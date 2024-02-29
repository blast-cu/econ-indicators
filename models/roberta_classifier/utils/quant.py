import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os

import data_utils.model_utils.dataset as d
import models.roberta_classifier.utils.general as gu


class QuantModel(nn.Module):
    """
    QuantModel is a PyTorch module that implements a quantification model based on the RoBERTa architecture.

    Args:
        model_checkpoint (str): The path or identifier of the pre-trained RoBERTa model.
        num_labels (int): The number of labels for the classification task.

    Attributes:
        config (RobertaConfig): The configuration of the RoBERTa model.
        dense (Linear): The linear layer for the dense connections.
        dropout (Dropout): The dropout layer for regularization.
        out_proj (Linear): The linear layer for the output projection.
        linear (Linear): The linear layer for combining CLS and indicator tokens.
        activation (ReLU): The activation function.
        roberta (RobertaModel): The pre-trained RoBERTa model.
        softmax (Softmax): The softmax function for probability calculation.

    Methods:
        from_pretrained(path, task): Loads the pre-trained model weights from the specified path.
        save(path, task): Saves the model weights to the specified path.
        forward(indices, excerpts, excerpt_attention_mask): Performs forward pass through the model.

    """

    def __init__(self, model_checkpoint, num_labels):
        super(QuantModel, self).__init__()

        self.config = RobertaConfig.from_pretrained(model_checkpoint)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size).to('cuda')
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout).to('cuda')
        self.out_proj = nn.Linear(self.config.hidden_size, num_labels).to('cuda')

        hidden_size = self.config.hidden_size  # num values per token
        self.linear = nn.Linear(hidden_size * 2, hidden_size).to('cuda')  # n size = [1536, 2]
        self.activation = nn.ReLU().to('cuda')
        self.roberta = RobertaModel.from_pretrained(model_checkpoint).to('cuda')
        self.softmax = nn.Softmax(dim=0).to('cuda')

    def from_pretrained(self, path, task):
        """
        Loads the pre-trained model weights from the specified path.

        Args:
            path (str): The path where the model weights are saved.
            task (str): The task name used as a suffix for the weight files.

        Returns:
            QuantModel: The QuantModel instance with loaded weights.

        """
        rob_path = os.path.join(path, task + '_roberta') 
        self.roberta = RobertaModel.from_pretrained(rob_path)

        dense_path = os.path.join(path, task + '_dense')
        self.dense.load_state_dict(torch.load(dense_path))
                                   
        dropout_path = os.path.join(path, task + '_dropout')
        self.dropout.load_state_dict(torch.load(dropout_path))

        out_proj_path = os.path.join(path, task + '_out_proj')
        self.out_proj.load_state_dict(torch.load(out_proj_path))

        linear_path = os.path.join(path, task + '_linear')
        self.linear.load_state_dict(torch.load(linear_path))

        activation_path = os.path.join(path, task + '_activation')
        self.activation.load_state_dict(torch.load(activation_path))

        return self
    
    def save(self, path, task):
        """
        Saves the model weights to the specified path.

        Args:
            path (str): The path where the model weights will be saved.
            task (str): The task name used as a suffix for the weight files.

        """
        rob_path = os.path.join(path, task + '_roberta')
        self.roberta.save_pretrained(rob_path)

        dense_path = os.path.join(path, task + '_dense')
        torch.save(self.dense.state_dict(), dense_path)

        dropout_path = os.path.join(path, task + '_dropout')
        torch.save(self.dropout.state_dict(), dropout_path)

        out_proj_path = os.path.join(path, task + '_out_proj')
        torch.save(self.out_proj.state_dict(), out_proj_path)

        linear_path = os.path.join(path, task + '_linear')
        torch.save(self.linear.state_dict(), linear_path)

        activation_path = os.path.join(path, task + '_activation')
        torch.save(self.activation.state_dict(), activation_path)

    def forward(self,
                indices: torch.Tensor,  # size = [8, 512, 769]
                excerpts: torch.Tensor,  # size = [8, 514](ish)
                excerpt_attention_mask: torch.Tensor  # size = [8, 512]
                ):
        """
        Performs forward pass through the model.

        Args:
            indices (Tensor): The indices tensor of shape [batch_size, sequence_length, num_values].
            excerpts (Tensor): The excerpts tensor of shape [batch_size, sequence_length].
            excerpt_attention_mask (Tensor): The attention mask tensor of shape [batch_size, sequence_length].

        Returns:
            Tensor: The logits tensor of shape [batch_size, num_labels].

        """
        batch_size = excerpts.shape[0]
        rob_out = self.roberta(excerpts, attention_mask=excerpt_attention_mask)

        last_layer = rob_out.last_hidden_state  # size = [8, 512, 768]
        padder = torch.zeros(batch_size, 512, 1).to('cuda')
        padded_x = torch.cat([last_layer, padder], dim = 2) # size = [8, 512, 769]

        spans = padded_x.gather(2, indices)  # index tensor must have the same number of dimensions as input tensor
        cls = last_layer[:, 0, :]  # size = [8, 768], CLS token

        indicator_token = spans.mean(dim=1)  # size = [8, 769]
        indicator_token = indicator_token[..., :-1]  # size = [8, 768]

        lin_in = torch.cat((cls, indicator_token), 1)  # size = [8, 1536]

        lin_out = self.linear(lin_in)
        activation = self.activation(lin_out)

        x = self.dropout(activation)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits


def setup(train_texts,
          test_texts,
          train_labels,
          test_labels,
          annotation_map,
          lr=2e-5,
          model_checkpoint: str = "roberta-base"):
    """
    Set up the training, validation, and testing data loaders, tokenizer, model, optimizer, and return them.

    Args:
        train_texts (list): List of training texts.
        test_texts (list): List of testing texts.
        train_labels (list): List of training labels.
        test_labels (list): List of testing labels.
        annotation_map (dict): Mapping of annotation labels to their corresponding indices.
        lr (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
        model_checkpoint (str, optional): Pretrained model checkpoint. Defaults to "roberta-base".

    Returns:
        tuple: A tuple containing the model, train loader, validation loader, test loader, and optimizer.
    """
    torch.manual_seed(42)  # Set random seed for reproducibility

    # split train into train and val
    train_texts, val_texts, train_labels, val_labels = \
        train_test_split(train_texts, train_labels,
                         test_size=0.1, random_state=42)

    tokenizer = RobertaTokenizerFast\
        .from_pretrained(pretrained_model_name_or_path=model_checkpoint,
                         problem_type="single_label_classification")

    max_length = 512
    train_data = d.QuantAnnClassificationDataset(texts=train_texts,
                                                 labels=train_labels,
                                                 tokenizer=tokenizer,
                                                 max_length=max_length)

    val_data = d.QuantAnnClassificationDataset(texts=val_texts,
                                               labels=val_labels,
                                               tokenizer=tokenizer,
                                               max_length=max_length)

    test_data = d.QuantAnnClassificationDataset(texts=test_texts,
                                                labels=test_labels,
                                                tokenizer=tokenizer,
                                                max_length=max_length)

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Define model
    num_labels = len(set(annotation_map.values()))
    model = QuantModel(model_checkpoint, num_labels=num_labels).to('cuda')

    # Define optimizer and loss function
    optimizer = torch.optim\
        .AdamW(model.parameters(), lr=lr)

    return model, train_loader, val_loader, test_loader, optimizer


def validate(model,
             val_loader):
    """
    Evaluate the performance of the model on the validation set.

    Args:
    - model (torch.nn.Module): The trained model to be evaluated.
    - val_loader (torch.utils.data.DataLoader): The validation data loader.
    - class_weights (torch.Tensor): The weights to be applied to each class during loss calculation.

    Returns:
    - float: The average validation loss.
    """

    model.eval()
    with torch.no_grad():

        predicted_labels = []
        true_labels = []

        for batch in val_loader:

            span = batch['span'].to('cuda')
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(span,
                            input_ids,
                            attention_mask)

            _, predicted = torch.max(outputs, 1)

            predicted_labels += predicted.tolist()
            true_labels += labels.tolist()

        f1 = f1_score(true_labels,
                      predicted_labels,
                      average='macro')

    return f1


def train(model,
          train_loader,
          val_loader,
          optimizer,
          class_weights):
    """
    Trains the given model using the provided training data and optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation data.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        class_weights (torch.Tensor): The weights for each class in the classification task.

    Returns:
        torch.nn.Module: The trained model.
    """

    improving = True
    val_f1_history = []

    history_len = 3
    epoch = 0
    
    while improving:
        model.train()
        for batch in train_loader:

            span = batch['span'].to('cuda')

            excerpt_input_ids = batch['input_ids'].to('cuda')
            excerpt_attention_mask = batch['attention_mask'].to('cuda')

            labels = batch['label'].to('cuda')

            outputs = model(span,
                            excerpt_input_ids,
                            excerpt_attention_mask=excerpt_attention_mask)
            
            loss = cross_entropy(outputs,
                                 labels,
                                 weight=class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_f1 = validate(model, val_loader, class_weights)

        improving, val_f1_history = gu.check_done(val_f1_history,
                                                  val_f1,
                                                  history_len)

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val F1: {val_f1_history[-1]:.4f}")
        epoch += 1

    return model


def test(model, test_loader):
    """
    Evaluate the performance of a given model on a test dataset.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
    - Tuple[List[int], List[int], float]: A tuple containing the true labels, predicted labels, and test accuracy.
    """
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        out_predicted = []
        out_labels = []

        for batch in test_loader:
            span = batch['span'].to('cuda')
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(span,
                            excerpts=input_ids,
                            excerpt_attention_mask=attention_mask)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            out_predicted += predicted.tolist()
            out_labels += labels.tolist()

        print(out_labels)
        print(out_predicted)

        test_acc = correct / total
        print(f"Test Accuracy: {test_acc:.4f}")

        # return macro f1 score
        test_f1 = f1_score(out_labels, out_predicted, average='macro')
        print(f"Test F1: {test_f1:.4f}")

        return out_labels, out_predicted, test_f1


def get_noise(annotation_component: str,
              task: str,
              noise_dict: {},
              test_article_ids: []):
    """
    Retrieves noise data from the given noise dictionary based on the specified annotation component and task.

    Args:
        annotation_component (str): The annotation component to retrieve noise data for.
        task (str): The task for which noise data is being retrieved.
        noise_dict (dict): The dictionary containing the noise data.

    Returns:
        tuple: A tuple containing two lists - texts and labels. The texts list contains pairs of indicator text and text with context,
               and the labels list contains the corresponding labels for each pair of texts.
    """

    texts = []  # list of [indicator text, text with context]
    labels = []

    for id in noise_dict.keys():
        if noise_dict[id][annotation_component] != '\x00':
            article_id, _ = id.split('_')
            if int(article_id) not in test_article_ids:
                if 'indicator' in noise_dict[id].keys(): # temp fix
                    indicator_text = noise_dict[id]['indicator']
                    excerpt_text = noise_dict[id]['excerpt']
                    text = [indicator_text, excerpt_text]

                    if type(noise_dict[id][annotation_component]) is not list:
                        noise_dict[id][annotation_component] = \
                            [noise_dict[id][annotation_component]]

                    for label in noise_dict[id][annotation_component]:
                        texts.append(text)
                        if label not in d.quant_label_maps[task].keys():
                            print(f"Label {label} not found in label maps")
                            print(id)
                            exit()
                        labels.append(d.quant_label_maps[task][label])

    return texts, labels
                   

def get_texts(
              annotation_component: str,
              task: str,
              qual_dict: dict,
              quant_dict: dict,
              article_ids: list,
              type_filter: list = []
              ):
    """
    Retrieves texts and labels for split from the given dictionaries for 
    the given task.

    Args:
        annotation_component (str): The annotation component to retrieve labels from.
        task (str): The task to perform.
        qual_dict (dict): Dictionary of article-level anns.
        quant_dict (dict): Dictionary of quant annotations.
        article_ids (list): The list of article IDs to retrieve texts and
                labels from.
        type_filter (list, optional): The list of ann types to filter the entries.
                Defaults to an empty list (no filter).

    Returns:
        tuple: A tuple containing the retrieved texts and labels.
            - texts (list): [indicator text, text with context]
            - labels (list): The list of labels for the given text
    """

    texts = []  # list of [indicator text, text with context]
    labels = []

    for id in article_ids:
        if 'quant_list' in qual_dict[id].keys():
            for quant_id in qual_dict[id]['quant_list']:
                if quant_id not in quant_dict.keys():
                    raise ValueError(f"Quant ID {quant_id} not found in quant_dict")

                if quant_dict[quant_id][annotation_component] != '\x00':

                    valid_entry = False
                    if len(type_filter) == 0:
                        valid_entry = True
                    elif 'type' in quant_dict[quant_id].keys():
                        if quant_dict[quant_id]['type'] in type_filter:
                            valid_entry = True

                    if valid_entry:
                        indicator_text = quant_dict[quant_id]['indicator']
                        excerpt_text = quant_dict[quant_id]['excerpt']
                        text = [indicator_text, excerpt_text]
                        texts.append(text)

                        label = quant_dict[quant_id][annotation_component]
                        labels.append(d.quant_label_maps[task][label])

    return texts, labels

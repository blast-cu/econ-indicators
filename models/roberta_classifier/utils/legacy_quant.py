
from transformers import RobertaTokenizerFast, RobertaModel, RobertaConfig
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import cross_entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
from data_utils.model_utils.dataset import quant_label_maps as label_maps


def find_sub_list(indicator_text, excerpt_encoding, text):

    sub_start = text.find(indicator_text)
    sub_end = sub_start + len(indicator_text)
    offset_map = excerpt_encoding['offset_mapping'].tolist()

    start_index = None
    end_index = None
    for i, token in enumerate(offset_map[0]):
        token_start = token[0]
        token_end = token[1]
        if start_index is None:
            if sub_start >= token_start and sub_start <= token_end:
                start_index = i
        else:
            if sub_end >= token_start and sub_end <= token_end:
                end_index = i + 1
                break

    return start_index, end_index


class QuantModel(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super(QuantModel, self).__init__()

        self.config = RobertaConfig.from_pretrained(model_checkpoint)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.config.hidden_size, num_labels)

        hidden_size = self.config.hidden_size  # num values per token
        self.linear = nn.Linear(hidden_size * 2, hidden_size)  # n size = [1536, 2]
        self.activation = nn.ReLU()
        self.roberta = RobertaModel.from_pretrained(model_checkpoint)
        self.softmax = nn.Softmax(dim=0)
    
    def from_pretrained(self, path, task):

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

        # load pretrained roberta from path
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
                start_index,
                end_index,
                excerpts,  # size = [8, 514]
                excerpt_attention_mask
                ):

        rob_out = self.roberta(excerpts, attention_mask=excerpt_attention_mask)
        last_layer = rob_out.last_hidden_state  # size = [8, 514, 768]

        cls = last_layer[:, 0, :]  # size = [8, 768], CLS token

        batch_size = excerpts.size(dim=0)
        indicator_token = torch.zeros((batch_size, 768)).to('cuda')
        for i in range(batch_size): # loop thru batch, get mean of each set of indicator tokens
            indicator_token[i] = torch.mean(last_layer[i][start_index[i]:end_index[i]], dim=0)

        lin_in = torch.cat((cls, indicator_token), 1)  # size = [8, 1536]

        lin_out = self.linear(lin_in).to('cuda')
        activation = self.activation(lin_out).to('cuda')

        x = self.dropout(activation).to('cuda')
        x = self.dense(x).to('cuda')
        x = torch.tanh(x)
        x = self.dropout(x).to('cuda')
        logits = self.out_proj(x).to('cuda')

        return logits


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels=[], ids=[], tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.indicator_texts = [t[0] for t in texts]
        self.texts = [t[1] for t in texts]
        self.labels = labels

        self.article_ids = []
        self.ann_ids = []
        for id in ids:
            article_id, ann_id = id.split('_')
            self.article_ids.append(int(article_id))
            self.ann_ids.append(int(ann_id))
        
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized  item from the dataset
        """
        text = self.texts[idx]
        if len(self.labels) > 0:
            label = self.labels[idx]
        else:
            label = -1

        if len(self.article_ids) > 0:
            article_id = self.article_ids[idx]
            ann_id = self.ann_ids[idx]
        else:
            article_id = -1
            ann_id = -1

        indicator_text = self.indicator_texts[idx]

        temp_encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=False,
            return_offsets_mapping=True
        )

        start_index, end_index = find_sub_list(indicator_text, temp_encoding, text)

        if end_index is None:
            end_index = start_index

        if start_index is None or end_index is None:
            print('Substring: ' + indicator_text)
            print('Original text: ' + text)
            print('Start index: ' + str(start_index))
            print('End index: ' + str(end_index))
            print('Offset mapping: \n')
            for i, token in enumerate(temp_encoding['offset_mapping'][0]):
                print(f"Token {i}: {token}")
            raise Exception('Could not find indicator text in excerpt')

        if end_index > self.max_length:
            text_start = end_index + int(self.max_length / 2) - self.max_length
            text = text[text_start:]

            start_index = start_index - text_start
            end_index = end_index - text_start

        excerpt_encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'start_index': torch.tensor(start_index),
            'end_index': torch.tensor(end_index),
            'input_ids': excerpt_encoding['input_ids'].flatten(),
            'attention_mask': excerpt_encoding['attention_mask'].flatten(),
            'label': torch.tensor(label),
            'article_ids': torch.tensor(article_id),
            'ann_ids': torch.tensor(ann_id)
        }

def setup(train_texts,
          test_texts,
          train_labels,
          test_labels,
          annotation_map,
          lr=2e-5,
          model_checkpoint: str = "roberta-base"):
    """
    Setup model and data loaders for fine-tuning a model on qualitative 
    annotation classification
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
    train_data = TextClassificationDataset(texts=train_texts,
                                           labels=train_labels,
                                           tokenizer=tokenizer,
                                           max_length=max_length)

    val_data = TextClassificationDataset(texts=val_texts,
                                         labels=val_labels,
                                         tokenizer=tokenizer,
                                         max_length=max_length)

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    # for train best setting, test_loader is None
    test_loader = None
    if test_texts:
        test_data = TextClassificationDataset(
            texts=test_texts,
            labels=test_labels,
            tokenizer=tokenizer,
            max_length=max_length
        )
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Define model
    num_labels = len(set(annotation_map.values()))
    model = QuantModel(model_checkpoint, num_labels=num_labels).to('cuda')

    # Define optimizer and loss function
    optimizer = torch.optim\
        .AdamW(model.parameters(), lr=lr)

    return model, train_loader, val_loader, test_loader, optimizer


def validate(model, val_loader, class_weights):
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

            start_index = batch['start_index'].to('cuda')
            end_index = batch['end_index'].to('cuda')
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(start_index,
                            end_index,
                            input_ids,
                            attention_mask)

            _, predicted = torch.max(outputs, 1)

            predicted_labels += predicted.tolist()
            true_labels += labels.tolist()

        f1 = f1_score(true_labels,
                      predicted_labels,
                      average='macro')

    return f1

def check_done(val_f1_history: list, val_f1, history_len):
    """
    Check if the model has stopped improving based on the validation loss history.

    Parameters:
    - val_loss_history (list): A list of previous validation losses.
    - val_loss (float): The current validation loss.
    - history_len (int): The maximum length of the validation loss history.

    Returns:
    - improving (bool): True if the model is still improving, False otherwise.
    - val_loss_history (list): The updated validation loss history.
    """

    improving = True
    if len(val_f1_history) == history_len:
        val_f1_history.pop(0)  # remove at index 0

        if val_f1_history[0] == val_f1_history[1]:
            if val_f1_history[1] == val_f1:
                improving = False

    val_f1_history.append(val_f1)
    return improving, val_f1_history


def train(model, train_loader, val_loader, optimizer, class_weights):

    improving = True
    val_f1_history = []

    history_len = 3
    epoch = 0
    
    while improving:
        model.train()
        for batch in train_loader:

            start_index = batch['start_index'].to('cuda')
            end_index = batch['end_index'].to('cuda')

            # for s in enumerate(batch['start_index']):
            #     print(s)
            #     print(batch['end_index'][s])
            #     print()
            # exit()

            excerpt_input_ids = batch['input_ids'].to('cuda')
            excerpt_attention_mask = batch['attention_mask'].to('cuda')

            labels = batch['label'].to('cuda')

            outputs = model(start_index,
                            end_index,
                            excerpt_input_ids,
                            excerpt_attention_mask=excerpt_attention_mask)
            
            loss = cross_entropy(outputs,
                                 labels,
                                 weight=class_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_f1 = validate(model, val_loader, class_weights)

        improving, val_f1_history = check_done(val_f1_history,
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
            start_index = batch['start_index'].to('cuda')
            end_index = batch['end_index'].to('cuda')
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(start_index=start_index,
                            end_index=end_index,
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
                        if label not in label_maps[task].keys():
                            print(f"Label {label} not found in label_maps")
                            print(id)
                            # print(noise_dict[id])
                            exit()
                        labels.append(label_maps[task][label])

    return texts, labels
                   

def get_texts(
              annotation_component: str,
              task: str,
              qual_dict: {},
              quant_dict: {},
              article_ids: [],
              type_filter: [] = []
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
                    print(f"Quant ID {quant_id} not found in quant_dict")
                    print(qual_dict[id])
                    print(quant_dict)
                    exit()
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
                        labels.append(label_maps[task][label])

    return texts, labels

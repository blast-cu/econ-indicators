import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torch.nn.functional import cross_entropy
from sklearn.utils.class_weight import compute_class_weight
from data_utils.dataset import qual_label_maps as label_maps
import data_utils.get_annotation_stats as gs


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels=[], tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.texts = texts
        self.labels = labels
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
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }


def check_done(val_f1_history: list, val_f1, patience, history_len):
    """
    Check if the model has stopped improving based on the validation loss history.

    Parameters:
    - val_loss_history (list): A list of previous validation losses.
    - val_loss (float): The current validation loss.
    - patience (float): The minimum difference between the current validation loss and the previous one to consider improvement.
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

            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(input_ids,
                            attention_mask)

            _, predicted = torch.max(outputs.logits, 1)

            predicted_labels += predicted.tolist()
            true_labels += labels.tolist()

        f1 = f1_score(true_labels,
                      predicted_labels,
                      average='macro')

    return f1


def train(model, train_loader, val_loader, optimizer, class_weights):
    """
    Trains the given model using the provided train and validation data loaders,
    optimizer, and class weights.

    Args:
    - model (torch.nn.Module): The model to train.
    - train_loader (torch.utils.data.DataLoader): The data loader for the training data.
    - val_loader (torch.utils.data.DataLoader): The data loader for the validation data.
    - optimizer (torch.optim.Optimizer): The optimizer to use during training.
    - class_weights (torch.Tensor): The class weights to use for the loss function.

    Returns:
    - The trained model.
    """
    
    improving = True
    # val_loss_history = []
    val_f1_history = []

    patience = 0.03
    history_len = 5
    epoch = 0

    while improving:
        model.train()
        for batch in train_loader:

            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            

            loss = cross_entropy(outputs.logits, labels, weight=class_weights)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        val_f1 = validate(model, val_loader, class_weights)
        improving, val_f1_history = check_done(val_f1_history,
                                               val_f1,
                                               patience,
                                               history_len)

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val F1: {val_f1_history[-1]:.4f}")
        epoch += 1

    return model


def test(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader):
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
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
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


def setup(train_texts, test_texts, train_labels, test_labels, annotation_map, lr=2e-5, model_checkpoint: str="roberta-base"):
    """
    Sets up the data and model for training and testing a RoBERTa model for text classification.

    Args:
        train_texts (list): List of training texts.
        test_texts (list): List of testing texts.
        train_labels (list): List of labels for training texts.
        test_labels (list): List of labels for testing texts.
        annotation_map (dict): A dictionary mapping label names to label indices.
        lr (float): Learning rate for the optimizer.

    Returns:
        tuple: A tuple containing the RoBERTa model, data loaders for training, validation, and testing,
        and the optimizer.
    """

    torch.manual_seed(42)  # Set random seed for reproducibility

    # split train into train and val
    train_texts, val_texts, train_labels, val_labels = \
        train_test_split(train_texts, train_labels,
                         test_size=0.1, random_state=42)

    tokenizer = RobertaTokenizer\
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

    test_data = TextClassificationDataset(texts=test_texts,
                                          labels=test_labels,
                                          tokenizer=tokenizer,
                                          max_length=max_length)

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Define model
    num_labels = len(annotation_map)
    model = RobertaForSequenceClassification\
        .from_pretrained(model_checkpoint, num_labels=num_labels).to('cuda')
    # model = RobertaForSequenceClassification\
    #     .from_pretrained(model_checkpoint, num_labels=num_labels)

    # Define optimizer and loss function
    optimizer = torch.optim\
        .AdamW(model.parameters(), lr=lr)

    return model, train_loader, val_loader, test_loader, optimizer


def get_weights(y, annotation_map: dict):
    """
    Computes class weights for weighted loss.

    Parameters:
    - y: list or array of target values
    - annotation_map: dictionary mapping target values to class labels

    Returns:
    - class_weights: tensor of class weights, to be used in loss function
    """

    # compute class weights for weighted loss
    classes = list(set(list(annotation_map.values())))

    if len(classes) != len(set(y)):
        y = y + classes

    weight = \
        compute_class_weight(class_weight='balanced', classes=classes, y=y)

    weight = [float(w) for w in weight]
    class_weights = torch.tensor(weight).to('cuda')
    return class_weights


def get_noise(db_filename: str,
              annotation_component: str,
              task: str,
              noise_dict: {}
             ):
    
    texts = []
    labels = []

    for id in noise_dict.keys():
        if noise_dict[id][annotation_component] !='\x00':
            
            if type(noise_dict[id][annotation_component]) is not list:
                noise_dict[id][annotation_component] = \
                    [noise_dict[id][annotation_component]]

            for label in noise_dict[id][annotation_component]:
                texts.append(gs.get_text(id, db_filename, clean=False))
                labels.append(label_maps[task][label])
    
    return texts, labels


def get_texts(db_filename: str,
              annotation_component: str,
              task: str,
              agreed_anns_dict: {},
              article_ids: []):

    texts = []
    labels = []

    for id in article_ids:
        if annotation_component in agreed_anns_dict[id].keys():
            if agreed_anns_dict[id][annotation_component] !='\x00':
                texts.append(gs.get_text(id, db_filename, clean=False))
                label = agreed_anns_dict[id][annotation_component]
                labels.append(label_maps[task][label])

    return texts, labels
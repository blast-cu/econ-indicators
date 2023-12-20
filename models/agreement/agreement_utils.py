
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import f1_score
import os
from torch.nn.functional import cross_entropy



def find_sub_list(sub_list, list):

    sub_len = len(sub_list)
    for ind in (i for i, e in enumerate(list) if e == sub_list[0]):
        if list[ind:ind+sub_len] == sub_list:
            return ind, ind+sub_len-1


class AgreementModel(nn.Module):
    def __init__(self, model_checkpoint, num_labels):
        super(AgreementModel, self).__init__()

        self.config = RobertaConfig.from_pretrained(model_checkpoint)
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(self.config.hidden_size, num_labels)

        hidden_size = self.config.hidden_size  # num values per token
        self.linear = nn.Linear(hidden_size * 4, hidden_size)  # n size = [3072, 2]
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
        

    # def forward(self,
    #             start_index,
    #             end_index,
    #             excerpts,  # size = [8, 514]
    #             excerpt_attention_mask
    #             ):
    def forward(self,
                n1_start_index,
                n1_end_index,
                n1_excerpt_input_ids,
                n1_excerpt_attention_mask,
                n2_start_index,
                n2_end_index,
                n2_excerpt_input_ids,
                n2_excerpt_attention_mask):

        # n1 
        n1_rob_out = self.roberta(n1_excerpt_input_ids, attention_mask=n1_excerpt_attention_mask)
        n1_last_layer = n1_rob_out.last_hidden_state  # size = [8, 514, 768]

        n1_cls = n1_last_layer[:, 0, :]  # size = [8, 768], CLS token

        batch_size = n1_excerpt_input_ids.size(dim=0)
        n1_indicator_token = torch.zeros((batch_size, 768)).to('cuda')
        for i in range(batch_size):
            if n1_start_index[i] == n1_end_index[i]:
                n1_indicator_token[i] = n1_last_layer[i][n1_start_index[i]]
            else: n1_indicator_token[i] = torch.mean(n1_last_layer[i][n1_start_index[i]:n1_end_index[i]], dim=0)

        # if torch.isnan(n1_last_layer).any():
        #     print("n1 last layer is nan")
        #     exit()
        # if torch.isnan(n1_cls).any():
        #     print("n1 cls is nan")
        #     exit()
        # if torch.isnan(n1_indicator_token[1]).any():
        #     print("n1 indicator token is nan")
        #     exit()
        # # print(n1_start_index[1])
        # # print(n1_end_index[1])
        # # print("n1 indicator token: ")
        # # print(n1_indicator_token[1])
        

        n1_lin_in = torch.cat((n1_cls, n1_indicator_token), 1)  # size = [8, 1536]
        # print("n1 linear layer input: ")
        # print(n1_lin_in)

        # n2
        n2_rob_out = self.roberta(n2_excerpt_input_ids, attention_mask=n2_excerpt_attention_mask)
        n2_last_layer = n2_rob_out.last_hidden_state  # size = [8, 514, 768]

        n2_cls = n2_last_layer[:, 0, :]  # size = [8, 768], CLS token

        batch_size = n2_excerpt_input_ids.size(dim=0)
        n2_indicator_token = torch.zeros((batch_size, 768)).to('cuda')
        for i in range(batch_size):
            if n2_start_index[i] == n2_end_index[i]:
                n2_indicator_token[i] = n2_last_layer[i][n2_start_index[i]]
            else:
                n2_indicator_token[i] = torch.mean(n2_last_layer[i][n2_start_index[i]:n2_end_index[i]], dim=0)

        n2_lin_in = torch.cat((n2_cls, n2_indicator_token), 1)  # size = [8, 1536]

        lin_in = torch.cat((n1_lin_in, n2_lin_in), 1)  # size = [8, 3072]
        # print("linear layer input: ")
        # print(lin_in)

        lin_out = self.linear(lin_in).to('cuda')
        activation = self.activation(lin_out).to('cuda')
        # probs = self.softmax(activation)  # size = [8, 2]
        x = self.dropout(activation).to('cuda')
        x = self.dense(x).to('cuda')
        x = torch.tanh(x)
        x = self.dropout(x).to('cuda')
        logits = self.out_proj(x).to('cuda')
        if torch.isnan(logits).any():
            print("logits is nan")
            exit()

        return logits


class AgreementPredictDataset(Dataset):
    def __init__(self, neighbors, labels=[], ids=[], tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.n1_indicator_texts = []
        self.n1_texts = []

        self.n2_indicator_texts = []
        self.n2_texts = []

        for n in neighbors:
            self.n1_indicator_texts.append(n[0][0])
            self.n1_texts.append(n[0][1])

            self.n2_indicator_texts.append(n[1][0])
            self.n2_texts.append(n[1][1])
            # print(n)
            # print()

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
        return len(self.n1_texts)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized  item from the dataset
        """
        n1_indicator_text = self.n1_indicator_texts[idx]
        n1_text = self.n1_texts[idx]
        n2_indicator_text = self.n2_indicator_texts[idx]
        n2_text = self.n2_texts[idx]

        # print(n1_indicator_text)
        # print(n2_indicator_text)
        # print()

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


        i = n1_text.find(n1_indicator_text)
        if i != 0:
            n1_indicator_text = ' ' + n1_indicator_text
        
        n1_indicator_encoding = self.tokenizer(
            n1_indicator_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        n1_excerpt_encoding = self.tokenizer(
            n1_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        complete = self.tokenizer.convert_ids_to_tokens(n1_excerpt_encoding['input_ids'].flatten())
        complete = [re.sub('Ġ', '', s) for s in complete if s != '<pad>']

        sub = self.tokenizer.convert_ids_to_tokens(n1_indicator_encoding['input_ids'].flatten())
        sub = [re.sub('Ġ', '', s) for s in sub if s != '<pad>']
        sub = [s for s in sub if s != '']
        sub = sub[1:-1]

        print_counter = 0
        n1_indicator_indices = None
        while n1_indicator_indices is None and len(sub) > 0:
            n1_indicator_indices = find_sub_list(sub, complete)
            sub = sub[1:]

            print_counter += 1
            # if print_counter > 2: 
            #     print(type(sub))
            #     print(complete)
        
        if n1_indicator_indices is None:
            # print('n1 Indicator not found in excerpt')
            # print(indicator_text)
            # print(text)
            # print()
            n1_indicator_indices = [0, 1]
        
        
        i = n2_text.find(n2_indicator_text)
        if i != 0:
            n2_indicator_text = ' ' + n2_indicator_text
        
        n2_indicator_encoding = self.tokenizer(
            n2_indicator_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        n2_excerpt_encoding = self.tokenizer(
            n2_text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        complete = self.tokenizer.convert_ids_to_tokens(n2_excerpt_encoding['input_ids'].flatten())
        complete = [re.sub('Ġ', '', s) for s in complete if s != '<pad>']

        sub = self.tokenizer.convert_ids_to_tokens(n2_indicator_encoding['input_ids'].flatten())
        sub = [re.sub('Ġ', '', s) for s in sub if s != '<pad>']
        sub = [s for s in sub if s != '']
        sub = sub[1:-1]

        print_counter = 0
        n2_indicator_indices = None
        while n2_indicator_indices is None and len(sub) > 0:
            n2_indicator_indices = find_sub_list(sub, complete)
            sub = sub[1:]

            print_counter += 1
            # if print_counter > 2: 
            #     print(type(sub))
            #     print(complete)
        
        if n2_indicator_indices is None:
            # print('n2 Indicator not found in excerpt')
            # print(indicator_text)
            # print(text)
            # print()
            n2_indicator_indices = [0, 1]

        # print(n1_indicator_indices)
        # print(n2_indicator_indices)


        return {
            'n1_start_index': torch.tensor(n1_indicator_indices[0]),
            'n1_end_index': torch.tensor(n1_indicator_indices[1]),
            'n1_input_ids': n1_excerpt_encoding['input_ids'].flatten(),
            'n1_attention_mask': n1_excerpt_encoding['attention_mask'].flatten(),
            'n2_start_index': torch.tensor(n2_indicator_indices[0]),
            'n2_end_index': torch.tensor(n2_indicator_indices[1]),
            'n2_input_ids': n2_excerpt_encoding['input_ids'].flatten(),
            'n2_attention_mask': n2_excerpt_encoding['attention_mask'].flatten(),
            'label': torch.tensor(label),
            'n1_article_ids': [],
            'n2_article_ids': [],
            'ann_ids': torch.tensor(ann_id)
        }

def setup(train_neighbors,
          test_neighbors,
          train_labels,
          test_labels,
          lr=2e-5,
          model_checkpoint: str = "roberta-base"):
    """
    Train a model on qualitative annotations
    """
    torch.manual_seed(42)  # Set random seed for reproducibility

    # split train into train and val
    train_neighbors, val_neighbors, train_labels, val_labels = \
        train_test_split(train_neighbors, train_labels,
                         test_size=0.1, random_state=42)

    tokenizer = RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path=model_checkpoint,
                         problem_type="single_label_classification")

    max_length = 512
    train_data = AgreementPredictDataset(neighbors=train_neighbors,
                                           labels=train_labels,
                                           tokenizer=tokenizer,
                                           max_length=max_length)

    val_data = AgreementPredictDataset(neighbors=val_neighbors,
                                         labels=val_labels,
                                         tokenizer=tokenizer,
                                         max_length=max_length)

    test_data = AgreementPredictDataset(neighbors=test_neighbors,
                                          labels=test_labels,
                                          tokenizer=tokenizer,
                                          max_length=max_length)

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Define model
    num_labels = 2
    model = AgreementModel(model_checkpoint, num_labels=num_labels).to('cuda')


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

            n1_start_index = batch['n1_start_index'].to('cuda')
            n1_end_index = batch['n1_end_index'].to('cuda')
            n1_excerpt_input_ids = batch['n1_input_ids'].to('cuda')
            n1_excerpt_attention_mask = batch['n1_attention_mask'].to('cuda')

            n2_start_index = batch['n2_start_index'].to('cuda')
            n2_end_index = batch['n2_end_index'].to('cuda')
            n2_excerpt_input_ids = batch['n2_input_ids'].to('cuda')
            n2_excerpt_attention_mask = batch['n2_attention_mask'].to('cuda')

            labels = batch['label'].to('cuda')

            outputs = model(n1_start_index,
                            n1_end_index,
                            n1_excerpt_input_ids,
                            n1_excerpt_attention_mask,
                            n2_start_index,
                            n2_end_index,
                            n2_excerpt_input_ids,
                            n2_excerpt_attention_mask)

            _, predicted = torch.max(outputs, 1)

            predicted_labels += predicted.tolist()
            true_labels += labels.tolist()

        f1 = f1_score(true_labels,
                      predicted_labels,
                      average='macro')

    return f1

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

        if val_f1_history[-1] > val_f1:  # overfitting training data
            improving = False
        elif val_f1_history[0] - val_f1 < patience:
            improving = False

    val_f1_history.append(val_f1)
    return improving, val_f1_history


def train(model, train_loader, val_loader, optimizer, class_weights):

    improving = True
    val_f1_history = []

    patience = 0.03
    history_len = 5
    epoch = 0

    while improving:
        model.train()
        for batch in train_loader:

            n1_start_index = batch['n1_start_index'].to('cuda')
            n1_end_index = batch['n1_end_index'].to('cuda')
            n1_excerpt_input_ids = batch['n1_input_ids'].to('cuda')
            n1_excerpt_attention_mask = batch['n1_attention_mask'].to('cuda')

            n2_start_index = batch['n2_start_index'].to('cuda')
            n2_end_index = batch['n2_end_index'].to('cuda')
            n2_excerpt_input_ids = batch['n2_input_ids'].to('cuda')
            n2_excerpt_attention_mask = batch['n2_attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            # print("in train func before model")
            # print(n1_start_index)
            # print(n1_end_index)
            # print(n1_excerpt_input_ids)
            # print(n1_excerpt_attention_mask)
            # print(n2_start_index)
            # print(n2_end_index)
            # print(n2_excerpt_input_ids)
            # print(n2_excerpt_attention_mask)
            # print(labels)
            # print()


            

            outputs = model(n1_start_index,
                            n1_end_index,
                            n1_excerpt_input_ids,
                            n1_excerpt_attention_mask,
                            n2_start_index,
                            n2_end_index,
                            n2_excerpt_input_ids,
                            n2_excerpt_attention_mask)
            # print("in train func before loss")
            # print(outputs.tolist())
            # # print(labels.tolist())
            # print()
            loss = cross_entropy(outputs,
                                 labels,
                                 weight=class_weights)
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
            n1_start_index = batch['n1_start_index'].to('cuda')
            n1_end_index = batch['n1_end_index'].to('cuda')
            n1_excerpt_input_ids = batch['n1_input_ids'].to('cuda')
            n1_excerpt_attention_mask = batch['n1_attention_mask'].to('cuda')

            n2_start_index = batch['n2_start_index'].to('cuda')
            n2_end_index = batch['n2_end_index'].to('cuda')
            n2_excerpt_input_ids = batch['n2_input_ids'].to('cuda')
            n2_excerpt_attention_mask = batch['n2_attention_mask'].to('cuda')

            labels = batch['label'].to('cuda')

            outputs = model(n1_start_index,
                            n1_end_index,
                            n1_excerpt_input_ids,
                            n1_excerpt_attention_mask,
                            n2_start_index,
                            n2_end_index,
                            n2_excerpt_input_ids,
                            n2_excerpt_attention_mask)

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
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import datetime

import data_utils.get_annotation_stats as gs

annotation_map = {
    'frame': {'business': 0, 'industry': 1, 'macro': 2, 'government': 3, 'other': 4},
    'econ_rate': {'good': 0, 'poor': 1, 'unsure': 2, 'irrelevant': 3, 'none': 4}
    }

def load_dataset(db_filename: str, annotation_component: str):
    """
    Takes file location of database, annotation component, and model name
    Returns text, labels from datasets for given annotation component
    """
    # frame - 0: business, 1: industry, 2: macro, 3: government, 4: other
    # econ_rate - 0: good, 1: poor, 2: unsure, 3: irrelevant, 4: none 
    frame_map = annotation_map[annotation_component]

    text = []
    labels = []
    qual_ann = gs.get_qual_dict(db_filename)
    agreed_qual_ann = gs.get_agreed_anns(qual_ann)

    for article_id in agreed_qual_ann.keys():
        if agreed_qual_ann[article_id][annotation_component] != '\0':
            label_dict = agreed_qual_ann[article_id]
            clean_text = gs.get_clean_text(article_id, db_filename)

            text.append(clean_text)
            label = frame_map[label_dict[annotation_component]]
            labels.append(label)

    return text, labels

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
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

def train(model, train_loader, val_loader, optimizer):
        # Train model
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                labels = batch['label'].to('cuda')

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item() * input_ids.size(0)

            val_loss /= len(val_loader.dataset)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

def test(model, test_loader):
        
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['label'].to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = correct / total
        print(f"Test Accuracy: {test_acc:.4f}")
        return test_acc

def setup(annotation_component: str): 
    
    torch.manual_seed(42) # Set random seed for reproducibility

    # Load training, validation, and test data
    texts, labels = load_dataset("data/data.db", annotation_component=annotation_component)
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    max_length = 512
    train_data = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_data = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    test_data = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)

    batch_size = 8
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Define model
    # TODO: Some weights of RobertaForSequenceClassification were not initialized from the model 
    # checkpoint at roberta-base and are newly initialized: ['classifier.out_proj.weight', 'classifier.dense.bias', 
    # 'classifier.out_proj.bias', 'classifier.dense.weight']
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
    model.to('cuda')

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5) # TODO: optimiz LR?
    # optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False) # deprecated

    return model, train_loader, val_loader, test_loader, optimizer


def to_csv(annotation_component: str, accuracy: float):
    with open(f"models/roberta/results/{annotation_component}_accuracy.csv", 'w') as f:
        f.write(f"{accuracy}" + ',')
        f.write(f"{datetime.datetime.now()}" + '\n')

def main():
    
    for annotation_component in ['frame', 'econ_rate']:
        model, train_loader, val_loader, test_loader, optimizer = setup(annotation_component='frame')
        train(model, train_loader, val_loader, optimizer)
        accuracy = test(model, test_loader)
        to_csv(annotation_component, accuracy)



if __name__ == '__main__':
    main()
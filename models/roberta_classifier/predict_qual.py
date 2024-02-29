import argparse
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from data_utils import get_annotation_stats as gs
import data_utils.model_utils.dataset as d


class PredictionDataset(Dataset):
    def __init__(self, articles:{}, tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.ids = list(articles.keys())
        self.texts = list(articles.values())
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
        id = self.ids[idx]
        text = self.texts[idx]

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
            'ids': id
        }

def main(args):
    """
    Runs the prediction pipeline for the qualitative component of the Roberta classifier
    on a random sample of articles without annotations. Outputs a csv file containing
    article ids and their predicted annotations.

    Args:
        args: An argparse.Namespace object containing the command line arguments.

    Returns:
        None
    """
    
    # load articles w/o annotations
    # key: article id, value: article text
    articles = gs.get_no_anns(db_filename=args.db, num_samples=args.ns)

    torch.manual_seed(42)  # Set random seed for reproducibility

    tokenizer = RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path="roberta-large",
                         problem_type="single_label_classification")

    data = PredictionDataset(articles=articles,
                             tokenizer=tokenizer,
                             max_length=512)

    batch_size = 8
    loader = DataLoader(data, batch_size=batch_size, shuffle=False) # check shuffle thing

    # load fine-tuned model for each annotation component
    models = {}
    for k in d.qual_predict_maps.keys():
        model_path = f"models/roberta_classifier/best_models/qual/{k}_model"
        models[k] = RobertaForSequenceClassification\
            .from_pretrained(model_path).to('cuda')

    # create dictionary to store annotations
    annotations = {}
    for id in articles.keys():
        annotations[id] = {}

    for annotation_component in models.keys():
        for batch in loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            ids = batch['ids'].to('cuda')

            cur_model = models[annotation_component]
            outputs = cur_model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)

            for i, id in enumerate(ids.tolist()):
                col_name = f"{annotation_component}_prediction"
                prediction = int(predicted[i].item())
                annotations[id][col_name] = d.qual_predict_maps[annotation_component][prediction]



    destination = "models/roberta_classifier/samples/qual_samples.csv"
    df = pd.DataFrame(annotations).transpose()
    df.to_csv(destination)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--db', type=str, required=True, help='path to database file from top level directory')
    parser.add_argument('--ns', type=int, default=10, help='number of article samples to load and label')
    args = parser.parse_args()
    main(args)
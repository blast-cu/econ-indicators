# from models.utils import dataset as d
# import argparse

import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pickle
import pandas as pd


label_maps = {
    'type': {
        0: 'macro',
        1: 'industry',
        2: 'government',
        3: 'personal',
        4: 'business',
        5: 'other'
    },
    'type-binary': {
        0: 'macro',
        1: 'other'
    },
    'spin': {
        0: 'positive',
        1: 'negative',
        2: 'neutral'
    },
    'macro_type': {
        0: 'jobs',
        1: 'retail',
        2: 'interest',
        3: 'prices',
        4: 'energy',
        5: 'wages',
        6: 'macro',
        7: 'market',
        8: 'currency',
        9: 'housing',
        10: 'other'
    }
}

class PredictionDataset(Dataset):
    def __init__(self, articles:{}, tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.ids = list(articles.keys())
        self.texts = list(articles.values())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.type = [] * len(self.texts)

    
    def set_type(self, idx, type):
        self.type[idx] = type

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

def remove_dates():
    """
    Remove articles with dates
    """

    return

def save_progress(to_save,
                  filename: str):
    """
    Save the progress to a file using pickle.

    Args:
        to_save: The object to be saved.
        filename (str): The name of the file to save the object to. Default is 'excerpts_dict'.
    """
    try:
        progress_file = open(filename, 'wb')
        pickle.dump(to_save, progress_file)
        progress_file.close()

    except:
        print("Something went wrong")



def main(args):

    quant_dict = pickle.load(open('data/clean/quant_dict', 'rb'))
    # annotations = {k: {'type': v['type'], 'spin': '', 'macro_type': ''} for k, v in annotations.items() if 'type' in v.keys() and v['type'] == 'macro'}
    annotations = {}
    for id, ann_dict in quant_dict.items():
        if 'type' in ann_dict.keys() and ann_dict['type'] == 'macro':
            annotations[id] = {'type': 'macro'}
            if 'spin' in ann_dict.keys():
                annotations[id]['spin'] = ann_dict['spin']
            else: 
                annotations[id]['spin'] = ''
            if 'macro_type' in ann_dict.keys():
                annotations[id]['macro_type'] = ann_dict['macro_type']
            else:
                annotations[id]['macro_type'] = ''


    # excerpts = d.get_excerpts_dict(args.db)
    # save_progress(excerpts, 'models/utils/splits/quant_excerpts_dict')

    excerpts = pickle.load(open('models/utils/splits/quant_excerpts_dict', 'rb'))

    # excerpts = {}
    # smaller excerpts for testing 
    # for k in list(temp_excerpts.keys())[:50]:
    #     excerpts[k] = temp_excerpts[k]


    # excerpts = remove_dates(excerpts)

    torch.manual_seed(42)  # Set random seed for reproducibility
    tokenizer = RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path="roberta-base",
                            problem_type="single_label_classification")

    data = PredictionDataset(articles=excerpts,
                                tokenizer=tokenizer,
                                max_length=512)

    batch_size = 8
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    models_dir = "models/roberta_classifier/best_models/original/quant/"
    type_model = RobertaForSequenceClassification.from_pretrained(models_dir + "type-binary_model").to("cuda")



    num_batches = len(loader)
    freq_report = 100
    for i, batch in enumerate(loader):
        if i % freq_report == 0:
            print(f"Batch {i+1}/{num_batches}")
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        ids = batch['ids']

        outputs = type_model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)

        for i, id in enumerate(ids):

            prediction = int(predicted[i].item())
            if prediction == 0 and id not in annotations:
                annotations[id] = {}
                annotations[id]['type'] = 'macro'
                annotations[id]['spin'] = ''
                annotations[id]['macro_type'] = ''

    print(">>> Saving type annotations")
    save_progress(annotations, 'outputs/annotations_type_predictions')

    filtered_text = {k: v for k, v in excerpts.items() if k in annotations.keys()}

    filtered_data = PredictionDataset(articles=filtered_text,
                                        tokenizer=tokenizer,
                                        max_length=512)
    filtered_loader = DataLoader(filtered_data, batch_size=batch_size)

    spin_model = RobertaForSequenceClassification.from_pretrained(models_dir + "spin_model").to("cuda")
    macro_type_model = RobertaForSequenceClassification.from_pretrained(models_dir + "macro_type_model").to("cuda")

    num_batches = len(filtered_loader)
    for i, batch in enumerate(filtered_loader):
        if i % freq_report == 0:
            print(f"Spin/Macro Type Batch {i+1}/{num_batches}")
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        ids = batch['ids']

        spin_outputs = spin_model(input_ids, attention_mask=attention_mask)
        _, spin_predicted = torch.max(spin_outputs.logits, 1)

        macro_type_outputs = macro_type_model(input_ids, attention_mask=attention_mask)
        _, macro_type_predicted = torch.max(macro_type_outputs.logits, 1)

        for i, id in enumerate(ids):
            spin_prediction = int(spin_predicted[i].item())
            macro_type_prediction = int(macro_type_predicted[i].item())
            if annotations[id]['spin'] == '':
                annotations[id]['spin'] = label_maps['spin'][spin_prediction]
            if annotations[id]['macro_type'] == '':
                annotations[id]['macro_type'] = label_maps['macro_type'][macro_type_prediction]


    # clean dictionary, save to csv 
    output_dict = {k: [v['type'], v['macro_type'], v['spin'] ] for k, v in annotations.items()}
    save_progress(output_dict, 'outputs/annotations_final')
    pd.DataFrame.from_dict(output_dict,
                        orient='index',
                        columns=['type', 'macro_type', 'spin']
                        ).to_csv('annotations.csv')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--db', type=str, required=True, help='path to database file from top level directory')
    parser.add_argument('--ns', type=int, default=10, help='number of article samples to load and label')
    args = parser.parse_args()
    main(args)

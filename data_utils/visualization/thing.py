import pickle
import argparse
import os
import sqlite3

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import data_utils.dataset as d
from sklearn.metrics import f1_score

from models.roberta_classifier.train_quant import label_maps
from models.roberta_classifier.quant_utils import QuantModel, TextClassificationDataset, test
import models.roberta_classifier.train_test_utils as tt

SPLIT_DIR = "data/clean/"
MODEL_CHECKPOINT = "models/roberta_classifier/tuned_models/acl_models/masked_folds/"

news_sites = ['foxnews', 'breitbart', 'nytimes', 'washingtonpost', 'huffpost', 'wsj', 'bbc']

def get_site(db, quant_id):
    con = sqlite3.connect(db)
    cur = con.cursor()
    query = 'select source from article where id = {}'.format(quant_id)
    res = cur.execute(query)
    site = res.fetchone()[0]
    return site

def get_texts(
              annotation_component: str,
              task: str,
              qual_dict: {},
              quant_dict: {},
              article_ids: [],
              db: str,
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
    sites = []

    for id in article_ids:
        if 'quant_list' in qual_dict[id].keys():
            article_site = get_site(db, id)
            for quant_id in qual_dict[id]['quant_list']:
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

                        sites.append(article_site)

    return texts, labels, sites

def main(args):

    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(SPLIT_DIR + 'quant_dict_clean', 'rb'))

    type_filters = {
        'type': [],
        'type-binary': [],
        'spin': [],
        'macro_type': []
    }

    results = {}
    for task in label_maps.keys():
        results[task] = {}
        for site in news_sites:
            results[task][site] = {}
            results[task][site]['labels'] = []
            results[task][site]['predictions'] = []

    for k, split in splits_dict.items():
        print("Fold " + str(k+1) + " of 5")
        print()
        
        split_test_ids = split['test']
        

        for task in list(label_maps.keys()):
            ann_component = task.split('-')[0]

            fold = 'fold{}'.format(k)
            task_name = task + '_model'
            model_checkpoint = os.path.join(MODEL_CHECKPOINT, fold, 'quant', task_name)

            test_texts, test_labels, curr_sites = \
                get_texts(ann_component,
                          task,
                          qual_dict,
                          quant_dict,
                          split_test_ids,
                          args.db,
                          type_filter=type_filters[task])
        

            tokenizer = RobertaTokenizer\
                .from_pretrained(pretrained_model_name_or_path='roberta-base',
                                 problem_type="single_label_classification")

            max_length = 512
            test_data = TextClassificationDataset(texts=test_texts,
                                                  labels=test_labels,
                                                  tokenizer=tokenizer,
                                                  max_length=max_length)

            batch_size = 8
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

            # Define model
            annotation_map = label_maps[task]
            num_labels = len(set(annotation_map.values()))
            model = QuantModel('roberta-base', num_labels=num_labels).to('cuda')
            model = model.from_pretrained(model_checkpoint, task).to('cuda')

            y, y_predicted, f1 = test(model,
                                      test_loader)

            for i, site in enumerate(curr_sites):
                results[task][site]['labels'].append(y[i])
                results[task][site]['predictions'].append(y_predicted[i])
        
    for task in label_maps.keys():
        print(task)
        for site in news_sites:
            # dest = f"models/roberta_classifier/result_scripts/site_results/{site}/"

            # os.makedirs(dest, exist_ok=True)

            # d.to_csv(task,
            #         results[task][site]['labels'],
            #         results[task][site]['predictions'],
            #         dest)
            print(site)
            macro_f1 = f1_score(results[task][site]['labels'], results[task][site]['predictions'], average='macro')
            print(round(macro_f1, 3))
            weighted_f1 = f1_score(results[task][site]['labels'], results[task][site]['predictions'], average='weighted')
            print(round(weighted_f1, 3))

        print('\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to db file")
    parser.add_argument("--model", required=False, default="roberta-base", help="model checkpoint")
    args = parser.parse_args()
    main(args)
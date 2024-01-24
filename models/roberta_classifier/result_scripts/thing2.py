import pickle
import argparse
import os
import sqlite3

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import models.utils.dataset as d
from sklearn.metrics import f1_score

from models.roberta_classifier.train_qual import label_maps
import models.roberta_classifier.train_test_utils as tt
import data_utils.get_annotation_stats as gs

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

def get_texts(db_filename: str,
              annotation_component: str,
              task: str,
              agreed_anns_dict: {},
              article_ids: []
              ):

    texts = []
    labels = []
    sites = []

    for id in article_ids:
        if annotation_component in agreed_anns_dict[id].keys():
            if agreed_anns_dict[id][annotation_component] !='\x00':
                texts.append(gs.get_text(id, db_filename, clean=False))
                label = agreed_anns_dict[id][annotation_component]
                labels.append(label_maps[task][label])
                sites.append(get_site(db_filename, id))

    return texts, labels, sites

def main(args):

    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))

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
            model_checkpoint = os.path.join(MODEL_CHECKPOINT, fold, 'qual', task_name)

            test_texts, test_labels, curr_sites = \
                get_texts(args.db,
                          ann_component,
                          task,
                          qual_dict,
                          split_test_ids
                          )
        
            test_texts = [t.replace('\n', '') for t in test_texts]

            tokenizer = RobertaTokenizer\
                .from_pretrained(pretrained_model_name_or_path='roberta-base',
                                 problem_type="single_label_classification")

            max_length = 512
            test_data = tt.TextClassificationDataset(texts=test_texts,
                                                     labels=test_labels,
                                                     tokenizer=tokenizer,
                                                     max_length=max_length)

            batch_size = 8
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

            model = RobertaForSequenceClassification.from_pretrained(model_checkpoint).to('cuda')


            y, y_predicted, f1 = tt.test(model,
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
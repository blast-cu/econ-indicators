import pickle
import argparse
import os
import sqlite3
from statistics import mode

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import models.utils.dataset as d
from sklearn.metrics import f1_score

from models.roberta_classifier.train_quant import label_maps, get_texts
from models.roberta_classifier.quant_utils import QuantModel, TextClassificationDataset, test
import models.roberta_classifier.train_test_utils as tt

SPLIT_DIR = "data/clean/"
MODEL_CHECKPOINT = "models/roberta_classifier/tuned_models/acl_models/masked_folds/"


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
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():
        print("Fold " + str(k+1) + " of 5")
        
        split_test_ids = split['test']
        

        for task in list(label_maps.keys()):
            ann_component = task.split('-')[0]

            print("Task: " + task)

            fold = 'fold{}'.format(k)
            task_name = task + '_model'
            model_checkpoint = os.path.join(MODEL_CHECKPOINT, fold, 'quant', task_name)

            test_texts, test_labels = \
                get_texts(ann_component,
                          task,
                          qual_dict,
                          quant_dict,
                          split_test_ids,
                          type_filter=type_filters[task])
            
            print("Number of test examples: " + str(len(test_texts)) + '\n')
        

            majority_label = mode(test_labels)
            y_predicted = [majority_label for i in range(len(test_labels))]

            results[task]['labels'] += test_labels
            results[task]['predictions'] += y_predicted


            

        
    # for task in label_maps.keys():
    #     print(task)

    #         # dest = f"models/roberta_classifier/result_scripts/site_results/{site}/"

    #         # os.makedirs(dest, exist_ok=True)

    #         # d.to_csv(task,
    #         #         results[task][site]['labels'],
    #         #         results[task][site]['predictions'],
    #         #         dest)

    #     macro_f1 = f1_score(results[task]['labels'], results[task]['predictions'], average='macro')
    #     print(round(macro_f1, 3))
    #     weighted_f1 = f1_score(results[task]['labels'], results[task]['predictions'], average='weighted')
    #     print(round(weighted_f1, 3))

    #     print('\n\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to db file")
    parser.add_argument("--model", required=False, default="roberta-base", help="model checkpoint")
    args = parser.parse_args()
    main(args)
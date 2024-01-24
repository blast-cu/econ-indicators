import pickle
import argparse
import os
import sqlite3
from statistics import mode

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import models.utils.dataset as d
from sklearn.metrics import f1_score

from models.roberta_classifier.train_qual import label_maps, get_texts
import models.roberta_classifier.train_test_utils as tt
import data_utils.get_annotation_stats as gs

SPLIT_DIR = "data/clean/"


def main(args):

    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))

    results = {}
    for task in label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():
        print("Fold " + str(k+1) + " of 5")
        print()
        
        split_test_ids = split['test']
        

        for task in list(label_maps.keys()):
            ann_component = task.split('-')[0]

            print("Task: " + task)

            fold = 'fold{}'.format(k)
            task_name = task + '_model'

            test_texts, test_labels = \
                get_texts(args.db,
                          ann_component,
                          task,
                          qual_dict,
                          split_test_ids
                          )
        
            majority_label = mode(test_labels)
            y_predicted = [majority_label for i in range(len(test_labels))]

            print("Number of test examples: " + str(len(test_texts)) + '\n')

            results[task]['labels'] += test_labels
            results[task]['predictions'] += y_predicted
        
    # for task in label_maps.keys():
    #     print(task)
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
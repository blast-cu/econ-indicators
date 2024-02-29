import pickle
import argparse
import os
from statistics import mode
import random

import data_utils.model_utils.dataset as d
from data_utils.model_utils.dataset import quant_label_maps, qual_label_maps
import models.utils.quant as qu
import models.roberta_classifier.qual_utils as tt

SPLIT_DIR = "data/clean/"

def main(args):

    random.seed(42)

    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(SPLIT_DIR + 'quant_dict', 'rb'))

    type_filters = {
        'type': [],
        'type-binary': [],
        'spin': [],
        'macro_type': []
    }

    results = {}
    for task in qual_label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():
        split_test_ids = split['test']
        for task in list(qual_label_maps.keys()):
            ann_component = task.split('-')[0]

            _, test_labels = \
                tt.get_texts(args.db,
                             ann_component,
                             task,
                             qual_dict,
                             split_test_ids
                            )
        
            # majority_label = mode(test_labels)
            # y_predicted = [majority_label for i in range(len(test_labels))]
            y_predicted = random.choices(list(qual_label_maps[task].values()), k=len(test_labels))


            results[task]['labels'] += test_labels
            results[task]['predictions'] += y_predicted

    for task in quant_label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():
        
        split_test_ids = split['test']
        

        for task in list(quant_label_maps.keys()):
            ann_component = task.split('-')[0]

            test_texts, test_labels = \
                qu.get_texts(ann_component,
                             task,
                             qual_dict,
                             quant_dict,
                             split_test_ids,
                             type_filter=type_filters[task])

            # majority_label = mode(test_labels)
            # y_predicted = [majority_label for i in range(len(test_labels))]
            y_predicted = random.choices(list(quant_label_maps[task].values()), k=len(test_labels))

            results[task]['labels'] += test_labels
            results[task]['predictions'] += y_predicted

    dest = f"data_utils/table_generators/results/random_label/"
    os.makedirs(dest, exist_ok=True)

    d.to_f1_csv(results, dest, 'macro')
    d.to_f1_csv(results, dest, 'weighted')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to db file")
    parser.add_argument("--model", required=False, default="roberta-base", help="model checkpoint")
    args = parser.parse_args()
    main(args)
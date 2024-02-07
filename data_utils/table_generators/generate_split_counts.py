import pickle
import pandas as pd

import data_utils.dataset as d
from sklearn.metrics import f1_score
from data_utils.dataset import DB_FILENAME, quant_label_maps, qual_label_maps
import models.roberta_classifier.quant_utils as qtu
import models.roberta_classifier.train_test_utils as tt


SPLIT_DIR = "data/clean/"

def main():

    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(SPLIT_DIR + 'quant_dict', 'rb'))

    noisy_qual_dict = pickle.load(open(SPLIT_DIR + 'noisy_qual_dict', 'rb'))
    noisy_quant_dict = pickle.load(open(SPLIT_DIR + 'noisy_quant_dict', 'rb'))

    split_counts = {}

    reverse_keys = splits_dict.keys()
    list(reverse_keys).reverse()
    for k in reverse_keys:
        split = splits_dict[k]
        
        split_test_ids = split['test']

        for task in list(qual_label_maps.keys()):
            ann_component = task.split('-')[0]

            agreed_key = task + "_agreed"
            noisy_key = task + "_noisy"
            if agreed_key not in split_counts.keys():
                split_counts[agreed_key] = []
                split_counts[noisy_key] = []

            _, test_labels = \
                tt.get_texts(DB_FILENAME,
                             ann_component,
                             task,
                             qual_dict,
                             split_test_ids
                             )
            _, noise_labels = \
                tt.get_noise(DB_FILENAME,
                             ann_component,
                             task,
                             noisy_qual_dict
                             )

            split_counts[agreed_key].append(len(test_labels))
            split_counts[noisy_key].append(len(noise_labels))
        

    type_filters = {
        'type': [],
        'type-binary': [],
        'spin': [],
        'macro_type': []
    }

    reverse_keys = splits_dict.keys()
    list(reverse_keys).reverse()
    for k in reverse_keys:
        split = splits_dict[k]

        split_test_ids = split['test']

        for task in list(quant_label_maps.keys()):

            agreed_key = task + "_agreed"
            noisy_key = task + "_noisy"

            if agreed_key not in split_counts.keys():
                split_counts[agreed_key] = []
                split_counts[noisy_key] = []

            ann_component = task.split('-')[0]

            _, test_labels = \
                qtu.get_texts(ann_component,
                          task,
                          qual_dict,
                          quant_dict,
                          split_test_ids,
                          type_filter=type_filters[task])

            _, test_labels_noisy = \
                qtu.get_noise(ann_component,
                              task,
                              noisy_quant_dict,
                              split_test_ids)
            
            split_counts[agreed_key].append(len(test_labels))
            split_counts[noisy_key].append(len(test_labels_noisy))

    pd.DataFrame(split_counts).to_csv("data_utils/table_generators/results/split_counts.csv", index=True, index_label="Fold")


if __name__ == "__main__":
    main()
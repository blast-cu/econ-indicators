from sklearn.model_selection import KFold
import argparse
import pickle
import os

import models.roberta_classifier.train_test_utils as tt
import models.utils.dataset as d
from data_utils import get_annotation_stats as gs

SETTING = "roberta_noise"
# SETTING = "test"
OUT_DIR = "models/roberta_classifier/tuned_models/qual_" + SETTING + "/"
SPLIT_DIR = "data/clean/"
# MODEL_CHECKPOINT = "models/roberta_classifier/tuned_models/masked"
MODEL_CHECKPOINT = "roberta-base"
DB_FILENAME = "data/data.db"

ADD_NOISE = True

label_maps = {
    'frame': {
            'business': 0,
            'industry': 1,
            'macro': 2,
            'government': 3,
            'other': 4},
    # 'frame-binary': {
    #         'business': 0,
    #         'industry': 1,
    #         'macro': 1,
    #         'government': 1,
    #         'other': 1},
    'econ_rate': {
            'good': 0,
            'poor': 1,
            'none': 2},
    'econ_change': {
            'better': 0,
            'worse': 1,
            'same': 2,
            'none': 3}
}

def get_noise(db_filename: str,
              annotation_component: str,
              task: str,
              noise_dict: {}
             ):
    
    texts = []
    labels = []

    for id in noise_dict.keys():
        if noise_dict[id][annotation_component] !='\x00':
            for label in noise_dict[id][annotation_component]:
                texts.append(gs.get_text(id, db_filename, clean=False))
                labels.append(label_maps[task][label])
    
    return texts, labels


def get_texts(db_filename: str,
              annotation_component: str,
              task: str,
              agreed_anns_dict: {},
              article_ids: []):

    texts = []
    labels = []

    for id in article_ids:
        if annotation_component in agreed_anns_dict[id].keys():
            if agreed_anns_dict[id][annotation_component] !='\x00':
                texts.append(gs.get_text(id, db_filename, clean=False))
                label = agreed_anns_dict[id][annotation_component]
                labels.append(label_maps[task][label])

    return texts, labels



# def main(args):
def main():
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file. The best model for each task is exported to 
    models/roberta/best_models.
    """

    split_dir = "data/clean/"
    splits_dict = pickle.load(open(split_dir + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))

    results = {}
    for task in label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []


    for k, split in splits_dict.items():

        print("Fold " + str(k+1) + " of 5")
        print()
        
        split_train_ids = split['train']
        split_test_ids = split['test']

        for task in list(label_maps.keys()):

            annotation_component = task.split('-')[0]

            train_texts, train_labels = \
                get_texts(DB_FILENAME,
                          annotation_component,
                          task,
                          qual_dict,
                          split_train_ids)
            
            if ADD_NOISE:
                noise_dict = pickle.load(open(split_dir + 'noisy_qual_dict', 'rb'))
                noise_ids = list(noise_dict.keys())
                noise_text, noise_labels = \
                    get_noise(DB_FILENAME,
                              annotation_component,
                              task,
                              noise_dict
                              )
                # print(">>> Number Noise texts: " + str(len(noise_text)))
                # print(">>> Number Noise labels: " + str(len(noise_labels)))
                train_texts += noise_text
                train_labels += noise_labels
            
            test_texts, test_labels = \
                get_texts(DB_FILENAME,
                          annotation_component,
                          task,
                          qual_dict,
                          split_test_ids)
            
            class_weights = tt.get_weights(
                train_labels,
                label_maps[task])

            print(">>> Annotation component: " + annotation_component)
            print(">>> Number Train texts: " + str(len(train_texts)))
            print(">>> Number Test texts: " + str(len(test_texts)))
            
            train_texts = [t.replace('\n', '') for t in train_texts]
            test_texts = [t.replace('\n', '') for t in test_texts]

            model, train_loader, val_loader, test_loader, optimizer = \
                tt.setup(train_texts,
                         test_texts,
                         train_labels,
                         test_labels,
                         label_maps[task],
                         model_checkpoint=MODEL_CHECKPOINT)

            tuned_model = tt.train(model, train_loader, val_loader,
                                   optimizer, class_weights)
            y, y_predicted, f1 = tt.test(tuned_model, test_loader)

            results[task]['labels'] += y
            results[task]['predictions'] += y_predicted

            print(">>> Labels: " + str(y))
            print(">>> Predictions: " + str(y_predicted))
            print('\n\n')

            dest = f"{OUT_DIR}fold{k}/"
            os.makedirs(dest, exist_ok=True)

            d.to_csv(
                task,
                y,
                y_predicted,
                dest)

            model_dest = dest + task + "_model"
            model.save_pretrained(model_dest)
            
    for task in label_maps.keys():
        dest = f"{OUT_DIR}results/"

        os.makedirs(dest, exist_ok=True)

        d.to_csv(task,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)
        
        d.to_f1_csv(results, dest, f1='macro')
        d.to_f1_csv(results, dest, f1='weighted')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--db", required=True, help="path to db file")
    # parser.add_argument("--model", required=False, default="roberta-base", help="model checkpoint")
    # args = parser.parse_args()
    # main(args)
    main()

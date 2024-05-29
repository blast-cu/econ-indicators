import argparse
import pickle
import os

import models.roberta_classifier.utils.qual as tt
import models.roberta_classifier.utils.general as gu
import data_utils.model_utils.dataset as d
import data_utils.model_utils.eval as e


def main(args):
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file. The best model for each task is exported to 
    models/roberta/best_models.
    """
    OUT_DIR, MODEL_CHECKPOINT, ADD_NOISE, BEST_NOISE = gu.settings(args, "qual")
    split_dir = "data/clean/"
    splits_dict = pickle.load(open(split_dir + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))

    results = {}
    for task in d.qual_label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():

        print("Fold " + str(k+1) + " of 5")
        print()
        
        split_train_ids = split['train']
        split_test_ids = split['test']

        for task in list(d.qual_label_maps.keys()):

            annotation_component = task.split('-')[0]

            train_texts, train_labels = \
                tt.get_texts(d.DB_FILENAME,
                             annotation_component,
                             task,
                             qual_dict,
                             split_train_ids)

            if ADD_NOISE:  # add noise to training set
                if BEST_NOISE:
                    f = open(d.SPLIT_DIR + 'noisy_best_qual_dict', 'rb')
                else:
                    f = open(d.SPLIT_DIR + 'noisy_qual_dict', 'rb')

                noise_dict = pickle.load(f)
                noise_text, noise_labels = \
                    tt.get_noise(d.DB_FILENAME,
                                 annotation_component,
                                 task,
                                 noise_dict)

                # print(">>> Number Noise texts: " + str(len(noise_text)))
                # print(">>> Number Noise labels: " + str(len(noise_labels)))
                train_texts += noise_text
                train_labels += noise_labels

            test_texts, test_labels = \
                tt.get_texts(d.DB_FILENAME,
                             annotation_component,
                             task,
                             qual_dict,
                             split_test_ids)

            class_weights = gu.get_weights(
                train_labels,
                d.qual_label_maps[task])

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
                         d.qual_label_maps[task],
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

            e.to_csv(
                task,
                y,
                y_predicted,
                dest)

            model_dest = dest + task + "_model"
            model.save_pretrained(model_dest)

    for task in d.qual_label_maps.keys():
        dest = f"{OUT_DIR}results/"

        os.makedirs(dest, exist_ok=True)

        e.to_csv(task,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)

    e.to_f1_csv(results, dest, f1='macro')
    e.to_f1_csv(results, dest, f1='weighted')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--m', required=True, help='Model checkpoint: "base" or "dapt"')
    parser.add_argument('--en', default=None, help='extra name for output directory')
    parser.add_argument('--n', required=False, help='Noise setting: "best" or "all". No noise will be added to training set if not specified.')
    args = parser.parse_args()
    main(args)

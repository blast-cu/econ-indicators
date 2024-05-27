import argparse
import pickle
import os

import models.roberta_classifier.utils.general as gu
import models.roberta_classifier.utils.quant as qu
import data_utils.model_utils.dataset as d
import data_utils.model_utils.eval as e

# takes about 12 hours to run on CURC


def main(args):
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file.
    """
    OUT_DIR, MODEL_CHECKPOINT, ADD_NOISE, BEST_NOISE = \
        gu.settings(args, "quant")

    splits_dict = pickle.load(open(d.SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(d.SPLIT_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(d.SPLIT_DIR + 'quant_dict', 'rb'))

    # dict for tracking results across folds
    results = {}
    for task in d.quant_label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():

        print("Fold " + str(k+1) + " of 5")
        print()

        split_train_ids = split['train']
        split_test_ids = split['test']

        for task in list(d.quant_label_maps.keys()):

            ann_component = task.split('-')[0]

            train_texts, train_labels = \
                qu.get_texts(ann_component,
                             task,
                             qual_dict,
                             quant_dict,
                             split_train_ids)

            if ADD_NOISE:
                if BEST_NOISE:
                    f = open(d.SPLIT_DIR + 'noisy_best_quant_dict', 'rb')
                else:
                    f = open(d.SPLIT_DIR + 'noisy_quant_dict', 'rb')
                
                noise_dict = pickle.load(f)

                noise_text, noise_labels = \
                    qu.get_noise(ann_component,
                                 task,
                                 noise_dict,
                                 split_test_ids)

                train_texts += noise_text
                train_labels += noise_labels

            test_texts, test_labels = \
                qu.get_texts(ann_component,
                             task,
                             qual_dict,
                             quant_dict,
                             split_test_ids)

            # gets class weights for loss function
            class_weights = gu.get_weights(train_labels,
                                           d.quant_label_maps[task])

            model, train_loader, val_loader, test_loader, optimizer = \
                qu.setup(train_texts,
                         test_texts,
                         train_labels,
                         test_labels,
                         d.quant_label_maps[task],
                         model_checkpoint=MODEL_CHECKPOINT)

            tuned_model = qu.train(model,
                                   train_loader,
                                   val_loader,
                                   optimizer,
                                   class_weights)

            y, y_predicted, f1 = qu.test(tuned_model,
                                         test_loader)

            results[task]['labels'] += y
            results[task]['predictions'] += y_predicted

            print(">>> Labels:\t" + str(y))
            print(">>> Predictions:\t" + str(y_predicted))
            print('\n\n')

            dest = os.path.join(OUT_DIR, f"fold{k}/{task}_model/")
            os.makedirs(dest, exist_ok=True)

            e.to_csv(
                task,
                y,
                y_predicted,
                dest)

            tuned_model.save(dest, task)

    for task in d.quant_label_maps.keys():
        dest = os.path.join(OUT_DIR, "results")
        os.makedirs(dest, exist_ok=True)
        dest += "/"

        e.to_csv(task,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)

    e.to_f1_csv(results, dest, f1='macro')
    e.to_f1_csv(results, dest, f1='weighted')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--m', required=True, help='Model checkpoint: "base" or "dapt"')
    parser.add_argument('--n', default=None, help='Noise setting: "best" or "all". No noise will be added to training set if not specified.')
    args = parser.parse_args()
    main(args)

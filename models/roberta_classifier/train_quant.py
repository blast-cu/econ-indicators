import argparse
import pickle
import os

import models.roberta_classifier.train_test_utils as tt
import data_utils.dataset as d
from data_utils.dataset import quant_label_maps as label_maps
import models.roberta_classifier.quant_utils as qu

# takes about 12 hours to run on CURC
SPLIT_DIR = "data/clean/"


def settings(args):

    global SETTING
    SETTING = args.s
    global OUT_DIR
    OUT_DIR = "models/roberta_classifier/tuned_models/quant_" + SETTING + "/"

    global MODEL_CHECKPOINT
    MODEL_CHECKPOINT = None
    global ADD_NOISE
    ADD_NOISE = False
    global BEST_NOISE
    BEST_NOISE = False

    if 'dapt' in SETTING:
        MODEL_CHECKPOINT = "data/masked/"
    elif 'base' in SETTING:
        MODEL_CHECKPOINT = "roberta-base"
    else:
        raise ValueError("Invalid setting: {}".format(SETTING))

    if 'noise' in SETTING:
        ADD_NOISE = True
        if 'best' in SETTING:
            BEST_NOISE = True
        elif 'all' in SETTING:
            BEST_NOISE = False
        else:
            raise ValueError("Invalid setting: {}".format(SETTING))


def main():
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file.
    """
    
    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(SPLIT_DIR + 'quant_dict', 'rb'))

    type_filters = {
        'type': [],
        'type-binary': [],
        'spin': [],
        'macro_type': []
    }

    # dict for tracking results across folds
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

            ann_component = task.split('-')[0]

            train_texts, train_labels = \
                qu.get_texts(ann_component,
                             task,
                             qual_dict,
                             quant_dict,
                             split_train_ids,
                             type_filter=type_filters[task])

            if ADD_NOISE:
                if BEST_NOISE:
                    f = open(SPLIT_DIR + 'noisy_best_quant_dict', 'rb')
                else:
                    f = open(SPLIT_DIR + 'noisy_quant_dict', 'rb')
                
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
                             split_test_ids,
                             type_filter=type_filters[task])

            # gets class weights for loss function
            class_weights = tt.get_weights(train_labels,
                                           label_maps[task])

            model, train_loader, val_loader, test_loader, optimizer = \
                qu.setup(train_texts,
                         test_texts,
                         train_labels,
                         test_labels,
                         label_maps[task],
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

            d.to_csv(
                task,
                y,
                y_predicted,
                dest)

            tuned_model.save(dest, task)

    for task in label_maps.keys():
        dest = os.path.join(OUT_DIR, "results")
        os.makedirs(dest, exist_ok=True)
        dest += "/"

        d.to_csv(task,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)

    d.to_f1_csv(results, dest, f1='macro')
    d.to_f1_csv(results, dest, f1='weighted')

if __name__ == "__main__":
    main()
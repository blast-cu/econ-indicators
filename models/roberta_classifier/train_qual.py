from sklearn.model_selection import KFold
import argparse
import pickle
import os

import models.roberta_classifier.train_test_utils as tt
import models.utils.dataset as d
from data_utils import get_annotation_stats as gs

label_maps = {
    'frame': {
            'business': 0,
            'industry': 1,
            'macro': 2,
            'government': 3,
            'other': 4},
    'frame-binary': {
            'business': 0,
            'industry': 1,
            'macro': 1,
            'government': 1,
            'other': 1},
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

def get_texts(db_filename: str,
              annotation_component: str,
              task: str,
              agreed_anns_dict: {},
              article_ids: []):

    texts = []
    labels = []

    for id in article_ids:
        if annotation_component in agreed_anns_dict[id].keys():
            texts.append(gs.get_text(id, db_filename, clean=False))
            label = agreed_anns_dict[id][annotation_component]
            labels.append(label_maps[task][label])

    return texts, labels



def main(args):
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file. The best model for each task is exported to 
    models/roberta/best_models.
    """

    split_dir = "models/utils/splits/"
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
                get_texts(args.db,
                          annotation_component,
                          task,
                          qual_dict,
                          split_train_ids)
            
            test_texts, test_labels = \
                get_texts(args.db,
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
            

            model, train_loader, val_loader, test_loader, optimizer = \
                tt.setup(train_texts,
                         test_texts,
                         train_labels,
                         test_labels,
                         label_maps[task],
                         model_checkpoint=args.model)

            tuned_model = tt.train(model, train_loader, val_loader,
                                   optimizer, class_weights)
            y, y_predicted, f1 = tt.test(tuned_model, test_loader)

            results[task]['labels'] += y
            results[task]['predictions'] += y_predicted

            print(">>> Labels: " + str(y))
            print(">>> Predictions: " + str(y_predicted))
            print('\n\n')

            dest = f"models/roberta_classifier/best_models/fold{k}/qual/"

            if not os.path.isdir(dest): 
                os.makedirs(dest)

            d.to_csv(
                task,
                y,
                y_predicted,
                dest)

            model_dest = dest + task + "_model"
            model.save_pretrained(model_dest)
            
    for task in label_maps.keys():
        dest = f"models/roberta_classifier/best_models/results/"
        if not os.path.isdir(dest):
            os.makedirs(dest)
        d.to_csv(task,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)


            

























    # k_folds = 5  # 4 train folds, 1 test fold

    # for task in list(label_maps.keys()):

    #     annotation_component = task.split('-')[0]

    #     best_f1 = 0 # best macro f1 score
    #     best_model = None
    #     best_labels = None
    #     best_predicted = None


    #     texts, labels = \
    #         d.load_qual_dataset(args.db,
    #                             annotation_component=annotation_component,
    #                             label_map=label_maps[task])

    #     kf = KFold(n_splits=5, random_state=42, shuffle=True)

    #     y_labels_tot = []
    #     y_predicted_tot = []

    #     for i, (train_index, test_index) in enumerate(kf.split(texts)):

    #         test_texts = [texts[i] for i in test_index]
    #         test_labels = [labels[i] for i in test_index]

    #         train_texts = [texts[i] for i in train_index]
    #         train_labels = [labels[i] for i in train_index]

    #         class_weights = tt.get_weights(train_labels,
    #                                        label_maps[task])

    #         print(f"\nFold {i+1}/{k_folds}")

    #         model, train_loader, val_loader, test_loader, optimizer = \
    #             tt.setup(train_texts,
    #                      test_texts,
    #                      train_labels,
    #                      test_labels,
    #                      label_maps[task],
    #                      model_checkpoint=args.model)

    #         tuned_model = tt.train(model, train_loader, val_loader,
    #                                optimizer, class_weights)
    #         y, y_predicted, f1 = tt.test(tuned_model, test_loader)

    #         if f1 > best_f1:
    #             best_f1 = f1
    #             best_model = tuned_model
    #             best_labels = y
    #             best_predicted = y_predicted

    #         y_labels_tot += y
    #         y_predicted_tot += y_predicted

    #     print("All labels: " + str(y_labels_tot))
    #     print("All predictions: " + str(y_predicted_tot))

    #     destination = "models/roberta_classifier/results/qual/"
    #     d.to_csv(task,
    #              y_labels_tot,
    #              y_predicted_tot,
    #              destination)

    #     # save best model and corresponding report to csv
    #     best_dest = "models/roberta_classifier/best_models/qual/"
    #     model_dest = best_dest + task + "_model"

    #     best_model.save_pretrained(model_dest)
    #     d.to_csv(task, best_labels, best_predicted, best_dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to database")
    parser.add_argument("--model", required=False, default="roberta-base", help="model checkpoint")
    args = parser.parse_args()
    main(args)

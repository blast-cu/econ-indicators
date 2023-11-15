from sklearn.model_selection import KFold
import argparse

import models.roberta_classifier.train_test_utils as tt
import models.utils.dataset as d

label_maps = {
    'frame': {
            'business': 0,
            'industry': 1,
            'macro': 2,
            'government': 3,
            'other': 4},
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


def main(args):
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file. The best model for each task is exported to 
    models/roberta/best_models.
    """

    k_folds = 5  # 4 train folds, 1 test fold

    for annotation_component in list(label_maps.keys()):

        best_f1 = 0 # best macro f1 score
        best_model = None
        best_labels = None
        best_predicted = None

        texts, labels = \
            d.load_qual_dataset(args.db,
                                annotation_component=annotation_component,
                                label_map=label_maps[annotation_component])

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        y_labels_tot = []
        y_predicted_tot = []

        for i, (train_index, test_index) in enumerate(kf.split(texts)):

            test_texts = [texts[i] for i in test_index]
            test_labels = [labels[i] for i in test_index]

            train_texts = [texts[i] for i in train_index]
            train_labels = [labels[i] for i in train_index]

            class_weights = tt.get_weights(train_labels,
                                           label_maps[annotation_component])

            print(f"\nFold {i+1}/{k_folds}")

            model, train_loader, val_loader, test_loader, optimizer = \
                tt.setup(train_texts,
                         test_texts,
                         train_labels,
                         test_labels,
                         label_maps[annotation_component])

            tuned_model = tt.train(model, train_loader, val_loader,
                                   optimizer, class_weights)
            y, y_predicted, f1 = tt.test(tuned_model, test_loader)

            if f1 > best_f1:
                best_f1 = f1
                best_model = tuned_model
                best_labels = y
                best_predicted = y_predicted

            y_labels_tot += y
            y_predicted_tot += y_predicted

        print("All labels: " + str(y_labels_tot))
        print("All predictions: " + str(y_predicted_tot))

        destination = "models/roberta_classifier/results/qual/"
        d.to_csv(annotation_component,
                 y_labels_tot,
                 y_predicted_tot,
                 destination)

        # save best model and corresponding report to csv
        best_dest = "models/roberta_classifier/best_models/qual/"
        model_dest = best_dest + annotation_component + "_model"

        best_model.save_pretrained(model_dest)
        d.to_csv(annotation_component, best_labels, best_predicted, best_dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)

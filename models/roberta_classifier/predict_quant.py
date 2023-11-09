from sklearn.model_selection import KFold
import train_test as tt
import models.utils.dataset as d

# nltk.download('punkt')

# maps annotation labels to integers for each prediction task
label_maps = {
    'type': {
            'industry': 0,
            'macro': 1},
    'spin': {
            'pos': 0,
            'neg': 1,
            'neutral': 2},
    'macro_type': {
            'jobs': 0,
            'retail': 1,
            'interest': 2,
            'prices': 3,
            'energy': 4,
            'wages': 5,
            'macro': 6,
            'market': 7,
            'currency': 8,
            'housing': 9,
            'other': 10}
}


def main():
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file.
    """

    k_folds = 5  # 4 train folds, 1 test fold

    # restrict dataset to annotations with agreed types
    type_filters = {
        'type': ['industry', 'macro'],
        'spin': ['industry', 'macro'],
        'macro_type': ['macro']
    }

    experiments = []
    for task in label_maps.keys():
        task_texts, task_labels = \
            d.load_quant_dataset("data/data.db", label_ann=task,
                                 label_map=label_maps[task],
                                 type_filter=type_filters[task])

        experiments.append((task_texts, task_labels, task))

    for texts, labels, task in experiments:

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        y_labels_tot = []
        y_predicted_tot = []

        for i, (train_index, test_index) in enumerate(kf.split(texts)):

            test_texts = [texts[i] for i in test_index]
            test_labels = [labels[i] for i in test_index]

            train_texts = [texts[i] for i in train_index]
            train_labels = [labels[i] for i in train_index]

            class_weights = tt.get_weights(train_labels, label_maps[task])

            print(f"\nFold {i+1}/{k_folds}")

            model, train_loader, val_loader, test_loader, optimizer = \
                tt.setup(train_texts, test_texts, train_labels, test_labels,
                         label_maps[task])

            # test before pretraining
            # tt.test(model, test_loader)

            tuned_model = tt.train(model, train_loader, val_loader,
                                   optimizer, class_weights)
            y, y_predicted, accuracy = tt.test(tuned_model, test_loader)

            y_labels_tot += y
            y_predicted_tot += y_predicted

        print(y_labels_tot)
        print(y_predicted_tot)

        destination = "models/roberta/results/quant"
        d.to_csv(task, y_labels_tot, y_predicted_tot, destination)
        # model.save_pretrained(f"models/roberta/{annotation_component}_model")


if __name__ == '__main__':
    main()

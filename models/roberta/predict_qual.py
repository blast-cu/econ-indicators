import data_utils.get_annotation_stats as gs
from sklearn.model_selection import KFold

import train_test as tt

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


def load_dataset(db_filename: str, annotation_component: str):
    """
    Load dataset from a database file and return text and labels for a given annotation component.

    Parameters:
    db_filename (str): The path to the database file.
    annotation_component (str): The annotation component to extract labels for.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two lists: text and labels.
    """
    
    label_map = label_maps[annotation_component]

    text = []
    labels = []

    # get all agreed annotations for given component
    qual_ann = gs.get_qual_dict(db_filename)  
    agreed_qual_ann = gs.get_agreed_anns(qual_ann)

    # get list of text and labels for given component
    for article_id in agreed_qual_ann.keys():
        if agreed_qual_ann[article_id][annotation_component] != '\0':
            article_dict = agreed_qual_ann[article_id]
            clean_text = gs.get_text(article_id, db_filename)

            text.append(clean_text)
            label = label_map[article_dict[annotation_component]]
            labels.append(label)

    return text, labels


def main():
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file.
    """

    k_folds = 5  # 4 train folds, 1 test fold

    for annotation_component in list(label_maps.keys()): 

        texts, labels = load_dataset("data/data.db",
                                     annotation_component=annotation_component)

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
                tt.setup(train_texts, test_texts, train_labels, test_labels,
                         label_maps[annotation_component])

            # test before pretraining
            # tt.test(model, test_loader)

            tuned_model = tt.train(model, train_loader, val_loader,
                                   optimizer, class_weights)
            y, y_predicted, accuracy = tt.test(tuned_model, test_loader)

            y_labels_tot.append(y)
            y_predicted_tot.append(y_predicted)

        y_labels_tot = [label for sublist in y_labels_tot for label in sublist]
        y_predicted_tot = [label for sublist in y_labels_tot for label in sublist]

        print(y_labels_tot)
        print(y_predicted_tot)

        destination = "models/roberta/results/test_refactor"
        tt.to_csv(annotation_component, y_labels_tot, y_predicted_tot,
                  destination)
        # model.save_pretrained(f"models/roberta/{annotation_component}_model")


if __name__ == '__main__':
    main()

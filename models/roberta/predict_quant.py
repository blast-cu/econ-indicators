from bs4 import BeautifulSoup
import nltk
from sklearn.model_selection import KFold

import data_utils.get_annotation_stats as gs
import train_test as tt

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


def get_article_dict(agreed_quant_ann: dict, label_ann: str):
    """
    Returns a dictionary containing the agreed-upon quantitative annotations 
    for each article.

    Parameters:
    agreed_quant_ann (dict): A dictionary containing the agreed-upon 
    quantitative annotations.
    label_ann (str): The label of the quantitative annotation to extract.

    Returns:
    dict: {key=article_id, value={key=local annotation id, value=label}
    """
    article_dict = {}
    for id in agreed_quant_ann.keys():

        # if annotation includes desired component
        if agreed_quant_ann[id][label_ann] != '\0':

            # split id into article_id and local annotation_id
            article_id, ann_id = id.split('_')

            # add article_id to dict if not already present
            if article_id not in article_dict:
                article_dict[article_id] = {}

            # add {key=local annotation_id, value=label} to article_dict
            article_dict[article_id][ann_id] = agreed_quant_ann[id][label_ann]

    return article_dict


def get_ann_dict(article_html: str, annotation_ids: list):
    """
    Extracts the text of the annotations with the given IDs from the HTML of an article.

    Parameters:
    article_html (str): The HTML content of the article.
    annotation_ids (list): A list of annotation IDs to extract.

    Returns:
    dict: {key=local annotation id, value=annotation text}
    """

    soup = BeautifulSoup(article_html, 'html.parser')
    ann_ex = soup.find_all('span', {'class': 'yellow'})

    ann_dict = {}
    for a in ann_ex:
        span_id = a['id']
        if span_id in annotation_ids:
            ann_dict[span_id] = a.text

    return ann_dict


def get_context(i: int, sentences: list):
    """
    Returns the context of a sentence at index i in a list of sentences.
    The context includes the previous and next sentences if they exist.
    
    Args:
    i (int): Index of the sentence to get context for.
    sentences (list): List of sentences.
    
    Returns:
    str: The context of the sentence at index i.
    """

    context = sentences[i]
    if i > 0:
        context = sentences[i-1] + ' ' + context
    if i < len(sentences) - 1:
        context = context + ' ' + sentences[i+1]

    return context


def load_dataset(db_filename: str, label_ann: str, type_filter: list = []):
    """
    Load dataset from a database file and return texts and labels.

    Args:
    - db_filename (str): path to the database file.
    - label_ann (str): label annotation.
    - type_filter (list): list of types to filter annotations.

    Returns:
    - texts (list): list of texts.
    - labels (list): list of labels.
    """

    texts_labels = set()

    quant_ann = gs.get_quant_dict(db_filename=db_filename)
    agreed_quant_ann = gs.get_agreed_anns(quant_ann, type_filter=type_filter)
    article_dict = get_article_dict(agreed_quant_ann, label_ann=label_ann)

    for article_id in article_dict.keys():

        ann_label_dict = article_dict[article_id]

        article_html = gs.get_text(article_id, 'data/data.db', clean=False)
        # {key=local annotation id, value=annotation text}
        ann_dict = get_ann_dict(article_html, ann_label_dict.keys())

        article_text = gs.extract_strings(article_html)  # remove span tags
        article_sentences = nltk.sent_tokenize(article_text)

        for ann_id in ann_dict.keys():

            ann_text = ann_dict[ann_id]
            found = False
            i = 0
            while not found or i < len(article_sentences): 
                if ann_text in article_sentences[i]:
                    context = get_context(i, article_sentences)
                    texts_labels.add((context, ann_label_dict[ann_id]))
                    found = True
                i += 1

    texts = [t[0] for t in texts_labels]
    labels = [label_maps[label_ann][t[1]] for t in texts_labels]

    return texts, labels


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
        task_texts, task_labels = load_dataset("data/data.db", label_ann=task, 
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
        tt.to_csv(task, y_labels_tot, y_predicted_tot, destination)
        # model.save_pretrained(f"models/roberta/{annotation_component}_model")


if __name__ == '__main__':
    main()

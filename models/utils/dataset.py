from bs4 import BeautifulSoup
import nltk
import data_utils.get_annotation_stats as gs
from sklearn.metrics import classification_report
import pandas as pd


def load_qual_dataset(db_filename: str, annotation_component: str,
                      label_map: dict = {}):
    """
    Load dataset from a database file and return text and labels for a given annotation component.

    Parameters:
    db_filename (str): The path to the database file.
    annotation_component (str): The annotation component to extract labels for.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two lists: text and labels.
    """

    text = []
    labels = []

    # get all agreed annotations for given component
    qual_ann = gs.get_qual_dict(db_filename)
    agreed_qual_ann = gs.get_agreed_anns(qual_ann)

    # get list of text and labels for given component
    for article_id in agreed_qual_ann.keys():
        if agreed_qual_ann[article_id][annotation_component] != '\0':
            article_dict = agreed_qual_ann[article_id]
            clean_text = gs.get_text(article_id, db_filename, clean=True)

            text.append(clean_text)
            label = article_dict[annotation_component]
            labels.append(label)

    if label_map != {}:
        labels = [label_map[label] for label in labels]

    return text, labels


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


def load_quant_dataset(db_filename: str, label_ann: str, label_map: dict,
                       type_filter: list = []):
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
    labels = [t[1] for t in texts_labels]

    if label_map != {}:
        labels = [label_map[label] for label in labels]

    return texts, labels

def to_csv(annotation_component, labels, predicted, destination: str= "models/roberta/results"):
    """
    Write classification report to a CSV file.

    Parameters:
    - annotation_component (str): Name of the annotation component.
    - labels (list): List of true labels.
    - predicted (list): List of predicted labels.
    - destination (str, optional): Path to save the CSV file. Defaults to "models/roberta/results".
    """
    report = classification_report(labels,
                                   predicted,
                                   output_dict=True,
                                   zero_division=0)

    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{destination}/{annotation_component}_classification_report.csv")
import data_utils.get_annotation_stats as gs  # msql queries

from bs4 import BeautifulSoup
import nltk
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import sys
import pickle


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


def get_ann_dict(article_html: str,
                 annotation_ids: list):
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
    
    if len(ann_dict.keys()) != len(annotation_ids):
        print(">>> ERROR: Annotations not found in html in get_ann_dict()")
        not_found = []
        for id in annotation_ids:
            if id not in ann_dict.keys():
                not_found.append(id)
        # print("Desired ids: " + str(annotation_ids))
        print("Ids not found: " + str(not_found))
        # print("Article html\n" + str(article_html))
        print()

    return ann_dict


def get_context(i: int,
                sentences: list):
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




def get_excerpts(ann_ids: list,
                 db_filename: str):
    """
    Retrieves excerpts from an article based on global annotation IDs.

    Args:
        ann_ids (list): List of annotation IDs, global ids from same article.
        db_filename (str): Filename of the database.

    Returns:
        dict: A dictionary containing the excerpts mapped to their 
        corresponding global IDs.
    """
    excerpt_dict = {}

    article_id = ann_ids[0].split('_')[0]
    article_html = gs.get_text(article_id, db_filename, clean=False)

    local_ann_ids = [ann_id.split('_')[1] for ann_id in ann_ids]
    ann_dict = get_ann_dict(article_html, local_ann_ids)

    article_text = gs.extract_strings(article_html)  # remove span tags
    article_sentences = nltk.sent_tokenize(article_text)

    for ann_id in ann_dict.keys():

        ann_text = ann_dict[ann_id]
        found = False
        i = 0
        while not found and i < len(article_sentences):
            if article_sentences[i].find(ann_text) != -1:
                context = get_context(i, article_sentences)
                global_id = f"{article_id}_{ann_id}"
                excerpt_dict[global_id] = [ann_text, context]
                found = True
            i += 1
            if i == len(article_sentences) and not found:
                print(">>> ERROR: Annotation not found in get_excerpts()")
                print(ann_id)
                print(ann_text)
                # print(article_sentences)
                print()

    return excerpt_dict


def save_progress(to_save,
                  filename: str):
    """
    Save the progress to a file using pickle.

    Args:
        to_save: The object to be saved.
        filename (str): The name of the file to save the object to. Default is 'excerpts_dict'.
    """
    try:
        progress_file = open(filename, 'wb')
        pickle.dump(to_save, progress_file)
        progress_file.close()

    except Exception as e:
        print(e)
        print("Could not dump object to picle file in save_progress()")


def get_excerpts_dict(db_filename: str):
    """
    Retrieves excerpts corresponding to all quant annotations from a database file and returns
    them as a dictionary.

    Args:
        db_filename (str): The filename of the database.

    Returns:
        dict: keys are global quant_ann IDs and the values are the corresponding excerpts + context.
    """

    try:
        # return id and text
        excerpt_dict = {}

        # get all global quant_ann ids
        excerpt_ids = gs.get_excerpts(db_filename)

        # put annotations from same article together to improve performance
        # {key=article_id, value=list of local annotation ids}
        article_dict = {}
        for id in excerpt_ids:
            # split id into article_id and local annotation_id
            article_id, ann_id = id.split('_')

            # add article_id to dict if not already present
            if article_id not in article_dict:
                article_dict[article_id] = []

            # add local annotation id to article_dict
            article_dict[article_id].append(ann_id)

        for article_id, ann_list in article_dict.items():

            # get article html 
            article_html = gs.get_text(article_id, db_filename, clean=False)

            # get annotation text for desired local annotations
            # {key=local annotation id, value=annotation text}
            ann_dict = get_ann_dict(article_html, ann_list)

            article_text = gs.extract_strings(article_html)  # remove span tags
            article_sentences = nltk.sent_tokenize(article_text)

            # search for each qual_ann excerpt in article
            for ann_id, ann_text in ann_dict.items():
                found = False
                i = 0
                while not found and i < len(article_sentences):

                    # if found, add sentence and context to excerpt_dict
                    if article_sentences[i].find(ann_text) != -1:
                        context = get_context(i, article_sentences)
                        id = f"{article_id}_{ann_id}"
                        text_list = [ann_text, context]
                        excerpt_dict[id] = text_list
                        found = True
                    i += 1

    except Exception as e:
        print(e)
        # If the program is interrupted, save the progress
        save_progress(excerpt_dict, 'data/clean/excerpts_dict_partial')
        sys.exit()

    return excerpt_dict


def to_csv(annotation_component: str,
           labels: list,
           predicted: list,
           destination: str = "models/roberta/results"):
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

def to_f1_csv(results,
              detination,
              f1):
    
    destination = destination + f1 + "_f1_report.csv"

    results_formatted = {}
    for task in results.keys():
        results_formatted[task] = []
        labels = results[task]['labels']
        predictions = results[task]['predictions']
        score = f1_score(labels, predictions, average=f1)
        results_formatted[task].append(score)

    df = pd.DataFrame(results_formatted)
    df.to_csv(destination)






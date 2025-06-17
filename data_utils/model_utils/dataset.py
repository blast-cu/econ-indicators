import data_utils.get_annotation_stats as gs  # msql queries
import torch
from torch.utils.data import Dataset
from bs4 import BeautifulSoup
import nltk
from tqdm import tqdm

import sys
import pickle

DB_FILENAME = 'data/data.db'  # database file location
ROBERTA_MODEL_DIR = 'models/roberta_classifier/tuned_models'  # directory to save model outputs
SPLIT_DIR = 'data/clean/'  # directory to save and load train/test splits

qual_label_maps = {  # maps raw annotations to numerical labels for model input
    'frame': {
            'business': 0,
            'industry': 1,
            'macro': 2,
            'government': 3,
            'other': 4,
            'personal': 5
    },
    'econ_rate': {
            'good': 0,
            'poor': 1,
            'none': 2,
            'irrelevant': 3},
    'econ_change': {
            'better': 0,
            'worse': 1,
            'same': 2,
            'none': 3,
            'irrelevant': 4}
}

quant_label_maps = {  # maps raw annotations to numerical labels for model input
    'type': {
            'macro': 0,
            'industry': 1,
            'government': 2,
            'personal': 3,
            'business': 4,
            'other': 5},
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
            'other': 10,
            'none': 11},
    'spin': {
            'pos': 0,
            'neg': 1,
            'neutral': 2}
}

qual_predict_maps = {  # maps model outputs to raw labels
    'frame': {
            0: 'business',
            1: 'industry',
            2: 'macro',
            3: 'government',
            4: 'other',
            5: 'personal'},
    'econ_rate': {
            0: 'good',
            1: 'poor',
            2: 'none',
            3: 'irrelevant'},
    'econ_change': {
            0: 'better',
            1: 'worse',
            2: 'same',
            3: 'none',
            4: 'irrelevant'}
}

quant_predict_maps = {  # maps model outputs to raw labels
    'type': {
            0: 'macro',
            1: 'industry',
            2: 'government',
            3: 'personal',
            4: 'business',
            5: 'other'},
    'macro_type': {
            0: 'jobs',
            1: 'retail',
            2: 'interest',
            3: 'prices',
            4: 'energy',
            5: 'wages',
            6: 'macro',
            7: 'market',
            8: 'currency',
            9: 'housing',
            10: 'other',
            11: 'none'},
    'spin': {
            0: 'pos',
            1: 'neg',
            2: 'neutral'}
}


class QualAnnClassificationDataset(Dataset):
    def __init__(self, texts, labels=[], tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized  item from the dataset
        """
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        ) 

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }


class QuantAnnClassificationDataset(Dataset):
    def __init__(self, texts, labels=[], ids=[], tokenizer=None, max_length=512):
        """
        Initializes a dataset for text classification
        """
        self.indicator_texts = [t[0] for t in texts]
        self.texts = [t[1] for t in texts]
        self.labels = labels

        self.article_ids = []
        self.ann_ids = []
        for id in ids:
            article_id, ann_id = id.split('_')
            self.article_ids.append(int(article_id))
            self.ann_ids.append(int(ann_id))
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        start_indices = []
        end_indices = []
        self.input_ids = []
        self.attention_masks = []

        for idx, text in enumerate(self.texts):

            indicator_text = self.indicator_texts[idx]

            temp_encoding = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=False,
                return_offsets_mapping=True
            )

            start_index, end_index = \
                self.get_indicator_indices(indicator_text,
                                           temp_encoding,
                                           text)

            if end_index is None:
                print("setting end index to start index")
                end_index = start_index

            if start_index is None or end_index is None:
                print('Substring: ' + indicator_text)
                print('Original text: ' + text)
                print('Start index: ' + str(start_index))
                print('End index: ' + str(end_index))
                print('Offset mapping: \n')
                for i, token in enumerate(temp_encoding['offset_mapping'][0]):
                    print(f"Token {i}: {token}")
                raise Exception('Could not find indicator text in excerpt')

            if end_index > self.max_length:
                text_start = end_index + int(self.max_length / 2) - self.max_length
                text = text[text_start:]

                start_index = start_index - text_start
                end_index = end_index - text_start

            excerpt_encoding = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=self.max_length,
                padding='max_length',
                truncation=True
            )

            start_indices.append(start_index)
            end_indices.append(end_index)
            self.input_ids.append(excerpt_encoding['input_ids'].flatten())
            self.attention_masks.append(excerpt_encoding['attention_mask'].flatten())

        self.spans = []
        for i, start_index in enumerate(start_indices):
            end_index = end_indices[i]

            span = []
            curr = start_index
            for j in range(512):
                inner_span = []

                if curr >= start_index and curr <= end_index:
                    inner_span = [i for i in range(769)]
                else:
                    inner_span = [768] * 769

                curr += 1
                span.append(inner_span)
            self.spans.append(span)

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.texts)
    
    def get_indicator_indices(self, indicator_text, excerpt_encoding, text):
        """
        Get the start and end indices of the indicator text within the given tokenized text.

        Parameters:
            indicator_text (str): The indicator text to search for.
            excerpt_encoding (dict): The encoding of the excerpt text.
            text (str): The full text to search within.

        Returns:
            tuple: A tuple containing the start and end indices of the indicator text within the text.
        """
        sub_start = text.find(indicator_text)
        sub_end = sub_start + len(indicator_text)
        offset_map = excerpt_encoding['offset_mapping'].tolist()

        start_index = None
        end_index = None
        for i, token in enumerate(offset_map[0]):
            token_start = token[0]
            token_end = token[1]
            if start_index is None:
                if sub_start >= token_start and sub_start <= token_end:
                    start_index = i
            else:
                if sub_end >= token_start and sub_end <= token_end:
                    end_index = i + 1
                    break

        return start_index, end_index
    
    def __getitem__(self, idx):
        """
        Returns a single tokenized  item from the dataset
        """
        text = self.texts[idx]
        if len(self.labels) > 0:
            label = self.labels[idx]
        else:
            label = -1

        if len(self.article_ids) > 0:
            article_id = self.article_ids[idx]
            ann_id = self.ann_ids[idx]
        else:
            article_id = -1
            ann_id = -1

        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        span = self.spans[idx]

        return {
            'span': torch.tensor(span),
            'input_ids': input_id,
            'attention_mask': attention_mask,
            'label': torch.tensor(label),
            'article_ids': torch.tensor(article_id),
            'ann_ids': torch.tensor(ann_id)
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
    not_found = []
    if len(ann_dict.keys()) != len(annotation_ids):
        # print(">>> ERROR: Annotations not found in html in get_ann_dict()")
        for id in annotation_ids:
            if id not in ann_dict.keys():
                not_found.append(id)
        # print(f"{len(not_found)} ids not found")
        # print()

    return ann_dict, not_found


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


def get_excerpts(ann_ids: list, db_filename: str) -> dict:
    """
    Retrieves excerpts from an article based on global annotation IDs.

    Args:
        ann_ids (list): List of annotation IDs, global ids from same article.
        db_filename (str): Filename of the database.

    Returns:
        dict: A dictionary containing [indicator span, excerpt] mapped to 
            their corresponding global IDs.
    """
    excerpt_dict = {}

    article_id = ann_ids[0].split('_')[0]
    article_html = gs.get_text(article_id, db_filename, clean=False)

    local_ann_ids = [ann_id.split('_')[1] for ann_id in ann_ids]
    ann_dict = get_ann_dict(article_html, local_ann_ids)

    article_text = gs.extract_strings(article_html)  # remove span tags
    article_sentences = nltk.sent_tokenize(article_text)

    for ann_id in ann_dict.keys():

        ann_text = ann_dict[ann_id].strip()
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
                print(article_id)
                print(ann_id)
                print(ann_text)
                print(article_sentences)

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


def get_excerpts_dict(db_filename: str, logger):
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

        errors = 0

        # get all global quant_ann ids
        excerpt_ids = gs.get_excerpts(db_filename)
        logger.info(f"Found {len(excerpt_ids)} quant annotations in database")

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

        pbar = tqdm(total=len(article_dict.keys()), desc="Processing articles")
        for article_id, ann_list in article_dict.items():

            # get article html 
            article_html = gs.get_text(article_id, db_filename, clean=False)

            # get annotation text for desired local annotations
            # {key=local annotation id, value=annotation text}
            ann_dict, local_errors = get_ann_dict(article_html, ann_list)
            errors += len(local_errors)

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
            pbar.update(1)
        pbar.close()
        logger.info(f"WARNING: {errors} of {len(excerpt_ids)} quants not found")

    except Exception as e:
        print(e)
        # If the program is interrupted, save the progress
        save_progress(excerpt_dict, 'data/clean/excerpts_dict_partial')
        sys.exit()

    return excerpt_dict
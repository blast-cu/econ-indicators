import pickle
import os
import torch
import math

from torch.utils.data import DataLoader

import models.psl.generate_rules as gd
import models.roberta_classifier.predict_qual as pq
import models.roberta_classifier.predict_quant as pqt
import data_utils.get_annotation_stats as gs

MODELS_DIR = 'models/roberta_classifier/tuned_models/'
OUT_DIR = 'models/psl/data'
DB_FILENAME = 'data/data.db'


def load_train_test_data(split_dict, qual_dict, quant_dict):
    """
    Load train and test data for the PSL model.

    Args:
        split_dict (dict): A dictionary containing train and test article
                IDs for given split.
        qual_dict (dict): A dictionary containing qualitative data for
                articles.
        quant_dict (dict): A dictionary containing quantitative data for 
                excerpts.

    Returns:
        tuple: A tuple containing train articles, train excerpts, test 
                articles, and test excerpts.
    """
    # load train article and excerpt dicts
    train_article_ids = split_dict['train']
    train_articles = {id: qual_dict[id] for id in train_article_ids}

    train_excerpt_ids = []
    for id in train_article_ids:
        train_excerpt_ids += qual_dict[id]['quant_list']
    train_excerpts = {id: quant_dict[id] for id in train_excerpt_ids}

    # load test article and excerpt dicts
    test_article_ids = split_dict['test']
    test_articles = {id: qual_dict[id] for id in test_article_ids}

    test_excerpt_ids = []
    for id in test_article_ids:
        test_excerpt_ids += qual_dict[id]['quant_list']
    test_excerpts = {id: quant_dict[id] for id in test_excerpt_ids}

    return train_articles, train_excerpts, test_articles, test_excerpts


def write_data_file(out_dir, predicate, file_type, values):
    """
    Write data to a file.

    Args:
        out_dir (str): The output directory where the file will be saved.
        predicate (str): The predicate associated with data.
        file_type (str): The type of file to be generated.
        values (list): The list of formatted values to be written to the file.

    Returns:
        None
    """
    if len(values) == 0:
        return

    filename = f'{predicate}_{file_type}.txt'
    file_path = os.path.join(out_dir, filename)
    with open(file_path, 'w') as f:
        for value in values:
            f.write(f'{value}\n')

    return


def write_val_files(out_dir, articles, file_type):
    """
    NOT CURRENTLY BEING USED
    Write validation files for each annotation type.

    Args:
        out_dir (str): The output directory to write the files to.
        articles (dict): A dictionary containing the articles and their annotations.
        file_type (str): The type of file to write.

    Returns:
        None
    """

    article_ann_dict = {}
    for article_id, article_dict in articles.items():
        for ann, value in article_dict.items():
            if value != '\x00' and ann != 'quant_list' and ann != 'excerpt':
                if ann not in article_ann_dict:
                    article_ann_dict[ann] = []
                value = f'{article_id}\t{value}'
                article_ann_dict[ann].append(value)

    for ann, values in article_ann_dict.items():
        ann = gd.camel_case(ann, upper=True)
        predicate = f'Val{ann}'
        write_data_file(out_dir, predicate, file_type, values)


def write_contains_file(out_dir, articles):
    """
    Write a file containing the 'Contains' predicate for the given articles 
    and their excerpts.

    Args:
        out_dir (str): The output directory where the file will be written.
        articles (dict): A dictionary containing the articles and their 
                associated excerpts.

    Returns:
        None
    """
    predicate = 'Contains'
    to_write = []
    for article_id, article_dict in articles.items():
        excerpt_ids = article_dict['quant_list']
        for excerpt_id in excerpt_ids:
            temp = [f'{article_id}\t{excerpt_id}\t1.0']
            to_write += temp

    write_data_file(out_dir, predicate, 'obs', to_write)


def write_target_files(out_dir, articles, map, truth=True):
    """
    Write target files based on the given articles and mapping.

    Args:
        out_dir (str): The output directory to write the target files.
        articles (dict): A dictionary containing article information.
        map (dict): A dictionary containing mapping information.
        truth (bool): A flag indicating whether to write truth files or not.
    """
    
    article_ann_dict = {}
    for id in articles.keys():
        for ann in map.keys():
            if ann != 'quant_list' and ann != 'excerpts':
                if ann not in article_ann_dict:
                    article_ann_dict[ann] = []
                for value in map[ann]:
                    if articles[id][ann] == value:
                        weight = 1.0
                    else:
                        weight = 0.0
                    to_write = f'{id}\t{value}\t{weight}'
                    article_ann_dict[ann].append(to_write)
    
    for ann, values in article_ann_dict.items():
        ann = gd.camel_case(ann, upper=True)
        predicate = f'Val{ann}'
        target_values = [value[:-4] for value in values]
        write_data_file(out_dir, predicate, 'target', target_values)

        if truth:
            write_data_file(out_dir, predicate, 'truth', values)


def logit_to_prob(logit):
    """
    Convert a logit value to a probability.

    Args:
        logit (float): The logit value.

    Returns:
        float: The corresponding probability value.
    """
    odds = math.exp(logit)
    prob = odds / (1 + odds)
    return prob


def predict_article_annotations(articles, split_num):
    """
    Predicts article annotations using fine-tuned models for each annotation
    component.

    Args:
        articles (dict): A dictionary where keys are article IDs and values 
                are article texts.
        split_num (int): The fold number for the fine-tuned models.

    Returns:
        dict: A dictionary where keys are annotation components and values are 
                lists of strings containing article IDs, annotation values, and 
                prediction probabilities.
    """

    articles = {k: gs.get_text(k, DB_FILENAME, clean=True) for k, v in articles.items()}

    torch.manual_seed(42)  # Set random seed for reproducibility

    tokenizer = pqt.RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path="roberta-base",
                         problem_type="single_label_classification")

    data = pq.PredictionDataset(articles=articles,
                                tokenizer=tokenizer,
                                max_length=512)

    batch_size = 8
    loader = DataLoader(data, batch_size=batch_size, shuffle=False) # check shuffle thing

    # load fine-tuned model for each annotation component
    models = {}
    for k in pq.label_maps.keys():
        model_path = f"models/roberta_classifier/tuned_models/fold{split_num}/qual/{k}_model"
        models[k] = pqt.RobertaForSequenceClassification\
            .from_pretrained(model_path).to('cuda')

    # create dictionary to store annotations
    annotations = {}
    for id in articles.keys():
        annotations[id] = {}

    # get dict where keys are annotation components and values are dicts where
    # # keys are annotation values and values are lists of tuples (article ids, pred_val)
    predict_dict = {}
    for annotation_component in gd.qual_map.keys():
        predict_dict[annotation_component] = []

    for annotation_component in models.keys():
        for batch in loader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            ids = batch['ids'].to('cuda')

            cur_model = models[annotation_component]
            outputs = cur_model(input_ids, attention_mask=attention_mask)
            outputs = outputs.logits.tolist()

            for i, id in enumerate(ids.tolist()):

                for j, output in enumerate(outputs[i]):
                    probability = logit_to_prob(output)
                    probability = round(probability, 4)
                    annotation_value = pq.label_maps[annotation_component][j]

                    to_write = f'{id}\t{annotation_value}\t{probability}'
                    predict_dict[annotation_component].append(to_write)

    return predict_dict


def generate_predict_excerpts(excerpts, split_num):
    """
    Generate type predictions for given excerpts using a pre-trained model.

    Args:
        excerpts (dict): A dictionary containing the excerpts to generate
            predictions for.
        split_num (int): The split number.

    Returns:
        dict: A dictionary containing the probabilities for each possible 
            value of each ann component.
    """

    excerpts = {k: v['excerpt'] for k, v in excerpts.items()}

    pqt.torch.manual_seed(42)  # Set random seed for reproducibility
    tokenizer = pq.RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path="roberta-base",
                         problem_type="single_label_classification")

    data = pq.PredictionDataset(articles=excerpts,
                                tokenizer=tokenizer,
                                max_length=512)
    
    batch_size = 8
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model_path = os.path.join(MODELS_DIR, f'fold{split_num}/quant/type_model')
    type_model = pqt.RobertaForSequenceClassification\
        .from_pretrained(model_path).to("cuda")

    # TODO: loop over all desired annotation components
    annotation_component = 'type'
    predict_dict = {}
    # for annotation_component in gd.qual_map.keys():
    predict_dict[annotation_component] = []

    for i, batch in enumerate(loader):

        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        ids = batch['ids']

        outputs = type_model(input_ids, attention_mask=attention_mask)
        type_outputs = outputs.logits.tolist()

        for i, id in enumerate(ids):

            for j, output in enumerate(type_outputs[i]):
                probability = logit_to_prob(output)
                probability = round(probability, 4)
                annotation_value = pqt.label_maps[annotation_component][j]

                to_write = f'{id}\t{annotation_value}\t{probability}'
                predict_dict[annotation_component].append(to_write)

    return predict_dict


def write_pred_files(out_dir, predict_dict):

    for annotation_component, values in predict_dict.items():
        annotation_component = gd.camel_case(annotation_component, upper=True)
        predicate = f'Pred{annotation_component}'
        write_data_file(out_dir, predicate, 'obs', values)


def main():
    """
    Main function for generating data.

    This function generates data for learn and eval by:
    1. Ensures that the output directory exists.
    2. Loads train and test articles from pickle files.
    3. Creates directories for split data.
    4. Loads learn and eval data for the split.
    5. For learn and eval: 
        a. Writes contains file linking articles and excerpts.
        b. Writes target and truth files for articles and excerpts.
        c. Writes prediction files for articles and excerpts.
    """
    
    # ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # load train and test articles
    split_dir = "data/clean/"
    splits_dict = pickle.load(open(split_dir + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(split_dir + 'quant_dict', 'rb'))

    split_num = 0
    # TODO: loop over splits
    # for split_num in splits_dict.keys():
    
    # make directories for split data
    split_learn_dir = os.path.join(OUT_DIR, f'split{split_num}/learn')
    os.makedirs(split_learn_dir, exist_ok=True)

    split_eval_dir = os.path.join(OUT_DIR, f'split{split_num}/eval')
    os.makedirs(split_eval_dir, exist_ok=True)

    # load train and test data for split    
    learn_articles, learn_excerpts, eval_articles, eval_excerpts = \
        load_train_test_data(splits_dict[split_num],
                             qual_dict,
                             quant_dict)

    # GENERATE LEARN DATA #
    # write contains file linking articles and excerpts
    # write_contains_file(split_learn_dir, learn_articles)  # contains

    # write target and truth files for validation data
    write_target_files(split_learn_dir, learn_articles, gd.qual_map, truth=True)  # isVal
    write_target_files(split_learn_dir, learn_excerpts, gd.quant_map, truth=True)  # isVal

    # predictions for validation set
    article_preds = predict_article_annotations(learn_articles, split_num)
    write_pred_files(split_learn_dir, article_preds)  # pred

    exerpt_preds = generate_predict_excerpts(learn_excerpts, split_num)
    write_pred_files(split_learn_dir, exerpt_preds)  # pred

    # GENERATE EVAL DATA #
    # write_contains_file(split_eval_dir, eval_articles)  # contains
    write_target_files(split_eval_dir, eval_articles, gd.qual_map, truth=True)  # isVal
    write_target_files(split_eval_dir, eval_excerpts, gd.quant_map, truth=True)  # isVal
    
    article_preds = predict_article_annotations(eval_articles, split_num)
    write_pred_files(split_eval_dir, article_preds)  # pred

    excerpt_preds = generate_predict_excerpts(eval_excerpts, split_num)
    write_pred_files(split_eval_dir, excerpt_preds)  # pred


if (__name__ == '__main__'):
    main()
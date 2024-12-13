import pickle
import os
import torch
import math

from torch.utils.data import DataLoader

import models.psl.generate_rules as gd
import models.roberta_classifier.predict_qual as pq
import models.roberta_classifier.predict_quant as pqt
# from data_utils.model_utils.dataset import QuantAnnClassificationDataset
import data_utils.get_annotation_stats as gs
import data_utils.model_utils.dataset as d
from data_utils.model_utils.dataset import DB_FILENAME
# import models.roberta_classifier.utils.quant as qu
import models.roberta_classifier.utils.legacy_quant as qu

OUT_DIR = 'models/psl/data'
NOISE = False

BEST_MODELS = {
    'frame': 'models/roberta_classifier/tuned_models/qual_roberta_base',
    'econ_rate': 'models/roberta_classifier/tuned_models/qual_roberta_dapt_512',
    'econ_change': 'models/roberta_classifier/tuned_models/qual_roberta_base',
    'type': 'models/roberta_classifier/tuned_models/quant_roberta_dapt_128',
    'macro_type': 'models/roberta_classifier/tuned_models/quant_roberta_dapt_512',
    'spin': 'models/roberta_classifier/tuned_models/quant_roberta_dapt_512'
}


def load_train_test_data(split_dict, qual_dict, quant_dict, qual_noise_dict={}, quant_noise_dict={}):
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
    test_article_ids = split_dict['test']

    train_articles = {id: qual_dict[id] for id in train_article_ids}

    train_excerpt_ids = []
    for id in train_article_ids:
        train_excerpt_ids += qual_dict[id]['quant_list']

    train_excerpts = {id: quant_dict[id] for id in train_excerpt_ids}

    # add noisy excerpts to train excerpts
    for noisy_id, noisy_ann in qual_noise_dict.items():
        if noisy_id not in train_articles:
            train_articles[noisy_id] = noisy_ann
        else:
            for k, v in noisy_ann.items():
                if train_articles[noisy_id][k] == '\x00':
                    train_articles[noisy_id][k] = v
                elif v == 'quant_list':
                    for q_id in v:
                        if q_id not in train_articles[noisy_id][k]:
                            train_articles[noisy_id][k].append(q_id)

    for noisy_id, noisy_ann in quant_noise_dict.items():
        noisy_article_id = int(noisy_id.split('_')[0])
        if noisy_article_id not in test_article_ids:
            if noisy_id not in train_excerpts:
                train_excerpts[noisy_id] = noisy_ann
            else:
                for k, v in noisy_ann.items():
                    if train_excerpts[noisy_id][k] == '\x00':
                        train_excerpts[noisy_id][k] = v

    # load test article and excerpt dicts
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

    print(f'Writing {file_path}')
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
    truth_ann_dict = {}
    for id in articles.keys():
        for ann in map.keys():
            if ann != 'quant_list' and ann != 'excerpts':
                if ann not in article_ann_dict:
                    article_ann_dict[ann] = []
                    truth_ann_dict[ann] = []
                for value in map[ann]:
                    to_write = f'{id}\t{value}'
                    article_ann_dict[ann].append(to_write)
                    if ann in articles[id] and articles[id][ann] == value:
                        truth_ann_dict[ann].append(to_write)

    for ann, values in article_ann_dict.items():
        ann2 = gd.camel_case(ann, upper=True)
        predicate = f'Val{ann2}'
        # target_values = [value[:-4] for value in values]
        write_data_file(out_dir, predicate, 'target', values)

        if truth:
            write_data_file(out_dir, predicate, 'truth', truth_ann_dict[ann])


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


def predict_article_annotations(articles, model_map, split_num=None):
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

    articles = {k: gs.get_text(k, DB_FILENAME, clean=False) for k, v in articles.items()}
    articles = {k: v.replace('\n', '') for k, v in articles.items()}
    
    torch.manual_seed(42)  # Set random seed for reproducibility

    tokenizer = pq.RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path="roberta-base",
                         problem_type="single_label_classification")

    data = pq.PredictionDataset(articles=articles,
                                tokenizer=tokenizer,
                                max_length=512)

    batch_size = 8
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)  # check shuffle thing

    # load fine-tuned model for each annotation component
    models = {}
    for k in d.qual_label_maps.keys():
        model_path = model_map[k]
        if split_num is not None:  # if split_num is provided, append fold number to model path
            model_path = os.path.join(model_path, f'fold{split_num}')
        # append task name to model path
        model_path = os.path.join(model_path, f'{k}_model')
        models[k] = pq.RobertaForSequenceClassification\
            .from_pretrained(model_path).to('cuda')

    # create dictionary to store annotations
    annotations = {}
    for id in articles.keys():
        annotations[id] = {}

    # get dict where keys are annotation components and values are dicts where
    # # keys are annotation values and values are lists of tuples (article ids, pred_val)
    predict_dict = {}
    for annotation_component in d.qual_label_maps.keys():
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
                probs = []
                for j, output in enumerate(outputs[i]):
                    probability = logit_to_prob(output)
                    probability = round(probability, 4)
                    annotation_value = d.qual_predict_maps[annotation_component][j]

                    to_write = f'{id}\t{annotation_value}\t{probability}'
                    predict_dict[annotation_component].append(to_write)

    return predict_dict


def generate_predict_excerpts(excerpts, model_map, split_num=None):
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
    predict_dict = {}

    for annotation_component in d.quant_label_maps.keys():

        texts = [[v['indicator'], v['excerpt']] for v in excerpts.values() if annotation_component not in v.keys() or v[annotation_component] != '\x00']
        ids = [k for k in excerpts.keys() if annotation_component not in excerpts[k].keys() or excerpts[k][annotation_component] != '\x00']

        pqt.torch.manual_seed(42)  # Set random seed for reproducibility
        tokenizer = pqt.RobertaTokenizerFast\
            .from_pretrained(pretrained_model_name_or_path="roberta-base",
                             problem_type="single_label_classification")

        data = qu.TextClassificationDataset(
            texts=texts,
            tokenizer=tokenizer,
            ids=ids,
            max_length=512
        )

        batch_size = 8
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        for annotation_component in d.quant_label_maps.keys():

            task = annotation_component
            num_labels = len(set(d.quant_label_maps[task].keys()))

            # get model path for split, task.
            model_path = model_map[task]
            if split_num is not None:  # if split_num is provided, append fold number to path
                model_path = os.path.join(model_path, f'fold{split_num}')
            model_path = os.path.join(model_path, f'{task}_model')  # append task name to model path

            # load model.
            type_model = qu.QuantModel('roberta-base', num_labels).to('cuda')
            type_model = type_model.from_pretrained(model_path, task).to('cuda')
            predict_dict[annotation_component] = []

            type_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    start_index = batch['start_index'].to('cuda')
                    end_index = batch['end_index'].to('cuda')
                    input_ids = batch['input_ids'].to('cuda')
                    attention_mask = batch['attention_mask'].to('cuda')
                    article_ids = batch['article_ids'].tolist()
                    ann_ids = batch['ann_ids'].tolist()

                    outputs = type_model(
                        start_index,
                        end_index,
                        input_ids,
                        attention_mask
                    )
                    type_outputs = outputs.tolist()
                    for i, id in enumerate(article_ids):

                        global_id = str(id) + '_' + str(ann_ids[i])
                        probs = []

                        for j, output in enumerate(type_outputs[i]):
                            probability = logit_to_prob(output)
                            probability = round(probability, 4)
                            probs.append(probability)
                            annotation_value = d.quant_predict_maps[annotation_component][j]

                            to_write = f'{global_id}\t{annotation_value}\t{probability}'
                            predict_dict[annotation_component].append(to_write)

    return predict_dict


def write_pred_files(out_dir, predict_dict):

    for annotation_component, values in predict_dict.items():
        annotation_component = gd.camel_case(annotation_component, upper=True)
        predicate = f'Pred{annotation_component}'
        write_data_file(out_dir, predicate, 'obs', values)


def write_preceeds_file(out_dir, articles):

    predicate = 'Precedes'
    to_write = []
    for article_id, article_dict in articles.items():
        excerpt_ids = article_dict['quant_list']
        for i, excerpt_id in enumerate(excerpt_ids):
            if i == 0:
                continue
            else: 
                prev = excerpt_ids[i-1].split('_')
                curr = excerpt_id.split('_')
                if prev[0] == curr[0]:  # same article
                    if int(prev[1]) + 1 == int(curr[1]):
                        temp = [f'{excerpt_ids[i-1]}\t{excerpt_id}\t1.0']
                        to_write += temp

    write_data_file(out_dir, predicate, 'obs', to_write)


def write_has_frame_ann_file(out_dir, excerpts, predicate='HasTypeAnn'):

    if predicate == 'HasTypeAnn':
        ann_comp = 'type'
    elif predicate == 'HasFrameAnn':
        ann_comp = 'frame'
    else:
        raise ValueError('Invalid predicate pased to write_has_frame_ann_file. Must be HasTypeAnn or HasFrameAnn.')

    to_write = []
    for article_id, ann_dict in excerpts.items():
        # if annotation is not empty OR ann comp not in dict because final data
        if ann_comp not in ann_dict.keys() or ann_dict[ann_comp] != '\x00': 
            temp = [f'{article_id}\t1.0']
            to_write += temp

    write_data_file(out_dir, predicate, 'obs', to_write)


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
    for split_num in range(5):
        # make directories for split data
        split_learn_dir = os.path.join(OUT_DIR, f'split{split_num}/learn')
        os.makedirs(split_learn_dir, exist_ok=True)

        split_eval_dir = os.path.join(OUT_DIR, f'split{split_num}/eval')
        os.makedirs(split_eval_dir, exist_ok=True)

        if NOISE:
            qual_noise_dict = pickle.load(open(split_dir + 'noisy_best_qual_dict', 'rb'))
            quant_noise_dict = pickle.load(open(split_dir + 'noisy_best_quant_dict', 'rb'))

            # load train and test data for split
            learn_articles, learn_excerpts, eval_articles, eval_excerpts = \
                load_train_test_data(splits_dict[split_num],
                                     qual_dict,
                                     quant_dict,
                                     qual_noise_dict,
                                     quant_noise_dict)
            
        else:
            # load train and test data for split    
            learn_articles, learn_excerpts, eval_articles, eval_excerpts = \
                load_train_test_data(splits_dict[split_num],
                                    qual_dict,
                                    quant_dict)

        # # GENERATE LEARN DATA #
        # # write contains file linking articles and excerpts
        write_contains_file(split_learn_dir, learn_articles)  # contains

        write_has_frame_ann_file(split_learn_dir, learn_excerpts)
        write_has_frame_ann_file(split_learn_dir, learn_articles, predicate="HasFrameAnn") 

        write_preceeds_file(split_learn_dir, learn_articles)  # preceeds

        # write target and truth files for validation data
        write_target_files(split_learn_dir, learn_articles, d.qual_label_maps, truth=True)  # isVal

        write_target_files(split_learn_dir, learn_excerpts, d.quant_label_maps, truth=True)  # isVal

        # # predictions for validation set
        article_preds = predict_article_annotations(learn_articles, BEST_MODELS, split_num)
        write_pred_files(split_learn_dir, article_preds)  # pred  

        exerpt_preds = generate_predict_excerpts(learn_excerpts, BEST_MODELS, split_num)
        write_pred_files(split_learn_dir, exerpt_preds)  # pred

        # # GENERATE EVAL DATA #
        write_contains_file(split_eval_dir, eval_articles)  # contains

        write_has_frame_ann_file(split_eval_dir, eval_excerpts)  # HasFrameAnn
        write_has_frame_ann_file(split_eval_dir, eval_articles, predicate="HasFrameAnn")

        write_preceeds_file(split_eval_dir, eval_articles)  # preceeds

        write_target_files(split_eval_dir, eval_articles, d.qual_label_maps, truth=True)  # isVal
        write_target_files(split_eval_dir, eval_excerpts, d.quant_label_maps, truth=True)  # isVal

        article_preds = predict_article_annotations(eval_articles, BEST_MODELS, split_num)
        write_pred_files(split_eval_dir, article_preds)  # pred

        excerpt_preds = generate_predict_excerpts(eval_excerpts, BEST_MODELS, split_num)
        write_pred_files(split_eval_dir, excerpt_preds)  # pred






if (__name__ == '__main__'):
    main()
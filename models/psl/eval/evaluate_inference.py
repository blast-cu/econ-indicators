from models.psl.generate_data import load_train_test_data
import models.psl.generate_rules as gd
import data_utils.model_utils.dataset as d
import data_utils.model_utils.eval as eval
import models.psl.run_inference as ri
import pickle
import os
import itertools
import argparse

from models.psl.SETTINGS import SETTINGS

DATA_DIR = 'models/psl/data'
RULE_DIR = ri.RULE_DIR


def get_pred_dict(filepath, doc_type='quant'):

    with open(filepath) as f:
        spin_lines = f.readlines()

    predictions = {}
    for line in spin_lines:
        line = line.strip()
        line = line.split('\t')

        id = line[0]
        if doc_type == 'qual':
                id = int(id)

        ann_value = line[1]
        ann_value_score = float(line[2])

        if id not in predictions.keys():
            predictions[id] = (ann_value, ann_value_score)

        else:
            curr_max = predictions[id][1]
            if ann_value_score > curr_max:
                predictions[id] = (ann_value, ann_value_score)

    for id in predictions.keys():
        predictions[id] = predictions[id][0]

    return predictions

def get_labels_dict(articles, annotation_type):
    
    labels_dict = {}
    for id, annotations in articles.items():
        if annotations[annotation_type] != '\0':
            labels_dict[id] = annotations[annotation_type]

    return labels_dict

def evaluate(annotation_map, eval_docs, inference_dir, report_dir, split_num, doc_type='quant'):

    results = {}

    for annotation_type in annotation_map.keys():
    # for annotation_type in ['spin']:

        results[annotation_type] = {}
        results[annotation_type]['id'] = []
        results[annotation_type]['predictions'] = []
        results[annotation_type]['labels'] = []

        file_pred = annotation_type.replace('_', '')
        file_pred = file_pred.upper()
        filename = f'VAL{file_pred}.txt'
        # filepath = f'models/psl/data/split{split_num}/eval/PredSpin_obs.txt'
        
        filepath = os.path.join(inference_dir, filename)
        print(filepath)

        try:
            predictions = get_pred_dict(filepath, doc_type)

        except Exception as e:
            print(e)
            print(f'No predictions for {annotation_type}. Skipping...')
            print()
            continue

        labels = get_labels_dict(eval_docs, annotation_type)

        prediction_list = []
        label_list = []
        
        # print(f'>>> {annotation_type} <<<')
        for id in labels.keys():
            # if labels[id] != '\0':
            if int(id) == 7: 
                print(split_num)
            prediction_list.append(predictions[id])
            label_list.append(labels[id])

            # if label_list[-1] != prediction_list[-1]:
            #     print(f'ID: {id} \t Prediction: {prediction_list[-1]} \t Label: {label_list[-1]}')

        results[annotation_type]['id'] = list(labels.keys())
        results[annotation_type]['predictions'] = prediction_list
        results[annotation_type]['labels'] = label_list  
    
        os.makedirs(report_dir, exist_ok=True)
        eval.to_csv(annotation_type, label_list, prediction_list, report_dir)
    
    return results

def main(args):

    # load train and test articles
    split_dir = "data/clean/"
    splits_dict = pickle.load(open(split_dir + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(split_dir + 'quant_dict', 'rb'))


    # establish setting parameters
    setting = args.s
    try:
        setting_dict = SETTINGS[setting]
    except KeyError:
        raise ValueError('Unknown setting: ' + setting)
    
    try:
        rule_files = os.listdir(setting_dict['rule_dir'])
    except FileNotFoundError:
        raise ValueError('Unknown rule directory: ' + setting_dict['rule_dir'])

    SETTING_OUT_DIR = os.path.join(DATA_DIR, 'results', setting)

    for rule_file in rule_files:

        rule_name = rule_file.split('.')[0]
        setting_rule_out_file = os.path.join(SETTING_OUT_DIR, rule_name)
        os.makedirs(setting_rule_out_file, exist_ok=False)

        full_qual_results = {}
        full_quant_results = {}

        for annotation_type in gd.qual_map.keys():
            full_qual_results[annotation_type] = {}
            full_qual_results[annotation_type]['ids'] = []
            full_qual_results[annotation_type]['predictions'] = []
            full_qual_results[annotation_type]['labels'] = []
        
        for annotation_type in gd.quant_map.keys():
            full_quant_results[annotation_type] = {}
            full_quant_results[annotation_type]['ids'] = []
            full_quant_results[annotation_type]['predictions'] = []
            full_quant_results[annotation_type]['labels'] = []
        
        for split_num in range(5):
            global SPLIT_DIR
            SPLIT_DIR = os.path.join(DATA_DIR, f'split{split_num}')
            SPLIT_SETTING_DIR = os.path.join(SPLIT_DIR, setting)

            split_setting_rule_dir = os.path.join(SPLIT_SETTING_DIR, rule_name)

            _, _, eval_articles, eval_excerpts = \
                load_train_test_data(
                    splits_dict[split_num],
                    qual_dict,
                    quant_dict
                )

            inference_dir = os.path.join(
                split_setting_rule_dir,
                'inferred_predicates'
            )

            report_dir = os.path.join(inference_dir, 'reports')
            os.makedirs(report_dir, exist_ok=True)

            qual_results = evaluate(gd.qual_map, eval_articles, inference_dir, report_dir, split_num, doc_type='qual')
            quant_results = evaluate(gd.quant_map, eval_excerpts, inference_dir, report_dir, split_num)

            for annotation_type in gd.qual_map.keys():

                ids = qual_results[annotation_type]['id']
                predictions = qual_results[annotation_type]['predictions']
                labels = qual_results[annotation_type]['labels']

                eval.to_csv(annotation_type, labels, predictions, split_setting_rule_dir)

                full_qual_results[annotation_type]['ids'] += ids
                full_qual_results[annotation_type]['predictions'] += predictions
                full_qual_results[annotation_type]['labels'] += labels


            for annotation_type in gd.quant_map.keys():

                ids = quant_results[annotation_type]['id']
                predictions = quant_results[annotation_type]['predictions']
                labels = quant_results[annotation_type]['labels']

                eval.to_csv(annotation_type, labels, predictions, split_setting_rule_dir)
                
                full_quant_results[annotation_type]['ids'] += ids

                full_quant_results[annotation_type]['predictions'] += \
                    quant_results[annotation_type]['predictions']

                full_quant_results[annotation_type]['labels'] += \
                    quant_results[annotation_type]['labels']

        for annotation_type in gd.qual_map.keys():
            labels = full_qual_results[annotation_type]['labels']
            predictions = full_qual_results[annotation_type]['predictions']
            eval.to_csv(
                annotation_type,
                labels,
                predictions,
                setting_rule_out_file
            )

        for annotation_type in gd.quant_map.keys():
            labels = full_quant_results[annotation_type]['labels']
            predictions = full_quant_results[annotation_type]['predictions']
            eval.to_csv(
                annotation_type,
                labels,
                predictions,
                setting_rule_out_file
            )

        pickle.dump(
            full_qual_results,
            open(os.path.join(d.SPLIT_DIR, 'best_qual_results'), 'wb')
        )
        pickle.dump(
            full_quant_results,
            open(os.path.join(d.SPLIT_DIR, 'best_quant_results'), 'wb')
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--s', required=True, help='setting mode')
    args = parser.parse_args()
    main(args)

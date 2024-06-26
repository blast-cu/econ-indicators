import sqlite3
import pickle
import os
from sklearn.metrics import f1_score
import pandas as pd

from data_utils.table_generators.thing2 import get_texts
from data_utils.model_utils.dataset import DB_FILENAME, qual_label_maps

DATA_DIR = "models/psl/data/"

news_sites = ['nytimes', 'wsj', 'washingtonpost', 'foxnews', 'breitbart', 'huffpost']

site_map = {
    'nytimes': 'New York Times',
    'wsj': 'Wall Street Journal',
    'washingtonpost': 'Washington Post',
    'foxnews': 'Fox News',
    'breitbart': 'Breitbart',
    'huffpost': 'Huffington Post'
}

def write_table(results, task):

    macro_f1s = {}
    weighted_f1s = {}
    macro_f1s['Publisher'] = list(site_map.values())
    weighted_f1s['Publisher'] = list(site_map.values())
    macro_f1s[task] = []
    weighted_f1s[task] = []

    for site in news_sites:
        # print(site)
        macro_f1 = f1_score(results[site]['labels'], results[site]['predictions'], average='macro')
        macro_f1 = round(macro_f1, 3)
        macro_f1s[task].append(macro_f1)

        weighted_f1 = f1_score(results[site]['labels'], results[site]['predictions'], average='weighted')
        weighted_f1 = round(weighted_f1, 3)
        weighted_f1s[task].append(weighted_f1)
    print(macro_f1s)
    pd.DataFrame(macro_f1s).to_csv('data_utils/table_generators/results/publisher_{}_macro_f1s.csv'.format(task), index=False)
    pd.DataFrame(macro_f1s).to_csv('data_utils/table_generators/results/publisher_{}_weighted_f1s.csv'.format(task), index=False)


def get_results(num_folds, setting, rule_name, task):
    results = []
    temp_task = task.replace('_', '')
    task_file = "VAL" + temp_task.upper() + ".txt"
    for split in range(0, num_folds):
        result_dict = {}

        result_file = os.path.join(DATA_DIR, 'split{}'.format(split),
                                   setting, rule_name,
                                   "inferred_predicates", task_file)

        with open(result_file, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]
        for line in lines:
            id, label, score = line.split('\t')
            id = int(id)
            label = qual_label_maps[task][label]
            if id not in result_dict.keys():
                result_dict[id] = [label, score]
            else:
                max_score = result_dict[id][1]
                if score > max_score:
                    result_dict[id] = [label, score]

        results_dict = {k: v[0] for k, v in result_dict.items()}
        results.append(results_dict)
    
    return results

def get_valid_ids(split_test_ids, qual_dict, task):
    valid_ids = []
    for id in split_test_ids:
        if qual_dict[id][task] !='\x00':
            valid_ids.append(id)
    return valid_ids

def main():

    # setting = "excerpt_article"
    # # rule_name = "ValSpin>>ValEconChange"
    # # task = "econ_change"
    # rule_name = "ValType>>ValFrame"
    setting = "precedes"
    rule_name = "ValType>>ValType"
    task = "econ_change"
    num_folds = 5

    splits_dict = pickle.load(open("data/clean/" + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open("data/clean/" + 'qual_dict', 'rb'))

    psl_results = get_results(num_folds, setting, rule_name, task)

    results = {}
    for site in news_sites:
        results[site] = {}
        results[site]['labels'] = []
        results[site]['predictions'] = []
    
    for k, split in splits_dict.items():
        print("Fold " + str(k+1) + " of 5")
        print()
        
        split_test_ids = split['test']
        ann_component = task.split('-')[0]

        print("Task: " + task)

        fold = 'fold{}'.format(k)
        task_name = task + '_model'
        valid_ids = get_valid_ids(split_test_ids, qual_dict, task)

        _, test_labels, curr_sites = get_texts(
            DB_FILENAME,
            ann_component,
            task,
            qual_dict,
            valid_ids
        )


        for i, id in enumerate(valid_ids):
            site = curr_sites[i]
            results[site]['labels'].append(test_labels[i])
            results[site]['predictions'].append(psl_results[k][id])

    write_table(results, task)
    


    

if __name__ == "__main__":
    main()

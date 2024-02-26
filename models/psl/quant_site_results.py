import sqlite3
import pickle
import os
from sklearn.metrics import f1_score

from data_utils.table_generators.thing import get_texts
from data_utils.dataset import DB_FILENAME, quant_label_maps
from models.psl import qual_site_results as sr

DATA_DIR = "models/psl/data/"

news_sites = ['nytimes', 'wsj', 'washingtonpost', 'foxnews', 'breitbart', 'huffpost']



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
            label = quant_label_maps[task][label]
            if id not in result_dict.keys():
                result_dict[id] = [label, score]
            else:
                max_score = result_dict[id][1]
                if score > max_score:
                    result_dict[id] = [label, score]

        results_dict = {k: v[0] for k, v in result_dict.items()}
        results.append(results_dict)
    
    return results

def get_valid_ids(article_ids, qual_dict, quant_dict, task):

    quant_ids = []

    for id in article_ids:
        if 'quant_list' in qual_dict[id].keys():

            for quant_id in qual_dict[id]['quant_list']:
                if quant_dict[quant_id][task] != '\x00':

                    quant_ids.append(quant_id)

    return quant_ids

def main():

    # setting = "neighbors"
    # rule_name = "ValType>>ValMacroType"
    setting = "precedes"
    rule_name = "ValType>>ValType"
    task = "spin"
    num_folds = 5

    splits_dict = pickle.load(open("data/clean/" + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open("data/clean/" + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open("data/clean/" + 'quant_dict', 'rb'))

    type_filters = {
        'type': [],
        'type-binary': [],
        'spin': [],
        'macro_type': []
    }


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

        valid_ids = get_valid_ids(split_test_ids, qual_dict, quant_dict, task)

        _, test_labels, curr_sites = \
                get_texts(ann_component,
                          task,
                          qual_dict,
                          quant_dict,
                          split_test_ids,
                          DB_FILENAME,
                          type_filter=type_filters[task])
        
        
        for i, id in enumerate(valid_ids):
            site = curr_sites[i]
            results[site]['labels'].append(test_labels[i])
            results[site]['predictions'].append(psl_results[k][id])

    # for site in news_sites:
    #     # print(site)
    #     macro_f1 = f1_score(results[site]['labels'], results[site]['predictions'], average='macro')
    #     print(round(macro_f1, 3))
    #     # weighted_f1 = f1_score(results[site]['labels'], results[site]['predictions'], average='weighted')
    #     # print(round(weighted_f1, 3))
    

    sr.write_table(results, task)

    

if __name__ == "__main__":
    main()

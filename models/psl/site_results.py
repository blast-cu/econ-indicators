import sqlite3
import pickle
import os
from sklearn.metrics import f1_score

from models.roberta_classifier.result_scripts.thing2 import get_site, get_texts
from models.roberta_classifier.train_qual import label_maps

DATA_DIR = "models/psl/data/"
DB_FILENAME = "data/data.db"

news_sites = ['foxnews', 'nytimes', 'wsj', 'washingtonpost', 'breitbart', 'huffpost','bbc']



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
            label = label_maps[task][label]
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

    setting = "excerpt_article"
    # rule_name = "ValSpin>>ValEconChange"
    # task = "econ_change"
    rule_name = "ValType>>ValFrame"
    task = "frame"
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

        _, test_labels, curr_sites = \
            get_texts(DB_FILENAME,
                      ann_component,
                      task,
                      qual_dict,
                      valid_ids
                     )
        
        

        for i, id in enumerate(valid_ids):
            site = curr_sites[i]
            results[site]['labels'].append(test_labels[i])
            results[site]['predictions'].append(psl_results[k][id])

    for site in news_sites:
        # print(site)
        macro_f1 = f1_score(results[site]['labels'], results[site]['predictions'], average='macro')
        print(round(macro_f1, 3))
        # weighted_f1 = f1_score(results[site]['labels'], results[site]['predictions'], average='weighted')
        # print(round(weighted_f1, 3))
    


    

if __name__ == "__main__":
    main()

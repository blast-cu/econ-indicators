import pickle
import os
import argparse
import pandas as pd
from models.psl.generate_data import load_train_test_data
from models.psl.SETTINGS import SETTINGS, PREDICATE_MAP, VALUE_MAP

DATA_DIR = 'models/psl/data/'
RULE_DIR = 'models/psl/data/rules/'

def get_co_occur(lhs_ann, rhs_ann, lhs_val, rhs_val, articles, excerpts):
    count = 0
    for article_id, article_dict in articles.items():
        # print(article_dict)
        if article_dict[rhs_ann] == rhs_val:
            excerpt_list = article_dict['quant_list']
        else: 
            continue

        for excerpt_id in excerpt_list:
            excerpt_dict = excerpts[excerpt_id]
            if excerpt_dict[lhs_ann] == lhs_val:
                count += 1

    return count

def get_pred_co_occur(lhs_ann, rhs_ann, lhs_val, rhs_val, excerpts):
    excerpt_keys = list(excerpts.keys())
    count = 0
    for index, global_id in enumerate(excerpt_keys):
        article_id, local_id = global_id.split('_')
        article_id = int(article_id)
        local_id = int(local_id)

        prev_global_id = excerpt_keys[index-1]
        prev_article_id, prev_local_id = prev_global_id.split('_')
        prev_article_id = int(prev_article_id)
        prev_local_id = int(prev_local_id)

        if article_id == prev_article_id:
            if local_id == prev_local_id + 1:
                if excerpts[prev_global_id][lhs_ann] == lhs_val:
                    if excerpts[global_id][rhs_ann] == rhs_val:
                        count += 1
    return count
        

def main(args):

    setting = args.s
    try:
        setting_dict = SETTINGS[setting]

    except Exception as e:
        print(e)
        raise ValueError('Unknown setting: ' + setting)

    split_dir = "data/clean/"
    splits_dict = pickle.load(open(split_dir + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(split_dir + 'quant_dict', 'rb'))

    big_learn_freq_report = {}
    big_eval_freq_report = {}
    for split_num in range(5):
        big_learn_freq_report[split_num] = {}
        big_eval_freq_report[split_num] = {}
        # load train and test data for split    
        learn_articles, learn_excerpts, eval_articles, eval_excerpts = \
            load_train_test_data(splits_dict[split_num],
                                 qual_dict,
                                 quant_dict)
        split_setting_dir = os.path.join(DATA_DIR, f'split{split_num}', setting)

        # loop over all combinations
        for c in setting_dict['combinations']:

            pred_left = c[0]
            pred_right = c[1]

            curr_rule_file = None
            rule_files = os.listdir(split_setting_dir)
            
            for rule_file in rule_files:
                lower_rule_file = (rule_file).split('>>')
                # print(lower_rule_file)
                # print(c)
                if pred_left in lower_rule_file[0] and pred_right in lower_rule_file[1]:
                    curr_rule_file = rule_file
                    break
            if curr_rule_file is None:
                raise ValueError('Unknown rule file: ' + str(c))
                
            
            split_setting_rule_dir = os.path.join(split_setting_dir, curr_rule_file)
            

            lhs_ann = PREDICATE_MAP[pred_left]
            rhs_ann = PREDICATE_MAP[pred_right]

            lhs_vals = VALUE_MAP[lhs_ann]
            rhs_vals = VALUE_MAP[rhs_ann]

            filename = 'learn_co_occur.csv'
            index = 0
            learn_count = 0
            learn_co_occur_dict = {}
            for lhs_val in lhs_vals:
                for rhs_val in rhs_vals:
                    learn_co_occur_dict[index] = {}
                    learn_co_occur_dict[index]['lhs'] = lhs_val
                    learn_co_occur_dict[index]['rhs'] = rhs_val
                    if setting == 'excerpt_article':
                        count = get_co_occur(lhs_ann, rhs_ann, lhs_val, rhs_val, learn_articles, learn_excerpts)
                    elif setting == 'neighbors':
                        count = get_pred_co_occur(lhs_ann, rhs_ann, lhs_val, rhs_val, learn_excerpts)

                    learn_co_occur_dict[index]['count'] = count
                    learn_count += count

                    index += 1
            # print(learn_co_occur_dict)
            big_learn_freq_report[split_num][curr_rule_file] = learn_count
            pd.DataFrame.from_dict(learn_co_occur_dict, orient='index').to_csv(os.path.join(split_setting_rule_dir, filename))

            filename = 'eval_co_occur.csv'
            index = 0
            learn_co_occur_dict = {}
            eval_count = 0
            for lhs_val in lhs_vals:
                for rhs_val in rhs_vals:
                    learn_co_occur_dict[index] = {}
                    learn_co_occur_dict[index]['lhs'] = lhs_val
                    learn_co_occur_dict[index]['rhs'] = rhs_val
                    if setting == 'excerpt_article':
                        count = get_co_occur(lhs_ann, rhs_ann, lhs_val, rhs_val, eval_articles, eval_excerpts)
                    elif setting == 'neighbors':
                        count = get_pred_co_occur(lhs_ann, rhs_ann, lhs_val, rhs_val, eval_excerpts)
                    learn_co_occur_dict[index]['count'] = count
                    eval_count += count
                    index += 1
            # print(learn_co_occur_dict)
            pd.DataFrame.from_dict(learn_co_occur_dict, orient='index').to_csv(os.path.join(split_setting_rule_dir, filename))

            big_eval_freq_report[split_num][curr_rule_file] = eval_count
            
    results_dir = os.path.join(DATA_DIR, 'results', setting, 'eval_co_occur.csv')
    pd.DataFrame.from_dict(big_eval_freq_report, orient='index').to_csv(results_dir)

    results_dir = os.path.join(DATA_DIR, 'results', setting, 'learn_co_occur.csv')
    pd.DataFrame.from_dict(big_learn_freq_report, orient='index').to_csv(results_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--s', required=True, help='setting mode')
    args = parser.parse_args()
    main(args)

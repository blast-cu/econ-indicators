#!/usr/bin/env python3

import os
import argparse

from tqdm import tqdm

from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from models.psl.SETTINGS import SETTINGS
from models.psl.run_inference import add_predicates, add_rules, add_learn_data, add_eval_data, learn, infer, write_results

MODEL_NAME = 'annotations'

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(THIS_DIR, 'data')
RULE_DIR = os.path.join(DATA_DIR, 'rules')

VERBOSE = True
ADDITIONAL_PSL_OPTIONS = {
    'runtime.log.level': 'INFO'
    # 'runtime.db.type': 'Postgres',
    # 'runtime.db.pg.name': 'psl',
}

"""
Script for running inference with train/test 5 split data
"""

def main(args):

    # establish setting parameters
    setting = args.s
    try:
        setting_dict = SETTINGS[setting]
    except Exception as e:
        print(e)
        raise ValueError('Unknown setting: ' + setting)

    # get rule directory from data/rules/[setting]
    try:
        rule_files = os.listdir(setting_dict['rule_dir'])
    except FileNotFoundError:
        raise ValueError('Unknown rule directory: ' + setting_dict['rule_dir'])


    print('Setting: ' + setting)
    global SPLIT_DIR
    SPLIT_DIR = os.path.join(DATA_DIR, f'final{args.final_split}')  # FINAL
    SPLIT_SETTING_DIR = os.path.join(SPLIT_DIR, setting)
    os.makedirs(SPLIT_SETTING_DIR, exist_ok=True)

    for rule_file in rule_files:

        rule_name = rule_file.split('.')[0]
        rule_file = os.path.join(setting_dict['rule_dir'], rule_file)
    
        # create directory for split data
        output_dir = os.path.join(SPLIT_SETTING_DIR, rule_name)
        os.makedirs(output_dir, exist_ok=True)

        # create model instance
        model_name = f'{MODEL_NAME}_{setting}_{rule_name}_final{args.final_split}'
        model = Model(model_name)

        predicates = add_predicates(model)
        add_rules(model, rule_file, args.no_constraints)

        # Weight Learning
        if setting_dict['learn']:
            learn(model, predicates)

        learned_rule_file = os.path.join(output_dir, 'learned_rules.txt')
        with open(learned_rule_file, 'w') as f:
            for rule in model.get_rules():
                f.write(str(rule) + '\n')

        # inference
        results = infer(model, predicates)
        write_results(results, model, output_dir)



if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--s', required=True, help='setting mode')
    parser.add_argument('--no_constraints', action='store_true', default=False, help='no constraints')
    parser.add_argument('--final_split', required=False, help='final split number')
    args = parser.parse_args()
    main(args)

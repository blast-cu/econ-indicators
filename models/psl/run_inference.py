#!/usr/bin/env python3

import os
import argparse

from tqdm import tqdm

from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from models.psl.SETTINGS import SETTINGS

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
    ## temporary for fixing lack of macro type
    # if split_num == 2:
    #     filtered_rule_files = rule_files
    # else:
    #     filtered_rule_files = [f for f in rule_files if 'MacroType' in f]

    # for rule_file in filtered_rule_files:
    #######

    
    for split_num in tqdm(range(5), desc="Processing splits"):
    
        global SPLIT_DIR
        SPLIT_DIR = os.path.join(DATA_DIR, f'split{split_num}')
        # SPLIT_DIR = os.path.join(DATA_DIR, f'final{args.final_split}')  # FINAL
        SPLIT_SETTING_DIR = os.path.join(SPLIT_DIR, setting)
        os.makedirs(SPLIT_SETTING_DIR, exist_ok=True)

        for rule_file in rule_files:

            rule_name = rule_file.split('.')[0]
            rule_file = os.path.join(setting_dict['rule_dir'], rule_file)
        
            # create directory for split data
            output_dir = os.path.join(SPLIT_SETTING_DIR, rule_name)
            os.makedirs(output_dir, exist_ok=True)

            # create model instance
            # model_name = f'{MODEL_NAME}_{setting}_{rule_name}_{split_num}'
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


def write_results(results, model, dir):
    out_dir = os.path.join(SPLIT_DIR, dir)
    out_dir = os.path.join(out_dir, 'inferred_predicates')
    os.makedirs(out_dir, exist_ok=True)

    for predicate in model.get_predicates().values():
        if (predicate not in results):
            continue

        out_path = os.path.join(out_dir, "%s.txt" % (predicate.name()))
        results[predicate].to_csv(out_path, sep="\t", header=False, index=False)
    print('Results written to ' + out_dir)


def add_predicates(model):

    if VERBOSE: 
        print('Adding predicates...')

    predicate_file = os.path.join(DATA_DIR, 'predicates.txt')
    with open(predicate_file) as f:
        predicate_strs = f.readlines()

    predicates = []
    for p in predicate_strs:
        p = p.strip()

        if VERBOSE:
            print(p)

        size = 2
        if p == 'HasTypeAnn' or p == 'HasFrameAnn':
            size = 1
        predicate = Predicate(p, size=size)
        model.add_predicate(predicate)
        predicates.append(predicate)

    return predicates


def add_rules(model, rule_file, no_hard_constraints):

    if VERBOSE:
        print('\nAdding rules...')

    # mutex constraints, default pred >> val
    constraint_file = os.path.join(DATA_DIR, 'constraints.txt')
    with open(constraint_file) as f:
        constraint_strs = f.readlines()

    for c in constraint_strs:
        c = c.strip()
        if VERBOSE:
            print(c)
        constraint = Rule(c)
        model.add_rule(constraint)

    if not no_hard_constraints:  # add macro type constraints
        constraint_file = os.path.join(DATA_DIR, 'hard_constraints.txt')
        with open(constraint_file) as f:
            constraint_strs = f.readlines()
        for c in constraint_strs:
            c = c.strip()
            if VERBOSE:
                print(c)
            constraint = Rule(c)
            model.add_rule(constraint)

    with open(rule_file) as f:
        rule_strs = f.readlines()

    for r in rule_strs:
        r = r.strip()
        if VERBOSE:
            print(r)
        rule = Rule(r)
        model.add_rule(rule)


def add_learn_data(predicates):
    _add_data('learn', predicates)


def add_eval_data(predicates):
    _add_data('eval', predicates)

def check_data(path):
    # make sure the data file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # read in the tab separated txt file
    with open(path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            raise ValueError(f"Data file is empty: {path}")
        
    # check that the final entry isn't nan
    clean_lines = []
    for line in lines:
        if 'nan' in line:
            print(f"Line removed from data file {path}: {line.strip()}")
        else:
            clean_lines.append(line)
    
    # overwrite the file with the cleaned lines
    if len(clean_lines) != len(lines):
        print(f"Replacing dirty data file {path} with cleaned version.")
        os.remove(path)  # remove the original file if it has been cleaned
        with open(path, 'w') as f:
            f.writelines(clean_lines) 

def _add_data(type, predicates):

    if VERBOSE: 
        print('\nAdding data files for ' + type + '...')

    split_data_dir = os.path.join(SPLIT_DIR, type)
    filenames = os.listdir(split_data_dir)
    files = [f.split('.')[0] for f in filenames]
    files = [[f.split('_')[0].upper(),f.split('_')[1]]  for f in files]

    for p in predicates:
        for i, f in enumerate(files): 
            if p.name() == f[0]:
                # if VERBOSE: print(f[0] + ' added to ' + p.name())
                filename = filenames[i]
                path = os.path.join(split_data_dir, filename)
                check_data(path)
                if f[1] == 'obs':
                    p.add_data_file(Partition.OBSERVATIONS, path)
                elif f[1] == 'truth':
                    p.add_data_file(Partition.TRUTH, path)
                elif f[1] == 'target':
                    p.add_data_file(Partition.TARGETS, path)
                else:
                    raise ValueError('Unknown predicate: ' + f[1])
                if VERBOSE: 
                    print(filename + ' added to ' + p.name())

def learn(model, predicates):
    add_learn_data(predicates)
    model.learn(psl_options=ADDITIONAL_PSL_OPTIONS)

def infer(model, predicates):
    add_eval_data(predicates)
    return model.infer(psl_options=ADDITIONAL_PSL_OPTIONS)


if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--s', required=True, help='setting mode')
    parser.add_argument('--no_constraints', action='store_true', default=False, help='no constraints')
    parser.add_argument('--final_split', required=False, help='final split number')
    args = parser.parse_args()
    main(args)

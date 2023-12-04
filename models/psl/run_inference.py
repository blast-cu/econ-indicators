#!/usr/bin/env python3

import os

from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

MODEL_NAME = 'annotations'

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(THIS_DIR, 'data')
SPLIT_DIR = os.path.join(DATA_DIR, 'split0')

VERBOSE = True

ADDITIONAL_PSL_OPTIONS = {
    'runtime.log.level': 'INFO'
    # 'runtime.db.type': 'Postgres',
    # 'runtime.db.pg.name': 'psl',
}

NUM_CATEGORIES = 7

def main():

    model = Model(MODEL_NAME)
    predicates = add_predicates(model)
    add_rules(model)

    # Weight Learning
    learn(model, predicates)

    print('Learned Rules:')
    for rule in model.get_rules():
        print('   ' + str(rule))

    # Inference
    results = infer(model, predicates)
    write_results(results, model)


def write_results(results, model):
    out_dir = os.path.join(SPLIT_DIR, 'inferred-predicates')
    os.makedirs(out_dir, exist_ok=True)

    for predicate in model.get_predicates().values():
        if (predicate not in results):
            continue

        out_path = os.path.join(out_dir, "%s.txt" % (predicate.name()))
        results[predicate].to_csv(out_path, sep="\t", header=False, index=False)


def add_predicates(model):

    if VERBOSE: print('Adding predicates...')

    predicate_file=os.path.join(DATA_DIR, 'predicates.txt')
    with open(predicate_file) as f:
        predicate_strs=f.readlines()

    predicates = []
    for p in predicate_strs:
        p = p.strip()
        if VERBOSE: print(p)
        predicate = Predicate(p, size=2)
        model.add_predicate(predicate)
        predicates.append(predicate)

    return predicates


def add_rules(model):

    if VERBOSE: print('\nAdding rules...')

    predicate_file = os.path.join(DATA_DIR, 'rules.txt')
    with open(predicate_file) as f:
        rule_strs = f.readlines()
    
    for r in rule_strs:
        r = r.strip()
        if VERBOSE: print(r)
        rule = Rule(r)
        model.add_rule(rule)


def add_learn_data(predicates):
    _add_data('learn', predicates)


def add_eval_data(predicates):
    _add_data('eval', predicates)

def _add_data(split, predicates):

    if VERBOSE: print('\nAdding data files for ' + split + '...')

    split_data_dir = os.path.join(SPLIT_DIR, split)
    filenames = os.listdir(split_data_dir)
    files = [f.split('.')[0] for f in filenames]
    files = [[f.split('_')[0].upper(),f.split('_')[1]]  for f in files]
    

    for p in predicates:
        for i, f in enumerate(files): 
            if p.name() == f[0]:
                # if VERBOSE: print(f[0] + ' added to ' + p.name())
                filename = filenames[i]
                if f[1] == 'obs':
                    path = os.path.join(split_data_dir, filename)
                    p.add_data_file(Partition.OBSERVATIONS, path)
                elif f[1] == 'truth':
                    path = os.path.join(split_data_dir, filename)
                    p.add_data_file(Partition.TRUTH, path)
                elif f[1] == 'target':
                    path = os.path.join(split_data_dir, filename)
                    p.add_data_file(Partition.TARGETS, path)
                else:
                    raise ValueError('Unknown predicate: ' + f[1])

                if VERBOSE: print(filename + ' added to ' + p.name())



def learn(model, predicates):
    add_learn_data(predicates)
    model.learn(psl_options = ADDITIONAL_PSL_OPTIONS)

def infer(model, predicates):
    add_eval_data(predicates)
    return model.infer(psl_options = ADDITIONAL_PSL_OPTIONS)

if (__name__ == '__main__'):
    main()

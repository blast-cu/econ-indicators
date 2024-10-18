import argparse
import os
import re
import pickle
import itertools
from torch.utils.data import DataLoader, Dataset

from itertools import permutations

OUT_DIR = 'models/psl/data'

qual_map = {
    'frame': {
        'business': 0,
        'industry': 1,
        'macro': 2,
        'government': 3,
        'other': 4,
        'personal': 5
    },
    'econ_rate': {
        'good': 0,
        'poor': 1,
        'none': 2,
        'irrelevant': 3
    },
    'econ_change': {
        'better': 0,
        'worse': 1,
        'same': 2,
        'none': 3,
        'irrelevant': 4
    },
}


quant_map = {

    'type': {
        'macro': 0,
        'industry': 1,
        'government': 2,
        'personal': 3,
        'business': 4,
        'other': 5
    },
    'macro_type': {
        'jobs': 0,
        'retail': 1,
        'interest': 2,
        'prices': 3,
        'energy': 4,
        'wages': 5,
        'macro': 6,
        'market': 7,
        'currency': 8,
        'housing': 9,
        'other': 10,
        'none': 11
    },
    'spin': {
        'pos': 0,
        'neg': 1,
        'neutral': 2
    }
}

qual_pred_map = {
    'ValFrame': 'frame',
    'ValEconRate': 'econ_rate',
    'ValEconChange': 'econ_change'}

quant_pred_map = {
    'ValType': 'type',
    'ValMacroType': 'macro_type',
    'ValIndustryType': 'industry_type',
    'ValGovType': 'gov_type',
    'ValExpenditureType': 'expenditure_type',
    'ValRevenueType': 'revenue_type',
    'ValSpin': 'spin'
}


def camel_case(s, upper: bool = False):
    """
    Convert a string to camel case.

    Args:
        s (str): The input string.
        upper (bool, optional): Whether to capitalize the first letter of the camel case string. Defaults to False.

    Returns:
        str: The camel case string.
    """

    # Use regular expression substitution to replace underscores and hyphens with spaces,
    # then title case the string (capitalize the first letter of each word), and remove spaces
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")

    # Join the string, ensuring the first letter is lowercase
    if upper:
        return ''.join([s[0].upper(), s[1:]])
    else:
        return ''.join([s[0].lower(), s[1:]])


def get_predicates(annotation_map, type: str):
    """
    Generate a list of predicates based on the given annotation map and type.

    Args:
        annotation_map (dict): A dictionary mapping annotations to their values.
        type (str): The type of the predicates.

    Returns:
        list: A list of predicates generated based on the annotation map and type.
    """
    predicates = []
    for annotation in annotation_map.keys():

        # to upper camel case
        annotation_formatted = camel_case(annotation, upper=True)
        predicate = f'{type}{annotation_formatted}'
        predicates.append(predicate)

    return predicates


def get_pred_val_rules(predicates, annotation_type):
    """
    Generate rules for predicting value based on predicates and annotation type.

    Args:
        predicates (list): List of tuples containing predicates for prediction and value.
        annotation_type (str): Type of annotation, either 'qual' or 'quant'.

    Returns:
        list: List of generated rules.

    """
    rules = []
    if annotation_type == 'qual' or annotation_type == 'quant':
        if annotation_type == 'qual':
            sym1 = 'A'
        elif annotation_type == 'quant':
            sym1 = 'E'
        
        for pred_predicate, val_predicate in predicates:
            rule = f'1.0: {pred_predicate}({sym1}, B) >> {val_predicate}({sym1}, B) ^2'
            rules.append(rule)

    elif annotation_type == 'agreement':
        sym1 = 'E1'
        sym2 = 'E2'
        for pred_predicate, val_predicate in predicates:
            rule = f'1.0: {pred_predicate}({sym1}, {sym2}, B) >> {val_predicate}({sym1}, {sym2}, B) ^2'
            rules.append(rule)


    return rules

def mutex_constraint(predicates, type: str):
    """
    Generate rules ensuring only one val prediction per annotation type.

    Args:
        predicates (list): List of predicates.
        type (str): Type of constraint ('qual' or 'quant').

    Returns:
        list: List of generated mutex constraints.
    """

    rules = []

    if type == 'qual':
        sym1 = 'A'
    elif type == 'quant':
        sym1 = 'E'

    for predicate in predicates:
        rule = f'{predicate}({sym1}, +B) = 1 .'
        rules.append(rule)

    return rules


def write_file(file_path, lines):
    """
    Write the given lines to a file at the specified file path.

    Args:
        file_path (str): The path of the file to write.
        lines (list): The lines to write to the file.

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')


def get_inter_rules(predicates, pred_map, map):
    """
    Generate inter-annotation type combinations based on the given predicates, predicate map, and map.

    Args:
        predicates (list): List of predicates.
        pred_map (dict): Mapping of predicates to annotation types.
        map (dict): Mapping of annotation types to values.

    Returns:
        list: List of generated inter-rule combinations.
    """
    rules = []

    pred_combinations = itertools.combinations(predicates, 2)
    for c in pred_combinations:
        ann_type_1 = pred_map[c[0]]
        ann_type_2 = pred_map[c[1]]
        vals_1 = map[ann_type_1].keys()
        vals_2 = map[ann_type_2].keys()
        for v1 in vals_1:
            for v2 in vals_2:
                rule_str = f"1.0: {c[0]}(A, '{v1}') >> {c[1]}(A, '{v2}') ^2"
                rules.append(rule_str)

                rule_str = f"1.0: {c[1]}(A, '{v2}') >> {c[0]}(A, '{v1}') ^2"
                rules.append(rule_str)

    return rules

def get_inter_rules_2(qual_predicates, quant_predicates):
    """
    Generate interrelatedness rules between qualitative and quantitative 
    predicates.

    Args:
        qual_predicates (list): List of qualitative predicates.
        quant_predicates (list): List of quantitative predicates.

    Returns:
        list: List of generated interaction rules.
    """
    rules = []

    for qual in qual_predicates:
        ann_type = qual_pred_map[qual]
        qual_vals = qual_map[ann_type].keys()

        for quant in quant_predicates:
            ann_type = quant_pred_map[quant]
            quant_vals = quant_map[ann_type].keys()

            for quant_val in quant_vals:
                for qual_val in qual_vals:
                    rule_str = f"1.0: Contains(A, E) & {qual}(A, '{qual_val}') >> {quant}(E, '{quant_val}') ^2"
                    rules.append(rule_str)
                    rule_str = f"1.0: Contains(A, E) & {quant}(E, '{quant_val}') >> {qual}(A, '{qual_val}') ^2"
                    rules.append(rule_str)

    return rules


def get_inter_rules_3(quant_predicates):

    rules = []

    for quant in quant_predicates:
        ann_type = quant_pred_map[quant]
        quant_vals = quant_map[ann_type].keys()

        for quant_val in quant_vals:

            for quant_type2 in quant_predicates:
                ann_type = quant_pred_map[quant_type2]
                quant_vals2 = quant_map[ann_type].keys()

                for quant_val in quant_vals:
                    for quant_val2 in quant_vals2:
                        rule_str = f"1.0: Precedes(A, E) & {quant}(A, '{quant_val}') >> {quant_type2}(E, '{quant_val2}') ^2"
                        rules.append(rule_str)

                        # rule_str = f"1.0: Neighbors(A, E) & {quant}(E, '{quant_val}') >> {quant_type2}(A, '{quant_val2}') ^2"
                        # rules.append(rule_str)

    rules = list(set(rules))

    return rules

def macro_type_constraints():

    constraints = []
    for val in quant_map['type'].keys():
        if val != 'macro':
            constraint_str = f"ValType(A, '{val}') >> ValMacroType(A, 'none') ."
            constraints.append(constraint_str)

    for val in quant_map['macro_type'].keys():
        if val != 'none':
            constraint_str = f"HasTypeAnn(A) & ValMacroType(A, '{val}') >> ValType(A, 'macro') ."
            constraints.append(constraint_str)
    
    constraint_str = f"HasTypeAnn(A) & ValMacroType(A, 'none') >> "
    for val in quant_map['type'].keys():
        if val != 'macro':
            constraint_str += f" ValType(A, '{val}') |"
    constraint_str = constraint_str[:-1]

    constraints.append(constraint_str + '.')

    return constraints

def main(args):

    # ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # generate all predicates
    val_qual_predicates = get_predicates(qual_map, 'Val')
    val_quant_predicates = get_predicates(quant_map, 'Val')
    pred_qual_predicates = get_predicates(qual_map, 'Pred')
    pred_quant_predicates = get_predicates(quant_map, 'Pred')

    agreement_pred_predicates = ['PredAgreeType', 'PredAgreeSpin', 'PredAgreeMacroType']
    agreement_val_predicates = ['ValAgreeType', 'ValAgreeSpin', 'ValAgreeMacroType']
    agreement_predicates = agreement_pred_predicates + agreement_val_predicates

    predicates = val_qual_predicates + val_quant_predicates + \
        pred_qual_predicates + pred_quant_predicates \
        + agreement_predicates \
        + ['Contains', 'Precedes', 'HasTypeAnn']

    # write predicates to file
    filename = 'predicates.txt'
    file_path = os.path.join(OUT_DIR, filename)
    # write_file(file_path, predicates)

    constraints = []
    # rules linking predictions and values
    qual_predicates = zip(pred_qual_predicates, val_qual_predicates)
    constraints += get_pred_val_rules(qual_predicates, 'qual')

    quant_predicates = zip(pred_quant_predicates, val_quant_predicates)
    constraints += get_pred_val_rules(quant_predicates, 'quant')

    agreement_predicates = zip(agreement_pred_predicates, agreement_val_predicates)
    
    constraints += get_pred_val_rules(agreement_predicates, 'agreement')

    # interrelatedness between qual and quant, respectively
    rules = get_inter_rules(val_qual_predicates, qual_pred_map, qual_map)
    rules += get_inter_rules(val_quant_predicates, quant_pred_map, quant_map)

    # interrelatedness between qual and quant
    rules += get_inter_rules_2(val_qual_predicates, val_quant_predicates)

    # interrelatedness between quants
    rules += get_inter_rules_3(val_quant_predicates)

    # # mutex constraints
    constraints += mutex_constraint(val_qual_predicates, 'qual')
    constraints += mutex_constraint(val_quant_predicates, 'quant')

    # macro type constraint
    constraints += macro_type_constraints()

    rules.sort()
    
    # write rules to file
    rule_dir = os.path.join(OUT_DIR, 'rules')
    os.makedirs(rule_dir, exist_ok=True)
    all_rules_dir = os.path.join(rule_dir, 'all_rules')
    os.makedirs(all_rules_dir, exist_ok=True)

    filename = os.path.join(all_rules_dir, 'all_rules.txt')
    # write_file(filename, rules)

    constraint_filename = os.path.join(OUT_DIR, 'constraints.txt')
    # write_file(constraint_filename, constraints)

    article_predicates = [ 'ValFrame', 'ValEconRate', 'ValEconChange']
    excerpt_predicates = ['ValSpin', 'ValType', 'ValMacroType']


    # INTER ARTICLE RULES ##########################################
    # generate inter-annotation rule files
    article_perms = list(permutations(article_predicates, 2))
    # print(article_perms)
    setting_dir = os.path.join(rule_dir, 'inter_article')
    os.makedirs(setting_dir, exist_ok=True)
    for p in article_perms:
        filename = f'{p[0]}>>{p[1]}.txt'
        file_path = os.path.join(setting_dir, filename)
        out_rules = []
        for r in rules:
            r_split = r.split('>>')
            if p[0] in r_split[0] and p[1] in r_split[1]:
                out_rules.append(r.strip())
        write_file(file_path, out_rules)

    
    # excerpt >> article rules ##########################################
    combinations = []
    setting_dir = os.path.join(rule_dir, 'excerpt_article')
    os.makedirs(setting_dir, exist_ok=True)

    for i in excerpt_predicates:
        for j in article_predicates:
            limit = [i,j]
            combinations.append(limit)
    # print(combinations)

    for p in combinations:
        filename = f'{p[0]}>>{p[1]}.txt'
        file_path = os.path.join(setting_dir, filename)
        out_rules = []
        for r in rules:
            r_split = r.split('>>')
            if p[0] in r_split[0] and p[1] in r_split[1]:
                out_rules.append(r.strip())
        # write_file(file_path, out_rules)
    


    # # NEIGHBOR RULES ##########################################
    combinations = []
    setting_dir = os.path.join(rule_dir, 'precedes')
    os.makedirs(setting_dir, exist_ok=True)
    for i in excerpt_predicates:
        for j in excerpt_predicates:
            limit = [i, j]
            combinations.append(limit)
    # print(combinations)
    for p in combinations:
        filename = f'{p[0]}>>{p[1]}.txt'
        file_path = os.path.join(setting_dir, filename)
        out_rules = []
        for r in rules:
            r_split = r.split('>>')
            if p[0] in r_split[0] and p[1] in r_split[1]:
                out_rules.append(r.strip())
        # write_file(file_path, out_rules)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=False)
    args = parser.parse_args()
    main(args)

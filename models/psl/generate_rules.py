import argparse
import os
import re
import pickle
import itertools
from torch.utils.data import DataLoader, Dataset

OUT_DIR = 'models/psl/data'

qual_map = {
    'frame': {
        'business': 0,
        'industry': 1,
        'macro': 2,
        'government': 3,
        'other': 4
    },
    'econ_rate': {
        'good': 0,
        'poor': 1,
        'none': 2
    },
    'econ_change': {
        'better': 0,
        'worse': 1,
        'same': 2,
        'none': 3
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
    }
    # },
    # 'macro_type': {}, 
    # 'industry_type': {},
    # 'gov_type': {},
    # 'expenditure_type': {},
    # 'revenue_type': {},
    # 'spin': {
    #     'pos': 0,
    #     'neg': 1,
    #     'neutral': 2
    # }
}

qual_pred_map = {
    'ValFrame':'frame',
    'ValEconRate': 'econ_rate',
    'ValEconChange': 'econ_change' }

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

    if annotation_type == 'qual':
        sym1 = 'A'
    elif annotation_type == 'quant':
        sym1 = 'E'
    
    for pred_predicate, val_predicate in predicates:
        rule = f'1.0: {pred_predicate}({sym1}, B) >> {val_predicate}({sym1}, B) ^2'
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

def main(args):

    # ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # generate all predicates
    val_qual_predicates = get_predicates(qual_map, 'Val')
    val_quant_predicates = get_predicates(quant_map, 'Val')
    pred_qual_predicates = get_predicates(qual_map, 'Pred')
    pred_quant_predicates = get_predicates(quant_map, 'Pred')

    predicates = val_qual_predicates + val_quant_predicates + \
        pred_qual_predicates + pred_quant_predicates \
        + ['Contains']

    # write predicates to file
    filename = 'predicates.txt'
    file_path = os.path.join(OUT_DIR, filename)
    write_file(file_path, predicates)

    # rules linking predictions and values
    qual_predicates = zip(pred_qual_predicates, val_qual_predicates)
    rules = get_pred_val_rules(qual_predicates, 'qual')

    quant_predicates = zip(pred_quant_predicates, val_quant_predicates)
    rules += get_pred_val_rules(quant_predicates, 'quant')

    # interrelatedness between qual and quant, respectively
    rules += get_inter_rules(val_qual_predicates, qual_pred_map, qual_map)
    rules += get_inter_rules(val_quant_predicates, quant_pred_map, quant_map)

    # interrelatedness between qual and quant
    rules += get_inter_rules_2(val_qual_predicates, val_quant_predicates)

    # mutex constraints
    rules += mutex_constraint(val_qual_predicates, 'qual')
    rules += mutex_constraint(val_quant_predicates, 'quant')
    
    # write rules to file
    filename = 'rules.txt'
    file_path = os.path.join(OUT_DIR, filename)
    write_file(file_path, rules)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=False)
    args = parser.parse_args()
    main(args)

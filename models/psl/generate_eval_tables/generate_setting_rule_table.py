import argparse
import os
import pandas as pd

RESULTS = 'models/psl/data/results/'

def main(args):

    arg_dir = (args.dir).strip()
    setting_dir = os.path.join(RESULTS, arg_dir)
    metric = -1  # -1 for weighted avg, -2 for macro avg
    try:
        rule_dirs = os.listdir(setting_dir)

    except FileNotFoundError:
        raise ValueError('Unknown results directory: ' + setting_dir)

    setting_dict = {}
    rows = ['frame', 'econ_rate', 'econ_change', 'type', 'spin', 'macro_type']

    for rule_name in rule_dirs:
        macro_rule_dict = {}
        weighted_rule_dict = {}
        for r in rows:
            filename = r + '_classification_report.csv'

            try:
                df = pd.read_csv(os.path.join(setting_dir, rule_name, filename), header=0)
                f1 = df.iloc[-1]['f1-score']
                weighted_rule_dict[r] = f1
                f1 = df.iloc[-2]['f1-score']
                macro_rule_dict[r] = f1
            except NotADirectoryError:
                print("Skipping unformatted directory")
                continue
        if len(macro_rule_dict) != 0:
            name = rule_name + '_macro'
            setting_dict[name] = macro_rule_dict
        if len(weighted_rule_dict) != 0:
            name = rule_name + '_weighted'
            setting_dict[name] = weighted_rule_dict

    df = pd.DataFrame().from_dict(setting_dict).reindex(rows)
    filename = f"{arg_dir}_classification_report.csv"
    df.to_csv(os.path.join(setting_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--dir', help='Path to the input directory')
    args = parser.parse_args()
    main(args)
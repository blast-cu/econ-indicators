import argparse
import os
import pandas as pd

RESULTS = 'models/psl/data/results/'

def main(args):

    arg_dir = (args.s).strip()
    setting_dir = os.path.join(RESULTS, arg_dir)
    metric = -1  # -1 for weighted avg, -2 for macro avg
    try:
        rule_dirs = os.listdir(setting_dir)

    except FileNotFoundError:
        raise ValueError('Unknown results directory: ' + setting_dir)

    macro_setting_dict = {}
    weighted_setting_dict = {}

    rows = ['frame', 'econ_rate', 'econ_change', 'type', 'macro_type', 'spin']

    for rule_name in rule_dirs:
        macro_rule_dict = {}
        weighted_rule_dict = {}
        for r in rows:
            filename = r + '_classification_report.csv'

            try:
                df = pd.read_csv(os.path.join(setting_dir, rule_name, filename), header=0)
                f1 = df.iloc[-1]['f1-score']
                weighted_rule_dict[r] = round(f1, 3)
                f1 = df.iloc[-2]['f1-score']
                macro_rule_dict[r] = round(f1, 3)
            except NotADirectoryError:
                print("Skipping unformatted directory")
                continue
        if len(macro_rule_dict) != 0:
            name = rule_name
            macro_setting_dict[name] = macro_rule_dict
        if len(weighted_rule_dict) != 0:
            name = rule_name
            weighted_setting_dict[name] = weighted_rule_dict

    df = pd.DataFrame().from_dict(macro_setting_dict).reindex(rows)
    filename = f"{arg_dir}_macro_classification_report.csv"
    df.to_csv(os.path.join(setting_dir, filename))

    df = pd.DataFrame().from_dict(weighted_setting_dict).reindex(rows)
    filename = f"{arg_dir}_weighted_classification_report.csv"
    df.to_csv(os.path.join(setting_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--s', help='rule setting to be evaluated')
    args = parser.parse_args()
    main(args)

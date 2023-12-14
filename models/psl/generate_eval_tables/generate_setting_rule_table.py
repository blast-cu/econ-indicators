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
        rule_dict = {}
        for r in rows:
            filename = r + '_classification_report.csv'

            try:
                df = pd.read_csv(os.path.join(setting_dir, rule_name, filename), header=0)
                f1 = df.iloc[metric]['f1-score']
                rule_dict[r] = f1
            except NotADirectoryError:
                print("Skipping unformatted directory")
                continue
        if len(rule_dict) != 0:
            setting_dict[rule_name] = rule_dict

    df = pd.DataFrame().from_dict(setting_dict).reindex(rows)
    if metric == -1:
        filename = 'weighted_f1.csv'
    elif metric == -2:
        filename = 'macro_f1.csv'
    else:
        raise ValueError('Unknown metric: ' + str(metric))
    df.to_csv(os.path.join(setting_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--dir', help='Path to the input directory')
    args = parser.parse_args()
    main(args)

import pickle
import data_utils.model_utils.dataset as d
from models.mistral.quant import def_map
from sklearn.metrics import classification_report
import pandas as pd
import os
import argparse


def main(args):
    SHOTS = args.s
    DICT_PATH = f'data/mistral_results/{SHOTS}_shot_results'
    OUT_DIR = f'data/mistral_results/{SHOTS}_shot/'
    os.makedirs(OUT_DIR, exist_ok=True)

    results = pickle.load(open(DICT_PATH, 'rb'))
    clean_results = {}
    for task in d.quant_label_maps.keys():
        print(task)
        clean_results[task] = {}
        clean_results[task]['predictions'] = []
        clean_results[task]['labels'] = results[task]['labels']

        predictions = results[task]['predictions']
        task_map = def_map[task]
        model_answer = list(task_map.keys())[-1]  # set to none of the above
        for idx, p in enumerate(predictions):
            p = p.split('[/INST]')[-1]
            for k, v in task_map.items():
                if v in p:
                    model_answer = k
                    break
            clean_results[task]['predictions'].append(model_answer)

    for task, results in clean_results.items():
        # print(results)
        all_labels = results['labels']
        all_predicted = results['predictions']
        report = classification_report(all_labels,
                                       all_predicted,
                                       output_dict=True,
                                       zero_division=0)

        df = pd.DataFrame(report).transpose()
        df.to_csv(f"{OUT_DIR}{task}classification_report.csv")

    with open(f"{OUT_DIR}clean_results.pkl", "wb") as f:
        pickle.dump(clean_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--s', required=True, help='Number of shots')
    args = parser.parse_args()
    main(args)
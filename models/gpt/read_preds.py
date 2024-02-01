import pickle
import os
from models.psl.evaluate_inference import load_train_test_data
import data_utils.dataset as d
PRED_DIR = 'models/gpt/predictions/'
LABEL_DIR = 'data/clean/'
RESULTS_DIR = 'models/gpt/results/'


label_maps = {
    'type': {
            'macro': 0,
            'industry': 1,
            'government': 2,
            'personal': 3,
            'business': 4,
            'other': 5},
    'spin': {
            'pos': 0,
            'neg': 1,
            'neutral': 2},
    # 'macro_type': {
    #         'jobs': 0,
    #         'retail': 1,
    #         'interest': 2,
    #         'prices': 3,
    #         'energy': 4,
    #         'wages': 5,
    #         'macro': 6,
    #         'market': 7,
    #         'currency': 8,
    #         'housing': 9,
    #         'other': 10,
    #         'none': 11}
}

def main():


    splits_dict = pickle.load(open(LABEL_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(LABEL_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(LABEL_DIR + 'quant_dict_clean', 'rb'))

    full_results = {}


    for split_num in range(5):

        split_results = {}

        prediction_file = os.path.join(PRED_DIR, f'preds_testsplit{split_num}_dict')
        predict_dict = pickle.load(open(prediction_file, 'rb'))

        for p in predict_dict.items():
            print(p)
            print()

        _, _, _, eval_excerpts = \
            load_train_test_data(splits_dict[split_num],
                                 qual_dict,
                                 quant_dict)
        
        for ann in ['type']:
            split_results[ann] = {}
            split_results[ann]['labels'] = []
            split_results[ann]['predictions'] = []

            for global_id in eval_excerpts.keys():
                if eval_excerpts[global_id][ann] != '\x00':
                    label = label_maps[ann][eval_excerpts[global_id][ann]]
                    split_results[ann]['labels'].append(label)
                    
                    pred_str = 'predicted_' + ann
                    pred = label_maps[ann][predict_dict[global_id][pred_str]]
                    split_results[ann]['predictions'].append(pred)
    
        split_results_file = os.path.join(RESULTS_DIR, f'split{split_num}')
        os.makedirs(split_results_file, exist_ok=True)
        for ann in split_results.keys():

            labels = split_results[ann]['labels']
            predictions = split_results[ann]['predictions']
            d.to_csv(ann, labels, predictions, split_results_file)

            if ann not in full_results.keys():
                full_results[ann] = {}
                full_results[ann]['labels'] = []
                full_results[ann]['predictions'] = []
            
            full_results[ann]['labels'] += labels
            full_results[ann]['predictions'] += predictions
    
    full_results_file = os.path.join(RESULTS_DIR, 'full')
    os.makedirs(full_results_file, exist_ok=True)
    for ann in full_results.keys():

        labels = full_results[ann]['labels']
        predictions = full_results[ann]['predictions']
        d.to_csv(ann, labels, predictions, full_results_file)

                    



    

if __name__ == "__main__":
    main()
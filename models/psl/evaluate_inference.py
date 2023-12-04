from models.psl.generate_data import load_train_test_data
import models.psl.generate_rules as gd
import models.utils.dataset as d
import pickle
import os
import sklearn.metrics

DATA_DIR = 'models/psl/data'

def get_pred_dict(filepath, doc_type='quant'):

    with open(filepath) as f:
        spin_lines = f.readlines()
    
    predictions = {}
    for line in spin_lines:
        line = line.strip()
        line = line.split('\t')
        id = line[0]

        if id not in predictions.keys():
            if doc_type == 'qual':
                id = int(id)
            predictions[id] = (line[1], line[2])
        else:
            curr_max = predictions[id][1]
            if line[2] > curr_max:
                predictions[id] = (line[1], line[2])

    for id in predictions.keys():
        
        predictions[id] = predictions[id][0]

    return predictions

def get_labels_dict(articles, annotation_type):
    
        labels_dict = {}
        for id, annotations in articles.items():
            labels_dict[id] = annotations[annotation_type]
    
        return labels_dict

def evaluate(annotation_map, eval_docs, inference_dir, report_dir, split_num, doc_type='quant'):

    for annotation_type in annotation_map.keys():
        file_pred = annotation_type.replace('_', '')
        file_pred = file_pred.upper()
        filename = f'VAL{file_pred}.txt'

        filepath = os.path.join(inference_dir, filename)
        try:
            predictions = get_pred_dict(filepath, doc_type)
        except:
            continue
        labels = get_labels_dict(eval_docs, annotation_type)


        prediction_list = []
        label_list = []

        for id in labels.keys():
            prediction_list.append(predictions[id])
            label_list.append(labels[id])

        sklearn.metrics.accuracy_score(label_list, prediction_list)

        os.makedirs(report_dir, exist_ok=True)
        d.to_csv(annotation_type, label_list, prediction_list, report_dir)

def main():

    # load train and test articles
    split_dir = "data/clean/"
    splits_dict = pickle.load(open(split_dir + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(split_dir + 'quant_dict', 'rb'))

    split_num = 0
    # TODO: loop over splits
    # for split_num in splits_dict.keys():

    _, _, eval_articles, eval_excerpts = \
        load_train_test_data(splits_dict[split_num],
                             qual_dict,
                             quant_dict)
    inference_dir = os.path.join(DATA_DIR, f'split{split_num}', 'inferred-predicates')
    report_dir = os.path.join(DATA_DIR, f'split{split_num}', 'psl_results')
    evaluate(gd.qual_map, eval_articles, inference_dir, report_dir, split_num, doc_type='qual')
    evaluate(gd.quant_map, eval_excerpts, inference_dir, report_dir, split_num)

if __name__ == "__main__":
    main()

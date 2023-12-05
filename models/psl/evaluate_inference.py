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
        if doc_type == 'qual':
                id = int(id)

        ann_value = line[1]
        ann_value_score = float(line[2])

        if id not in predictions.keys():
            predictions[id] = (ann_value, ann_value_score)

        else:
            curr_max = predictions[id][1]
            if ann_value_score > curr_max:
                predictions[id] = (ann_value, ann_value_score)

    for id in predictions.keys():
        predictions[id] = predictions[id][0]

    return predictions

def get_labels_dict(articles, annotation_type):
    
        labels_dict = {}
        for id, annotations in articles.items():
            if annotations[annotation_type] != '\0':
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

        except FileNotFoundError:
            print(f'No predictions for {annotation_type}. Skipping...')
            continue

        labels = get_labels_dict(eval_docs, annotation_type)

        prediction_list = []
        label_list = []
        
        print(f'>>> {annotation_type}')
        for id in labels.keys():
            # if labels[id] != '\0':
            prediction_list.append(predictions[id])
            label_list.append(labels[id])

            if label_list[-1] != prediction_list[-1]:
                print(f'ID: {id} \t Prediction: {prediction_list[-1]} \t Label: {label_list[-1]}')

                

        os.makedirs(report_dir, exist_ok=True)
        d.to_csv(annotation_type, label_list, prediction_list, report_dir)

def write_data_file(articles, excerpts, report_dir, type):

    filepath = os.path.join(report_dir, f'{type}_data.txt')
    with open(filepath, 'w') as f:
        for id, annotations in articles.items():
            line = f'id: {id}'
            for annotation_type in annotations.keys():
                if annotation_type != 'quant_list':
                    ann_val = annotations[annotation_type]
                    if ann_val == '\0':
                        ann_val = 'None'
                    ann = f'\t{annotation_type}: {ann_val} '
                    line += ann
            f.write(line + '\n')
            
            for excerpt_id in annotations['quant_list']:
                cur_excerpt = excerpts[excerpt_id]
                line = f'\texcerpt: {excerpt_id}'
                for annotation_type in cur_excerpt.keys():
                    if annotation_type != 'excerpt':
                        ann_val = cur_excerpt[annotation_type]
                        if ann_val == '\0':
                            ann_val = 'None'
                        line += f'\t{annotation_type}: {ann_val}'
                f.write(line + '\n')
            f.write('\n')

def main():

    # load train and test articles
    split_dir = "data/clean/"
    splits_dict = pickle.load(open(split_dir + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(split_dir + 'quant_dict', 'rb'))

    split_num = 0
    # TODO: loop over splits
    # for split_num in splits_dict.keys():

    learn_articles, learn_excerpts, eval_articles, eval_excerpts = \
        load_train_test_data(splits_dict[split_num],
                             qual_dict,
                             quant_dict)

    inference_dir = os.path.join(DATA_DIR, f'split{split_num}', 'inferred-predicates')
    report_dir = os.path.join(DATA_DIR, f'split{split_num}', 'psl_results')

    write_data_file(learn_articles, learn_excerpts, report_dir, type='learn')
    write_data_file(eval_articles, eval_excerpts, report_dir, type='eval')

    evaluate(gd.qual_map, eval_articles, inference_dir, report_dir, split_num, doc_type='qual')
    evaluate(gd.quant_map, eval_excerpts, inference_dir, report_dir, split_num)

if __name__ == "__main__":
    main()

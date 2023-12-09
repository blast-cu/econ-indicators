import torch
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
import pickle
import os
import random
import re

import models.roberta_classifier.quant_utils as qu
# import nltk
# nltk.download('punkt')

MODELS_DIR = "models/roberta_classifier/tuned_models/quant_predict_models"
SPLIT_DIR = "data/clean/"

label_maps = {
    'type': {
        0: 'macro',
        1: 'industry',
        2: 'government',
        3: 'personal',
        4: 'business',
        5: 'other'
    },
    'type-binary': {
        0: 'macro',
        1: 'other'
    },
    'spin': {
        0: 'positive',
        1: 'negative',
        2: 'neutral'
    },
    'macro_type': {
        0: 'jobs',
        1: 'retail',
        2: 'interest',
        3: 'prices',
        4: 'energy',
        5: 'wages',
        6: 'macro',
        7: 'market',
        8: 'currency',
        9: 'housing',
        10: 'other'
    }
}


def clean_dict(dirty_dict: {}):

    month_list = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november',
                  'december']
    
    weekday_list = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                    'saturday', 'sunday']
    
    year_regex = [re.compile('20[0-9][0-9]\s'), re.compile('19[0-9][0-9]\s')]
    
    clean_dict = {}
    for id, text in dirty_dict.items():
        indicator_text = text['indicator']
        excerpt_text = text['excerpt']

        indicator_text = indicator_text.replace('\n', ' ')
        excerpt_text = excerpt_text.replace('\n', ' ')

        valid_entry = True
        test_indicator_text = indicator_text.lower()
        for month in month_list:
            if month in test_indicator_text:
                valid_entry = False
                print("month")
                break
        for weekday in weekday_list:
            if weekday in test_indicator_text:
                print("weekday")
                valid_entry = False
                break
        
        for regex in year_regex:
            if regex.search(test_indicator_text) != None:
                print("year")
                valid_entry = False
                break

        if valid_entry:
            clean_dict[id] = {'indicator': indicator_text, 'excerpt': excerpt_text}
        else:
            print(f"Excluded: {test_indicator_text}")

    return clean_dict

def save_progress(to_save,
                  filename: str):
    """
    Save the progress to a file using pickle.

    Args:
        to_save: The object to be saved.
        filename (str): The name of the file to save the object to. Default is 'excerpts_dict'.
    """
    try:
        progress_file = open(filename, 'wb')
        pickle.dump(to_save, progress_file)
        progress_file.close()

    except Exception as e:
        print(e)
        print("Something went wrong")



def main():

    quant_dict_file = os.path.join(SPLIT_DIR, 'quant_dict')
    quant_dict = pickle.load(open(quant_dict_file, 'rb'))
    # annotations = {k: {'type': v['type'], 'spin': '', 'macro_type': ''} for k, v in annotations.items() if 'type' in v.keys() and v['type'] == 'macro'}
    annotations = {}
    for id, ann_dict in quant_dict.items():
        if 'type' in ann_dict.keys() and ann_dict['type'] == 'macro':
            annotations[id] = {'type': 'macro'}
            if 'spin' in ann_dict.keys():
                annotations[id]['spin'] = ann_dict['spin']
            else: 
                annotations[id]['spin'] = ''
            if 'macro_type' in ann_dict.keys():
                annotations[id]['macro_type'] = ann_dict['macro_type']
            else:
                annotations[id]['macro_type'] = ''
    # annotations = {}

    # excerpts_dict = d.get_excerpts_dict(args.db)
    # print(f"Retrieved {len(excerpts_dict.keys())} excerpts")
    excerpts_file = os.path.join(SPLIT_DIR, 'quant_excerpts_dict')

    # excerpt_dict = {k: {'indicator': v[0], 'excerpt': v[1]} for k, v in excerpts_dict.items()}
    # save_progress(excerpt_dict, excerpts_file)

    excerpt_dict = pickle.load(open(excerpts_file, 'rb'))
    
    # excerpt_dict = clean_dict(excerpt_dict)
    # clean_file = os.path.join(SPLIT_DIR, 'quant_excerpts_dict_clean')
    # save_progress(excerpt_dict, excerpts_file)

    # # smaller excerpts for testing
    # random_keys = random.sample(excerpt_dict.keys(), 100)
    # excerpt_dict_small = {k: excerpt_dict[k] for k in random_keys}
    # excerpt_dict = excerpt_dict_small

    texts = [[v['indicator'], v['excerpt']] for v in excerpt_dict.values()]
    ids = [k for k in excerpt_dict.keys()]

    # for index, id in enumerate(ids):
    #     print(f"ID: {id}")
    #     print(f"Text: {texts[index]}")
    #     print()
    print(f">>> Loaded {len(texts)} excerpts")


    torch.manual_seed(42)  # Set random seed for reproducibility
    tokenizer = RobertaTokenizer\
        .from_pretrained(pretrained_model_name_or_path="roberta-base",
                         problem_type="single_label_classification")

    data = qu.TextClassificationDataset(texts=texts,
                                        tokenizer=tokenizer,
                                        ids=ids,
                                        max_length=512)

    batch_size = 8
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    task = 'type-binary'
    num_labels = len(set(label_maps[task].keys()))
    path = os.path.join(MODELS_DIR, 'type-binary_model')
    type_model = qu.QuantModel('roberta-base', num_labels).to('cuda')
    type_model = type_model.from_pretrained(path, task).to('cuda')

    num_batches = len(loader)
    freq_report = 100
    for i, batch in enumerate(loader):
        if i % freq_report == 0:
            print(f"Type Batch {i+1}/{num_batches}")
        start_index = batch['start_index'].to('cuda')
        end_index = batch['end_index'].to('cuda')
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        article_ids = batch['article_ids'].tolist()
        ann_ids = batch['ann_ids'].tolist()

        outputs = type_model(start_index,
                             end_index,
                             input_ids,
                             attention_mask)

        _, predicted = torch.max(outputs, 1)


        for i, id in enumerate(article_ids):
            global_id = str(id) + '_' + str(ann_ids[i])
            prediction = int(predicted[i].item())
            # print(f"{global_id}: {prediction}")
            if prediction == 0 and global_id not in annotations:
                annotations[global_id] = {}
                annotations[global_id]['type'] = 'macro'
                annotations[global_id]['spin'] = ''
                annotations[global_id]['macro_type'] = ''

        
    print(">>> Saving type annotations")
    save_progress(annotations, 'outputs/annotations_type_predictions')

    # for id, ann_dict in annotations.items():
    #     print(f"{id}: {ann_dict}")

    filtered_text = {k: v for k, v in excerpt_dict.items() if k in annotations.keys()}

    excerpt_dict = filtered_text
    texts = [[v['indicator'], v['excerpt']] for v in excerpt_dict.values()]
    ids = [k for k in excerpt_dict.keys()]

    data = qu.TextClassificationDataset(texts=texts,
                                        tokenizer=tokenizer,
                                        ids=ids,
                                        max_length=512)

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    task = 'spin'
    num_labels = len(set(label_maps[task].keys()))
    path = os.path.join(MODELS_DIR, 'spin_model')
    spin_model = qu.QuantModel('roberta-base', num_labels).to('cuda')
    spin_model = spin_model.from_pretrained(path, task).to('cuda')

    task = 'macro_type'
    num_labels = len(set(label_maps[task].keys()))
    path = os.path.join(MODELS_DIR, 'macro_type_model')
    macro_type_model = qu.QuantModel('roberta-base', num_labels).to('cuda')
    macro_type_model = macro_type_model.from_pretrained(path, task).to('cuda')
    

    num_batches = len(loader)
    for i, batch in enumerate(loader):
        if i % freq_report == 0:
            print(f"Spin/Macro Type Batch {i+1}/{num_batches}")

        start_index = batch['start_index'].to('cuda')
        end_index = batch['end_index'].to('cuda')
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        article_ids = batch['article_ids'].tolist()
        ann_ids = batch['ann_ids'].tolist()

        spin_outputs = spin_model(start_index,
                                  end_index,
                                  input_ids,
                                  attention_mask)

        _, spin_predicted = torch.max(spin_outputs, 1)

        macro_type_outputs = macro_type_model(start_index,
                                              end_index,
                                              input_ids,
                                              attention_mask)

        _, macro_type_predicted = torch.max(macro_type_outputs, 1)

        for i, id in enumerate(article_ids):
            global_id = str(id) + '_' + str(ann_ids[i])
            spin_prediction = int(spin_predicted[i].item())
            macro_type_prediction = int(macro_type_predicted[i].item())

            if annotations[global_id]['spin'] == '':
                annotations[global_id]['spin'] = label_maps['spin'][spin_prediction]
            if annotations[global_id]['macro_type'] == '' or annotations[global_id]['macro_type'] == '\x00':
                annotations[global_id]['macro_type'] = label_maps['macro_type'][macro_type_prediction]

    


    # clean dictionary, save to csv 
    output_dict = {k: [v['type'], v['macro_type'], v['spin'] ] for k, v in annotations.items()}
    save_progress(output_dict, 'outputs/annotations_final')
    # pd.DataFrame.from_dict(output_dict,
    #                        orient='index',
    #                        columns=['type', 'macro_type', 'spin']
    #                        ).to_csv('annotations.csv')

    # check = pickle.load(open('outputs/annotations_final', 'rb'))
    # for id, ann_dict in output_dict.items():
    #     print(f"{id}: {ann_dict}")



if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('--db', type=str, default='data/data.db', help='path to database file from top level directory')
    # parser.add_argument('--ns', type=int, default=10, help='number of article samples to load and label')
    # args = parser.parse_args()
    # main(args)
    main()

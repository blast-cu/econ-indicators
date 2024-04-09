import pickle
import pandas as pd
import re
import random

import data_utils.get_annotation_stats as gs
import data_utils.inter_annotator_agreement as iaa
from data_utils.model_utils.dataset import DB_FILENAME, qual_label_maps

OUTPUT_DIR = "potato_annotation/article_annotate/data_files"
QUANT_OUTPUT_DIR = "potato_annotation/quant_annotate/data_files"

SETTING = "distributed" # distributed, fill, agreed
global NUM_ARTICLES
NUM_ARTICLES = 300

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

def get_distributed_articles(num_articles, priority, qual_dict, predict_dict):
    random.seed(42)
    bins = gs.get_article_bins(predict_dict, qual_dict)
    bin_counts = {}
    for indicator in priority:
        bin_counts[indicator] = 0

    choices = []
    while len(choices) < num_articles:
        for indicator in priority:
            curr_ids = bins[indicator]
            id = random.choice(curr_ids)
            choices.append(id)
            curr_ids.remove(id)
            bin_counts[indicator] += 1
            if len(choices) == num_articles:
                break

    for k, v in bin_counts.items():
        print(k, v)

    return choices

def get_quant_text(quant_id, quant_excerpts_dict):

    indicator = quant_excerpts_dict[quant_id]['indicator']
    excerpt = quant_excerpts_dict[quant_id]['excerpt']

    indicator = re.escape(indicator)

    if indicator in excerpt:
        print("Indicator found in excerpt")

    try:
        match_iter = list(re.finditer(indicator, excerpt))
        match = list(match_iter)[0]
    except Exception:
        print(f"ERROR: No match found for indicator {id} in iteration {i}")
        print(">>> Match Iterator: " + str(match_iter))
        print(">>> Indicator: " + indicator)
        print(">>> Excerpt: " + excerpt)
        print('\n\n')
        exit()

    start = match.start()
    end = match.end()

    excerpt = excerpt[:start] + "<span>" + excerpt[start:end] + "</span>" + excerpt[end:]
    return excerpt

def get_choices(SETTING, NUM_ARTICLES):

    article_choices = []
    quant_choices = {}

    qual_dict = pickle.load(open("data/clean/qual_dict", "rb"))
    predict_dict = pickle.load(open("data/quant_predictions", "rb"))
    quant_excerpts_dict = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))



    if SETTING == "fill":    
        qual_dict = pickle.load(open('data/clean/qual_dict', 'rb'))
        contendors = []
        for id, anns in qual_dict.items():
            if anns['frame'] == '\x00':
                if anns['econ_rate'] == '\x00':
                    if anns['econ_change'] == '\x00':
                        contendors.append(id)
        print(len(contendors))
        article_choices = random.choices(contendors, k=25)
        articles = {}
        for id in article_choices:
            text = gs.get_text(id, db_filename=DB_FILENAME, clean=False, headline=True)
            articles[id] = text

        num_articles = NUM_ARTICLES - len(articles)
        more_articles = gs.get_no_anns(db_filename=DB_FILENAME,
                                num_samples=num_articles,
                                clean=False,
                                headline=True)
        
        articles.update(more_articles)

    # json
    # data_list = []
    # out_file = OUTPUT_DIR + '/articles.json'
    # for id, text in articles.items():
    #     temp_dict = {}
    #     temp_dict['id'] = str(id)
    #     temp_dict['text'] = text
    #     # data_list.append(temp_dict)
    #     with open(out_file, 'a+') as f:
    #         f.write('\n')
    #         json.dump(temp_dict, f)
        
    elif SETTING == "distributed":
        priority = ['jobs', 'interest', 'energy', 'retail', 'macro', 'market', 'currency', 'other', 'prices', 'housing', 'wages']
        article_choices = get_distributed_articles(NUM_ARTICLES, priority, qual_dict, predict_dict)

        

    elif SETTING == "agreed":

        qual_dict = pickle.load(open('data/clean/qual_dict', 'rb'))
        all_anns, _ = iaa.get_anns(DB_FILENAME)

        macro_count = 0
        not_macro_count = 0

        bin_choices = {}
        
        for article_id, anns in qual_dict.items():
            if anns['frame'] != '\x00':
                if len(all_anns[article_id]['frame']) == 3:
                    frame_val = anns['frame']
                    if frame_val == 'macro':
                        rate_val = anns['econ_rate']
                        change_val = anns['econ_change']
                        if len(all_anns[article_id]['econ_rate']) == 3:
                            if rate_val != '\x00':
                                if len(all_anns[article_id]['econ_change']) == 3:
                                    if change_val != '\x00':
                                        if 'macro' not in bin_choices:
                                            bin_choices['macro'] = []
                                        bin_choices['macro'].append(article_id)
                    else: 
                        if frame_val in qual_label_maps['frame']:
                            if frame_val not in bin_choices:
                                bin_choices[frame_val] = []
                            bin_choices[frame_val].append(article_id)

        bin_counts = {}
        while len(article_choices) < NUM_ARTICLES:
            for k, v in bin_choices.items():
                if k not in bin_counts:
                    bin_counts[k] = 0
                if len(v) > 0:
                    new_id = random.choice(v)
                    bin_choices[k].remove(new_id)
                    article_choices.append(new_id)
                    bin_counts[k] += 1
        for k, v in bin_counts.items():
            print(k, v)

    # get quants based on articles selected 
    for quant_id, ann in predict_dict.items():
        article_id = (quant_id.split('_')[0])
        if article_id in article_choices:
            text = get_quant_text(quant_id, quant_excerpts_dict)
            quant_tuple = (quant_id, text)
            if article_id not in quant_choices:
                quant_choices[article_id] = []
            quant_choices[article_id].append(quant_tuple)
    
    return article_choices, quant_choices

def main():

    col_names = ['id', 'text']
    article_choices, quant_choices = get_choices(SETTING, NUM_ARTICLES)

    if len(set(article_choices)) != NUM_ARTICLES:
        print(len(set(article_choices)))
        raise ValueError("Not enough articles")

    articles = {}
    for id in article_choices:
        text = gs.get_text(id, db_filename=DB_FILENAME, clean=False, headline=True)
        articles[id] = text

    for id, text in articles.items():
        headline = "<h3>" + text[0] + "</h3>"
        articles[id] = headline + text[1]

    step = 100
    for counter, idx in enumerate(range(0, NUM_ARTICLES, step)):
        # csv
        article_csv_dict = {}
        article_csv_dict['id'] = list(articles.keys())[idx:idx+step]
        article_csv_dict['text'] = list(articles.values())[idx:idx+step]

        df = pd.DataFrame(article_csv_dict)
        df.to_csv(OUTPUT_DIR + f'/articles{counter}.csv', index=False)

        quant_csv_dict = {}
        quant_csv_dict['id'] = []
        quant_csv_dict['text'] = []

        
        for article_id in article_csv_dict['id']:
            random.seed(42)
            small_choices = quant_choices[article_id]
            small_choices = random.choices(small_choices, k=5)
            for quant_id, text in small_choices:
                quant_csv_dict['id'].append(quant_id)
                quant_csv_dict['text'].append(text)

        df = pd.DataFrame(quant_csv_dict)
        df.to_csv(QUANT_OUTPUT_DIR + f'/quants{counter}.csv', index=False)



    # temp = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))
    # article_excerpt_dict = {}
    # for global_id in temp.keys():
    #     article_id, local_id = global_id.split('_')
    #     article_id = int(article_id)
    #     if article_id not in article_excerpt_dict:
    #         article_excerpt_dict[article_id] = []
    #     article_excerpt_dict[article_id].append(global_id)

    # for id in article_excerpt_dict.items():
    #     print(id)
    # # save_progress(article_excerpt_dict, 'data/clean/article_excerpt_dict')
    


if __name__ == "__main__":
    main()

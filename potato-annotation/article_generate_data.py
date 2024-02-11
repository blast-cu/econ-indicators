import pickle
import pandas as pd
import json
import random

import data_utils.get_annotation_stats as gs

OUTPUT_DIR = "potato-annotation/article_annotate/data_files"
DB_FILENAME =  "data/data.db"

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
    bins = gs.get_article_bins(predict_dict, qual_dict)
    # for k, v in bins.items():
    #     print(k, len(v))
    choices = []
    while len(choices) < num_articles:
        for indicator in priority:
            curr_ids = bins[indicator]
            id = random.choice(curr_ids)
            choices.append(id)
    return choices

def main():

    num_articles = 25
    col_names = ['id', 'text']
    
    # qual_dict = pickle.load(open('data/clean/qual_dict', 'rb'))
    # contendors = []
    # for id, anns in qual_dict.items():
    #     if anns['frame'] == '\x00':
    #         if anns['econ_rate'] == '\x00':
    #             if anns['econ_change'] == '\x00':
    #                 contendors.append(id)
    # print(len(contendors))
    # article_choices = random.choices(contendors, k=25)
    # articles = {}
    # for id in article_choices:
    #     text = gs.get_text(id, db_filename=DB_FILENAME, clean=False, headline=True)
    #     articles[id] = text

    # num_articles = num_articles - len(articles)
    # more_articles = gs.get_no_anns(db_filename=DB_FILENAME,
    #                         num_samples=num_articles,
    #                         clean=False,
    #                         headline=True)
    
    # articles.update(more_articles)

    # print(len(articles))

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
    qual_dict = pickle.load(open("data/clean/qual_dict", "rb"))
    predict_dict = pickle.load(open("data/quant_predictions", "rb"))
    priority = ['jobs', 'market', 'macro', 'prices', 'energy', 'wages', 'prices', 'interest', 'housing']
    article_choices = get_distributed_articles(num_articles, priority, qual_dict, predict_dict)

    articles = {}
    for id in article_choices:
        text = gs.get_text(id, db_filename=DB_FILENAME, clean=False, headline=True)
        articles[id] = text

    for id, text in articles.items():
        headline = "<h3>" + text[0] + "</h3>"
        articles[id] = headline + text[1]

    # csv
    csv_dict = {}
    print(articles.keys())
    csv_dict['id'] = list(articles.keys())
    csv_dict['text'] = list(articles.values())

    df = pd.DataFrame(csv_dict)
    df.to_csv(OUTPUT_DIR + '/articles.csv', index=False)



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

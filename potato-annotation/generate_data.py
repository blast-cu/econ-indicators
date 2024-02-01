import pickle
import pandas as pd
import json

import data_utils.get_annotation_stats as gs

OUTPUT_DIR = "potato-annotation/trying_again/data_files"
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


def main():

    num_articles = 10
    col_names = ['id', 'text']
    

    articles = gs.get_no_anns(db_filename=DB_FILENAME, num_samples=num_articles, clean=False)
    # print(articles)

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

    # # csv
    # csv_dict = {}
    # csv_dict['id'] = list(articles.keys())
    # csv_dict['text'] = list(articles.values())

    # df = pd.DataFrame(csv_dict)
    # df.to_csv(OUTPUT_DIR + '/articles.csv', index=False)



    temp = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))
    article_excerpt_dict = {}
    for global_id in temp.keys():
        article_id, local_id = global_id.split('_')
        article_id = int(article_id)
        if article_id not in article_excerpt_dict:
            article_excerpt_dict[article_id] = []
        article_excerpt_dict[article_id].append(global_id)

    for id in article_excerpt_dict.items():
        print(id)
    # save_progress(article_excerpt_dict, 'data/clean/article_excerpt_dict')
    


if __name__ == "__main__":
    main()

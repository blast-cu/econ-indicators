import pickle
import pandas as pd

import data_utils.get_annotation_stats as gs

OUTPUT_DIR = "potato-annotation/temp/data_files"
DB_FILENAME =  "data/data.db"

def main():

    num_articles = 2
    col_names = ['id', 'text']
    

    articles = gs.get_no_anns(db_filename=DB_FILENAME, num_samples=num_articles)
    # print(articles)

    csv_dict = {}
    csv_dict['id'] = list(articles.keys())
    csv_dict['text'] = list(articles.values())

    df = pd.DataFrame(csv_dict)
    df.to_csv(OUTPUT_DIR + '/articles.csv', index=False)

    # split_dir = 'data/clean/'
    # qual_dict = pickle.load(open(split_dir + 'qual_dict', 'rb'))
    # # quant_dict = pickle.load(open(split_dir + 'quant_dict_clean', 'rb'))
    # print(qual_dict)

if __name__ == "__main__":
    main()

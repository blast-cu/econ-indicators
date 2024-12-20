"""
Script to get json of articles and excerpts to get preds for
"""
import os
import json
import sqlite3
import pickle

from data_utils import get_annotation_stats as gs
import data_utils.model_utils.dataset as d
from models.roberta_classifier.predict_quant import clean_dict


def get_all_articles():
    """
    Get all articles from the database, add clean text and return
    """
    conn = sqlite3.connect(d.DB_FILENAME)
    c = conn.cursor()
    c.execute("SELECT id FROM article")
    article_ids = c.fetchall()
    conn.close()

    articles = {}
    for article_id in article_ids:
        article_dict = {}
        article_id = str(article_id[0])
        article_dict['text'] = gs.get_text(article_id, d.DB_FILENAME)
        article_dict['quant_list'] = []
        articles[article_id] = article_dict

    return articles


def add_excerpts(articles, excerpts):
    """
    Populate 'quant_list' key in articles with excerpt ids
    """
    for excerpt_id in excerpts:
        article_id = excerpt_id.split('_')[0]
        articles[article_id]['quant_list'].append(excerpt_id)

    return articles


def main():

    split_dir = "data/clean/"

    print("Loading all articles...")
    articles = get_all_articles()
    
    # try to open existing quant excerpt dict
    quant_excerpt_path = os.path.join(split_dir, 'quant_excerpts_dict')
    try:
        print("Loading existing quant excerpt dict...")
        excerpts = pickle.load(open(quant_excerpt_path, 'rb'))

    except FileNotFoundError:  # if not found, create it
        print("Creating new quant excerpt dict...")
        excerpts = d.get_excerpts_dict(d.DB_FILENAME)
        print(f"Retrieved {len(excerpts.keys())} excerpts")
        excerpts = {k: {'indicator': v[0], 'excerpt': v[1]} for k, v in excerpts.items()}
        with open(quant_excerpt_path, 'wb') as file:
            pickle.dump(excerpts, file)

    # add excerpts to article dict
    print("Adding excerpts to articles...")
    articles = add_excerpts(articles, excerpts)

    # export
    print("Exporting...")
    with open('data/clean/all_articles.json', 'w') as f:
        json.dump(articles, f, indent=4)

    with open('data/clean/all_excerpts.json', 'w') as f:
        json.dump(excerpts, f, indent=4)


if __name__ == "__main__":
    main()

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

# set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    fail_count = 0
    for excerpt_id in excerpts:
        article_id = excerpt_id.split('_')[0]
        if article_id in articles:
            articles[article_id]['quant_list'].append(excerpt_id)
        else:
            fail_count += 1

    return articles, fail_count


def main():

    split_dir = "data/clean/"

    logger.info("Loading all articles...")
    articles = get_all_articles()
    
    # try to open existing quant excerpt dict
    quant_excerpt_path = os.path.join(split_dir, 'quant_excerpts_dict')
        # try:
        #     logger.info("Loading existing quant excerpt dict...")
        #     excerpts = pickle.load(open(quant_excerpt_path, 'rb'))

        # except FileNotFoundError:  # if not found, create it
    logger.info("Creating new quant excerpt dict...")
    excerpts = d.get_excerpts_dict(d.DB_FILENAME, logger)
    logger.info(f"Retrieved {len(excerpts.keys())} excerpts")
    excerpts = {k: {'indicator': v[0], 'excerpt': v[1]} for k, v in excerpts.items()}
    with open(quant_excerpt_path, 'wb') as file:
        pickle.dump(excerpts, file)

    # add excerpts to article dict
    logger.info("Adding excerpts to articles...")
    articles, failed_excerpt_count = add_excerpts(articles, excerpts)
    logger.info(f"Added excerpts to articles, {failed_excerpt_count} excerpts failed bc article doesn't exist.")

    # export
    logger.info("Exporting...")
    with open('data/clean/all_articles.json', 'w') as f:
        json.dump(articles, f, indent=4)

    with open('data/clean/all_excerpts.json', 'w') as f:
        json.dump(excerpts, f, indent=4)
    
    logger.info("Exported all articles and excerpts to json files.")


if __name__ == "__main__": 
    main()

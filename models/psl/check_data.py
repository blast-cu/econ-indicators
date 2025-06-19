import pickle
import os
import json
import sqlite3
import data_utils.model_utils.dataset as d
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Set up logging
logger = logging.getLogger(__name__)

from models.psl.generate_data import write_contains_file, \
    write_has_frame_ann_file, write_preceeds_file, \
    write_target_files, write_pred_files, predict_article_annotations, \
    generate_predict_excerpts, OUT_DIR

def load_train_test_data(ann_qual_dict, ann_quant_dict, qual_dict, quant_dict):

    conn = sqlite3.connect(d.DB_FILENAME)
    c = conn.cursor()
    c.execute("SELECT id FROM article")
    articles = c.fetchall()
    print(c.description)
    conn.close()

    return articles


def main():

    # ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # check the database
    conn = sqlite3.connect(d.DB_FILENAME)
    c = conn.cursor()
    c.execute("SELECT id FROM article")
    articles = c.fetchall()
    c.execute("SELECT id FROM quantity")
    excerpts = c.fetchall()
    logger.info(f"There are {len(articles)} articles and {len(excerpts)} excerpts in the database.")

    # check train set
    split_dir = "data/clean/"
    learn_articles = json.load(open(split_dir + 'agreed_qual_dict.json', 'r'))
    learn_excerpts = json.load(open(split_dir + 'agreed_quant_dict.json', 'r'))
    logger.info(f"There are {len(learn_articles)} articles and {len(learn_excerpts)} excerpts in the learn set loaded from .json files.")

    
    # check eval set
    eval_articles = json.load(open(split_dir + 'all_articles.json', 'r'))
    eval_excerpts = json.load(open(split_dir + 'all_excerpts.json', 'r'))
    learn_ids = set([k for k in list(learn_articles.keys())])
    for quant_id in list(learn_excerpts.keys()):
        article_id = quant_id.split('_')[0]
        learn_ids.add(article_id)
    
    eval_articles = {k: v for k, v in eval_articles.items() if k not in learn_ids}
    eval_excerpts = {k: v for k, v in eval_excerpts.items() if k.split('_')[0] not in learn_ids}
    logger.info(f"There are {len(eval_articles)} articles and {len(eval_excerpts)} excerpts in the eval set loaded from .json files (after removing learn set).")


    # filter articles and excerpts that have already been annotated
    processed_articles = set()
    eval_dir = os.path.join(OUT_DIR, f'final/eval')
    with open(os.path.join(eval_dir, "ValFrame_target.txt"), 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                processed_articles.add(line.split('\t')[0])
    logger.info(f"There are {len(processed_articles)} articles that have already been annotated in the eval set.")

    # check if all articles are in the 



if (__name__ == '__main__'):
    main()

import pickle
import os
import json
import sqlite3

from models.psl.generate_data import write_contains_file, \
    write_has_frame_ann_file, write_preceeds_file, \
    write_target_files, write_pred_files, predict_article_annotations, \
    generate_predict_excerpts, OUT_DIR

import models.psl.generate_rules as gd
BEST_MODELS = {
    'frame': 'data/models/final_classifier',
    'econ_rate': 'data/models/final_classifier',
    'econ_change': 'data/models/final_classifier',
    'type': 'data/models/final_classifier',
    'macro_type': 'data/models/final_classifier',
    'spin': 'data/models/final_classifier'
}


def load_train_test_data(ann_qual_dict, ann_quant_dict, qual_dict, quant_dict):

    conn = sqlite3.connect(d.DB_FILENAME)
    c = conn.cursor()
    c.execute("SELECT id FROM article")
    articles = c.fetchall()
    print(c.description)
    conn.close()

    return articles


def main():
    """
    Main function for generating data.

    This function generates data for learn and eval by:
    1. Ensures that the output directory exists.
    2. Loads train and test articles from pickle files.
    3. Creates directories for split data.
    4. Loads learn and eval data for the split.
    5. For learn and eval:
        a. Writes contains file linking articles and excerpts.
        b. Writes target and truth files for articles and excerpts.
        c. Writes prediction files for articles and excerpts.
    """

    # ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # load train and test articles
    split_dir = "data/clean/"
    learn_articles = json.load(open(split_dir + 'agreed_qual_dict.json', 'r'))
    learn_excerpts = json.load(open(split_dir + 'agreed_quant_dict.json', 'r'))

    # make directories for split data
    learn_dir = os.path.join(OUT_DIR, f'final/learn')
    os.makedirs(learn_dir, exist_ok=True)

    eval_dir = os.path.join(OUT_DIR, f'final/eval')
    os.makedirs(eval_dir, exist_ok=True)

    eval_articles = json.load(open(split_dir + 'all_articles.json', 'r'))
    eval_excerpts = json.load(open(split_dir + 'all_excerpts.json', 'r'))
    # GENERATE LEARN DATA #
    # write contains file linking articles and excerpts
    # write_contains_file(learn_dir, learn_articles)  # contains

    # write_has_frame_ann_file(learn_dir, learn_excerpts)
    # write_has_frame_ann_file(
    #     learn_dir, learn_articles, predicate="HasFrameAnn"
    # )

    # write_preceeds_file(learn_dir, learn_articles)  # preceeds

    # # write target and truth files for validation data
    # write_target_files(learn_dir, learn_articles, gd.qual_map, truth=True)  # isVal

    # write_target_files(learn_dir, learn_excerpts, gd.quant_map, truth=True)  # isVal

    # # predictions for validation set
    # article_preds = predict_article_annotations(learn_articles, BEST_MODELS)
    # write_pred_files(learn_dir, article_preds)  # pred

    # exerpt_preds = generate_predict_excerpts(learn_excerpts, BEST_MODELS)
    # write_pred_files(learn_dir, exerpt_preds)  # pred

    # GENERATE EVAL DATA #
    # write_contains_file(eval_dir, eval_articles)  # contains

    # write_has_frame_ann_file(eval_dir, eval_excerpts)  # HasFrameAnn TODO
    # write_has_frame_ann_file(eval_dir, eval_articles, predicate="HasFrameAnn")  # TODO

    # write_preceeds_file(eval_dir, eval_articles)  # preceeds

    # write_target_files(eval_dir, eval_articles, gd.qual_map, truth=False)  # isVal
    # write_target_files(eval_dir, eval_excerpts, gd.quant_map, truth=False)  # isVal

    # article_preds = predict_article_annotations(
    #     eval_articles, BEST_MODELS
    # )
    # write_pred_files(eval_dir, article_preds)  # pred

    excerpt_preds = generate_predict_excerpts(
        eval_excerpts, BEST_MODELS
    )
    write_pred_files(eval_dir, excerpt_preds)  # pred


if (__name__ == '__main__'):
    main()

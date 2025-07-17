import sqlite3
import data_utils.model_utils.dataset as d
import os
import argparse
import pandas as pd
from data_utils.collection.article import Article
import spacy
from spacy.language import Language
from tqdm import tqdm
import re
import json
import csv
from tqdm import tqdm

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MIN_KEYWORDS = 5  # min economic keywords in an article to be added to db

"""
Script to count the number of articles per day for each publisher
Counts both economic and all articles, and saves the counts in a json file.
"""


def main(args):

    in_path = args.in_path

    # loop over all csv files
    article_count_data = {}  # article count per day, per publisher
    for publisher in tqdm(os.listdir(in_path), desc="Processing articles for publishers"):
        # read in articles.json from publisher directory
        articles_json_path = os.path.join(in_path, publisher, 'articles.json')
        if not os.path.exists(articles_json_path):
            logger.warning(f"File {articles_json_path} does not exist, skipping...")
            continue

        with open(articles_json_path, 'r') as f:
            econ_articles = json.load(f)

        # each item in the list is an article dict
        for art in econ_articles:
            date = art['date']

            # add count to json
            if date not in article_count_data:
                article_count_data[date] = {}
                article_count_data[date]['all'] = {}
                article_count_data[date]['econ'] = {}

            # check if article is economic
            if art['is_econ']:
                if publisher not in article_count_data[date]['econ']:
                    article_count_data[date]['econ'][publisher] = 0

                article_count_data[date]['econ'][publisher] += 1
            
            # add to all articles count
            if publisher not in article_count_data[date]['all']:
                article_count_data[date]['all'][publisher] = 0

            article_count_data[date]['all'][publisher] += 1

    # sort the article count data by date
    article_count_data = dict(sorted(article_count_data.items()))

    # save article counts to json file
    with open('data/article_counts.json', 'w') as f:
        json.dump(article_count_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/text')
    args = parser.parse_args()
    main(args)

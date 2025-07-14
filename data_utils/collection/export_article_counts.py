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

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MIN_KEYWORDS = 5  # min economic keywords in an article to be added to db

"""
Script to add new article data from csv (or articles.json) files to the 'article' and 'quantity' table in the database
"""


def get_unique_articles(file_path: str, pub_ids: list, logger: logging.Logger) -> list:
    """
    Read and process data from csv file
    """
    article_count = 0
    df = pd.read_csv(file_path)

    # if 'text' column is not present, return empty list
    if 'text' not in df.columns:
        logger.error(f"Column 'text' not found in {file_path} with columns {df.columns}")
        return []

    df = df.dropna(subset=['text'])  # filter out rows with no text
    for i in range(len(df)):  # iterate over rows

        row = df.iloc[i]
        id = row['article_id']  # for deduplication (can exist on multiple dates)

        if id not in pub_ids:  # skip if id already exists
            article_count += 1
            pub_ids.append(id)  # add id to list of seen ids

    return pub_ids, article_count


def main(args):

    in_path = args.in_path

    # loop over all csv files
    article_count_data = {}  # article count per day, per publisher
    new_articles_count = 0
    for publisher in os.listdir(in_path):
        # skip non-directories
        if not os.path.isdir(os.path.join(in_path, publisher)):
            continue

        logger.info("-----------------------------------------")
        logger.info(f"Processing publisher '{publisher}'...")
        pub_path = os.path.join(in_path, publisher)
        pub_articles = []

        for file in tqdm(os.listdir(pub_path)):
            if file.endswith(".csv"):
                date = file.split('.')[0]  # get date from file name
                file_path = os.path.join(pub_path, file)

                # get all articles from file which have an economic keyword
                pub_articles, article_count = get_unique_articles(file_path, pub_articles, logger)
                
                # add count to json
                if date not in article_count_data:
                    article_count_data[date] = {}

                article_count_data[date][publisher] = article_count
                new_articles_count += article_count


    # save article counts to json file
    with open('data/article_counts.json', 'w') as f:
        json.dump(article_count_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/text')
    args = parser.parse_args()
    main(args)

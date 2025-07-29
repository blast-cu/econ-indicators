import data_utils.model_utils.dataset as d
import os
import pandas as pd
from data_utils.collection.article import Article
from spacy.language import Language
from tqdm import tqdm
import json
import spacy
import csv
from data_utils.collection.add_data import get_data
import argparse
import logging
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):

    # load in urls from database
    logger.info("Loading URLs from database...")
    con = sqlite3.connect('data/data.db')
    cur = con.cursor()
    cur.execute("SELECT url FROM article WHERE url IS NOT NULL")
    urls = [row[0] for row in cur.fetchall()]
    con.close()
    print(urls[:5])  # print first 5 urls for debugging

    in_path = args.dataset
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')  # model to tokenize text into sents

    keywords = []  # load keywords from csv file
    with open("data_utils/collection/econ-keywords.csv") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            keyword = row[0]
            keyword = keyword.replace('*', '\w*')
            keywords.append(keyword)
    economic_keywords = r'(\W+|^)(' + "|".join(keywords) + r')(\W+|$)'
    economic_keywords = r'{}'.format(economic_keywords)

    for publisher in os.listdir(in_path):

        # Skip if not a directory
        if not os.path.isdir(os.path.join(in_path, publisher)):
            logger.info(f"Skipping '{publisher}' as it is not a directory.")
            continue

        pub_articles = []
        pub_urls = urls  # list of all article ids in this publisher for deduplication
        pub_path = os.path.join(in_path, publisher)

        # if articles.json already exists, skip this publisher
        if os.path.exists(os.path.join(pub_path, 'articles.json')):
            logger.info(f"Skipping publisher '{publisher}' as 'articles.json' already exists.")
            continue

        logger.info(f"Reading data from .csv files in '{pub_path}'...")
        for file in tqdm(os.listdir(pub_path)):
            if file.endswith(".csv"):
                file_path = os.path.join(pub_path, file)

                # get all articles from file which have an economic keyword
                pub_urls, articles = get_data(file_path, nlp, economic_keywords, pub_urls, logger)
                pub_articles.extend(articles)

        # save progress
        logger.info(f"Saving {len(pub_articles)} articles to 'articles.json'...\n\n")
        articles_json = [art.to_json() for art in pub_articles]
        json.dump(
            articles_json,
            open(os.path.join(pub_path, 'articles.json'), 'w+'),
            indent=4
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()
    main(args)
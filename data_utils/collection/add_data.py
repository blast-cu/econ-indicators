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

"""
Script to add 2024 data from csv files to the 'article' and 'quantity' table in the database
"""


def get_text(nlp: Language, text: str, economic_keywords: str) -> tuple:
    """
    Tokenize text and find sentences with economic keywords
    args:
        nlp: spacy model
        text: str, text to tokenize
        economic_keywords: str, regex pattern to match economic keywords
    returns:
        text: str, tokenized text
        is_econ: bool, True if text contains an economic keyword
        sentences: list of tuples, start and end indices of sentences with economic keywords
        keywords: list of str, economic keywords found in text
    """
    doc = nlp(text)  # tokenize text

    sentences = []
    keywords = []
    is_econ = False

    # if one sentence contains an economic keyword, the article is about economy
    for sentence in doc.sents:
        found_one = False
        for match in re.finditer(economic_keywords, sentence.text.lower()):
            keywords.append(match.group(2))
            found_one = True

        if found_one:
            sentences.append((sentence.start_char, sentence.end_char))
            is_econ = True

    keywords = list(set(keywords))
    return (text, is_econ, sentences, keywords)


def get_data(file_path: str, nlp: Language, econ_keywords: str, logger: logging.Logger) -> list:
    """
    Read and process data from csv file
    args:
        file_path: str, path to csv file with columns 'article_id,text,url,publisher,datetime,title,authors,publish_date,top_image,rank'
        econ_keywords: str, regex pattern to match economic keywords
    returns:
        articles: list of Article objects
    """
    articles = []
    df = pd.read_csv(file_path)

    # if 'text' column is not present, return empty list
    if 'text' not in df.columns:
        logger.error(f"Column 'text' not found in {file_path} with columns {df.columns}")
        return []

    df = df.dropna(subset=['text'])  # filter out rows with no text
    for i in range(len(df)):  # iterate over rows

        row = df.iloc[i]
        headline = row['title']
        id = None  # generate new id later
        text = row['text']

        text, is_econ, econ_sentences, keywords_used = \
            get_text(nlp, text, econ_keywords)

        source = row['publisher']
        url = row['url']
        date = row['datetime']  # must parse to datetime object

        if is_econ:  # only add if it has an econ keyword
            article = Article(
                id=id,
                headline=headline,
                text=text,
                source=source,
                url=url,
                is_econ=is_econ,
                econ_sentences=econ_sentences,
                econ_keywords=keywords_used,
                num_keywords=len(econ_sentences),
                date=date
            )
            articles.append(article)
            # print(f"Added article '{headline}' from '{source}'")
    return articles


def add_to_db(articles: list):
    """
    Add articles to the 'article' table in the database and quants to the 'quantity' table
    args:
        articles: list of Article objects
    """
    pbar = tqdm(total=len(articles), desc='saving articles and quants to database')
    conn = sqlite3.connect(d.DB_FILENAME)
    c = conn.cursor()

    # insert articles into 'article' table.
    for art in articles:
        # clean list and string values.
        keywords = ','.join(art['keywords']).strip()
        headline = art['headline'].replace("'", "''")
        text = art['text'].replace("'", "''")

        # check for duplicates in the database
        check_dup_query = f"SELECT * FROM article WHERE headline = '{headline}' AND source = '{art['source']}' AND date = '{art['date']}'"
        c.execute(check_dup_query)
        rows = c.fetchall()
        if len(rows) > 0:
            logger.info(f"Article '{headline}' from '{art['source']}' already exists in database.")
            exit()  # remove this line to continue
        
        else:  # no duplicates, insert into database
            insert_query = f'''INSERT INTO article (id, headline, source, keywords, num_keywords, text, date, url, relevance) \
                VALUES ({art['id']}, '{headline}', '{art['source']}', '{keywords}', {art['num_keywords']}, '{text}', '{art['date']}', '{art['url']}', -1)'''

            try:
                c.execute(insert_query)
            except Exception as e:
                logger.error(e)
                logger.error(insert_query)
                exit()

            # insert quants into 'quantity' table
            for quant in art['quants']:
                global_id = f"{art['id']}_{quant}"
                quant_insert_query = f'''INSERT INTO quantity (id, local_id, article_id) \
                    VALUES ('{global_id}', {quant}, {art['id']})'''

                try:
                    c.execute(quant_insert_query)
                except Exception as e:
                    logger.error(e)
                    logger.error(quant_insert_query)
                    exit()

        conn.commit()
        pbar.update(1)

    pbar.close()
    conn.close()


def main(args):

    MIN_KEYWORDS = 5  # min economic keywords in an article to be added to db
    in_path = args.in_path
    nlp = spacy.load('en_core_web_sm')  # model to tokenize text into sents

    keywords = []  # load keywords from csv file
    with open(args.econ_words) as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            keyword = row[0]
            keyword = keyword.replace('*', '\w*')
            keywords.append(keyword)
    economic_keywords = r'(\W+|^)(' + "|".join(keywords) + r')(\W+|$)'
    economic_keywords = r'{}'.format(economic_keywords)

    # loop over all csv files
    new_articles = []
    for publisher in os.listdir(in_path):
        # skip non-directories
        if not os.path.isdir(os.path.join(in_path, publisher)):
            continue
        logger.info(f"Processing publisher '{publisher}'...")
        pub_path = os.path.join(in_path, publisher)
        pub_articles = []

        # check for existing .json file of articles.
        if 'articles.json' in os.listdir(pub_path):
            logger.info(f"Reading data from 'articles_gen_headlines.json' in '{pub_path}'...")
            json_path = os.path.join(pub_path, 'articles_gen_headlines.json')
            if not os.path.exists(json_path):
                json_path = os.path.join(pub_path, 'articles.json')
                logger.info(f"Using 'articles.json' instead of 'articles_gen_headlines.json'")
            articles_json = json.load(open(json_path, 'r'))
            for article in articles_json:
                art = Article.from_json(article)
                pub_articles.append(art)
                new_articles.append(art)

        else:
            logger.info(f"Reading data from .csv files in '{pub_path}'...")
            for file in tqdm(os.listdir(pub_path)):
                if file.endswith(".csv"):
                    file_path = os.path.join(pub_path, file)

                    # get all articles from file which have an economic keyword
                    articles = get_data(file_path, nlp, economic_keywords, logger)
                    if len(articles) > 0:
                        pub_articles.extend(articles)
                        new_articles.extend(articles)

            # save progress
            logger.info(f"Saving {len(new_articles)} articles to 'articles.json'...\n\n")
            articles_json = [art.to_json() for art in pub_articles]
            json.dump(
                articles_json,
                open(os.path.join(pub_path, 'articles.json'), 'w+'),
                indent=4
            )

    # print stats
    logger.info(f"Found {len(new_articles)} articles with economic keywords")  # 492359

    # get last (max) article id in database
    conn = sqlite3.connect(d.DB_FILENAME)
    c = conn.cursor()
    c.execute("SELECT id FROM article")
    article_ids = [row[0] for row in c.fetchall()]
    last_id = max(article_ids)
    logger.info(f"Last article id in database: {last_id}")  # 96827
    conn.close()

    seen_text = set()  # keep track of duplicate texts and urls
    seen_url = set()
    idx = 0

    pbar = tqdm(total=len(new_articles), desc='cleaning articles')
    lengths = []
    clean_articles = []
    for art in new_articles:
        # 99 percentile is ~27k, no need to keep the full length of outliers that blow up the memory
        ret_text = art.text[:]
        lengths.append(len(ret_text))
        if len(ret_text) > 30000:
            pbar.update(1)
            continue

        # Removing duplicates, errors and checking for substantial economy content
        bad_headlines = set(['Access Denied', 'Wayback Machine'])
        if art.is_econ and art.num_keywords >= MIN_KEYWORDS \
                and art.text not in seen_text \
                and art.url not in seen_url \
                and art.headline not in bad_headlines \
                and 'page not found' not in art.text.lower() \
                and not (art.source == 'wsj' and (art.text.endswith('...') or art.text.endswith('Continue reading your article with\na WSJ membership'))):

            quant_span_id = 0
            quants = {}
            sentences = {}

            for sent_idx, (sent_start, sent_end) \
                    in enumerate(art.econ_sentences):

                sent = art.text[sent_start:sent_end]
                pattern = re.escape(sent)
                sentences[sent_idx] = {}
                sentences[sent_idx]['text'] = sent
                sentences[sent_idx]['span'] = (sent_start, sent_end)
                ret_text = re.sub(pattern, '<span id="{}" class="red">{}</span>'.format(sent_idx, sent.replace('\\', r'\\')), ret_text, flags=re.I)

            # find all quantities in the text
            for match in re.finditer(r"\w*(?:\s+|^)\$*[0-9,]*[0-9.]*[0-9]+%*(?:\s+|$|\.|,)\w*", art.text):
                quant = art.text[match.start():match.end()]
                pattern = re.escape(quant)
                ret_text = re.sub(pattern, '<span id="{}" class="yellow" onclick="annotateQuant(this);">{}</span>'.format(quant_span_id, quant.replace('\\', r'\\')), ret_text, flags=re.I)
                quants[quant_span_id] = {}
                quants[quant_span_id]['text'] = quant
                quants[quant_span_id]['span'] = match.span()
                quant_span_id += 1

            article_id = last_id + 1  # generate new id
            last_id = article_id
            clean_article = {
                'headline': art.headline,
                'keywords': art.econ_keywords,
                'num_keywords': art.num_keywords,
                'source': art.source,
                'url': art.url,
                'id': article_id,
                'date': art.date,
                'text': ret_text,
                'quants': quants,
                'econ_sentences': sentences
            }
            clean_articles.append(clean_article)
            idx += 1

            seen_text.add(art.text)
            seen_url.add(art.url)

        pbar.update(1)
    pbar.close()

    # add articles and quants to database
    logger.info(f"Adding {len(clean_articles)} articles to database...")  # 42604
    add_to_db(clean_articles)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/text')
    parser.add_argument('--econ_words', default='data_utils/collection/econ-keywords.csv', type=str)
    args = parser.parse_args()
    main(args)

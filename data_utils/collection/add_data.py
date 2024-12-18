import sqlite3
import data_utils.model_utils.dataset as d
import os
import argparse
import pandas as pd
from data_utils.collection.article import Article
import spacy
from tqdm import tqdm
import re
import json
import csv

"""
Script to add data from parquet files to the 'article' table in the database
"""


def get_text(nlp, text: str, economic_keywords: str) -> tuple:
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


def get_data(file_path: str, nlp, econ_keywords: str) -> list:
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
    return articles


def main(args):

    in_path = 'data/2024_dump/text'
    nlp = spacy.load('en_core_web_sm')  # model to tokenize text

    keywords = []  # load keywords from csv file
    with open(args.econ_words) as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            keyword = row[0]
            keyword = keyword.replace('*', '\w*')
            keywords.append(keyword)
    economic_keywords = '(\W+|^)(' + "|".join(keywords) + ')(\W+|$)'
    economic_keywords = r'{}'.format(economic_keywords)

    # loop over all csv files
    new_articles = []
    for publisher in os.listdir(in_path):
        pub_path = os.path.join(in_path, publisher)
        pub_articles = []
        print(f"Reading data from '{pub_path}'...")
        for file in tqdm(os.listdir(pub_path)):
            if file.endswith(".csv"):
                file_path = os.path.join(pub_path, file)

                # get all articles from file which have an economic keyword
                articles = get_data(file_path, nlp, args.econ_words)
                pub_articles.extend(articles)
                new_articles.extend(articles)

        # save progress
        print(f"Saving {len(new_articles)} articles to 'articles.json'...\n\n")
        articles_json = [art.to_json() for art in pub_articles]
        json.dump(
            articles_json,
            open(os.path.join(pub_path, 'articles.json'), 'w+'),
            indent=4
        )

    exit()

    # get last (max) article id in database
    conn = sqlite3.connect(d.DB_FILENAME)
    c = conn.cursor()
    c.execute("SELECT id FROM article")
    article_ids = [row[0] for row in c.fetchall()]
    last_id = max(article_ids)
    conn.close()

    seen_text = set()
    seen_url = set()
    idx = 0

    pbar = tqdm(total=len(articles), desc='recomputing features')
    lengths = []
    clean_articles = []
    for art in new_articles:
        '''
        if art.id == 'ab233372bab8a53dcc38cd7f4453c068':
            pbar.update(1)
            continue
        '''
        # 99 percentile is ~27k, no need to keep the full length of outliers that blow up the memory
        ret_text = art.text[:]
        lengths.append(len(ret_text))
        if len(ret_text) > 30000:
            pbar.update(1)
            continue

        '''
        doc = nlp(art.text)
        for sentence in doc.sents:
            print(sentence)
            print('-----')
        exit()
        '''

        # Removing duplicates, errors and checking for substantial economy content
        bad_headlines = set(['Access Denied', 'Wayback Machine'])
        if art.is_econ and art.num_keywords >= 5 \
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
                # print(sent)
                # print('------')
                # pattern = escape_special_chars(sent)
                pattern = re.escape(sent)
                sentences[sent_idx] = {}
                sentences[sent_idx]['text'] = sent
                sentences[sent_idx]['span'] = (sent_start, sent_end)
                # print(pattern)
                # print('##########')
                ret_text = re.sub(pattern, '<span id="{}" class="red">{}</span>'.format(sent_idx, sent.replace('\\', r'\\')), ret_text, flags=re.I)

            for match in re.finditer(r"\w*(?:\s+|^)\$*[0-9,]*[0-9.]*[0-9]+%*(?:\s+|$|\.|,)\w*", art.text):
                quant = art.text[match.start():match.end()]
                # print(quant)
                # print('------')
                # check if it is a date/time
                # pattern = escape_special_chars(quant)
                pattern = re.escape(quant)
                ret_text = re.sub(pattern, '<span id="{}" class="yellow" onclick="annotateQuant(this);">{}</span>'.format(quant_span_id, quant.replace('\\', r'\\')), ret_text, flags=re.I)
                quants[quant_span_id] = {}
                quants[quant_span_id]['text'] = quant
                quants[quant_span_id]['span'] = match.span()
                quant_span_id += 1

            article_id = last_id + 1  # generate new one
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


    # # print example article
    # conn = sqlite3.connect(d.DB_FILENAME)
    # c = conn.cursor()
    # c.execute("SELECT * FROM article")
    # articles = c.fetchall()
    # print(articles[0])
    # print(c.description)
    # # id, headline, source, keywords, num_keywords, relevance, text, distance, date, url, cluster_id
    # conn.close()

        # check relevance


        # extract all indicators

        # add to quant table in database
        # conn = sqlite3.connect(d.DB_FILENAME)
        # c = conn.cursor()
        # c.execute("SELECT id FROM article")
        # article_ids = c.fetchall()
        # conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='data/clean/2024_dump/text')
    parser.add_argument('--econ_words', default='data_utils/collection/econ-keywords.csv', type=str)
    args = parser.parse_args()
    main(args)

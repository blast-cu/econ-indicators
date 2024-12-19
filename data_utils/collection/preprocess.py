import argparse
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from data_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from data_utils.collection.article import Article
from tqdm import tqdm
#from sutime import SUTime
from dateparser.search import search_dates
import spacy
from scipy import stats
import pandas as pd


def escape_special_chars(pattern):
    pattern = pattern.replace('\\', '\\\\')
    pattern = pattern.replace('(', '\(')
    pattern = pattern.replace(')', '\)')
    pattern = pattern.replace('[', '\[')
    pattern = pattern.replace(']', '\]')
    pattern = pattern.replace('*', '\*')
    pattern = pattern.replace('+', '\+')
    pattern = pattern.replace('.', '\.')
    pattern = pattern.replace('?', '\?')
    pattern = pattern.replace('$', '\$')
    pattern = pattern.replace('^', '\^')
    pattern = pattern.replace('|', '\|')
    return pattern

def main(args):
    if args.parse_data:

        if not args.mysql and args.datadir is not None:
            articles = parse_data(args.datadir, args.econ_words)
        elif args.mysql:
            articles = parse_data_mysql(args.host, args.user, args.password, args.database, args.econ_words)
        else:
            print('One of the following options is mandatory:')
            print('\t--datadir')
            print('\t--mysql --host [host] --user [user] --password [password] --database [database]')
            exit(-1)

        with open(os.path.join(args.outdir, 'preprocessed_data.json'), 'w') as fp:
            dictionary = articles_to_json(articles)
            json.dump(dictionary, fp)

    elif args.load_json:
        dictionary = json.load(open(os.path.join(args.outdir, 'preprocessed_data.json')))
        articles = [Article.from_json(dictionary[id]) for id in dictionary]
    else:
        print('One of two options --[parse_data|load_json] need to be passed')
        exit(-1)

    # extract TF-IDF representation and create files for annotation-gui
    # only if they have been marked as having to do with the economy
    corpus = []
    bad_headlines = set(['Access Denied', 'Wayback Machine'])

    if args.recompute_features:
        seen_text = set(); seen_url = set()
        pos2features = {}
        idx = 0

        #nlp = spacy.load('en_core_web_sm')
        pbar = tqdm(total=len(articles), desc='recomputing features')
        max_len = 0
        lengths = []
        for art in articles:
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
            if art.is_econ and art.num_keywords >= 5 and not art.text in seen_text and not art.url in seen_url and\
               dictionary[art.id]['headline'] not in bad_headlines and 'page not found' not in art.text.lower() and\
               not (art.source == 'wsj' and (art.text.endswith('...') or art.text.endswith('Continue reading your article with\na WSJ membership'))):
                quant_span_id = 0
                quants = {}; sentences = {}

                for sent_idx, (sent_start, sent_end) in enumerate(art.econ_sentences):
                    sent = art.text[sent_start:sent_end]
                    #print(sent)
                    #print('------')
                    #pattern = escape_special_chars(sent)
                    pattern = re.escape(sent)
                    sentences[sent_idx] = {}
                    sentences[sent_idx]['text'] = sent
                    sentences[sent_idx]['span'] = (sent_start, sent_end)
                    #print(pattern)
                    #print('##########')
                    ret_text = re.sub(pattern, '<span id="{}" class="red">{}</span>'.format(sent_idx, sent.replace('\\', r'\\')), ret_text, flags=re.I)

                for match in re.finditer(r"\w*(?:\s+|^)\$*[0-9,]*[0-9.]*[0-9]+%*(?:\s+|$|\.|,)\w*", art.text):
                    quant = art.text[match.start():match.end()]
                    #print(quant)
                    #print('------')
                    # check if it is a date/time
                    #pattern = escape_special_chars(quant)
                    pattern = re.escape(quant)
                    ret_text = re.sub(pattern, '<span id="{}" class="yellow" onclick="annotateQuant(this);">{}</span>'.format(quant_span_id, quant.replace('\\', r'\\')), ret_text, flags=re.I)
                    quants[quant_span_id] = {}
                    quants[quant_span_id]['text'] = quant
                    quants[quant_span_id]['span'] = match.span()
                    quant_span_id += 1

                pos2features[idx] = {
                    'headline': art.headline,
                    'keywords': art.econ_keywords,
                    'num_keywords': art.num_keywords,
                    'source': art.source,
                    'url': art.url,
                    'id': art.id,
                    'date': art.date,
                    'text': ret_text,
                    'quants': quants,
                    'econ_sentences': sentences
                }

                corpus.append(ret_text)
                idx += 1

                seen_text.add(art.text)
                seen_url.add(art.url)

            pbar.update(1)
        pbar.close()

        df = pd.DataFrame({ 'lengths': lengths })

        print(stats.describe(lengths))
        print(df.lengths.describe())
        print(np.percentile(lengths, 90))
        print(np.percentile(lengths, 95))
        print(np.percentile(lengths, 99))
        #exit()
        with open(os.path.join(args.outdir, 'article_features.json'), 'w') as fp:
            json.dump(pos2features, fp)
    else:
        pos2features = json.load(open(os.path.join(args.outdir, 'article_features.json')))
        for i in range(0, len(pos2features)):
            corpus.append(pos2features[str(i)]['text'])

    if args.tfidf:
        print("vectorizing...")
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                     min_df=2, stop_words='english',
                                     use_idf=True)
        X = vectorizer.fit_transform(corpus).toarray()

        # Save numpy matrix for tfidf rep, text and headlines
        np.save(os.path.join(args.outdir, 'article_embed.npy'), X)
        #np.save(os.path.join(args.outdir, 'article_text.npy'), np.array(corpus))
        print('done.')
    else:
        # create a similarity matrix NxN where sim(i,j) = max similarity between any two paragraphs in i and j
        # use sentence bert to represent paragraphs
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--mysql', action='store_true', default=False)
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--user', type=str, default='root')
    parser.add_argument('--database', type=str, default='newsobs')
    parser.add_argument('--password', type=str, default=None)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--parse_data', action='store_true', default=False)
    parser.add_argument('--recompute_features', action='store_true', default=False)
    parser.add_argument('--load_json', action='store_true', default=False)
    parser.add_argument('--tfidf', action='store_true', default=False)
    parser.add_argument('--econ_words', default='econ-keywords.csv', type=str)
    args = parser.parse_args()
    main(args)

import csv
import os
import re
from tqdm import tqdm
from article import Article
import spacy
from mysql.connector import connect, Error

def parse_data(data_dir, econ_words):
    nlp = spacy.load('en_core_web_sm')
    metadata = os.path.join(data_dir, 'metadata.csv')
    articles = []
    with open(metadata) as fp:
        reader = csv.reader(fp)
        rows = [l for l in reader]
        indices = {}
        pbar = tqdm(total=len(rows), desc='preprocessing data [parse_data]')
        for i, row in enumerate(rows):
            if i == 0:
                # Get headers
                for j, header in enumerate(row):
                    indices[header] = j
            else:
                # Get the rest of the data
                headline = row[indices['title']]
                id = row[0]
                text_path = os.path.join(data_dir, 'readability', '{}.txt'.format(id))
                fp = open(text_path)
                text = fp.read()
                text, is_econ, econ_sentences, econ_keywords =\
                    parse_text(nlp, text, econ_words)

                url = row[indices['url']]
                date = row[indices['publish_date']]
                source = url.split('/')[2]
                article = Article(
                            id=id,
                            headline=headline,
                            text=text,
                            source=source,
                            url=url,
                            is_econ=is_econ,
                            econ_sentences=econ_sentences,
                            econ_keywords=econ_keywords,
                            num_keywords=len(econ_sentences),
                            date=date)
                articles.append(article)
            pbar.update(1)
        pbar.close()

    return articles

def parse_data_mysql(host, user, password, database, econ_words):
    nlp = spacy.load('en_core_web_sm')
    articles = []

    connection = connect(
        host=host,
        user=user,
        password=password,
        database=database)
    cursor = connection.cursor(buffered=True)

    # Iterate over articles
    cursor.execute("select count(*) from articles")
    for count in cursor:
        total = count[0]

    pbar = tqdm(total=total, desc='preprocessing data [parse_data_mysql]')

    cursor.execute("select * from articles")
    for art in cursor:
        (headline, html_checksum, source, href, _, _, scrape_time, scrape_id, fp_id) = art
        new_cursor = connection.cursor()
        query = "select trafilatura from content where html_checksum = '{0}'".format(html_checksum)
        new_cursor.execute(query)
        for text in new_cursor:
            text = text[0]
            text, is_econ, econ_sentences, econ_keywords = parse_text(nlp, text, econ_words)

            article = Article(
                id=scrape_id,
                headline=headline,
                text=text,
                source=source,
                url=href,
                is_econ=is_econ,
                econ_sentences=econ_sentences,
                econ_keywords=econ_keywords,
                num_keywords=len(econ_sentences),
                date=str(scrape_time))
            articles.append(article)
        pbar.update(1)
    pbar.close()

    return articles

def parse_text(nlp, text, econ_words):
    # read csv
    keywords = []
    with open(econ_words) as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            keyword = row[0]
            keyword = keyword.replace('*', '\w*')
            keywords.append(keyword)
    economic_keywords = '(\W+|^)(' + "|".join(keywords) + ')(\W+|$)'
    economic_keywords = r'{}'.format(economic_keywords)

    #economic_keywords = r'(\W+|^)(econo\w*|unemploy\w*|job\w*|earning\w*|inflation|stock|income|revenue|poverty|price|cost|housing|nasdaq|dow jones)(\W+|$)'
    doc = nlp(text)

    sentences = []; keywords = []
    is_econ = False

    for sentence in doc.sents:
        found_one = False
        for match in re.finditer(economic_keywords, sentence.text.lower()):
            keywords.append(match.group(2))
            found_one = True

        if found_one:
            sentences.append((sentence.start_char, sentence.end_char))
            is_econ = True

    keywords = list(set(keywords))
    #print(keywords)
    #print(sentences)
    #exit()
    return (text, is_econ, sentences, keywords)

def articles_to_json(articles):
    ret = {}
    for art in articles:
        ret[art.id] = art.to_json()
    return ret


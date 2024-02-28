import data_utils.get_annotation_stats as gs
from data_utils.dataset import quant_label_maps

import sqlite3
import pickle
import pandas as pd
import numpy as np

DB_FILENAME = 'data/data.db'

def get_quant_counts(sites, excerpts_dict):

    quant_counts = []
    conn = sqlite3.connect('data/data.db')
    publisher_count = {}
    for global_id in list(excerpts_dict.keys()):
        article_id = global_id.split('_')[0]
        
        c = conn.cursor()
        c.execute(f'SELECT source FROM article WHERE id = {article_id}')

        source = c.fetchone()[0]
        if source not in publisher_count.keys():
            publisher_count[source] = 0
        publisher_count[source] += 1

    conn.close()

    total_quants = 0
    for site in sites:
        count = int(publisher_count[site])
        quant_counts.append(count)
        total_quants += count
    
    return quant_counts, total_quants

def get_quant_ann_counts(sites):

    quant_ann_counts = []
    annotation_count = {}
    conn = sqlite3.connect('data/data.db')
    c = conn.cursor()
    c.execute(f'SELECT quantity_id FROM quantityann')
    annotations = c.fetchall()
    annotations = [a[0] for a in annotations]

    ann_count = {}
    for a in annotations:
        if a not in ann_count.keys():
            ann_count[a] = 0
        ann_count[a] += 1

    cross_validated = [a for a in ann_count.keys() if ann_count[a] > 1]
    cross_validated_count = len(cross_validated)


    for id in annotations:
        c = conn.cursor()
        article_id = id.split('_')[0]
        c.execute(f'SELECT source FROM article WHERE id = {article_id}')
        source = c.fetchone()[0]
        if source != 'bbc':
            if source not in annotation_count.keys():
                annotation_count[source] = 0
            annotation_count[source] += 1
    
    
    total_quant_anns = 0
    for p in sites:
        quant_ann_counts.append(int(annotation_count[p]))
        total_quant_anns += annotation_count[p]


    conn.close()

    return quant_ann_counts, total_quant_anns, cross_validated_count

def get_qual_ann_counts(sites):

    qual_ann_counts = []
    annotation_count = {}
    conn = sqlite3.connect('data/data.db')
    c = conn.cursor()
    c.execute(f'SELECT article_id FROM articleann')
    annotations = c.fetchall()
    annotations = [a[0] for a in annotations]

    ann_count = {}
    for a in annotations:
        if a not in ann_count.keys():
            ann_count[a] = 0
        ann_count[a] += 1

    cross_validated = [a for a in ann_count.keys() if ann_count[a] > 1]
    cross_validated_count = len(cross_validated)


    for article_id in annotations:
        c = conn.cursor()
        c.execute(f'SELECT source FROM article WHERE id = {article_id}')
        source = c.fetchone()[0]
        if source != 'bbc':
            if source not in annotation_count.keys():
                annotation_count[source] = 0
            annotation_count[source] += 1
    
    
    total_qual_anns = 0
    for p in sites:
        qual_ann_counts.append(int(annotation_count[p]))
        total_qual_anns += annotation_count[p]


    conn.close()

    return qual_ann_counts, total_qual_anns, cross_validated_count


def main():

    table = {}
    table['site'] = ['nytimes', 'wsj', 'washingtonpost', 'foxnews', 'breitbart', 'huffpost']
    # table['econ_arts'] = []
    table['quants'] = []
    table['art_anns'] = []
    table['quant_anns'] = []

    excerpts_dict = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))
    quant_counts, total_quants = get_quant_counts(table['site'], excerpts_dict)
    quant_ann_counts, total_quant_anns, quant_cross_validated_count = \
        get_quant_ann_counts(table['site'])
    
    qual_ann_counts, total_qual_anns, qual_cross_validated_count = \
        get_qual_ann_counts(table['site'])
    
    table['quants'] = quant_counts
    table['quant_anns'] = quant_ann_counts
    table['art_anns'] = qual_ann_counts

    table['site'].append('total')
    table['quants'].append(total_quants)
    table['quant_anns'].append(total_quant_anns)
    table['art_anns'].append(total_qual_anns)

    table['site'].append('cross-validated')
    table['quants'].append(np.nan)
    table['quant_anns'].append(quant_cross_validated_count)
    table['art_anns'].append(qual_cross_validated_count)

    pd.DataFrame(table).to_csv('data_utils/table_generators/results/annotation_stats.csv', index=False)

    # conn = sqlite3.connect('data/data.db')
    # c = conn.cursor()
    # c.execute(f'SELECT * FROM article WHERE source = "nytimes"')
    # articles = c.fetchall()
    # # articles = [a[0] for a in articles]
    # conn.close()
    # print(len(articles))

if __name__ == "__main__":
    main()
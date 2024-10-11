import data_utils.inter_annotator_agreement as iaa
import argparse
import sqlite3
from collections import Counter
import random
import csv
import re
import matplotlib.pyplot as plt
import numpy as np

def get_article_bins(predict_dict: dict, qual_dict: dict):
    article_preds = {} # {article_id: [preds]}

    for quant_id, preds in predict_dict.items():

        article_id = quant_id.split('_')[0]
        if article_id not in article_preds:
            article_preds[article_id] = []

        pred = preds[1]
        article_preds[article_id].append(pred)

    article_labels = {} # {article_id: [labels]}
    # TODO: get article labels from quant preds, aggregate
    for article_id, preds in article_preds.items():
        if len(preds) >= 2: # at least two quants
            c = Counter(preds).most_common()
            if c[0][1] >= 2: # at least two quants agree, doesn't account for ties
                article_labels[article_id] = c[0][0]

    bins = {} # {frame label: [article_ids]}

    for article_id, label in article_labels.items():
        if article_id not in qual_dict:
            try:
                if label not in bins:
                    bins[label] = []
                bins[label].append(article_id)
            except KeyError as e:
                print(f'Key error: {e} not in article frames')
                continue
    print(bins.keys())
    return bins

def get_site(article_id, db_filename):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()
    query = "SELECT source FROM article WHERE id is " + str(article_id) + ";"
    site = cur.execute(query).fetchone()
    con.close()
    return site[0]


def get_qual_dict(db_filename: str):
    """
    Takes file location of database
    Return dictionary where keys are article_id
        and values are dictionary where keys are frame component
        and values are list of tuples (annotator_id, annotation value)
    """
    con = sqlite3.connect(db_filename)
    cur = con.cursor()
    ann = {}

    for comp in ['frame', 'econ_rate', 'econ_change']:
        query = 'SELECT article_id, user_id, ' + comp \
                + ' FROM articleann ' \
                + 'WHERE ' + comp + ' is NOT NULL and ' + comp + ' != "None";'
        res = cur.execute(query)
        iaa.retrieve_anns(ann, res, comp)

    con.close()

    clean_ann = {}
    for a in ann.keys():
        if get_site(a, db_filename) != 'bbc':
            clean_ann[a] = ann[a]

    return clean_ann

def get_quant_dict(db_filename: str): 
    """
    Takes file location of database
    Return dictionary where keys are article_id
        and values are dictionary where keys are frame component
        and values are list of tuples (annotator_id, annotation value)
    """
    con = sqlite3.connect(db_filename)
    cur = con.cursor()
    ann = {}

    query = 'SELECT quantity_id, user_id, type, macro_type, industry_type, \
                gov_type, expenditure_type, revenue_type, spin \
                FROM quantityann;'

    res = cur.execute(query)
    iaa.retrieve_quant_anns(ann, res)
    con.close()

    clean_ann = {}
    for a in ann.keys():
        a_id = int(a.split('_')[0])
        if get_site(a_id, db_filename) != 'bbc':
            clean_ann[a] = ann[a]

    return clean_ann


def get_agreed_anns(ann_dict: dict, label_maps: dict, type_filter: list = []):
    """
    Takes a nested dictionary of annotations (list) and returns
    nested dictionary of final annotations (str) wrt full agreement
    """

    agreed_dict = {}

    for id in ann_dict.keys(): 
        curr_ent = ann_dict[id]
        # TODO: for quant anns, check type before subtypes
        # print(curr_ent)
        for type in curr_ent.keys():
            if type in label_maps:
                curr_t = curr_ent[type]
                result = '\0'

                if len(curr_t) >= 2:  # 2 or more annotations

                    anns = [a[1] for a in curr_t if a[1] in label_maps[type]]
                    if len(anns) >= 2:
                        c = Counter(anns).most_common()

                        # check for tie (first result count matches second)
                        if len(c) == 1 or c[0][1] != c[1][1]:

                            result = c[0][0]

                if id not in agreed_dict:
                    agreed_dict[id] = {}

                agreed_dict[id][type] = result

    if type_filter != []:
        filtered_dict = {}
        for id in agreed_dict.keys():
            curr_ent = agreed_dict[id]
            if curr_ent['type'] in type_filter:
                filtered_dict[id] = curr_ent

        agreed_dict = filtered_dict

    return agreed_dict

def get_noisy_anns(ann_dict: dict, label_maps: dict):

    noisy_dict = {}

    for id in ann_dict.keys():
        curr_ent = ann_dict[id]
        for type in curr_ent.keys():
            if type in label_maps:
                curr_t = curr_ent[type]
                result = []
                if len(curr_t) >= 2:  # 2 or more annotations
                    anns = [a[1] for a in curr_t if a[1] in label_maps[type]]
                    c = Counter(anns).most_common()

                    # check for tie (first result count matches second)-> no consensus
                    if len(c) > 1 and c[0][1] == c[1][1]:
                        for ann in c:
                            result.append(ann[0])
        
                elif len(curr_t) == 1 and curr_t[0][1] in label_maps[type]:
                    result.append(curr_t[0][1])

                if id not in noisy_dict:
                    noisy_dict[id] = {}

                if result != []:
                    noisy_dict[id][type] = result
                else:
                    noisy_dict[id][type] = '\0'
    
    return noisy_dict
    

def get_best_noisy_anns(ann_dict: dict, label_maps: dict, db_filename: str, quant: bool = False):

    ann_dict, quantity2ann = iaa.get_anns(db_filename)
    if quant:
        ann_dict = quantity2ann

    user_ann_disagreement = {}
    for ann_name in label_maps.keys():
        user_ann_disagreement[ann_name] = {}
        user_disagreements = {}
        user_total_anns = {}
        iaa.measure_percentage_agreement(ann_dict, ann_name, user_disagreements, user_total_anns)
        for user in sorted(user_disagreements.keys()):
            percent_disagree = round(user_disagreements[user]/user_total_anns[user], 2)
            user_ann_disagreement[ann_name][user] = percent_disagree

    noisy_dict = {}

    for id in ann_dict.keys():
        curr_ent = ann_dict[id]
        for type in curr_ent.keys():
            if type in label_maps:
                curr_t = curr_ent[type]
                result = '\0'

                if len(curr_t) >= 2:  # 2 or more annotations
                    anns = [a[1] for a in curr_t if a[1] in label_maps[type]]
                    c = Counter(anns).most_common()

                    # check for tie (first result count matches second)-> no consensus
                    if len(c) > 1 and c[0][1] == c[1][1]:
                        min_disagreement = 1
                        best_ann = curr_t
                        for ann in curr_t:
                            if ann[0] not in user_ann_disagreement[type]:
                                print(f"User {ann[0]} not in user_ann_disagreement")
                                print(ann[0])
                                print(user_ann_disagreement[type])
                                # exit()
                                continue
                            curr_disagreement = user_ann_disagreement[type][ann[0]]
                            if curr_disagreement < min_disagreement:
                                min_disagreement = curr_disagreement
                                best_ann = ann
                        result = best_ann[1]
        
                elif len(curr_t) == 1 and curr_t[0][1] in label_maps[type]:
                    result = curr_t[0][1]

                if id not in noisy_dict:
                    noisy_dict[id] = {}
                noisy_dict[id][type] = result
                # if '(62896882543' in result:
                #     print(result)
                #     exit()
    
    return noisy_dict
        


def print_agreed_anns_counts(ann_dict: dict):
    """
    Takes dictionary output of get_agreed_anns func and prints
        agreed annotation stats

    Returns dictionary where keys are annotation components and
        values are dictionary where keys are annotation component
        assignment and values are number of articles with given
        value assigned to given annotation component :D
    """

    labels = {}

    for id in ann_dict.keys():  # for each article entry
        curr_article = ann_dict[id]

        for type in curr_article.keys():  # for each ann component
            curr_t = curr_article[type]
            if type not in labels:
                labels[type] = {}

            if curr_t != '\0': 
                if curr_t in labels[type]:
                    labels[type][curr_t] += 1
                else:
                    labels[type][curr_t] = 1
    label_counts = {}
    for type in labels.keys():
        count = sum(labels[type].values())
        print(type + " - count: " + str(count))
        print(str(labels[type]) + '\n')
        label_counts[type] = count
    
    return labels, label_counts


def export_quants_to_csv(ann: dict, filename: str):
    """
    Takes as input the labels dictionary output by print_agreed_anns_counts
    and formats results into csv
    """

    field_names = ['Type', 'Subtype', 'Nested Subtype', 'Annotations']
    row_list = []
    filename = "data_utils/data_summary/" + filename

    def add_row(type, subtype, n_subtype, ann_ct):
        row = {}
        row['Type'] = type
        row['Subtype'] = subtype
        row['Nested Subtype'] = n_subtype
        row['Annotations'] = ann_ct
        row_list.append(row)

    macro_dict = ann['macro_type']
    for mt in macro_dict.keys():
        add_row('Macro', mt, 'NA', str(macro_dict[mt]))

    add_row('Macro', 'Total', 'NA', str(ann['type']['macro']))

    expen_dict = ann['expenditure_type']
    for et in expen_dict.keys():
        add_row('Government', 'Expenditures', et, str(expen_dict[et]))

    rev_dict = ann['revenue_type']
    for rt in rev_dict.keys():
        add_row('Government', 'Revenues', rt, str(rev_dict[rt]))

    add_row('Government', 'Debt and Deficit', 'NA', str(ann['gov_type']['deficit']))
    add_row('Government', 'Total', rt, ann['type']['government'])

    ind_dict = ann['industry_type']
    for it in ind_dict.keys():
        add_row('Industry', it, 'NA', str(ind_dict[it]))

    add_row('Industry', 'Total', 'NA', str(ann['type']['industry']))
    add_row('Business', 'NA', 'NA', str(ann['type']['business']))
    add_row('Personal', 'NA', 'NA', str(ann['type']['personal']))
    add_row('Other', 'NA', 'NA', str(ann['type']['other']))
    add_row('Total', 'NA', 'NA', str(sum(ann['type'].values())))

    with open(filename, 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(row_list)


def extract_strings(dirty_str: str):
    clean = re.sub('<[^>]+>', '', dirty_str)
    return clean


def print_article_examples(comp: str, ann_dict: dict, filename: str, db_filename: str):
    """
    Takes annotation component as string, dictionary of annotations in which
        key is article_id and value is dictionary where key is frame component
        and value is annotation value, the desired output filename and the file
        location of the db

    Outputs file of article text and associated annotation component into
        data_summary directory
    """
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    filename = "data_utils/data_summary/" + filename
    with open(filename, 'w+') as f:
        f.write("###\n")
        for article_id in ann_dict.keys():
            label = ann_dict[article_id][comp]
            if label != '\0':
                query = 'SELECT text\
                    FROM article ' \
                    + 'WHERE id is ' + str(article_id) + ';'
                article_txt = cur.execute(query).fetchone()
                # print query result
                clean_text = extract_strings(article_txt[0])
                
                f.write('article: "' + clean_text + '"' + '\n')
                f.write(comp + " of article is: " + label + '\n')
                f.write("###\n")

    con.close()

def get_text(article_id: int, db_filename: str, clean: bool = True, headline: bool = False):
    """
    Takes article_id and db filename
    Returns cleaned text of article as string
    """
    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    text = ''

    query = 'SELECT text\
        FROM article ' \
        + 'WHERE id is ' + str(article_id) + ';'
    article_txt = cur.execute(query).fetchone()

    text = article_txt[0]
    if clean:
        text = extract_strings(article_txt[0])
    if headline:
        query = "SELECT headline FROM article WHERE id is " + str(article_id) + ";"
        headline = cur.execute(query).fetchone()
        text = [headline[0], text]
        
    con.close()

    return text

def get_no_anns(db_filename: str, num_samples: int = None, clean: bool = True, headline: bool = False):
    """
    Retrieves a dictionary of articles with no annotations from a SQLite database.

    Args:
        db_filename (str): The path to the SQLite database file.
        num_samples (int, optional): The number of samples to retrieve. If None, retrieves all samples.

    Returns:
        dict: A dictionary where the keys are article IDs and the values are the article text.
    """

    articles = {}  # keys: article_id, values: article text

    con = sqlite3.connect(db_filename)
    cur = con.cursor()

    query = "SELECT id, text FROM article WHERE id NOT IN (SELECT article_id FROM articleann);"
    # query = 'SELECT * FROM article;' # 96828
    # query = 'SELECT * FROM articleann;' # 1189

    ret = cur.execute(query)
    samples = ret.fetchall()

    if num_samples is not None:
        random.seed(42)
        samples = random.sample(samples, num_samples)

    for id, text in samples:
        if clean:
            text = extract_strings(text)
        if headline:
            query = "SELECT headline FROM article WHERE id is " + str(id) + ";"
            headline = cur.execute(query).fetchone()
            articles[id] = (headline[0], text)
        else: 
            articles[id] = text

    con.close()
    return articles

def get_excerpts(db_filename: str):
    con = sqlite3.connect(db_filename)
    cur = con.cursor()
    ann = {}

    query = 'SELECT id FROM quantity'
    res = cur.execute(query)
    ids = res.fetchall()
    ids = [id[0] for id in ids]
    con.close()

    return ids

def gpt_cost(db_filename: str,  ann_dict: dict, price_per_k):
    """
    Takes file location of database; dictionary of annotations in which key is
        article_id and value is dictionary where key is frame component and value
        is annotation value; and price of input to gpt per thousand tokens

    Returns estimated cost of classifying each annotation component (0-shot, 1
        label at a time)
    """

    word_count = 0
    for article_id in ann_dict.keys():

        label_dict = ann_dict[article_id]
        clean_text = get_text(article_id, db_filename)

        clean_list = re.split(r'[\n\t\f\v\r]+', clean_text)  # split on w space
        article_word_count = len(clean_list)

        num_labels = 0  # count labels with agreed value
        for label in label_dict.keys():
            if label_dict[label] != '\0':
                num_labels += 1

        word_count += (num_labels * article_word_count)

    token_count = (100 * word_count) * 75  # calc via open ai
    price = price_per_k * (token_count / 1000)
    return price


def get_all_text(db_filename: str, clean: bool = True):
    """
    Retrieves the text data from the specified SQLite database file.

    Parameters:
    - db_filename (str): The path to the SQLite database file.
    - clean (bool): Flag indicating whether to clean the text data. Default is True.

    Returns:
    - text (list): The list of text data retrieved from the database.
    """
    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()

    query = 'SELECT text FROM article;'
    res = cur.execute(query)
    text = [t[0] for t in res.fetchall()]

    if clean:
        text = [extract_strings(t) for t in text]

    conn.close()
    return text


def visualize_anns(ann_dict: dict):

    bins = len(ann_dict.keys())
    X_axis = np.arange(bins)
    tot_articles = [199996]*bins

    bin_names = ['type', 'econ_conditions', 'econ_change']


    plt.bar(X_axis, tot_articles, 0.4, color='#FED766')
    plt.bar(X_axis, ann_dict.values(), 0.4, label='annotated', color='#a4649c') 

    plt.xticks(X_axis, bin_names)
    plt.yscale('log')
    plt.legend()

    plt.ylabel('Number of Articles (log scale)')
    plt.xlabel('Annotation Component')
    plt.savefig(fname='plot.pdf')
    
    plt.show()


def main(args):

    # qual_ann = get_qual_dict(args.db)
    # agreed_qual_ann = get_agreed_anns(qual_ann)
    # print(gpt_cost(args.db, agreed_qual_ann, 0.0015))
    # qual_label_counts, label_counts = print_agreed_anns_counts(agreed_qual_ann)
    # visualize_anns(label_counts)
    # print_article_examples('econ_change', agreed_qual_ann, 'econ_change.txt', args.db)


    # quant_ann = get_quant_dict(args.db)
    # agreed_quant_ann = get_agreed_anns(quant_ann, quant_label_maps)
    # quant_labels = print_agreed_anns_counts(agreed_quant_ann)
    # export_quants_to_csv(quant_labels, 'annotation_count.csv')

    # con = sqlite3.connect(args.db)
    # cur = con.cursor()

    # # query = 'SELECT name FROM sqlite_master WHERE type="table";'
    # # query = 'SELECT * FROM articleann;'
    # query = "SELECT COUNT(DISTINCT articleann_id) FROM topics;"
    # article_txt = cur.execute(query)
    # # for col in article_txt:
    # #     print(col[0])
    # # description = article_txt.description
    # # for d in description:
    # #     print(d[0])
    # topics = article_txt.fetchall()
    # for t in topics:
    #     print(t)

    # get_no_anns(args.db)
    # db_filename = args.db
    # qual_ann = get_qual_dict(db_filename)
    # noisy = get_noisy_anns(qual_ann)
    # for n in noisy.items():
    #     print(n)
    print("temp")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)
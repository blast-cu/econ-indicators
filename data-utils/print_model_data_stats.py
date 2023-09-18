from inter_annotator_agreement import retrieve_anns, retrieve_quant_anns
import argparse
import sqlite3
from collections import Counter


def format_dict(d): 
    for k in d.keys():
        print(str(k) + ": " + str(d[k]))

def get_qual_tags(qual_dict: dict):
    """takes a nested dictionary of annotations (list) and returns
    nested dictionary of final annotations (str) wrt full agreement """

    for id in qual_dict.keys(): 
        curr_ent = qual_dict[id]
        for type in ['frame', 'econ_rate', 'econ_change']:
            curr_t = curr_ent[type]
            result = '\0'
            if len(curr_t) > 1: 
                anns = [a[1] for a in curr_t]
                c = Counter(anns).most_common()
                if len(c) == 1 or c[0][1] != c[1][1]: # check for tie
                    result = c[0][0]
            qual_dict[id][type] = result

    return qual_dict

def print_qual_tags(qual_dict: dict):
    """takes dictionary output of get_qual_tags func and prints
    annotation stats"""

    labels = {'frame': {}, 'econ_rate': {}, 'econ_change': {}}

    for id in qual_dict.keys():
        curr_ent = qual_dict[id]
        for type in labels.keys():
            curr_t = curr_ent[type]
            if curr_t != '\0': 
                if curr_t in labels[type]:
                    labels[type][curr_t] += 1
                else:
                    labels[type][curr_t] = 1

    for type in labels.keys():
        count = sum(labels[type].values())
        print(type + " - count: " + str(count))
        print(str(labels[type]) + '\n')


def main(args):
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    article2ann = {}
    quantity2ann = {}

    # Frame
    query = 'select article_id, user_id, frame from articleann where frame is not null and frame != "None"';
    res = cur.execute(query)
    retrieve_anns(article2ann, res, 'frame')
    # Econ rate
    query = 'select article_id, user_id, econ_rate from articleann where econ_rate is not null and econ_rate != "None"';
    res = cur.execute(query)
    retrieve_anns(article2ann, res, 'econ_rate')
    # Econ change
    query = 'select article_id, user_id, econ_change from articleann where econ_change is not null and econ_change != "None"';
    res = cur.execute(query)
    retrieve_anns(article2ann, res, 'econ_change')
    # Quantities
    query = 'select quantity_id, user_id, type, macro_type, industry_type, gov_type, expenditure_type, revenue_type, spin from quantityann';
    res = cur.execute(query)
    retrieve_quant_anns(quantity2ann, res)

    article2ann = get_qual_tags(article2ann)
    # print(format_dict(article2ann))
    print_qual_tags(article2ann)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)
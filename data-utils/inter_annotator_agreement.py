import argparse
import sqlite3
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import interval_distance, binary_distance
from collections import Counter

def retrieve_anns(article2ann, query_res, ann_name):
    """
    Takes as input a SELECT query result for a frame component ann_name
    adds results to article2ann dict 
    """
    for (article_id, user_id, ann) in query_res:
        if article_id not in article2ann:
            article2ann[article_id] = {'frame': [], 'econ_rate': [], 'econ_change': []}
        if ann_name == 'econ_rate' and ann == 'excellent':
            ann = 'good'
        if ann_name == 'econ_rate' and ann == 'fair':
            ann = 'poor'
        article2ann[article_id][ann_name].append((user_id, ann))

def retrieve_quant_anns(quantity2ann, query_res):
    values = []
    for (quant_id, user_id, type, macro_type, industry_type, gov_type, expenditure_type, revenue_type, spin)  in query_res:
        if quant_id not in quantity2ann:
            quantity2ann[quant_id] = {'type': [], 'macro_type': [], 'industry_type': [], 'gov_type': [], 'expenditure_type': [], 'revenue_type': [], 'spin': []}
        quantity2ann[quant_id]['type'].append((user_id, type))
        if spin and spin != "None":
            quantity2ann[quant_id]['spin'].append((user_id, spin))
        if macro_type and macro_type != "None":
            quantity2ann[quant_id]['macro_type'].append((user_id, macro_type))
        if industry_type and industry_type != "None":
            quantity2ann[quant_id]['industry_type'].append((user_id, industry_type))
        if gov_type and gov_type != "None":
            quantity2ann[quant_id]['gov_type'].append((user_id, gov_type))
        if revenue_type and revenue_type != "None":
            quantity2ann[quant_id]['revenue_type'].append((user_id, revenue_type))
        if expenditure_type and expenditure_type != "None":
            quantity2ann[quant_id]['expenditure_type'].append((user_id, expenditure_type))
            #values.append(expenditure_type)
    #print(Counter(values))

def create_triplets(article2ann, ann_name, min_ann):
    ann_triplets = []
    for article_id in article2ann:
        _anns = []
        for elem in article2ann[article_id][ann_name]:
            if elem[1] != "none":
                _anns.append(elem)

        if len(_anns) >= min_ann:
            for (user_id, ann) in article2ann[article_id][ann_name]:
                ann_triplets.append((user_id, article_id, ann))

    return ann_triplets

def measure_percentage_agreement(article2ann, ann_name, user_disagreements, user_total_anns):
    num_full = 0; num_partial = 0; num_total = 0
    # Measure proportion
    for article_id in article2ann:
        if ann_name in article2ann[article_id]:
            _anns = [frame for (_, frame) in article2ann[article_id][ann_name]]
            if len(set(_anns)) == 1 and len(_anns) >= 2:
                num_full += 1; num_total += 1
                for (user, _) in article2ann[article_id][ann_name]:
                    if user not in user_total_anns:
                        user_total_anns[user] = 0
                    user_total_anns[user] += 1

            elif len(_anns) != len(set(_anns)) and len(_anns) >= 2:
                num_partial += 1; num_total += 1
                c = Counter(_anns)
                value, count = c.most_common()[0]
                for (user, ann) in article2ann[article_id][ann_name]:
                    if ann != value:
                        if user not in user_disagreements:
                            user_disagreements[user] = 0
                        user_disagreements[user] += 1
                    if user not in user_total_anns:
                        user_total_anns[user] = 0
                    user_total_anns[user] += 1
            elif len(_anns) >= 2:
                num_total += 1

    print("{} full".format(ann_name), round(num_full/num_total*100, 2), "partial", round((num_full + num_partial)/num_total*100, 2))

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

    article_ids = tuple(article2ann.keys())
    # Get article info
    query = 'select id, headline, source, url from article where id in {}'.format(article_ids)
    res = cur.execute(query)
    for article_id, headline, source, url in res:
        article2ann[article_id]['headline'] = headline
        article2ann[article_id]['source'] = source
        article2ann[article_id]['url'] = url

    frame_triplets = create_triplets(article2ann, 'frame', 2)
    t = AnnotationTask(frame_triplets, distance=binary_distance)
    result = t.alpha()
    print("Frame", round(result, 2))

    econ_rate_triplets = create_triplets(article2ann, 'econ_rate', 2)
    t = AnnotationTask(econ_rate_triplets, distance=binary_distance)
    result = t.alpha()
    print("Econ-rate", round(result, 2))

    econ_change_triplets = create_triplets(article2ann, 'econ_change', 2)
    t = AnnotationTask(econ_change_triplets, distance=binary_distance)
    result = t.alpha()
    print("Econ-change", round(result, 2))

    quantity_type_triplets = create_triplets(quantity2ann, 'type', 2)
    t = AnnotationTask(quantity_type_triplets, distance=binary_distance)
    result = t.alpha()
    print('Quantity Type', round(result, 2))

    quantity_spin_triplets = create_triplets(quantity2ann, 'spin', 2)
    t = AnnotationTask(quantity_spin_triplets, distance=binary_distance)
    result = t.alpha()
    print('Quantity Spin', round(result, 2))

    quantity_macro_triplets = create_triplets(quantity2ann, 'macro_type', 2)
    t = AnnotationTask(quantity_macro_triplets, distance=binary_distance)
    result = t.alpha()
    print('Macro', round(result, 2))

    quantity_industry_triplets = create_triplets(quantity2ann, 'industry_type', 2)
    t = AnnotationTask(quantity_industry_triplets, distance=binary_distance)
    result = t.alpha()
    print('Industry', round(result, 2))

    quantity_gov_triplets = create_triplets(quantity2ann, 'gov_type', 2)
    t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
    result = t.alpha()
    print('Gov', round(result, 2))

    quantity_gov_triplets = create_triplets(quantity2ann, 'revenue_type', 2)
    t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
    result = t.alpha()
    print('Revenue', round(result, 2))

    quantity_gov_triplets = create_triplets(quantity2ann, 'expenditure_type', 2)
    t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
    result = t.alpha()
    print('Expenditure', round(result, 2))

    user_disagreements = {}; user_total_anns = {}
    for ann_name in ['frame', 'econ_rate', 'econ_change']:
        measure_percentage_agreement(article2ann, ann_name, user_disagreements, user_total_anns)
    for ann_name in ['type', 'spin', 'macro_type', 'industry_type', 'gov_type', 'revenue_type', 'expenditure_type']:
        measure_percentage_agreement(quantity2ann, ann_name, user_disagreements, user_total_anns)

    for user in user_disagreements:
        print("user", user, ":", user_disagreements[user]/user_total_anns[user], user_total_anns[user])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)

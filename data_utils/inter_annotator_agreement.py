import argparse
import sqlite3
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import interval_distance, binary_distance
from collections import Counter

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import numpy as np
import itertools

ann_structure = {
    'frame': {
            'business': 0,
            'industry': 1,
            'macro': 2,
            'government': 3,
            'other': 4},
    'econ_rate': {
            'good': 0,
            'poor': 1,
            'none': 2},
    'econ_change': {
            'better': 0,
            'worse': 1,
            'same': 2,
            'none': 3}
}

component_name = {
    'frame': 'Article Type',
    'econ_rate': 'Economic Conditions',
    'econ_change': 'Economic Direction'
}


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
        if ann_name == 'econ_rate' and ann in ['unsure', 'irrelevant']:
            ann = 'none'
        if ann_name == 'econ_change' and ann in ['unsure', 'irrelevant']:
            ann = 'none'
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

def create_triplets(article2ann, ann_name, min_ann, max_ann=None):
    ann_triplets = []
    for article_id in article2ann:
        _anns = []
        for elem in article2ann[article_id][ann_name]:
            if elem[1] != "none":
                _anns.append(elem)

        if len(_anns) >= min_ann:
            if max_ann is None or len(_anns) <= max_ann:
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


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Annotator B',
           xlabel='Annotator A')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_agreement_matrix(article2ann):

    for ann_comp in ann_structure.keys():
        x_ann = []
        y_ann = []
        classes = np.array(list(ann_structure[ann_comp].keys()))
        label_map = ann_structure[ann_comp]
        plot_title = component_name[ann_comp] + " Agreement Confusion Matrix"

        for article_id in article2ann:
            if len(article2ann[article_id][ann_comp]) >= 2:
                combinations = itertools.combinations(article2ann[article_id][ann_comp], 2)
                for c in combinations:
                   
                    if c[0][1] != c[1][1]:
                        print(c)
                    x_ann.append(label_map[c[0][1]])
                    y_ann.append(label_map[c[1][1]])

        # plot_confusion_matrix(x_ann, y_ann, classes, normalize=False, title=plot_title)
        # plt.savefig("confusion_matrix_{}.png".format(ann_comp), dpi=300)

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

    for min, max in [(2, 2), (3, 3), [4, None]]:

        print("Min", min, "Max", max)

        frame_triplets = create_triplets(article2ann, 'frame', min, max)
        print(frame_triplets)
        t = AnnotationTask(frame_triplets, distance=binary_distance)
        result = t.alpha()
        print("Frame", round(result, 2))

        econ_rate_triplets = create_triplets(article2ann, 'econ_rate', min, max)
        t = AnnotationTask(econ_rate_triplets, distance=binary_distance)
        result = t.alpha()
        print("Econ-rate", round(result, 2))

        econ_change_triplets = create_triplets(article2ann, 'econ_change', min, max)
        t = AnnotationTask(econ_change_triplets, distance=binary_distance)
        result = t.alpha()
        print("Econ-change", round(result, 2))

        quantity_type_triplets = create_triplets(quantity2ann, 'type', min, max)
        t = AnnotationTask(quantity_type_triplets, distance=binary_distance)
        result = t.alpha()
        print('Quantity Type', round(result, 2))

        quantity_spin_triplets = create_triplets(quantity2ann, 'spin', min, max)
        t = AnnotationTask(quantity_spin_triplets, distance=binary_distance)
        result = t.alpha()
        print('Quantity Spin', round(result, 2))

        quantity_macro_triplets = create_triplets(quantity2ann, 'macro_type', min, max)
        t = AnnotationTask(quantity_macro_triplets, distance=binary_distance)
        result = t.alpha()
        print('Macro', round(result, 2))

        quantity_industry_triplets = create_triplets(quantity2ann, 'industry_type', min, max)
        t = AnnotationTask(quantity_industry_triplets, distance=binary_distance)
        result = t.alpha()
        print('Industry', round(result, 2))

        quantity_gov_triplets = create_triplets(quantity2ann, 'gov_type', min, max)
        t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
        result = t.alpha()
        print('Gov', round(result, 2))

        quantity_gov_triplets = create_triplets(quantity2ann, 'revenue_type', min, max)
        t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
        result = t.alpha()
        print('Revenue', round(result, 2))

        quantity_gov_triplets = create_triplets(quantity2ann, 'expenditure_type', min, max)
        t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
        result = t.alpha()
        print('Expenditure', round(result, 2))

        print('\n')

    user_disagreements = {}; user_total_anns = {}
    for ann_name in ['frame', 'econ_rate', 'econ_change']:
        measure_percentage_agreement(article2ann, ann_name, user_disagreements, user_total_anns)
    for ann_name in ['type', 'spin', 'macro_type', 'industry_type', 'gov_type', 'revenue_type', 'expenditure_type']:
        measure_percentage_agreement(quantity2ann, ann_name, user_disagreements, user_total_anns)

    # for user in sorted(user_disagreements.keys()):
    #     print("user", user, ":", round(user_disagreements[user]/user_total_anns[user], 2), user_total_anns[user])
    #     print(user_disagreements[user])

    get_agreement_matrix(article2ann)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)

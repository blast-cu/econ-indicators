import pickle
from collections import Counter

from data_utils.dataset import qual_label_maps

def main():
    qual_dict = pickle.load(open("data/clean/qual_dict", "rb"))
    predict_dict = pickle.load(open("data/annotations_type_predictions", "rb"))

    article_preds = {} # {article_id: [preds]}

    for quant_id, preds in predict_dict.items():
        article_id = quant_id.split('_')[0]
        if article_id not in article_preds:
            article_preds[article_id] = []

        pred = preds['type']
        print(pred)
        article_preds[article_id].append(pred)

    article_labels = {} # {article_id: [labels]}
    # TODO: get article labels from quant preds, aggregate
    for article_id, preds in article_preds.items():
        if len(preds) >= 2: # at least two quants
            c = Counter(preds).most_common()
            if c[0][1] >= 2: # at least two quants agree, doesn't account for ties
                article_labels[article_id] = c[0][0]

    bins = {} # {frame label: [article_ids]}
    for key in qual_label_maps['frame'].keys():
        bins[key] = []

    for article_id, label in article_labels.items():
        if article_id not in qual_dict:
            bins[label].append(article_id)
    
    for key, value in bins.items():
        print(key, len(value))


if __name__ == "__main__":
    main()
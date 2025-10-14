import json
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_utils.get_annotation_stats import get_site, get_date
from data_utils.model_utils.dataset import DB_FILENAME

IN_DIR = 'models/psl/data'
OUT_DIR = f"{IN_DIR}/final_combined"
ANN_DIR = 'data/clean/'
NUM_SPLITS = 50  # number of splits to combine
DEBUG = False

article_ann_dict = {
    "frame": "VALFRAME",
    "econ_change": "VALECONCHANGE",
    "econ_rate": "VALECONRATE"
}
quant_ann_dict = {
    "type": "VALTYPE",
    "macro_type": "VALMACROTYPE",
    "spin": "VALSPIN"
}


def add_pred(temp_dict, l, ann_type):
    id, ann_val, ann_score = l.strip().split('\t')
    ann_score = float(ann_score)
    if id not in temp_dict:
        temp_dict[id] = {}
    if ann_type not in temp_dict[id]:
        temp_dict[id][ann_type] = []
    temp_dict[id][ann_type].append((ann_val, ann_score))

    return temp_dict


def set_max_ann(in_dict):

    out_dict = {}
    for id, ann_dict in in_dict.items():
        out_dict[id] = {}
        for ann_type, ann_list in ann_dict.items():
            # sort by score
            ann_list.sort(key=lambda x: x[1], reverse=True)
            final_ann = ann_list[0][0]

            if "frame" in out_dict[id].keys() and final_ann == "irrelevant" and out_dict[id]["frame"] == "macro":
                final_ann = ann_list[1][0]
            elif "type" in out_dict[id].keys() and final_ann == "none" and out_dict[id]["type"] == "macro":
                final_ann = ann_list[1][0]
            
            out_dict[id][ann_type] = final_ann

    return out_dict


def main():

    global NUM_SPLITS
    if DEBUG:  # reduce number of splits for debugging
        NUM_SPLITS = 1
    logger.info(f"Combining data from {NUM_SPLITS} splits.")

    # loop over all inferred pred files
    article_dict = {} # article_id -> annotations
    quant_dict = {} # article_id -> annotations
    for i in range(NUM_SPLITS):
        pred_dir = f"{IN_DIR}/final{i}/no_inter/no_inter/inferred_predicates/"
        if not os.path.exists(pred_dir):
            logger.warning(f"Pred directory {pred_dir} does not exist. Skipping.")
            continue
        for article_ann in article_ann_dict.items():
            with open(f"{pred_dir}{article_ann[1]}.txt", 'r') as f:
                lines = f.readlines()
                for l in lines:
                    article_dict = add_pred(article_dict, l, article_ann[0])
                    
        for quant_ann in quant_ann_dict.items():
            with open(f"{pred_dir}{quant_ann[1]}.txt", 'r') as f:
                lines = f.readlines()
                for l in lines:
                    quant_dict = add_pred(quant_dict, l, quant_ann[0])
    
    # set ann with max score for each article and quant
    article_dict = set_max_ann(article_dict)
    quant_dict = set_max_ann(quant_dict)

    # add hand annotations
    agree_articles = json.load(open(f"{ANN_DIR}agreed_qual_dict.json", 'r'))
    agree_quants = json.load(open(f"{ANN_DIR}agreed_quant_dict.json", 'r'))

    for article_id, ann_dict in agree_articles.items():
        if article_id not in article_dict:
                article_dict[article_id] = {}  # add empty dict if not already present
        for ann_type in article_ann_dict.keys():  # replace with or add hand annotations
            article_dict[article_id][ann_type] = ann_dict[ann_type]
    
    for quant_id, ann_dict in agree_quants.items():
        if quant_id not in quant_dict:
                quant_dict[quant_id] = {}
        for ann_type in quant_ann_dict.keys():
            quant_dict[quant_id][ann_type] = ann_dict[ann_type]
    

    # add source and date to article_dict
    for article_id in article_dict.keys():
        article_dict[article_id]["source"] = get_site(article_id, DB_FILENAME)
        article_dict[article_id]["date"] = get_date(article_id, DB_FILENAME)

    # article_id to quant dict
    for quant_id in quant_dict.keys():
        article_id = quant_id.split('_')[0]
        quant_dict[quant_id]["article_id"] = article_id


    # export article and quant preds to json format
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "articles.json"), 'w+') as f:
        json.dump(article_dict, f, indent=4)
    logger.info(f"Exported {len(article_dict)} articles to {os.path.join(OUT_DIR, 'articles.json')}")

    with open(os.path.join(OUT_DIR, "quants.json"), 'w+') as f:
        json.dump(quant_dict, f, indent=4)
    logger.info(f"Exported {len(quant_dict)} quants to {os.path.join(OUT_DIR, 'quants.json')}")


if __name__ == "__main__":
    main()
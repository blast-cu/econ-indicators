from data_utils import inter_annotator_agreement as iaa
from data_utils import get_annotation_stats as gs
from data_utils.model_utils import dataset as d
import data_utils.visualization.generate_agree_table as at

import os
import pickle
import pandas as pd
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance

from potato_annotation.eval.read_article_annotations import get_potato_article_anns

ANN_DIR = "potato_annotation/article_annotate/annotation_output/pilot_4_20"
# ANN_DIR = "potato_annotation/article_annotate_output/quant_pilot1"



def get_qual_potato_dict(potato_anns):  # article_id, user_id, ann
    clean_ann = {}
    ann_dict = {}

    for comp, anns in potato_anns.items():
        for ann in anns:
            article_id = ann[0]
            user_id = ann[1]
            ann = ann[2]
            if article_id not in clean_ann: # article id
                clean_ann[article_id] = {}
                for qual_comp in d.qual_label_maps.keys():
                    clean_ann[article_id][qual_comp] = []
            ann = (user_id, ann)
            clean_ann[article_id][comp].append(ann)
        
    return clean_ann

def get_quant_potato_dict(potato_anns):  # article_id, user_id, ann
    
    clean_ann = {}
    for ann in potato_anns:
        quant_id = ann["quant_id"]
        
        user_id = ann["user_id"]
        quant_type = ann["type"]
        macro_type = ann["macro_type"]
        spin = ann["spin"]
        if quant_id not in clean_ann:
            clean_ann[quant_id] = {}
            clean_ann[quant_id]['type'] = []
            clean_ann[quant_id]['macro_type'] = []
            clean_ann[quant_id]['spin'] = []

        clean_ann[quant_id]['type'].append((user_id, quant_type))

        if macro_type != "none":
            clean_ann[quant_id]['macro_type'].append((user_id, macro_type))
        if spin != "none":
            clean_ann[quant_id]['spin'].append((user_id, spin))
    return clean_ann

def count_agreed(agreed_anns, label_map, report_dir):
    agree_count = {}
    for comp in label_map.keys():
        agree_count[comp] = 0
        for id in agreed_anns.keys():
            if agreed_anns[id][comp] != '\x00':
                agree_count[comp] += 1
    for k, v in agree_count.items():
        print(k, v)

def generate_disagree_examples(quantity2ann, quant_dict, report_dir):
    with open (os.path.join(report_dir, "disagree_examples.txt"), "w") as f:
        for id, anns in quantity2ann.items():
            for k, ann in anns.items():
                if len(ann) >= 2:
                    if ann[0][1] != ann[1][1]:

                        f.write(f"indicator: {quant_dict[id]['indicator']}\n")
                        f.write(f"full excerpt: {quant_dict[id]['excerpt']}\n")
                        f.write("--------\n")
                        f.write(f"potato: {ann[0][1]}\n")
                        f.write(f"house: {ann[1][1]}\n\n\n")

    print("Disagree examples written to file")

def main():

    # a = gs.get_qual_dict(d.DB_FILENAME)
    # # print(gs.get_agreed_anns(a, d.qual_label_maps))
    # print(a)
    # exit()

    article_potato, _ = get_potato_article_anns(ann_output_dir=ANN_DIR)
    # for k, v in article_potato.items():
    #     print(k, v)
    #     print() 
    # exit()
    article_potato = get_qual_potato_dict(article_potato)

    agreed_qual_potato = gs.get_agreed_anns(article_potato, d.qual_label_maps)

    agreed_qual_ann = pickle.load(open("data/clean/qual_dict", "rb"))

    article_anns = {}
    article_anns['frame'] = []
    article_anns['econ_rate'] = []
    article_anns['econ_change'] = []
    for id, anns in agreed_qual_potato.items():
        for k in anns.keys():
            # article_id, user_id, ann
            if anns[k] == '\x00':
                anns[k] = "none"
            article_anns[k].append((id, "potato", anns[k]))

    for id, anns in agreed_qual_ann.items():
        for k in anns.keys():
            if k != 'quant_list':
                if id in agreed_qual_potato:
                    if anns[k] == '\x00':
                        anns[k] = "none"
                    article_anns[k].append((id, "house", anns[k]))

    article2ann = {}
    for ann_name, anns in article_anns.items():
        iaa.retrieve_anns(article2ann, anns, ann_name)

    report_dir = os.path.join(ANN_DIR, "reports/")
    os.makedirs(report_dir, exist_ok=True)
    at.generate_agree_table(article2ann, {}, filepath=report_dir, filename="potato_house_qual_agree.csv")
    at.generate_ka_table(article2ann, {}, filepath=report_dir, filename="potato_house_qual_ka.csv")


if __name__ == "__main__":
    main()
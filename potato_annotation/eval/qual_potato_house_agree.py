from data_utils import inter_annotator_agreement as iaa
from data_utils import get_annotation_stats as gs
from data_utils.model_utils import dataset as d
import data_utils.visualization.generate_agree_table as at

import os
import pickle
import pandas as pd
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance

# from potato_annotation.eval.read_article_annotations import get_potato_article_anns
from potato_annotation.eval.read_quant_annotations import get_potato_quant_anns

ANN_DIR = "potato_annotation/quant_annotate/annotation_output/pilot"
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
    
    report_dir = os.path.join(ANN_DIR, "reports/")
    os.makedirs(report_dir, exist_ok=True)

    quant_potato = get_potato_quant_anns(ann_output_dir=ANN_DIR)
    quant_potato = get_quant_potato_dict(quant_potato)
    agreed_quant_potato = gs.get_agreed_anns(quant_potato, d.quant_label_maps)
    
    quant_ann = gs.get_quant_dict(d.DB_FILENAME)
    agreed_quant_ann = gs.get_agreed_anns(quant_ann, d.quant_label_maps)

    quant_dict = pickle.load(open("data/clean/quant_excerpts_dict", "rb"))

    to_retrieve = []
    for id, anns in agreed_quant_potato.items():
        # print(anns)
        for k, ann in anns.items():
            if ann == '\x00':
                anns[k] = "None"
        # print(id)
        curr = [id, "potato", anns['type'], anns['macro_type'], "None", "None", "None", "None", anns['spin']]
        to_retrieve.append(curr)

    for id, anns in agreed_quant_ann.items():
        if id in agreed_quant_potato:
            # print(anns)
            for k, ann in anns.items():
                if ann == '\x00':
                    anns[k] = "None"
            curr = [id, "house", anns['type'], anns['macro_type'], "None", "None", "None", "None", anns['spin']]
            to_retrieve.append(curr)

    quantity2ann = {}
    iaa.retrieve_quant_anns(quantity2ann, to_retrieve)
    generate_disagree_examples(quantity2ann, quant_dict, report_dir)
    at.generate_agree_table({},
                            quantity2ann,
                            filepath=report_dir,
                            filename="potato_house_agree")

    at.generate_ka_table({},
                         quantity2ann,
                         filepath=report_dir,
                         filename="potato_house_ka")


    # agreed_qual_potato = gs.get_agreed_anns(qual_potato, d.qual_label_maps)
    # qual_potato = get_potato_article_anns()
    # qual_potato = get_qual_potato_dict(qual_potato)
    # qual_ann = gs.get_qual_dict(d.DB_FILENAME)
    # agreed_qual_ann = gs.get_agreed_anns(qual_ann, d.qual_label_maps)


    # article_anns = {}
    # article_anns['frame'] = []
    # article_anns['econ_rate'] = []
    # article_anns['econ_change'] = []
    # for id, anns in agreed_qual_potato.items():
    #     for k in anns.keys():
    #         # article_id, user_id, ann
    #         if anns[k] == '\x00':
    #             anns[k] = "none"
    #         article_anns[k].append((id, "potato", anns[k]))

    # for id, anns in agreed_qual_ann.items():
    #     for k in anns.keys():
    #         if id in agreed_qual_potato:
    #             if anns[k] == '\x00':
    #                 anns[k] = "none"
    #             article_anns[k].append((id, "house", anns[k]))

    # article2ann = {}
    # for ann_name, anns in article_anns.items():
    #     iaa.retrieve_anns(article2ann, anns, ann_name)

    # at.generate_agree_table(article2ann, {}, "potato_house_qual_agree.csv")
    # at.generate_ka_table(article2ann, {}, "potato_house_qual_ka.csv")

if __name__ == "__main__":
    main()
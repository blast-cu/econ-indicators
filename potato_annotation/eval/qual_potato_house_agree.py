from data_utils import inter_annotator_agreement as iaa
from data_utils import get_annotation_stats as gs
from data_utils.model_utils import dataset as d
import data_utils.visualization.generate_agree_table as at

import os
import pickle
import pandas as pd
import argparse
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance

from potato_annotation.eval.read_article_annotations import get_potato_article_anns
from data_utils.model_utils.dataset import DB_FILENAME
from data_utils.get_annotation_stats import get_text

ANN_DIR = ""


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


def generate_disagree_examples(
        article2ann,
        report_dir,
        filename="disagree_examples.html"
):

    with open(os.path.join(report_dir, "disagree_examples.html"), "w") as f:
        # f.write("<style>span {background-color: yellow;}</style>")
        f.write("<h3>Disagree Examples</h3><br><br>")
        for k in ["frame", "econ_rate", "econ_change"]:
            f.write("----<br>")
            f.write("<b>Annotation Component: " + k + "</b><br>")
            f.write("----<br><br>")

            for id, anns in article2ann.items():
                potato_ann = anns[k][0][1]
                house_ann = anns[k][1][1]

                if potato_ann != house_ann:

                    text = get_text(
                        id,
                        db_filename=DB_FILENAME,
                        clean=False,
                        headline=True
                    )
                    headline = text[0].replace("\n", "<br>")
                    body = text[1].replace("\n", "<br>")
                    body = body.replace(headline, "")

                    f.write("<h1>" + headline + "</h1><br>")
                    f.write(body + "<br><br>")

                    f.write("PROLIFIC: <i>" + potato_ann + "</i><br>")
                    f.write("HOUSE: <i>" + house_ann + "</i><br>")
                    f.write("<br><br><br>")


    print("Disagree examples written to file")

def main(args):

    # a = gs.get_qual_dict(d.DB_FILENAME)
    # # print(gs.get_agreed_anns(a, d.qual_label_maps))
    # print(a)
    # exit()\
    global ANN_DIR
    ANN_DIR = f"potato_annotation/article_annotate/annotation_output/{args.sn}"

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
    generate_disagree_examples(
        article2ann,
        report_dir,
        filename="potato_house_disagree_examples.html"
    )
    at.generate_agree_table(article2ann, {}, filepath=report_dir, filename="potato_house_qual_agree.csv")
    at.generate_ka_table(article2ann, {}, filepath=report_dir, filename="potato_house_qual_ka.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sn",
        type=str,
        help="Name of the study to generate reports for."
    )
    args = parser.parse_args()
    main(args)
import os
import pandas as pd
from potato_annotation.eval.read_article_annotations import get_potato_quant_anns
from potato_annotation.eval.quant_potato_house_agree import get_quant_potato_dict
import data_utils.get_annotation_stats as gs
from data_utils.model_utils import dataset as d
from data_utils import inter_annotator_agreement as iaa

ANN_DIR = "potato_annotation/qual_annotate/annotation_output/pilot"
# ANN_DIR = "potato_annotation/article_annotate_output/quant_pilot1"


def get_user_ann_disagreement(anns):
    user_ann_disagreement = {}
    total_disagreements = {}
    for ann_name in d.quant_label_maps.keys():
        user_ann_disagreement[ann_name] = {}
        user_disagreements = {}
        user_total_anns = {}
        iaa.measure_percentage_agreement(anns, ann_name, user_disagreements, user_total_anns)
        for user in sorted(user_disagreements.keys()):
            if user not in total_disagreements:
                total_disagreements[user] = [0, 0]
            percent_disagree = round(user_disagreements[user]/user_total_anns[user], 2)
            user_ann_disagreement[ann_name][user] = percent_disagree
            total_disagreements[user][0] += user_disagreements[user]
            total_disagreements[user][1] += user_total_anns[user]

    user_ann_disagreement['total'] = {}
    for user in total_disagreements:
        percent_disagree = round(total_disagreements[user][0]/total_disagreements[user][1], 2)
        user_ann_disagreement['total'][user] = percent_disagree

    return user_ann_disagreement

def main():
    report_dir = os.path.join(ANN_DIR, "reports/")
    os.makedirs(report_dir, exist_ok=True)

    quant_potato, annotator_stats = get_potato_quant_anns(ann_output_dir=ANN_DIR,
                                                          annotator_stats=True)
    quant_potato = get_quant_potato_dict(quant_potato)

    user_ann_disagreement = get_user_ann_disagreement(quant_potato)

    annotator_stats['type_disagreement'] = []
    annotator_stats['macro_type_disagreement'] = []
    annotator_stats['spin_disagreement'] = []
    annotator_stats['total_disagreement'] = []

    for user in annotator_stats['user_id']:
        annotator_stats['type_disagreement'].append(user_ann_disagreement['type'][user])
        annotator_stats['macro_type_disagreement'].append(user_ann_disagreement['macro_type'][user])
        annotator_stats['spin_disagreement'].append(user_ann_disagreement['spin'][user])
        annotator_stats['total_disagreement'].append(user_ann_disagreement['total'][user])

    filepath = report_dir
    filename = "annotator_stats"
    print(f"Saving {filename} table to {filepath}{filename}.csv")
    pd.DataFrame(annotator_stats).to_csv(f'{filepath}{filename}.csv', index=False)

if __name__ == "__main__":
    main()
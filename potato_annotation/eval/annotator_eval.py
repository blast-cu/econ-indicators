import os
import pandas as pd
from potato_annotation.eval.read_quant_annotations import get_potato_quant_anns
from potato_annotation.eval.potato_house_agree import get_quant_potato_dict
import data_utils.get_annotation_stats as gs
from data_utils import dataset as d
from data_utils import inter_annotator_agreement as iaa

ANN_DIR = "potato_annotation/quant_annotate/annotation_output/pilot"
# ANN_DIR = "potato_annotation/article_annotate_output/quant_pilot1"


def main():
    report_dir = os.path.join(ANN_DIR, "reports/")
    os.makedirs(report_dir, exist_ok=True)


    user_ann_disagreement = {}
    quant_potato, annotator_stats = get_potato_quant_anns(ann_output_dir=ANN_DIR,
                                                          annotator_stats=True)
    quant_potato = get_quant_potato_dict(quant_potato)

    for ann_name in d.quant_label_maps.keys():
        user_ann_disagreement[ann_name] = {}
        user_disagreements = {}
        user_total_anns = {}
        iaa.measure_percentage_agreement(quant_potato, ann_name, user_disagreements, user_total_anns)
        for user in sorted(user_disagreements.keys()):
            percent_disagree = round(user_disagreements[user]/user_total_anns[user], 2)
            user_ann_disagreement[ann_name][user] = percent_disagree
 
    # for ann_name, anns in user_ann_disagreement.items():
    #     print(ann_name)
    #     for user, percent_disagree in anns.items():
    #         print(f"{user}: {percent_disagree}")
    #     print()
    annotator_stats['type_disagreement'] = []
    annotator_stats['macro_type_disagreement'] = []
    annotator_stats['spin_disagreement'] = []
    for user in annotator_stats['user_id']:
        annotator_stats['type_disagreement'].append(user_ann_disagreement['type'][user])
        annotator_stats['macro_type_disagreement'].append(user_ann_disagreement['macro_type'][user])
        annotator_stats['spin_disagreement'].append(user_ann_disagreement['spin'][user])
    
    filepath = report_dir
    filename = "annotator_stats"
    print(f"Saving agreement table to {filepath}{filename}.csv")
    pd.DataFrame(annotator_stats).to_csv(f'{filepath}{filename}.csv', index=False)

if __name__ == "__main__":
    main()
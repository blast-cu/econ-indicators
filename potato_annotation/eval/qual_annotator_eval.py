import os
import pandas as pd
from potato_annotation.eval.read_article_annotations \
    import get_potato_article_anns
from potato_annotation.eval.qual_potato_house_agree \
    import get_qual_potato_dict
from data_utils.visualization.generate_agree_table import \
    generate_agree_table, generate_ka_table
import data_utils.get_annotation_stats as gs
from data_utils.model_utils import dataset as d
from data_utils import inter_annotator_agreement as iaa

ANN_DIR = "potato_annotation/article_annotate/annotation_output/pilot_5_16"
# ANN_DIR = "potato_annotation/article_annotate_output/quant_pilot1"


def get_user_ann_disagreement(anns):
    user_ann_disagreement = {}
    total_disagreements = {}
    for ann_name in d.qual_label_maps.keys():
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
        ratio_disagree = total_disagreements[user][0]/total_disagreements[user][1]
        percent_disagree = round(ratio_disagree, 2)
        user_ann_disagreement['total'][user] = percent_disagree

    return user_ann_disagreement

def main():
    report_dir = os.path.join(ANN_DIR, "reports/")
    os.makedirs(report_dir, exist_ok=True)

    qual_potato, annotator_stats = get_potato_article_anns(
        ann_output_dir=ANN_DIR
    )
    qual_potato = get_qual_potato_dict(qual_potato)
    # print(qual_potato)

    user_ann_disagreement = get_user_ann_disagreement(qual_potato)

    for k in user_ann_disagreement:
        print(k, user_ann_disagreement[k])

    annotator_stats['frame_disagreement'] = []
    annotator_stats['econ_rate_disagreement'] = []
    annotator_stats['econ_change_disagreement'] = []
    annotator_stats['total_disagreement'] = []

    for user in annotator_stats['user_id']:
        annotator_stats['frame_disagreement'].append(
            user_ann_disagreement['frame'][user]
        )
        annotator_stats['econ_rate_disagreement'].append(
            user_ann_disagreement['econ_rate'][user]
        )
        annotator_stats['econ_change_disagreement'].append(
            user_ann_disagreement['econ_change'][user]
        )
        annotator_stats['total_disagreement'].append(
            user_ann_disagreement['total'][user]
        )

    filepath = report_dir
    filename = "annotator_stats"
    print(f"Saving {filename} table to {filepath}{filename}.csv")
    pd.DataFrame(annotator_stats).to_csv(f'{filepath}{filename}.csv', index=False)

    # for a_id in qual_potato:
    #     print(a_id, qual_potato[a_id])
    #     print()

    generate_agree_table(
        qual_potato,
        {},
        filepath=report_dir,
        filename="inter_annotator_agreement"
    )

    generate_ka_table(
        qual_potato,
        {},
        filepath=report_dir,
        filename="inter_ka"
    )

if __name__ == "__main__":
    main()
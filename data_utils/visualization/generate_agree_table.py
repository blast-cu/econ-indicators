import pandas as pd
import os
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance

from data_utils.inter_annotator_agreement import \
    measure_percentage_agreement, get_anns, create_triplets, retrieve_anns, retrieve_quant_anns

from potato_annotation.eval.read_article_annotations import get_potato_article_anns
from potato_annotation.eval.read_quant_annotations import get_potato_quant_anns

SETTING = 'original'


def generate_agree_table(article2ann,
                         quantity2ann,
                         extended=False,  # if True, include all annotations
                         filepath="data_utils/visualization/results/annotation/",
                         filename="annotation_agreement"):

    user_disagreements = {}
    user_total_anns = {}
    agree_table = {}
    agree_table['annotation'] = []
    agree_table['full'] = []
    agree_table['partial'] = []
    if article2ann != {}:
        for ann_name in ['frame', 'econ_rate', 'econ_change']:
            full, partial = measure_percentage_agreement(article2ann, ann_name, user_disagreements, user_total_anns)
            agree_table['annotation'].append(ann_name)
            agree_table['full'].append(full)
            agree_table['partial'].append(partial)
    if quantity2ann != {}:
        quant_anns = ['type', 'macro_type', 'spin']
        if extended:
            quant_anns += ['industry_type', 'gov_type', 'revenue_type', 'expenditure_type']
        for ann_name in quant_anns:
            full, partial = measure_percentage_agreement(quantity2ann,
                                                         ann_name,
                                                         user_disagreements,
                                                         user_total_anns)
            # populate agree_table
            agree_table['annotation'].append(ann_name)
            agree_table['full'].append(full)
            agree_table['partial'].append(partial)

    print(f"Saving agreement table to {filepath}{filename}.csv")
    pd.DataFrame(agree_table).to_csv(f'{filepath}{filename}.csv', index=False)


def generate_ka_table(article2ann,
                      quantity2ann,
                      extended=False,  # if True, include all annotations
                      filepath="data_utils/visualization/results/annotation/",
                      filename='annotation_ka'):

    ka_table = {}
    ka_table['annotation'] = []
    ka_table['ka'] = []

    if article2ann != {}:
        frame_triplets = create_triplets(article2ann, 'frame')
        t = AnnotationTask(frame_triplets, distance=binary_distance)
        result = t.alpha()
        ka_table['annotation'].append('frame')
        ka_table['ka'].append(round(result, 2))

        econ_rate_triplets = create_triplets(article2ann, 'econ_rate')
        t = AnnotationTask(econ_rate_triplets, distance=binary_distance)
        result = t.alpha()
        ka_table['annotation'].append('econ_rate')
        ka_table['ka'].append(round(result, 2))

        econ_change_triplets = create_triplets(article2ann, 'econ_change')
        t = AnnotationTask(econ_change_triplets, distance=binary_distance)
        result = t.alpha()
        ka_table['annotation'].append('econ_change')
        ka_table['ka'].append(round(result, 2))

    if quantity2ann != {}:

        quantity_type_triplets = create_triplets(quantity2ann, 'type')
        t = AnnotationTask(quantity_type_triplets, distance=binary_distance)
        result = t.alpha()
        ka_table['annotation'].append('type')
        ka_table['ka'].append(round(result, 2))

        quantity_macro_triplets = create_triplets(quantity2ann, 'macro_type')
        t = AnnotationTask(quantity_macro_triplets, distance=binary_distance)
        result = t.alpha()
        ka_table['annotation'].append('macro_type')
        ka_table['ka'].append(round(result, 2))

        quantity_spin_triplets = create_triplets(quantity2ann, 'spin')
        t = AnnotationTask(quantity_spin_triplets, distance=binary_distance)
        result = t.alpha()
        ka_table['annotation'].append('spin')
        ka_table['ka'].append(round(result, 2))

        if extended:

            quantity_industry_triplets = create_triplets(quantity2ann, 'industry_type')
            t = AnnotationTask(quantity_industry_triplets, distance=binary_distance)
            result = t.alpha()
            ka_table['annotation'].append('industry_type')
            ka_table['ka'].append(round(result, 2))

            quantity_gov_triplets = create_triplets(quantity2ann, 'gov_type')
            t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
            result = t.alpha()
            ka_table['annotation'].append('gov_type')
            ka_table['ka'].append(round(result, 2))

            quantity_gov_triplets = create_triplets(quantity2ann, 'revenue_type')
            t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
            result = t.alpha()
            ka_table['annotation'].append('revenue_type')
            ka_table['ka'].append(round(result, 2))

            quantity_gov_triplets = create_triplets(quantity2ann, 'expenditure_type')
            t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
            result = t.alpha()
            ka_table['annotation'].append('expenditure_type')
            ka_table['ka'].append(round(result, 2))

    print(f"Saving KA table to {filepath}{filename}.csv")
    pd.DataFrame(ka_table).to_csv(f"{filepath}{filename}.csv", index=False)


def main():

    if SETTING == 'potato_qual':
        article_anns = get_potato_article_anns()
    
        article2ann = {}
        for ann_name, anns in article_anns.items():
            retrieve_anns(article2ann, anns, ann_name)

        generate_agree_table(article2ann, {})
        generate_ka_table(article2ann, {})
    
    elif SETTING == 'potato_quant':
        # ann_dir = "potato_annotation/quant_annotate/annotation_output/pilot"
        ann_dir = "potato_annotation/article_annotate_output/quant_pilot1"
        report_dir = os.path.join(ann_dir, "reports/")
        os.makedirs(report_dir, exist_ok=True)
        quant_anns = get_potato_quant_anns(ann_output_dir=ann_dir)
        
        # quant_id, user_id, type, macro_type, industry_type, gov_type, expenditure_type, revenue_type, spin
        to_retrieve = []
        for anns in quant_anns:
            curr = [anns['quant_id'], anns['user_id'], anns['type'], anns['macro_type'], "none", "none", "none", "none", anns['spin']]
            to_retrieve.append(curr)
        quantity2ann = {}
        retrieve_quant_anns(quantity2ann, to_retrieve)
        generate_agree_table({}, quantity2ann, filepath=report_dir, filename="agreement")
        generate_ka_table({}, quantity2ann, filepath=report_dir, filename="ka")

    elif SETTING == 'original':

        filter = True
        if filter: 
            # article_anns = get_potato_article_anns()
    
            # potato_article2ann = {}
            # for ann_name, anns in article_anns.items():
            #     retrieve_anns(potato_article2ann, anns, ann_name)
            
            article2ann, quantity2ann = get_anns(db_filename='data/data.db')

            # potato_keys = potato_article2ann.keys()
            # article2ann = {k: v for k, v in article2ann.items() if k in potato_keys}
            # print(article2ann)
            # generate_agree_table(article2ann, {}, "overlap_annotation_agreement.csv")
            # generate_ka_table(article2ann, {}, "overlap_annotation_krippendorff_alpha.csv")

            # quant_anns = get_potato_quant_anns()
        
            # quant_id, user_id, type, macro_type, industry_type, gov_type, expenditure_type, revenue_type, spin
            # to_retrieve = []
            # for anns in quant_anns:
            #     curr = [anns['quant_id'], anns['user_id'], anns['type'], anns['macro_type'], "none", "none", "none", "none", anns['spin']]
            #     to_retrieve.append(curr)
            # potato_quantity2ann = {}
            # retrieve_quant_anns(potato_quantity2ann, to_retrieve)
            
            # potato_keys = potato_quantity2ann.keys()
            # quantity2ann = {k: v for k, v in quantity2ann.items() if k in potato_keys}

            generate_agree_table(article2ann, quantity2ann, "quant_overlap_annotation_agreement.csv")
            generate_ka_table(article2ann, quantity2ann, "quant_overlap_annotation_krippendorff_alpha.csv")

    # see = article_anns['frame']
    # see.sort()
    # for s in see:
    #     print(s)

        
    
if __name__ == "__main__":
    main()
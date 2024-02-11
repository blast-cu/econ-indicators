import pandas as pd
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance

from data_utils.inter_annotator_agreement import \
    measure_percentage_agreement, get_anns, create_triplets, retrieve_anns

from potato_annotation.read_annotations import get_potato_article_anns

def generate_agree_table(article2ann, quantity2ann):

    user_disagreements = {}; user_total_anns = {}
    agree_table = {}
    agree_table['annotation'] = []
    agree_table['full'] = []
    agree_table['partial'] = []
    for ann_name in ['frame', 'econ_rate', 'econ_change']:
        full, partial = measure_percentage_agreement(article2ann, ann_name, user_disagreements, user_total_anns)
        agree_table['annotation'].append(ann_name)
        agree_table['full'].append(full)
        agree_table['partial'].append(partial)

    # for ann_name in ['type', 'spin', 'macro_type', 'industry_type', 'gov_type', 'revenue_type', 'expenditure_type']:
    #     full, partial = measure_percentage_agreement(quantity2ann, ann_name, user_disagreements, user_total_anns)
    #     agree_table['annotation'].append(ann_name)
    #     agree_table['full'].append(full)
    #     agree_table['partial'].append(partial)

    # pd.DataFrame(agree_table).to_csv('data_utils/table_generators/results/annotation_agreement.csv', index=False)
    pd.DataFrame(agree_table).to_csv('data_utils/table_generators/results/potato_pilot_annotation_agreement.csv', index=False)
        


def generate_ka_table(article2ann, quantity2ann):

    ka_table = {}
    ka_table['annotation'] = []
    ka_table['ka'] = []

    frame_triplets = create_triplets(article2ann, 'frame')
    for t in frame_triplets:
        print(t)
    t = AnnotationTask(frame_triplets, distance=binary_distance)

    result = t.alpha()
    ka_table['annotation'].append('frame')
    ka_table['ka'].append(round(result, 2))
    print(round(result, 2))

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

    # quantity_type_triplets = create_triplets(quantity2ann, 'type')
    # t = AnnotationTask(quantity_type_triplets, distance=binary_distance)
    # result = t.alpha()
    # ka_table['annotation'].append('type')
    # ka_table['ka'].append(round(result, 2))

    # quantity_macro_triplets = create_triplets(quantity2ann, 'macro_type')
    # t = AnnotationTask(quantity_macro_triplets, distance=binary_distance)
    # result = t.alpha()
    # ka_table['annotation'].append('macro_type')
    # ka_table['ka'].append(round(result, 2))

    # quantity_spin_triplets = create_triplets(quantity2ann, 'spin')
    # t = AnnotationTask(quantity_spin_triplets, distance=binary_distance)
    # result = t.alpha()
    # ka_table['annotation'].append('spin')
    # ka_table['ka'].append(round(result, 2))

    # quantity_industry_triplets = create_triplets(quantity2ann, 'industry_type')
    # t = AnnotationTask(quantity_industry_triplets, distance=binary_distance)
    # result = t.alpha()
    # ka_table['annotation'].append('industry_type')
    # ka_table['ka'].append(round(result, 2))

    # quantity_gov_triplets = create_triplets(quantity2ann, 'gov_type')
    # t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
    # result = t.alpha()
    # ka_table['annotation'].append('gov_type')
    # ka_table['ka'].append(round(result, 2))

    # quantity_gov_triplets = create_triplets(quantity2ann, 'revenue_type')
    # t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
    # result = t.alpha()
    # ka_table['annotation'].append('revenue_type')
    # ka_table['ka'].append(round(result, 2))

    # quantity_gov_triplets = create_triplets(quantity2ann, 'expenditure_type')
    # t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
    # result = t.alpha()
    # ka_table['annotation'].append('expenditure_type')
    # ka_table['ka'].append(round(result, 2))

    # pd.DataFrame(ka_table).to_csv('data_utils/table_generators/results/annotation_krippendorff_alpha.csv', index=False)
    pd.DataFrame(ka_table).to_csv('data_utils/table_generators/results/potato_pilot_annotation_krippendorff_alpha.csv', index=False)


def main():

    # article2ann, quantity2ann = get_anns(db_filename='data/data.db')
    # print(article2ann)
    # generate_agree_table(article2ann, quantity2ann)
    # generate_ka_table(article2ann, quantity2ann)
    article_anns = get_potato_article_anns()
    # see = article_anns['frame']
    # see.sort()
    # for s in see:
    #     print(s)
    # exit()
    article2ann = {}
    for ann_name, anns in article_anns.items():
        retrieve_anns(article2ann, anns, ann_name)

    
    generate_agree_table(article2ann, {})
    generate_ka_table(article2ann, {})
        
    
if __name__ == "__main__":
    main()
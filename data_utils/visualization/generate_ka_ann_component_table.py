from data_utils.inter_annotator_agreement import create_triplets, get_anns
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import binary_distance
from data_utils.model_utils.dataset import DB_FILENAME
import pandas as pd

def main():
    
    article2ann, quantity2ann = get_anns(DB_FILENAME)

    table = {}
    table['annotator_count'] = []

    for min, max in [(2, 2), (3, 3), [4, None]]:

        table['annotator_count'].append(min)

        frame_triplets = create_triplets(article2ann, 'frame', min, max)
        # print(frame_triplets)
        t = AnnotationTask(frame_triplets, distance=binary_distance)
        result = t.alpha()
        if "frame" not in table:
            table['frame'] = []
        table['frame'].append(round(result, 2))

        econ_rate_triplets = create_triplets(article2ann, 'econ_rate', min, max)
        t = AnnotationTask(econ_rate_triplets, distance=binary_distance)
        result = t.alpha()
        if "econ_rate" not in table:
            table['econ_rate'] = []
        table['econ_rate'].append(round(result, 2))

        econ_change_triplets = create_triplets(article2ann, 'econ_change', min, max)
        t = AnnotationTask(econ_change_triplets, distance=binary_distance)
        result = t.alpha()
        if "econ_change" not in table:
            table['econ_change'] = []
        table['econ_change'].append(round(result, 2))

        quantity_type_triplets = create_triplets(quantity2ann, 'type', min, max)
        t = AnnotationTask(quantity_type_triplets, distance=binary_distance)
        result = t.alpha()
        if "quantity_type" not in table:
            table['quantity_type'] = []
        table['quantity_type'].append(round(result, 2))

        quantity_spin_triplets = create_triplets(quantity2ann, 'spin', min, max)
        t = AnnotationTask(quantity_spin_triplets, distance=binary_distance)
        result = t.alpha()
        if "quantity_spin" not in table:
            table['quantity_spin'] = []
        table['quantity_spin'].append(round(result, 2))

        quantity_macro_triplets = create_triplets(quantity2ann, 'macro_type', min, max)
        t = AnnotationTask(quantity_macro_triplets, distance=binary_distance)
        result = t.alpha()
        if "macro_type" not in table:
            table['macro_type'] = []
        table['macro_type'].append(round(result, 2))

        # quantity_industry_triplets = create_triplets(quantity2ann, 'industry_type', min, max)
        # t = AnnotationTask(quantity_industry_triplets, distance=binary_distance)
        # result = t.alpha()
        # print('Industry', round(result, 2))

        # quantity_gov_triplets = create_triplets(quantity2ann, 'gov_type', min, max)
        # t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
        # result = t.alpha()
        # print('Gov', round(result, 2))

        # quantity_gov_triplets = create_triplets(quantity2ann, 'revenue_type', min, max)
        # t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
        # result = t.alpha()
        # print('Revenue', round(result, 2))

        # quantity_gov_triplets = create_triplets(quantity2ann, 'expenditure_type', min, max)
        # t = AnnotationTask(quantity_gov_triplets, distance=binary_distance)
        # result = t.alpha()
        # print('Expenditure', round(result, 2))

    pd.DataFrame(table).to_csv('data_utils/table_generators/results/ka_ann_component.csv')



if __name__ == "__main__":
    main()
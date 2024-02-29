from data_utils.model_utils.dataset import DB_FILENAME, qual_label_maps, quant_label_maps
from data_utils.inter_annotator_agreement import get_anns, annotator_percentages

import pandas as pd


def main():
    article2ann, quantity2ann = get_anns(DB_FILENAME)

    qual_table = annotator_percentages(article2ann, qual_label_maps)
    quant_table = annotator_percentages(quantity2ann, quant_label_maps)

    for k in quant_table.keys():
        qual_table[k] += quant_table[k]

    pd.DataFrame(qual_table).to_csv('data_utils/table_generators/results/annotator_percentages.csv')

if __name__ == "__main__":
    main()
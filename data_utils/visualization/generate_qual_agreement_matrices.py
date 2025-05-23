from data_utils.inter_annotator_agreement import get_agreement_matrix, get_anns
from data_utils.model_utils.dataset import DB_FILENAME, qual_label_maps, quant_label_maps

import pickle
def main():

    article2ann, quantity2ann = get_anns(DB_FILENAME)
    output_path = 'data_utils/visualization/results/annotation/'
    get_agreement_matrix(article2ann, qual_label_maps, False, output_path)

    excerpts_dict = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))
    get_agreement_matrix(quantity2ann, quant_label_maps, True, output_path, excerpts_dict)

    

if __name__ == "__main__":
    main()
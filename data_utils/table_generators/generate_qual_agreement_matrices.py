from data_utils.inter_annotator_agreement import get_agreement_matrix, get_anns
from data_utils.dataset import DB_FILENAME, qual_label_maps
def main():

    article2ann, quantity2ann = get_anns(DB_FILENAME)
    output_path = 'data_utils/table_generators/results/'
    get_agreement_matrix(article2ann, qual_label_maps, False, output_path)
    

if __name__ == "__main__":
    main()
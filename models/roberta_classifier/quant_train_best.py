import argparse
import pickle
import os
import json

# import models.roberta_classifier.utils.general as gu
# # import models.roberta_classifier.utils.quant as qu
# import models.roberta_classifier.utils.legacy_quant as qu
import data_utils.model_utils.dataset as d
# import data_utils.model_utils.eval as e
from models.roberta_classifier.quant_train_best import get_checkpoint_path

# takes about 12 hours to run on CURC


def main(args):

    # convert args to dict
    model_setup = vars(args)

    # load data
    split_dir = "data/clean/"
    ann_data = json.load(open(split_dir + 'agreed_quant_dict.json', 'r'))
    text_data = pickle.load(open(split_dir + 'quant_dict', 'rb'))
    train_ids = list(ann_data.keys())

    for task in d.quant_label_maps.keys():

        model_checkpoint = get_checkpoint_path(model_setup[task])


   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get setup for each ann component")
    parser.add_argument("--type", type=str, help="type ann component model setup")
    parser.add_argument("--macro_type", type=str, help="macro_type ann component model setup")
    parser.add_argument("--spin", type=str, help="spin ann component model setup")
    args = parser.parse_args()
    main(args)
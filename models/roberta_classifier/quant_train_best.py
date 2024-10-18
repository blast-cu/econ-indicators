import argparse
import pickle
import os
import json

import models.roberta_classifier.utils.general as gu
import models.roberta_classifier.utils.legacy_quant as qu
import data_utils.model_utils.dataset as d
# import data_utils.model_utils.eval as e
from models.roberta_classifier.qual_train_best import get_checkpoint_path

# takes about 12 hours to run on CURC
OUT_DIR = "data/models/final_classifier/"

def get_texts(annotation_component, task, ann_data):

    texts = []  # list of [indicator text, text with context]
    labels = []

    for id, ann in ann_data.items():
        if ann[annotation_component] != '\x00':
            indicator_text = ann['indicator']
            excerpt_text = ann['excerpt']
            text = [indicator_text, excerpt_text]
            texts.append(text)

            label = ann_data[id][annotation_component]
            labels.append(d.quant_label_maps[task][label])

    return texts, labels


def main(args):

    # convert args to dict
    model_setup = vars(args)

    # load data
    split_dir = "data/clean/"
    ann_data = json.load(open(split_dir + 'agreed_quant_dict.json', 'r'))

    for task in d.quant_label_maps.keys():

        model_checkpoint = get_checkpoint_path(model_setup[task])
        for task in list(d.quant_label_maps.keys()):

            ann_component = task.split('-')[0]
            train_texts, train_labels = \
                get_texts(
                    ann_component,
                    task,
                    ann_data
                )

            # gets class weights for loss function
            class_weights = gu.get_weights(
                train_labels,
                d.quant_label_maps[task]
            )

            model, train_loader, val_loader, _, optimizer = \
                qu.setup(train_texts,
                         None,
                         train_labels,
                         None,
                         d.quant_label_maps[task],
                         model_checkpoint=model_checkpoint)

            tuned_model = qu.train(model,
                                   train_loader,
                                   val_loader,
                                   optimizer,
                                   class_weights)

            # save model
            dest = f"{OUT_DIR}/"
            os.makedirs(dest, exist_ok=True)
            model_dest = dest + task + "_model"
            tuned_model.save_pretrained(model_dest)  # save model

            # save training setup
            config = {}
            config['setup'] = model_setup[task]
            config['checkpoint'] = model_checkpoint
            config['class_weights'] = class_weights.tolist()
            config['train length'] = len(train_texts)
            json.dump(config, open(os.path.join(model_dest, 'train_config.json'), 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get setup for each ann component")
    parser.add_argument("--type", type=str, help="type ann component model setup")
    parser.add_argument("--macro_type", type=str, help="macro_type ann component model setup")
    parser.add_argument("--spin", type=str, help="spin ann component model setup")
    args = parser.parse_args()
    main(args)
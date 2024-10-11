import argparse
import json
import pickle
import os

import models.roberta_classifier.utils.qual as tt
import models.roberta_classifier.utils.general as gu
import data_utils.model_utils.dataset as d
import data_utils.model_utils.eval as e

OUT_DIR = "data/models/final_classifier/"


def get_checkpoint_path(model_setup):

    if model_setup == 'dapt':  # using domain-adapted model
        model_checkpoint = "data/masked/"
    elif model_setup == 'dapt_128':
        model_checkpoint = "data/models/roberta_base_dapt_128/"
    elif model_setup == 'dapt_512':
        model_checkpoint = "data/models/roberta_base_dapt_512/"
    elif model_setup == 'base':  # using base model
        model_checkpoint = "roberta-base"
    elif model_setup == 'large':
        model_checkpoint = "roberta-large"
    elif model_setup == 'no_leak':
        model_checkpoint = "models/roberta_classifier/tuned_models/roberta-base-dapt-no-leak"
    else:
        raise ValueError("Invalid setting: {}".format(model_setup))
    return model_checkpoint


def main(args):
    
    # convert args to dict
    model_setup = vars(args)

    # load data
    split_dir = "data/clean/"
    ann_data = json.load(open(split_dir + 'agreed_qual_dict.json', 'r'))
    text_data = pickle.load(open(split_dir + 'qual_dict', 'rb'))
    train_ids = [int(k) for k in ann_data.keys()]

    for task in d.qual_label_maps.keys():

        model_checkpoint = get_checkpoint_path(model_setup[task])
        annotation_component = task.split('-')[0]

        train_texts, train_labels = \
            tt.get_texts(
                d.DB_FILENAME,
                annotation_component,
                task,
                text_data,
                train_ids
            )
        class_weights = gu.get_weights(
                train_labels,
                d.qual_label_maps[task]
        )

        print(">>> Annotation component: " + annotation_component)
        print(">>> Number Train texts: " + str(len(train_texts)))

        train_texts = [t.replace('\n', '') for t in train_texts]
        model, train_loader, val_loader, _, optimizer = \
            tt.setup(
                train_texts,
                None,  # no test texts
                train_labels,
                None,  # no test labels
                d.qual_label_maps[task],
                model_checkpoint=model_checkpoint
            )

        tuned_model = tt.train(
            model, train_loader, val_loader,
            optimizer, class_weights
        )

        dest = f"{OUT_DIR}/"
        os.makedirs(dest, exist_ok=True)
        model_dest = dest + task + "_model"
        tuned_model.save_pretrained(model_dest)  # save model

        config = {}
        config['setup'] = model_setup[task]
        config['checkpoint'] = model_checkpoint
        config['class_weights'] = class_weights.tolist()
        config['train length'] = len(train_texts)
        json.dump(config, open(os.path.join(model_dest, 'train_config.json'), 'w'), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get setup for each ann component")
    parser.add_argument("--frame", type=str, help="frame ann component model setup")
    parser.add_argument("--econ_rate", type=str, help="econ_rate ann component model setup")
    parser.add_argument("--econ_change", type=str, help="econ_change ann component model setup")
    args = parser.parse_args()
    main(args)
import data_utils.get_annotation_stats as gs # msql queries
import dataset as d

import argparse
from sklearn.model_selection import KFold
import pickle


def remove_empty(ann_dict: dict):
    """
    Remove empty annotations from the given annotation dictionary.

    Args:
        ann_dict (dict): The annotation dictionary to remove empty annotations from.

    Returns:
        dict: The new dictionary with non-empty annotations.
    """

    # {key=article_id, value={key=local annotation id, value=label}}
    new_dict = {}  # to return

    for id, id_dict in ann_dict.items():
        empty = True

        # add non-empty annotations to clean_ann
        for ann, label in id_dict.items():
            if label != '\0':
                empty = False

        # if an article has at least one non-empty annotation, include
        if not empty:
            new_dict[id] = id_dict

    return new_dict


def main(args):
    """
    Generate train/test splits for econ indicator models. Save the splits and 
    qual/quant dictionaries to models/utils/splits as pickles. 

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        None
    """

    db_filename = args.db

    # get agreed article-level annotations
    # {key = articleid, value = dict of annotations}
    qual_ann = gs.get_qual_dict(db_filename)
    agreed_qual_ann = gs.get_agreed_anns(qual_ann)
    agreed_qual_ann = remove_empty(agreed_qual_ann)
    
    for k in agreed_qual_ann.keys():
        agreed_qual_ann[k]['quant_list'] = []

    # get agreed quantitative annotations in dict where 
    # {key = "articleid_localid", value = dict of annotations}
    quant_ann = gs.get_quant_dict(db_filename)
    agreed_quant_ann = gs.get_agreed_anns(quant_ann)
    agreed_quant_ann = remove_empty(agreed_quant_ann)


    # add quant_ids to agreed_qual_ann dict
    for quant_id in agreed_quant_ann.keys():
        article_id, local_id = quant_id.split('_')
        article_id = int(article_id)
        if article_id not in agreed_qual_ann.keys():
            agreed_qual_ann[article_id] = {}
            agreed_qual_ann[article_id]['frame'] = '\x00'
            agreed_qual_ann[article_id]['econ_rate'] = '\x00'
            agreed_qual_ann[article_id]['econ_change'] = '\x00'
            agreed_qual_ann[article_id]['quant_list'] = []

        agreed_qual_ann[article_id]['quant_list'].append(quant_id)

    # add text excerpts w/ context to agreed_quant_ann dict
    for id in agreed_qual_ann.keys():
        if len(agreed_qual_ann[id]['quant_list']) > 0:
            excerpts = d.get_excerpts(agreed_qual_ann[id]['quant_list'],
                                      db_filename)

            for id, text in excerpts.items():
                agreed_quant_ann[id]['excerpt'] = text


    # create splits 
    split_dict = {}
    article_ids = list(agreed_qual_ann.keys())
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    split = kf.split(article_ids)

    # format split dictionary {key = fold number, value = dict of train/test ids}
    for i, (train_index, test_index) in enumerate(split):
        split_dict[i] = {}
        split_dict[i]['train'] = [article_ids[t] for t in train_index]
        split_dict[i]['test'] = [article_ids[t] for t in test_index]

    for k, v in agreed_quant_ann.items():
        print(k, v)

    # print split dictionary
    for k in split_dict.keys():
        print(f"Fold {k}")
        print(f"Train: {split_dict[k]['train']}")
        print(f"Test: {split_dict[k]['test']}")

    # save dictionaries as pickles 
    base_dir = 'data/clean/'
    d.save_progress(split_dict, f'{base_dir}splits_dict')
    d.save_progress(agreed_quant_ann, f'{base_dir}quant_dict')
    d.save_progress(agreed_qual_ann, f'{base_dir}qual_dict')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line arguments.')
    parser.add_argument('--db', required=True, help='Path to the input file')
    args = parser.parse_args()
    main(args)

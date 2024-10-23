import data_utils.get_annotation_stats as gs # msql queries
import data_utils.model_utils.dataset as d
from data_utils.model_utils.dataset import qual_label_maps, quant_label_maps
from data_utils.model_utils.dataset import DB_FILENAME
from sklearn.model_selection import KFold
import pickle
import nltk
import json
nltk.download('punkt')


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

def populate_quant_list(qual_dict: dict, quant_dict: dict):
    """
    Populates the quant_list in the qual_dict with the keys from quant_dict.
    
    Args:
        qual_dict (dict): The dictionary containing quality indicators.
        quant_dict (dict): The dictionary containing quant indicators.
    
    Returns:
        dict: The updated qual_dict with the quant_list populated.
    """
    for quant_id in quant_dict.keys():
        article_id, local_id = quant_id.split('_')
        article_id = int(article_id)
        if article_id not in qual_dict.keys():
            qual_dict[article_id] = {}
            qual_dict[article_id]['frame'] = '\x00'
            qual_dict[article_id]['econ_rate'] = '\x00'
            qual_dict[article_id]['econ_change'] = '\x00'
            qual_dict[article_id]['quant_list'] = []

        qual_dict[article_id]['quant_list'].append(quant_id)

    return qual_dict


def populate_quant_text(qual_dict: dict, quant_dict: dict, db_filename: str):
    """
    Populates the 'indicator' and 'excerpt' fields in the quant_dict dictionary
    using the quant_list from the qual_dict dictionary and excerpts from the
    database file specified by db_filename.

    Args:
        qual_dict (dict): A dictionary containing qualitative data.
        quant_dict (dict): A dictionary containing quantitative data.
        db_filename (str): The filename of the database file.

    Returns:
        dict: The updated quant_dict dictionary.
    """
    for id in qual_dict.keys():
        if len(qual_dict[id]['quant_list']) > 0:
            excerpts = d.get_excerpts(qual_dict[id]['quant_list'],
                                      db_filename)
            for id, text in excerpts.items():
                quant_dict[id]['indicator'] = text[0]
                quant_dict[id]['excerpt'] = text[1]
    
    return quant_dict


def get_split_dict(qual_dict: dict):
    """
    Generate a dictionary of train/test splits for a given quality dictionary.

    Args:
        qual_dict (dict): A dictionary containing quality information for articles.

    Returns:
        dict: A dictionary of train/test splits, where each key represents a fold number and the value is a dictionary
              containing the train and test article IDs for that fold.
    """
    split_dict = {}
    article_ids = list(qual_dict.keys())
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    split = kf.split(article_ids)

    # format split dictionary {key = fold number, value = dict of train/test ids}
    for i, (train_index, test_index) in enumerate(split):
        split_dict[i] = {}
        split_dict[i]['train'] = [article_ids[t] for t in train_index]
        split_dict[i]['test'] = [article_ids[t] for t in test_index]

    return split_dict

def add_none(ann_dict: dict, quant=True):

    if quant:
        none_val = 'none'
        type_comp = 'type'
        change_anns = ['macro_type']
    else:
        none_val = 'irrelevant'
        type_comp = 'frame'
        change_anns = ['econ_rate', 'econ_change']
    # print(ann_dict)
    none_ids = []
    for id in ann_dict.keys():
        if ann_dict[id][type_comp] != '\x00':
            if ann_dict[id][type_comp] != 'macro':
                for ann in change_anns:
                    ann_dict[id][ann] = none_val
                    none_ids.append(id)
                

    return ann_dict, none_ids

def remove_nones(ann_dict: dict, none_ids: list):
    for id in none_ids:
        if id in ann_dict.keys():
            ann_dict.pop(id)
    return ann_dict


def sanity_check(agreed_dict: dict, noisy_dict: dict):
    # sanity check
    for id, anns in agreed_dict.items():
        if id in noisy_dict.keys():
            noisy_ann = noisy_dict[id]
            for type in anns.keys():
                if type not in ['indicator', 'excerpt', 'quant_list']:
                    if noisy_ann[type] != '\0' and anns[type] != '\0':
                        print(f"Article {id} has both agreed and noisy annotations for {type}: {anns[type]} and {noisy_ann[type]}")

def none_sanity_check(ann_dict: dict, quant=False):
    if quant:
        none_val = 'none'
        type_comp = 'type'
        change_anns = ['macro_type']
    else:
        none_val = 'irrelevant'
        type_comp = 'frame'
        change_anns = ['econ_rate', 'econ_change']

    for id, anns in ann_dict.items():
        if anns[type_comp] != '\x00':
            if anns[type_comp] == 'macro':
                for ann in change_anns:
                    if anns[ann] == none_val:
                        print(f"Article {id} has a none annotation for {ann}")
            else: 
                for ann in change_anns:
                    if anns[ann] != none_val:
                        print(f"Article {id} has an annotation for {ann} but is macro")



def main():
    """
    Generate train/test splits for econ indicator models. Save the splits and 
    qual/quant dictionaries to models/utils/splits as pickles. 

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        None
    """ 

    db_filename = DB_FILENAME

    # get agreed article-level annotations
    # {key = articleid, value = dict of annotations}
    qual_ann = gs.get_qual_dict(db_filename)
    agreed_qual_ann = gs.get_agreed_anns(qual_ann, qual_label_maps)
    agreed_qual_ann, none_ids = add_none(agreed_qual_ann, quant=False)
    agreed_qual_ann = remove_empty(agreed_qual_ann)

    noisy_qual_ann = gs.get_noisy_anns(qual_ann, qual_label_maps)
    noisy_qual_ann = remove_empty(noisy_qual_ann)
    noisy_qual_ann = remove_nones(noisy_qual_ann, none_ids)

    # TODO: implement this
    noisy_best_qual_ann = gs.get_best_noisy_anns(qual_ann, qual_label_maps, DB_FILENAME)
    noisy_best_qual_ann = remove_empty(noisy_best_qual_ann)
    # noisy_best_qual_ann = remove_nones(noisy_best_qual_ann, none_ids)
    noisy_best_qual_ann, _ = add_none(noisy_best_qual_ann, quant=False)
   

    # get agreed quantitative annotations in dict where 
    # {key = "articleid_localid", value = dict of annotations}
    quant_ann = gs.get_quant_dict(db_filename)
    agreed_quant_ann = gs.get_agreed_anns(quant_ann, quant_label_maps)
    agreed_quant_ann, none_ids = add_none(agreed_quant_ann)
    agreed_quant_ann = remove_empty(agreed_quant_ann)

    noisy_quant_ann = gs.get_noisy_anns(quant_ann, quant_label_maps)
    noisy_quant_ann = remove_empty(noisy_quant_ann)
    noisy_quant_ann = remove_nones(noisy_quant_ann, none_ids)

    noisy_best_quant_ann = gs.get_best_noisy_anns(quant_ann, quant_label_maps, DB_FILENAME, quant=True)
    noisy_best_quant_ann = remove_empty(noisy_best_quant_ann)
    noisy_best_quant_ann, _ = add_none(noisy_best_quant_ann, quant=True)
    
    
    for k in agreed_qual_ann.keys():
        agreed_qual_ann[k]['quant_list'] = []

    for k in noisy_qual_ann.keys():
        noisy_qual_ann[k]['quant_list'] = []

    # print(noisy_best_qual_ann)
    for k in noisy_best_qual_ann.keys():
        noisy_best_qual_ann[k]['quant_list'] = []

    # add quant_ids to agreed_qual_ann dict
    agreed_qual_ann = populate_quant_list(agreed_qual_ann, agreed_quant_ann)
    noisy_qual_ann = populate_quant_list(noisy_qual_ann, noisy_quant_ann)
    noisy_best_qual_ann = populate_quant_list(noisy_best_qual_ann, noisy_best_quant_ann)

    # add text excerpts w/ context to agreed_quant_ann dict
    agreed_quant_ann = populate_quant_text(agreed_qual_ann, agreed_quant_ann, db_filename)
    noisy_quant_ann = populate_quant_text(noisy_qual_ann, noisy_quant_ann, db_filename)
    noisy_best_quant_ann = populate_quant_text(noisy_best_qual_ann, noisy_best_quant_ann, db_filename)

    # save full data dictionaries as pickles
    d.save_progress(agreed_qual_ann, 'data/clean/agreed_qual_dict')
    d.save_progress(agreed_quant_ann, 'data/clean/agreed_quant_dict')
    agreed_qual_ann = {int(k): v for k, v in agreed_qual_ann.items()}
    

    # create splits
    split_dict = get_split_dict(agreed_qual_ann)

    # for k, v in agreed_quant_ann.items():
    #     print(k, v)

    # # print split dictionary
    # for k in split_dict.keys():
    #     print(f"Fold {k}")
    #     print(f"Train: {split_dict[k]['train']}")
    #     # print(f"Test: {split_dict[k]['test']}")
    #     for id in split_dict[k]['train']:
    #         print(id, agreed_qual_ann[id])

    # sanity_check(agreed_qual_ann, noisy_qual_ann)
    # sanity_check(agreed_quant_ann, noisy_quant_ann)
    # sanity_check(agreed_qual_ann, noisy_best_qual_ann)
    # sanity_check(agreed_quant_ann, noisy_best_quant_ann)

    # none_sanity_check(agreed_qual_ann, quant=False)
    # none_sanity_check(agreed_quant_ann, quant=True)

    counts = {}
    counts['frame'] = 0
    counts['econ_rate'] = 0
    counts['econ_change'] = 0
    counts['quant_support'] = 0
    for k, v in agreed_qual_ann.items():
        for ann in ['frame', 'econ_rate', 'econ_change']:
            if v[ann] != '\x00':
                counts[ann] += 1
        if len(v['quant_list']) > 0:
            counts['quant_support'] += 1
    print(counts)

    counts = {}
    counts['type'] = 0
    counts['macro_type'] = 0
    counts['spin'] = 0
    for k, v in agreed_quant_ann.items():
        for ann in ['type', 'macro_type', 'spin']:
            if v[ann] != "\x00":
                counts[ann] += 1
    # print(counts)

    print(len(agreed_qual_ann))
    print(len(agreed_quant_ann))

    # # # save dictionaries as pickles 
    base_dir = 'data/clean/'
    d.save_progress(split_dict, f'{base_dir}splits_dict')
    d.save_progress(agreed_quant_ann, f'{base_dir}quant_dict')
    d.save_progress(agreed_qual_ann, f'{base_dir}qual_dict')

    # save noisy dictionaries as pickles
    d.save_progress(noisy_qual_ann, f'{base_dir}noisy_qual_dict')
    d.save_progress(noisy_quant_ann, f'{base_dir}noisy_quant_dict')

    d.save_progress(noisy_best_qual_ann, f'{base_dir}noisy_best_qual_dict')
    d.save_progress(noisy_best_quant_ann, f'{base_dir}noisy_best_quant_dict')

    # add text to qual dict
    for id in agreed_qual_ann.keys():
        agreed_qual_ann[id]['text'] = gs.get_text(id, db_filename, clean=True, headline=True)

    json.dump(
        agreed_qual_ann, 
        open('data/clean/agreed_qual_dict.json', 'w'),
        indent=4
    )
    json.dump(
        agreed_quant_ann, 
        open('data/clean/agreed_quant_dict.json', 'w'),
        indent=4
    )


if __name__ == '__main__':
    main()

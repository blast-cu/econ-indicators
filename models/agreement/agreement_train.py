from models.psl.generate_data import load_train_test_data
import models.agreement.agreement_utils as au
import models.roberta_classifier.train_test_utils as tt
import data_utils.dataset as d

import pickle
import os

DATA_DIR = "data/clean/"
OUT_DIR = 'models/agreement/tuned_models/'
    
label_maps = {
    'type': {
            'macro': 0,
            'industry': 1,
            'government': 2,
            'personal': 3,
            'business': 4,
            'other': 5},
    'spin': {
            'pos': 0,
            'neg': 1,
            'neutral': 2},
    'macro_type': {
            'jobs': 0,
            'retail': 1,
            'interest': 2,
            'prices': 3,
            'energy': 4,
            'wages': 5,
            'macro': 6,
            'market': 7,
            'currency': 8,
            'housing': 9,
            'other': 10,
            'none': 11}
}

agreement_map = {
    'agree': 1,
    'disagree': 0
}

def get_neighbors(articles):


    neighbors = []
    for article_id, article_dict in articles.items():
        excerpt_ids = article_dict['quant_list']
        for i, excerpt_id in enumerate(excerpt_ids):
            if i == 0:
                continue
            else: 
                prev = excerpt_ids[i-1].split('_')
                curr = excerpt_id.split('_')
                if prev[0] == curr[0]:  # same article
                    if int(prev[1]) + 1 == int(curr[1]):
                        temp = [excerpt_ids[i-1], excerpt_id]
                        neighbors.append(temp)

    return neighbors

def get_texts(
              annotation_component: str,
              quant_dict: {},
              neighbors: [[]],
              ):
    """
    Retrieves texts and labels for split from the given dictionaries for 
    the given task.

    Args:
        annotation_component (str): The annotation component to retrieve labels from.
        task (str): The task to perform.
        qual_dict (dict): Dictionary of article-level anns.
        quant_dict (dict): Dictionary of quant annotations.
        article_ids (list): The list of article IDs to retrieve texts and
                labels from.
        type_filter (list, optional): The list of ann types to filter the entries.
                Defaults to an empty list (no filter).

    Returns:
        tuple: A tuple containing the retrieved texts and labels.
            - texts (list): [indicator text, text with context]
            - labels (list): The list of labels for the given text
    """
    
    labels = []
    neighbor_texts = []

    for id1, id2 in neighbors:

        if quant_dict[id1][annotation_component] == '\x00' or \
            quant_dict[id2][annotation_component] == '\x00':
            continue

        else:

            neighbor_text = []
            indicator_text = quant_dict[id1]['indicator']
            excerpt_text = quant_dict[id1]['excerpt']

            if indicator_text not in excerpt_text:
                print("ERROR: indicator not in excerpt 1")

            text = [indicator_text, excerpt_text]
            neighbor_text.append(text)

            indicator_text = quant_dict[id2]['indicator']
            excerpt_text = quant_dict[id2]['excerpt']
            text = [indicator_text, excerpt_text]
            neighbor_text.append(text)

            if indicator_text not in excerpt_text:
                print("ERROR: indicator not in excerpt 2")

            if quant_dict[id1][annotation_component] == \
                quant_dict[id2][annotation_component]:
                label = 1
            else:
                label = 0
            
            neighbor_texts.append(neighbor_text)
            labels.append(label)

    return neighbor_texts, labels


def main():

    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file.
    """
    # model_checkpoint = 'models/roberta_classifier/tuned_models/masked'
    model_checkpoint = 'data/masked/'
    
    splits_dict = pickle.load(open(DATA_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(DATA_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(DATA_DIR + 'quant_dict_clean', 'rb'))

    # dict for tracking results across folds
    results = {}
    for task in label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for split_num in range(5):

        print("Fold " + str(split_num) + " of 4")
        print()


        train_articles, train_excerpts, test_articles, test_excerpts = \
            load_train_test_data(splits_dict[split_num],
                                 qual_dict,
                                 quant_dict)
        
        train_neighbors_raw = get_neighbors(train_articles)
        test_neighbors_raw = get_neighbors(test_articles)


        for task in list(label_maps.keys()):

            print("Training on task: " + task)
        
            ann_component = task.split('-')[0]

            train_neighbors, train_labels = \
                get_texts(ann_component,
                          quant_dict,
                          train_neighbors_raw)

            test_neighbors, test_labels = \
                get_texts(ann_component,
                          quant_dict,
                          test_neighbors_raw)
            
            class_weights = tt.get_weights(train_labels,
                                           agreement_map)
            
            # print(class_weights)


            model, train_loader, val_loader, test_loader, optimizer = \
                au.setup(train_neighbors,
                         test_neighbors,
                         train_labels,
                         test_labels,
                         model_checkpoint=model_checkpoint)

            tuned_model = au.train(model,
                                   train_loader,
                                   val_loader,
                                   optimizer,
                                   class_weights)
            
            y, y_predicted, f1 = au.test(tuned_model,
                                         test_loader)

            results[task]['labels'] += y
            results[task]['predictions'] += y_predicted

            print(">>> Labels:\t" + str(y))
            print(">>> Predictions:\t" + str(y_predicted))
            print('\n\n')

            dest = os.path.join(OUT_DIR, f"fold{split_num}/{task}_agreement_model/")
            os.makedirs(dest, exist_ok=True)

            d.to_csv(
                task,
                y,
                y_predicted,
                dest)

            tuned_model.save(dest, task)

    for task in label_maps.keys():
        dest = os.path.join(OUT_DIR, "results")
        os.makedirs(dest, exist_ok=True)
        name = f'{task}_agreement'
        d.to_csv(name,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)


if __name__ == "__main__":
    main()
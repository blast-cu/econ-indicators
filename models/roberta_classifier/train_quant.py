from torch import nn

import argparse
import pickle
import os

import models.roberta_classifier.train_test_utils as tt
import models.utils.dataset as d

# nltk.download('punkt')

OUT_DIR = "models/roberta_classifier/tuned_models/quant_edits/"
SPLIT_DIR = "data/clean/"

# maps annotation labels to integers for each prediction task
label_maps = {
    # 'type': {
    #         'macro': 0,
    #         'industry': 1,
    #         'government': 2,
    #         'personal': 3,
    #         'business': 4,
    #         'other': 5},
    'type-binary': {
            'macro': 0,
            'industry': 1,
            'government': 1,
            'personal': 1,
            'business': 1,
            'other': 1}
    # 'spin': {
    #         'pos': 0,
    #         'neg': 1,
    #         'neutral': 2}, 
    # 'macro_type': {
    #         'jobs': 0,
    #         'retail': 1,
    #         'interest': 2,
    #         'prices': 3,
    #         'energy': 4,
    #         'wages': 5,
    #         'macro': 6,
    #         'market': 7,
    #         'currency': 8,
    #         'housing': 9,
    #         'other': 10}
}

# class QuantModel(nn.Module):
#     def __init__(self):
#         super(QuantModel, self).__init__()

#         self.linear1 = nn.Linear(100, 200)
#         self.activation = nn.ReLU()
#         self.linear2 = nn.Linear(200, 10)
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.activation(x)
#         x = self.linear2(x)
#         x = self.softmax(x)
#         return x


def get_texts(
              annotation_component: str,
              task: str,
              qual_dict: {},
              quant_dict: {},
              article_ids: [],
              type_filter: [] = []
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
    
    texts = []
    labels = []

    for id in article_ids:
        if 'quant_list' in qual_dict[id].keys():
            for quant_id in qual_dict[id]['quant_list']:
                if quant_dict[quant_id][annotation_component] != '\x00':
                    valid_entry = False
                    if len(type_filter) == 0:
                        valid_entry = True
                    elif 'type' in quant_dict[quant_id].keys():
                        if quant_dict[quant_id]['type'] in type_filter:
                            valid_entry = True
                    
                    if valid_entry:
                        indicator_text = quant_dict[quant_id]['indicator']
                        excerpt_text = quant_dict[quant_id]['excerpt']
                        text = [indicator_text, excerpt_text]
                        texts.append(text)

                        label = quant_dict[quant_id][annotation_component]
                        labels.append(label_maps[task][label])

    return texts, labels


def qual_train():
    """
    Train a model on qualitative annotations
    """
    return


def main(args):
    """
    Performs k-fold cross-validation for a set of classification tasks on
    quantitative annotations, trains a model for each fold, and saves the
    results to a CSV file.
    """

    model_checkpoint = args.model
  
    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))
    quant_dict = pickle.load(open(SPLIT_DIR + 'quant_dict', 'rb'))

    type_filters = {
        'type': ['industry', 'macro'],
        'type-binary': [],
        'spin': ['industry', 'macro'],
        'macro_type': ['macro']
    }

    # dict for tracking results across folds
    results = {}
    for task in label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []

    for k, split in splits_dict.items():

        print("Fold " + str(k+1) + " of 5")
        print()

        split_train_ids = split['train']
        split_test_ids = split['test']

        for task in list(label_maps.keys()):

            ann_component = task.split('-')[0]

            train_texts, train_labels = \
                get_texts(ann_component,
                          task,
                          qual_dict,
                          quant_dict,
                          split_train_ids,
                          type_filter=type_filters[task])

            test_texts, test_labels = \
                get_texts(ann_component,
                          task,
                          qual_dict,
                          quant_dict,
                          split_test_ids,
                          type_filter=type_filters[task])

            # gets class weights for loss function
            class_weights = tt.get_weights(train_labels,
                                           label_maps[task])

            model, train_loader, val_loader, test_loader, optimizer = \
                tt.setup(train_texts,
                         test_texts,
                         train_labels,
                         test_labels,
                         label_maps[task],
                         model_checkpoint=model_checkpoint)

            tuned_model = tt.train(model,
                                   train_loader,
                                   val_loader,
                                   optimizer,
                                   class_weights)

            y, y_predicted, f1 = tt.test(tuned_model,
                                         test_loader)

            results[task]['labels'] += y
            results[task]['predictions'] += y_predicted

            print(">>> Labels:\t" + str(y))
            print(">>> Predictions:\t" + str(y_predicted))
            print('\n\n')

            dest = os.path.join(OUT_DIR, "fold{k}/quant/")
            os.makedirs(dest, exist_ok=True)

            d.to_csv(
                task,
                y,
                y_predicted,
                dest)

            model_dest = dest + task + "_model"
            model.save_pretrained(model_dest)

    for task in label_maps.keys():
        dest = os.path.join(OUT_DIR, "quant_results/")
        os.makedirs(dest, exist_ok=True)

        d.to_csv(task,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to database")
    parser.add_argument("--model", required=False, default="roberta-base", help="model checkpoint")
    args = parser.parse_args()
    main(args)

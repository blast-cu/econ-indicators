import pickle
import argparse
from torch.utils.data import DataLoader, Dataset

from models.roberta_classifier.train_quant import label_maps, get_texts
from models.roberta_classifier.quant_utils import QuantModel, TextClassificationDataset
import models.roberta_classifier.train_test_utils as tt

SPLIT_DIR = "data/clean/"

def main(args):

    splits_dict = pickle.load(open(SPLIT_DIR + 'splits_dict', 'rb'))
    qual_dict = pickle.load(open(SPLIT_DIR + 'qual_dict', 'rb'))

    results = {}
    for task in label_maps.keys():
        results[task] = {}
        results[task]['labels'] = []
        results[task]['predictions'] = []# Your code here

    for k, split in splits_dict.items():
        print("Fold " + str(k+1) + " of 5")
        print()
        
        split_test_ids = split['test']

        for task in list(label_maps.keys()):
            annotation_component = task.split('-')[0]
            test_texts, test_labels = \
                get_texts(args.db,
                          annotation_component,
                          task,
                          qual_dict,
                          split_test_ids)
        
            test_texts = [t.replace('\n', '') for t in test_texts]
            tokenizer = RobertaTokenizerFast\
                        .from_pretrained(pretrained_model_name_or_path=model_checkpoint,
                                        problem_type="single_label_classification")
            max_length = 512
            test_data = TextClassificationDataset(texts=test_texts,
                                                  labels=test_labels,
                                                  tokenizer=tokenizer,
                                                  max_length=max_length)

            batch_size = 8
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

            # Define model
            num_labels = len(set(annotation_map.values()))
            model = QuantModel(model_checkpoint, num_labels=num_labels).to('cuda')

            results[task]['labels'] += y
            results[task]['predictions'] += y_predicted
        
    for task in label_maps.keys():
        dest = f"models/roberta_classifier/tuned_models/roberta_base_unfiltered/results/"

        os.makedirs(dest, exist_ok=True)

        d.to_csv(task,
                 results[task]['labels'],
                 results[task]['predictions'],
                 dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="path to db file")
    parser.add_argument("--model", required=False, default="roberta-base", help="model checkpoint")
    args = parser.parse_args()
    main(args)

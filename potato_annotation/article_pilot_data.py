import pickle
import random
import pandas as pd
import data_utils.inter_annotator_agreement as iaa
from data_utils.model_utils.dataset import DB_FILENAME, qual_label_maps
import data_utils.get_annotation_stats as gs


def main():
    OUTPUT_DIR = "potato_annotation/article_annotate/data_files"
    random.seed(42)
    qual_dict = pickle.load(open('data/clean/qual_dict', 'rb'))
    all_anns, _ = iaa.get_anns(DB_FILENAME)

    bin_choices = {}
    
    for article_id, anns in qual_dict.items():
        if anns['frame'] != '\x00':
            if len(all_anns[article_id]['frame']) == 3:
                frame_val = anns['frame']
                if frame_val == 'macro':
                    rate_val = anns['econ_rate']
                    change_val = anns['econ_change']
                    if len(all_anns[article_id]['econ_rate']) == 3:
                        if rate_val != '\x00':
                            if len(all_anns[article_id]['econ_change']) == 3:
                                if change_val != '\x00':
                                    if 'macro' not in bin_choices:
                                        bin_choices['macro'] = []
                                    bin_choices['macro'].append(article_id)
                else:
                    if frame_val in qual_label_maps['frame']:
                        if frame_val not in bin_choices:
                            bin_choices[frame_val] = []
                        bin_choices[frame_val].append(article_id)

    bin_counts = {}
    article_choices = []
    NUM_ARTICLES = 25
    while len(article_choices) < NUM_ARTICLES:
        for k, v in bin_choices.items():
            if k not in bin_counts:
                bin_counts[k] = 0
            if len(v) > 0:
                new_id = random.choice(v)
                bin_choices[k].remove(new_id)
                article_choices.append(new_id)
                bin_counts[k] += 1
    for k, v in bin_counts.items():
        print(k, v)

    articles = {}
    for id in article_choices:
        text = gs.get_text(id, db_filename=DB_FILENAME, clean=False, headline=True)
        articles[id] = text

    article_csv_dict = {}
    article_csv_dict['id'] = list(articles.keys())
    article_csv_dict['text'] = list(articles.values())

    df = pd.DataFrame(article_csv_dict)
    df.to_csv(OUTPUT_DIR + f'/articles_pilot.csv', index=False)

    quant_csv_dict = {}
    quant_csv_dict['id'] = []
    quant_csv_dict['text'] = []


    for k, v in bin_choices.items():
        for id in v:
            text = gs.get_text(id, db_filename=DB_FILENAME, clean=True, headline=True)
            text[0] = text[0].replace('"', "'")
            text[1] = text[1].replace('"', "'")
            print(qual_dict[id])
            print(text)
            print()

if __name__ == "__main__":
    main()
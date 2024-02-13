import pickle
import re
import pandas as pd
import random

OUTPUT_DIR = "potato_annotation/quant_annotate/data_files"
NUM_EXCERPTS = 45

def main():

    quant_excerpts_dict = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))
    clean = pickle.load(open('data/clean/quant_dict', 'rb'))

    ids = []
    texts = []

    bin_choices = {}
    for id, ann in clean.items():
        ann_type = ann['type']
        if ann_type != '\x00':
            if ann_type not in bin_choices:
                bin_choices[ann_type] = []
            
            if ann_type == 'macro':
                if ann['macro_type'] != '\x00':
                    if ann['spin'] != '\x00':
                        bin_choices[ann_type].append(id)
            else:
                bin_choices[ann_type].append(id)
    # for b in bin_choices:
    #     print(b, len(bin_choices[b]))
    

    bin_counts = {}
    
    while len(ids) < NUM_EXCERPTS:
        for k, v in bin_choices.items():
            if k not in bin_counts:
                bin_counts[k] = 0
            if len(v) > 0:
                new_id = random.choice(v)
                bin_choices[k].remove(new_id)
                ann = clean[new_id]
                indicator = ann['indicator']
                excerpt = ann['excerpt']

                indicator = re.escape(indicator)

                if indicator in excerpt:
                    print("Indicator found in excerpt")

                try: 
                    match_iter = list(re.finditer(indicator, excerpt))
                    match = list(match_iter)[0]
                except Exception:
                    print(f"ERROR: No match found for indicator {id} in iteration {i}")
                    print(">>> Match Iterator: " + str(match_iter))
                    print(">>> Indicator: " + indicator)
                    print(">>> Excerpt: " + excerpt)
                    print('\n\n')
                    continue

                start = match.start()
                end = match.end()

                excerpt = excerpt[:start] + "<span>" + excerpt[start:end] + "</span>" + excerpt[end:]

                ids.append(id)
                texts.append(excerpt)
                bin_counts[k] += 1
    
    for k, v in bin_counts.items():
        print(k, v)

        

    csv_dict = {}
    csv_dict['id'] = ids
    csv_dict['text'] = texts

    df = pd.DataFrame(csv_dict)
    df.to_csv(OUTPUT_DIR + '/quants.csv', index=False)


if __name__ == "__main__":
    main()

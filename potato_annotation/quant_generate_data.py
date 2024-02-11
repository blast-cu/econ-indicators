import pickle
import re
import pandas as pd

OUTPUT_DIR = "potato-annotation/quant_annotate/data_files"

def main():

    quant_excerpts_dict = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))
    clean = pickle.load(open('data/clean/quant_dict_clean', 'rb'))

    ids = []
    texts = []

    for i in range(10):
        id, val = clean.popitem()
        indicator = val['indicator']
        excerpt = val['excerpt']

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

    csv_dict = {}
    csv_dict['id'] = ids
    csv_dict['text'] = texts

    df = pd.DataFrame(csv_dict)
    df.to_csv(OUTPUT_DIR + '/quants.csv', index=False)


if __name__ == "__main__":
    main()

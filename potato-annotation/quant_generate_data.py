import pickle
import re
import pandas as pd

OUTPUT_DIR = "potato-annotation/quant_annotate/data_files"

def main():

    quant_excerpts_dict = pickle.load(open('data/clean/quant_excerpts_dict', 'rb'))

    ids = []
    texts = []

    for i in range(10):
        id, val = quant_excerpts_dict.popitem()
        indicator = val['indicator']
        excerpt = val['excerpt']

        match = list(re.finditer(indicator, excerpt))[0]
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



    # for article_id, excerpts in quant_excerpts_dict.items():
    #     print(article_id)
    #     print(excerpts)
    #     print()
    #     exit()


if __name__ == "__main__":
    main()

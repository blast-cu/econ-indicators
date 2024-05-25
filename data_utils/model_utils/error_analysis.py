import pickle
import os
import sqlite3
from data_utils.model_utils.dataset import qual_label_maps, quant_label_maps, SPLIT_DIR, DB_FILENAME
from data_utils.get_annotation_stats import get_text

def get_site(db, quant_id):
    con = sqlite3.connect(db)
    cur = con.cursor()
    query = 'select source from article where id = {}'.format(quant_id)
    res = cur.execute(query)
    site = res.fetchone()[0]
    return site

def main():

    # start with qual, split 0
    qual_results_dict = pickle.load(
        open(
            os.path.join(SPLIT_DIR, 'best_qual_results'),
            'rb'
        )
    )

    with open('data/error_analysis.html', 'w') as f:
        for annotation_type in qual_label_maps.keys():
            f.write(f'<h1>{annotation_type}</h1><br>')

            curr_results = qual_results_dict[annotation_type]

            curr_results['source'] = [  # get site for each id
                get_site(DB_FILENAME, id) for id in curr_results['ids']
            ]
            sites = list(set(curr_results['source']))
            sites.sort()

            for site in sites:
                site_indices = [
                    i for i, s in enumerate(curr_results['source']) if s == site
                ]

                # site_results = {
                #     'text': [
                #         get_text(['ids'][i]) for i in site_indices
                #     ],
                #     'labels': [
                #         qual_results_dict['labels'][i] for i in site_indices
                #     ],
                #     'predictions': [
                #         qual_results_dict['predictions'][i] for i in site_indices
                #     ]
                # }

                # per_site_results[site] = site_results

                f.write(f'<h2>{site}</h2><br>')
                for idx in site_indices:
                    label = curr_results['labels'][idx]
                    prediction = curr_results['predictions'][idx]
                    # if label != prediction:
                    id = curr_results['ids'][idx]
                    text = get_text(id, db_filename=DB_FILENAME, clean=False, headline=True)
                    headline = text[0].replace("\n", "")
                    headline = "<h4>" + headline + "</h4>"
                    id = "<p>Id: " + str(id) + "</p>"
                    body = text[1].replace("\n", "<br>")
                    body = body.replace(headline, "")
                    body = f'<p>{body}</p>'
                    f.write(headline + id + body)
                    f.write(f'<p>Label: {label}</p>')
                    f.write(f'<p>Prediction: {prediction}</p><br>')


if __name__ == "__main__":
    main()
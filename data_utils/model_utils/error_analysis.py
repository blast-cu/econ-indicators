import pickle
import os
import sqlite3
from data_utils.model_utils.dataset import qual_label_maps, quant_label_maps, SPLIT_DIR, DB_FILENAME
from data_utils.get_annotation_stats import get_text
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

quant_dict = pickle.load(open(os.path.join(SPLIT_DIR, 'quant_dict'), 'rb'))

def get_site(db, article_id):
    con = sqlite3.connect(db)
    cur = con.cursor()
    query = 'select source from article where id = {}'.format(article_id)
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

    quant_results_dict = pickle.load(
        open(
            os.path.join(SPLIT_DIR, 'best_quant_results'),
            'rb'
        )
    )

    wsj_results = {}

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

                # for wsj confusion matrix
                wsj_results[annotation_type] = {}
                wsj_results[annotation_type]['labels'] = [curr_results['labels'][i] for i in site_indices]
                wsj_results[annotation_type]['predictions'] = [curr_results['predictions'][i] for i in site_indices]

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

        for annotation_type in quant_label_maps.keys():
            f.write(f'<h1>{annotation_type}</h1><br>')

            curr_results = quant_results_dict[annotation_type]
            curr_results['article_ids'] = [id.split("_")[0] for id in curr_results['ids']]

            curr_results['source'] = [  # get site for each id
                get_site(DB_FILENAME, id) for id in curr_results['article_ids']
            ]
            sites = list(set(curr_results['source']))
            sites.sort()

            for site in sites:
                site_indices = [
                    i for i, s in enumerate(curr_results['source']) if s == site
                ]

                f.write(f'<h2>{site}</h2><br>')
                for idx in site_indices:
                    label = curr_results['labels'][idx]
                    prediction = curr_results['predictions'][idx]
                    # if label != prediction:
                    id = curr_results['ids'][idx]
                    excerpt = quant_dict[id]['excerpt']
                    indicator = quant_dict[id]['indicator']
                    body = f'<p>Excerpt: {excerpt}</p>'
                    body += f'<p>Indicator: {indicator}</p>'
                    f.write(body)
                    f.write(f'<p>Label: {label}</p>')
                    f.write(f'<p>Prediction: {prediction}</p><br>')
    # print(wsj_results)

    classes = list(qual_label_maps['frame'].keys())
    print(classes)
    y_true = wsj_results['frame']['labels']
    y_true = [qual_label_maps['frame'][label] for label in y_true]

    y_pred = wsj_results['frame']['predictions']
    y_pred = [qual_label_maps['frame'][label] for label in y_pred]
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    normalize = False
    cmap = plt.cm.Blues
    title = 'WSJ Article-Level Type Prediction Errors'
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('data/wsj_frame_errors.png', dpi=300)
    
    

if __name__ == "__main__":
    main()
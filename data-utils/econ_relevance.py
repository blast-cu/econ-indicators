import argparse
import json
import wikipedia
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def main(args):
    print(wikipedia.search("Economy of the United States"))
    article = wikipedia.page("Economy of the United States")
    #print(article.content)

    dataset = json.load(open(args.feature_file))
    X = []
    for id in range(0, len(dataset)):
        text = dataset[str(id)]['text']
        X.append(text)

    X.append(article.content)

    vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)

    X = vectorizer.fit_transform(X).toarray()

    wikipedia_index = len(X) - 1
    wikipedia_vector = X[wikipedia_index]

    ret = []
    for elem in dataset:
        vector = X[int(elem)]
        cosine = np.dot(vector, wikipedia_vector) / (norm(vector)*norm(wikipedia_vector))
        dataset[str(elem)]['relevance'] = cosine
        ret.append((cosine, dataset[elem]['headline']))

    ret.sort()

    '''
    for (cos, headline) in ret:
        print(cos, headline)
    '''

    with open(args.feature_file, "w") as fp:
        json.dump(dataset, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', type=str, required=True)
    args = parser.parse_args()
    main(args)

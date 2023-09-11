import os
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from data_utils import *
from article import Article


def source_count_fig(source_counts, fig_name):
    x = []; y = []
    for key, value in source_counts.most_common(20):
        x.append(key)
        y.append(source_counts[key])

    plt.figure(layout='constrained')
    print('############')
    print(fig_name)
    print(x)
    print(y)
    print('############')
    ax = sns.barplot(x=x, y=y)
    plt.xticks(rotation=90)
    plt.savefig('out/{}.png'.format(fig_name))

def dump_non_econ_headlines(articles, dataset):
    with open('out/not_economy.csv', 'w') as fp:
        all_headlines = []
        writer = csv.writer(fp)
        writer.writerow(['source', 'id', 'headline', 'init-headline'])
        for art in articles:
            if not art.is_econ:
                writer.writerow([art.source, art.id, art.headline, dataset[art.id]['headline']])
                all_headlines.append(dataset[art.id]['headline'])
        print(Counter(all_headlines).most_common(50))

def main(args):
    if args.parse_data:
        articles = parse_data(args.datadir)
        with open(os.path.join(args.datadir, 'preprocessed_data.json'), 'w') as fp:
            dictionary = articles_to_json(articles)
            json.dump(dictionary, fp)
    elif args.load_json:
        dictionary = json.load(open(os.path.join(args.datadir, 'preprocessed_data.json')))
        articles = [Article.from_json(dictionary[id]) for id in dictionary]
    else:
        print('One of two options --[parse_data|load_json] need to be passed')
        exit(-1)


    # remove duplicates
    bad_headlines = set(['Access Denied', 'Wayback Machine'])
    seen_text = set(); unseen_articles = []
    counted_articles = set()
    skipped_articles_access = []; skipped_articles_stemmed = []; skipped_articles_repetition = []
    for a in articles:
        if a.text not in seen_text and dictionary[a.id]['headline'] not in bad_headlines and 'page not found' not in a.text.lower() and\
            not (a.source == 'wsj' and (a.text.endswith('...') or a.text.endswith('Continue reading your article with\na WSJ membership'))):
            unseen_articles.append(a)
            counted_articles.add(a.id)
            seen_text.add(a.text)
        elif dictionary[a.id]['headline'] in bad_headlines or 'page not found' in a.text.lower():
            skipped_articles_access.append(a)
        elif a.source == 'wsj' and (a.text.endswith('...') or a.text.endswith('Continue reading your article with\na WSJ membership')):
            skipped_articles_stemmed.append(a)
        elif a.text in seen_text:
            skipped_articles_repetition.append(a)


    source_counts = Counter([a.source for a in unseen_articles])
    source_count_fig(source_counts, 'good_articles')

    source_counts = Counter([a.source for a in skipped_articles_access])
    source_count_fig(source_counts, 'skipped_articles_access')

    source_counts = Counter([a.source for a in skipped_articles_stemmed])
    source_count_fig(source_counts, 'skipped_articles_stemmed')

    source_counts = Counter([a.source for a in skipped_articles_repetition])
    source_count_fig(source_counts, 'skipped_articles_repetition')

    # Counting top-most sources if keyword is present
    source_counts = Counter([a.source for a in unseen_articles if a.is_econ])
    source_count_fig(source_counts, 'top_20_sources_econ')

    # Counting top-most sources if at least 3 mentions to keywords
    source_counts = Counter([a.source for a in unseen_articles if a.num_keywords >= 3])
    source_count_fig(source_counts, 'top_20_sources_econ_3_econ_sentences')

    for a in articles:
        if args.other_errors and a.source not in ['wsj', 'bbc', 'pbsnewshour'] and a.id not in counted_articles and a.text not in seen_text:
            print(a.id, "\t", a.source, "\t", a.url, '\t', dictionary[a.id]['headline'])
        if args.wsj_errors and a.source == 'wsj' and a.id not in counted_articles and a.text not in seen_text:
            if dictionary[a.id]['headline'] in bad_headlines or 'page not found' in a.text.lower():
                reason = "denied/not-found"
            elif a.source == 'wsj' and (a.text.endswith('...') or a.text.endswith('Continue reading your article with\na WSJ membership')):
                reason = "stemmed"
            else:
                reason = "unknown"
            print(a.id, "\t", a.source, "\t", a.url, '\t', reason)
    #dump_non_econ_headlines(unseen_articles, dictionary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--parse_data', action='store_true', default=False)
    parser.add_argument('--load_json', action='store_true', default=False)
    parser.add_argument('--wsj_errors', action='store_true', default=False)
    parser.add_argument('--other_errors', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

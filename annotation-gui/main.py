# -*- coding: utf-8 -*-
from __main__ import db, article_embed, article_features
from flask import Blueprint, render_template, request, flash, redirect, url_for, session, current_app
from sqlalchemy import desc, sql, func, case
from flask_login import login_required, current_user

# Other libs
import argparse
import datetime
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import umap
from scipy.sparse import csr_matrix
import ast
import os
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing  # to normalise existing X
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
from tqdm.auto import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
from wordcloud import WordCloud
from scipy.spatial import KDTree
from sentence_transformers import SentenceTransformer, util
import torch
import json
from numpyencode import EncodeFromNumpy, DecodeToNumpy
import time
from sklearn.manifold import TSNE
import seaborn as sns
import torch.nn.functional as F
from collections import Counter
from nltk.corpus import stopwords
import hdbscan
import copy
import nltk
import re
from bertopic import BERTopic

from forms import ClusterMethodForm, DataForm, SymbolEditForm, TopicForm, EditTopicForm, ArticleForm, QuantityEditForm
from models import Article, Topic, Cluster, topics, Quantity, Articleann, Quantityann, User, clusters, articles

main = Blueprint('main', __name__)

@main.before_app_first_request
def create_tables():
    #db.drop_all()
    db.create_all()

    global article_embed
    #global article_text
    global article_features

    article_embed = np.load(os.path.join(current_app.config['UPLOAD_FOLDER'], 'article_embed.npy'))
    #article_text = np.load(os.path.join(current_app.config['UPLOAD_FOLDER'], 'article_text.npy'))
    article_features = json.load(open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'article_features.json')))

@login_required
@main.route('/clustering', methods=['GET', 'POST'])
def clustering():
    pass

@login_required
@main.route('/', methods=['GET', 'POST'])
def index():
    if current_user.is_authenticated:
        return render_template('index.html', name=current_user.email)
    else:
        return redirect(url_for('auth.login'))

@login_required
@main.route('/recluster', methods=['GET', 'POST'])
def recluster():
    if not current_user.is_authenticated or not current_user.email == "maria.pacheco@colorado.edu":
        return redirect(url_for('main.index'))

    form = ClusterMethodForm()
    print(request.form)
    global K
    if form.validate_on_submit() and 'restart' in request.form:
        # Restart cluster assignments by:
        # 1) Set all article-cluster assignments to NULL
        #curr_articles = Article.query.all()
        #for art in curr_articles:
        #    art.cluster_id = sql.null()
        # 2) Delete cluster table
        Cluster.query.delete()
        db.session.commit()

        K = form.k.data

        if form.method.data == 'kmeans':
            return redirect(url_for('main.loading'))
        elif form.method.data == 'hdbscan':
            return redirect(url_for('main.loadinghdbscan'))
        elif form.method.data == 'bertopic':
            return redirect(url_for('main.loadingbertopic'))
    elif form.validate_on_submit() and 'assign' in request.form:
        return redirect(url_for('main.loadingassignment'))

    return render_template('recluster.html', form=form)

@login_required
@main.route('/loadingassignment', methods=['GET', 'POST'])
def loadingassignment():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    return render_template('loadingassignment.html')

@login_required
@main.route('/loading', methods=['GET', 'POST'])
def loading():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    return render_template("loading.html")

@login_required
@main.route('/loadinghdbscan', methods=['GET', 'POST'])
def loadinghdbscan():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    return render_template('loadinghdbscan.html')

@login_required
@main.route('/loadingbertopic', methods=['GET', 'POST'])
def loadingbertopic():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    return render_template('loadingbertopic.html')

def _update_articles():
    article_ids = list(range(0, article_embed.shape[0]))
    pbar = tqdm(total=len(article_ids), desc='Updating articles')
    for article_id in article_ids:
        art = Article.query.get(article_id)

        if art is None:
            if 'T' in article_features[str(article_id)]['date']:
                date_str = article_features[str(article_id)]['date'].split('T')[0]
            else:
                date_str = article_features[str(article_id)]['date'].split()[0]

            art = Article(id=article_id, headline=article_features[str(article_id)]['headline'],
                          keywords=",".join(article_features[str(article_id)]['keywords']),
                          num_keywords=article_features[str(article_id)]['num_keywords'],
                          relevance=article_features[str(article_id)]['relevance'],
                          source=article_features[str(article_id)]['source'].replace('www.', '').replace('.com', ''),
                          text=article_features[str(article_id)]['text'], cluster=None, distance=0.0,
                          date=datetime.datetime.strptime(date_str, '%Y-%m-%d').date(),
                          url=article_features[str(article_id)]['url'])
            db.session.add(art)

            for quant_id in article_features[str(article_id)]['quants']:
                unique_id = "{}_{}".format(article_id, quant_id)
                quant = Quantity(id=unique_id, local_id=quant_id, article=art)
                db.session.add(quant)

        pbar.update(1)
    pbar.close()
    db.session.commit()

def _update_annotations(user_id):
    article_ids = list(range(0, article_embed.shape[0]))
    pbar = tqdm(total=len(article_ids), desc='Updating annotations')
    for article_id in article_ids:
        art_ann = Articleann.query.filter_by(article_id=article_id, user_id=user_id).first()

        # Update annotation if DB has been created before
        if art_ann is None:
            art = Article.query.get(article_id)

            # Create the annotation record for the current user
            art_ann = Articleann(user_id=user_id, article_id=article_id, text=art.text)
            db.session.add(art_ann)

            for quant in art.quantities:
                quant_ann = Quantityann.query.filter_by(quantity_id=quant.id, user_id=user_id).first()
                if quant_ann is None:
                    quant_ann = Quantityann(user_id=user_id, quantity_id=quant.id)
                    db.session.add(quant_ann)

        pbar.update(1)
    pbar.close()
    db.session.commit()

@login_required
@main.route('/kmeans', methods=['POST', 'GET'])
def kmeans():
    print(current_user)
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    # Potentially we would like to further reduce the IDs we do this on in the future
    # (E.g. iterative interactions)

    # Filter IDs that pass threshold
    curr_articles = Article.query.filter(Article.relevance >= 0.15)
    article_ids = [art.id for art in curr_articles]

    #article_ids = list(range(0, article_embed.shape[0]))

    X_Norm = preprocessing.normalize(article_embed[article_ids])

    kmeans = KMeans(n_clusters=K, random_state=0, verbose=True).fit(X_Norm)
    # Create clusters if they don't exist
    for cluster_idx in range(0, K):
        cluster_name = "KMeans_{}".format(cluster_idx)
        exists = db.session.query(Cluster.id).filter_by(name=cluster_name).first() is not None
        if not exists:
            clstr = Cluster(name=cluster_name)
            db.session.add(clstr)
    db.session.commit()

    centroids = kmeans.cluster_centers_

    pbar = tqdm(total=len(article_ids), desc='calculating distances and storing articles')
    for j, i in zip(article_ids, range(0, len(article_ids))):
        cluster_idx = kmeans.labels_[i]
        cluster_name = "KMeans_{}".format(cluster_idx)
        cluster = Cluster.query.filter_by(name=cluster_name).first()

        distance_to_centroid = distance.cosine(article_embed[j], centroids[cluster_idx])
        distance_to_centroid = round(distance_to_centroid,4)
        art = Article.query.filter_by(id=j).first()

        art.cluster=cluster
        art.distance=distance_to_centroid

        pbar.update(1)

    pbar.close()

    db.session.commit()
    return redirect(url_for('main.data'))

@login_required
@main.route('/bertopic', methods=['POST', 'GET'])
def run_bertopic():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    # Potentially we would like to further reduce the IDs we do this on in the future
    # (E.g. iterative interactions)

    # Filter IDs that pass threshold
    curr_articles = Article.query.filter(Article.relevance >= 0.10)
    article_texts = [art.text for art in curr_articles]
    article_ids = [art.id for art in curr_articles]

    print("Running BERTopic...")
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(article_texts)
    topic_labels = topic_model.generate_topic_labels(nr_words=10, separator='_')
    #print(probs)

    # Create a cluster for outliers
    cluster_name = "Outliers"
    exists = db.session.query(Cluster.id).filter_by(name=cluster_name).first() is not None
    if not exists:
        clstr = Cluster(name=cluster_name)
        db.session.add(clstr)
    db.session.commit()

    # Create clusters if they don't exist
    for cluster_idx in range(0, len(topic_labels)):
        cluster_name = "BERTopic_{}".format(cluster_idx)
        exists = db.session.query(Cluster.id).filter_by(name=cluster_name).first() is not None
        if not exists:
            clstr = Cluster(name=cluster_name, explanation=topic_labels[cluster_idx])
            db.session.add(clstr)

    # We are gonna use 1 - probability here instead of distance to centroid
    pbar = tqdm(total=len(article_ids), desc='calculating distances and storing articles')
    for j, i in zip(article_ids, range(0, len(article_ids))):
        cluster_idx = topics[i]
        if cluster_idx >= 0:
            cluster_name = "BERTopic_{}".format(cluster_idx)
            #print(theme_name)
            cluster = Cluster.query.filter_by(name=cluster_name).first()
            #print(theme)
            distance_to_centroid = 1 - probs[i]
            distance_to_centroid = round(distance_to_centroid,4)
            art = Article.query.filter_by(id=j).first()

            art.cluster=cluster
            art.distance=distance_to_centroid
            #print(i, probs[i], 1 - probs[i], distance_to_centroid, art.distance)

        else:
            cluster_name = "Outliers"
            cluster = Cluster.query.filter_by(name=cluster_name).first()
            art = Article.query.filter_by(id=j).first()
            distance_to_centroid = 1.0

            art.cluster=cluster
            art.distance=distance_to_centroid

        pbar.update(1)

    pbar.close()

    db.session.commit()
    return redirect(url_for('main.data'))



@login_required
@main.route('/hdbscan', methods=['POST', 'GET'])
def run_hdbscan():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    # Potentially we would like to further reduce the IDs we do this on in the future
    # (E.g. iterative interactions)

    # Filter IDs that pass threshold
    curr_articles = Article.query.filter(Article.relevance >= 0.10)
    article_ids = [art.id for art in curr_articles]

    #article_ids = list(range(0, article_embed.shape[0]))

    X_Norm = article_embed[article_ids]
    print("Reducing dimentionality...")

    #pca = PCA(n_components=50)
    #X_Norm = pca.fit_transform(X_Norm)

    #svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
    #X_Norm = csr_matrix(X_Norm)
    #X_Norm = svd.fit_transform(X_Norm)

    X_Norm = umap.UMAP(n_neighbors=15, n_components=100, metric='cosine', min_dist=0.0).fit_transform(X_Norm)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=25, gen_min_span_tree=True)
    print(clusterer)
    print('fitting...')
    clusterer.fit(X_Norm)
    print('DONE.')

    for cluster_idx in range(0, clusterer.labels_.max() + 1):
        cluster_name = "HDBSCAN_{}".format(cluster_idx)
        exists = db.session.query(Cluster.id).filter_by(name=cluster_name).first() is not None
        if not exists:
            clstr = Cluster(name=cluster_name)
            db.session.add(clstr)

    # Create a cluster for outliers
    cluster_name = "Outliers"
    exists = db.session.query(Cluster.id).filter_by(name=cluster_name).first() is not None
    if not exists:
        clstr = Cluster(name=cluster_name)
        db.session.add(clstr)
    db.session.commit()

    # We are gonna use 1 - probability here instead of distance to centroid
    pbar = tqdm(total=len(article_ids), desc='calculating distances and storing articles')
    for j, i in zip(article_ids, range(0, len(article_ids))):
        cluster_idx = clusterer.labels_[i]
        if cluster_idx >= 0:
            cluster_name = "HDBSCAN_{}".format(cluster_idx)
            #print(theme_name)
            cluster = Cluster.query.filter_by(name=cluster_name).first()
            #print(theme)

            distance_to_centroid = 1 - clusterer.probabilities_[i]
            distance_to_centroid = round(distance_to_centroid,4)
            art = Article.query.filter_by(id=j).first()

            art.cluster=cluster
            art.distance=distance_to_centroid

        else:
            cluster_name = "Outliers"
            cluster = Cluster.query.filter_by(name=cluster_name).first()
            art = Article.query.filter_by(id=j).first()
            distance_to_centroid = 1.0

            art.cluster=cluster
            art.distance=distance_to_centroid

        pbar.update(1)

    pbar.close()

    db.session.commit()
    return redirect(url_for('main.data'))

def _assign_topics(request, user_id):
    print(request)
    # Assign to theme scenario
    topic_choice = request.form['topics']
    topic_name = request.form['new_topic']
    topic_description = request.form['description']

    topic_relevance = False
    if 'relevance' in request.form:
        topic_relevance = (request.form['relevance'] == "y")

    if topic_name == "" and topic_choice == "N/A":
        flash("You have to specify either an existing topic or a new name")
        return False
    else:
        if topic_name == "":
            topic_name = topic_choice
            t = Topic.query.filter_by(name=topic_name).first()
        else:
            # Adding new code
            t = Topic(name=topic_name, relevant=topic_relevance, description=topic_description)
            print(topic_name, topic_relevance, topic_description)
            db.session.add(t)
            db.session.commit()

        # continuing with the re-assignment
        article_ids = request.form['articleids'].strip().split()
        print(article_ids)
        for article_id in article_ids:
            annotation = Articleann.query.filter_by(user_id=user_id, article_id=article_id).first()
            if annotation is None:
                print(article_id)
                real_article = Article.query.filter_by(id=article_id).first()
                art_ann = Articleann(user_id=user_id, article_id=article_id, text=real_article.text)
                db.session.add(art_ann)
                annotation = Articleann.query.filter_by(user_id=user_id, article_id=article_id).first()

            annotation.topics.append(t)
            db.session.commit()

        return True

def _assign_type(request, user_id):
    print(request, user_id)
    # Assign to type scenario
    type_choice = request.form['frame']
    type_comments = request.form['comments']

    article_ids = request.form['articleidstype'].strip().split()
    for article_id in article_ids:
        annotation = Articleann.query.filter_by(user_id=user_id, article_id=article_id).first()
        if annotation is None:
            real_article = Article.query.filter_by(id=article_id).first()
            art_ann = Articleann(user_id=user_id, article_id=article_id, text=real_article.text)
            db.session.add(art_ann)
            annotation = Articleann.query.filter_by(user_id=user_id, article_id=article_id).first()

        annotation.frame = type_choice
        annotation.comments = type_comments
        db.session.commit()

@login_required
@main.route('/data', methods=['GET', 'POST'])
def data():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    #if not current_user.admin:
    # Find clusters that have assignments
    article_ids = db.session.query(articles).filter(articles.c.user_id==current_user.id).all()
    article_ids = [article_id for (article_id, user_id) in article_ids]
    curr_articles = Article.query.filter(Article.id.in_(article_ids)).all()
    cluster_ids = [a.cluster_id for a in curr_articles]
    cluster_choices = [t.name for t in Cluster.query.filter(Cluster.id.in_(cluster_ids)).order_by('name').all()]
    '''
    else:
        article_ids = db.session.query(articles).all()
        article_ids = [article_id for (article_id, user_id) in article_ids]
        curr_articles = Article.query.filter(Article.id.in_(article_ids)).all()
        cluster_ids = [a.cluster_id for a in curr_articles]
        cluster_choices = [t.name for t in Cluster.query.filter(Cluster.id.in_(cluster_ids)).order_by('name').all()]
    '''
    print(current_user.admin, cluster_choices)

    form_explore = DataForm(k=10)
    form_explore.cluster.choices = cluster_choices
    empty_table = True; curr_articles = None; columns=None

    form_assign = SymbolEditForm(topics=["N/A"])
    topic_names = ["N/A"] + [t.name for t in Topic.query.all()]
    form_assign.topics.choices = topic_names

    form_article = ArticleForm()
    print(form_explore.cluster.data)

    if form_explore.validate_on_submit():
        subquery_ann = db.session.query(Articleann).filter_by(user_id=current_user.id).subquery()
        annotations = db.session.query(subquery_ann).all()
        print("annotations", len(annotations))
        if len(annotations) > 0:
            print(annotations[0].keys())

        curr_articles = db.session.query(Article, subquery_ann, func.group_concat(Topic.name)).filter(Article.id.in_(article_ids))\
                    .join(Article.cluster, aliased=True).filter_by(name=form_explore.cluster.data)\
                    .join(subquery_ann, Article.id==subquery_ann.c.article_id, isouter=True)\
                    .join(topics, topics.c.articleann_id==subquery_ann.c.id, isouter=True)\
                    .join(Topic, Topic.id==topics.c.topic_id, isouter=True)\
                    .group_by(Article.id)\
                    .order_by('distance').limit(25).all()

        print("articles", len(curr_articles))

        if len(curr_articles) > 0:
            columns = curr_articles[0].keys()
            empty_table = False
            print(columns, len(columns))
            print(curr_articles[0], len(curr_articles[0]))

    print("request.method", request.method)
    if request.method == 'POST':
        print("request.form", request.form)
        if 'articletoedit' in request.form and request.form['articletoedit'].strip():
            article_id = request.form['articletoedit'].strip()
            return redirect(url_for('main.edit_symbols', article_id=article_id))
        if 'articleids' in request.form:
            if 'topics' in request.form:
                if _assign_topics(request, user_id=current_user.id):
                     return redirect(url_for('main.data'))
        if 'articleidstype' in request.form:
            if 'frame' in request.form:
                _assign_type(request, user_id=current_user.id)
                return redirect(url_for('main.data'))

        if 'explore-similar-button' in request.form and request.form['explore-similar-button'] == "Clicked":
            print(request.form.getlist('checkbox'))
            return redirect(url_for('main.similar', article_ids=request.form.getlist('checkbox')))

    return render_template("data.html", rows=curr_articles, columns=columns, form=form_explore,
                           form_assign=form_assign, form_article=form_article, not_empty=not empty_table)

@login_required
@main.route('/data/<article_ids>', methods=['GET', 'POST'])
def similar(article_ids):
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    form_assign = SymbolEditForm(topics=["N/A"])
    topic_names = ["N/A"] + [t.name for t in Topic.query.all()]
    form_assign.topics.choices = topic_names

    form_article = ArticleForm()

    article_ids_before = article_ids
    article_ids = ast.literal_eval(article_ids)
    article_ids = [int(id) for id in article_ids]
    query_articles = Article.query.filter(Article.id.in_(article_ids)).all()
    #article_ids = [a.id for a in query_articles]

    average = torch.from_numpy(np.mean(article_embed[article_ids], axis=0).reshape(1, -1))
    article_embed_torch = torch.from_numpy(article_embed)

    cos_scores = util.cos_sim(average, article_embed_torch)[0]
    top_results = torch.topk(cos_scores, k=10)

    top_scores = [x.item() for x in top_results.values]
    top_results = [str(x.item()) for x in top_results.indices]

    curr_articles = db.session.query(Article, Articleann)\
        .join(Articleann, isouter=True).filter(Article.id.in_(top_results))\
        .filter_by(user_id=current_user.id).all()

    empty_table = False
    first = Article.query.first()
    columns = first.__table__.columns.keys()

    if request.method == 'POST':
        print("request.form", request.form)
        if 'articletoedit' in request.form and request.form['articletoedit'].strip():
            article_id = request.form['articletoedit'].strip()
            return redirect(url_for('main.edit_symbols', article_id=article_id))
        if 'articleids' in request.form:
            if 'topics' in request.form:
                if _assign_topics(request, user_id=current_user.id):
                     return redirect(url_for('main.similar', article_ids=article_ids_before))
        if 'articleidstype' in request.form:
            if 'frame' in request.form:
                _assign_type(request, user_id=current_user.id)
                return redirect(url_for('main.similar', article_ids=article_ids_before))
        if 'explore-similar-button' in request.form and request.form['explore-similar-button'] == "Clicked":
            return redirect(url_for('main.similar', article_ids=request.form.getlist('checkbox')))

    return render_template('similar.html', rows=curr_articles, columns=columns, not_empty=not empty_table,
                           form_assign=form_assign, form_article=form_article, cos_scores=cos_scores, arts=query_articles)

@login_required
@main.route('/topics', methods=['GET', 'POST'])
def explore_topics():
    if not current_user.is_authenticated or not current_user.admin:
        return redirect(url_for('main.index'))

    topic_choices = [t.name for t in Topic.query.order_by('name').all()]

    form_explore = TopicForm()
    form_explore.topic.choices = topic_choices
    empty_table = True; curr_articles = None; columns=None

    form_edit_topic = EditTopicForm()

    if form_explore.validate_on_submit() and 'search' in request.form:
        topic_id = db.session.query(Topic.id).filter_by(name=form_explore.topic.data).first()[0]
        _annotations = db.session.query(topics).filter(topics.c.topic_id==topic_id).all()
        annotation_ids = [ann_id for (topic_id, ann_id) in _annotations]

        curr_articles = db.session.query(Article, Articleann, User)\
                .join(Articleann, Articleann.article_id==Article.id).filter(Articleann.id.in_(annotation_ids))\
                .join(User, Articleann.user_id==User.id)\
                .order_by(Article.id)\
                .all()

        if len(curr_articles) > 0:
            empty_table = False
            first = Article.query.first()
            columns = first.__table__.columns.keys()
    if form_explore.validate_on_submit() and 'delete' in request.form:
        # Get topic
        topic_id = db.session.query(Topic.id).filter_by(name=form_explore.topic.data).first()[0]
        topic = Topic.query.filter_by(id=topic_id).first()

        # Get articles assigned to it
        _annotations = db.session.query(topics).filter(topics.c.topic_id==topic_id).all()
        annotation_ids = [ann_id for (topic_id, ann_id) in _annotations]

        annotations = db.session.query(Articleann).filter(Articleann.id.in_(annotation_ids))\
                .all()

        # Unmap articles
        for _ann in annotations:
            _ann.topics.remove(topic)
        # Remove topic
        db.session.delete(topic)
        db.session.commit()
        return redirect(url_for('main.explore_topics'))
    if form_edit_topic.validate_on_submit() and 'topic_name' in request.form:
        topic_name = request.form['topic_name']
        # Get topic
        topic_id = db.session.query(Topic.id).filter_by(name=topic_name).first()[0]
        topic = Topic.query.filter_by(id=topic_id).first()
        topic.name = form_edit_topic.name.data
        db.session.commit()
        return redirect(url_for('main.explore_topics'))

    if request.method == 'POST':
        print(request.form)
        if 'articletoedit' in request.form and request.form['articletoedit'].strip():
            print("HERE")
            article_id = request.form['articletoedit'].strip()
            return redirect(url_for('main.edit_symbols', article_id=article_id))

    return render_template("topics.html", rows=curr_articles, columns=columns, form=form_explore,
                           form_edit_topic=form_edit_topic,
                           not_empty=not empty_table)


@login_required
@main.route('/edit_symbols/<article_id>', methods=['GET', 'POST'])
def edit_symbols(article_id):
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    art_ann = Articleann.query.filter_by(article_id=article_id, user_id=current_user.id).first()
    if art_ann is None:
        real_article = Article.query.filter_by(id=article_id).first()
        art_ann = Articleann(user_id=current_user.id, article_id=article_id, text=real_article.text)
        db.session.add(art_ann)
        db.session.commit()

    article = db.session.query(Article, Articleann)\
                .join(Articleann, Article.id==Articleann.article_id)\
                .filter_by(article_id=article_id, user_id=current_user.id).first()

    topic_names = [t.name for t in Topic.query.all()]

    form_edit = SymbolEditForm()
    form_edit.topics.choices = topic_names

    form_article = ArticleForm()
    form_quantity = QuantityEditForm()

    if request.method == 'POST':
        print('POST REQUEST:', request.form)

        for key in request.form:
            if key.startswith('remove'):
                topic_name = '-'.join(key.strip().split('-')[1:])
                article[1].topics.remove(Topic.query.filter_by(name=topic_name).first())
                db.session.commit()
                return redirect(url_for('main.edit_symbols', article_id=article_id))

        if 'assign_topic' in request.form:
            assigned_topic = request.form['topics']
            article[1].topics.append(Topic.query.filter_by(name=assigned_topic).first())
            db.session.commit()
            return redirect(url_for('main.edit_symbols', article_id=article_id))

        if 'new_topic' in request.form:
            topic_name = request.form['new_topic']
            topic_relevance = False
            if 'relevance' in request.form:
                topic_relevance = True
            topic_description = request.form['description']
            t = Topic(name=topic_name, relevant=topic_relevance, description=topic_description)
            db.session.add(t)
            article[1].topics.append(t)
            db.session.commit()
            return redirect(url_for('main.edit_symbols', article_id=article_id))

        if 'macro' in request.form:
            if request.form['macro'] == 'yes':
                article[1].frame = 'macro'
                article[1].econ_rate = request.form['econ_rate']
                article[1].econ_change = request.form['econ_change']

            if 'frame' in request.form and request.form['macro'] == 'no':
                article_frame = request.form['frame']
                article[1].frame = article_frame

            if 'comments' in request.form:
                article_comment = request.form['comments']
                article[1].comments = article_comment
            db.session.commit()
            flash("Article updated")
            return redirect(url_for('main.edit_symbols', article_id=article_id))

        if 'type' in request.form:
            quant_type = request.form['type']
            quant_spin = request.form['spin']
            quant_id = request.form['spanId']

            unique_id = "{}_{}".format(article_id, quant_id)

            quant = db.session.query(Quantity, Quantityann)\
                .join(Quantityann, isouter=True).filter_by(quantity_id=unique_id, user_id=current_user.id).first()
            if quant is None:
                quant_ann = Quantityann(user_id=current_user.id, quantity_id=unique_id)
                db.session.add(quant_ann)
                quant = db.session.query(Quantity, Quantityann)\
                    .join(Quantityann, isouter=True).filter_by(quantity_id=unique_id, user_id=current_user.id).first()

            quant[1].type = quant_type
            quant[1].spin = quant_spin
            if quant_type == 'macro':
                quant[1].macro_type = request.form['macro_type']
                if request.form['macro_type'] == 'other':
                    quant[1].comments = request.form['comments']
            elif quant_type == 'industry':
                quant[1].industry_type = request.form['industry_type']
                if request.form['industry_type'] == 'other':
                    quant[1].comments = request.form['comments']
            elif quant_type == 'government':
                quant[1].gov_level = request.form['government_level']
                quant[1].gov_type = request.form['gov_type']
                if quant[1].gov_type == 'other':
                    quant[1].comments = request.form['comments']
                if quant[1].gov_type == 'revenue':
                    quant[1].revenue_type = request.form['gov_revenue_type']
                    if quant[1].revenue_type == 'other':
                        quant[1].comments = request.form['comments']
                elif quant[1].gov_type == 'expenditures':
                    quant[1].expenditure_type = request.form['gov_expenditure_type']
                    if quant[1].expenditure_type == 'other':
                        quant[1].comments = request.form['comments']
            elif quant_type == 'business' or quant_type == 'personal':
                quant[1].comments = request.form['comments']

            color = 'yellow'; superscript = ''; subscript = ''
            if quant_type is None or quant_type == 'other':
                color = 'yellow'; superscript = ''
            elif quant_type == 'macro':
                color = 'blue'; superscript = 'macro'
                subscript = quant[1].macro_type
            elif quant_type == 'industry':
                color = 'orange'; superscript = 'ind'
                subscript = quant[1].industry_type
            elif quant_type == 'business':
                color = 'green'; superscript = 'firm'
            elif quant_type == 'government':
                color = 'purple'; superscript = 'gov'
                print(quant[1].gov_level, quant[1].gov_type)
                subscript = quant[1].gov_level + "_" + quant[1].gov_type
                if subscript == 'revenue':
                    subscript += "_" + quant[1].gov_revenue_type
                elif subscript == 'expenditures':
                    subscript += "_" + quant[1].gov_expenditure_type
            elif quant_type == 'personal':
                color = 'pink'; superscript = 'per'

            print(color, superscript, subscript)
            subscript += "_[" + quant_spin[:3] + "]"

            # edit color for text
            prev_pattern = r'<span id="{}" class="\w+" onclick="(.*?)">(.*?)(<sup>.*?</sup>)*?(<sub>.*?</sub>)*?</span>'.format(quant_id)
            replacement = r'<span id="{}" class="{}" onclick="\1">\2<sup><b>{}</b></sup><sub>{}</sub></span>'.format(quant_id, color, superscript, subscript)

            article[1].text = re.sub(prev_pattern, replacement, article[1].text)

            db.session.commit()

    return render_template("edit_symbols.html", article=article, form_edit=form_edit, form_article=form_article,
                           form_quantity=form_quantity)


def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

@login_required
@main.route('/global', methods=['GET', 'POST'])
def global_state():
    if not current_user.is_authenticated:
        return redirect(url_for('main.index'))

    clusters = Cluster.query.all()
    cluster_names = []; counts = []
    for cluster in clusters:
        num_samples = Article.query.join(Article.cluster, aliased=True).filter_by(name=cluster.name).count()
        if num_samples > 0:
            counts.append(num_samples)
            cluster_names.append(cluster.name)

    clrs = ['grey' if name == 'Outliers' else 'red' for name in cluster_names]
    ind = np.arange(len(counts)).tolist()
    print(cluster_names)
    print(counts)

    f, ax = plt.subplots(figsize=(15,5)) # set the size that you'd like (width, height)
    bars = ax.bar(ind, counts, color=clrs)
    ax.set_ylabel('Num. Articles')
    ax.set_xlabel('Cluster')
    ax.set_xticks(ind, cluster_names)
    ax.bar_label(bars)
    bar_plot_img = json.dumps(mpld3.fig_to_dict(f), default=convert)

    print(type(bar_plot_img))

    return render_template("global.html", bar_plot_img=bar_plot_img)


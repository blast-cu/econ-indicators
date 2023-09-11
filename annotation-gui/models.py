from __main__ import db
from sqlalchemy import ForeignKey
from sqlalchemy import JSON
from flask_login import UserMixin

topics = db.Table('topics',
                  db.Column('topic_id', db.Integer, db.ForeignKey('topic.id'), primary_key=True),
                  db.Column('articleann_id', db.Integer, db.ForeignKey('articleann.id'), primary_key=True),
)

clusters = db.Table('clusters',
                db.Column('cluster_id', db.Integer, db.ForeignKey('cluster.id'), primary_key=True),
                db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
)

articles = db.Table('articles',
                       db.Column('article_id', db.Integer, db.ForeignKey('article.id'), primary_key=True),
                       db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
)

class Articleann(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    article_id = db.Column(db.Integer, db.ForeignKey('article.id'), nullable=False)

    article = db.relationship('User', backref=db.backref('articleanns', lazy=True))
    user = db.relationship('Article', backref=db.backref('articleanns', lazy=True))

    frame = db.Column(db.String, nullable=True)
    econ_rate = db.Column(db.String, nullable=True)
    econ_change = db.Column(db.String, nullable=True)
    comments = db.Column(db.String(1000), nullable=True)
    topics = db.relationship('Topic', secondary=topics, lazy='subquery', backref=db.backref('articleanns', lazy=True))
    text = db.Column(db.String(5000), nullable=False)

class Quantityann(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quantity_id = db.Column(db.Integer, db.ForeignKey('quantity.id'), nullable=False)

    type = db.Column(db.String(1000), nullable=True)
    macro_type = db.Column(db.String(1000), nullable=True)
    industry_type = db.Column(db.String(1000), nullable=True)
    gov_level = db.Column(db.String(1000), nullable=True)
    gov_type = db.Column(db.String(1000), nullable=True)
    expenditure_type = db.Column(db.String(1000), nullable=True)
    revenue_type = db.Column(db.String(1000), nullable=True)

    comments = db.Column(db.String(1000), nullable=True)
    spin = db.Column(db.String(1000), nullable=True)

    user = db.relationship('User', backref=db.backref('quantityanns', lazy=True))
    quantity = db.relationship('Quantity', backref=db.backref('quantityanns', lazy=True))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    admin = db.Column(db.Boolean, default=False, nullable=False)

    clusters = db.relationship('Cluster', secondary=clusters, lazy='subquery', backref=db.backref('users', lazy=True))
    articles = db.relationship('Article', secondary=articles, lazy='subquery', backref=db.backref('users', lazy=True))

class Article(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    headline = db.Column(db.String(1000), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    keywords = db.Column(db.String(100), nullable=False)
    num_keywords = db.Column(db.Integer, nullable=False)
    relevance = db.Column(db.Float, nullable=False)
    text = db.Column(db.String(5000), nullable=False)
    distance = db.Column(db.Float)
    date = db.Column(db.DateTime, nullable=False)
    url = db.Column(db.String(100), nullable=False)

    cluster_id = db.Column(db.Integer, ForeignKey('cluster.id'), nullable=True)
    cluster = db.relationship('Cluster', backref=db.backref('articles', lazy=True))

class Topic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(1000), nullable=True)
    relevant = db.Column(db.Boolean, default=False, nullable=False)
    name = db.Column(db.String(1000), nullable=False, unique=True)

class Cluster(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000), nullable=False, unique=True)
    explanation = db.Column(db.String(1000), nullable=True)

class Quantity(db.Model):
    id = db.Column(db.String(1000), primary_key=True)
    local_id = db.Column(db.Integer, nullable=False)

    article_id = db.Column(db.Integer, ForeignKey('article.id'), nullable=False)
    article = db.relationship('Article', backref=db.backref('quantities', lazy=True))

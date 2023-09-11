from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField, BooleanField, PasswordField, IntegerField, FormField, SelectField, FieldList, RadioField
from flask_wtf.file import FileField, FileRequired
from wtforms.validators import DataRequired, Length, InputRequired
from wtforms.fields import *


class LogInForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    login = SubmitField('Log In')

class SignUpForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    submit = SubmitField('Sign Up')

class ClusterMethodForm(FlaskForm):
    k = IntegerField('K (# initial clusters, only needed if using K-means)', default=10)
    method = SelectField(choices=[('kmeans', 'K-Means'), ('hdbscan', 'HDBSCAN'), ('bertopic', 'BERTopic')])

    submit = SubmitField('Recluster')
    assign = SubmitField('Assign articles to topics')
    restart = SubmitField('Start from scratch')

class DataForm(FlaskForm):
    cluster = SelectField('Cluster')
    k = IntegerField('K (# of norms to show)')
    close = SubmitField('Explore Close Data Points')
    distant = SubmitField('Explore Distant Data Points')

class TopicForm(FlaskForm):
    topic = SelectField('Topic')
    search = SubmitField('See Assignments')
    edit = SubmitField('Edit Topic')
    delete = SubmitField('Delete Topic')

class EditTopicForm(FlaskForm):
    name = StringField('Name', validators=[InputRequired()])
    submit = SubmitField('Submit')

class SymbolEditForm(FlaskForm):
    topics = SelectField('Select Topic to Assign')
    assign_topic = SubmitField('Assign Topic')
    add_topic = SubmitField('+New Topic')
    new_topic = StringField('Topic Name', validators=[InputRequired()])
    relevance = BooleanField('Relevant?')
    description = StringField('Please provide a short description')
    role_emotion_submit = SubmitField('Submit')

class ArticleForm(FlaskForm):
    macro = SelectField('Does this article discuss or touches on the economy as a whole?',
                        choices=[('yes', 'Yes'),
                                 ('no', 'No')])

    frame = SelectField('General Article Framing (Select the most prominent one)',
                       choices=[('industry', 'Industry-Specific'),
                                ('business', 'Firm-Specific'),
                                ('government', 'Government and/or Politics'),
                                ('other', 'Other')])
    econ_rate = SelectField('How does this article rate economic conditions in the US?',
                            choices=[
                                ('good', 'Good'),
                                ('poor', 'Poor'),
                                ('none', 'No Opinion'),
                                ('unsure', 'Coder is unsure'),
                                ('irrelevant', 'Not relevant to the US economy')
                            ])
    econ_change = SelectField('Does this article state/imply that economic conditions in the US as a whole are getting better or getting worse?',
                            choices=[
                                ('better', 'Better'),
                                ('worse', 'Worse'),
                                ('same', 'Same'),
                                ('none', 'No Opinion'),
                                ('unsure', 'Coder is unsure'),
                                ('irrelevant', 'Not relevant to the US economy')
                            ])

    comments = StringField('Please provide a short explanation')

class QuantityEditForm(FlaskForm):
    type = SelectField('Quantity Type',
                       choices=[('other', 'Other/Cannot Tell'),
                                ('macro', 'Macro-Economic'),
                                ('industry', 'Industry-Specific'),
                                ('business', 'Firm-Specific'),
                                ('government', 'Government Revenue and Expenditures'),
                                ('personal', 'Personal')])
    macro_type = SelectField('Macro-Economic Indicator',
                             choices=[
                                ('other', 'Other/Cannot Tell'),
                                ('jobs', 'Jobs Numbers (Jobs, Unemployment)'),
                                ('market', 'Market Numbers (Any Financial Market)'),
                                ('housing', 'Housing (Start, Sales, Pricing)'),
                                ('macro', 'Macro Economy (GDP, etc.)'),
                                ('wages', 'Wages'),
                                ('prices', 'Prices (CPI, PPI)'),
                                ('confidence', 'Confidence'),
                                ('retail', 'Retail Sales'),
                                ('interest', 'Interest Rates (Fed, Mortgage)'),
                                ('currency', 'Currency Values'),
                                ('energy', 'Energy Prices')
                             ])
    industry_type = SelectField('Industry Type',
                                choices=[
                                    ('other', 'Other/Cannot Tell (not gov.)'),
                                    ('agro-forestry-hunting', 'Agriculture, Forestry and Hunting'),
                                    ('mining', 'Mining'),
                                    ('utilities', 'Utilities'),
                                    ('construction', 'Construction'),
                                    ('manufacturing', 'Manufacturing'),
                                    ('wholesale', 'Wholesale Trade'),
                                    ('retail', 'Retail Trade'),
                                    ('trans-warehouse', 'Transportation and Warehousing'),
                                    ('info', 'information'),
                                    ('finance-insurance-realestate', 'Finance, Insurance, Real Estate and Leasing'),
                                    ('professional', 'Professional and Business Services'),
                                    ('education-health-social', 'Health Care, Educational Services and Social Assistance'),
                                    ('art-entertain-hotel-restaurant', 'Arts, Entertainment, Recreation, Accomodation and Food Services'),
                                ])

    government_level = SelectField('Governmental Level',
                                   choices=[
                                        ('other', 'Other/Cannot Tell'),
                                        ('federal', 'Federal'),
                                        ('local', 'State/Local'),
                                        ('foreign', 'Non-US'),
                                   ])

    gov_type = SelectField('Type',
                           choices=[
                               ('expenditures', 'Expenditures'),
                               ('revenue', 'Revenue'),
                               ('deficit', 'Debt/Deficit')])

    gov_expenditure_type = SelectField('Type of Expenditure',
                                        choices=[
                                            ('social-security', 'Social Security and Public Welfare'),
                                            ('health', 'Health and Hospitals'),
                                            ('defense', 'National Defense'),
                                            ('police', 'Police'),
                                            ('transportation', 'Transportation'),
                                            ('research', 'Research'),
                                            ('education', 'Education'),
                                            ('employment', 'Employment'),
                                            ('housing', 'Housing'),
                                            ('corrections', 'Corrections'),
                                            ('courts', 'Courts'),
                                            ('net-interest', 'Net Interest'),
                                            ('other', 'Other/Cannot Tell')
                                        ])

    gov_revenue_type = SelectField('Type of Revenue',
                                    choices=[
                                        ('tax', 'Taxes and Other Compulsory Transfers'),
                                        ('income', 'Income Derived from Assets'),
                                        ('sales', 'Sales of Goods and Services'),
                                        ('voluntary', 'Voluntary Transfers'),
                                        ('other', 'Other/Cannot Tell')
                                    ])

    spin = SelectField = SelectField('Framing of quantity',
                                     choices=[
                                        ('pos', 'Positive (e.g. inflation dropped)'),
                                        ('neg', 'Negative (e.g. prices skyrocketed)'),
                                        ('neutral', 'Neutral (e.g. GDP was worth X US dollars)'),
                                        ('unknown', 'Cannot tell')
                                     ])

    comments = StringField('Please provide a short explanation')

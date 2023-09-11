# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, flash, Markup, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from sqlalchemy import desc
from flask_login import LoginManager

import argparse
import numpy as np
import json
import os

app = Flask(__name__)
db = SQLAlchemy()

article_embed = None
#article_text = None
article_features = None

if __name__ == '__main__':
    #app.run(debug=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=80)
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--static', type=str, required=True)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    app.secret_key = 'dev'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{}'.format(args.database)

    # set default button sytle and size, will be overwritten by macro parameters
    app.config['BOOTSTRAP_BTN_STYLE'] = 'primary'
    app.config['BOOTSTRAP_BTN_SIZE'] = 'sm'
    #app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'quartz'  # uncomment this line to test bootswatch theme

    # set default icon title of table actions
    app.config['BOOTSTRAP_TABLE_VIEW_TITLE'] = 'Read'
    app.config['BOOTSTRAP_TABLE_EDIT_TITLE'] = 'Update'
    app.config['BOOTSTRAP_TABLE_DELETE_TITLE'] = 'Remove'
    app.config['BOOTSTRAP_TABLE_NEW_TITLE'] = 'Create'

    app.config['UPLOAD_FOLDER'] = args.static
    # 500 MB
    app.config['MAX_CONTENT_PATH'] = 500000000

    bootstrap = Bootstrap(app)
    db.init_app(app)
    csrf = CSRFProtect(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    from models import User

    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    # blueprint for auth routes in our app
    from auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    # blueprint for non-auth parts of app
    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    if args.debug:
        app.run(debug=True, host=args.host, port=args.port)
    else:
        from waitress import serve
        serve(app, host=args.host, port=args.port)



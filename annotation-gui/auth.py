from __main__ import db
from flask import Blueprint, render_template, redirect, url_for, request, flash, Markup
from werkzeug.security import generate_password_hash, check_password_hash
from models import User
from forms import LogInForm, SignUpForm
from flask_login import login_user, login_required, logout_user
from main import _update_annotations, _update_articles

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    form = LogInForm()
    return render_template('login.html', form=form)

@auth.route('/login', methods=['POST'])
def login_post():
    # login code goes here
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()
    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('auth.login')) # if the user doesn't exist or password is wrong, reload the page

    # if the above check passes, then we know the user has the right credentials
    login_user(user, remember=remember)

    # if the above check passes, then we know the user has the right credentials
    return redirect(url_for('main.data'))

@auth.route('/signup')
def signup():
    form = SignUpForm()
    return render_template('signup.html', form=form)

@auth.route('/signup', methods=['POST'])
def signup_post():
    # code to validate and add user to database goes here
    email = request.form.get('email')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first()

    if user: # if a user is found, we want to redirect back to signup page so user can try again
        flash(Markup('Email address already exists, go to the <a href="/login" class="alert-link">login</a> page'))
        return redirect(url_for('auth.signup'))

    # create a new user with the form data. Hash the password so the plaintext version isn't saved.
    new_user = User(email=email, password=generate_password_hash(password, method='sha256'), admin=False)

    # add the new user to the database
    db.session.add(new_user)
    db.session.commit()

    _update_articles()
    #_update_annotations(new_user.id)

    # code to validate and add user to database goes here
    return redirect(url_for('auth.login'))

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))

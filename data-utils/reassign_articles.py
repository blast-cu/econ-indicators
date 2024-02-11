import argparse
import sqlite3

def get_users(cur):
    query = 'select id, email from user'
    res = cur.execute(query)
    user2email = {}
    for user_id, email in res:
        user2email[user_id] = email
    return user2email

def get_article_assignments(cur, user_id):
    query = 'select article_id from articles where user_id = {}'.format(user_id)
    res = cur.execute(query)
    articles = [a[0] for a in res]

    return articles

def unassign_completed_tasks(cur, user_id, assigned_articles):
    print("Deleting completed tasks for {}...".format(user_id))
    query = 'select article_id from articleann where user_id = {} and frame is not null'.format(user_id)
    res = cur.execute(query)
    annotated_articles = [a[0] for a in res]
    annotated_articles = set(annotated_articles)
    assigned_articles = set(assigned_articles)

    for a in annotated_articles & assigned_articles:
        query = 'delete from articles where user_id = {} and article_id = {}'.format(user_id, a)
        cur.execute(query)
    print("Done")

def assign_incomplete_articles(cur, user2email, email2user, user_email, limit):
    query = 'select distinct article_id from articleann'
    res = cur.execute(query)
    articles = [a[0] for a in res]

    num_assign = 0
    for article in articles:
        query = 'select user_id, frame from articleann where frame is not null and article_id = {}'.format(article)
        res = cur.execute(query)
        users = set([user_email[a[0]] for a in res])
        frames = [a[1] for a in res]
        if len(users) <= 1 or len(set(frames)) != len(frames):

            if user_email not in users and num_assign < limit:
                user_id = email2user[user_email]
                query = 'insert into articles (user_id, article_id) values ({}, {})'.format(user_id, article)
                cur.execute(query)
                num_assign += 1


def main(args):
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Get users
    user2email = get_users(cur)
    email2user = {v: k for k, v in user2email.items()}
    print(user2email)
    print(email2user)

    if args.remove_completed:
        for user_id in user2email:
            articles = get_article_assignments(cur, user_id)
            if len(articles) > 0:
                unassign_completed_tasks(cur, user_id, articles)
        conn.commit()

    if args.assign_incomplete:
        if not args.user_email or not args.limit:
            print('You must specify --user_email and --limit when using --assign_incomplete')
            exit()
        assign_incomplete_articles(cur, user2email, email2user, args.user_email, args.limit)
        conn.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    parser.add_argument('--remove_completed', default=False, action='store_true')
    parser.add_argument('--assign_incomplete', default=False, action='store_true')
    parser.add_argument('--user_email', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    main(args)


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

def get_annotation_stats(cur, user_id, articles):
    articles = [str(a) for a in articles]
    query = 'select count(*) from articleann where user_id = {} and frame is not null and article_id in ({})'.format(user_id, ",".join(articles))
    res = cur.execute(query)
    return [a[0] for a in res][0]

def main(args):
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Get users
    user2email = get_users(cur)
    print(user2email)

    for user_id in user2email:
        articles = get_article_assignments(cur, user_id)
        if len(articles) > 0:
            # find annotations
            count_ann = get_annotation_stats(cur, user_id, articles)
            print(user2email[user_id], "{}/{}".format(count_ann, len(articles)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    main(args)



import argparse
import sqlite3
from tqdm import tqdm
import os

def main(args):
    source2url = {
        'nytimes': 'https://www.nytimes.com',
        'foxnews': 'https://www.foxnews.com',
        'bbc': 'https://www.bbc.com',
        'breitbart': 'https://www.breitbart.com'
    }

    con = sqlite3.connect(args.database)
    cur = con.cursor()

    res = cur.execute("SELECT id, source, url FROM article").fetchall()
    pbar = tqdm(total=len(res))
    for (_id, _source, _url) in res:
        new_url = source2url[_source] + _url
        cur.execute('UPDATE article SET url = "{}" where id = {}'.format(new_url, _id))
        con.commit()
        pbar.update(1)
    pbar.close()
    con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    args = parser.parse_args()
    main(args)

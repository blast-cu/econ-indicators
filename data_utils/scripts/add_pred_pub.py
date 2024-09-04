import pandas as pd
import sqlite3

from data_utils.model_utils.dataset import DB_FILENAME


def get_publisher(row):
    id = row['article_id']

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    cur.execute(f"SELECT source FROM article WHERE id={id}")
    publisher = cur.fetchone()[0]
    con.close()

    return publisher


def get_date(row):
    id = row['article_id']

    con = sqlite3.connect(DB_FILENAME)
    cur = con.cursor()
    cur.execute(f"SELECT date FROM article WHERE id={id}")
    year = cur.fetchone()[0]
    con.close()

    return year

def main():
    pred_path = "data/predictions"
    original_preds = f"{pred_path}/qual_predictions.csv"
    df = pd.read_csv(original_preds)
    print(df.head())

    df['publisher'] = df.apply(get_publisher, axis=1)
    df['date'] = df.apply(get_date, axis=1)
    print(df.head())

    edited_preds = f"{pred_path}/qual_predictions_edited.csv"
    df.to_csv(edited_preds, index=False)

if __name__ == "__main__":
    main()

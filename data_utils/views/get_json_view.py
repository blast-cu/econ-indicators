import argparse
import json
import sqlite3
import os
import logging

# set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_count(data, f):
    """
    Get the count of items in json.`
    """
    count = len(data)
    f.write(f"- row count: {count}\n")

def get_publisher_article_count(articles, cursor, f):
    """
    Get number of articles per publisher.
    """
    # create a string with all article ids separated by commas
    article_ids = [str(article_id) for article_id in articles.keys()]
    article_ids_str = ",".join(article_ids)
    cursor.execute("""
        SELECT source, COUNT(*) 
        FROM article 
        WHERE id IN ({}) 
        GROUP BY source
    """.format(article_ids_str))

    rows = cursor.fetchall()
    f.write("- per publisher:\n")
    for row in rows:
        f.write(f"  - {row[0]}: {row[1]}\n")

def get_publisher_quantity_count(quants, cursor, f):
    """
    Get number of quantities per publisher.
    """
    quant_ids = [str(quant_id) for quant_id in quants.keys()]
    quant_ids_str = ','.join(quant_ids)
    # join quantity with article to get the publisher
    cursor.execute("""
        SELECT a.source, COUNT(q.id) 
        FROM quantity q
        JOIN article a ON q.article_id = a.id
        WHERE q.article_id IN ({})
        GROUP BY a.source
    """.format(quant_ids_str))

    rows = cursor.fetchall()
    f.write("- per publisher:\n")
    for row in rows:
        f.write(f"  - {row[0]}: {row[1]}\n")

def main(args):

    # Connect to the SQLite database
    logger.info(f"Connecting to database: {args.db}")
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    # load the json files
    logger.info(f"Loading JSON files from: {args.data_dir}")
    with open(args.data_dir + "/all_articles.json", "r") as f:
        articles = json.load(f)

    with open(args.data_dir + "/all_excerpts.json", "r") as f:
        excerpts = json.load(f)

    # create a file for the report
    logger.info("Creating report file")
    db_name = (args.db.split(".")[0]).split("/")[-1]
    report_path = "data/reports"
    os.makedirs(report_path, exist_ok=True)

    report_file = os.path.join(report_path, f"all_jsons_report.txt")
    with open(report_file, "w") as f:
        f.write(f"JSON Report: {db_name}\n")
        f.write("================\n\n")

        f.write("ARTICLES\n")
        get_count(articles, f)
        get_publisher_article_count(articles, cursor, f)

        f.write("\n\nEXCERPTS\n")
        get_count(excerpts, f)
        get_publisher_quantity_count(excerpts, cursor, f)


    # Close the connection
    conn.close()
    logger.info(f"Report created at: {report_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Database operations script')
    parser.add_argument('--db', type=str, help='Path of the database')
    parser.add_argument("--data_dir", type=str, default="data/clean", help='path to directory containing json data')
    args = parser.parse_args()
    main(args)
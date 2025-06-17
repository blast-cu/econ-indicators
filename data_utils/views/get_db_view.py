import argparse
import sqlite3
import os
import logging

# set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_count(cursor, table_name, f):
    """
    Get the count of rows in a table.
    """
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    f.write(f"- row count: {count}\n")

def get_publisher_article_count(cursor, f):
    """
    Get number of articles per publisher.
    """
    cursor.execute("SELECT source, COUNT(*) FROM article GROUP BY source")
    rows = cursor.fetchall()
    f.write("- per publisher:\n")
    for row in rows:
        f.write(f"  - {row[0]}: {row[1]}\n")

def get_publisher_quantity_count(cursor, f):
    """
    Get number of quantities per publisher.
    """
    # join quantity with article to get the publisher
    cursor.execute("""
        SELECT a.source, COUNT(q.id) 
        FROM quantity q
        JOIN article a ON q.article_id = a.id
        GROUP BY a.source
    """)
    rows = cursor.fetchall()
    f.write("- per publisher:\n")
    for row in rows:
        f.write(f"  - {row[0]}: {row[1]}\n")

def main(args):

    # Connect to the SQLite database
    logger.info(f"Connecting to database: {args.db}")
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    # create a file for the report
    logger.info("Creating report file...")
    db_name = (args.db.split(".")[0]).split("/")[-1]
    report_path = "data/reports"
    os.makedirs(report_path, exist_ok=True)

    report_file = os.path.join(report_path, f"{db_name}_report.txt")
    with open(report_file, "w") as f:
        f.write(f"Database Report: {db_name}\n")
        f.write("================\n\n")

        f.write("ARTICLE TABLE\n")
        get_count(cursor, "article", f)
        get_publisher_article_count(cursor, f)

        f.write("\n\nQUANTITY TABLE\n")
        get_count(cursor, "quantity", f)
        get_publisher_quantity_count(cursor, f)


    # Close the connection
    conn.close()
    logger.info(f"Report created and saved to {report_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Database operations script')
    parser.add_argument('--db', type=str, help='Path of the database')
    args = parser.parse_args()
    main(args)




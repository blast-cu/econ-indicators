import sqlite3
import os
import re
from potato_annotation.eval.read_article_annotations import get_potato_article_anns
from potato_annotation.eval.read_quant_annotations import get_potato_quant_anns
from data_utils.model_utils.dataset import DB_FILENAME
import argparse

def get_type_arg(args):
    qual = False
    quant = False
    if args.type == "qual":
        qual = True
    elif args.type == "quant":
        quant = True
    elif args.type == "both":
        qual = True
        quant = True
    else:
        raise ValueError("Invalid type argument")

    return qual, quant


def get_ann_dict(anns):
    ann_dict = {}
    for ann_comp, anns in anns.items():
        for ann in anns:
            article_id, user_id, val = ann
            if article_id not in ann_dict:
                ann_dict[article_id] = {}
            if user_id not in ann_dict[article_id]:
                ann_dict[article_id][user_id] = {}
            ann_dict[article_id][user_id][ann_comp] = val
    return ann_dict


def get_article_text(article_id, cursor):

    cursor.execute(f"SELECT text FROM article WHERE id = '{article_id}'")
    text = cursor.fetchone()[0]
    return text


def main(args):

    qual, quant = get_type_arg(args)
    ARTICLE_ANN_DIR = "potato_annotation/article_annotate/to_read/"
    QUANT_ANN_DIR = "potato_annotation/quant_annotate/to_read/"

    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    if qual:
        ann_dirs = os.listdir(ARTICLE_ANN_DIR)

        ## add source column to annotations table
        # cursor.execute("ALTER TABLE articleann ADD COLUMN source TEXT")

        for ann_dir in ann_dirs:
            article_annotations, _ = get_potato_article_anns(
                f"{ARTICLE_ANN_DIR}{ann_dir}"
            )
            ann_dict = get_ann_dict(article_annotations)
            for article_id, user_anns in ann_dict.items():
                article_text = get_article_text(article_id, cursor)
                article_text = article_text.replace("'", "''")

                for user_id, ann in user_anns.items():
                    # remove duplicate
                    cursor.execute(f"DELETE FROM articleann WHERE user_id = '{user_id}' AND article_id = '{article_id}'")
                    user_id = re.sub(r"[A-Za-z]", "", user_id)
                    user_id = int(user_id)
                    article_id = int(article_id)

                    # check for a duplicate
                    cursor.execute(f"SELECT * FROM articleann WHERE user_id = {user_id} AND article_id = {article_id}")
                    if cursor.fetchone():
                        print(f"DUPLICATE: {user_id} {article_id}")
                        continue
                    if ann['frame'] == 'macro':
                        insert_query = f"INSERT INTO articleann (user_id, article_id, frame, econ_rate, econ_change, text, source)\
                            VALUES ({user_id}, {article_id}, '{ann['frame']}', '{ann['econ_rate']}', '{ann['econ_change']}', '{article_text}', 'pototo')"
                    elif ann['frame'] == 'NA':
                        insert_query = f"INSERT INTO articleann (user_id, article_id, text, source)\
                            VALUES ({user_id}, {article_id}, '{article_text}', 'potato')"
                    else:
                        insert_query = f"INSERT INTO articleann (user_id, article_id, frame, text, source)\
                            VALUES ({user_id}, {article_id}, '{ann['frame']}', '{article_text}', 'potato')"

                    cursor.execute(insert_query)

            # TEST
            # select_query = "SELECT frame, econ_rate, econ_change FROM articleann"
            # cursor.execute(select_query)
            # for row in cursor.fetchall():
            #     print(row)
            # exit()

            # if source is null, set source to "house"
            cursor.execute("UPDATE articleann SET source = 'house' WHERE source IS NULL")

    if quant:

        try:  # add source column to annotations table
            cursor.execute("ALTER TABLE quantityann ADD COLUMN source TEXT")
        except sqlite3.OperationalError:
            pass

        # loop ober directories in 'to_read' directory
        ann_dirs = os.listdir(QUANT_ANN_DIR)
        for ann_dir in ann_dirs:
            quant_annotations = get_potato_quant_anns(
                f"{QUANT_ANN_DIR}{ann_dir}"
            )

            for ann in quant_annotations:
                quant_id = ann['quant_id']
                # print(quant_id)
                user_id = ann["user_id"]
                frame = ann["type"]
                macro_type = ann["macro_type"]
                spin = ann["spin"]

                # remove duplicate
                cursor.execute(
                    f"DELETE FROM quantityann WHERE user_id = '{user_id}' AND quantity_id = '{quant_id}'"
                )
                user_id = re.sub(r"[A-Za-z]", "", user_id)
                user_id = int(user_id)

                # check for a duplicate
                cursor.execute(
                    f"SELECT * FROM quantityann WHERE user_id = {user_id} AND quantity_id = '{quant_id}'"
                )
                if cursor.fetchone():
                    print(f"DUPLICATE: {user_id} {quant_id}")
                    # delete duplicate
                    cursor.execute(
                        f"DELETE FROM quantityann WHERE user_id = {user_id} AND quantity_id = '{quant_id}'"
                    )

                    cursor.execute(
                        f"DELETE FROM quantityann WHERE user_id = {user_id} AND quantity_id = {int(quant_id)}"
                    )
                
                # insert into database
                insert_query = f"INSERT INTO quantityann (user_id, quantity_id, type, macro_type, spin, source)\
                    VALUES ({user_id}, '{quant_id}', '{frame}', '{macro_type}', '{spin}', 'potato')"
                cursor.execute(insert_query)

        # if source is null, set source to "house"
        cursor.execute("UPDATE quantityann SET source = 'house' WHERE source IS NULL")

        # set macro_type and spin to None if 'None'
        cursor.execute("UPDATE quantityann SET macro_type = NULL WHERE macro_type = 'None'")
        cursor.execute("UPDATE quantityann SET spin = NULL WHERE spin = 'None'")

        # TEST
        select_query = "SELECT quantity_id, type, macro_type, spin, source FROM quantityann"
        cursor.execute(select_query)
        for row in cursor.fetchall():
            print(row)
 
    # check number of potato article anns
    cursor.execute("SELECT COUNT(*) FROM articleann WHERE source = 'potato'")
    print(cursor.fetchone())

    # check number of potato quant anns
    cursor.execute("SELECT COUNT(*) FROM quantityann WHERE source = 'potato'")
    print(cursor.fetchone())

    # # remove potato anns
    # cursor.execute("DELETE FROM articleann WHERE source = 'potato'")
    # cursor.execute("DELETE FROM quantityann WHERE source = 'potato'")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="both",
        help="qual, quant or both"
    )
    args = parser.parse_args()
    main(args)
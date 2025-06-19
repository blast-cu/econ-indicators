import argparse
import json
import sqlite3
import os
import logging
import tqdm

# set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):

    # create a file for the report
    logger.info("Creating report file")
    report_path = "data/reports"
    os.makedirs(report_path, exist_ok=True)
    report_file = os.path.join(report_path, f"psl_final_data_report.txt")

    with open(report_file, "w") as f:
        f.write(f"PSL Final Data Report\n")
        f.write("================\n\n")

        # Connect to the SQLite database
        logger.info(f"Reading in eval data")
        eval_dir = f"{args.data_dir}/eval/"

        f.write(f"Eval Directory: {eval_dir}\n")
        for filename in tqdm(os.listdir(eval_dir)):
            with open(f"{eval_dir}{filename}", 'r') as cf:
                lines = cf.readlines()
            
            ids = [line.split('\t')[0] for line in lines if line.strip() != '']
            ids = list(set(ids))  # remove duplicates

            type = "quants" if "_" in ids[0] else "articles"
            with open(report_file, "a") as f:
                f.write(f"- filename: {filename}, type: {type}, count: {len(ids)}\n")

        f.write("\n\nLearn Directory:\n")
        learn_dir = f"{args.data_dir}/learn/"
        logger.info(f"Reading in learn data")
        for filename in tqdm(os.listdir(learn_dir)):
            with open(f"{learn_dir}{filename}", 'r') as cf:
                lines = cf.readlines()
            
            ids = [line.split('\t')[0] for line in lines if line.strip() != '']
            ids = list(set(ids))
            type = "quants" if "_" in ids[0] else "articles"
            with open(report_file, "a") as f:
                f.write(f"- filename: {filename}, type: {type}, count: {len(ids)}\n")

    logger.info(f"Report created at: {report_file}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Database operations script')
    parser.add_argument("--data_dir", type=str, default="data/clean", help='path to directory containing final psl data')
    args = parser.parse_args()
    main(args)
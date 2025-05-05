import argparse
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This script de-duplicates JSON files in a specified directory based on a given key ('id' or 'url').
"""

def de_duplicate_articles(articles, key):
    """
    De-duplicate articles json based on the specified key ('id' or 'url').
    """
    seen = set()
    unique_articles = []
    for article in articles:
        u_id = article[key]
        
        if u_id not in seen:
            seen.add(u_id)
            unique_articles.append(article)
    
    return unique_articles

def main(args):

    in_path = args.input_dir
    logger.info(f"Processing files in '{in_path}'...")
    for publisher in os.listdir(in_path):

        pub_path = os.path.join(in_path, publisher)
        # Skip if not a directory
        if not os.path.isdir(pub_path):
            logger.info(f"Skipping '{publisher}' as it is not a directory.")
            continue
        
        for file in ["articles.json", "articles_gen_headlines.json"]:
            file_path = os.path.join(pub_path, file)
            if not os.path.exists(file_path):
                logger.info(f"File '{file}' not found in '{publisher}'. Skipping.")
                continue
            
            # Load article texts
            try:
                dataset = json.load(open(file_path))
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from '{file_path}': {e}")
                continue

            # De-duplicate articles
            logger.info(f"De-duplicating articles in '{file_path}'...")
            unique_articles = de_duplicate_articles(dataset, args.key)
            logger.info(f"Original count: {len(dataset)}. Unique count: {len(unique_articles)}.")

            # Save de-duplicated articles (overwriting the original file)
            with open(file_path, "w") as f:
                json.dump(unique_articles, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="De-duplicate files in a directory.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing files to de-duplicate.",
    )
    parser.add_argument(
        "--key",
        type=str,
        required=True,
        help="Key to use for de-duplication. Can be 'id' or 'url'.",
    )
    args = parser.parse_args()
    if args.key not in ["id", "url"]:
        raise ValueError("Key must be either 'id' or 'url'.")
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist.")
    main(args)
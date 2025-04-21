import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
from dotenv import load_dotenv

from helper import S3Functions

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environmental variables
load_dotenv()
PUBLISHERS = os.getenv("publishers", "").split(",")  # default, overwritten by command line arg
AWS_ACCESS_KEY_ID = os.getenv("aws_access_key_id")
AWS_SECRET_ACCESS_KEY = os.getenv("aws_secret_access_key")
BUCKET_NAME = os.getenv("bias_detector_bucket")

# Check that keys and bucket loaded
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not BUCKET_NAME:
    logger.error("AWS credentials or bucket name not found in .env file")
    sys.exit(1)


class S3DataDownloader:
    """Class to handle S3 data downloading operations"""

    def __init__(
        self,
        publishers: list = PUBLISHERS,
        aws_access_key_id: str = AWS_ACCESS_KEY_ID,
        aws_secret_access_key: str = AWS_SECRET_ACCESS_KEY,
        bucket_name: str = BUCKET_NAME,
    ):
        self.s3_helper = S3Functions(
            bucket_name=bucket_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            log_file="../logs/download_s3_data.log",
        )
        self.publishers = publishers

    def query_all_publishers(self, query: str) -> Dict[str, pd.DataFrame]:
        """
        Queries all publishers for the given query

        Args:
            query (str): SQL-like query to run on all publishers

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of results for each publisher
        """
        results = {}
        for publisher in self.publishers:
            logger.info(f"Querying data for publisher: {publisher}")
            # TODO: Implement query functionality
            results[publisher] = self.query_single_publisher(publisher, query)
        return results

    def query_single_publisher(self, publisher: str, query: str) -> pd.DataFrame:
        """
        Queries a single publisher for the given query

        Args:
            publisher (str): Publisher to query
            query (str): SQL-like query to run

        Returns:
            pd.DataFrame: Query results
        """
        # TODO: Implement query functionality
        # Check s3 query docs

    def get_file_in_range(
        self, start_date: datetime, end_date: datetime, data_type: str
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get data for all publishers in the given date range

        Args:
            start_date (datetime): Start date for the range
            end_date (datetime): End date for the range
            data_type (str): Type of data to download ('text' or 'urls')

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: Nested dictionary of publisher data
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        publisher_data = {}
        if data_type not in ["text", "urls"]:
            raise ValueError("Invalid data type. Must be 'text' or 'urls'")
        if data_type == "text":
            file_suffix = "texts"
        else:
            file_suffix = "urls"

        for publisher in self.publishers:
            logger.info(f"Downloading {data_type} data for publisher: {publisher}")
            publisher_data[publisher] = {}

            for date in tqdm(dates):
                year, month, day = date.year, date.month, date.day
                s3_key = (
                    f"{publisher}/{data_type}/{month:02d}_{year}/"
                    f"{month:02d}_{day:02d}_{file_suffix}.csv"
                )
                try:
                    # should return a pandas dataframe
                    df = self.s3_helper.read_csv_from_s3(key_path=s3_key)
                    publisher_data[publisher][f"{year}_{month:02d}_{day:02d}"] = df
                except Exception as e:
                    logger.error(f"Error downloading {s3_key}: {str(e)}")
                    publisher_data[publisher][
                        f"{year}_{month:02d}_{day:02d}"
                    ] = pd.DataFrame()

        return publisher_data


def save_downloaded_data(
    data: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str,
    data_type: str,
    file_format: str = "csv",
    compress: bool = False,
) -> None:
    """
    Saves downloaded data to files in an organized directory structure.

    Args:
        data (Dict[str, Dict[str, pd.DataFrame]]): Nested dictionary containing publisher data
        output_dir (str): Base directory for saving files
        data_type (str): Type of data being saved (e.g., 'text', 'urls')
        file_format (str): Format to save files in ('csv', 'parquet', or 'json')
        compress (bool): Whether to compress the output files

    Directory structure created:
    output_dir/
    ├── data_type/
    │   ├── publisher1/
    │   │   ├── YYYY_MM_DD.csv
    │   │   └── ...
    │   ├── publisher2/
    │   │   ├── YYYY_MM_DD.csv
    │   │   └── ...
    │   └── ...
    └── metadata.json
    """
    try:
        # Create base directory if it doesn't exist
        base_dir = Path(output_dir) / data_type
        base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        metadata = {
            "data_type": data_type,
            "created_at": datetime.now().isoformat(),
            "publishers": {},
            "file_format": file_format,
            "compressed": compress,
        }

        # Process each publisher's data
        for publisher, date_data in data.items():
            publisher_dir = base_dir / publisher
            publisher_dir.mkdir(exist_ok=True)

            metadata["publishers"][publisher] = {
                "file_count": 0,
                "total_rows": 0,
                "dates": [],
            }

            # Save each date's data
            for date_str, df in date_data.items():
                if df is None or df.empty:
                    continue

                file_name = f"{date_str}"
                metadata["publishers"][publisher]["dates"].append(date_str)
                metadata["publishers"][publisher]["total_rows"] += len(df)
                metadata["publishers"][publisher]["file_count"] += 1

                if file_format == "csv":
                    file_path = publisher_dir / f"{file_name}.csv"
                    if compress:
                        file_path = file_path.with_suffix(".csv.gz")
                        df.to_csv(file_path, compression="gzip", index=False)
                    else:
                        df.to_csv(file_path, index=False)

                elif file_format == "parquet":
                    file_path = publisher_dir / f"{file_name}.parquet"
                    df.to_parquet(file_path, compression="snappy" if compress else None)

                elif file_format == "json":
                    file_path = publisher_dir / f"{file_name}.json"
                    if compress:
                        file_path = file_path.with_suffix(".json.gz")
                        df.to_json(file_path, compression="gzip", orient="records")
                    else:
                        df.to_json(file_path, orient="records")

                logging.info(f"Saved {file_path}")

        # Save metadata
        metadata_path = base_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Data successfully saved to {base_dir}")
        logging.info(f"Metadata saved to {metadata_path}")

    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Download and process S3 data")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--publishers", type=str, help="Comma-separated list of publishers"
    )
    # TODO: Support more data types
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["text", "urls"],
        default="text",
        help="Type of data to download",
    )
    parser.add_argument("--query", type=str, help="Query to run on the data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./downloaded_data",
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "json"],
        default="csv",
        help="Output file format",
    )
    parser.add_argument("--compress", action="store_true", help="Compress output files")
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    downloader = S3DataDownloader()

    # TODO: Implement query functionality
    if args.query:
        results = downloader.query_all_publishers(args.query)
        for publisher, data in results.items():
            print(f"\nResults for {publisher}:")
            print(data)

    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        data = downloader.get_file_in_range(start_date, end_date, args.data_type)
        print(
            f"\nDownloaded {args.data_type} data for date range: "
            f"{args.start_date} to {args.end_date}"
        )
        print("Saving downloaded data...")
        save_downloaded_data(
            data,
            output_dir=args.output_dir,
            data_type=args.data_type,
            file_format=args.format,
            compress=args.compress,
        )


if __name__ == "__main__":
    main()

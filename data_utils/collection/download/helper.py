import json
import logging
import os
import re
import sys
from io import StringIO

import boto3
import botocore
import pandas as pd
from dotenv import load_dotenv

sys.path.append("../NewsClassifier")
from preprocessing import preprocess_dict


class S3Functions:
    def __init__(self, bucket_name, aws_access_key_id, aws_secret_access_key, log_file):
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.s3_resource = session.resource("s3")
        self.bucket_name = bucket_name
        self.bucket = self.s3_resource.Bucket(name=bucket_name)
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )

        # Logging specific
        logging.basicConfig(
            filename=log_file, format="%(asctime)s %(message)s", filemode="a"
        )

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # DEBUG, INFO, WARN

    ############## UPLOAD ##############

    def upload_to_s3(self, in_file, out_path):
        """
        Upload data to the object.

        :param data: The data to upload. This can either be bytes or a string. When this
                        argument is a string, it is interpreted as a file name, which is
                        opened in read bytes mode.
        """
        try:
            print(f"Uploading {in_file}")
            self.bucket.upload_file(Filename=in_file, Key=out_path)
        except Exception as e:
            self.logger.error(e)

    ############## DOWNLOAD ##############

    def download_from_s3(self, in_key, out_file):
        """
        Download data from s3 bucket to local bucket object

        Input:
            in_key: str - Key to the file in s3
            out_file: str - Path to save the downloaded file
        Output: None
        """
        try:
            self.bucket.download_file(in_key, out_file)
            print("downloaded file")
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("The object does not exist.")
            else:
                raise

    ################ READ ##############

    def read_csv_from_s3(self, key_path):
        """
        Read CSV file from s3 to pandas dataframe

        Input: key_path: str - Key path to the CSV file
        Output: pd.DataFrame - DataFrame of CSV file
        """
        try:
            csv_obj = self.s3_resource.Object(
                bucket_name=self.bucket_name, key=key_path
            )
            body = csv_obj.get()["Body"]
            csv_string = body.read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_string))
            return df
        except Exception as e:
            print(f"Error reading from S3 bucket {e}")

    def read_json_from_s3(self, key_path):
        """
        Read JSONL file (no commas between objects!) from s3 to pandas dataframe

        Input: key_path: str - Key path to the JSONL file
        Output: pd.DataFrame - DataFrame of JSONL file
        """
        try:
            json_obj = self.client.get_object(Bucket=self.bucket_name, Key=key_path)
            body = json_obj["Body"]
            json_lines = body.read().decode("utf-8").splitlines()
            df = pd.DataFrame([json.loads(line) for line in json_lines])
            return df
        except Exception as e:
            print(f"Error in read_json_from_s3 {e}")

    def filter_files(self, prefix):
        """
        Provide a key prefix then filter for objects under that key

        Input: prefix: str - Key prefix to filter for
        Output: list - List of objects under the key prefix
        """
        return self.bucket.objects.filter(Prefix=prefix)


###########################################################################################


def deduplicate(df):
    """
    Remove duplicate articles from a day of scraped text, keeps most recent version

    Input:
        df: pd.DataFrame - DataFrame of scraped articles
    Output:
        pd.DataFrame - DataFrame of deduplicated articles
    """
    top_ten = df[df["rank"] <= 10]
    return top_ten.sort_values(by="datetime").drop_duplicates(subset="url", keep="last")


def preprocess(text, publisher):
    """
    Preprocesses text using publisher-specific preprocessing strings

    Input:
        text: str - Article text
        publisher: str - Publisher name
    Output:
        str - Preprocessed text
    """
    # if type(text) is str:
    #     text = text.replace("\n", " ")
    if isinstance(text, str):
        text = text.replace("\n", " ")
    else:
        logging.info(
            f"Article text for {publisher} is not a string. Returning empty string"
        )
        return ""

    keys = [publisher, "all", "regex_patterns"]
    for key in keys:
        for bad_phrase in preprocess_dict[key]:
            try:
                bad_phrase = bad_phrase.strip()
                text = re.sub(bad_phrase, "", text)
            except Exception as e:
                logging.error(f"Error processing phrase '{bad_phrase}': {e}")

    return text.strip()

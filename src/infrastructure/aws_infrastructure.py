import yaml
import os
from pathlib import Path
from src.globals import PROJECT_DIR
import boto3
import torchdata.datapipes as dp
import requests
import time


def get_aws_credentials():
    with open(Path(PROJECT_DIR / "aws_credentials.yaml"), "r") as file:
        content = yaml.safe_load(file)
    return content


def set_s3_credentials(aws_credentials):
    os.environ["AWS_ACCESS_KEY_ID"] = aws_credentials["s3"]["public_key"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_credentials["s3"]["secret_key"]


def download_from_s3(bucket: str, remote_path: str, local_path: str, s3_client=None):
    if not s3_client:
        s3_client = boto3.client("s3")
    s3_client.download_file(bucket, remote_path, local_path)


def upload_file_to_s3(bucket: str, remote_path: str, local_path: str, s3_client=None):
    if not s3_client:
        s3_client = boto3.client("s3")
    s3_client.upload_file(Filename=local_path, Bucket=bucket, Key=remote_path)


def split_bucket_from_path(x: str):
    """Separate bucket name and path in bucket from full string.

    Args:
        x (str): full string

    Returns:
        tuple: (bucket_name, path)
    """
    return "".join(x.split("/")[2:3]), "/".join(x.split("/")[3:])


def get_s3_folder_content(bucket_path=f"s3://mi4people-soil-project/BigEarthNet-v1.0/"):
    """Gets all top level-elements in buckets path.

    Args:
        bucket_path (string, optional): Path in bucket to list elements from. Defaults to f"s3://mi4people-soil-project/BigEarthNet-v1.0/".

    Returns:
        list: list of full paths of all top-level-elements.
    """
    top_pipe = dp.iter.IterableWrapper([bucket_path])
    top_pipe = top_pipe.list_files_by_fsspec()
    folders = list(top_pipe)
    return folders


def spot_instance_terminating():
    status_code = requests.get("http://169.254.169.254/latest/meta-data/spot/instance-action").status_code
    if status_code != 404:
        return True
    return False

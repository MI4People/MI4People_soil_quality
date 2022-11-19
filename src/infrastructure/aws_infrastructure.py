import yaml
import os
from pathlib import Path
from src.globals import PROJECT_DIR


def get_aws_credentials():
    with open(Path(PROJECT_DIR / "aws_credentials.yaml"), "r") as file:
        content = yaml.safe_load(file)
    return content


def set_s3_credentials(aws_credentials):
    os.environ["AWS_ACCESS_KEY_ID"] = aws_credentials["s3"]["public_key"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_credentials["s3"]["secret_key"]

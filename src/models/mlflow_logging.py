import mlflow
import sys
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


def create_or_set_experiment(experiment_name: str):
    credentials = get_aws_credentials()
    mlflow.set_tracking_uri(
        f"mysql+pymysql://{credentials['mysql']['user']}:{credentials['mysql']['password']}@mlflow-backend.chf6ry9cdkyl.eu-central-1.rds.amazonaws.com:3306/mlflowbackend"
    )
    set_s3_credentials(credentials)
    s3_bucket = "s3://mi4people-soil-project/mlflow-artifacts/"

    mlflow.end_run()
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not existing_experiment:
        print("Creating new Mlflow-Experiment.")
        mlflow.create_experiment(experiment_name, artifact_location=s3_bucket)
    else:
        print("An experiment with that name already exists, logging new run into it.")
    mlflow.set_experiment(experiment_name)


def start_auto_logging(experiment_name: str, model_library: str):
    create_or_set_experiment(experiment_name)

    if model_library.lower() == "sklearn":
        mlflow.sklearn.autolog()
    elif model_library.lower() == "pytorch":
        mlflow.pytorch.autolog()
    else:
        print(
            "Please choose <sklearn> or <pytorch> for autologging or use manual mlflow logging, as described in the mlflow logging notebook. \n Aborting run."
        )
        sys.exit()

import mlflow

# import pymysql
import sys
import yaml
from pathlib import Path
from src.globals import PROJECT_DIR


def get_mlflow_credentials():
    # print(src.globals.PROJECT_DIR)
    print(Path(PROJECT_DIR / "aws_credentials.yaml"))
    with open(Path(PROJECT_DIR / "aws_credentials.yaml"), "r") as file:
        content = yaml.safe_load(file)
    return content


def create_or_set_experiment(experiment_name: str):
    credentials = get_mlflow_credentials()
    mlflow.set_tracking_uri(
        f"mysql+pymysql://{credentials['user']}:{credentials['password']}@mlflow-backend.chf6ry9cdkyl.eu-central-1.rds.amazonaws.com:3306/mlflowbackend"
    )
    s3_bucket = "s3://mi4people-soil-project/mlflow-artifacts/"
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if not existing_experiment:
        print("Creating new Mlflow-Experiment.")
        mlflow.create_experiment(experiment_name, artifact_location=s3_bucket)
    else:
        # TODO test if tracking & artifact-locations are still right then
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


if __name__ == "__main__":
    print(get_mlflow_credentials())

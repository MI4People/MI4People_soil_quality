import mlflow
import sys
from src.infrastructure.aws_infrastructure import (
    get_aws_credentials,
    set_s3_credentials,
)


def establish_mlflow_connection():
    """Set tracking uri for mlflow database. Enables logging and retrieving info from our mlflow db.
    """
    credentials = get_aws_credentials()
    mlflow.set_tracking_uri(
        f"mysql+pymysql://{credentials['mysql']['user']}:{credentials['mysql']['password']}@mlflow-backend.chf6ry9cdkyl.eu-central-1.rds.amazonaws.com:3306/mlflowbackend"
    )
    set_s3_credentials(credentials)


def create_or_find_experiment(experiment_name: str):
    """Creates a new experiment by name or retrieves its data from the database if it already exists.

    Args:
        experiment_name (str)

    Returns:
        mlflow.entities.experiment.Experiment
    """
    establish_mlflow_connection()
    s3_bucket = "s3://mi4people-soil-project/mlflow-artifacts/"
    # end existing runs to cancel weird side-effects
    mlflow.end_run()
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment:
        print("An experiment with that name already exists, logging new run into it.")
        return existing_experiment
    else:
        print("Creating new Mlflow-Experiment.")
        mlflow.create_experiment(experiment_name, artifact_location=s3_bucket)
        new_experiment = mlflow.get_experiment_by_name(experiment_name)
        return new_experiment


def start_auto_logging(experiment: str | mlflow.entities.experiment.Experiment, model_library: str):
    """Start mlflow autologging and sets the experiment as the active one.
       Models and metrics still need to be logged by calling the appropiate functions, except when using Pytorch Lightning.
       Creates or sets an experiment first if called with an experiment name string.

    Args:
        experiment (str | mlflow.Experiment): Name of the experiment or Experiment object.
        model_library (str): "Sklearn" or "Pytorch".

    Returns:
        mlflow.Experiment: The experiment for which logging is started.
    """
    establish_mlflow_connection()
    if isinstance(experiment, (str, mlflow.entities.experiment.Experiment)):
        # Returns always an Experiment object
        experiment = create_or_find_experiment(experiment)

    if model_library.lower() == "sklearn":
        mlflow.sklearn.autolog()
    elif model_library.lower() == "pytorch":
        mlflow.pytorch.autolog()
    else:
        print(
            "Please choose <sklearn> or <pytorch> for autologging or use manual mlflow logging, as described in the mlflow logging notebook. \n Aborting run."
        )
        sys.exit()
    mlflow.set_experiment(experiment.name)
    return experiment


def get_latest_model(model_name: str, model_version: str = "latest"):
    """Returns the latest model from the model registry.

    Args:
        # TODO other frameworks than pytorch
        model_name (str): Name under which the model was registered (see the table "registered models").
        model_version (str, optional): Which version to retrieve. Either an int-like string or "latest". Defaults to "latest".

    Returns:
        Model of the respective type.
    """
    establish_mlflow_connection()
    # TODO sklearn
    model = mlflow.pytorch.load_model(f"models:/{model_name}/{str(model_version)}")
    return model


if __name__ == "__main__":
    # Example: Get latest pytorch model from s3 which is registered in our mlflow-db
    get_latest_model("ben_res50")

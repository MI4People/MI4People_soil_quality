"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from mi4people_soil_quality.pipelines import test_pipeline as tp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    test_pipeline = tp.create_pipeline()
    return {"tp": test_pipeline, "__default__": pipeline([])}

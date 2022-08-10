from .nodes import get_unique_uuids
from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                get_unique_uuids,
                inputs="raw_targets_and_inputs",
                outputs="unique_uuids",
                name="get_unique_uuids",
            )
        ]
    )

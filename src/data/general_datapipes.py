import torchdata.datapipes as dp
import random


def get_s3_folder_content(bucket_path=f"s3://mi4people-soil-project/BigEarthNet-v1.0/"):
    """Gets all top level-elements in buckets path.

    Args:
        bucket_path (string, optional): Path in bucket to list elements from. Defaults to f"s3://mi4people-soil-project/BigEarthNet-v1.0/".
    Returns:
        list: list of full paths of all top-level-elements.
    """
    top_pipe = dp.iter.IterableWrapper([bucket_path])
    top_pipe = top_pipe.list_files_by_fsspec()
    return top_pipe


def split_pipe_to_train_test(datapipe, test_percentage):
    # TODO extend to train-test-val
    # TODO build something similar for geospatial split?
    """Generates two pipes from one, splitting the output to create a train-test-split.

    Args:
        datapipe (_type_): The original pipe
        test_percentage (_type_): Fraction of data used for the test-set.
    """
    def assign_sample(n):
        return random.uniform(0, 1) < test_percentage
    train_pipe, test_pipe = datapipe.demux(num_instances=2, classifier_fn=assign_sample)
    return train_pipe, test_pipe

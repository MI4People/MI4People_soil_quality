import numpy as np
import torchdata.datapipes as dp
import json
import fsspec
from PIL import Image
from src.globals import LABELS_TO_INDS
from ingestion import combine_and_resize_bands
from src.features.img_engineering import get_first_n_pcs


""" Functions for torchdata datapipes specific for the bigearthnet dataset as well as pipeline definitions.
"""


def group_key_by_folder(path_and_stream):
    """Generates string of image-name as a key to group all bands and labels of one image.

    Args:
        path_and_stream (tuple): Generated by list_files_by_fsspec, pairs of file-(s3)-path and StreamWrapper objects.

    Returns:
        str: image id
    """
    return "".join(path_and_stream[0].split("/")[-1].split("_")[:5])


def chunk_to_dataloader_dict(chunk):
    """Transforms chunks into dict for datalaoder.

    Args:
        chunk : Output of groupBy

    Returns:
        dict: dict with keys "label" and "data",
    """
    json_files = [path for path in chunk if path.endswith(".json")]
    assert len(json_files) == 1, "Only one label json per image is permitted."
    image_files = [path for path in chunk if path.endswith(".tif")]
    return {"label": json_files[0], "data": image_files}


def read_json_from_path(path):
    # reads a json to a dict from a s3-path.
    json_content = json.loads(fsspec.open(path, mode="r").open().read())
    return json_content


def read_imgs_from_paths(path_list):
    # reads an image file to a dict from a s3-path.
    image_contents = [
        np.array(Image.open(fsspec.open(path, mode="rb").open())) for path in path_list
    ]
    return image_contents


def get_labels(meta_dict):
    return meta_dict["labels"]


def get_label_inds(label_list):
    # Converts lists of multiple bigearth labels per image in a list to lists of ints.
    return [LABELS_TO_INDS[label] for label in label_list]


def pca_on_label_and_data(combined_image):
    # Perform PCA on a single array of shape (height, width, bands) to (height, width, 3).
    return get_first_n_pcs(combined_image, num_components=3)


def get_bigearth_pca_pipe(folders):
    """Given a list of paths (in a bucket), returns a datapipe with simple 3-component PCA performed on images.
        May be passed in the contruction of a dataloader in  the dataset argument.

    Args:
        folders (list): List of folders to process. Each folder must contain 12 tifs, one for each band and one json with metadata.

    Returns:
        torch.utils.data.datapipes.iter.callable.MapperIterDataPipe: Iterable datapipe yielding dicts with keys "data" and "label".
            Data is a single numpy array of the 3 first PCs of all bands, label a list of integers from /data/raw/bigearth_labels.json.
    """
    img_pipe = dp.iter.IterableWrapper(folders)
    img_pipe = img_pipe.list_files_by_fsspec()
    img_pipe = img_pipe.groupby(group_key_fn=group_key_by_folder, group_size=13)
    img_pipe = img_pipe.map(chunk_to_dataloader_dict)
    img_pipe = img_pipe.map(read_json_from_path, input_col="label")
    img_pipe = img_pipe.map(get_labels, input_col="label")
    img_pipe = img_pipe.map(get_label_inds, input_col="label")
    img_pipe = img_pipe.map(read_imgs_from_paths, input_col="data")
    img_pipe = img_pipe.map(combine_and_resize_bands, input_col="data")
    img_pipe = img_pipe.map(pca_on_label_and_data, input_col="data")
    return img_pipe

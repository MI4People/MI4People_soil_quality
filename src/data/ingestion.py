import pandas as pd
import intake
from typing import List
import numpy as np
import cv2
from src.globals import PROJECT_DIR, SRC_DIR, CATALOG
from src.data.decorators import csv_ingestion_decorator, img_ingestion_decorator


@csv_ingestion_decorator
def read_csv(path):
    return pd.read_csv(path)


@csv_ingestion_decorator
def read_csv_from_catalog(datasource_name: str):
    cat = intake.open_catalog(CATALOG)
    datasource = cat[datasource_name].read()
    return datasource


@csv_ingestion_decorator
def delete_broken_rows(df: pd.DataFrame):
    try:
        # No sentinel data for this olc_id available
        df = df[df.olc_id != "5G55HPG4+MM7"]
    except KeyError or ValueError:
        print(
            "Tried to delete olc_id, '5G55HPG4+MM7', did not find it. Passing uncleaned df."
        )
    return df


def combine_and_resize_bands(arrays: List[np.array], max_res=(120, 120)):
    """Combines bands into one array, upscaling each band to max_res.

    Args:
        arrays (list): list of bands
        max_res (tuple, optional): Max height & width to resize bands to. Defaults to (120, 120).

    Returns:
        np.array: Concatenated & resized bands.
    """
    
    result = np.zeros(shape=(*max_res, len(arrays)))
    for i, array in enumerate(arrays):
        assert array.shape[0] <= max_res[0]
        assert array.shape[1] <= max_res[1]
        result[:, :, i] = cv2.resize(array, dsize=max_res)

    # TODO: Add additional transformations of the input data here
    # Suggestion: Make of of library torchvision.transforms as in the example but not mandatory, other approaches should also be possible
    # Think about if some transformations should only be applied to training data and not to validation data. 
    # If so, think about if train_test-split should be done already before that step
        
    # Example:
    # data_transforms = {
    #     'train':
    #     transforms.Compose([
    #         transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), # TODO: an idea, check if necessary
    #         transforms.RandomHorizontalFlip(), # TODO: an idea, check if necessary
    #         transforms.ToTensor(), # TODO: check if necessary or if it should be done later
    #     ]),
    #     'validation':
    #     transforms.Compose([
    #         transforms.ToTensor(), # TODO: check if necessary or if it should be done later
    #     ]),
    # }
    return result

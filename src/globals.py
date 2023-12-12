from pathlib import Path
from typing import Union
import pickle
import warnings


PROJECT_DIR = Path(__file__).parents[1]
SRC_DIR = Path(__file__).parents[0]
DATA_DIR = PROJECT_DIR / "data"
CATALOG = Path(__file__).parents[1] / "catalog.yaml"


def get_bigearth_labels_from_json() -> Union[dict, None]:
    """Returns all potential labels for BigEarthNet dataset if available"""
    try:
        with open(DATA_DIR / "raw/ben_labels.pickle", "rb") as label_file:
            ind_to_labels = pickle.load(label_file)
            labels_to_ind = {l: int(i) for (i, l) in ind_to_labels.items()} if ind_to_labels else None
        return labels_to_ind
    except FileNotFoundError:
        warnings.warn("If you want to make use of global variable LABELS_TO_INDS containing all labels for the BigEarthNet-v1.0 (BEN) dataset, you need to execute the notebook `bigearthnet_label_creation.ipnyb` first.")
        return None

#LABELS_TO_INDS = get_bigearth_labels_from_json()

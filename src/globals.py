from pathlib import Path
import json
import pickle

PROJECT_DIR = Path(__file__).parents[1]
SRC_DIR = Path(__file__).parents[0]
DATA_DIR = PROJECT_DIR / "data"
CATALOG = Path(__file__).parents[1] / "catalog.yaml"


def get_bigearth_labels_from_json():
    try:
        with open(DATA_DIR / "raw/bigearth_labels.json", "r") as label_file:
            ind_to_labels = json.load(label_file)
    except FileNotFoundError:
        with open(DATA_DIR / "raw/ben_labels", "rb") as label_file:
            ind_to_labels = pickle.load(label_file)
    labels_to_ind = {l: int(i) for (i, l) in ind_to_labels.items()}
    return labels_to_ind


LABELS_TO_INDS = get_bigearth_labels_from_json()

import pandas as pd
import numpy as np
from pathlib import Path

from src.globals import PROJECT_DIR, SRC_DIR, CATALOG
from src.features.decorators import csv_preprocessing_decorator


@csv_preprocessing_decorator
def filter_corrupted_sites(df):
    corrupted_sites = pd.read_csv(
        PROJECT_DIR / "data/raw/RaCA2016_and_ISCND_points.csv"
    )["olc_id"]
    df = df[~df["olc_id"].isin(corrupted_sites)]
    return df


@csv_preprocessing_decorator
def drop_high_oc(df: pd.DataFrame):
    return df[df["oc"] < 120]


@csv_preprocessing_decorator
def fill_na(data: pd.DataFrame, fill_with: int = 0):
    return data.fillna(fill_with, inplace=True)


@csv_preprocessing_decorator
def minmax_scale_columns(data: pd.DataFrame):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    data = (data - mins) / (maxs - mins)
    return data


@csv_preprocessing_decorator
def log_column(data: pd.DataFrame):
    return np.log(data, axis=0)

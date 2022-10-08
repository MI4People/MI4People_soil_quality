import pandas as pd
import numpy as np

from src.features.decorators import csv_preprocessing_decorator


@csv_preprocessing_decorator
def drop_high_oc(df):
    return df[df["oc"] < 120]


@csv_preprocessing_decorator
def fill_na(data, fill_with=0):
    return data.fillna(fill_with, inplace=True)


@csv_preprocessing_decorator
def minmax_scale_columns(data: pd.DataFrame):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    data = (data - mins) / (maxs - mins)
    return data


@csv_preprocessing_decorator
def log_column(data):
    return np.log(data, axis=0)

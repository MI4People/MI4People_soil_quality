import pandas as pd
import numpy as np


# target_and_input_OpenLandMap = target_and_input_OpenLandMap[target_and_input_OpenLandMap.olc_id != "5G55HPG4+MM7"]


def drop_high_oc(df):
    return df[df["oc"] < 120]


def fill_na(data, fill_with=0):
    return data.fillna(fill_with, inplace=True)


def minmax_scale_columns(data: pd.DataFrame):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    data = (data - mins) / (maxs - mins)
    return data


def log_column(data):
    return np.log(data, axis=0)

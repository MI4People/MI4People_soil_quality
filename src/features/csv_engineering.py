import pandas as pd
import numpy as np

from src.features.decorators import csv_preprocessing_decorator


@csv_preprocessing_decorator
def drop_empty_oc(df):
    return df.dropna(subset=['oc'], inplace=True)


@csv_preprocessing_decorator
def drop_high_oc(df):
    return df[df["oc"] < 120]


@csv_preprocessing_decorator
def fill_na(data, fill_with=0):
    return data.fillna(fill_with, inplace=True)


@csv_preprocessing_decorator
def drop_columns(df):
    df = df.drop(['olc_id', 'confidence_degree', 'uuid', 'site_obsdate', 'source_db', 'layer_sequence.f', 'hzn_top',
                  'hzn_bot', 'n_tot', 'ph_h2o'], axis=1, inplace=True)
    return df


@csv_preprocessing_decorator
def minmax_scale_columns(data: pd.DataFrame):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    data = (data - mins) / (maxs - mins)
    return data


@csv_preprocessing_decorator
def log_column(data):
    return np.log(data, axis=0)


@csv_preprocessing_decorator
def separate_input_data_target(data, target_name):
    input_data = data.drop(['oc'], axis=1)
    target = data[target_name]
    return input_data, target


@csv_preprocessing_decorator
def separate_train_validate_test(data, frac_train=.6, frac_validate=.8):
    train, validate, test = np.split(data.sample(frac=1, random_state=42),
                                     [int(frac_train * len(data)), int(frac_validate * len(data))]
                                     )
    return train, validate, test


def to_numpy(data):
    return data.to_numpy()

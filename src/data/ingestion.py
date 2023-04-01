import pandas as pd

from src.data.decorators import csv_ingestion_decorator


@csv_ingestion_decorator
def read_csv(path):
    return pd.read_csv(path)


@csv_ingestion_decorator
def delete_broken_rows(df):
    try:
        # No sentinel data for this olc_id available
        df = df[df.olc_id != "5G55HPG4+MM7"]
    except KeyError or ValueError:
        print(
            "Tried to delete olc_id, '5G55HPG4+MM7', did not find it. Passing uncleaned df."
        )
    return df


@csv_ingestion_decorator
def merge_df(df, df_sentinel_2, on='olc_id'):
    df = df.merge(df_sentinel_2, on=on, how='left')
    return df

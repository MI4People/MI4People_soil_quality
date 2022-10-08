import pandas as pd

from src.data.decorators import ingestion_decorator


@ingestion_decorator
def read_csv(path):
    return pd.read_csv(path)


@ingestion_decorator
def delete_broken_rows(df):
    try:
        # No sentinel data for this olc_id available
        df = df[df.olc_id != "5G55HPG4+MM7"]
    except KeyError or ValueError:
        print(
            "Tried to delete olc_id, '5G55HPG4+MM7', did not find it. Passing uncleaned df."
        )
    return df

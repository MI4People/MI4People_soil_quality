import pandas as pd
import intake

from src.globals import PROJECT_DIR, SRC_DIR, CATALOG
from src.data.decorators import csv_ingestion_decorator


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

import pandas as pd

from src.data.decorators import ingestion_decorator


@ingestion_decorator
def read_csv(path):
    return pd.read_csv(path)

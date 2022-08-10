import pandas as pd


def get_unique_uuids(data: pd.DataFrame) -> list:
    return data["uuid"].unique()

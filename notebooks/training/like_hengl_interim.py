import src.data.ingestion as ingestion
import src.features.csv_engineering as csv_engineering

# TODO: Add explanations
# TODO: Search file

# Ingest data
df = ingestion.read_csv("Target_and_input.csv")
df = ingestion.delete_broken_rows(df)
df_sentinel_2 = ingestion.read_csv("pixels_df.csv")
data = ingestion.merge_df(df, df_sentinel_2)

# Data cleaning
data = csv_engineering.drop_empty_oc(data)
data = csv_engineering.drop_high_oc(data)
data = csv_engineering.drop_columns(data)
# Neue Reihenfolge
data = csv_engineering.fill_na(data)

# Model training


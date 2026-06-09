from google.cloud import bigquery
from sqlalchemy import create_engine, text
import pandas as pd
from config import (PROJECT_ID, DATASET, TABLE, CREDENTIALS_PATH, PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB)

def extract_from_bigquery():
    print("Connecting to BigQuery...")
    client = bigquery.Client.from_service_account_json(CREDENTIALS_PATH, project=PROJECT_ID)

    query = f"""
    SELECT N, P, K, temperature, humidity, ph, rainfall, label
    FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
""" 

    print("Extracting data from BigQuery...")
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    print(f"{len(df)} rows extracted from BigQuery.")

    return df

def load_to_postgresql(df):
    print("Connecting to PostgreSQL...")
    engine = create_engine(f'postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}')

    print("Loading data into PostgreSQL...")
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS crop"))
        conn.commit()

    df.to_sql('crop_data', engine, schema='crop', if_exists='replace', index=False)
    print(f"{len(df)} rows loaded into PostgreSQL.")

if __name__ == "__main__":
    df = extract_from_bigquery()
    load_to_postgresql(df)
    print("ETL process completed successfully.")
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine

from config import PG_DB, PG_HOST, PG_PASSWORD, PG_PORT, PG_USER

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_data_from_postgres() -> pd.DataFrame:
    print("Loading data from PostgreSQL...")
    engine = create_engine(
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
    df = pd.read_sql("SELECT * FROM crop.crop_data", engine)
    print(f"{len(df)} rows loaded.")
    return df


def train_classifier(df: pd.DataFrame) -> None:
    print("\nTraining classifier (XGBoost)...")
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, target_names=le.classes_)
    )

    os.makedirs("models", exist_ok=True)
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump((model, le), f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Classifier saved.")


def train_regressor(df: pd.DataFrame) -> None:
    print("\nTraining regressor (XGBoost)...")

    np.random.seed(42)

    df["yield"] = (
        df["N"] * 0.3
        + df["P"] * 0.2
        + df["K"] * 0.2
        + df["rainfall"] * 0.1
        + df["temperature"] * 0.5
        + np.random.normal(0, 5, len(df))
    ).round(2)

    X = df[FEATURE_COLUMNS]
    y = df["yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"RMSE: {rmse:.2f}")

    with open("models/regressor.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Regressor saved.")


if __name__ == "__main__":
    df = load_data_from_postgres()
    train_classifier(df)
    train_regressor(df)
    print("\nTraining completed!")

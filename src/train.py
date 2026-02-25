# src/train.py
from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import json
import joblib
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from src.config import (
    DATA_RAW, MODELS_DIR, REPORTS_DIR,
    TARGET_COL, TEST_SIZE, RANDOM_STATE
)
from src.preprocessing import load_raw, basic_clean, split_features_target, infer_column_types

def build_pipeline(numeric_cols, categorical_cols) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def main():
    df = basic_clean(load_raw(DATA_RAW))
    X, y = split_features_target(df)
    numeric_cols, categorical_cols = infer_column_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "xgb_fair_price.joblib"
    joblib.dump(pipe, model_path)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "model_path": str(model_path),
    }

    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to:", model_path)
    print("MAE:", mae)
    print("RMSE:", rmse)


if __name__ == "__main__":
    main()
# src/preprocessing.py
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import pandas as pd
from pathlib import Path
from typing import Tuple, List

from src.config import TARGET_COL

def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load raw data from CSV into pandas dataframe. Standardize column names by stripping whitespace."""
    df = pd.read_csv(csv_path)
    # Standardize column names a bit by removing leading/trailing whitespace
    df.columns = [c.strip() for c in df.columns]
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning steps:
    - Drop rows missing target
    - Remove obviously invalid target values (e.g. zero or negative prices)
    - Try to coerce numerics where possible (e.g. "10000" -> 10000)
    """
    # Drop rows missing target
    df = df.dropna(subset=[TARGET_COL]).copy()

    # Remove obviously invalid target values -- e.g. zero or negative prices
    df = df[df[TARGET_COL] > 0]

    # Try to coerce numerics where possible
    for col in df.columns:
        if col == TARGET_COL:
            continue
        # If it's object, attempt numeric conversion
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target. Raises if target column is missing."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def save_processed(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
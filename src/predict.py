# src/predict.py
from __future__ import annotations

import joblib
import pandas as pd

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.config import (
    DATA_RAW, MODELS_DIR, REPORTS_DIR,
    TARGET_COL, FLIP_COST, SAFETY_MARGIN
)
from src.preprocessing import load_raw, basic_clean



def main():
    df = basic_clean(load_raw(DATA_RAW))

    model_path = MODELS_DIR / "xgb_fair_price.joblib"
    pipe = joblib.load(model_path)

    X = df.drop(columns=[TARGET_COL])
    df["predicted_price"] = pipe.predict(X)
    df["undervalued"] = df["predicted_price"] - df[TARGET_COL]
    df["estimated_profit"] = df["undervalued"] - FLIP_COST
    df["deal_flag"] = df["estimated_profit"] > SAFETY_MARGIN

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "deals.csv"
    df.sort_values("estimated_profit", ascending=False).to_csv(out_path, index=False)

    print("Wrote deals to:", out_path)
    print("Top 10 deals:")
    print(df.sort_values("estimated_profit", ascending=False).head(10)[
        [TARGET_COL, "predicted_price", "undervalued", "estimated_profit", "deal_flag"]
    ])


if __name__ == "__main__":
    main()
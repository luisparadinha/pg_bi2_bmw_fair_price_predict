import numpy as np
import pandas as pd
from pathlib import Path
import pandas as pd

def add_profit_columns(
    df: pd.DataFrame,
    price_col: str = "price",
    market_value_col: str = "market_value",
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Adds:
      - profit: market_value - price
      - profit_capped: profit clipped between the 1st and 99th percentiles (like Excel PERCENTILE.INC)
    """
    out = df.copy()

    # Ensure numeric
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out[market_value_col] = pd.to_numeric(out[market_value_col], errors="coerce")

    # Profit
    out["profit"] = out[market_value_col] - out[price_col]

    # Percentile caps (Excel PERCENTILE.INC equivalent)
    valid_profit = out["profit"].dropna()
    if len(valid_profit) == 0:
        out["profit_capped"] = np.nan
        return out

    p_low = valid_profit.quantile(lower_pct)
    p_high = valid_profit.quantile(upper_pct)

    # Profit capped = MIN(MAX(profit, p_low), p_high)
    out["profit_capped"] = out["profit"].clip(lower=p_low, upper=p_high)

    return out


def save_features_to_csv(
    df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
    filename: str = "bmw_feat_eng.csv",
) -> Path:
    """
    Save engineered features to CSV.
    Does not auto-run — must be called explicitly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)

    print(f"Saved engineered features to: {output_path}")
    return output_path
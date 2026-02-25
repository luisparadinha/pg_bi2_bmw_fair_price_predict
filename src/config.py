# src/config.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "bmw.csv"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED = DATA_PROCESSED_DIR / "bmw_processed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Modeling
TARGET_COL = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# "Flipping" assumptions (edit later)
FLIP_COST = 1500          # repairs/fees/transport buffer
SAFETY_MARGIN = 500       # minimum profit to call it a "deal"
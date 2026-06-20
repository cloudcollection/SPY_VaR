from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
FORECASTS_DIR = OUTPUT_DIR / "forecasts"
REPORT_DIR = PROJECT_ROOT / "report"

WINDOW_SIZE = 500
FORECAST_HORIZON = 1
VAR_LEVELS = [0.01, 0.05, 0.10]
VAR_COLUMN_MAP = {0.01: "VaR_1", 0.05: "VaR_5", 0.10: "VaR_10"}
REQUIRED_COLUMNS = ["log_ret", "rv5", "bv"]


def ensure_directories() -> None:
    for path in [DATA_DIR, FIGURES_DIR, TABLES_DIR, FORECASTS_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _find_date_column(columns: Iterable[str]) -> str | None:
    candidates = ["date", "datetime", "time", "timestamp", "Date", "DATE"]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_spy_data(path: Path | None = None) -> pd.DataFrame:
    """Load SPY data without disturbing the original time order."""
    ensure_directories()
    path = path or DATA_DIR / "spy_data.csv"
    df = pd.read_csv(path)

    # A CSV saved from a DataFrame often has an unnamed date index column.
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    date_col = _find_date_column(df.columns)
    if date_col is None and unnamed_cols:
        parsed = pd.to_datetime(df[unnamed_cols[0]], errors="coerce")
        if parsed.notna().mean() > 0.95:
            date_col = unnamed_cols[0]

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        df.index.name = "date"
    else:
        df.index.name = "index"

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def index_label(index_value, position: int):
    if isinstance(index_value, pd.Timestamp):
        return index_value.strftime("%Y-%m-%d")
    return position


def forecast_base_frame(df: pd.DataFrame, start: int = WINDOW_SIZE) -> pd.DataFrame:
    rows = []
    for pos in range(start, len(df)):
        rows.append(
            {
                "date": index_label(df.index[pos], pos),
                "realized_log_ret": df["log_ret"].iloc[pos],
            }
        )
    return pd.DataFrame(rows)


def save_forecast(df: pd.DataFrame, filename: str) -> Path:
    ensure_directories()
    path = FORECASTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def normal_quantile_table(values: np.ndarray) -> dict[float, float]:
    return {alpha: float(np.quantile(values, alpha)) for alpha in VAR_LEVELS}

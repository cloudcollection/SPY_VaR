"""Load the existing data snapshot without guessing its upstream provenance."""
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["log_ret", "rv5", "bv"]


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_col = next((name for name in ("date", "Date", "datetime", "Unnamed: 0") if name in frame), None)
    if date_col is None:
        raise ValueError("Data must contain a date column or a saved date index")
    missing = set(FEATURES) - set(frame)
    if missing:
        raise ValueError(f"Missing data fields: {sorted(missing)}")
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame = frame.set_index(date_col)[FEATURES].apply(pd.to_numeric, errors="raise")
    frame.index.name = "date"
    if frame.empty or frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Data dates must be nonempty, valid, unique and increasing")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Data contains missing or nonfinite values; document preprocessing before training")
    return frame

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import FIGURES_DIR, ensure_directories, load_spy_data


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(module_name: str, script_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / script_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_forecasts(window_size: int = 1000, decay: float = 0.98) -> dict[str, pd.DataFrame]:
    df = load_spy_data()
    compare_module = load_module("compare_windows", "07_compare_weighted_hs_windows.py")
    weighted_module = load_module("weighted_hs", "02_historical_simulation.py")
    kde_module = load_module("kde_weighted_hs", "04_kde_weighted_hs.py")
    return {
        "Ordinary HS": compare_module.ordinary_hs_var(df, window_size=window_size),
        "Time-weighted HS": weighted_module.historical_simulation_var(df, window_size=window_size, decay=decay),
        "KDE Time-weighted HS": kde_module.kde_weighted_hs_var(df, window_size=window_size, decay=decay),
    }


def plot_var_forecasts(forecasts: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    first = next(iter(forecasts.values())).copy()
    dates = pd.to_datetime(first["date"])
    ax.plot(dates, first["realized_log_ret"], color="black", linewidth=0.55, alpha=0.65, label="Realized log return")
    for name, frame in forecasts.items():
        ax.plot(pd.to_datetime(frame["date"]), frame["VaR_1"], linewidth=0.9, label=f"{name} 1% VaR")
    ax.set_title("1% VaR Forecasts for Historical Simulation Variants (W=1000)")
    ax.set_ylabel("Log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "hs_1pct_var_comparison_w1000.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_violation_indicators(forecasts: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(13, 4))
    offsets = {"Ordinary HS": 2, "Time-weighted HS": 1, "KDE Time-weighted HS": 0}
    for name, frame in forecasts.items():
        dates = pd.to_datetime(frame["date"])
        violation = (frame["realized_log_ret"] < frame["VaR_1"]).astype(int)
        y = np.where(violation.to_numpy() == 1, offsets[name] + 1.0, np.nan)
        ax.scatter(dates, y, s=8, label=name)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["KDE TWHS", "TWHS", "Ordinary HS"])
    ax.set_title("1% VaR Violation Indicators for Historical Simulation Variants (W=1000)")
    ax.set_ylim(0.6, 3.4)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "hs_1pct_violation_indicators_w1000.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    ensure_directories()
    forecasts = build_forecasts()
    path1 = plot_var_forecasts(forecasts)
    path2 = plot_violation_indicators(forecasts)
    print(f"Saved {path1}")
    print(f"Saved {path2}")


if __name__ == "__main__":
    main()

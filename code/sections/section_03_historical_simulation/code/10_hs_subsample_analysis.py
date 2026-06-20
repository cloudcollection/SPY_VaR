from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from utils import TABLES_DIR, VAR_COLUMN_MAP, VAR_LEVELS, ensure_directories, load_spy_data


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(module_name: str, script_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / script_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ordinary_hs_var(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    rows = []
    returns = df["log_ret"]
    for pos in range(window_size, len(df)):
        window = returns.iloc[pos - window_size : pos].dropna()
        row = {
            "date": df.index[pos].strftime("%Y-%m-%d") if hasattr(df.index[pos], "strftime") else pos,
            "realized_log_ret": returns.iloc[pos],
        }
        for alpha in VAR_LEVELS:
            row[VAR_COLUMN_MAP[alpha]] = float(window.quantile(alpha))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_subsample(forecasts: pd.DataFrame, model: str, period_name: str, start: str, end: str) -> list[dict]:
    tmp = forecasts.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    period = tmp.loc[(tmp["date"] >= pd.Timestamp(start)) & (tmp["date"] <= pd.Timestamp(end))].copy()
    rows = []
    for alpha in VAR_LEVELS:
        var_col = VAR_COLUMN_MAP[alpha]
        clean = period[["realized_log_ret", var_col]].dropna()
        violations = clean["realized_log_ret"].to_numpy() < clean[var_col].to_numpy()
        n_obs = len(clean)
        n_viol = int(violations.sum())
        rows.append(
            {
                "model": model,
                "period": period_name,
                "start": start,
                "end": end,
                "alpha": alpha,
                "n_obs": n_obs,
                "expected_violations": n_obs * alpha,
                "n_violations": n_viol,
                "failure_rate": n_viol / n_obs if n_obs else np.nan,
                "avg_var": float(clean[var_col].mean()) if n_obs else np.nan,
            }
        )
    return rows


def main() -> None:
    ensure_directories()
    df = load_spy_data()
    df = df.sort_index()

    crisis_start, crisis_end = "2007-09-01", "2009-06-30"
    calm_start, calm_end = "2012-01-01", "2016-12-31"
    crisis = df.loc[crisis_start:crisis_end, "log_ret"].dropna()
    calm = df.loc[calm_start:calm_end, "log_ret"].dropna()

    data_rows = [
        {
            "metric": "full_sample_excess_kurtosis",
            "value": float(kurtosis(df["log_ret"].dropna(), fisher=True, bias=False)),
            "interpretation": "Excess kurtosis of SPY daily log returns; values far above zero reject Gaussian-style tail intuition.",
        },
        {
            "metric": "crisis_std",
            "value": float(crisis.std(ddof=1)),
            "interpretation": "Standard deviation from 2007-09-01 to 2009-06-30.",
        },
        {
            "metric": "calm_std",
            "value": float(calm.std(ddof=1)),
            "interpretation": "Standard deviation from 2012-01-01 to 2016-12-31.",
        },
        {
            "metric": "crisis_to_calm_std_ratio",
            "value": float(crisis.std(ddof=1) / calm.std(ddof=1)),
            "interpretation": "Volatility regime ratio; values materially above one indicate fixed-window regime mixing.",
        },
    ]
    data_summary = pd.DataFrame(data_rows)
    data_summary.to_csv(TABLES_DIR / "hs_data_regime_diagnostics.csv", index=False)

    weighted_hs = load_module("weighted_hs", "02_historical_simulation.py")
    kde_hs = load_module("kde_hs", "04_kde_weighted_hs.py")

    forecasts = {
        "Ordinary HS": ordinary_hs_var(df, window_size=1000),
        "Time-weighted HS": weighted_hs.historical_simulation_var(df, window_size=1000, decay=0.98),
        "KDE Time-weighted HS": kde_hs.kde_weighted_hs_var(df, window_size=1000, decay=0.98),
    }

    rows = []
    for model, fc in forecasts.items():
        rows.extend(summarize_subsample(fc, model, "Crisis", crisis_start, crisis_end))
        rows.extend(summarize_subsample(fc, model, "Post-crisis calm", calm_start, calm_end))
    out = pd.DataFrame(rows)
    for col in ["expected_violations", "failure_rate", "avg_var"]:
        out[col] = out[col].round(4)
    out.to_csv(TABLES_DIR / "hs_w1000_subsample_backtesting.csv", index=False)

    print("Saved data diagnostics to", TABLES_DIR / "hs_data_regime_diagnostics.csv")
    print("Saved subsample backtesting to", TABLES_DIR / "hs_w1000_subsample_backtesting.csv")
    print(data_summary.to_string(index=False))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

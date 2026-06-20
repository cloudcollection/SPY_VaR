from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from utils import TABLES_DIR, VAR_COLUMN_MAP, VAR_LEVELS, ensure_directories, load_spy_data


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(module_name: str, script_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / script_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_forecasts(forecasts: pd.DataFrame, model_name: str, window_size: int, decay: float, backtesting_module) -> list[dict]:
    rows = []
    for alpha in VAR_LEVELS:
        var_col = VAR_COLUMN_MAP[alpha]
        tmp = forecasts[["realized_log_ret", var_col]].dropna()
        realized = tmp["realized_log_ret"].to_numpy()
        var_forecast = tmp[var_col].to_numpy()
        violations = (realized < var_forecast).astype(int)
        kupiec_lr, kupiec_p = backtesting_module.kupiec_test(violations, alpha)
        ind_lr, ind_p = backtesting_module.christoffersen_independence_test(violations)
        lopez = backtesting_module.lopez_loss(realized, var_forecast)
        n_obs = len(violations)
        n_violations = int(violations.sum())
        avg_var = float(tmp[var_col].mean())
        rows.append(
            {
                "model": model_name,
                "window_size": window_size,
                "decay": decay,
                "alpha": alpha,
                "n_obs": n_obs,
                "n_violations": n_violations,
                "expected_violations": n_obs * alpha,
                "failure_rate": n_violations / n_obs if n_obs else np.nan,
                "avg_var": avg_var,
                "kupiec_LR": kupiec_lr,
                "kupiec_pvalue": kupiec_p,
                "christoffersen_ind_LR": ind_lr,
                "christoffersen_ind_pvalue": ind_p,
                "lopez_loss": lopez,
            }
        )
    return rows


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


def compare_windows(windows: list[int] | None = None, decay: float = 0.98) -> pd.DataFrame:
    ensure_directories()
    windows = windows or [250, 500, 1000]
    hs_module = load_module("weighted_hs", "02_historical_simulation.py")
    kde_module = load_module("kde_weighted_hs", "04_kde_weighted_hs.py")
    backtesting_module = load_module("backtesting", "05_backtesting.py")
    df = load_spy_data()

    rows = []
    for window_size in windows:
        print(f"Evaluating ordinary HS with window_size={window_size}")
        ordinary_forecasts = ordinary_hs_var(df, window_size=window_size)
        rows.extend(evaluate_forecasts(ordinary_forecasts, "Ordinary HS", window_size, np.nan, backtesting_module))

        print(f"Evaluating weighted HS with window_size={window_size}, decay={decay}")
        forecasts = hs_module.historical_simulation_var(df, window_size=window_size, decay=decay)
        rows.extend(evaluate_forecasts(forecasts, "Time-weighted HS", window_size, decay, backtesting_module))

        print(f"Evaluating Gaussian KDE weighted HS with window_size={window_size}, decay={decay}")
        kde_forecasts = kde_module.kde_weighted_hs_var(df, window_size=window_size, decay=decay)
        rows.extend(evaluate_forecasts(kde_forecasts, "KDE Time-weighted HS", window_size, decay, backtesting_module))
    return pd.DataFrame(rows)


def main() -> None:
    results = compare_windows()
    output_path = TABLES_DIR / "weighted_hs_window_comparison.csv"
    results.to_csv(output_path, index=False)
    report_table = results[
        [
            "model",
            "window_size",
            "alpha",
            "n_violations",
            "expected_violations",
            "failure_rate",
            "avg_var",
            "kupiec_pvalue",
            "christoffersen_ind_pvalue",
            "lopez_loss",
        ]
    ].copy()
    report_table["viol_exp"] = report_table["n_violations"].astype(str) + " / " + report_table["expected_violations"].round(1).astype(str)
    report_table["expected_violations"] = report_table["expected_violations"].round(1)
    report_table["failure_rate"] = report_table["failure_rate"].round(4)
    report_table["avg_var"] = report_table["avg_var"].round(4)
    report_table["kupiec_pvalue"] = report_table["kupiec_pvalue"].round(4)
    report_table["christoffersen_ind_pvalue"] = report_table["christoffersen_ind_pvalue"].round(4)
    report_table["lopez_loss"] = report_table["lopez_loss"].round(6)
    report_table = report_table[
        [
            "model",
            "window_size",
            "alpha",
            "viol_exp",
            "failure_rate",
            "avg_var",
            "kupiec_pvalue",
            "christoffersen_ind_pvalue",
            "lopez_loss",
        ]
    ]
    report_path = TABLES_DIR / "hs_window_comparison_report_table.csv"
    report_table.to_csv(report_path, index=False)

    for window_size in sorted(report_table["window_size"].unique()):
        window_table = report_table.loc[report_table["window_size"] == window_size].copy()
        window_path = TABLES_DIR / f"hs_window_{window_size}_report_table.csv"
        window_table.to_csv(window_path, index=False)
        print(f"Saved report-ready table for W={window_size} to {window_path}")

    print(f"Saved weighted HS window comparison to {output_path}")
    print(f"Saved report-ready HS window comparison table to {report_path}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

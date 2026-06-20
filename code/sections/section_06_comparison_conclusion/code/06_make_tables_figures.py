from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import FIGURES_DIR, FORECASTS_DIR, TABLES_DIR, VAR_COLUMN_MAP, VAR_LEVELS, ensure_directories, load_spy_data


FORECAST_FILES = {
    "Weighted Historical Simulation": "var_historical_simulation.csv",
    "Gaussian KDE Weighted HS": "var_kde_weighted_hs.csv",
    "GARCH(1,1)-t": "var_garch_t.csv",
    "GJR-GARCH(1,1)-t": "var_gjr_garch_t.csv",
    "MLP Quantile": "var_mlp_quantile.csv",
}


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "_")


def plot_var_forecasts(model_name: str, path: Path) -> None:
    forecasts = pd.read_csv(path)
    x = pd.to_datetime(forecasts["date"], errors="coerce")
    if x.isna().all():
        x = forecasts.index

    for alpha in VAR_LEVELS:
        var_col = VAR_COLUMN_MAP[alpha]
        valid = forecasts[["realized_log_ret", var_col]].dropna()
        violation_mask = valid["realized_log_ret"] < valid[var_col]
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(x, forecasts["realized_log_ret"], linewidth=0.6, label="realized log_ret", alpha=0.8)
        ax.plot(x, forecasts[var_col], linewidth=0.9, label=f"{int(alpha * 100)}% VaR")
        if violation_mask.any():
            violation_x = x[valid.index[violation_mask]]
            ax.scatter(violation_x, valid.loc[violation_mask, "realized_log_ret"], color="red", s=12, label="violations")
        ax.set_title(f"{model_name}: {int(alpha * 100)}% One-Day-Ahead VaR")
        ax.set_ylabel("log return")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"var_forecast_{_safe_name(model_name)}_{int(alpha * 100)}pct.png", dpi=150)
        plt.close(fig)


def plot_failure_rates(backtesting: pd.DataFrame) -> None:
    if backtesting.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = backtesting.pivot(index="model", columns="alpha", values="failure_rate")
    pivot.plot(kind="bar", ax=ax)
    for alpha in VAR_LEVELS:
        ax.axhline(alpha, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Failure Rate Comparison by Model and VaR Level")
    ax.set_ylabel("failure rate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_rate_comparison.png", dpi=150)
    plt.close(fig)


def save_pvalue_table(backtesting: pd.DataFrame) -> None:
    if backtesting.empty:
        return
    cols = ["model", "alpha", "kupiec_pvalue", "christoffersen_ind_pvalue"]
    if "duration_pvalue" in backtesting.columns:
        cols.append("duration_pvalue")
    backtesting[cols].to_csv(TABLES_DIR / "backtesting_pvalue_comparison.csv", index=False)


def save_model_evaluation_summary(backtesting: pd.DataFrame) -> None:
    if backtesting.empty:
        return
    cols = [
        "model",
        "alpha",
        "n_obs",
        "n_violations",
        "expected_violations",
        "failure_rate",
        "kupiec_pvalue",
        "christoffersen_ind_pvalue",
        "christoffersen_cc_pvalue",
        "duration_pvalue",
        "lopez_loss",
    ]
    summary = backtesting[cols].copy()
    numeric_cols = summary.select_dtypes(include="number").columns
    summary[numeric_cols] = summary[numeric_cols].round(6)
    summary.to_csv(TABLES_DIR / "model_evaluation_summary.csv", index=False)


def plot_crisis_zoom_if_available() -> None:
    hs_path = FORECASTS_DIR / "var_historical_simulation.csv"
    if not hs_path.exists():
        return
    forecasts = pd.read_csv(hs_path)
    dates = pd.to_datetime(forecasts["date"], errors="coerce")
    if dates.isna().all():
        return
    mask = (dates >= "2008-01-01") & (dates <= "2009-12-31")
    if mask.sum() < 20:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates[mask], forecasts.loc[mask, "realized_log_ret"], linewidth=0.7, label="realized log_ret")
    ax.plot(dates[mask], forecasts.loc[mask, "VaR_1"], linewidth=0.9, label="Historical Simulation 1% VaR")
    ax.set_title("Crisis Period Zoom-In: 2008-2009")
    ax.set_ylabel("log return")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "crisis_zoom_2008_2009.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    for model_name, filename in FORECAST_FILES.items():
        path = FORECASTS_DIR / filename
        if path.exists():
            plot_var_forecasts(model_name, path)
            print(f"Saved VaR forecast plots for {model_name}")
        else:
            print(f"Skipping plots for {model_name}: missing {path}")

    backtesting_path = TABLES_DIR / "backtesting_results.csv"
    if backtesting_path.exists():
        backtesting = pd.read_csv(backtesting_path)
        plot_failure_rates(backtesting)
        save_pvalue_table(backtesting)
        save_model_evaluation_summary(backtesting)
        print("Saved failure-rate chart and p-value comparison table")
    else:
        print(f"Skipping backtesting comparison outputs: missing {backtesting_path}")

    plot_crisis_zoom_if_available()


if __name__ == "__main__":
    main()

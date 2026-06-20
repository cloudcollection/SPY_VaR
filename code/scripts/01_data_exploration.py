from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from utils import FIGURES_DIR, REQUIRED_COLUMNS, TABLES_DIR, ensure_directories, load_spy_data


def save_dataset_summary(df: pd.DataFrame) -> Path:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_values": [int(df[col].isna().sum()) for col in df.columns],
            "missing_pct": [float(df[col].isna().mean()) for col in df.columns],
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "index_name": df.index.name or "",
            "index_type": type(df.index).__name__,
        }
    )
    path = TABLES_DIR / "dataset_summary.csv"
    summary.to_csv(path, index=False)
    return path


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[REQUIRED_COLUMNS]
    return pd.DataFrame(
        {
            "count": numeric.count(),
            "mean": numeric.mean(),
            "std": numeric.std(),
            "min": numeric.min(),
            "q01": numeric.quantile(0.01),
            "q05": numeric.quantile(0.05),
            "q10": numeric.quantile(0.10),
            "median": numeric.median(),
            "max": numeric.max(),
            "skewness": numeric.skew(),
            "kurtosis": numeric.kurtosis(),
        }
    )


def save_worst_returns(df: pd.DataFrame) -> Path:
    worst = df[["log_ret", "rv5", "bv"]].nsmallest(10, "log_ret").copy()
    worst.insert(0, "date", [idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else idx for idx in worst.index])
    path = TABLES_DIR / "worst_returns.csv"
    worst.to_csv(path, index=False)
    return path


def save_line_plot(series: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    series.plot(ax=ax, linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_log_ret_time_series(df: pd.DataFrame) -> None:
    save_line_plot(df["log_ret"], "SPY Daily Log Returns", "log return", FIGURES_DIR / "log_ret_time_series.png")


def plot_rv5_bv_time_series(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    df[["rv5", "bv"]].plot(ax=ax, linewidth=0.8)
    ax.set_title("Realized Volatility and Bipower Variation")
    ax.set_ylabel("volatility measure")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rv5_bv_time_series.png", dpi=150)
    plt.close(fig)


def plot_log_ret_histogram_with_normal(df: pd.DataFrame) -> None:
    values = df["log_ret"].dropna()
    mu = values.mean()
    sigma = values.std()
    x_grid = np.linspace(values.min(), values.max(), 500)
    normal_pdf = stats.norm.pdf(x_grid, loc=mu, scale=sigma)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=80, density=True, alpha=0.65, label="empirical density")
    ax.plot(x_grid, normal_pdf, color="red", linewidth=1.5, label="normal density")
    ax.set_title("Histogram of SPY Log Returns with Normal Density")
    ax.set_xlabel("log return")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "log_ret_histogram_normal_density.png", dpi=150)
    plt.close(fig)


def plot_log_ret_qq(df: pd.DataFrame) -> None:
    fig = sm.qqplot(df["log_ret"].dropna(), line="s")
    fig.axes[0].set_title("QQ Plot of SPY Log Returns")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "log_ret_qq_plot.png", dpi=150)
    plt.close(fig)


def plot_rolling_volatility(df: pd.DataFrame) -> None:
    rolling_vol = df["log_ret"].rolling(60).std()
    save_line_plot(
        rolling_vol,
        "60-Day Rolling Volatility of SPY Log Returns",
        "rolling std",
        FIGURES_DIR / "rolling_volatility_60d.png",
    )


def plot_acf_figures(df: pd.DataFrame) -> None:
    log_ret = df["log_ret"].dropna()
    squared_log_ret = log_ret**2

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(log_ret, lags=40, ax=axes[0], zero=False)
    axes[0].set_title("ACF of SPY Log Returns")
    plot_acf(squared_log_ret, lags=40, ax=axes[1], zero=False)
    axes[1].set_title("ACF of Squared SPY Log Returns")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "acf_log_ret_and_squared_log_ret.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[REQUIRED_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Correlation Heatmap: log_ret, rv5, bv")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def generate_figures(df: pd.DataFrame) -> None:
    plot_log_ret_time_series(df)
    plot_rv5_bv_time_series(df)
    plot_log_ret_histogram_with_normal(df)
    plot_log_ret_qq(df)
    plot_rolling_volatility(df)
    plot_acf_figures(df)
    plot_correlation_heatmap(df)


def main() -> None:
    ensure_directories()
    df = load_spy_data()
    print(f"Loaded data with shape {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values by column:")
    print(df.isna().sum())

    dataset_summary_path = save_dataset_summary(df)
    print(f"Saved dataset summary to {dataset_summary_path}")

    summary = compute_summary_statistics(df)
    summary_path = TABLES_DIR / "summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")

    worst_returns_path = save_worst_returns(df)
    print(f"Saved worst 10 returns to {worst_returns_path}")

    generate_figures(df)
    print(f"Saved exploration figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

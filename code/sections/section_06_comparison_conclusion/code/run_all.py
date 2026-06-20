from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
FORECASTS_DIR = ROOT / "outputs" / "forecasts"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
VAR_LEVELS = [0.01, 0.05, 0.10]
VAR_COLUMNS = {0.01: "VaR_1", 0.05: "VaR_5", 0.10: "VaR_10"}
FORECAST_FILES = {
    "Weighted Historical Simulation": "var_historical_simulation.csv",
    "Gaussian KDE Weighted HS": "var_kde_weighted_hs.csv",
    "GARCH(1,1)-t": "var_garch_t.csv",
    "GJR-GARCH(1,1)-t": "var_gjr_garch_t.csv",
    "MLP Quantile": "var_mlp_quantile.csv",
}


def safe_log(x: float) -> float:
    return math.log(max(float(x), 1e-12))


def kupiec_test(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    n = len(violations)
    x = int(violations.sum())
    phat = x / n
    unrestricted = x * safe_log(phat) + (n - x) * safe_log(1 - phat)
    restricted = x * math.log(alpha) + (n - x) * math.log(1 - alpha)
    lr = -2 * (restricted - unrestricted)
    return lr, 1 - chi2.cdf(lr, 1)


def christoffersen_independence(violations: np.ndarray) -> tuple[float, float]:
    prev = violations[:-1].astype(int)
    curr = violations[1:].astype(int)
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    ll_null = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    ll_alt = n00 * safe_log(1 - pi0) + n01 * safe_log(pi0) + n10 * safe_log(1 - pi1) + n11 * safe_log(pi1)
    lr = -2 * (ll_null - ll_alt)
    return lr, 1 - chi2.cdf(lr, 1)


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    diff = y - q
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def load_forecasts() -> dict[str, pd.DataFrame]:
    return {
        model: pd.read_csv(FORECASTS_DIR / filename, parse_dates=["date"])
        for model, filename in FORECAST_FILES.items()
    }


def common_dates(frames: dict[str, pd.DataFrame]) -> pd.Index:
    dates = None
    for frame in frames.values():
        frame_dates = pd.Index(frame["date"].dropna().unique())
        dates = frame_dates if dates is None else dates.intersection(frame_dates)
    return dates.sort_values()


def build_aligned_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dates = common_dates(frames)
    rows = []
    for model, frame in frames.items():
        aligned = frame[frame["date"].isin(dates)].sort_values("date")
        for alpha in VAR_LEVELS:
            var_col = VAR_COLUMNS[alpha]
            tmp = aligned[["realized_log_ret", var_col]].dropna()
            realized = tmp["realized_log_ret"].to_numpy()
            forecast = tmp[var_col].to_numpy()
            violations = (realized < forecast).astype(int)
            _, kupiec_p = kupiec_test(violations, alpha)
            _, ind_p = christoffersen_independence(violations)
            rows.append(
                {
                    "model": model,
                    "alpha": alpha,
                    "n_obs": len(tmp),
                    "n_violations": int(violations.sum()),
                    "expected_violations": round(alpha * len(tmp), 2),
                    "failure_rate": float(violations.mean()),
                    "avg_var": float(forecast.mean()),
                    "kupiec_pvalue": float(kupiec_p),
                    "christoffersen_ind_pvalue": float(ind_p),
                    "pinball_loss": pinball_loss(realized, forecast, alpha),
                }
            )
    result = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(TABLES_DIR / "aligned_model_comparison.csv", index=False)
    return result


def plot_failure_rates(table: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, alpha in zip(axes, VAR_LEVELS):
        subset = table[table["alpha"] == alpha]
        ax.barh(subset["model"], subset["failure_rate"], color="#4C78A8")
        ax.axvline(alpha, color="#D62728", linestyle="--", linewidth=1.2)
        ax.set_title(f"{int(alpha * 100)}% VaR failure rate")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "aligned_failure_rates.png", dpi=180)
    plt.close(fig)


def main() -> None:
    frames = load_forecasts()
    table = build_aligned_table(frames)
    plot_failure_rates(table)
    print("Saved outputs/tables/aligned_model_comparison.csv")
    print("Saved outputs/figures/aligned_failure_rates.png")
    print(table[["model", "alpha", "n_obs", "n_violations", "failure_rate", "kupiec_pvalue", "christoffersen_ind_pvalue", "pinball_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()

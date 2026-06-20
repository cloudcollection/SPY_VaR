from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils import FIGURES_DIR, FORECASTS_DIR, TABLES_DIR, ensure_directories


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"


def arch_lm_test(z: pd.Series, lags: int = 10) -> dict[str, float]:
    y = pd.Series(z).replace([np.inf, -np.inf], np.nan).dropna()
    y = y - y.mean()
    sq = y.pow(2).rename("sq")
    frame = pd.concat([sq] + [sq.shift(i).rename(f"lag_{i}") for i in range(1, lags + 1)], axis=1).dropna()
    target = frame["sq"].to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame[[f"lag_{i}" for i in range(1, lags + 1)]].to_numpy()])
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    fitted = x @ beta
    sst = float(((target - target.mean()) ** 2).sum())
    ssr = float(((target - fitted) ** 2).sum())
    r2 = 0.0 if sst <= 0 else max(0.0, 1.0 - ssr / sst)
    lm_stat = len(frame) * r2
    return {
        "lags": lags,
        "n_obs": int(len(frame)),
        "lm_stat": float(lm_stat),
        "p_value": float(1.0 - stats.chi2.cdf(lm_stat, df=lags)),
    }


def standardized_residuals(forecast_name: str, params_name: str) -> pd.DataFrame:
    forecast = pd.read_csv(FORECASTS_DIR / forecast_name, parse_dates=["date"])
    params = pd.read_csv(TABLES_DIR / params_name, parse_dates=["date"])
    merged = forecast[["date", "realized_log_ret"]].merge(params[["date", "mu", "sigma_next"]], on="date", how="inner")
    merged["std_resid"] = (merged["realized_log_ret"] - merged["mu"]) / merged["sigma_next"]
    return merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["std_resid"])


def make_section4_diagnostics() -> None:
    rows = []
    residual_sets = {}
    for label, key in [
        ("GARCH(1,1)-t", ("var_garch_t_w1000.csv", "section4_garch_params_w1000.csv")),
        ("GJR-GARCH(1,1)-t", ("var_gjr_garch11_t_w1000.csv", "section4_gjr_garch11_t_params_w1000.csv")),
    ]:
        resid = standardized_residuals(*key)
        residual_sets[label] = resid["std_resid"]
        row = {"model": label}
        row.update(arch_lm_test(resid["std_resid"], lags=10))
        rows.append(row)

    pd.DataFrame(rows).to_csv(TABLES_DIR / "section4_arch_lm_diagnostics_w1000.csv", index=False)

    gjr = pd.read_csv(TABLES_DIR / "section4_gjr_garch11_t_params_w1000.csv")
    gamma = gjr["gamma_1"].dropna()
    gamma_summary = pd.DataFrame(
        [
            {
                "parameter": "gamma_1",
                "median": gamma.median(),
                "q25": gamma.quantile(0.25),
                "q75": gamma.quantile(0.75),
                "positive_share": (gamma > 0).mean(),
                "interpretation": "Positive gamma means negative shocks increase next-day variance more than positive shocks in the GJR recursion.",
            }
        ]
    )
    gamma_summary.to_csv(TABLES_DIR / "section4_gjr_gamma_summary_w1000.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (label, resid) in zip(axes, residual_sets.items()):
        stats.probplot(resid.clip(-8, 8), dist="norm", plot=ax)
        ax.set_title(f"{label}: standardized residual QQ plot")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_4_standardized_residual_qq_w1000.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_section5_diagnostics() -> None:
    preds = pd.read_csv(RESULTS_DIR / "section5_model_c_garch_anchored_predictions.csv")
    c2 = preds[preds["model_name"] == "Model C2: Conservative GARCH-anchored MLP-QR"].copy()
    rows = []
    for alpha in [0.01, 0.05, 0.10]:
        suffix = f"{alpha:.2f}"
        base = c2[f"base_VaR_{suffix}"]
        var = c2[f"VaR_{suffix}"]
        corr = c2[f"correction_{suffix}"]
        rows.append(
            {
                "alpha": alpha,
                "base_avg_var": base.mean(),
                "c2_avg_var": var.mean(),
                "avg_correction": corr.mean(),
                "median_correction": corr.median(),
                "q25_correction": corr.quantile(0.25),
                "q75_correction": corr.quantile(0.75),
                "c2_var_q05": var.quantile(0.05),
                "c2_var_q50": var.quantile(0.50),
                "c2_var_q95": var.quantile(0.95),
                "share_below_minus_10pct": (var <= -0.10).mean(),
                "failure_rate": c2[f"violation_{suffix}"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "section5_c2_conservatism_diagnostics.csv", index=False)

    wd = pd.read_csv(RESULTS_DIR / "section5_model_c_weight_decay_tuning.csv")
    wd_compact = wd.pivot_table(index="weight_decay", columns="alpha", values="failure_rate", aggfunc="first")
    wd_compact.columns = [f"failure_rate_{alpha:.2f}" for alpha in wd_compact.columns]
    wd_pinball = wd.pivot_table(index="weight_decay", columns="alpha", values="pinball_loss", aggfunc="first")
    wd_pinball.columns = [f"pinball_loss_{alpha:.2f}" for alpha in wd_pinball.columns]
    pd.concat([wd_compact, wd_pinball], axis=1).reset_index().to_csv(
        RESULTS_DIR / "section5_weight_decay_tuning_compact.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for alpha, color in [(0.01, "tab:blue"), (0.05, "tab:orange"), (0.10, "tab:green")]:
        suffix = f"{alpha:.2f}"
        ax.hist(c2[f"correction_{suffix}"], bins=60, alpha=0.45, label=f"{int(alpha * 100)}%", color=color)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Model C2 correction distribution relative to GARCH baseline")
    ax.set_xlabel("C2 VaR minus GARCH VaR")
    ax.set_ylabel("Frequency")
    ax.legend(title="VaR level")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "section5_c2_correction_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, alpha in zip(axes, [0.01, 0.05, 0.10]):
        suffix = f"{alpha:.2f}"
        ax.boxplot([c2[f"base_VaR_{suffix}"], c2[f"VaR_{suffix}"]], tick_labels=["GARCH", "C2"])
        ax.axhline(-0.10, color="red", lw=0.8, ls="--", label="-10%")
        ax.set_title(f"{int(alpha * 100)}% VaR")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Log-return VaR")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Model C2 VaR distribution versus GARCH baseline")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "section5_c2_var_distribution_vs_garch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_directories()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    make_section4_diagnostics()
    make_section5_diagnostics()
    print(f"Saved Section 4 diagnostics to {TABLES_DIR}")
    print(f"Saved Section 5 diagnostics to {RESULTS_DIR}")
    print(f"Saved diagnostic figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

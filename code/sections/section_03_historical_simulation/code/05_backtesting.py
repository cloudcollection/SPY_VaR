from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2

from utils import FORECASTS_DIR, TABLES_DIR, VAR_COLUMN_MAP, VAR_LEVELS, ensure_directories


FORECAST_FILES = {
    "Weighted Historical Simulation": "var_historical_simulation.csv",
    "Gaussian KDE Weighted HS": "var_kde_weighted_hs.csv",
    "GARCH(1,1)-t": "var_garch_t.csv",
    "GJR-GARCH(1,1)-t": "var_gjr_garch_t.csv",
    "MLP Quantile": "var_mlp_quantile.csv",
}


def _safe_log(x: float) -> float:
    return math.log(max(float(x), 1e-12))


def kupiec_test(violations: np.ndarray, alpha: float) -> tuple[float, float]:
    n = len(violations)
    x = int(violations.sum())
    if n == 0:
        return np.nan, np.nan
    phat = x / n
    ll_null = (n - x) * _safe_log(1 - alpha) + x * _safe_log(alpha)
    ll_alt = (n - x) * _safe_log(1 - phat) + x * _safe_log(phat) if 0 < x < n else 0.0
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return lr, float(1.0 - chi2.cdf(lr, df=1))


def christoffersen_independence_test(violations: np.ndarray) -> tuple[float, float]:
    if len(violations) < 2:
        return np.nan, np.nan
    prev = violations[:-1].astype(int)
    curr = violations[1:].astype(int)
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    n0 = n00 + n01
    n1 = n10 + n11
    total_viol = n01 + n11
    total = n0 + n1
    if total == 0:
        return np.nan, np.nan

    pi = total_viol / total
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0
    ll_restricted = (total - total_viol) * _safe_log(1 - pi) + total_viol * _safe_log(pi)
    ll_unrestricted = (
        n00 * _safe_log(1 - pi0)
        + n01 * _safe_log(pi0)
        + n10 * _safe_log(1 - pi1)
        + n11 * _safe_log(pi1)
    )
    lr = max(0.0, -2.0 * (ll_restricted - ll_unrestricted))
    return lr, float(1.0 - chi2.cdf(lr, df=1))


def duration_test(violations: np.ndarray) -> tuple[float, float]:
    """Weibull duration test of violation clustering.

    Under independence, violation durations should be memoryless. This test
    compares unrestricted Weibull durations with the exponential special case
    where the Weibull shape parameter equals one.
    """
    violation_positions = np.flatnonzero(violations)
    if len(violation_positions) < 3:
        return np.nan, np.nan

    durations = np.diff(violation_positions).astype(float)
    durations = durations[durations > 0]
    if len(durations) < 2:
        return np.nan, np.nan

    def weibull_loglik(scale: float, shape: float) -> float:
        if scale <= 0 or shape <= 0:
            return -np.inf
        z = durations / scale
        return float(np.sum(np.log(shape) - shape * np.log(scale) + (shape - 1.0) * np.log(durations) - z**shape))

    restricted_scale = float(durations.mean())
    ll_restricted = weibull_loglik(restricted_scale, 1.0)

    def objective(params: np.ndarray) -> float:
        scale = float(np.exp(params[0]))
        shape = float(np.exp(params[1]))
        return -weibull_loglik(scale, shape)

    result = optimize.minimize(
        objective,
        x0=np.log([restricted_scale, 1.0]),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
    )
    if not result.success:
        return np.nan, np.nan
    ll_unrestricted = -float(result.fun)
    lr = max(0.0, 2.0 * (ll_unrestricted - ll_restricted))
    return lr, float(1.0 - chi2.cdf(lr, df=1))


def lopez_loss(realized: np.ndarray, var_forecast: np.ndarray) -> float:
    """Average Lopez regulatory loss for VaR forecasts.

    A non-violation receives zero loss. A violation receives one plus the
    squared exceedance size, so the score penalizes both failure occurrence
    and violation depth.
    """
    realized = np.asarray(realized, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    valid = np.isfinite(realized) & np.isfinite(var_forecast)
    if not valid.any():
        return np.nan
    realized = realized[valid]
    var_forecast = var_forecast[valid]
    violations = realized < var_forecast
    losses = np.where(violations, 1.0 + (realized - var_forecast) ** 2, 0.0)
    return float(losses.mean())


def conclusion(alpha: float, failure_rate: float, kupiec_p: float, ind_p: float) -> str:
    if np.isnan(failure_rate):
        return "Insufficient valid forecasts"
    messages = []
    messages.append("coverage accepted" if kupiec_p >= 0.05 else "coverage rejected")
    messages.append("independence accepted" if ind_p >= 0.05 else "independence rejected")
    direction = "too many violations" if failure_rate > alpha else "too few violations"
    return f"{'; '.join(messages)} ({direction})"


def backtest_file(model_name: str, path: Path) -> list[dict]:
    forecasts = pd.read_csv(path)
    results = []
    for alpha in VAR_LEVELS:
        var_col = VAR_COLUMN_MAP[alpha]
        tmp = forecasts[["realized_log_ret", var_col]].dropna()
        realized = tmp["realized_log_ret"].to_numpy()
        var_forecast = tmp[var_col].to_numpy()
        violations = (realized < var_forecast).astype(int)
        n = len(violations)
        n_viol = int(violations.sum())
        failure_rate = n_viol / n if n else np.nan
        kupiec_lr, kupiec_p = kupiec_test(violations, alpha)
        ind_lr, ind_p = christoffersen_independence_test(violations)
        cc_lr = kupiec_lr + ind_lr if not (np.isnan(kupiec_lr) or np.isnan(ind_lr)) else np.nan
        cc_p = float(1.0 - chi2.cdf(cc_lr, df=2)) if not np.isnan(cc_lr) else np.nan
        dur_lr, dur_p = duration_test(violations)
        lopez = lopez_loss(realized, var_forecast)
        results.append(
            {
                "model": model_name,
                "alpha": alpha,
                "n_obs": n,
                "n_violations": n_viol,
                "expected_violations": n * alpha if n else np.nan,
                "failure_rate": failure_rate,
                "kupiec_LR": kupiec_lr,
                "kupiec_pvalue": kupiec_p,
                "christoffersen_ind_LR": ind_lr,
                "christoffersen_ind_pvalue": ind_p,
                "christoffersen_cc_LR": cc_lr,
                "christoffersen_cc_pvalue": cc_p,
                "duration_LR": dur_lr,
                "duration_pvalue": dur_p,
                "lopez_loss": lopez,
                "conclusion": conclusion(alpha, failure_rate, kupiec_p, ind_p),
            }
        )
    return results


def run_backtests() -> pd.DataFrame:
    rows = []
    for model_name, filename in FORECAST_FILES.items():
        path = FORECASTS_DIR / filename
        if not path.exists():
            print(f"Skipping {model_name}: missing {path}")
            continue
        rows.extend(backtest_file(model_name, path))
    return pd.DataFrame(rows)


def main() -> None:
    ensure_directories()
    results = run_backtests()
    output_path = TABLES_DIR / "backtesting_results.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved backtesting results to {output_path}")


if __name__ == "__main__":
    main()

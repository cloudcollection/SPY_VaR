"""Common-sample evaluation with explicit handling of missing observations."""
import math

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from .config import LEVELS, VAR_COLUMNS


def _binary(values):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isin(values, [0, 1]).all():
        raise ValueError("Violations must be a one-dimensional binary sequence")
    return values.astype(int)


def kupiec_test(violations, alpha):
    values = _binary(violations)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(values)
    if n == 0:
        return math.nan, math.nan
    x = int(values.sum())
    p = x / n
    ll_alt = xlogy(x, p) + xlogy(n - x, 1 - p)
    ll_null = xlogy(x, alpha) + xlogy(n - x, 1 - alpha)
    lr = max(0.0, float(2 * (ll_alt - ll_null)))
    return lr, float(chi2.sf(lr, 1))


def christoffersen_independence(violations, positions=None):
    values = _binary(violations)
    if len(values) < 2:
        return math.nan, math.nan
    adjacent = np.ones(len(values) - 1, dtype=bool)
    if positions is not None:
        positions = np.asarray(positions)
        if len(positions) != len(values):
            raise ValueError("positions and violations must have the same length")
        adjacent = np.diff(positions) == 1
    prev, curr = values[:-1][adjacent], values[1:][adjacent]
    counts = np.array([((prev == a) & (curr == b)).sum() for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))])
    n00, n01, n10, n11 = counts
    if counts.sum() == 0:
        return math.nan, math.nan
    # No visits to one of the states: the two-state alternative is unidentified.
    if n00 + n01 == 0 or n10 + n11 == 0:
        return math.nan, math.nan
    p = (n01 + n11) / counts.sum()
    p0, p1 = n01 / (n00 + n01), n11 / (n10 + n11)
    ll_null = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    ll_alt = xlogy(n00, 1 - p0) + xlogy(n01, p0) + xlogy(n10, 1 - p1) + xlogy(n11, p1)
    lr = max(0.0, float(2 * (ll_alt - ll_null)))
    return lr, float(chi2.sf(lr, 1))


def pinball_loss(realized, forecast, alpha):
    errors = np.asarray(realized) - np.asarray(forecast)
    return float(np.maximum(alpha * errors, (alpha - 1) * errors).mean()) if errors.size else math.nan


def normalize_forecast(frame):
    frame = frame.rename(columns={
        "date/index": "date", "realized_return": "realized_log_ret",
        "VaR_0.01": "VaR_1", "VaR_0.05": "VaR_5", "VaR_0.10": "VaR_10",
    }).copy()
    required = ["date", "realized_log_ret", *VAR_COLUMNS.values()]
    if set(required) - set(frame):
        raise ValueError(f"Forecast needs columns {required}")
    frame = frame[required]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.empty or frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Forecast dates must be nonempty, valid and unique")
    for col in required[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    return frame.sort_values("date").set_index("date")


def evaluate(frames: dict[str, pd.DataFrame], calendar=None):
    if not frames:
        raise ValueError("No forecasts supplied")
    frames = {name: normalize_forecast(frame) for name, frame in frames.items()}
    union = pd.DatetimeIndex(sorted(set().union(*(frame.index for frame in frames.values()))))
    timeline = pd.DatetimeIndex(calendar) if calendar is not None else union
    if not timeline.is_unique or not timeline.is_monotonic_increasing or timeline.hasnans:
        raise ValueError("Calendar must be unique and increasing")
    if not union.isin(timeline).all():
        raise ValueError("Forecast dates fall outside the evaluation calendar")
    # Conflicting realized returns must never silently enter a model comparison.
    realized = pd.concat({name: frame["realized_log_ret"] for name, frame in frames.items()}, axis=1).reindex(union)
    for col in realized:
        reference = realized.bfill(axis=1).iloc[:, 0]
        mask = realized[col].notna() & reference.notna()
        if not np.allclose(realized.loc[mask, col], reference[mask], rtol=1e-7, atol=1e-10):
            raise ValueError("Models disagree on realized returns for the same date")
    common = union
    for frame in frames.values():
        common = common.intersection(frame.index)
    if common.empty:
        raise ValueError("Models have no common forecast dates")
    rows, date_rows = [], []
    for alpha, var_col in VAR_COLUMNS.items():
        valid = np.ones(len(common), dtype=bool)
        for frame in frames.values():
            valid &= np.isfinite(frame.loc[common, ["realized_log_ret", var_col]].to_numpy()).all(axis=1)
        dates = common[valid]
        if dates.empty:
            raise ValueError(f"No common finite forecasts at alpha={alpha}")
        positions = timeline.get_indexer(dates)
        date_rows.extend({"alpha": alpha, "date": date} for date in dates)
        for name, frame in frames.items():
            r, q = frame.loc[dates, ["realized_log_ret", var_col]].to_numpy().T
            violations = (r < q).astype(int)
            uc_lr, uc_p = kupiec_test(violations, alpha)
            ind_lr, ind_p = christoffersen_independence(violations, positions)
            cc_lr = uc_lr + ind_lr
            own = frame[["realized_log_ret", var_col]].to_numpy()
            rows.append({
                "model": name, "alpha": alpha, "n_obs": len(dates),
                "start_date": dates.min().date().isoformat(), "end_date": dates.max().date().isoformat(),
                "n_violations": int(violations.sum()), "expected_violations": alpha * len(dates),
                "failure_rate": float(violations.mean()), "avg_var": float(q.mean()),
                "kupiec_pvalue": uc_p, "christoffersen_ind_pvalue": ind_p,
                "christoffersen_cc_pvalue": float(chi2.sf(cc_lr, 2)),
                "pinball_loss": pinball_loss(r, q, alpha),
                "n_invalid_own": int((~np.isfinite(own).all(axis=1)).sum()),
                "n_excluded_from_own": len(frame) - len(dates),
                "n_independence_transitions": int((np.diff(positions) == 1).sum()),
            })
    return pd.DataFrame(rows), pd.DataFrame(date_rows)

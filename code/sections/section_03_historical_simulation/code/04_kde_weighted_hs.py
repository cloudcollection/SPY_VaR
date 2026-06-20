from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from utils import VAR_COLUMN_MAP, VAR_LEVELS, ensure_directories, load_spy_data, save_forecast


DEFAULT_WINDOW_SIZE = 1000
DEFAULT_DECAY = 0.98


def exponential_weights(n_obs: int, decay: float = DEFAULT_DECAY) -> np.ndarray:
    weights = decay ** np.arange(n_obs - 1, -1, -1)
    return weights / weights.sum()


def kde_quantile(values: pd.Series, alpha: float, decay: float = DEFAULT_DECAY, bw_method: str | float = "scott") -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 20:
        return float("nan")

    weights = exponential_weights(len(clean), decay=decay)
    kde = stats.gaussian_kde(clean, bw_method=bw_method, weights=weights)

    bandwidth = float(np.sqrt(kde.covariance.squeeze()))
    lower = float(clean.min() - 8.0 * bandwidth)
    upper = float(clean.max() + 8.0 * bandwidth)

    def centered_cdf(x: float) -> float:
        return float(kde.integrate_box_1d(-np.inf, x) - alpha)

    try:
        return float(optimize.brentq(centered_cdf, lower, upper, maxiter=100))
    except ValueError:
        grid = np.linspace(lower, upper, 2000)
        cdf = np.array([kde.integrate_box_1d(-np.inf, x) for x in grid])
        return float(np.interp(alpha, cdf, grid))


def kde_weighted_hs_var(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    decay: float = DEFAULT_DECAY,
    bw_method: str | float = "scott",
) -> pd.DataFrame:
    rows = []
    returns = df["log_ret"]
    for pos in range(window_size, len(df)):
        window = returns.iloc[pos - window_size : pos]
        row = {
            "date": df.index[pos].strftime("%Y-%m-%d") if hasattr(df.index[pos], "strftime") else pos,
            "realized_log_ret": returns.iloc[pos],
        }
        for alpha in VAR_LEVELS:
            row[VAR_COLUMN_MAP[alpha]] = kde_quantile(window, alpha, decay=decay, bw_method=bw_method)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ensure_directories()
    df = load_spy_data()
    forecasts = kde_weighted_hs_var(df)
    path = save_forecast(forecasts, "var_kde_weighted_hs.csv")
    print(
        "Saved Gaussian KDE weighted historical simulation VaR forecasts "
        f"(window_size={DEFAULT_WINDOW_SIZE}, decay={DEFAULT_DECAY}) to {path}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import pandas as pd

from utils import VAR_COLUMN_MAP, VAR_LEVELS, WINDOW_SIZE, ensure_directories, load_spy_data, save_forecast


def weighted_quantile(values: pd.Series, alpha: float, decay: float = 0.98) -> float:
    """Compute an exponentially weighted lower-tail quantile.

    The newest observation receives the largest weight. This is an
    age-weighted historical simulation variant, not plain empirical HS.
    """
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    n = len(clean)
    weights = decay ** (n - 1 - pd.Series(range(n), index=clean.index))
    weights = weights / weights.sum()
    ordered = pd.DataFrame({"value": clean.to_numpy(), "weight": weights.to_numpy()}).sort_values("value")
    cumulative_weight = ordered["weight"].cumsum()
    return float(ordered.loc[cumulative_weight >= alpha, "value"].iloc[0])


def historical_simulation_var(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    decay: float = 0.98,
) -> pd.DataFrame:
    rows = []
    returns = df["log_ret"]
    for pos in range(window_size, len(df)):
        window = returns.iloc[pos - window_size : pos].dropna()
        row = {
            "date": df.index[pos].strftime("%Y-%m-%d") if hasattr(df.index[pos], "strftime") else pos,
            "realized_log_ret": returns.iloc[pos],
        }
        for alpha in VAR_LEVELS:
            row[VAR_COLUMN_MAP[alpha]] = weighted_quantile(window, alpha, decay=decay)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ensure_directories()
    df = load_spy_data()
    forecasts = historical_simulation_var(df)
    path = save_forecast(forecasts, "var_historical_simulation.csv")
    print(f"Saved exponentially weighted historical simulation VaR forecasts to {path}")


if __name__ == "__main__":
    main()

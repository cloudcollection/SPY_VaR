from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from utils import VAR_COLUMN_MAP, VAR_LEVELS, WINDOW_SIZE, ensure_directories, load_spy_data, save_forecast


def _student_t_quantile(alpha: float, nu: float) -> float:
    # arch standardizes Student-t innovations to unit variance.
    raw_q = stats.t.ppf(alpha, df=nu)
    return float(raw_q * np.sqrt((nu - 2.0) / nu)) if nu > 2 else float(raw_q)


def garch_var(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    distribution: str = "t",
    scale_returns: float = 100.0,
    asymmetric: bool = False,
) -> pd.DataFrame:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("The 'arch' package is required. Install dependencies with: pip install -r requirements.txt") from exc

    rows = []
    returns = df["log_ret"]
    dist_name = "StudentsT" if distribution.lower() in {"t", "student", "student-t"} else "normal"

    for pos in range(window_size, len(df)):
        window = returns.iloc[pos - window_size : pos].dropna() * scale_returns
        row = {
            "date": df.index[pos].strftime("%Y-%m-%d") if hasattr(df.index[pos], "strftime") else pos,
            "realized_log_ret": returns.iloc[pos],
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(
                    window,
                    mean="Constant",
                    vol="GARCH",
                    p=1,
                    o=1 if asymmetric else 0,
                    q=1,
                    dist=dist_name,
                    rescale=False,
                )
                result = model.fit(disp="off", show_warning=False)
                forecast = result.forecast(horizon=1, reindex=False)
            mu = float(forecast.mean.iloc[-1, 0])
            sigma = float(np.sqrt(forecast.variance.iloc[-1, 0]))
            nu = float(result.params.get("nu", np.nan))

            for alpha in VAR_LEVELS:
                q = _student_t_quantile(alpha, nu) if dist_name == "StudentsT" else float(stats.norm.ppf(alpha))
                row[VAR_COLUMN_MAP[alpha]] = (mu + sigma * q) / scale_returns
        except Exception as exc:
            print(f"GARCH failed at position {pos}: {exc}")
            for alpha in VAR_LEVELS:
                row[VAR_COLUMN_MAP[alpha]] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ensure_directories()
    df = load_spy_data()
    garch_forecasts = garch_var(df, distribution="t", asymmetric=False)
    garch_path = save_forecast(garch_forecasts, "var_garch_t.csv")
    print(f"Saved GARCH-t VaR forecasts to {garch_path}")

    gjr_forecasts = garch_var(df, distribution="t", asymmetric=True)
    gjr_path = save_forecast(gjr_forecasts, "var_gjr_garch_t.csv")
    print(f"Saved GJR-GARCH-t VaR forecasts to {gjr_path}")


if __name__ == "__main__":
    main()

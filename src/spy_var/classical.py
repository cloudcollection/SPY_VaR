"""Historical simulation and rolling Student-t volatility benchmarks."""
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .config import LEVELS, VAR_COLUMNS


def historical_forecast(data, config, kind):
    rows = []
    for pos in range(config.window_size, len(data)):
        values = data.log_ret.iloc[pos - config.window_size:pos].to_numpy()
        weights = config.decay ** np.arange(len(values) - 1, -1, -1)
        weights /= weights.sum()
        if kind == "hs":
            quantiles = np.quantile(values, LEVELS)
        elif kind == "weighted_hs":
            order = np.argsort(values)
            cumulative = np.cumsum(weights[order])
            quantiles = [values[order[min(np.searchsorted(cumulative, alpha), len(values) - 1)]] for alpha in LEVELS]
        elif kind == "kde_hs":
            # The degenerate constant-return case has a point-mass distribution.
            if np.ptp(values) == 0:
                quantiles = np.repeat(values[0], len(LEVELS))
            else:
                kde = stats.gaussian_kde(values, weights=weights, bw_method="scott")
                bandwidth = float(np.sqrt(kde.covariance.squeeze()))
                bounds = (float(values.min() - 8 * bandwidth), float(values.max() + 8 * bandwidth))
                quantiles = [optimize.brentq(lambda x: kde.integrate_box_1d(-np.inf, x) - alpha, *bounds) for alpha in LEVELS]
        else:
            raise ValueError(f"Unknown historical simulation variant: {kind}")
        rows.append({"date": data.index[pos], "realized_log_ret": data.log_ret.iloc[pos],
                     **{VAR_COLUMNS[alpha]: float(q) for alpha, q in zip(LEVELS, quantiles)}})
    return pd.DataFrame(rows)


def garch_forecast(data, config, asymmetric=False):
    from arch import arch_model

    rows = []
    for pos in range(config.window_size, len(data)):
        row = {"date": data.index[pos], "realized_log_ret": data.log_ret.iloc[pos],
               "sigma_next": np.nan, "convergence_flag": -1, "fit_warning": "", "fit_error": ""}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = arch_model(data.log_ret.iloc[pos - config.window_size:pos] * 100,
                                   mean="Constant", vol="GARCH", p=1, o=int(asymmetric), q=1,
                                   dist="StudentsT", rescale=False)
                fitted = model.fit(disp="off", show_warning=True)
                row["convergence_flag"] = int(fitted.convergence_flag)
                row["fit_warning"] = " | ".join(str(item.message) for item in caught)
                if fitted.convergence_flag:
                    raise RuntimeError(f"Optimizer convergence flag {fitted.convergence_flag}")
                forecast = fitted.forecast(horizon=1, reindex=False)
            nu = float(fitted.params["nu"])
            if nu <= 2:
                raise ValueError("Student-t degrees of freedom must exceed 2")
            mu = float(forecast.mean.iloc[-1, 0]) / 100
            sigma = float(np.sqrt(forecast.variance.iloc[-1, 0])) / 100
            row["sigma_next"] = sigma
            for alpha in LEVELS:
                row[VAR_COLUMNS[alpha]] = mu + sigma * stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)
        except Exception as exc:
            row["fit_error"] = f"{type(exc).__name__}: {exc}"
            row.update({column: np.nan for column in VAR_COLUMNS.values()})
        rows.append(row)
    return pd.DataFrame(rows)

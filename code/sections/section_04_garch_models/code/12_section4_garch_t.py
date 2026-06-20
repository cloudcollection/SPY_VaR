from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from utils import (
    FIGURES_DIR,
    FORECASTS_DIR,
    TABLES_DIR,
    VAR_COLUMN_MAP,
    VAR_LEVELS,
    ensure_directories,
    load_spy_data,
)


WINDOWS = [250, 500, 1000]
CRISIS_START = "2007-09-01"
CRISIS_END = "2009-06-30"
CALM_START = "2012-01-01"
CALM_END = "2016-12-31"


def _student_t_quantile(alpha: float, nu: float) -> float:
    raw_q = stats.t.ppf(alpha, df=nu)
    return float(raw_q * math.sqrt((nu - 2.0) / nu)) if nu > 2 else float(raw_q)


def kupiec_test(violations: pd.Series, alpha: float) -> float:
    v = int(violations.sum())
    t = int(violations.notna().sum())
    if v == 0 or v == t:
        return np.nan
    p_hat = v / t
    lr = -2 * (
        (t - v) * np.log((1 - alpha) / (1 - p_hat))
        + v * np.log(alpha / p_hat)
    )
    return float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_independence_test(violations: pd.Series) -> float:
    v = violations.dropna().astype(int).to_numpy()
    if len(v) < 2:
        return np.nan
    t00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    t01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    t10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    t11 = np.sum((v[:-1] == 1) & (v[1:] == 1))
    if (t00 + t01) == 0 or (t10 + t11) == 0:
        return np.nan

    pi01 = t01 / (t00 + t01)
    pi11 = t11 / (t10 + t11)
    p = (t01 + t11) / (t00 + t01 + t10 + t11)

    def xlogx(n: int, prob: float) -> float:
        if n == 0:
            return 0.0
        return n * np.log(max(prob, 1e-12))

    ll_null = xlogx(t00 + t10, 1 - p) + xlogx(t01 + t11, p)
    ll_alt = (
        xlogx(t00, 1 - pi01)
        + xlogx(t01, pi01)
        + xlogx(t10, 1 - pi11)
        + xlogx(t11, pi11)
    )
    lr = max(-2 * (ll_null - ll_alt), 0.0)
    return float(1 - stats.chi2.cdf(lr, df=1))


def duration_test(violations: pd.Series, alpha: float) -> float:
    idx = np.where(violations.dropna().astype(int).to_numpy() == 1)[0]
    if len(idx) < 3:
        return np.nan
    durations = np.diff(idx).astype(float)
    if len(durations) < 2:
        return np.nan

    def weibull_nll(params: np.ndarray) -> float:
        a, b = params
        if a <= 0 or b <= 0:
            return 1e12
        log_pdf = np.log(a * b) + (b - 1) * np.log(a * durations) - (a * durations) ** b
        return float(-np.sum(log_pdf))

    null_nll = float(-np.sum(np.log(alpha) - alpha * durations))
    res = minimize(
        weibull_nll,
        x0=np.array([alpha, 1.0]),
        bounds=[(1e-6, 1.0), (0.1, 10.0)],
        method="L-BFGS-B",
    )
    if not res.success:
        return np.nan
    lr = max(2 * (null_nll - float(res.fun)), 0.0)
    return float(1 - stats.chi2.cdf(lr, df=1))


def lopez_loss(returns: pd.Series, var_forecast: pd.Series, violations: pd.Series) -> float:
    exceedance = returns - var_forecast
    return float(np.where(violations, 1.0 + exceedance**2, 0.0).mean())


def run_backtest(
    returns: pd.Series,
    var_forecast: pd.Series,
    alpha: float,
    window: int,
    model_name: str = "GARCH(1,1)-t",
) -> dict[str, object]:
    mask = returns.notna() & var_forecast.notna()
    r = returns.loc[mask]
    v = var_forecast.loc[mask]
    violations = r < v
    obs = int(mask.sum())
    count = int(violations.sum())
    return {
        "Model": model_name,
        "Window": window,
        "Alpha": f"{int(alpha * 100)}%",
        "Viol./Exp.": f"{count} / {alpha * obs:.1f}",
        "Fail. rate": round(count / obs, 4),
        "Avg VaR": round(float(v.mean()), 4),
        "Kupiec p": round(kupiec_test(violations, alpha), 4),
        "Christoffersen p": round(christoffersen_independence_test(violations), 4),
        "Duration p": round(duration_test(violations, alpha), 4),
        "Lopez loss": round(lopez_loss(r, v, violations), 6),
        "n_obs": obs,
        "n_violations": count,
        "expected_violations": round(alpha * obs, 1),
    }


def fit_garch_t_rolling(
    returns: pd.Series,
    window: int,
    asymmetric: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("Install project dependencies first: pip install -r requirements.txt") from exc

    values = returns.to_numpy()
    dates = returns.index
    forecast_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    last_params = None

    for pos in range(window, len(returns)):
        train = values[pos - window : pos] * 100.0
        row: dict[str, object] = {
            "date": dates[pos],
            "realized_log_ret": values[pos],
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = arch_model(
                    train,
                    mean="Constant",
                    vol="GARCH",
                    p=1,
                    o=1 if asymmetric else 0,
                    q=1,
                    dist="StudentsT",
                    rescale=False,
                )
                fit = model.fit(
                    disp="off",
                    show_warning=False,
                    starting_values=last_params,
                    options={"maxiter": 300},
                )
            last_params = fit.params.to_numpy()
            forecast = fit.forecast(horizon=1, reindex=False)
            mu_pct = float(forecast.mean.iloc[-1, 0])
            sigma_pct = float(np.sqrt(forecast.variance.iloc[-1, 0]))
            nu = float(fit.params["nu"])
            for alpha, col in VAR_COLUMN_MAP.items():
                row[col] = (mu_pct + sigma_pct * _student_t_quantile(alpha, nu)) / 100.0
            param_rows.append(
                {
                    "date": dates[pos],
                    "mu": float(fit.params["mu"]) / 100.0,
                    "omega": float(fit.params["omega"]),
                    "alpha_1": float(fit.params["alpha[1]"]),
                    "gamma_1": float(fit.params.get("gamma[1]", 0.0)),
                    "beta_1": float(fit.params["beta[1]"]),
                    "persistence": float(
                        fit.params["alpha[1]"]
                        + fit.params["beta[1]"]
                        + 0.5 * fit.params.get("gamma[1]", 0.0)
                    ),
                    "nu": nu,
                    "sigma_next": sigma_pct / 100.0,
                    "converged": True,
                }
            )
        except Exception as exc:
            for col in VAR_COLUMN_MAP.values():
                row[col] = np.nan
            param_rows.append(
                {
                    "date": dates[pos],
                    "mu": np.nan,
                    "omega": np.nan,
                    "alpha_1": np.nan,
                    "gamma_1": np.nan,
                    "beta_1": np.nan,
                    "persistence": np.nan,
                    "nu": np.nan,
                    "sigma_next": np.nan,
                    "converged": False,
                    "error": str(exc),
                }
            )
        forecast_rows.append(row)
        if (pos - window + 1) % 500 == 0:
            print(f"W={window}: fitted {pos - window + 1}/{len(returns) - window}")

    forecasts = pd.DataFrame(forecast_rows)
    params = pd.DataFrame(param_rows)
    return forecasts, params


def _standardized_t_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    scale = math.sqrt((nu - 2.0) / nu)
    return stats.t.logpdf(z / scale, df=nu) - math.log(scale)


def _garchx_neg_loglik(params: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    mu, omega, alpha, beta, gamma, nu = params
    if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0 or nu <= 2.05 or alpha + beta >= 0.999:
        return 1e12

    eps = y - mu
    h = np.empty_like(y)
    unconditional = max(float(np.var(y)), 1e-6)
    h[0] = unconditional
    for i in range(1, len(y)):
        h[i] = omega + alpha * eps[i - 1] ** 2 + beta * h[i - 1] + gamma * x[i - 1]
        if not np.isfinite(h[i]) or h[i] <= 1e-10:
            return 1e12

    z = eps / np.sqrt(h)
    ll = _standardized_t_logpdf(z, nu) - 0.5 * np.log(h)
    if not np.all(np.isfinite(ll)):
        return 1e12
    return float(-np.sum(ll))


def fit_garchx_t_rolling(
    df: pd.DataFrame,
    window: int,
    realized_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = df["log_ret"].dropna()
    aligned = df.loc[returns.index, ["log_ret", realized_col]].dropna()
    y_all = (aligned["log_ret"] * 100.0).to_numpy()
    # rv5 and bv are daily variance-scale realized measures. Multiplying by
    # 10000 converts them to percent-squared units, matching y_all.
    x_all = (aligned[realized_col].clip(lower=1e-12) * 10000.0).to_numpy()
    dates = aligned.index

    forecast_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    last_params = None

    for pos in range(window, len(aligned)):
        y = y_all[pos - window : pos]
        x = x_all[pos - window : pos]
        row: dict[str, object] = {
            "date": dates[pos],
            "realized_log_ret": aligned["log_ret"].iloc[pos],
        }
        start = (
            last_params
            if last_params is not None
            else np.array([float(np.mean(y)), max(float(np.var(y)) * 0.02, 1e-5), 0.06, 0.88, 0.04, 8.0])
        )
        try:
            res = minimize(
                _garchx_neg_loglik,
                start,
                args=(y, x),
                method="SLSQP",
                bounds=[
                    (-2.0, 2.0),
                    (1e-8, 20.0),
                    (0.0, 0.998),
                    (0.0, 0.998),
                    (0.0, 20.0),
                    (2.05, 80.0),
                ],
                constraints=({"type": "ineq", "fun": lambda p: 0.998 - p[2] - p[3]},),
                options={"maxiter": 250, "ftol": 1e-7, "disp": False},
            )
            if not res.success:
                raise RuntimeError(res.message)

            mu, omega, alpha, beta, gamma, nu = [float(v) for v in res.x]
            last_params = res.x
            eps = y - mu
            h = np.empty_like(y)
            h[0] = max(float(np.var(y)), 1e-6)
            for i in range(1, len(y)):
                h[i] = omega + alpha * eps[i - 1] ** 2 + beta * h[i - 1] + gamma * x[i - 1]
            h_next = omega + alpha * eps[-1] ** 2 + beta * h[-1] + gamma * x[-1]
            sigma_next = math.sqrt(max(h_next, 1e-10)) / 100.0
            for alpha_level, col in VAR_COLUMN_MAP.items():
                row[col] = mu / 100.0 + sigma_next * _student_t_quantile(alpha_level, nu)
            param_rows.append(
                {
                    "date": dates[pos],
                    "mu": mu / 100.0,
                    "omega": omega,
                    "alpha_1": alpha,
                    "beta_1": beta,
                    "gamma_x": gamma,
                    "persistence": alpha + beta,
                    "nu": nu,
                    "sigma_next": sigma_next,
                    "realized_col": realized_col,
                    "converged": True,
                }
            )
        except Exception as exc:
            for col in VAR_COLUMN_MAP.values():
                row[col] = np.nan
            param_rows.append(
                {
                    "date": dates[pos],
                    "mu": np.nan,
                    "omega": np.nan,
                    "alpha_1": np.nan,
                    "beta_1": np.nan,
                    "gamma_x": np.nan,
                    "persistence": np.nan,
                    "nu": np.nan,
                    "sigma_next": np.nan,
                    "realized_col": realized_col,
                    "converged": False,
                    "error": str(exc),
                }
            )
        forecast_rows.append(row)
        if (pos - window + 1) % 500 == 0:
            print(f"GARCH-X-{realized_col}, W={window}: fitted {pos - window + 1}/{len(aligned) - window}")

    return pd.DataFrame(forecast_rows), pd.DataFrame(param_rows)


def subsample_rows(returns: pd.Series, forecasts: pd.DataFrame, period: str, start: str, end: str) -> list[dict[str, object]]:
    f = forecasts.set_index("date")
    rows = []
    for alpha, col in VAR_COLUMN_MAP.items():
        sub_r = returns.loc[start:end]
        sub_v = f[col].loc[start:end]
        mask = sub_r.notna() & sub_v.notna()
        r = sub_r.loc[mask]
        v = sub_v.loc[mask]
        violations = r < v
        obs = int(mask.sum())
        count = int(violations.sum())
        rows.append(
            {
                "Model": "GARCH(1,1)-t",
                "Period": period,
                "Start": start,
                "End": end,
                "Alpha": f"{int(alpha * 100)}%",
                "Obs.": obs,
                "Viol./Exp.": f"{count} / {alpha * obs:.1f}",
                "Fail. rate": round(count / obs, 4),
                "Avg VaR": round(float(v.mean()), 4),
            }
        )
    return rows


def parameter_summary(params: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = [
        ("mu", r"$\hat{\mu}$", "Daily mean return"),
        ("omega", r"$\hat{\omega}$", "Baseline variance"),
        ("alpha_1", r"$\hat{\alpha}_1$", "ARCH effect (shock sensitivity)"),
        ("beta_1", r"$\hat{\beta}_1$", "Volatility persistence"),
        ("persistence", r"$\hat{\alpha}_1+\hat{\beta}_1$", "Total persistence"),
        ("nu", r"$\hat{\nu}$", "Tail degrees of freedom"),
    ]
    for col, label, interpretation in labels:
        series = params[col].dropna()
        q25, q75 = series.quantile([0.25, 0.75])
        rows.append(
            {
                "Parameter": label,
                "Median": round(float(series.median()), 6),
                "IQR": f"{q25:.6f} to {q75:.6f}",
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def plot_var_forecast(returns: pd.Series, forecasts: pd.DataFrame, window: int) -> Path:
    f = forecasts.set_index("date")
    var = f["VaR_1"].dropna()
    fig, ax = plt.subplots(figsize=(12, 4))
    r_plot = returns.dropna()
    ax.fill_between(r_plot.index, r_plot, 0, where=(r_plot < 0), color="#CCCCCC", alpha=0.4)
    ax.plot(r_plot.index, r_plot, color="#888888", lw=0.5, label="Realized log return", alpha=0.6)
    ax.plot(var.index, var, lw=1.2, color="#1A5F9E", label="GARCH(1,1)-t 1% VaR")
    ax.axhline(0, color="black", lw=0.4, ls="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Log return", fontsize=10)
    ax.set_title(f"1% VaR Forecasts for GARCH(1,1)-t, W={window}", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_1_garch_t_var_forecast_w1000.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_violation_indicators(returns: pd.Series, forecasts: pd.DataFrame, window: int) -> Path:
    f = forecasts.set_index("date")
    var = f["VaR_1"].reindex(returns.index)
    mask = returns.notna() & var.notna()
    viol_dates = returns.loc[mask & (returns < var)].index
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.scatter(viol_dates, np.ones(len(viol_dates)), s=10, color="#1A5F9E", alpha=0.85)
    ax.set_yticks([1])
    ax.set_yticklabels(["GARCH(1,1)-t"], fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"1% VaR Violation Indicators for GARCH(1,1)-t, W={window}", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_2_garch_t_violation_indicators_w1000.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_conditional_volatility(params: pd.DataFrame, window: int) -> Path:
    p = params.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.set_index("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    annualized_vol = p["sigma_next"] * np.sqrt(252) * 100
    axes[0].plot(annualized_vol.index, annualized_vol, color="#1A5F9E", lw=0.8)
    axes[0].set_ylabel("Annualized sigma (%)", fontsize=9)
    axes[0].set_title(f"GARCH(1,1)-t Conditional Volatility, W={window}", fontsize=11)
    axes[1].plot(p.index, p["nu"], color="#C04828", lw=0.8)
    axes[1].axhline(p["nu"].median(), ls="--", color="gray", lw=0.6, label=f"Median nu = {p['nu'].median():.1f}")
    axes[1].set_ylabel("Degrees of freedom nu", fontsize=9)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    out = FIGURES_DIR / "fig4_3_garch_t_conditional_volatility_w1000.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ensure_directories()
    df = load_spy_data()
    returns = df["log_ret"].dropna()
    all_backtests: list[dict[str, object]] = []
    all_subsamples: list[dict[str, object]] = []
    params_w1000 = None
    forecasts_w1000 = None

    for window in WINDOWS:
        forecasts_out = FORECASTS_DIR / f"var_garch_t_w{window}.csv"
        params_out = TABLES_DIR / f"section4_garch_params_w{window}.csv"
        if forecasts_out.exists() and params_out.exists():
            print(f"Using cached rolling GARCH(1,1)-t results for W={window}")
            forecasts = pd.read_csv(forecasts_out, parse_dates=["date"])
            params = pd.read_csv(params_out, parse_dates=["date"])
        else:
            print(f"Running rolling GARCH(1,1)-t for W={window}")
            forecasts, params = fit_garch_t_rolling(returns, window)
            forecasts.to_csv(forecasts_out, index=False)
            params.to_csv(params_out, index=False)
        f_indexed = forecasts.set_index("date")
        for alpha, col in VAR_COLUMN_MAP.items():
            all_backtests.append(run_backtest(returns, f_indexed[col], alpha, window))
        if window == 1000:
            forecasts_w1000 = forecasts
            params_w1000 = params
            all_subsamples.extend(subsample_rows(returns, forecasts, "Crisis", CRISIS_START, CRISIS_END))
            all_subsamples.extend(subsample_rows(returns, forecasts, "Post-crisis calm", CALM_START, CALM_END))

    backtest_df = pd.DataFrame(all_backtests)
    backtest_df.to_csv(TABLES_DIR / "section4_garch_backtest.csv", index=False)
    pd.DataFrame(all_subsamples).to_csv(TABLES_DIR / "section4_garch_subsample.csv", index=False)

    if params_w1000 is not None and forecasts_w1000 is not None:
        parameter_summary(params_w1000).to_csv(TABLES_DIR / "section4_garch_parameter_summary_w1000.csv", index=False)
        plot_var_forecast(returns, forecasts_w1000, 1000)
        plot_violation_indicators(returns, forecasts_w1000, 1000)
        plot_conditional_volatility(params_w1000, 1000)

    robustness_rows = []
    if forecasts_w1000 is not None:
        baseline_indexed = forecasts_w1000.set_index("date")
        for alpha, col in VAR_COLUMN_MAP.items():
            robustness_rows.append(run_backtest(returns, baseline_indexed[col], alpha, 1000, "GARCH(1,1)-t"))

    for model_name, asymmetric, realized_col in [
        ("GJR-GARCH(1,1)-t", True, None),
        ("GARCH-X-rv5-t", False, "rv5"),
        ("GARCH-X-bv-t", False, "bv"),
    ]:
        safe_name = model_name.lower().replace("(", "").replace(")", "").replace(",", "").replace("-", "_")
        forecasts_out = FORECASTS_DIR / f"var_{safe_name}_w1000.csv"
        params_out = TABLES_DIR / f"section4_{safe_name}_params_w1000.csv"
        if forecasts_out.exists() and params_out.exists():
            print(f"Using cached {model_name} robustness results")
            forecasts = pd.read_csv(forecasts_out, parse_dates=["date"])
            params = pd.read_csv(params_out, parse_dates=["date"])
        elif realized_col is None:
            print(f"Running {model_name} robustness for W=1000")
            forecasts, params = fit_garch_t_rolling(returns, 1000, asymmetric=asymmetric)
            forecasts.to_csv(forecasts_out, index=False)
            params.to_csv(params_out, index=False)
        else:
            print(f"Running {model_name} robustness for W=1000")
            forecasts, params = fit_garchx_t_rolling(df, 1000, realized_col)
            forecasts.to_csv(forecasts_out, index=False)
            params.to_csv(params_out, index=False)
        f_indexed = forecasts.set_index("date")
        for alpha, col in VAR_COLUMN_MAP.items():
            robustness_rows.append(run_backtest(returns, f_indexed[col], alpha, 1000, model_name))

    pd.DataFrame(robustness_rows).to_csv(TABLES_DIR / "section4_garch_robustness_w1000.csv", index=False)

    print(f"Saved {TABLES_DIR / 'section4_garch_backtest.csv'}")
    print(f"Saved {TABLES_DIR / 'section4_garch_subsample.csv'}")
    print(f"Saved {TABLES_DIR / 'section4_garch_robustness_w1000.csv'}")


if __name__ == "__main__":
    main()

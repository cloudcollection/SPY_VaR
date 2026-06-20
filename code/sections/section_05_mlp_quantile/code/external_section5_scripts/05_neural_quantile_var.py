from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import chi2

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required. Install it with: pip install torch") from exc


SEED = 42
ALPHAS = np.array([0.01, 0.05, 0.10], dtype=np.float32)
WINDOW_SIZE = 1000
RETRAIN_EVERY = 20
MAX_EPOCHS = 200
PATIENCE = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

BASE_FEATURES = [
    "log_ret_lag1",
    "log_ret_lag2",
    "log_ret_lag3",
    "log_ret_lag4",
    "log_ret_lag5",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_std_22",
    "rolling_min_22",
    "rolling_max_22",
]
RV_FEATURES = ["rv5_lag1", "bv_lag1", "rv5_roll_5", "bv_roll_5"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directories() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def format_time_label(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return str(value)


def find_data_path() -> Path:
    candidates = [
        ROOT / "spy_data.csv",
        ROOT / "VaR_Project" / "data" / "spy_data.csv",
        ROOT / "data" / "spy_data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find spy_data.csv. Put it in the current directory, data/, "
        "or VaR_Project/data/."
    )


def load_spy_data(path: Path | None = None) -> pd.DataFrame:
    path = path or find_data_path()
    df = pd.read_csv(path)

    date_col = None
    for candidate in ["date", "Date", "datetime", "time", "timestamp"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        unnamed = [col for col in df.columns if str(col).startswith("Unnamed")]
        if unnamed:
            parsed = pd.to_datetime(df[unnamed[0]], errors="coerce")
            if parsed.notna().mean() > 0.95:
                date_col = unnamed[0]

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
        df.index.name = "date"
    else:
        df.index.name = "index"

    required = ["log_ret", "rv5", "bv"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in range(1, 6):
        out[f"log_ret_lag{lag}"] = out["log_ret"].shift(lag)

    shifted_ret = out["log_ret"].shift(1)
    out["rolling_mean_5"] = shifted_ret.rolling(5).mean()
    out["rolling_std_5"] = shifted_ret.rolling(5).std()
    out["rolling_std_22"] = shifted_ret.rolling(22).std()
    out["rolling_min_22"] = shifted_ret.rolling(22).min()
    out["rolling_max_22"] = shifted_ret.rolling(22).max()

    out["rv5_lag1"] = out["rv5"].shift(1)
    out["bv_lag1"] = out["bv"].shift(1)
    out["rv5_roll_5"] = out["rv5"].shift(1).rolling(5).mean()
    out["bv_roll_5"] = out["bv"].shift(1).rolling(5).mean()

    out["target"] = out["log_ret"].shift(-1)
    out["target_date"] = out.index.to_series().shift(-1).to_numpy()
    needed = BASE_FEATURES + RV_FEATURES + ["target", "target_date"]
    return out.dropna(subset=needed).copy()


class QuantileDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class MLPQuantileNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(ALPHAS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, alphas: torch.Tensor | None = None) -> torch.Tensor:
    if alphas is None:
        alphas = torch.as_tensor(ALPHAS, dtype=pred.dtype, device=pred.device).view(1, -1)
    errors = target - pred
    return torch.maximum(alphas * errors, (alphas - 1.0) * errors).mean()


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def train_one_model(
    x_window: np.ndarray,
    y_window: np.ndarray,
    input_dim: int,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[MLPQuantileNet, Standardizer]:
    n_train = int(len(x_window) * 0.8)
    x_train_raw, y_train = x_window[:n_train], y_window[:n_train]
    x_val_raw, y_val = x_window[n_train:], y_window[n_train:]

    scaler = Standardizer.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)

    model = MLPQuantileNet(input_dim)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(QuantileDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_x = torch.as_tensor(x_val, dtype=torch.float32)
    val_y = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    stale = 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = pinball_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss(model(val_x), val_y))

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def crossing_rate(raw_predictions: list[np.ndarray]) -> float:
    if not raw_predictions:
        return np.nan
    arr = np.vstack(raw_predictions)
    crossed = (arr[:, 0] > arr[:, 1]) | (arr[:, 1] > arr[:, 2])
    return float(crossed.mean())


def rolling_train_predict(data: pd.DataFrame, feature_cols: list[str], model_name: str) -> tuple[pd.DataFrame, float]:
    x_all = data[feature_cols].to_numpy(dtype=np.float32)
    y_all = data["target"].to_numpy(dtype=np.float32)
    target_dates = data["target_date"].to_numpy()

    rows = []
    raw_preds = []
    model = None
    scaler = None
    model_state = None

    for i in range(WINDOW_SIZE, len(data)):
        if model is None or (i - WINDOW_SIZE) % RETRAIN_EVERY == 0:
            x_window = x_all[i - WINDOW_SIZE : i]
            y_window = y_all[i - WINDOW_SIZE : i]
            valid = np.isfinite(x_window).all(axis=1) & np.isfinite(y_window)
            model, scaler = train_one_model(x_window[valid], y_window[valid], len(feature_cols), model_state)
            model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        x_pred = scaler.transform(x_all[i : i + 1])
        with torch.no_grad():
            raw = model(torch.as_tensor(x_pred, dtype=torch.float32)).numpy().reshape(-1)
        sorted_pred = np.sort(raw)
        raw_preds.append(raw)

        realized = float(y_all[i])
        date_label = format_time_label(target_dates[i])

        rows.append(
            {
                "date/index": date_label,
                "realized_return": realized,
                "model_name": model_name,
                "VaR_0.01": float(sorted_pred[0]),
                "VaR_0.05": float(sorted_pred[1]),
                "VaR_0.10": float(sorted_pred[2]),
                "violation_0.01": int(realized < sorted_pred[0]),
                "violation_0.05": int(realized < sorted_pred[1]),
                "violation_0.10": int(realized < sorted_pred[2]),
            }
        )

    return pd.DataFrame(rows), crossing_rate(raw_preds)


def _safe_log(x: float) -> float:
    return math.log(max(float(x), 1e-12))


def kupiec_test(violations: np.ndarray, alpha: float) -> float:
    n = len(violations)
    x = int(violations.sum())
    if n == 0:
        return np.nan
    phat = x / n
    ll_null = (n - x) * _safe_log(1 - alpha) + x * _safe_log(alpha)
    ll_alt = (n - x) * _safe_log(1 - phat) + x * _safe_log(phat) if 0 < x < n else 0.0
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return float(1.0 - chi2.cdf(lr, df=1))


def christoffersen_independence_test(violations: np.ndarray) -> float:
    if len(violations) < 2:
        return np.nan
    prev = violations[:-1].astype(int)
    curr = violations[1:].astype(int)
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    n0 = n00 + n01
    n1 = n10 + n11
    total = n0 + n1
    total_viol = n01 + n11
    if total == 0:
        return np.nan

    pi = total_viol / total
    pi0 = n01 / n0 if n0 else 0.0
    pi1 = n11 / n1 if n1 else 0.0
    ll_restricted = (total - total_viol) * _safe_log(1 - pi) + total_viol * _safe_log(pi)
    ll_unrestricted = (
        n00 * _safe_log(1 - pi0)
        + n01 * _safe_log(pi0)
        + n10 * _safe_log(1 - pi1)
        + n11 * _safe_log(pi1)
    )
    lr = max(0.0, -2.0 * (ll_restricted - ll_unrestricted))
    return float(1.0 - chi2.cdf(lr, df=1))


def duration_test(violations: np.ndarray) -> float:
    positions = np.flatnonzero(violations)
    if len(positions) < 3:
        warnings.warn("Duration test returned NaN because there are fewer than three violations.")
        return np.nan

    durations = np.diff(positions).astype(float)
    durations = durations[durations > 0]
    if len(durations) < 2:
        warnings.warn("Duration test returned NaN because violation durations are insufficient.")
        return np.nan

    def weibull_loglik(scale: float, shape: float) -> float:
        if scale <= 0 or shape <= 0:
            return -np.inf
        z = durations / scale
        return float(np.sum(np.log(shape) - shape * np.log(scale) + (shape - 1.0) * np.log(durations) - z**shape))

    restricted_scale = float(durations.mean())
    ll_restricted = weibull_loglik(restricted_scale, 1.0)

    def objective(params: np.ndarray) -> float:
        return -weibull_loglik(float(np.exp(params[0])), float(np.exp(params[1])))

    try:
        result = optimize.minimize(
            objective,
            x0=np.log([restricted_scale, 1.0]),
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8},
        )
    except Exception as exc:
        warnings.warn(f"Duration test failed and returned NaN: {exc}")
        return np.nan

    if not result.success:
        warnings.warn("Duration test optimization failed and returned NaN.")
        return np.nan

    lr = max(0.0, 2.0 * (-float(result.fun) - ll_restricted))
    return float(1.0 - chi2.cdf(lr, df=1))


def lopez_loss(realized: np.ndarray, var_forecast: np.ndarray) -> float:
    violations = realized < var_forecast
    losses = np.where(violations, 1.0 + (realized - var_forecast) ** 2, 0.0)
    return float(losses.mean())


def pinball_loss_np(realized: np.ndarray, var_forecast: np.ndarray, alpha: float) -> float:
    errors = realized - var_forecast
    return float(np.maximum(alpha * errors, (alpha - 1.0) * errors).mean())


def backtest_predictions(predictions: pd.DataFrame, crossing_rates: dict[str, float]) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model_name", sort=False):
        realized = group["realized_return"].to_numpy(dtype=float)
        for alpha in [0.01, 0.05, 0.10]:
            var_col = f"VaR_{alpha:.2f}"
            var_forecast = group[var_col].to_numpy(dtype=float)
            valid = np.isfinite(realized) & np.isfinite(var_forecast)
            r = realized[valid]
            v = var_forecast[valid]
            violations = (r < v).astype(int)
            rows.append(
                {
                    "model_name": model_name,
                    "alpha": alpha,
                    "n_forecasts": int(len(violations)),
                    "violations": int(violations.sum()),
                    "expected_violations": float(len(violations) * alpha),
                    "failure_rate": float(violations.mean()) if len(violations) else np.nan,
                    "avg_var": float(v.mean()) if len(v) else np.nan,
                    "kupiec_p": kupiec_test(violations, alpha),
                    "christoffersen_p": christoffersen_independence_test(violations),
                    "duration_p": duration_test(violations),
                    "lopez_loss": lopez_loss(r, v) if len(v) else np.nan,
                    "pinball_loss": pinball_loss_np(r, v, alpha) if len(v) else np.nan,
                    "crossing_rate": crossing_rates.get(model_name, np.nan),
                }
            )
    return pd.DataFrame(rows)


def plot_var_comparison(predictions: pd.DataFrame) -> None:
    pivot = predictions.pivot(index="date/index", columns="model_name")
    x = np.arange(len(pivot))
    step = max(1, len(x) // 10)

    plt.figure(figsize=(13, 6))
    plt.plot(x, pivot["realized_return"].iloc[:, 0], color="0.65", linewidth=0.8, label="Realized log returns")
    for model_name, color in [("Model A: MLP-QR", "tab:blue"), ("Model B: MLP-QR + RV", "tab:orange")]:
        plt.plot(x, pivot["VaR_0.01"][model_name], linewidth=1.0, label=f"{model_name} 1% VaR", color=color)
    plt.xticks(x[::step], pivot.index[::step], rotation=45, ha="right")
    plt.title("One-day-ahead 1% VaR forecasts from MLP quantile models")
    plt.ylabel("Log return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "section5_nn_var_1pct_comparison.png", dpi=200)
    plt.close()


def plot_violation_indicators(predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 4))
    for offset, (model_name, group) in enumerate(predictions.groupby("model_name", sort=False)):
        x = np.arange(len(group))
        y = group["violation_0.01"].to_numpy() + offset * 1.2
        plt.step(x, y, where="post", label=model_name)
    step = max(1, len(predictions["date/index"].unique()) // 10)
    labels = predictions["date/index"].drop_duplicates().to_numpy()
    x_ticks = np.arange(len(labels))
    plt.xticks(x_ticks[::step], labels[::step], rotation=45, ha="right")
    plt.yticks([0, 1, 1.2, 2.2], ["A no", "A yes", "B no", "B yes"])
    plt.title("1% VaR violation indicators")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "section5_nn_violation_1pct.png", dpi=200)
    plt.close()


def plot_pinball_summary(summary: pd.DataFrame) -> None:
    pivot = summary.pivot(index="alpha", columns="model_name", values="pinball_loss")
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Pinball loss by quantile level")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean pinball loss")
    ax.set_xticklabels([f"{idx:.0%}" for idx in pivot.index], rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "section5_nn_pinball_loss_comparison.png", dpi=200)
    plt.close()


def write_interpretation(summary: pd.DataFrame) -> None:
    def model_row(model_name: str, alpha: float) -> pd.Series:
        return summary[(summary["model_name"] == model_name) & (summary["alpha"] == alpha)].iloc[0]

    a1 = model_row("Model A: MLP-QR", 0.01)
    b1 = model_row("Model B: MLP-QR + RV", 0.01)
    mean_pinball = summary.groupby("model_name")["pinball_loss"].mean()
    better = mean_pinball.idxmin()
    rv_statement = (
        "The realized-volatility augmented specification delivers a lower average pinball loss in this run."
        if better == "Model B: MLP-QR + RV"
        else "The realized-volatility augmented specification does not deliver a lower average pinball loss in this run."
    )

    text = f"""# Section 5. Neural Quantile Regression for One-day-ahead SPY VaR

## 5.1 Motivation

This section evaluates a neural quantile regression approach for one-day-ahead
SPY Value-at-Risk forecasting. The objective is deliberately modest: the neural
network is used as a flexible conditional quantile function, not as evidence
that nonlinear machine learning should dominate the classical historical
simulation and GARCH benchmarks.

## 5.2 Feature construction

The target variable is the next-day log return, so the information set at day
t is mapped to log_ret at day t+1. Model A uses lagged returns and rolling
return summaries only. Model B augments the same information set with lagged
and rolling realized-volatility proxies, rv5 and bv. All rolling features are
computed from shifted series, which prevents the forecast for day t+1 from
using information beyond day t.

## 5.3 MLP quantile regression model

Both specifications use a feed-forward MLP with two hidden layers of 64 and 32
units, ReLU activations, and dropout in the first hidden block. The output layer
contains three units corresponding to the 1%, 5%, and 10% conditional quantiles.
The model is trained by the average pinball loss across the three quantile
levels. Predicted quantiles are sorted after inference so that
VaR_1% <= VaR_5% <= VaR_10%.

## 5.4 Rolling training design

The forecasting exercise uses a 1000-observation rolling estimation window.
For each refit, the first 80% of the current window is used for parameter
training and the final 20% is used for validation-based early stopping. The
standardization parameters are fitted only on the training portion of the
current rolling window. The network is retrained every 20 trading days, with
one-day-ahead forecasts generated between refits.

## 5.5 Empirical results summary

At the 1% VaR level, Model A records {int(a1['violations'])} violations from
{int(a1['n_forecasts'])} forecasts, compared with {int(b1['violations'])}
violations for Model B. The corresponding 1% failure rates are
{a1['failure_rate']:.4f} and {b1['failure_rate']:.4f}. The average pinball loss
across the three reported quantiles is {mean_pinball['Model A: MLP-QR']:.6f}
for Model A and {mean_pinball['Model B: MLP-QR + RV']:.6f} for Model B.
{rv_statement}

## 5.6 Interpretation

The results should be interpreted as an incremental robustness check rather
than a replacement for the classical VaR models. The realized-volatility inputs
rv5 and bv provide additional state variables for volatility clustering, but
their empirical value depends on whether they improve both calibration and
loss-based accuracy out of sample. A final judgment on whether the neural model
is superior to historical simulation or GARCH-type methods should be made in
Section 6 using the common backtesting and loss metrics across all models.
"""
    (RESULTS_DIR / "section5_nn_interpretation.md").write_text(text, encoding="utf-8")


def main() -> None:
    set_seed()
    ensure_directories()
    raw = load_spy_data()
    data = make_features(raw)
    if len(data) <= WINDOW_SIZE:
        raise ValueError(f"Need more than {WINDOW_SIZE} usable rows after feature engineering; found {len(data)}.")

    model_specs = {
        "Model A: MLP-QR": BASE_FEATURES,
        "Model B: MLP-QR + RV": BASE_FEATURES + RV_FEATURES,
    }

    all_predictions = []
    crossing_rates = {}
    for model_name, features in model_specs.items():
        print(f"Training {model_name} with {len(features)} features...")
        predictions, rate = rolling_train_predict(data, features, model_name)
        all_predictions.append(predictions)
        crossing_rates[model_name] = rate
        print(f"  crossing_rate={rate:.4f}, forecasts={len(predictions)}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = backtest_predictions(predictions, crossing_rates)

    predictions.to_csv(RESULTS_DIR / "section5_nn_var_predictions.csv", index=False)
    summary.to_csv(RESULTS_DIR / "section5_nn_backtesting_summary.csv", index=False)
    plot_var_comparison(predictions)
    plot_violation_indicators(predictions)
    plot_pinball_summary(summary)
    write_interpretation(summary)

    print(f"Saved predictions to {RESULTS_DIR / 'section5_nn_var_predictions.csv'}")
    print(f"Saved backtesting summary to {RESULTS_DIR / 'section5_nn_backtesting_summary.csv'}")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved interpretation to {RESULTS_DIR / 'section5_nn_interpretation.md'}")


if __name__ == "__main__":
    main()

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
VALIDATION_FRACTION = 0.10
CONSERVATIVE_SCALE = 1.0
WEIGHT_DECAY = 1e-5

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PROJECT_DIR = ROOT / "VaR_Project"

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
GARCH_FEATURES = ["base_VaR_0.01", "base_VaR_0.05", "base_VaR_0.10", "base_sigma"]


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
    for path in [ROOT / "spy_data.csv", PROJECT_DIR / "data" / "spy_data.csv", ROOT / "data" / "spy_data.csv"]:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find spy_data.csv in the current directory, data/, or VaR_Project/data/.")


def load_spy_data(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or find_data_path())
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
    return out.dropna(subset=BASE_FEATURES + RV_FEATURES + ["target", "target_date"]).copy()


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_forecast_file(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "date/index"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    rename = {
        date_col: "date",
        "realized_log_ret": "realized_return",
        "VaR_1": "VaR_0.01",
        "VaR_5": "VaR_0.05",
        "VaR_10": "VaR_0.10",
    }
    df = df.rename(columns=rename)
    keep = ["date", "realized_return", "VaR_0.01", "VaR_0.05", "VaR_0.10"]
    missing = [col for col in keep if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required forecast columns: {missing}")
    out = df[keep].copy()
    out["model_name"] = model_name
    return out


def load_garch_baseline() -> pd.DataFrame:
    forecast_path = first_existing(
        [
            PROJECT_DIR / "outputs" / "forecasts" / "var_garch_t_w1000.csv",
            PROJECT_DIR / "Section4_GARCH_t_Output" / "forecasts" / "var_garch_t_w1000.csv",
        ]
    )
    if forecast_path is None:
        raise FileNotFoundError(
            "No cached GARCH-t W=1000 forecast file found. Rerun Section 4 first or add a GARCH forecast file."
        )
    base = normalize_forecast_file(forecast_path, "GARCH(1,1)-t")
    base = base.rename(
        columns={
            "VaR_0.01": "base_VaR_0.01",
            "VaR_0.05": "base_VaR_0.05",
            "VaR_0.10": "base_VaR_0.10",
        }
    )
    params_path = first_existing(
        [
            PROJECT_DIR / "outputs" / "tables" / "section4_garch_params_w1000.csv",
            PROJECT_DIR / "Section4_GARCH_t_Output" / "tables" / "section4_garch_params_w1000.csv",
        ]
    )
    if params_path is not None:
        params = pd.read_csv(params_path)
        if "date" in params.columns and "sigma_next" in params.columns:
            params["date"] = pd.to_datetime(params["date"], errors="coerce")
            base = base.merge(params[["date", "sigma_next"]], on="date", how="left")
            base = base.rename(columns={"sigma_next": "base_sigma"})
    if "base_sigma" not in base.columns:
        base["base_sigma"] = np.nan
    base["base_model"] = "GARCH(1,1)-t"
    return base


def attach_baseline(feature_df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    out["forecast_date"] = pd.to_datetime(out["target_date"], errors="coerce")
    merged = out.merge(
        baseline[
            [
                "date",
                "base_model",
                "base_VaR_0.01",
                "base_VaR_0.05",
                "base_VaR_0.10",
                "base_sigma",
            ]
        ],
        left_on="forecast_date",
        right_on="date",
        how="inner",
    )
    merged = merged.drop(columns=["date"])
    if merged["base_sigma"].isna().all():
        # sigma is optional; use a neutral finite column if Section 4 params are unavailable.
        merged["base_sigma"] = 0.0
    return merged.dropna(subset=BASE_FEATURES + RV_FEATURES + GARCH_FEATURES + ["target"]).copy()


class CorrectionDataset(Dataset):
    def __init__(self, x: np.ndarray, base: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.base = torch.as_tensor(base, dtype=torch.float32)
        self.y = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.base[idx], self.y[idx]


class NeuralCorrectionNet(nn.Module):
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


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    alphas = torch.as_tensor(ALPHAS, dtype=pred.dtype, device=pred.device).view(1, -1)
    errors = target - pred
    return torch.maximum(alphas * errors, (alphas - 1.0) * errors).mean()


def final_var(base: torch.Tensor, raw: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "additive":
        return base + raw
    if variant == "conservative":
        return base - CONSERVATIVE_SCALE * torch.nn.functional.softplus(raw)
    raise ValueError(f"Unknown variant: {variant}")


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


def train_correction_model(
    x_window: np.ndarray,
    base_window: np.ndarray,
    y_window: np.ndarray,
    input_dim: int,
    variant: str,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[NeuralCorrectionNet, Standardizer]:
    n_train = int(len(x_window) * (1.0 - VALIDATION_FRACTION))
    x_train_raw, base_train, y_train = x_window[:n_train], base_window[:n_train], y_window[:n_train]
    x_val_raw, base_val, y_val = x_window[n_train:], base_window[n_train:], y_window[n_train:]

    scaler = Standardizer.fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)

    model = NeuralCorrectionNet(input_dim)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    elif variant == "conservative":
        # Start the conservative adjustment near a small positive correction
        # rather than an unrealistically large softplus(0) shift.
        final_layer = model.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, -5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(CorrectionDataset(x_train, base_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_x = torch.as_tensor(x_val, dtype=torch.float32)
    val_base = torch.as_tensor(base_val, dtype=torch.float32)
    val_y = torch.as_tensor(y_val.reshape(-1, 1), dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    stale = 0

    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, baseb, yb in loader:
            optimizer.zero_grad()
            pred = final_var(baseb, model(xb), variant)
            loss = pinball_loss(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss(final_var(val_base, model(val_x), variant), val_y))
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
    return float(((arr[:, 0] > arr[:, 1]) | (arr[:, 1] > arr[:, 2])).mean())


def rolling_train_predict(data: pd.DataFrame, feature_cols: list[str], variant: str, model_name: str) -> tuple[pd.DataFrame, float]:
    x_all = data[feature_cols].to_numpy(dtype=np.float32)
    base_all = data[["base_VaR_0.01", "base_VaR_0.05", "base_VaR_0.10"]].to_numpy(dtype=np.float32)
    y_all = data["target"].to_numpy(dtype=np.float32)
    target_dates = data["forecast_date"].to_numpy()

    rows = []
    raw_final_preds = []
    model = None
    scaler = None
    model_state = None

    for i in range(WINDOW_SIZE, len(data)):
        if model is None or (i - WINDOW_SIZE) % RETRAIN_EVERY == 0:
            x_window = x_all[i - WINDOW_SIZE : i]
            base_window = base_all[i - WINDOW_SIZE : i]
            y_window = y_all[i - WINDOW_SIZE : i]
            valid = np.isfinite(x_window).all(axis=1) & np.isfinite(base_window).all(axis=1) & np.isfinite(y_window)
            model, scaler = train_correction_model(
                x_window[valid],
                base_window[valid],
                y_window[valid],
                len(feature_cols),
                variant,
                model_state,
            )
            model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        x_pred = scaler.transform(x_all[i : i + 1])
        base_pred = base_all[i : i + 1].copy()
        with torch.no_grad():
            raw = model(torch.as_tensor(x_pred, dtype=torch.float32))
            final_raw = final_var(torch.as_tensor(base_pred, dtype=torch.float32), raw, variant).numpy().reshape(-1)
            raw_correction = (final_raw - base_pred.reshape(-1)).reshape(-1)

        sorted_final = np.sort(final_raw)
        raw_final_preds.append(final_raw)
        realized = float(y_all[i])

        rows.append(
            {
                "date/index": format_time_label(target_dates[i]),
                "realized_return": realized,
                "model_name": model_name,
                "base_model": str(data["base_model"].iloc[i]),
                "base_VaR_0.01": float(base_pred.reshape(-1)[0]),
                "base_VaR_0.05": float(base_pred.reshape(-1)[1]),
                "base_VaR_0.10": float(base_pred.reshape(-1)[2]),
                "VaR_0.01": float(sorted_final[0]),
                "VaR_0.05": float(sorted_final[1]),
                "VaR_0.10": float(sorted_final[2]),
                "correction_0.01": float(raw_correction[0]),
                "correction_0.05": float(raw_correction[1]),
                "correction_0.10": float(raw_correction[2]),
                "violation_0.01": int(realized < sorted_final[0]),
                "violation_0.05": int(realized < sorted_final[1]),
                "violation_0.10": int(realized < sorted_final[2]),
            }
        )

    return pd.DataFrame(rows), crossing_rate(raw_final_preds)


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
    return float(np.where(violations, 1.0 + (realized - var_forecast) ** 2, 0.0).mean())


def pinball_loss_np(realized: np.ndarray, var_forecast: np.ndarray, alpha: float) -> float:
    errors = realized - var_forecast
    return float(np.maximum(alpha * errors, (alpha - 1.0) * errors).mean())


def backtest_prediction_frame(predictions: pd.DataFrame, crossing_rates: dict[str, float]) -> pd.DataFrame:
    rows = []
    for model_name, group in predictions.groupby("model_name", sort=False):
        realized = group["realized_return"].to_numpy(dtype=float)
        base_model = group["base_model"].iloc[0] if "base_model" in group.columns else ""
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
                    "base_model": base_model,
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


def backtest_external_forecast(forecast: pd.DataFrame, model_name: str, base_model: str = "") -> pd.DataFrame:
    frame = forecast.copy()
    frame["model_name"] = model_name
    frame["base_model"] = base_model
    return backtest_prediction_frame(frame, {model_name: 0.0})


def load_existing_comparisons(model_c_summary: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    baseline_nn = RESULTS_DIR / "section5_nn_backtesting_summary.csv"
    if baseline_nn.exists():
        nn = pd.read_csv(baseline_nn)
        if "base_model" not in nn.columns:
            nn["base_model"] = ""
        pieces.append(nn)

    # Classical GARCH-family baselines from cached forecasts.
    for name, paths in {
        "GARCH(1,1)-t baseline": [
            PROJECT_DIR / "outputs" / "forecasts" / "var_garch_t_w1000.csv",
            PROJECT_DIR / "Section4_GARCH_t_Output" / "forecasts" / "var_garch_t_w1000.csv",
        ],
        "GJR-GARCH(1,1)-t baseline": [
            PROJECT_DIR / "outputs" / "forecasts" / "var_gjr_garch11_t_w1000.csv",
            PROJECT_DIR / "Section4_GARCH_t_Output" / "forecasts" / "var_gjr_garch11_t_w1000.csv",
        ],
    }.items():
        path = first_existing(paths)
        if path is not None:
            ext = normalize_forecast_file(path, name)
            pieces.append(backtest_external_forecast(ext, name))

    # HS-family compact tables use different names; include available rows with shared metrics.
    hs_path = PROJECT_DIR / "outputs" / "tables" / "hs_window_1000_report_table.csv"
    if hs_path.exists():
        hs = pd.read_csv(hs_path)
        model_col = "Model" if "Model" in hs.columns else "model" if "model" in hs.columns else None
        alpha_col = "Alpha" if "Alpha" in hs.columns else "alpha" if "alpha" in hs.columns else None
        if model_col and alpha_col:
            rows = []
            for _, row in hs.iterrows():
                alpha_value = row.get(alpha_col)
                alpha = float(str(alpha_value).strip("%")) / 100.0 if "%" in str(alpha_value) else float(alpha_value)
                rows.append(
                    {
                        "model_name": row.get(model_col),
                        "base_model": "",
                        "alpha": alpha,
                        "n_forecasts": np.nan,
                        "violations": np.nan,
                        "expected_violations": np.nan,
                        "failure_rate": row.get("Fail. rate", row.get("failure_rate", np.nan)),
                        "avg_var": row.get("Avg VaR", row.get("avg_var", np.nan)),
                        "kupiec_p": row.get("Kupiec p", row.get("kupiec_pvalue", np.nan)),
                        "christoffersen_p": row.get(
                            "Christoffersen p",
                            row.get("christoffersen_ind_pvalue", np.nan),
                        ),
                        "duration_p": row.get("Duration p", row.get("duration_pvalue", np.nan)),
                        "lopez_loss": row.get("Lopez loss", row.get("lopez_loss", np.nan)),
                        "pinball_loss": np.nan,
                        "crossing_rate": np.nan,
                    }
                )
            pieces.append(pd.DataFrame(rows))

    pieces.append(model_c_summary)
    cols = [
        "model_name",
        "base_model",
        "alpha",
        "n_forecasts",
        "violations",
        "expected_violations",
        "failure_rate",
        "avg_var",
        "kupiec_p",
        "christoffersen_p",
        "duration_p",
        "lopez_loss",
        "pinball_loss",
        "crossing_rate",
    ]
    return pd.concat([p.reindex(columns=cols) for p in pieces], ignore_index=True)


def plot_var_comparison(model_c_predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    dates = pd.to_datetime(model_c_predictions["date/index"].drop_duplicates())
    first_model = model_c_predictions.groupby("model_name", sort=False).get_group(model_c_predictions["model_name"].iloc[0])
    ax.plot(dates, first_model["realized_return"], color="0.65", linewidth=0.8, label="Realized log returns")
    ax.plot(dates, first_model["base_VaR_0.01"], linewidth=1.0, color="black", label="GARCH baseline 1% VaR")

    nn_path = RESULTS_DIR / "section5_nn_var_predictions.csv"
    if nn_path.exists():
        nn = pd.read_csv(nn_path)
        for model_name, color in [("Model A: MLP-QR", "tab:blue"), ("Model B: MLP-QR + RV", "tab:orange")]:
            sub = nn[nn["model_name"] == model_name].copy()
            if not sub.empty:
                ax.plot(pd.to_datetime(sub["date/index"]), sub["VaR_0.01"], linewidth=0.8, label=f"{model_name} 1% VaR", color=color, alpha=0.8)

    for model_name, color in [
        ("Model C1: GARCH-anchored MLP-QR", "tab:green"),
        ("Model C2: Conservative GARCH-anchored MLP-QR", "tab:red"),
    ]:
        sub = model_c_predictions[model_c_predictions["model_name"] == model_name].copy()
        if not sub.empty:
            ax.plot(pd.to_datetime(sub["date/index"]), sub["VaR_0.01"], linewidth=1.0, label=f"{model_name} 1% VaR", color=color)

    ax.set_title("1% VaR comparison: baseline MLP, GARCH anchor, and Model C")
    ax.set_ylabel("Log return")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "section5_model_c_1pct_var_comparison.png", dpi=200)
    plt.close(fig)


def plot_metric_comparisons(comparison: pd.DataFrame) -> None:
    focused = comparison[
        comparison["model_name"].isin(
            [
                "Model A: MLP-QR",
                "Model B: MLP-QR + RV",
                "Model C1: GARCH-anchored MLP-QR",
                "Model C2: Conservative GARCH-anchored MLP-QR",
                "GARCH(1,1)-t baseline",
                "GJR-GARCH(1,1)-t baseline",
            ]
        )
    ].copy()
    focused["alpha_label"] = focused["alpha"].map(lambda x: f"{x:.0%}")
    alpha_order = ["1%", "5%", "10%"]

    for metric, path, ylabel in [
        ("pinball_loss", FIGURES_DIR / "section5_model_c_pinball_loss_comparison.png", "Mean pinball loss"),
        ("failure_rate", FIGURES_DIR / "section5_model_c_failure_rate_comparison.png", "Failure rate"),
    ]:
        pivot = focused.pivot_table(index="alpha_label", columns="model_name", values=metric, aggfunc="first")
        pivot = pivot.reindex(alpha_order)
        ax = pivot.plot(kind="bar", figsize=(12, 5))
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Alpha")
        ax.set_ylabel(ylabel)
        if metric == "failure_rate":
            for ref in [0.01, 0.05, 0.10]:
                ax.axhline(ref, color="0.4", linewidth=0.7, linestyle="--")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()


def write_interpretation(summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    def row(model: str, alpha: float) -> pd.Series:
        return summary[(summary["model_name"] == model) & (summary["alpha"] == alpha)].iloc[0]

    c1_1 = row("Model C1: GARCH-anchored MLP-QR", 0.01)
    c2_1 = row("Model C2: Conservative GARCH-anchored MLP-QR", 0.01)

    baseline = comparison[comparison["model_name"].isin(["Model A: MLP-QR", "Model B: MLP-QR + RV"])]
    c_models = comparison[comparison["model_name"].isin(["Model C1: GARCH-anchored MLP-QR", "Model C2: Conservative GARCH-anchored MLP-QR"])]
    best_baseline_pinball = baseline.groupby("model_name")["pinball_loss"].mean().min() if not baseline.empty else np.nan
    best_c_pinball = c_models.groupby("model_name")["pinball_loss"].mean().min() if not c_models.empty else np.nan
    pinball_sentence = (
        "The anchored specification also improves the average pinball loss relative to the better baseline MLP model."
        if np.isfinite(best_baseline_pinball) and np.isfinite(best_c_pinball) and best_c_pinball < best_baseline_pinball
        else "The anchored specification does not uniformly improve average pinball loss relative to the better baseline MLP model."
    )
    fail_sentence = (
        "At the 1% level, both anchored variants materially reduce the severe underestimation observed in the baseline MLP-QR models."
        if c1_1["failure_rate"] < 0.0672 and c2_1["failure_rate"] < 0.0672
        else "At the 1% level, the anchored variants do not fully remove the severe underestimation observed in the baseline MLP-QR models."
    )

    text = f"""# Section 5 Model C: GARCH-Anchored Neural Quantile Correction

## 5.8 Motivation for GARCH-anchored Neural Correction

The baseline MLP-QR models severely underestimated SPY lower-tail risk. Their
1% failure rates were far above the nominal 1% target, which indicates that a
neural network trained directly on rolling-window samples may not learn a stable
extreme-tail VaR level. Section 4 showed the opposite strength and weakness for
GARCH-type models: GARCH-t improved the timing of violations by modeling
conditional volatility, but it still suffered from unconditional coverage
errors. Model C therefore uses GARCH-t VaR as a structured baseline and trains
a neural network to correct the VaR level.

## 5.9 Model Specification

The additive anchored specification is

VaR_NN_alpha,t+1 = VaR_BASE_alpha,t+1 + g_theta(X_t).

VaR_BASE is the GARCH-t VaR forecast, and g_theta(X_t) is the neural correction.
The information set X_t contains lagged returns, rolling return summaries, rv5,
bv, GARCH VaR forecasts, and the GARCH volatility forecast when available.

The conservative version is

VaR_NN_alpha,t+1 = VaR_BASE_alpha,t+1 - softplus(g_theta(X_t)).

This version only allows the neural correction to make the GARCH baseline more
conservative.

## 5.10 Rolling Training Design

The rolling design uses W = 1000 observations and one-day-ahead forecasts. Each
rolling window is split internally into 90% training and 10% validation. The
feature scaler is fitted only on the training part of the current rolling
window. The network is retrained every 20 trading days with Adam, learning rate
0.001, batch size 64, maximum 200 epochs, and early stopping patience 10.

## 5.11 Empirical Results

Model C1 records {int(c1_1['violations'])} 1% violations from
{int(c1_1['n_forecasts'])} forecasts, for a failure rate of
{c1_1['failure_rate']:.4f}. Model C2 records {int(c2_1['violations'])}
1% violations, for a failure rate of {c2_1['failure_rate']:.4f}.
{fail_sentence} {pinball_sentence}

The Kupiec, Christoffersen, and Duration p-values should still be interpreted
cautiously because the anchored correction is trained on a difficult left-tail
forecasting problem with limited extreme observations.

## 5.12 Interpretation

The results should not be interpreted as evidence that a neural network alone
dominates classical VaR models. The relevant question is whether neural
flexibility becomes more useful when it is tied to a financial-risk structure.
If Model C improves calibration relative to Model A and Model B, the evidence
supports the idea that GARCH anchoring helps neural quantile forecasting. If it
does not improve all diagnostics, the conclusion is narrower: even GARCH
anchoring is insufficient by itself, and stronger quantile dynamics or
post-training calibration may be needed.
"""
    (RESULTS_DIR / "section5_model_c_interpretation.md").write_text(text, encoding="utf-8")


def print_compact_summary(summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    compact = summary[["model_name", "alpha", "failure_rate", "kupiec_p", "christoffersen_p", "duration_p", "pinball_loss"]]
    print("\nModel C compact summary:")
    print(compact.to_string(index=False))

    baselines = comparison[comparison["model_name"].isin(["Model A: MLP-QR", "Model B: MLP-QR + RV"])]
    if not baselines.empty:
        base_1 = baselines[baselines["alpha"] == 0.01][["model_name", "failure_rate", "pinball_loss"]]
        c_1 = summary[summary["alpha"] == 0.01][["model_name", "failure_rate", "pinball_loss"]]
        print("\n1% comparison against Model A/B:")
        print(pd.concat([base_1, c_1], ignore_index=True).to_string(index=False))


def main() -> None:
    set_seed()
    ensure_directories()

    raw = load_spy_data()
    features = make_features(raw)
    baseline = load_garch_baseline()
    data = attach_baseline(features, baseline)
    if len(data) <= WINDOW_SIZE:
        raise ValueError(f"Need more than {WINDOW_SIZE} usable rows after GARCH alignment; found {len(data)}.")

    feature_cols = BASE_FEATURES + RV_FEATURES + GARCH_FEATURES
    specs = [
        ("additive", "Model C1: GARCH-anchored MLP-QR"),
        ("conservative", "Model C2: Conservative GARCH-anchored MLP-QR"),
    ]

    all_predictions = []
    crossing_rates = {}
    for variant, model_name in specs:
        print(f"Training {model_name} with {len(feature_cols)} features...")
        preds, rate = rolling_train_predict(data, feature_cols, variant, model_name)
        all_predictions.append(preds)
        crossing_rates[model_name] = rate
        print(f"  crossing_rate={rate:.4f}, forecasts={len(preds)}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    summary = backtest_prediction_frame(predictions, crossing_rates)
    comparison = load_existing_comparisons(summary)

    pred_path = RESULTS_DIR / "section5_model_c_garch_anchored_predictions.csv"
    summary_path = RESULTS_DIR / "section5_model_c_backtesting_summary.csv"
    comparison_path = RESULTS_DIR / "section5_model_c_comparison_with_existing.csv"
    predictions.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    plot_var_comparison(predictions)
    plot_metric_comparisons(comparison)
    write_interpretation(summary, comparison)
    print_compact_summary(summary, comparison)

    print(f"\nUsable forecasts per Model C variant: {len(predictions) // len(specs)}")
    print(f"Saved predictions to {pred_path}")
    print(f"Saved Model C backtesting summary to {summary_path}")
    print(f"Saved comparison table to {comparison_path}")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved interpretation to {RESULTS_DIR / 'section5_model_c_interpretation.md'}")


if __name__ == "__main__":
    main()

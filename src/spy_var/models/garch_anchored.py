"""C2 training implementation migrated from the original chapter script.

Data loading and evaluation are supplied by the canonical pipeline.
Inherited ex-post hyperparameters remain exploratory.
"""
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

TRAINING_HISTORY = []

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


def format_time_label(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed.strftime("%Y-%m-%d")
    return str(value)


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

    fit_id = len({row["fit_id"] for row in TRAINING_HISTORY})
    for _epoch in range(MAX_EPOCHS):
        total_loss, n_batches = 0.0, 0
        model.train()
        for xb, baseb, yb in loader:
            optimizer.zero_grad()
            pred = final_var(baseb, model(xb), variant)
            loss = pinball_loss(pred, yb)
            total_loss += float(loss.detach())
            n_batches += 1
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss(final_var(val_base, model(val_x), variant), val_y))
        TRAINING_HISTORY.append({"fit_id": fit_id, "epoch": _epoch + 1, "train_loss": total_loss / max(n_batches, 1), "validation_loss": val_loss})
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
            history_start = len(TRAINING_HISTORY)
            model, scaler = train_correction_model(
                x_window[valid],
                base_window[valid],
                y_window[valid],
                len(feature_cols),
                variant,
                model_state,
            )
            for record in TRAINING_HISTORY[history_start:]:
                record["refit_forecast_date"] = format_time_label(target_dates[i])
                record["window_start_date"] = format_time_label(target_dates[i - WINDOW_SIZE])
                record["window_end_date"] = format_time_label(target_dates[i - 1])
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



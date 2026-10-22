from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import LEVELS as VAR_LEVELS, VAR_COLUMNS as VAR_COLUMN_MAP
from ..data import FEATURES as REQUIRED_COLUMNS

WINDOW_SIZE = 1000
TRAINING_HISTORY = []


SEED = 42


def set_seed(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class MLPConfig:
    lag: int = 20
    window_size: int = WINDOW_SIZE
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    validation_fraction: float = 0.2
    patience: int = 5
    step_refit: int = 20


def build_supervised_window(values: np.ndarray, start: int, end: int, lag: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    for target_pos in range(start + lag, end):
        x_rows.append(values[target_pos - lag : target_pos, :].reshape(-1))
        y_rows.append(values[target_pos, 0])
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32).reshape(-1, 1)


def pinball_loss(pred, target, quantiles):
    import torch

    pred = torch.sort(pred, dim=1)[0]
    errors = target - pred
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).view(1, -1)
    return torch.maximum(q * errors, (q - 1.0) * errors).mean()


def train_model(x_train: np.ndarray, y_train: np.ndarray, config: MLPConfig):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    n_features = x_train.shape[1]
    n_val = max(1, int(len(x_train) * config.validation_fraction))
    n_fit = max(1, len(x_train) - n_val)
    x_fit, y_fit = x_train[:n_fit], y_train[:n_fit]
    x_val, y_val = x_train[n_fit:], y_train[n_fit:]

    mean = x_fit.mean(axis=0, keepdims=True)
    std = x_fit.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    x_fit_s = (x_fit - mean) / std
    x_val_s = (x_val - mean) / std

    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, len(VAR_LEVELS)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit_s), torch.from_numpy(y_fit)),
        batch_size=config.batch_size,
        shuffle=True,
    )

    best_state = None
    best_val = float("inf")
    stale = 0
    fit_id = len({row["fit_id"] for row in TRAINING_HISTORY})
    for _epoch in range(config.epochs):
        total_loss, n_batches = 0.0, 0
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = pinball_loss(model(xb), yb, VAR_LEVELS)
            total_loss += float(loss.detach())
            n_batches += 1
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(pinball_loss(model(torch.from_numpy(x_val_s)), torch.from_numpy(y_val), VAR_LEVELS))
        TRAINING_HISTORY.append({"fit_id": fit_id, "epoch": _epoch + 1, "train_loss": total_loss / max(n_batches, 1), "validation_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mean.astype(np.float32), std.astype(np.float32)


def neural_quantile_var(df: pd.DataFrame, config: MLPConfig = MLPConfig()) -> pd.DataFrame:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required. Install dependencies with: pip install -r requirements.txt") from exc

    set_seed(SEED)
    values = df[REQUIRED_COLUMNS].to_numpy(dtype=np.float32)
    returns = df["log_ret"]
    rows = []
    model = None
    mean = None
    std = None

    for forecast_pos in range(config.window_size, len(df)):
        train_start = forecast_pos - config.window_size
        train_end = forecast_pos
        need_refit = model is None or (forecast_pos - config.window_size) % config.step_refit == 0

        if need_refit:
            x_train, y_train = build_supervised_window(values, train_start, train_end, config.lag)
            valid_mask = np.isfinite(x_train).all(axis=1) & np.isfinite(y_train).all(axis=1)
            x_train, y_train = x_train[valid_mask], y_train[valid_mask]
            if len(x_train) < 10:
                model = None
            else:
                history_start = len(TRAINING_HISTORY)
                model, mean, std = train_model(x_train, y_train, config)
                for record in TRAINING_HISTORY[history_start:]:
                    record["refit_forecast_date"] = str(df.index[forecast_pos].date())
                    record["window_start_date"] = str(df.index[train_start].date())
                    record["window_end_date"] = str(df.index[train_end - 1].date())

        row = {
            "date": df.index[forecast_pos].strftime("%Y-%m-%d") if hasattr(df.index[forecast_pos], "strftime") else forecast_pos,
            "realized_log_ret": returns.iloc[forecast_pos],
        }
        if model is None or forecast_pos < config.lag:
            for alpha in VAR_LEVELS:
                row[VAR_COLUMN_MAP[alpha]] = np.nan
        else:
            x_pred = values[forecast_pos - config.lag : forecast_pos, :].reshape(1, -1).astype(np.float32)
            if not np.isfinite(x_pred).all():
                preds = [np.nan] * len(VAR_LEVELS)
            else:
                x_pred = (x_pred - mean) / std
                model.eval()
                with torch.no_grad():
                    preds = model(torch.from_numpy(x_pred)).numpy().reshape(-1)
                preds = np.sort(preds)
            for alpha, pred in zip(VAR_LEVELS, preds):
                row[VAR_COLUMN_MAP[alpha]] = float(pred)
        rows.append(row)

    return pd.DataFrame(rows)


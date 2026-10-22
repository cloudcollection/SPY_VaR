"""A small, from-scratch PatchTST-inspired quantile adaptation.

Nie et al. (ICLR 2023): https://arxiv.org/abs/2211.14730
Changes: univariate one-step quantiles, pinball loss, LayerNorm encoder,
no end padding, and reversible per-window normalization without affine terms.
This is an exploratory financial adaptation, not the official paper model.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import LEVELS, VAR_COLUMNS


class PatchQuantileTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_patches = 1 + (config.lookback - config.patch_length) // config.stride
        self.projection = nn.Linear(config.patch_length, config.d_model)
        self.position = nn.Parameter(torch.empty(1, self.n_patches, config.d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.heads, dim_feedforward=config.feedforward,
            dropout=config.dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.head = nn.Linear(self.n_patches * config.d_model, len(LEVELS))

    def forward(self, history):
        # All moments are computed from the observed input window only.
        mean = history.mean(dim=1, keepdim=True).detach()
        scale = (history.var(dim=1, keepdim=True, unbiased=False) + self.config.norm_epsilon).sqrt().detach()
        normalized = (history - mean) / scale
        patches = normalized.unfold(1, self.config.patch_length, self.config.stride)
        encoded = self.encoder(self.projection(patches) + self.position)
        raw = self.head(encoded.flatten(start_dim=1))
        return torch.sort(mean + scale * raw, dim=1).values


def pinball(prediction, target):
    alpha = prediction.new_tensor(LEVELS).view(1, -1)
    errors = target.reshape(-1, 1) - prediction
    return torch.maximum(alpha * errors, (alpha - 1) * errors).mean()


def supervised_window(returns, start, end, lookback):
    targets = np.arange(start + lookback, end)
    if not len(targets):
        raise ValueError("No supervised samples in the specified window")
    x = np.stack([returns[t - lookback:t] for t in targets]).astype(np.float32)
    y = np.asarray(returns[targets], dtype=np.float32)
    return x, y, targets


def fit_window(x, y, config):
    split = int(len(x) * (1 - config.validation_fraction))
    if split < 2 or len(x) - split < 1:
        raise ValueError("Need nonempty chronological training and validation sets")
    model = PatchQuantileTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:split]), torch.from_numpy(y[:split])),
                        batch_size=config.batch_size, shuffle=True)
    val_x, val_y = torch.from_numpy(x[split:]), torch.from_numpy(y[split:])
    best_state, best_loss, stale = None, float("inf"), 0
    history = []
    for epoch in range(config.epochs):
        model.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = pinball(model(xb), yb)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite Transformer training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        model.eval()
        with torch.no_grad():
            validation = float(pinball(model(val_x), val_y))
        if not np.isfinite(validation):
            raise ValueError("Nonfinite Transformer validation loss")
        history.append({"epoch": epoch + 1, "train_loss": total / count, "validation_loss": validation})
        if validation < best_loss:
            best_loss, stale, best_state = validation, 0, deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, split


def rolling_forecast(data, experiment):
    from .mlp import set_seed

    set_seed(experiment.seed)
    config = experiment.patch
    returns = data.log_ret.to_numpy(dtype=np.float32)
    rows, logs, fit_id = [], [], 0
    model = None
    for pos in range(experiment.window_size, len(data)):
        if model is None or (pos - experiment.window_size) % config.refit_every == 0:
            x, y, targets = supervised_window(returns, pos - experiment.window_size, pos, config.lookback)
            model, history, split = fit_window(x, y, config)
            for record in history:
                record.update({
                    "fit_id": fit_id, "refit_forecast_date": str(data.index[pos].date()),
                    "train_target_start": str(data.index[targets[0]].date()),
                    "train_target_end": str(data.index[targets[split - 1]].date()),
                    "validation_target_start": str(data.index[targets[split]].date()),
                    "validation_target_end": str(data.index[targets[-1]].date()),
                })
            logs.extend(history)
            fit_id += 1
        with torch.no_grad():
            quantiles = model(torch.from_numpy(returns[pos - config.lookback:pos].copy()).view(1, -1)).numpy()[0]
        rows.append({"date": data.index[pos], "realized_log_ret": float(data.log_ret.iloc[pos]),
                     **{VAR_COLUMNS[a]: float(q) for a, q in zip(LEVELS, quantiles)}})
    return pd.DataFrame(rows), pd.DataFrame(logs)

"""Explicit, serializable experiment settings (no implicit chapter defaults)."""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

MODEL_NAMES = {
    "hs": "Ordinary Historical Simulation",
    "weighted_hs": "Time-weighted Historical Simulation",
    "kde_hs": "Gaussian KDE Time-weighted HS",
    "garch_t": "GARCH(1,1)-t",
    "gjr_garch_t": "GJR-GARCH(1,1)-t",
    "mlp": "MLP Quantile (lag-window specification)",
    "c2": "C2 Conservative GARCH-anchored Neural Correction",
}
LEVELS = (0.01, 0.05, 0.10)
VAR_COLUMNS = {0.01: "VaR_1", 0.05: "VaR_5", 0.10: "VaR_10"}


@dataclass
class NeuralConfig:
    lag: int = 20
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 0.001
    validation_fraction: float = 0.2
    patience: int = 5
    refit_every: int = 20


@dataclass
class C2Config:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    validation_fraction: float = 0.1
    patience: int = 10
    refit_every: int = 20
    conservative_scale: float = 1.0
    weight_decay: float = 0.00001


@dataclass
class ExperimentConfig:
    window_size: int = 1000
    seed: int = 42
    decay: float = 0.98
    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    mlp: NeuralConfig = field(default_factory=NeuralConfig)
    c2: C2Config = field(default_factory=C2Config)

    def validate(self):
        if not isinstance(self.window_size, int) or self.window_size < 50:
            raise ValueError("window_size must be an integer >= 50")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not 0 < self.decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if not self.models or len(set(self.models)) != len(self.models) or set(self.models) - MODEL_NAMES.keys():
            raise ValueError(f"models must be a nonempty unique subset of {list(MODEL_NAMES)}")
        for config in (self.mlp, self.c2):
            for key in ("epochs", "patience", "refit_every", "batch_size"):
                if not isinstance(getattr(config, key), int) or getattr(config, key) < 1:
                    raise ValueError(f"{key} must be a positive integer")
            if not 0 < config.validation_fraction < 1:
                raise ValueError("validation_fraction must be in (0, 1)")
            if not config.learning_rate > 0:
                raise ValueError("learning_rate must be positive")
        if self.mlp.lag < 1 or self.mlp.lag >= self.window_size - 10:
            raise ValueError("mlp.lag must leave at least 10 supervised observations")
        if self.mlp.batch_size < 1 or self.mlp.learning_rate <= 0:
            raise ValueError("MLP batch size and learning rate must be positive")
        if self.c2.conservative_scale <= 0 or self.c2.weight_decay < 0:
            raise ValueError("C2 scale must be positive and weight decay nonnegative")
        return self

    def to_dict(self):
        return asdict(self)


def _strict(cls, values):
    unknown = set(values) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def load_config(path: Path) -> ExperimentConfig:
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    values["mlp"] = _strict(NeuralConfig, values.get("mlp", {}))
    values["c2"] = _strict(C2Config, values.get("c2", {}))
    return _strict(ExperimentConfig, values).validate()

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_C_PATH = ROOT / "05c_garch_anchored_neural_var.py"
RESULTS_DIR = ROOT / "results"


def load_model_c_module():
    spec = importlib.util.spec_from_file_location("model_c", MODEL_C_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {MODEL_C_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_scale(module, scale: float) -> pd.DataFrame:
    module.CONSERVATIVE_SCALE = scale
    module.WEIGHT_DECAY = 0.0
    module.set_seed(module.SEED)
    raw = module.load_spy_data()
    features = module.make_features(raw)
    baseline = module.load_garch_baseline()
    data = module.attach_baseline(features, baseline)
    feature_cols = module.BASE_FEATURES + module.RV_FEATURES + module.GARCH_FEATURES
    model_name = f"Model C2 scale={scale:g}"
    print(f"Running {model_name}...")
    preds, crossing = module.rolling_train_predict(data, feature_cols, "conservative", model_name)
    summary = module.backtest_prediction_frame(preds, {model_name: crossing})
    summary["conservative_scale"] = scale
    return summary


def run_weight_decay(module, weight_decay: float, scale: float = 1.0) -> pd.DataFrame:
    module.CONSERVATIVE_SCALE = scale
    module.WEIGHT_DECAY = weight_decay
    module.set_seed(module.SEED)
    raw = module.load_spy_data()
    features = module.make_features(raw)
    baseline = module.load_garch_baseline()
    data = module.attach_baseline(features, baseline)
    feature_cols = module.BASE_FEATURES + module.RV_FEATURES + module.GARCH_FEATURES
    model_name = f"Model C2 wd={weight_decay:g}"
    print(f"Running {model_name}...")
    preds, crossing = module.rolling_train_predict(data, feature_cols, "conservative", model_name)
    summary = module.backtest_prediction_frame(preds, {model_name: crossing})
    summary["conservative_scale"] = scale
    summary["weight_decay"] = weight_decay
    return summary


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    module = load_model_c_module()
    path = RESULTS_DIR / "section5_model_c_conservative_scale_tuning.csv"
    if path.exists():
        print(f"Using existing scale tuning summary from {path}")
        out = pd.read_csv(path)
    else:
        scales = [0.75, 1.0, 1.25, 1.5]
        summaries = []
        for scale in scales:
            summaries.append(run_scale(module, scale))
        out = pd.concat(summaries, ignore_index=True)
        out.to_csv(path, index=False)
    compact = out[["conservative_scale", "alpha", "failure_rate", "kupiec_p", "christoffersen_p", "duration_p", "pinball_loss", "crossing_rate"]]
    print(compact.to_string(index=False))
    print(f"Saved tuning summary to {path}")

    wd_summaries = []
    for weight_decay in [0.0, 1e-5, 1e-4, 5e-4]:
        wd_summaries.append(run_weight_decay(module, weight_decay))
    wd_out = pd.concat(wd_summaries, ignore_index=True)
    wd_path = RESULTS_DIR / "section5_model_c_weight_decay_tuning.csv"
    wd_out.to_csv(wd_path, index=False)
    wd_compact = wd_out[["weight_decay", "alpha", "failure_rate", "kupiec_p", "christoffersen_p", "duration_p", "pinball_loss", "crossing_rate"]]
    print(wd_compact.to_string(index=False))
    print(f"Saved weight-decay tuning summary to {wd_path}")


if __name__ == "__main__":
    main()

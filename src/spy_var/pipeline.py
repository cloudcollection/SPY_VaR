"""Run experiments and preserve their inputs, configuration and diagnostics."""
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import pandas as pd

from .backtesting import evaluate
from .config import MODEL_NAMES
from .data import load_data
from .plotting import plot_failure_rates


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance(root):
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    versions = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "arch", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions,
        "source_sha256": {str(path.relative_to(root)): fingerprint(path) for path in sorted((root / "src/spy_var").glob("*.py"))},
    }


def save_evaluation(frames, output, calendar):
    table, dates = evaluate(frames, calendar=calendar)
    (output / "tables").mkdir(exist_ok=True)
    table.to_csv(output / "tables/aligned_model_comparison.csv", index=False)
    dates.to_csv(output / "tables/evaluation_dates.csv", index=False)
    plot_failure_rates(table, output / "figures/aligned_failure_rates.png")
    return table


def run_experiment(root, data_path, config, output, max_observations=None):
    from . import anchored, neural
    from .classical import garch_forecast, historical_forecast

    config.validate()
    data = load_data(data_path)
    if max_observations is not None:
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        data = data.iloc[:max_observations]
    minimum = 2 * config.window_size + 1 if "c2" in config.models else config.window_size + 1
    if len(data) < minimum:
        raise ValueError(f"Need at least {minimum} rows for the selected models; found {len(data)}")
    output = prepare_output(output)
    manifest = provenance(root)
    manifest.update({
        "mode": "train", "status": "running", "config": config.to_dict(),
        "data": {"path": str(data_path.resolve()), "sha256": fingerprint(data_path),
                 "n_rows_used": len(data), "start": str(data.index.min().date()), "end": str(data.index.max().date())},
        "max_observations": max_observations, "models": {},
        "interpretation": "exploratory; inherited C2 weight decay was selected after inspecting historical test results",
        "report_replication": "New configuration-driven experiment; not an exact reconstruction of every archived report table.",
    })
    write_json(output / "config.json", config.to_dict())
    write_json(output / "manifest.json", manifest)
    start = time.monotonic()
    try:
        (output / "forecasts").mkdir()
        (output / "training").mkdir()
        generated, frames = {}, {}
        # C2 requires one-step GARCH forecasts as inputs, even if not scored separately.
        requested = list(config.models)
        order = (["garch_t"] if "c2" in requested and "garch_t" not in requested else []) + requested
        if "c2" in order and order.index("c2") < order.index("garch_t"):
            order.remove("garch_t")
            order.insert(0, "garch_t")
        for model in order:
            print(f"Training {MODEL_NAMES[model]} (seed={config.seed}, W={config.window_size})", flush=True)
            model_start = time.monotonic()
            if model in {"hs", "weighted_hs", "kde_hs"}:
                prediction = historical_forecast(data, config, model)
            elif model in {"garch_t", "gjr_garch_t"}:
                prediction = garch_forecast(data, config, asymmetric=model == "gjr_garch_t")
            elif model == "mlp":
                neural.SEED = config.seed
                neural.TRAINING_HISTORY.clear()
                settings = neural.MLPConfig(
                    **{k: v for k, v in vars(config.mlp).items() if k != "refit_every"},
                    window_size=config.window_size, step_refit=config.mlp.refit_every,
                )
                prediction = neural.neural_quantile_var(data, settings)
                pd.DataFrame(neural.TRAINING_HISTORY).to_csv(output / "training/mlp.csv", index=False)
            elif model == "c2":
                anchored.WINDOW_SIZE = config.window_size
                anchored.MAX_EPOCHS = config.c2.epochs
                anchored.BATCH_SIZE = config.c2.batch_size
                anchored.LEARNING_RATE = config.c2.learning_rate
                anchored.VALIDATION_FRACTION = config.c2.validation_fraction
                anchored.PATIENCE = config.c2.patience
                anchored.RETRAIN_EVERY = config.c2.refit_every
                anchored.CONSERVATIVE_SCALE = config.c2.conservative_scale
                anchored.WEIGHT_DECAY = config.c2.weight_decay
                anchored.TRAINING_HISTORY.clear()
                anchored.set_seed(config.seed)
                baseline = generated["garch_t"].rename(columns={
                    "VaR_1": "base_VaR_0.01", "VaR_5": "base_VaR_0.05", "VaR_10": "base_VaR_0.10", "sigma_next": "base_sigma",
                }).copy()
                baseline["base_model"] = MODEL_NAMES["garch_t"]
                prepared = anchored.attach_baseline(anchored.make_features(data), baseline)
                if len(prepared) <= config.window_size:
                    raise ValueError("Not enough valid GARCH forecasts to train C2")
                prediction, crossing = anchored.rolling_train_predict(
                    prepared, anchored.BASE_FEATURES + anchored.RV_FEATURES + anchored.GARCH_FEATURES,
                    "conservative", MODEL_NAMES["c2"],
                )
                manifest["c2_raw_crossing_rate"] = crossing
                pd.DataFrame(anchored.TRAINING_HISTORY).to_csv(output / "training/c2.csv", index=False)
            if prediction.empty:
                raise ValueError(f"No predictions generated for {model}")
            prediction.to_csv(output / "forecasts" / f"{model}.csv", index=False)
            generated[model] = prediction
            if model in requested:
                frames[MODEL_NAMES[model]] = prediction
            manifest["models"][model] = {
                "n_rows": len(prediction), "seconds": time.monotonic() - model_start,
                "forecast_sha256": fingerprint(output / "forecasts" / f"{model}.csv"),
                "n_fit_errors": int(prediction.fit_error.ne("").sum()) if "fit_error" in prediction else 0,
            }
            write_json(output / "manifest.json", manifest)
        table = save_evaluation(frames, output, data.index)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start
        write_json(output / "manifest.json", manifest)
    print(f"Saved experiment to {output}", flush=True)
    return table


def evaluate_cached(root, output, include_c2=False):
    cached = root / "code/outputs/forecasts"
    files = {
        "weighted_hs": cached / "var_historical_simulation.csv",
        "kde_hs": cached / "var_kde_weighted_hs.csv",
        "garch_t": cached / "var_garch_t.csv",
        "gjr_garch_t": cached / "var_gjr_garch_t.csv",
        "mlp": cached / "var_mlp_quantile.csv",
    }
    frames = {MODEL_NAMES[key]: pd.read_csv(path) for key, path in files.items()}
    if include_c2:
        path = root / "code/sections/section_05_mlp_quantile/outputs/external_section5_results/section5_model_c_garch_anchored_predictions.csv"
        prediction = pd.read_csv(path)
        frames[MODEL_NAMES["c2"]] = prediction.loc[prediction.model_name.eq("Model C2: Conservative GARCH-anchored MLP-QR")].copy()
        files["c2"] = path
    data_path = root / "code/data/spy_data.csv"
    calendar = load_data(data_path).index
    output = prepare_output(output)
    manifest = provenance(root)
    manifest.update({
        "mode": "evaluate_cached", "status": "running", "include_c2": include_c2,
        "training_performed": False,
        "historical_training_config": "unknown; archived files predate per-run manifests",
        "inputs": {str(path.relative_to(root)): fingerprint(path) for path in [data_path, *files.values()]},
        "interpretation": "Historical exploratory predictions; adding C2 changes the common evaluation window.",
    })
    try:
        table = save_evaluation(frames, output, calendar)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    print(f"Evaluated cached forecasts in {output}; no training performed.")
    return table


def run_seeds(root, data_path, config, output, seeds, max_observations=None):
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Seeds must be nonempty and unique")
    for seed in seeds:
        replace(config, seed=seed).validate()
    output = prepare_output(output)
    tables = []
    for seed in seeds:
        table = run_experiment(root, data_path, replace(config, seed=seed), output / f"seed-{seed}", max_observations)
        tables.append(table.assign(seed=seed))
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / "all_seeds.csv", index=False)
    # Compare seeds only if every model/seed was evaluated on the same dates.
    date_sets = [pd.read_csv(output / f"seed-{seed}/tables/evaluation_dates.csv") for seed in seeds]
    if any(not frame.equals(date_sets[0]) for frame in date_sets[1:]):
        raise ValueError("Evaluation dates differ across seeds; inspect per-run dates before pooling")
    combined.groupby(["model", "alpha"])[["failure_rate", "pinball_loss"]].agg(["mean", "std", "count"]).to_csv(output / "seed_summary.csv")

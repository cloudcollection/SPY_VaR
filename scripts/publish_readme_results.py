"""Publish only verified final evidence and update README tables from real CSVs.

Detailed runs stay outside the repository. This command checks their predictions,
input hash and model source hash before retaining compact final result tables.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spy_var.artifacts import load_archived_forecasts
from spy_var.backtesting import evaluate
from spy_var.config import MODEL_NAMES
from spy_var.data import load_data

LABELS = {
    MODEL_NAMES["weighted_hs"]: "Time-weighted HS", MODEL_NAMES["kde_hs"]: "KDE weighted HS",
    MODEL_NAMES["garch_t"]: "GARCH-t", MODEL_NAMES["gjr_garch_t"]: "GJR-GARCH-t",
    MODEL_NAMES["mlp"]: "Lag-window MLP", MODEL_NAMES["c2"]: "C2 correction",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def historical_markdown(table):
    lines = ["| Model | Violations / 2,640 | Rate | Kupiec p | Independence p | Pinball loss |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in table.loc[table.alpha == 0.01].iterrows():
        lines.append(f"| {LABELS[row.model]} | {int(row.n_violations)} | {100 * row.failure_rate:.2f}% | "
                     f"{row.kupiec_pvalue:.3g} | {row.christoffersen_ind_pvalue:.3g} | {row.pinball_loss:.6f} |")
    return "\n".join(lines)


def modern_markdown(summary):
    lines = ["| Nominal quantile | Observed rate, mean ± SD | Pinball loss, mean ± SD | Dates per seed |",
             "| --- | ---: | ---: | ---: |"]
    for _, row in summary.iterrows():
        lines.append(f"| {row.alpha:.0%} | {100 * row.failure_rate_mean:.2f}% ± {100 * row.failure_rate_std:.2f} pp | "
                     f"{row.pinball_loss_mean:.6f} ± {row.pinball_loss_std:.6f} | {int(row.n_obs)} |")
    return "\n".join(lines)


def replace_block(content, tag, value):
    pattern = rf"(<!-- BEGIN {tag} -->).*?(<!-- END {tag} -->)"
    result, count = re.subn(pattern, lambda m: m.group(1) + "\n" + value + "\n" + m.group(2), content, flags=re.S)
    if count != 1:
        raise ValueError(f"README needs exactly one {tag} result block")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    data_path = ROOT / "code/data/spy_data.csv"
    data = load_data(data_path)
    frames, files = load_archived_forecasts(ROOT, include_c2=True)
    historical, _ = evaluate(frames, calendar=data.index)
    results = ROOT / "docs/results"
    results.mkdir(parents=True, exist_ok=True)
    historical.to_csv(results / "historical_comparison.csv", index=False)
    tables, run_records, date_frames = [], [], []
    expected_model_hash = digest(ROOT / "src/spy_var/models/patch_transformer.py")
    for folder in sorted(args.runs.glob("seed-*")):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        seed = manifest["config"]["seed"]
        if manifest["status"] != "complete" or manifest["config"]["models"] != ["patchtst_qr"]:
            raise ValueError("Only completed, single-model patch experiments can be published here")
        if manifest["data"]["sha256"] != digest(data_path) or manifest["max_observations"] is not None:
            raise ValueError("Run input differs from the complete repository snapshot")
        sources = {key.replace("\\", "/"): value for key, value in manifest["source_sha256"].items()}
        if sources["src/spy_var/models/patch_transformer.py"] != expected_model_hash:
            raise ValueError("The trained model source differs from the published model source")
        pred_path = folder / "forecasts/patchtst_qr.csv"
        if digest(pred_path) != manifest["models"]["patchtst_qr"]["forecast_sha256"]:
            raise ValueError("Predictions changed after the run was recorded")
        predictions = pd.read_csv(pred_path)
        dates = pd.to_datetime(predictions.date)
        if not np.allclose(predictions.realized_log_ret, data.loc[dates, "log_ret"], rtol=1e-10, atol=1e-12):
            raise ValueError("Forecast actual returns do not match the real input data")
        table, evaluation_dates = evaluate({MODEL_NAMES["patchtst_qr"]: predictions}, calendar=data.index)
        table["seed"] = seed
        tables.append(table)
        date_frames.append(evaluation_dates)
        logs = pd.read_csv(folder / "training/patchtst_qr.csv")
        if not np.isfinite(logs[["train_loss", "validation_loss"]]).all().all():
            raise ValueError("Nonfinite training or validation loss")
        if not (pd.to_datetime(logs.train_target_end) < pd.to_datetime(logs.validation_target_start)).all():
            raise ValueError("Training and validation targets overlap")
        if not (pd.to_datetime(logs.validation_target_end) < pd.to_datetime(logs.refit_forecast_date)).all():
            raise ValueError("A refit used its future target")
        run_records.append({"seed": seed, "seconds": manifest["elapsed_seconds"],
                            "prediction_sha256": digest(pred_path), "refits": int(logs.fit_id.nunique())})
    if len(tables) != 3 or {record["seed"] for record in run_records} != {42, 43, 44}:
        raise ValueError("Expected the three prespecified distinct seeds")
    if any(not frame.equals(date_frames[0]) for frame in date_frames[1:]):
        raise ValueError("Seeds were evaluated on different target dates")
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(results / "patch_transformer_by_seed.csv", index=False)
    summary = combined.groupby("alpha", as_index=False).agg(
        failure_rate_mean=("failure_rate", "mean"), failure_rate_std=("failure_rate", "std"),
        pinball_loss_mean=("pinball_loss", "mean"), pinball_loss_std=("pinball_loss", "std"),
        n_obs=("n_obs", "first"), n_seeds=("seed", "nunique"))
    summary.to_csv(results / "patch_transformer_summary.csv", index=False)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = replace_block(readme, "HISTORICAL_RESULTS", historical_markdown(historical))
    readme = replace_block(readme, "MODERN_RESULTS", modern_markdown(summary))
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    records = "\n".join(f"| {r['seed']} | {r['refits']} | {r['seconds']:.2f} | `{r['prediction_sha256']}` |" for r in run_records)
    sources = "\n".join(f"- `{path.relative_to(ROOT).as_posix()}`: `{digest(path)}`" for path in [data_path, *files.values()])
    data_lf_hash = sha256(data_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    report = f"""# Experiment evidence — 2026-10-22

## Retained final evidence

`results/historical_comparison.csv` recalculates all three quantiles from the
archived predictions on their common C2-aligned dates. It is a descriptive,
retrospective comparison; historical training settings are not reconstructed.

`results/patch_transformer_by_seed.csv` is recomputed from each new run's actual
predictions. `results/patch_transformer_summary.csv` reports means and sample
standard deviations across seeds 42, 43 and 44. The configurations were fixed
before these runs; no test-driven tuning was performed after seeing the results.
Detailed run directories, layouts, crops and QA reports are not committed.

## New model run verification

All three complete-snapshot rolling runs used W=1000 and the fixed configuration
in `../configs/modern_quantile.json`. They use CPU and the locked numerical stack.
There are 3,640 forecasts per seed, 182 refits per seed and three quantiles.
The model has 18,531 trainable parameters. Training targets precede validation
targets, and validation targets strictly precede the refit's forecast target.
The publication helper checks the data hash, model hash, forecast hash and
realized-return agreement before writing the final tables.

| Seed | Refits | CPU run wall time (seconds) | Prediction SHA-256 |
| --- | ---: | ---: | --- |
{records}

Trained model source SHA-256: `{expected_model_hash}`.

The independent seed runs reuse the same historical market observations; three
seeds do not triple the effective empirical sample size. Their SD describes
optimization variability, not a confidence interval. Per-seed p-values are
retained and never averaged. No outperformance claim follows from these runs.

## Archived input fingerprints

The raw-byte fingerprints below record the actual Windows inputs. Git's text
checkout conversion can change CRLF/LF bytes without changing any numeric value.
The included data snapshot's LF-normalized SHA-256 is `{data_lf_hash}`.
The model source is kept at LF line endings by `.gitattributes`.

{sources}

## Reproduce

Run new experiments outside the checkout, then publish compact final evidence:

```sh
python scripts/reproduce.py train --config configs/modern_quantile.json --seeds 42 43 44 --output ../spy-var-runs/patch-seeds
python scripts/publish_readme_results.py --runs ../spy-var-runs/patch-seeds
```

See `METHODS.md` for model changes and timing, `REFERENCES.md` for original papers
and the high-star README selection, and `VALIDATION.md` for executed checks.
"""
    (ROOT / "docs/EXPERIMENTS.md").write_text(report, encoding="utf-8")
    print("Published verified final CSV evidence and README tables.")


if __name__ == "__main__":
    main()

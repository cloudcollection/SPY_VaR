# Reproducibility and experiment identity

## Two distinct workflows

`evaluate` reads stored predictions and recomputes metrics/figures. It does not
refit any model. `train` starts from the checked-in data snapshot, generates new
predictions, and evaluates them. Every command requires a new output directory;
an existing nonempty directory is rejected.

Run commands from the repository root after installing `requirements-lock.txt`.
The entry script also works when invoked by absolute path from another directory.

```sh
python scripts/reproduce.py evaluate --output ../spy-var-runs/cached-check
python scripts/reproduce.py evaluate --include-c2 --output ../spy-var-runs/cached-c2-check
python scripts/reproduce.py train --config configs/smoke.json --max-observations 160 --output ../spy-var-runs/smoke
python scripts/reproduce.py train --config configs/baselines.json --output ../spy-var-runs/baselines
python scripts/reproduce.py train --config configs/full_exploratory.json --output ../spy-var-runs/full-exploratory
python scripts/reproduce.py train --config configs/full_exploratory.json --seeds 42 43 44 --output ../spy-var-runs/multiple-seeds
```

The eight-model smoke configuration checks the pipeline with a chronological data prefix and
two training epochs. Its metrics are not research evidence. Full rolling GARCH
and neural experiments take substantially longer; runtime depends on hardware.

## Canonical implementation and historical materials

`src/spy_var/` is the authoritative implementation for new experiments.
Model implementations are grouped under `src/spy_var/models/`; legacy imports
remain compatibility aliases. New Transformer settings and changes are described
in `METHODS.md`, and actual three-seed runs in `EXPERIMENTS.md`.
`code/data/spy_data.csv` is the single input used by that implementation.
`code/sections/`, the original individual chapter scripts, and `code/src/*_original.py`
remain an archive so that report links and historical outputs remain inspectable.
The duplicated chapter data are not inputs to the new pipeline. They are not
silently deleted or substituted during this migration.

`code/run_all.py` redirects to cached evaluation with an output directory beside the checkout. `code/scripts/main.py` redirects
to the full configuration-driven pipeline. Use the new CLI for explicit options.

## Historical table mapping

| Historical artifact | Actual source / coverage |
| --- | --- |
| `code/outputs/tables/aligned_model_comparison.csv` | Five saved forecast files; 3,640 common observations, excluding C2. The CSV named `var_historical_simulation.csv` is **time-weighted** HS. |
| Chapter 5 C2 prediction CSV | Two variants (C1 and C2); cached evaluation selects C2 explicitly. Original C2 has 2,640 predictions, beginning 2008-01-03. |
| Chapter 6 C2-aligned table | Broader chapter-specific model set, including ordinary HS and GARCH-X. The quick CLI's six-model C2 comparison is not a reconstruction of that entire table. |
| New `train` runs | New experiment IDs, explicit configuration and environment. Uniform W=1000 by default; these results are not asserted to reproduce all historical values. |

The historical generic utility default was W=500, KDE used W=1000, and chapter
implementations differ from the generic MLP. The exact training settings of saved
files cannot be recovered from defaults alone. Historical manifests therefore
record that their training configuration is unknown rather than inventing it.

## Saved run record

Each training directory contains:

- `config.json`: expanded settings, including defaults and random seed.
- `manifest.json`: status, data SHA-256, sample dates, code fingerprints, Git
  revision, installed package versions, model runtime and fitting error counts.
- `forecasts/`: predictions; GARCH files also retain volatility, optimizer flags,
  warnings and errors. Unconverged fits are marked missing rather than scored.
- `training/`: per-refit epoch training/validation losses for MLP and C2.
- `tables/aligned_model_comparison.csv`: coverage, independence, conditional
  coverage and pinball loss on common finite dates at each quantile.
- `tables/evaluation_dates.csv`: the exact dates used for each quantile.
- `figures/aligned_failure_rates.png`: observed and target violation rates.

If a run fails, its manifest records the error and retains intermediate outputs.
Multi-seed runs retain each seed separately and summarize mean/std only when their
evaluation dates agree. A single seed has an undefined sample standard deviation.

## Forecast timing and methods

For HS, GARCH and the lag-window MLP, a forecast on date t uses observations
strictly before t. Scaling and early stopping use a chronological training /
validation split inside the current estimation window. Quantile outputs from
the neural models are sorted to remove crossings.

The C2 implementation migrates the original chapter training logic: lagged and
rolling features at t predict return at t+1; a separately fitted one-step GARCH
forecast supplies that target day's anchor. After aligning available anchors,
C2 needs another estimation window before its first forecast, so its evaluation
starts later. It warm-starts successive rolling fits as the original did. The
new pipeline supplies actual GARCH sigma instead of the old missing-sigma fallback.
The original feature timing is conservative: features at t already use shifted
observations. This is preserved rather than silently redefining the method.

Ordinary HS uses NumPy's default linear empirical quantile. Time-weighted HS uses
the inverse weighted empirical CDF with exponential decay 0.98. KDE HS uses the
same weights, Scott bandwidth and numerical Gaussian-mixture CDF inversion.
GARCH/GJR use a constant mean, Student-t innovations standardized to unit variance,
and percentage scaling during fitting, with forecasts returned to decimal units.

At each quantile, every model is evaluated on the same finite-date intersection.
Conflicting realized returns and duplicate dates are errors. Independence tests
use only transitions between adjacent observations in the source-data calendar;
they do not bridge excluded trading days. Degenerate sequences with an
unidentified two-state alternative return NaN for independence/conditional
coverage. Kupiec tests still report the unconditional-coverage statistic.

## Interpretation and unresolved research work

The original C2 weight decay was chosen after inspecting final out-of-sample
failure rates. The inherited setting and its p-values remain exploratory.
Reorganizing files, fixing paths, or adding seeds does not remove that selection
bias. A validation-only tuning protocol and an untouched future/independent test
sample are required for a confirmatory result; this revision does not invent a
new untouched sample inside an already inspected dataset.

Report rejection or non-rejection at a stated threshold. Do not describe a
non-rejection as proof of correctness, rank models by maximum p-value, or equate
coverage calibration with uniformly better pinball loss. New multi-seed runs
describe training variability; they do not by themselves establish superiority.

The data fields and integrity checks are documented in `code/data/README.md`.
No DOI or publication claim has been fabricated. `CITATION.cff` identifies the repository maintainer by GitHub
handle; it does not assert a publication or invent a preferred author identity.

## Rebuild the final editable figures

The original diagrams are generated from code and the real CSVs. Install the
optional figure dependencies and obtain the user-selected MIT-licensed tool at
the pinned revision in `REFERENCES.md`, outside the checkout:

```sh
python -m pip install -e .[figures]
python scripts/build_readme_figures.py --tool-root /path/to/EditableImage2PPTSkill --work-dir /path/to/private-figure-build
```

On Windows with PowerPoint installed, export the saved deck:

```powershell
.\scripts\export_readme_figures.ps1 -Deck .\docs\figures\spy_var_methods.pptx -OutputDirectory .\docs\figures -WorkDirectory D:\path\to\private-ppt-export
```

The work directory must be outside the checkout. Only the four final PNGs and
the editable PPTX belong in `docs/figures/`; extracted data plots, source sheets,
layout JSONs, compatibility copies and QA reports are private build materials.
All ordinary diagram text/shapes remain editable; numerical plots are separate
picture objects inside the deck. No reference-paper image is substituted for
this project's methods or measurements.

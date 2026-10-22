# Experiment evidence — 2026-10-22

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
| 42 | 182 | 269.70 | `badec70bc70d833aeaa64cb6b60479a01267e78998a9bc833fc8bbfc7c24a1e4` |
| 43 | 182 | 265.12 | `feb8a533f1798c1eb8854a4a90b6d105ea361d00cb0341659ef2f0767b5435f4` |
| 44 | 182 | 258.94 | `276aff99968ee737eb0a62e78e72352f231dcfac992fe5b91d1b888787119431` |

Trained model source SHA-256: `3e221fea45ecf5331eee3fb98013114b4c5bc6a8b98a4fcb113c39be345b5590`.

The independent seed runs reuse the same historical market observations; three
seeds do not triple the effective empirical sample size. Their SD describes
optimization variability, not a confidence interval. Per-seed p-values are
retained and never averaged. No outperformance claim follows from these runs.

## Archived input fingerprints

The raw-byte fingerprints below record the actual Windows inputs. Git's text
checkout conversion can change CRLF/LF bytes without changing any numeric value.
The included data snapshot's LF-normalized SHA-256 is `277406a832c1418de30221396bd8dbf12a444c3f5430ad88c7cc75d7a7f573a3`.
The model source is kept at LF line endings by `.gitattributes`.

- `code/data/spy_data.csv`: `98e5b96f02b8dea9a1e42a0f247eadb1ad89df0d99ce7428c5de2f8542ee55ab`
- `code/outputs/forecasts/var_historical_simulation.csv`: `2fd75e28e15f754e2fbeead0878cedf6f185e46afef1f52fddf515093f129fe6`
- `code/outputs/forecasts/var_kde_weighted_hs.csv`: `cb1084a9a8fb9fe1be7459c9df4b3c6b6a9e7f9f4e24f2bac95d793e5e04b807`
- `code/outputs/forecasts/var_garch_t.csv`: `b3ca516ec596db08d34a97e7fdaece9df2f73ca2e6bdcd3e94fadb484e57ef04`
- `code/outputs/forecasts/var_gjr_garch_t.csv`: `920100264f9d5ffb2eb06d9b2caa348e44ba7946cdd0f1cb8436417de2a627a5`
- `code/outputs/forecasts/var_mlp_quantile.csv`: `73758bfd533a81d64deb900cb41ef853b8f64dedac4600db7042da5d5c1a6859`
- `code/sections/section_05_mlp_quantile/outputs/external_section5_results/section5_model_c_garch_anchored_predictions.csv`: `9dff1ec0979c15ea4e2bc4284c38d7705a70c651cbd40e112d990689bf64c3ad`

## Reproduce

Run new experiments outside the checkout, then publish compact final evidence:

```sh
python scripts/reproduce.py train --config configs/modern_quantile.json --seeds 42 43 44 --output ../spy-var-runs/patch-seeds
python scripts/publish_readme_results.py --runs ../spy-var-runs/patch-seeds
```

See `METHODS.md` for model changes and timing, `REFERENCES.md` for original papers
and the high-star README selection, and `VALIDATION.md` for executed checks.

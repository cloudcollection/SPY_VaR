# Methods and forecast timing

## Task

Let r_t be the decimal daily SPY log return. A forecast q_(alpha,t) is a lower
conditional return quantile at alpha in {0.01, 0.05, 0.10}; a violation occurs
when r_t < q_(alpha,t). More negative values represent deeper downside thresholds.
The horizon is one trading observation. A forecast for t must use information
available strictly before its target return.

The snapshot has 4,640 observations from 2000-01-04 through 2018-06-27. The
loader validates chronological unique dates and finite log_ret, rv5 and bv.
No observations are imputed or removed by the new training loader.

## Statistical and MLP baselines

Ordinary HS uses the linear empirical quantile. Time-weighted HS uses exponential
age weights with decay 0.98 and the inverse weighted empirical CDF. KDE uses
the same weights, a Gaussian kernel, Scott bandwidth and numerical CDF inversion.
GARCH/GJR use a constant mean, GARCH(1,1) / asymmetric GJR(1,1) variance and
standardized Student-t innovations; decimal returns are multiplied by 100 for
fitting and forecasts are converted back. Failed optimizer convergence is
retained as a missing forecast with an explicit diagnostic.

The generic MLP flattens 20 lags of the three snapshot variables (60 inputs),
then uses dense widths 64, 32 and 3 with ReLU and first-block dropout 0.05.
It minimizes mean pinball loss with chronological validation inside each rolling
window. Its specification differs from the archived Chapter 5 Model A/B models.

## Conservative anchored correction (C2)

Implementation: `src/spy_var/models/garch_anchored.py`.

The input vector has 18 entries: 10 lagged/rolling return statistics, four
lagged/rolling rv5/bv statistics, and three one-step GARCH quantiles plus sigma.
A dense 18→64→32→3 network uses ReLU and dropout 0.1 after the first hidden block.
The raw correction passes through softplus and is subtracted from the GARCH
quantile vector, multiplied by scale 1. The three results are sorted.
The loss is mean pinball loss over the three quantiles.

The inherited chapter convention constructs lagged features at row t for target
t+1; this conservative information lag is retained. Baseline forecasts are
joined by their actual target date. C2 then needs a further rolling estimation
window of aligned anchor/feature records, so it starts later than the simple
baselines. The new training pipeline supplies actual fitted GARCH sigma.

The historical regularization choice followed inspection of test-period results.
C2 therefore remains exploratory; a near-nominal violation frequency does not
establish untouched-test performance or dominance over other scoring rules.

## PatchTST-inspired quantile Transformer

Implementation: `src/spy_var/models/patch_transformer.py`.
Fixed settings: `configs/modern_quantile.json`. Primary sources are listed in
`REFERENCES.md`.

| Component | Actual implementation |
| --- | --- |
| Input | One return channel; last 64 observed log returns |
| Normalization | Per-window mean and sqrt(population variance + 1e-5), detached; no learnable affine terms |
| Patches | Length 16, stride 8, no end padding; 7 tokens |
| Embedding | Shared Linear(16,32) plus learned 7×32 position parameters |
| Encoder | 2 standard Transformer encoder layers, 4 attention heads, FFN 64, GELU, LayerNorm, dropout 0.1 |
| Head | Flatten 7×32; Linear(224,3); inverse normalization and quantile sorting |
| Objective | Mean pinball loss for 1%, 5%, 10% quantiles |
| Optimization | Adam 0.001; batch 64; maximum 20 epochs; patience 4 |
| Rolling protocol | W=1000; refit every 20 days; last 20% of supervised targets used only for early stopping |
| Initialization | Train from scratch at each refit, with experiment seed set at the start of each run |

This is a small project adaptation inspired by PatchTST, not the official
long-horizon point-forecasting model. It changes the horizon, objective, output
head, configuration and normalization details. Only one channel is used, so
no multivariate channel-independence benefit is claimed. No pretrained weights
or undocumented pretraining corpus enter the experiment.

For a refit forecasting t, the rolling return array ends at t−1. Supervised
targets range from window_start+64 through t−1. Every target's 64 inputs precede
that target. The training/validation split is by target time; validation inputs
may legitimately include preceding training observations, but no future target
is read. Per-window normalization uses each observed input window alone.
Early stopping uses validation pinball loss, never test coverage or maximum p-value.

The settings were fixed before the new runs and are not advertised as tuned
optima. The experiment is retrospective because this historical dataset has
already been inspected. Repartitioning it would not create a genuinely untouched
confirmation sample. A W=1000 window contains only about 10 observations in a
nominal 1% tail; this is a sample-size calculation, not an observed exceedance count.

## Evaluation

All models in a comparison share exactly the same finite target dates at each
quantile. Duplicate dates and conflicting realized returns are errors.
Kupiec unconditional coverage, Christoffersen independence/conditional coverage,
pinball loss, fit errors and exclusions are recorded. Independence transitions
are restricted to adjacent observations in the original data calendar; missing
observations are not bridged. Degenerate unidentified alternatives return NaN.

The historical main table is recalculated from archived predictions on the
2,640 C2-aligned dates. The new Transformer runs use 3,640 target dates, so their
separate table is not a head-to-head ranking against the historical table.
Multi-seed means and sample standard deviations summarize optimization
variability, not statistical confidence intervals. P-values are retained by
seed; they are not averaged or used as model-ranking scores.

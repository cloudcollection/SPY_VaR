# Section 5. Neural Quantile Regression for One-day-ahead SPY VaR

## 5.1 Motivation

This section evaluates a neural quantile regression approach for one-day-ahead
SPY Value-at-Risk forecasting. The objective is deliberately modest: the neural
network is used as a flexible conditional quantile function, not as evidence
that nonlinear machine learning should dominate the classical historical
simulation and GARCH benchmarks.

## 5.2 Feature construction

The target variable is the next-day log return, so the information set at day
t is mapped to log_ret at day t+1. Model A uses lagged returns and rolling
return summaries only. Model B augments the same information set with lagged
and rolling realized-volatility proxies, rv5 and bv. All rolling features are
computed from shifted series, which prevents the forecast for day t+1 from
using information beyond day t.

## 5.3 MLP quantile regression model

Both specifications use a feed-forward MLP with two hidden layers of 64 and 32
units, ReLU activations, and dropout in the first hidden block. The output layer
contains three units corresponding to the 1%, 5%, and 10% conditional quantiles.
The model is trained by the average pinball loss across the three quantile
levels. Predicted quantiles are sorted after inference so that
VaR_1% <= VaR_5% <= VaR_10%.

## 5.4 Rolling training design

The forecasting exercise uses a 1000-observation rolling estimation window.
For each refit, the first 80% of the current window is used for parameter
training and the final 20% is used for validation-based early stopping. The
standardization parameters are fitted only on the training portion of the
current rolling window. The network is retrained every 20 trading days, with
one-day-ahead forecasts generated between refits.

## 5.5 Empirical results summary

At the 1% VaR level, Model A records 260 violations from
3617 forecasts, compared with 243
violations for Model B. The corresponding 1% failure rates are
0.0719 and 0.0672. The average pinball loss
across the three reported quantiles is 0.002663
for Model A and 0.003800 for Model B.
The realized-volatility augmented specification does not deliver a lower average pinball loss in this run.

## 5.6 Interpretation

The results should be interpreted as an incremental robustness check rather
than a replacement for the classical VaR models. The realized-volatility inputs
rv5 and bv provide additional state variables for volatility clustering, but
their empirical value depends on whether they improve both calibration and
loss-based accuracy out of sample. A final judgment on whether the neural model
is superior to historical simulation or GARCH-type methods should be made in
Section 6 using the common backtesting and loss metrics across all models.

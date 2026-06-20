# Section 5 Model C: GARCH-Anchored Neural Quantile Correction

## 5.8 Motivation for GARCH-anchored Neural Correction

The baseline MLP-QR models severely underestimated SPY lower-tail risk. Their
1% failure rates were far above the nominal 1% target, which indicates that a
neural network trained directly on rolling-window samples may not learn a stable
extreme-tail VaR level. Section 4 showed the opposite strength and weakness for
GARCH-type models: GARCH-t improved the timing of violations by modeling
conditional volatility, but it still suffered from unconditional coverage
errors. Model C therefore uses GARCH-t VaR as a structured baseline and trains
a neural network to correct the VaR level.

## 5.9 Model Specification

The additive anchored specification is

VaR_NN_alpha,t+1 = VaR_BASE_alpha,t+1 + g_theta(X_t).

VaR_BASE is the GARCH-t VaR forecast, and g_theta(X_t) is the neural correction.
The information set X_t contains lagged returns, rolling return summaries, rv5,
bv, GARCH VaR forecasts, and the GARCH volatility forecast when available.

The conservative version is

VaR_NN_alpha,t+1 = VaR_BASE_alpha,t+1 - softplus(g_theta(X_t)).

This version only allows the neural correction to make the GARCH baseline more
conservative.

## 5.10 Rolling Training Design

The rolling design uses W = 1000 observations and one-day-ahead forecasts. Each
rolling window is split internally into 90% training and 10% validation. The
feature scaler is fitted only on the training part of the current rolling
window. The network is retrained every 20 trading days with Adam, learning rate
0.001, batch size 64, maximum 200 epochs, and early stopping patience 10.

## 5.11 Empirical Results

Model C1 records 104 1% violations from
2640 forecasts, for a failure rate of
0.0394. Model C2 records 25
1% violations, for a failure rate of 0.0095.
At the 1% level, both anchored variants materially reduce the severe underestimation observed in the baseline MLP-QR models. The anchored specification also improves the average pinball loss relative to the better baseline MLP model.

The Kupiec, Christoffersen, and Duration p-values should still be interpreted
cautiously because the anchored correction is trained on a difficult left-tail
forecasting problem with limited extreme observations.

## 5.12 Interpretation

The results should not be interpreted as evidence that a neural network alone
dominates classical VaR models. The relevant question is whether neural
flexibility becomes more useful when it is tied to a financial-risk structure.
If Model C improves calibration relative to Model A and Model B, the evidence
supports the idea that GARCH anchoring helps neural quantile forecasting. If it
does not improve all diagnostics, the conclusion is narrower: even GARCH
anchoring is insufficient by itself, and stronger quantile dynamics or
post-training calibration may be needed.

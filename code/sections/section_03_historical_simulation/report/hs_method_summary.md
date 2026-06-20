# Historical Simulation VaR Optimization Summary

## 1. Objective

The goal of this part of the project is to improve the Historical Simulation (HS) VaR model for SPY daily log returns. The original ordinary HS method is simple and transparent, but it has two important weaknesses:

1. It treats all observations in the rolling window equally.
2. It estimates VaR from a discrete empirical distribution, which is especially unstable for the 1% tail.

Because of these limitations, I tested two improvements:

- Time-weighted Historical Simulation
- Gaussian KDE-smoothed Time-weighted Historical Simulation

The backtesting metrics used for comparison are:

- Failure rate
- Kupiec unconditional coverage test
- Christoffersen independence test
- Duration test

## 2. Ordinary Historical Simulation

Ordinary HS estimates VaR as the empirical quantile of past returns in the rolling window:

```text
VaR_{alpha,t+1} = Q_alpha(r_{t-W+1}, ..., r_t)
```

This method is distribution-free and easy to explain. However, it assumes that all returns in the window are equally relevant for tomorrow's risk. In financial data, this is often unrealistic because volatility clusters and market regimes change over time.

In the initial backtesting, ordinary HS had acceptable performance around the 5% level, but it showed problems at the 1% and 10% levels. In particular, the 1% VaR had too many violations, suggesting that the model underestimated extreme downside risk.

## 3. Time-weighted Historical Simulation

To improve ordinary HS, I implemented an exponentially time-weighted HS model. The idea follows the age-weighted historical simulation logic used in risk-management literature: recent observations receive larger weights, while older observations receive smaller weights.

The weight structure is:

```text
w_i proportional to lambda^age_i
```

where:

- `age_i = 0` for the newest observation
- older observations have larger age
- `lambda` controls how fast the weights decay

I tested different decay values and found that `lambda = 0.98` worked better than `lambda = 0.94`. A decay value of `0.94` gave too much weight to very recent observations and made the VaR estimates too reactive. A value of `0.98` provided a better balance between responsiveness and stability.

I also compared rolling windows:

```text
T = 500
T = 1000
```

The `T = 1000` window was slightly better overall, especially in the Christoffersen independence test. The longer window provides more tail observations and more stable violation dynamics.

## 4. Time-weighted HS Backtesting Results

Using `lambda = 0.98`, the comparison between `T = 500` and `T = 1000` showed:

| Window | Alpha | Failure Rate | Kupiec p-value | Christoffersen p-value | Interpretation |
|---:|---:|---:|---:|---:|---|
| 500 | 1% | 0.0162 | 0.0002 | 0.0273 | Fails coverage and independence |
| 500 | 5% | 0.0527 | 0.4366 | 0.1060 | Good |
| 500 | 10% | 0.0995 | 0.9174 | 0.2354 | Very good |
| 1000 | 1% | 0.0168 | 0.0002 | 0.0218 | Fails coverage and independence |
| 1000 | 5% | 0.0530 | 0.4072 | 0.2366 | Good |
| 1000 | 10% | 0.1019 | 0.6998 | 0.3569 | Good |

The main conclusion is:

- Time-weighted HS works well for 5% and 10% VaR.
- It still struggles at the 1% tail.
- The 1% level has too few tail observations, so the empirical quantile remains unstable even after time weighting.

## 5. Gaussian KDE-smoothed Time-weighted HS

To improve the 1% tail, I implemented a Gaussian kernel density estimation (KDE) version of time-weighted HS.

The motivation is that ordinary HS and time-weighted HS both use discrete historical observations. At the 1% level, this means VaR depends heavily on a very small number of extreme observations. KDE smooths the weighted empirical distribution into a continuous probability distribution.

The procedure is:

1. Take a rolling window of past returns.
2. Apply exponential time weights with `lambda = 0.98`.
3. Fit a Gaussian KDE using the weighted return sample.
4. Numerically invert the KDE-implied CDF to obtain the 1%, 5%, and 10% VaR forecasts.

The KDE-smoothed model uses:

```text
T = 1000
lambda = 0.98
Gaussian kernel
Scott bandwidth rule
```

## 6. KDE Backtesting Results

The KDE method improved the 1% VaR coverage substantially:

| Model | Alpha | Failure Rate | Kupiec p-value | Christoffersen p-value | Duration p-value | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Time-weighted HS | 1% | 0.0162 | 0.0002 | 0.0273 | 0.0121 | Poor |
| KDE Time-weighted HS | 1% | 0.0129 | 0.0911 | 0.0246 | 0.1258 | Better coverage |
| Time-weighted HS | 5% | 0.0527 | 0.4366 | 0.1060 | 0.0121 | Good coverage |
| KDE Time-weighted HS | 5% | 0.0459 | 0.2476 | 0.0638 | 0.0003 | Acceptable coverage, weak duration result |
| Time-weighted HS | 10% | 0.0995 | 0.9174 | 0.2354 | 0.9885 | Very good |
| KDE Time-weighted HS | 10% | 0.0898 | 0.0379 | 0.1349 | 0.8685 | Too conservative |

The most important improvement is at the 1% level:

```text
1% Kupiec p-value:
Time-weighted HS      0.0002
KDE Time-weighted HS  0.0911
```

This means KDE time-weighted HS is no longer rejected by the Kupiec coverage test at the 5% significance level for 1% VaR.

However, the KDE model is not uniformly better:

- It improves 1% VaR coverage.
- It makes 10% VaR too conservative.
- At 5%, the result is acceptable but not clearly superior to time-weighted HS.

## 7. Duration Test Interpretation

I also added a duration-based test. The motivation is that Christoffersen's independence test only uses a first-order transition structure, while the duration test directly examines the time gaps between VaR violations.

If VaR violations are independent, the durations between violations should behave like a memoryless process. A low duration p-value suggests violation clustering.

The duration test provides additional information:

- Time-weighted HS has good 5% coverage, but its 5% duration p-value is low.
- KDE time-weighted HS improves 1% duration performance relative to time-weighted HS.
- For 10% VaR, time-weighted HS performs very well across coverage, independence, and duration tests.

## 8. Recommended Final Historical Simulation Specification

The best practical specification depends on the VaR level:

| VaR Level | Recommended Method | Reason |
|---:|---|---|
| 1% | KDE Time-weighted HS, T=1000, lambda=0.98 | Best tail coverage among HS variants |
| 5% | Time-weighted HS, T=1000, lambda=0.98 | Stable coverage and independence |
| 10% | Time-weighted HS, T=1000, lambda=0.98 | Best overall performance |

If a single HS-family model must be selected for all VaR levels, I would choose:

```text
Time-weighted HS with T = 1000 and lambda = 0.98
```

This is the most balanced and interpretable model. It performs well at 5% and 10%, although it still underestimates the 1% tail.

If the report is allowed to use level-specific model selection, I would choose:

```text
1% VaR: KDE Time-weighted HS
5% VaR: Time-weighted HS
10% VaR: Time-weighted HS
```

This is the best performance-driven choice.

## 9. Can Historical Simulation Be Improved Further?

There are still possible improvements, but they would make the HS section more complex:

1. Tune the KDE bandwidth instead of using Scott's rule.
2. Use adaptive bandwidth KDE so the tail receives different smoothing from the center.
3. Combine KDE with extreme value theory for the far tail.
4. Use filtered historical simulation, where returns are standardized by a volatility model before applying HS.
5. Estimate the decay parameter `lambda` using a validation objective instead of choosing it manually.

However, for this mini project, the current HS optimization is already sufficient:

- Ordinary HS was improved with time weighting.
- The rolling window was compared at 500 and 1000.
- KDE smoothing was tested for the 1% tail.
- Backtesting now includes coverage, independence, and duration diagnostics.

Further improvements would likely shift the project toward a more advanced tail-risk modeling paper rather than a clean VaR forecasting mini project.

## 10. Final Conclusion

The HS-family models show a clear tradeoff:

- Time-weighted HS is simple, interpretable, and strong at 5% and 10%.
- KDE time-weighted HS improves the difficult 1% tail by smoothing the discrete empirical distribution.
- No HS variant fully solves all VaR levels at once.

The strongest conclusion for the report is:

```text
Time weighting improves ordinary Historical Simulation by making the model more responsive to recent volatility regimes. Gaussian KDE smoothing further improves the 1% tail by converting the discrete weighted empirical distribution into a continuous density estimate. Empirically, KDE improves 1% coverage, while time-weighted HS remains preferable for 5% and 10% VaR.
```


## Section 3: Historical Simulation Methods

### 3.1 Motivation

Historical Simulation (HS) is used in this project as the non-parametric benchmark for one-day-ahead Value-at-Risk (VaR) forecasting of SPY daily log returns. The data provide a quantitative reason for starting from a non-parametric method. The full-sample excess kurtosis of SPY log returns is 8.22, far above the Gaussian benchmark of zero. The crisis-period standard deviation from 2007-09-01 to 2009-06-30 is 0.0224, whereas the post-crisis calm-period standard deviation from 2012-01-01 to 2016-12-31 is 0.0081. The ratio is 2.78. These two facts are central to the modelling strategy: heavy tails make normal-distribution VaR unattractive, and the volatility-regime gap makes fixed-window estimation fragile. At the same time, the simplicity of HS is also the source of its main weakness: the method uses past returns as if they were drawn from a locally stable distribution.

The purpose of this section is not to treat HS as a single mechanical benchmark. Instead, the analysis starts with ordinary HS and then introduces two targeted improvements: time-weighted HS and KDE-smoothed time-weighted HS. This design follows the logic of Boudoukh, Richardson and Whitelaw (1998), who show that hybrid historical methods can retain the transparency of HS while improving the treatment of time-varying volatility. It also responds to the critique in Pritsker (2006), who shows that standard HS can systematically underestimate tail risk when the return period of extreme events is long relative to the estimation window. In this data set, the maximum daily log-return loss is -9.69%, an event too rare to be represented reliably in short rolling windows.

The empirical question is therefore twofold. First, does equal-weighted HS provide acceptable unconditional coverage at the 1%, 5%, and 10% VaR levels? Second, if it fails, is the failure due to the number of violations, the temporal clustering of violations, or both? The answer matters because a model with the correct average failure rate can still be weak for risk management if violations arrive in clusters during crises.

### 3.2 Rolling Window Design

For each forecast date, the estimation window contains only past observations. If the forecast is made at time t, the information set used by the HS-family models is

$$
\mathcal{F}_t(W)=\{r_{t-W+1},r_{t-W+2},\ldots,r_t\},
\tag{1}
$$

where $r_t$ denotes the SPY daily log return and $W$ is the rolling window length. The VaR forecast constructed from this window is evaluated against $r_{t+1}$. This timing convention avoids look-ahead bias because the realized return being forecast is never included in the estimation sample.

The assignment does not specify a unique rolling window length. I therefore compare W = 250, 500, and 1000 trading days. These windows correspond approximately to one, two, and four trading years. The comparison is statistically important because HS quantiles are sensitive to the number of observations available in the left tail. With W = 250, the 1% empirical tail contains only about two or three observations. With W = 1000, it contains about ten observations, which is still small but materially more stable. This tension is closely related to the horizon and regime-switching problem discussed by Danielsson, Ergun, de Haan and de Vries (2016): a longer window improves tail precision but may mix different volatility regimes, while a shorter window adapts faster but has weak tail information.

The full sample contains 4,640 daily observations from 2000-01-04 to 2018-06-27. The out-of-sample period depends on W. For W = 250, forecasts run from 2001-01-02 to 2018-06-27 with 4,390 forecast days. For W = 500, forecasts run from 2002-01-09 to 2018-06-27 with 4,140 forecast days. For W = 1000, forecasts run from 2004-01-08 to 2018-06-27 with 3,640 forecast days. The W = 1000 window is emphasized in the final interpretation because it contains more tail observations and allows a cleaner crisis-versus-calm subsample comparison. All backtests are interpreted at the 5% significance level unless otherwise stated.

### 3.3 Ordinary Historical Simulation

Ordinary HS estimates VaR as the empirical lower-tail quantile of the rolling return sample:

$$
\widehat{\mathrm{VaR}}^{HS}_{\alpha,t+1}
=Q_{\alpha}(r_{t-W+1},\ldots,r_t).
\tag{2}
$$

Here $Q_{\alpha}(\cdot)$ is the empirical $\alpha$-quantile, with $\alpha$ equal to 1%, 5%, or 10%. Each return in the rolling window receives weight $1/W$. This estimator is transparent and easy to audit, which explains why HS remains widely used in applied risk management. However, ordinary HS treats an observation from a calm regime and an observation from a recent stress regime as equally informative. This assumption is difficult to justify for financial returns, where volatility is persistent and crisis periods produce clustered losses.

The ordinary HS results in this project are consistent with the warning in Pritsker (2006): the method can look acceptable at moderate quantiles while still failing in the extreme tail or in independence diagnostics. In the W = 1000 case, ordinary HS has a reasonable 5% failure rate, but the Christoffersen independence p-value is 0.0004, indicating that violations are not randomly scattered over time. This means that the model captures the unconditional frequency of 5% losses but not the conditional dynamics of risk.

### 3.4 Time-weighted Historical Simulation

Time-weighted HS modifies ordinary HS by assigning exponentially larger weights to more recent observations. The unnormalized weight of observation i is

$$
\tilde{w}_i=\lambda^{\mathrm{age}_i}, \qquad 0<\lambda<1,
\tag{3}
$$

where $\mathrm{age}_i$ equals zero for the most recent observation and increases for older observations. The normalized weight is

$$
w_i=\frac{\lambda^{\mathrm{age}_i}}{\sum_{j=0}^{W-1}\lambda^j}.
\tag{4}
$$

The time-weighted VaR is the smallest value $q$ for which the cumulative weight of returns below $q$ reaches $\alpha$:

$$
\widehat{\mathrm{VaR}}^{TWHS}_{\alpha,t+1}
=\inf\left\{q:\sum_i w_i\mathbf{1}(r_i\le q)\ge \alpha\right\}.
\tag{5}
$$

I use $\lambda = 0.98$. This choice follows the spirit of Boudoukh, Richardson and Whitelaw (1998), who show that historical simulation can be improved by allowing recent observations to receive greater weight. A smaller value such as 0.94 makes the estimator more reactive, but it also lowers the effective sample size and makes tail estimates more unstable. In this data set, $\lambda = 0.98$ provides a practical compromise between responsiveness and tail stability.

The results corroborate Boudoukh, Richardson and Whitelaw (1998) in that time weighting improves the dynamics of violations without abandoning the non-parametric structure of HS. The improvement is strongest at the 5% and 10% levels. This is exactly where time weighting should help most: these quantiles have enough observations for reweighting to matter, while still being sensitive to volatility persistence. At the 1% level, however, the effective number of tail observations remains too small. For W = 1000, time-weighted HS passes both the Kupiec and Christoffersen independence tests at the 5% and 10% levels, but it still fails at the 1% level. This failure is economically meaningful rather than only statistical: the 1% expected number of violations is 36.4, whereas time-weighted HS produces 61 violations.

### 3.5 KDE-smoothed Time-weighted Historical Simulation

The second extension addresses the discreteness of empirical quantiles. Time weighting changes the relevance of observations, but the quantile is still selected from a finite set of historical returns. This is problematic in the 1% tail, where the effective number of observations is small even when W = 1000. I therefore estimate a Gaussian-kernel-smoothed density on the weighted return sample:

$$
\hat{f}_t(x)=\frac{1}{h}\sum_i w_iK\left(\frac{x-r_i}{h}\right),
\qquad
K(u)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{u^2}{2}\right).
\tag{6}
$$

The KDE VaR is obtained by inverting the estimated cumulative distribution:

$$
\widehat{\mathrm{VaR}}^{KDE}_{\alpha,t+1}
=\inf\left\{x:\int_{-\infty}^{x}\hat{f}_t(u)\,du\ge \alpha\right\}.
\tag{7}
$$

The implementation uses `scipy.stats.gaussian_kde` with observation weights and Scott bandwidth selection. Conceptually, the weighted Scott bandwidth is proportional to $\sigma_w n_{\mathrm{eff}}^{-1/5}$, where $\sigma_w$ is the weighted dispersion of returns and $n_{\mathrm{eff}} = 1 / \sum_i w_i^2$ when weights sum to one. This detail matters because exponential weighting reduces the effective sample size, which increases smoothing.

KDE smoothing is useful, but it is not a free improvement. Quantile estimation based on kernel smoothing is affected by bandwidth bias. At a boundary or tail point, the relevant non-parametric convergence rate is closer to n^{-2/5} under standard bandwidth choices rather than the parametric n^{-1/2} rate. This slower rate helps explain why the 1% tail remains unstable: smoothing reduces discreteness, but it cannot create genuine tail information that is absent from the sample. The W = 1000 result illustrates this tradeoff. KDE time-weighted HS improves the 1% Kupiec p-value to 0.0911, but it creates a weaker 10% coverage result and a strong 5% Duration rejection. This finding is consistent with Silverman (1986) and Bowman and Azzalini (1997), who emphasize that bandwidth rules designed for smooth density estimation can perform poorly under non-Gaussian, heavy-tailed data. In the present application, Scott bandwidth smoothing helps the far-left 1% tail but distorts the intermediate 5% tail.

### 3.6 Backtesting Methodology

For each VaR forecast, define the violation indicator as

$$
I_{t+1}=\mathbf{1}\left(r_{t+1}<\widehat{\mathrm{VaR}}_{\alpha,t+1}\right).
\tag{8}
$$

The sample failure rate is

$$
\widehat{p}=\frac{1}{T}\sum_{t=1}^{T}I_t.
\tag{9}
$$

Kupiec (1995) tests unconditional coverage. The null hypothesis is $H_0:p=\alpha$, where $p$ is the true violation probability. If $V$ denotes the number of violations in $T$ out-of-sample forecasts, the likelihood-ratio statistic is

$$
LR_{uc}
=-2\log\left[
\frac{(1-\alpha)^{T-V}\alpha^V}
{(1-\hat{p})^{T-V}\hat{p}^{V}}
\right]\sim\chi^2(1).
\tag{10}
$$

The finite-sample power of this test is limited at the 1% level. For $T = 3640$ and $\alpha = 1\%$, the 5% two-sided Kupiec rejection region based on the binomial likelihood ratio is $V \le 25$ or $V \ge 49$, with an acceptance region from 26 to 48 violations. The upper rejection boundary corresponds to a failure rate of $49/3640 = 1.35\%$. A binomial power calculation shows that, on the upper side, the true failure probability must be approximately 1.50% before the test reaches 80% power. Thus, a model can materially underestimate 1% tail risk and still be difficult to reject in finite samples. This calculation is important for interpreting the KDE result: passing the 1% Kupiec test does not prove that the extreme tail is fully solved.

Christoffersen (1998) adds an independence requirement. Let T_{ij} be the number of transitions from state i to state j in the violation sequence, where 0 denotes no violation and 1 denotes violation. The estimated transition probabilities are

$$
\hat{\pi}_{01}=\frac{T_{01}}{T_{00}+T_{01}},
\qquad
\hat{\pi}_{11}=\frac{T_{11}}{T_{10}+T_{11}}.
\tag{11}
$$

The independence null is $H_0:\pi_{01}=\pi_{11}$. The likelihood-ratio statistic is

$$
LR_{ind}
=-2\log\left[
\frac{(1-\hat{p})^{T_{00}+T_{10}}\hat{p}^{T_{01}+T_{11}}}
{(1-\hat{\pi}_{01})^{T_{00}}\hat{\pi}_{01}^{T_{01}}
(1-\hat{\pi}_{11})^{T_{10}}\hat{\pi}_{11}^{T_{11}}}
\right]\sim\chi^2(1).
\tag{12}
$$

The conditional coverage statistic is

$$
LR_{cc}=LR_{uc}+LR_{ind}\sim\chi^2(2).
\tag{13}
$$

This decomposition is useful because a rejection of conditional coverage can arise from a biased violation frequency, clustered violations, or both.

The Duration test of Christoffersen and Pelletier (2004) examines the waiting times between violations. If tau_i is the date of the i-th violation, then

$$
D_i=\tau_i-\tau_{i-1}.
\tag{14}
$$

Under correct coverage and independence, durations are memoryless. In discrete time, the corresponding distribution is geometric:

$$
P(D=d)=(1-\alpha)^{d-1}\alpha,\qquad d=1,2,3,\ldots .
\tag{15}
$$

Christoffersen and Pelletier (2004) use a Weibull alternative,

$$
f(d;a,b)=ab(ad)^{b-1}\exp[-(ad)^b],
\tag{16}
$$

where b = 1 corresponds to the exponential distribution and hence to the memoryless benchmark. The likelihood-ratio statistic is

$$
LR_{dur}=-2\{\ell(\hat{a},1)-\ell(\hat{a},\hat{b})\}\sim\chi^2(1).
\tag{17}
$$

Duration testing is particularly useful at $\alpha = 1\%$, where the transition count $T_{11}$ can be small and the Markov-chain independence test can have weak finite-sample behavior.

As a complementary scoring metric, I also report the Lopez regulatory loss:

$$
L_t =
\begin{cases}
1 + (r_t-\widehat{\mathrm{VaR}}_t)^2, & r_t < \widehat{\mathrm{VaR}}_t,\\
0, & r_t \ge \widehat{\mathrm{VaR}}_t.
\end{cases}
\tag{18}
$$

Unlike Kupiec and Christoffersen tests, Lopez loss is not a hypothesis test. It is a loss score that penalizes both the occurrence of a VaR violation and the depth of the exceedance. In this data set, returns are measured in decimal log-return units, so the squared exceedance component is numerically small and Lopez loss is close to the failure rate. Its value is still useful because it provides a consistent ranking criterion when models have similar coverage but different violation magnitudes.

### 3.7 Empirical Results

Table 3.1 reports the W = 250 results. The one-year window adapts quickly, but it has very few observations in the 1% tail. Ordinary HS has acceptable 5% and 10% unconditional coverage, but the Christoffersen p-values indicate violation clustering. Time-weighted HS improves the 5% and 10% independence diagnostics, while KDE improves 1% coverage at the cost of conservativeness at 10%. This finding is consistent with Pritsker (2006), who emphasizes that short-window HS can be unreliable for tail risk because the empirical tail is too thin.

Table 3.1. Backtesting results for HS-family models, W = 250.

| Model | Alpha | Viol./Exp. | Fail. rate | Avg VaR | Kupiec p | Christoffersen p | Lopez loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary HS | 1% | 72 / 43.9 | 0.0164 | -0.0284 | 0.0001 | 0.0370 | 0.016404 |
| Ordinary HS | 5% | 237 / 219.5 | 0.0540 | -0.0180 | 0.2313 | 0.0000 | 0.053995 |
| Ordinary HS | 10% | 450 / 439.0 | 0.1025 | -0.0129 | 0.5814 | 0.0001 | 0.102522 |
| Time-weighted HS | 1% | 72 / 43.9 | 0.0164 | -0.0275 | 0.0001 | 0.0370 | 0.016402 |
| Time-weighted HS | 5% | 232 / 219.5 | 0.0528 | -0.0177 | 0.3909 | 0.0998 | 0.052853 |
| Time-weighted HS | 10% | 438 / 439.0 | 0.0998 | -0.0129 | 0.9599 | 0.1240 | 0.099783 |
| KDE Time-weighted HS | 1% | 54 / 43.9 | 0.0123 | -0.0289 | 0.1392 | 0.0323 | 0.012302 |
| KDE Time-weighted HS | 5% | 194 / 219.5 | 0.0442 | -0.0188 | 0.0719 | 0.0069 | 0.044196 |
| KDE Time-weighted HS | 10% | 383 / 439.0 | 0.0872 | -0.0139 | 0.0041 | 0.0517 | 0.087253 |

Note: The Christoffersen column reports the independence test p-value. Time-weighted models use $\lambda = 0.98$. The 5% significance level is used for rejection decisions.

Table 3.2 reports the W = 500 results. The main pattern is unchanged, but the advantage of time weighting becomes clearer. Ordinary HS again has reasonable unconditional coverage at the 5% and 10% levels, yet its independence p-values are effectively zero. Time-weighted HS keeps the failure rates close to the nominal levels and improves independence. KDE continues to help the 1% failure rate but makes the 10% quantile too conservative.

Table 3.2. Backtesting results for HS-family models, W = 500.

| Model | Alpha | Viol./Exp. | Fail. rate | Avg VaR | Kupiec p | Christoffersen p | Lopez loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary HS | 1% | 67 / 41.4 | 0.0162 | -0.0303 | 0.0002 | 0.0049 | 0.016187 |
| Ordinary HS | 5% | 214 / 207.0 | 0.0517 | -0.0182 | 0.6195 | 0.0000 | 0.051702 |
| Ordinary HS | 10% | 396 / 414.0 | 0.0957 | -0.0129 | 0.3479 | 0.0000 | 0.095671 |
| Time-weighted HS | 1% | 67 / 41.4 | 0.0162 | -0.0270 | 0.0002 | 0.0273 | 0.016185 |
| Time-weighted HS | 5% | 218 / 207.0 | 0.0527 | -0.0174 | 0.4366 | 0.1060 | 0.052662 |
| Time-weighted HS | 10% | 412 / 414.0 | 0.0995 | -0.0126 | 0.9174 | 0.2354 | 0.099528 |
| KDE Time-weighted HS | 1% | 51 / 41.4 | 0.0123 | -0.0283 | 0.1479 | 0.0274 | 0.012320 |
| KDE Time-weighted HS | 5% | 186 / 207.0 | 0.0449 | -0.0185 | 0.1279 | 0.0129 | 0.044932 |
| KDE Time-weighted HS | 10% | 362 / 414.0 | 0.0874 | -0.0136 | 0.0060 | 0.0530 | 0.087449 |

Note: The Christoffersen column reports the independence test p-value. Time-weighted models use $\lambda = 0.98$. The 5% significance level is used for rejection decisions.

Table 3.3 reports the W = 1000 results. This window provides the strongest overall evidence. Ordinary HS passes the 5% Kupiec test but fails independence, which indicates that the average number of violations is not enough to establish model adequacy. Time-weighted HS performs best at the 5% and 10% levels, with both coverage and independence p-values above 5%. KDE time-weighted HS is the only HS-family specification that passes the 1% Kupiec test, but the Christoffersen p-value remains below 5%, and the 10% Kupiec p-value indicates excessive conservativeness.

Table 3.3. Backtesting results for HS-family models, W = 1000.

| Model | Alpha | Viol./Exp. | Fail. rate | Avg VaR | Kupiec p | Christoffersen p | Lopez loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordinary HS | 1% | 56 / 36.4 | 0.0154 | -0.0334 | 0.0025 | 0.0002 | 0.015389 |
| Ordinary HS | 5% | 177 / 182.0 | 0.0486 | -0.0187 | 0.7025 | 0.0004 | 0.048641 |
| Ordinary HS | 10% | 303 / 364.0 | 0.0832 | -0.0129 | 0.0005 | 0.0000 | 0.083265 |
| Time-weighted HS | 1% | 61 / 36.4 | 0.0168 | -0.0264 | 0.0002 | 0.0218 | 0.016759 |
| Time-weighted HS | 5% | 193 / 182.0 | 0.0530 | -0.0169 | 0.4072 | 0.2366 | 0.053027 |
| Time-weighted HS | 10% | 371 / 364.0 | 0.1019 | -0.0120 | 0.6998 | 0.3569 | 0.101935 |
| KDE Time-weighted HS | 1% | 47 / 36.4 | 0.0129 | -0.0276 | 0.0911 | 0.0246 | 0.012913 |
| KDE Time-weighted HS | 5% | 167 / 182.0 | 0.0459 | -0.0178 | 0.2476 | 0.0638 | 0.045884 |
| KDE Time-weighted HS | 10% | 327 / 364.0 | 0.0898 | -0.0129 | 0.0379 | 0.1349 | 0.089845 |

Note: The Christoffersen column reports the independence test p-value. Time-weighted models use $\lambda = 0.98$. The 5% significance level is used for rejection decisions.

The Avg VaR column is economically informative. At $W = 1000$ and $\alpha = 1\%$, ordinary HS has an average VaR of -0.0334, while time-weighted HS has an average VaR of -0.0264. The latter is less conservative on average because older crisis observations receive less weight. This explains why time-weighted HS can improve 5% and 10% dynamics while worsening 1% underestimation. KDE time-weighted HS moves the average 1% VaR to -0.0276, which is slightly more conservative than time-weighted HS but still less conservative than ordinary HS. The result is a better balance for 1% unconditional coverage, although not a full solution to violation clustering.

The repeated 1% violation counts for ordinary HS and time-weighted HS at W = 250 and W = 500 are not evidence of duplicated calculations. The average VaR values differ, and diagnostic checks show that the violation dates also differ. There are 40 different 1% violation dates for W = 250 and 60 different 1% violation dates for W = 500. The aggregate counts are equal because some dates are added as violations and others are removed.

Table 3.4 reports the Duration test p-values available for the preferred W = 1000 comparisons. The results clarify why the KDE improvement should be interpreted carefully. KDE time-weighted HS improves 1% duration behavior relative to time-weighted HS, with the p-value increasing from 0.0121 to 0.1258. However, at the 5% level, KDE produces a Duration p-value of 0.0003. This is not merely a statistical artefact. It is consistent with bandwidth mismatch: a global Gaussian bandwidth that stabilizes the far-left tail can distort the intermediate tail, producing periods with too few violations followed by clustered corrections when volatility changes.

Table 3.4. Duration-test diagnostics for W = 1000.

| Model | 1% Duration p | 5% Duration p | 10% Duration p |
|---|---:|---:|---:|
| Time-weighted HS | 0.0121 | 0.0121 | 0.9885 |
| KDE Time-weighted HS | 0.1258 | 0.0003 | 0.8685 |

Note: Duration p-values are based on the Christoffersen and Pelletier (2004) duration-based independence test. The 5% significance level is used for rejection decisions.

The 1% underestimation is not evenly distributed over time. Table 3.5 separates the W = 1000 forecasts into the crisis period from 2007-09-01 to 2009-06-30 and the post-crisis calm period from 2012-01-01 to 2016-12-31. The contrast is sharp. Ordinary HS has a 1% crisis-period failure rate of 7.17%, compared with an expected rate of 1%. In the calm period, the corresponding failure rate is only 0.64%. This shows that the overall failure of ordinary HS is driven by volatility-regime switching rather than by a constant bias over the full sample. During a regime switch, ordinary HS reacts slowly because the estimation window still contains many calm-period observations. Time-weighted HS reacts faster and reduces the crisis-period 1% failure rate to 2.39%. KDE time-weighted HS reduces it further to 1.74%, but the value remains above the nominal level.

Table 3.5. W = 1000 subsample failure rates.

| Model | Period | Alpha | Obs. | Viol./Exp. | Fail. rate | Avg VaR |
|---|---|---:|---:|---:|---:|---:|
| Ordinary HS | Crisis | 1% | 460 | 33 / 4.6 | 0.0717 | -0.0335 |
| Ordinary HS | Crisis | 5% | 460 | 86 / 23.0 | 0.1870 | -0.0167 |
| Ordinary HS | Crisis | 10% | 460 | 117 / 46.0 | 0.2543 | -0.0112 |
| Time-weighted HS | Crisis | 1% | 460 | 11 / 4.6 | 0.0239 | -0.0502 |
| Time-weighted HS | Crisis | 5% | 460 | 23 / 23.0 | 0.0500 | -0.0355 |
| Time-weighted HS | Crisis | 10% | 460 | 50 / 46.0 | 0.1087 | -0.0266 |
| KDE Time-weighted HS | Crisis | 1% | 460 | 8 / 4.6 | 0.0174 | -0.0527 |
| KDE Time-weighted HS | Crisis | 5% | 460 | 23 / 23.0 | 0.0500 | -0.0373 |
| KDE Time-weighted HS | Crisis | 10% | 460 | 48 / 46.0 | 0.1043 | -0.0282 |
| Ordinary HS | Calm | 1% | 1258 | 8 / 12.6 | 0.0064 | -0.0318 |
| Time-weighted HS | Calm | 1% | 1258 | 16 / 12.6 | 0.0127 | -0.0231 |
| KDE Time-weighted HS | Calm | 1% | 1258 | 15 / 12.6 | 0.0119 | -0.0240 |

Note: The crisis period is 2007-09-01 to 2009-06-30. The calm period is 2012-01-01 to 2016-12-31. The table reports all crisis-period VaR levels and the 1% calm-period comparison to keep the subsample evidence compact.

The subsample evidence explains the overall results more precisely. In the crisis period, time-weighted HS and KDE time-weighted HS bring the 5% failure rate exactly to 5.00% and the 10% failure rate close to 10%. This is strong evidence that time weighting is effective for intermediate tails during regime changes. In the calm period, ordinary HS becomes conservative at 1%, while the two weighted methods are close to the nominal level. This pattern is consistent with Danielsson et al. (2016), who emphasize that fixed rolling windows can mix incompatible market states.

The analysis also involves multiple testing. For each model, there are three tail levels and three backtesting dimensions, which implies nine simultaneous hypotheses. A Bonferroni correction would use a per-test threshold of 0.05/9 = 0.0056. Under this stricter threshold, some marginal rejections would no longer be classified as significant. For example, the 1% Christoffersen p-value of 0.0218 for time-weighted HS and the 10% Kupiec p-value of 0.0379 for KDE time-weighted HS would not survive the correction. However, the main conclusions do not change: ordinary HS still fails several diagnostics strongly, time-weighted HS remains the best overall choice at 5% and 10%, and KDE remains a targeted improvement for 1% coverage rather than a universal replacement.

Figure 3.1 compares the 1% VaR forecasts for the three HS-family models at W = 1000. Figure 3.2 plots the corresponding 1% violation indicators. The figures support the table-based interpretation by showing whether violations occur as isolated events or in clusters.

![Figure 3.1. 1% VaR forecasts for HS-family models, W=1000](../outputs/figures/hs_1pct_var_comparison_w1000.png)

![Figure 3.2. 1% violation indicators for HS-family models, W=1000](../outputs/figures/hs_1pct_violation_indicators_w1000.png)

### 3.8 Model Choice

The evidence supports a tail-level-specific model choice. For 1% VaR, KDE time-weighted HS with W = 1000 is preferred within the HS family because it is the only specification that passes the Kupiec unconditional coverage test at the 5% level. This conclusion must be qualified because the Christoffersen independence p-value remains below 5%, and the finite-sample power calculation shows that 1% VaR backtests have limited ability to detect moderate misspecification.

For 5% and 10% VaR, time-weighted HS with $W = 1000$ and $\lambda = 0.98$ is preferred. It maintains failure rates close to the nominal levels and substantially improves independence relative to ordinary HS. This finding is consistent with Boudoukh, Richardson and Whitelaw (1998), who argue that weighting schemes can improve HS when volatility is persistent.

If a single HS-family model must be selected for operational use, the most defensible choice is time-weighted HS with $W = 1000$ and $\lambda = 0.98$. It is transparent, easy to implement, and robust at the 5% and 10% levels. The remaining weakness at 1% should be reported rather than hidden. A practical system could therefore use time-weighted HS as the main non-parametric benchmark and report KDE-smoothed 1% VaR as a sensitivity analysis.

The limitations also point to more advanced alternatives. Barone-Adesi, Giannopoulos and Vosper (1999) propose filtered historical simulation, which combines volatility filtering with empirical innovations and is a natural next step when ordinary HS fails in volatile regimes. Engle and Manganelli (2004) propose CAViaR models that directly model conditional quantiles and avoid estimating the full return distribution. McNeil and Frey (2000) combine GARCH volatility dynamics with extreme value theory, which is especially relevant for the 1% tail. These alternatives are beyond the scope of the present HS section, but they identify the correct direction for improving extreme-tail risk measurement.

### 3.9 Summary

This section shows that historical simulation should not be evaluated only by average failure rates. Ordinary HS is transparent but suffers from violation clustering and sparse-tail instability. Time-weighted HS improves the dynamic behavior of violations by giving more relevance to recent observations. KDE-smoothed time-weighted HS improves 1% unconditional coverage by replacing the discrete empirical distribution with a continuous density estimate, but it can distort intermediate quantiles when the bandwidth is not locally appropriate.

The preferred non-parametric specification depends on the VaR level. For 1% VaR, KDE time-weighted HS with $W = 1000$ is the best HS-family candidate, although independence remains imperfect. For 5% and 10% VaR, time-weighted HS with $W = 1000$ and $\lambda = 0.98$ is the most balanced model. This conclusion reflects a statistical tradeoff among tail sample size, responsiveness to volatility regimes, unconditional coverage, and violation independence. It also establishes a clear benchmark for the later parametric and machine-learning models in the project.

### References

Boudoukh, J., Richardson, M., and Whitelaw, R. (1998). The best of both worlds: A hybrid approach to calculating value at risk. Risk, 11(5), 64-67.

Barone-Adesi, G., Giannopoulos, K., and Vosper, L. (1999). VaR without correlations for nonlinear portfolios. Journal of Futures Markets, 19(5), 583-602.

Bowman, A. W., and Azzalini, A. (1997). Applied Smoothing Techniques for Data Analysis: The Kernel Approach with S-Plus Illustrations. Oxford University Press.

Christoffersen, P. F. (1998). Evaluating interval forecasts. International Economic Review, 39(4), 841-862.

Christoffersen, P. F., and Pelletier, D. (2004). Backtesting value-at-risk: A duration-based approach. Journal of Financial Econometrics, 2(1), 84-108.

Danielsson, J., Ergun, L. M., de Haan, L., and de Vries, C. G. (2016). Tail index estimation: Quantile driven threshold selection. LSE Systemic Risk Centre Discussion Paper.

Engle, R. F., and Manganelli, S. (2004). CAViaR: Conditional autoregressive value at risk by regression quantiles. Journal of Business & Economic Statistics, 22(4), 367-381.

Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. The Journal of Derivatives, 3(2), 73-84.

McNeil, A. J., and Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: An extreme value approach. Journal of Empirical Finance, 7(3-4), 271-300.

Pritsker, M. (2006). The hidden dangers of historical simulation. Journal of Banking & Finance, 30(2), 561-582.

Scott, D. W. (1979). On optimal and data-based histograms. Biometrika, 66(3), 605-610.

Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis. Chapman and Hall.

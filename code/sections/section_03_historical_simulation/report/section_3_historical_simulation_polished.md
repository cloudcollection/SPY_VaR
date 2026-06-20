## Section 3: Historical Simulation Methods

### 3.1 Motivation

Historical Simulation (HS) is used as the non-parametric benchmark in this project because it estimates Value-at-Risk directly from the empirical distribution of observed returns. Unlike parametric models, HS does not impose a Gaussian, Student-t, or other pre-specified distributional form on returns. This is important for equity-index returns, which typically exhibit skewness, excess kurtosis, volatility clustering, and extreme losses that are poorly represented by a normal distribution.

The second advantage of HS is interpretability. A HS VaR forecast can be traced to observed historical returns rather than to latent parameters or black-box model states. This makes it a useful benchmark for financial risk management, where model outputs must be explainable and auditable. In this project, HS also provides a disciplined reference point for evaluating the more structured GARCH model and the more flexible neural quantile model. If a complex model cannot improve on a transparent empirical quantile method, the additional modeling complexity is difficult to justify.

At the same time, ordinary HS has two well-known weaknesses. First, it assigns equal weight to all observations in the rolling window, even though financial volatility changes over time. A return from a calm market regime may not be as informative as a recent return from a turbulent regime. Second, ordinary HS estimates VaR from a discrete empirical distribution. At the 1% level, even a 1000-day window contains only about ten observations in the relevant tail. This makes the 1% VaR estimate sensitive to a small number of observations and can generate discontinuous jumps when an extreme return enters or leaves the rolling window. These limitations motivate the time-weighted and KDE-smoothed extensions below.

### 3.2 Rolling Window Design

All HS variants are implemented in a strict rolling-window forecasting framework. For a one-day-ahead forecast made at time \(t\), the model uses only the information set

$$
\mathcal{F}_t = \{r_{t-W+1}, r_{t-W+2}, \ldots, r_t\},
$$

where \(r_t\) denotes the SPY daily log return and \(W\) is the rolling window length. The forecast is then evaluated against the realized return \(r_{t+1}\). The window advances by one trading day and the procedure is repeated until the end of the sample. This design avoids look-ahead bias because \(r_{t+1}\) and all later observations are excluded from the estimation window.

Two window lengths were considered: \(W=500\) and \(W=1000\). The 500-day window is more adaptive to recent market conditions, while the 1000-day window provides a larger effective tail sample. The empirical comparison supports the use of \(W=1000\) for the HS family. At the 5% and 10% VaR levels, both windows perform reasonably well, but the 1000-day window produces more stable violation dynamics in the Christoffersen independence test. At the 1% level, the longer window is also statistically preferable because it contains roughly twice as many tail observations as the 500-day window.

### 3.3 Ordinary Historical Simulation

The ordinary HS estimator defines the one-day-ahead VaR forecast as the empirical lower-tail quantile of the rolling return sample:

$$
\widehat{\mathrm{VaR}}_{\alpha,t+1}^{HS}
= Q_{\alpha}\left(r_{t-W+1}, \ldots, r_t\right),
$$

where \(Q_{\alpha}(\cdot)\) denotes the empirical \(\alpha\)-quantile and \(\alpha \in \{0.01,0.05,0.10\}\). Equivalently, after sorting the rolling returns from smallest to largest, the VaR forecast is selected from the lower tail of this ordered sample. In ordinary HS, every observation receives weight \(1/W\).

The strength of this estimator is that it is fully non-parametric. It can capture skewness and heavy tails present in the historical sample without estimating a parametric density. Its weakness is that it treats all observations as equally relevant. This is restrictive for financial returns because volatility is persistent. When the recent volatility regime differs from the earlier part of the rolling window, the equal-weighted empirical quantile can respond too slowly.

### 3.4 Time-weighted Historical Simulation

To address the equal-weighting limitation, I implement an exponentially time-weighted HS estimator. This approach follows the age-weighted HS idea in Boudoukh, Richardson, and Whitelaw (1998): recent observations receive larger weights, while older observations receive progressively smaller weights. Let \(\text{age}_i\) denote the number of days between observation \(r_i\) and the forecast origin \(t\), so that the most recent return has \(\text{age}=0\). The unnormalized weight is

$$
\tilde{w}_i = \lambda^{\text{age}_i}, \qquad 0 < \lambda < 1.
$$

The normalized weight is

$$
w_i =
\frac{\lambda^{\text{age}_i}}
{\sum_{j=0}^{W-1}\lambda^j}.
$$

The time-weighted VaR is then the weighted quantile of the rolling sample:

$$
\widehat{\mathrm{VaR}}_{\alpha,t+1}^{TWHS}
= \inf \left\{
q:
\sum_{i=t-W+1}^{t} w_i \mathbf{1}(r_i \leq q) \geq \alpha
\right\}.
$$

This estimator preserves the non-parametric nature of HS but makes the model more responsive to changing volatility conditions. I compared several decay factors and found that \(\lambda=0.98\) gives the best balance in this dataset. A smaller value such as \(\lambda=0.94\) makes the model too reactive because most of the probability mass is concentrated on a very short recent period. In contrast, \(\lambda=0.98\) retains meaningful information from a broader part of the window while still emphasizing recent observations. In the final implementation, the recommended time-weighted HS specification uses \(W=1000\) and \(\lambda=0.98\).

### 3.5 KDE-smoothed Time-weighted Historical Simulation

Although time weighting improves the relevance of the historical sample, it does not solve the discreteness of the empirical tail. This limitation is most visible at the 1% VaR level. With \(W=1000\), the 1% empirical tail corresponds to approximately ten observations. As a result, the VaR estimate can be dominated by a few extreme returns. This creates a ghost effect: an old crisis observation may continue to determine VaR until it exits the window, after which the forecast can jump abruptly.

To smooth the discrete weighted empirical distribution, I implement a Gaussian kernel density estimator (KDE) on the time-weighted rolling sample. The weighted KDE is

$$
\hat{f}_t(x)
=
\frac{1}{h}
\sum_{i=t-W+1}^{t}
w_i
K\left(\frac{x-r_i}{h}\right),
$$

where \(w_i\) are the exponential time weights, \(h\) is the bandwidth, and the Gaussian kernel is

$$
K(u) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{u^2}{2}\right).
$$

The VaR forecast is obtained by inverting the KDE-implied cumulative distribution:

$$
\widehat{\mathrm{VaR}}_{\alpha,t+1}^{KDE}
=
\inf \left\{
x:
\int_{-\infty}^{x} \hat{f}_t(u)\,du \geq \alpha
\right\}.
$$

The KDE variant keeps the same time-weighting structure with \(W=1000\) and \(\lambda=0.98\), but replaces the discrete weighted quantile with a continuous probability model. This is especially useful for the 1% tail because it uses information from nearby tail observations rather than relying only on the exact order statistic. The bandwidth is selected by Scott's rule for computational stability in the rolling loop. This is a practical choice for a forecasting exercise, although adaptive bandwidth selection would be a natural extension for heavy-tailed financial returns.

### 3.6 Backtesting Framework

For each VaR forecast, I define the violation indicator

$$
I_{t+1} =
\mathbf{1}\left(r_{t+1} < \widehat{\mathrm{VaR}}_{\alpha,t+1}\right).
$$

A correctly specified VaR model should satisfy two conditions. First, the unconditional violation probability should equal \(\alpha\). Second, violations should not be serially clustered. If violations arrive in clusters, the model may be too slow to adjust during stress periods even when the average failure rate appears acceptable.

The failure rate is computed as

$$
\widehat{p} = \frac{1}{T}\sum_{t=1}^{T} I_t,
$$

where \(T\) is the number of out-of-sample forecasts. The Kupiec unconditional coverage test evaluates

$$
H_0: p = \alpha.
$$

Let \(V=\sum_t I_t\) be the number of violations. The likelihood-ratio statistic is

$$
LR_{uc}
=
-2\log
\left[
\frac{(1-\alpha)^{T-V}\alpha^V}
{(1-\widehat{p})^{T-V}\widehat{p}^{V}}
\right]
\sim \chi^2(1).
$$

The Christoffersen independence test evaluates whether the violation sequence is serially independent. Let \(T_{ij}\) denote the number of transitions from state \(i\) to state \(j\), where \(i,j \in \{0,1\}\). The test compares the restricted model with a common violation probability against a first-order Markov alternative with separate transition probabilities \(\pi_{01}\) and \(\pi_{11}\). A rejection indicates violation clustering.

I also include a duration-based test following the logic of Christoffersen and Pelletier (2004). Instead of using only one-step transition counts, the duration test examines the time gaps between violations. Under independence, violation durations should behave like a memoryless process. This test is particularly useful for the 1% VaR level because consecutive violations are rare, making \(T_{11}\) sparse and the Markov transition test less stable.

### 3.7 Empirical Comparison of HS Variants

The comparison between \(W=500\) and \(W=1000\) for time-weighted HS shows that the longer window is preferable overall. With \(\lambda=0.98\), the 1000-day window achieves acceptable Kupiec and Christoffersen p-values at the 5% and 10% levels. The 1% level remains difficult for both windows, which confirms that sparse tail information is the main limitation rather than the window length alone.

The KDE-smoothed model is most useful at the 1% level. The 1% failure rate decreases from 0.0162 under time-weighted HS to 0.0129 under KDE-smoothed time-weighted HS. More importantly, the Kupiec p-value increases from 0.0002 to 0.0911, so the 1% KDE VaR is no longer rejected by the unconditional coverage test at the 5% significance level. The duration p-value also improves from 0.0121 to 0.1258, suggesting that the KDE model reduces the clustering of extreme-tail violations.

However, KDE smoothing is not uniformly better. At the 10% level, the KDE model becomes too conservative: the failure rate falls to 0.0898 and the Kupiec p-value is 0.0379. This indicates that smoothing the entire distribution with a single bandwidth improves the far left tail but may distort intermediate tail regions. The 5% duration result also weakens under KDE, which suggests that a global Gaussian bandwidth may not be optimal for all quantile levels.

The final HS-family recommendation is therefore level-specific. For the 1% VaR, KDE-smoothed time-weighted HS is preferred because it materially improves tail coverage. For the 5% and 10% VaR levels, time-weighted HS with \(W=1000\) and \(\lambda=0.98\) is preferred because it provides more stable overall backtesting performance.

### 3.8 Summary

The HS analysis shows that simple non-parametric methods can be made substantially more informative through statistically motivated modifications. Ordinary HS provides a transparent baseline but ignores time variation in volatility and suffers from discrete-tail instability. Time-weighted HS addresses the first problem by assigning greater importance to recent observations. KDE-smoothed time-weighted HS addresses the second problem by converting the weighted empirical distribution into a continuous density estimate.

The empirical results support a clear interpretation. Time weighting improves the practical usefulness of HS at the 5% and 10% VaR levels, while KDE smoothing is valuable for the sparse 1% tail. This demonstrates both the strength and the limitation of non-parametric VaR forecasting: distributional assumptions can be avoided, but tail estimation still requires careful treatment of sample size, weighting, smoothing, and violation dynamics.

References: Boudoukh, Richardson, and Whitelaw (1998); Kupiec (1995); Christoffersen (1998); Christoffersen and Pelletier (2004).

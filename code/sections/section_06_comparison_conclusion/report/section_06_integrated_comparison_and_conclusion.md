## Section 6: Integrated Discussion and Final Model Assessment

### 6.1 Purpose of the Integrated Assessment

The preceding sections should be read as a single modelling argument rather than as a sequence of unrelated experiments. The empirical problem is one-day-ahead Value-at-Risk forecasting for SPY daily log returns, and the data show three features that any credible model must address: heavy tails, volatility clustering, and nonlinear state dependence. No single modelling family automatically satisfies all three requirements. Historical Simulation captures the empirical tail without imposing a distribution, but it is weak when market regimes change. GARCH-type models introduce an explicit conditional volatility structure, but the symmetric GARCH-t specification remains biased in unconditional coverage. Direct neural quantile regression is flexible, but without a structural anchor it performs poorly in the extreme tail under rolling-window estimation.

This section therefore evaluates the models through a unified thesis: accurate VaR forecasting requires the joint presence of empirical tail information, conditional volatility structure, and controlled nonlinear flexibility. The purpose is not to claim that a neural network dominates classical risk models in general. The evidence is more specific. Neural flexibility becomes useful only after the model is anchored to a statistically meaningful risk structure. In this project, the best empirical support for that claim comes from Model C2, the conservative GARCH-anchored neural quantile correction.

### 6.2 Evidence from the Data and Backtesting Framework

Section 2 establishes why a Gaussian benchmark is not sufficient for this application. The full-sample excess kurtosis of SPY daily log returns is approximately 8.21, far above the Gaussian benchmark of zero. The empirical lower-tail quantiles are also economically large: the 1%, 5%, and 10% daily log-return quantiles are approximately -3.45%, -1.88%, and -1.29%. These facts justify the use of methods that can handle heavy tails directly. They also explain why Student-t innovations are preferable to Gaussian innovations in the GARCH specification. The Student-t distribution adds a degrees-of-freedom parameter, and the rolling GARCH estimates in Section 4 produce a median degrees of freedom around 6.39, which is substantially heavier-tailed than the normal distribution.

The common backtesting framework is essential for interpreting the results. The Kupiec test evaluates unconditional coverage: whether the total number of violations is consistent with the nominal tail probability. The Christoffersen independence test evaluates whether violations are serially independent. A model can pass one test and fail the other. This distinction becomes central to the project. Historical Simulation can produce reasonable average failure rates at some quantile levels while still generating clustered violations. GARCH-t reduces clustering by adapting to volatility regimes, but it still produces too many violations in total. Model C2 is strongest in the unconditional-coverage dimension because it improves the violation-frequency problem while preserving much of the dynamic structure supplied by the GARCH anchor.

### 6.3 What Each Model Family Contributes

The Historical Simulation family provides the first benchmark because it is transparent and distribution-free. Ordinary Historical Simulation uses the empirical return distribution directly, so it avoids the false precision of a Gaussian parametric model. However, the equal-weighted rolling window reacts slowly when the market shifts from a calm regime into a stress regime. This is visible in the crisis-period evidence: ordinary Historical Simulation has a 1% crisis failure rate of 7.17%, far above the nominal rate. Time-weighted Historical Simulation improves this by assigning more weight to recent returns, and KDE-smoothed time-weighted Historical Simulation further improves the 1% unconditional coverage by smoothing the empirical tail. At W = 1000, KDE time-weighted Historical Simulation reaches a 1% failure rate of 1.29% with a Kupiec p-value of 0.0911, but its 1% Christoffersen p-value remains 0.0246. Thus, the HS family demonstrates that empirical tail information matters, but also that a non-parametric tail alone does not fully solve conditional risk dynamics.

GARCH-t addresses the conditional-risk weakness more directly. By modelling the conditional variance recursively and using Student-t innovations, it responds to volatility clustering and thick-tailed standardized shocks. The W = 1000 GARCH-t results show this clearly: the Christoffersen p-values at the 1%, 5%, and 10% levels are 0.3424, 0.8973, and 0.1363, respectively. These are much stronger than the ordinary HS independence results. However, the same model fails unconditional coverage. Its failure rates are 1.62%, 6.32%, and 11.48%, and all three Kupiec p-values are below conventional significance thresholds. The model therefore improves the timing of violations but leaves the VaR level too shallow on average.

The robustness extensions refine this interpretation. GJR-GARCH-t improves calibration by adding an asymmetric leverage channel: the 10% Kupiec p-value rises to 0.0719, and the 1% failure rate falls from 1.62% to 1.46%. The realized-measure GARCH-X models using rv5 and bv reduce volatility persistence, which confirms that intraday realized measures contain information about the latent volatility state. Nevertheless, these linear variance-side extensions do not fully fix VaR coverage. Their value is therefore mainly diagnostic: they show that realized measures should be retained as information variables, but that a linear GARCH-X recursion is too restrictive to be the final model.

Direct neural quantile regression tests the opposite extreme. Models A and B allow nonlinear conditional quantile functions, with Model B adding realized-volatility features. In principle, this flexibility should help capture interactions among lagged returns, realized volatility, and tail risk. In practice, both direct MLP specifications fail badly. Model A records a 1% failure rate of 7.19%, and Model B records a 1% failure rate of 6.72%. Both are far above the nominal 1% target. The reason is structural. With a 1000-day rolling window, the 1% tail contains only about 10 extreme observations inside each estimation sample. A neural network asked to learn both volatility dynamics and the absolute lower-tail level from this small number of events has too much freedom relative to the information available. The direct neural results therefore do not reject machine learning; they show that unconstrained flexibility is not enough for rare-event VaR forecasting.

### 6.4 Why Model C2 Is the Most Defensible Final Specification

Model C2 combines the useful parts of the previous models. The GARCH-t forecast supplies a structured baseline for conditional volatility and heavy-tailed innovations. The neural network then learns a correction term using lagged returns, realized volatility proxies, GARCH VaR forecasts, and the GARCH volatility forecast. The conservative softplus design restricts the correction so that it can only move the baseline VaR in a more conservative direction:

$$
\widehat{\mathrm{VaR}}^{C2}_{\alpha,t+1}
= \widehat{\mathrm{VaR}}^{GARCH}_{\alpha,t+1}
- \operatorname{softplus}(g_{\theta,\alpha}(X_t)).
$$

This restriction is not an ad hoc repair of a failed neural model. It encodes a prior learned from Section 4. GARCH-t produces too many violations across all alpha levels and rolling windows, which means that the absolute VaR level is systematically too shallow. The softplus correction turns this empirical bias into a modelling constraint: the network only needs to learn the magnitude of the additional conservatism, not its direction. In a limited rolling sample, this is a deliberate bias-variance tradeoff. It reduces estimation variance by narrowing the function class to corrections that are economically and statistically consistent with the GARCH evidence.

The empirical results support this design choice in the coverage dimension. Model C2 records 25 1% violations from 2640 forecasts, compared with 26.4 expected violations, for a failure rate of 0.95%. At the 5% level, it records 131 violations against 132 expected, for a failure rate of 4.96%. At the 10% level, it records 235 violations against 264 expected, for a failure rate of 8.90%. The Kupiec p-values are 0.7823, 0.9288, and 0.0557. These values are substantially stronger than the direct MLP models and, on the C2-aligned window, closer to nominal coverage than the GARCH-family baselines. This is a coverage result rather than a blanket loss-dominance result.

The independence diagnostics are more mixed. Model C2 has Christoffersen p-values of 0.0217, 0.5196, and 0.4737 at the 1%, 5%, and 10% levels. The 5% and 10% results are satisfactory, but the 1% result still indicates residual clustering of extreme violations. This caveat matters. The appropriate conclusion is that Model C2 is the best model in this project for unconditional VaR calibration, not that it completely solves all dynamic backtesting diagnostics. The remaining 1% clustering suggests that extreme-tail dynamics may require a richer quantile process, a stronger regime component, or post-training calibration.

### 6.5 Cross-model Comparison

The comparison must be made on a common evaluation window. The original Section 3 and Section 4 W = 1000 benchmarks begin in 2004 and contain 3640 forecast days, whereas Model C2 begins on 2008-01-03 and contains 2640 aligned forecast days. Directly placing these rows in one table gives C2 a different market sample and excludes the relatively calm 2004-2007 period from the neural comparison. Table 6.1 therefore recomputes the Historical Simulation and GARCH-family benchmarks on exactly the C2 forecast dates. This is the apple-to-apple comparison that should carry the main empirical conclusion.

Table 6.1. C2-aligned cross-model backtesting comparison, 2008-01-03 onward.

| Model | N | FR 1% | Kupiec 1% | Christ. 1% | FR 5% | Kupiec 5% | Christ. 5% | FR 10% | Kupiec 10% | Christ. 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordinary HS | 2640 | 0.0155 | 0.0082 | 0.0003 | 0.0530 | 0.4791 | 0.0012 | 0.0955 | 0.4331 | 0.0000 |
| Time-weighted HS | 2640 | 0.0155 | 0.0082 | 0.0275 | 0.0515 | 0.7222 | 0.6988 | 0.1015 | 0.7957 | 0.7059 |
| KDE Time-weighted HS | 2640 | 0.0136 | 0.0752 | 0.0127 | 0.0451 | 0.2381 | 0.2640 | 0.0883 | 0.0406 | 0.4171 |
| GARCH-t | 2640 | 0.0155 | 0.0082 | 0.1632 | 0.0655 | 0.0005 | 0.2654 | 0.1148 | 0.0132 | 0.0535 |
| GJR-GARCH-t | 2640 | 0.0144 | 0.0333 | 0.5768 | 0.0659 | 0.0003 | 0.2489 | 0.1121 | 0.0413 | 0.0214 |
| GARCH-X-rv5-t | 2640 | 0.0182 | 0.0002 | 0.2907 | 0.0697 | 0.0000 | 0.5749 | 0.1152 | 0.0111 | 0.1139 |
| GARCH-X-bv-t | 2640 | 0.0197 | 0.0000 | 0.9801 | 0.0667 | 0.0002 | 0.5783 | 0.1129 | 0.0303 | 0.4717 |
| Model C2 | 2640 | 0.0095 | 0.7823 | 0.0217 | 0.0496 | 0.9288 | 0.5196 | 0.0890 | 0.0557 | 0.4737 |

The aligned table gives a narrower and more defensible conclusion than the unaligned comparison. Model C2 is closest to nominal unconditional coverage at all three alpha levels, and it has the strongest Kupiec p-values. However, its 1% Christoffersen p-value remains 0.0217. Under a conventional single-test 5% rule, this still signals residual clustering. Under a Bonferroni threshold of approximately 0.0056, it is not rejected. The conclusion is therefore not that C2 fully solves independence, but that its independence weakness is borderline while its unconditional coverage is clearly stronger.

Coverage is not the same as forecast-loss dominance. Table 6.2 reports Diebold-Mariano tests using paired pinball-loss series on the same C2-aligned dates. The statistic is based on the loss difference C2 minus benchmark, so a negative value favours C2.

Table 6.2. Diebold-Mariano tests for paired pinball loss, C2 versus benchmarks.

| Benchmark | DM p 1% | DM p 5% | DM p 10% | Interpretation |
|---|---:|---:|---:|---|
| Ordinary HS | 0.0000 | 0.0000 | 0.0000 | C2 lower loss |
| Time-weighted HS | 0.0001 | 0.0228 | 0.0918 | C2 lower at 1% and 5%, not 10% |
| KDE Time-weighted HS | 0.0001 | 0.0087 | 0.1076 | C2 lower at 1% and 5%, not 10% |
| GARCH-t | 0.3558 | 0.1949 | 0.9950 | No significant loss difference |
| GJR-GARCH-t | 0.1032 | 0.5023 | 0.2072 | No significant loss difference |
| GARCH-X-rv5-t | 0.2706 | 0.0374 | 0.0226 | C2 not uniformly lower; benchmark has lower 5% and 10% loss |
| GARCH-X-bv-t | 0.0451 | 0.0055 | 0.0020 | C2 not uniformly lower; benchmark has lower loss |

These tests discipline the interpretation. C2 materially improves unconditional coverage, but the improvement is not a formal pinball-loss dominance result over the GARCH-family benchmarks. Relative to GARCH-t and GJR-GARCH-t, the paired loss differences are statistically indistinguishable at all three alpha levels. Relative to GARCH-X, C2 has better coverage but higher pinball loss at several quantiles. The correct claim is therefore: C2 is the best-calibrated coverage model in this project, not the uniformly best scoring model.

Table 6.3 reports Lopez loss on the same common window after converting return differences to percentage points before squaring the exceedance. This avoids the near-equivalence between decimal-scale Lopez loss and failure rate.

Table 6.3. Lopez loss on percent-return scale, C2-aligned window.

| Model | 1% | 5% | 10% |
|---|---:|---:|---:|
| Ordinary HS | 0.0758 | 0.2450 | 0.3928 |
| Time-weighted HS | 0.0301 | 0.1160 | 0.2438 |
| KDE Time-weighted HS | 0.0250 | 0.1027 | 0.2086 |
| GARCH-t | 0.0239 | 0.1285 | 0.2496 |
| GJR-GARCH-t | 0.0209 | 0.1180 | 0.2295 |
| GARCH-X-rv5-t | 0.0248 | 0.1181 | 0.2267 |
| GARCH-X-bv-t | 0.0259 | 0.1135 | 0.2216 |
| Model C2 | 0.0147 | 0.0904 | 0.1798 |

![Figure 6.1. Failure-rate comparison across VaR model families on the C2-aligned window.](../outputs/figures/failure_rate_comparison.png)

![Figure 6.2. Percent-scale Lopez loss comparison across VaR model families on the C2-aligned window.](../outputs/figures/lopez_loss_comparison.png)

The comparison shows why the final narrative should not be framed as a sequence of failed attempts. Each stage identifies one necessary component of a reliable VaR model. Historical Simulation establishes the importance of the empirical tail. Time weighting and GARCH establish the importance of conditional dynamics. Direct neural quantile regression establishes the risk of using flexible nonlinear models without enough tail observations. Model C2 is the synthesis: it keeps the conditional structure of GARCH, uses realized-measure and lagged-return information in a nonlinear correction, and imposes a conservative direction consistent with the previous coverage bias.

The empirical answer to the thesis is therefore nuanced. Model C2 comes closest to satisfying the coverage requirement, but it does not satisfy every diagnostic perfectly. It inherits thick-tail and conditional-volatility information from the GARCH-t anchor, it uses rv5 and bv as state variables for nonlinear correction, and it controls finite-sample variance through the one-sided softplus restriction. However, the 1% Christoffersen p-value remains below the unadjusted 5% threshold, and the DM tests do not show pinball-loss dominance over the GARCH-family benchmarks. The proper conclusion is not full dominance, but disciplined improvement: Model C2 materially improves unconditional calibration while leaving clear targets for future dynamic-tail modelling and pre-specified model selection.

### 6.6 Limitations

Several limitations should be reported explicitly. First, the extreme-tail sample size is small by construction. Even with a 1000-day rolling window, the 1% quantile is informed by roughly 10 observations in each training window. This makes all 1% VaR comparisons statistically fragile, especially for flexible models. Second, the neural-network hyperparameters are not the result of a fully pre-specified tuning protocol. In particular, weight decay was selected after inspecting the final out-of-sample failure-rate table, so the C2 p-values are conditional on this ex post choice and should not be presented as untouched test-sample evidence. Third, epoch-level training and validation losses were not persisted by the original rolling scripts, which limits reproducibility of the convergence claim even though early stopping was used. Fourth, Model C2 improves unconditional coverage partly by imposing one-sided conservatism. This is defensible because the GARCH evidence points to systematic underestimation, but it may become too conservative in a different market sample or asset class. Fifth, all models are evaluated on SPY daily returns only. The conclusions should therefore be interpreted as evidence for this data set rather than a universal ranking of VaR methods.

### 6.7 Future Work

There are several natural extensions. First, CAViaR models directly specify the dynamics of the VaR process without first estimating a conditional variance equation. This would provide a useful benchmark between GARCH and neural quantile regression. Second, Realized GARCH would integrate rv5 and bv into a joint model for returns and realized measures, rather than using realized measures only in a simplified variance-side GARCH-X recursion or as neural-network inputs. Third, the neural architecture could be extended beyond a feed-forward MLP. Sequence models such as LSTM or temporal convolutional networks may capture persistent state dependence more effectively, although they would also increase the risk of overfitting in the 1% tail. Fourth, post-training calibration methods could be used to adjust neural VaR levels while preserving the shape of the learned conditional quantile function.

### 6.8 Final Conclusion

The central conclusion of the project is that VaR forecasting accuracy depends on combining structure with flexibility. A model that only uses the empirical distribution reacts too slowly to volatility regimes. A model that only uses GARCH structure improves timing but can still be biased in coverage. A model that only uses neural flexibility can collapse in the extreme tail because the rolling training sample contains too few rare events. The best-performing specification in this project is therefore not the most complex model in isolation, but the model with the most coherent allocation of tasks: GARCH-t supplies the conditional risk anchor, realized measures and lagged returns provide state information, and the neural network learns a constrained conservative correction.

Model C2 is the most defensible final specification for unconditional coverage under the C2-aligned backtesting framework. It achieves near-nominal unconditional coverage at the 1% and 5% levels and remains statistically reasonable at the 10% level. Its remaining weaknesses are the borderline 1% independence diagnostic, the ex post regularization choice, and the absence of formal pinball-loss dominance over GARCH-family benchmarks. The final interpretation is consequently balanced: the conservative GARCH-anchored neural correction materially improves VaR calibration, but it should be presented as a structured coverage improvement over classical benchmarks rather than as a complete or uniformly dominant solution to extreme-tail risk forecasting.

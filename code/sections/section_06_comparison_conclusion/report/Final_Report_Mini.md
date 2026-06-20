# Value-at-Risk Forecasting Mini Program for SPY Daily Returns

    ## 1. Introduction and Task

    This project forecasts one-day-ahead Value-at-Risk (VaR) for SPY daily log
    returns at the 1%, 5%, and 10% lower-tail levels. The final project is
    organized as a clean mini program rather than a claim of a new model
    contribution. The main workflow connects data exploration, rolling-window
    VaR estimation, statistical backtesting, and model comparison.

    The retained main models are Historical Simulation, KDE-weighted Historical
    Simulation, GARCH-t, GJR-GARCH-t, and direct MLP quantile regression.

    ## 2. Data and Forecasting Framework

    The data set contains 4,640 daily observations for SPY from 2000-01-04 to
    2018-06-27. The target variable is daily log return. The evidence shows
    heavy tails and volatility clustering, which motivates both empirical-tail
    methods and conditional-volatility models.

    VaR is treated as a lower conditional quantile. For tail probability alpha,
    a correct VaR forecast should be violated approximately alpha of the time.
    The final comparison uses 3640 common forecast dates across all retained
    main models, so the table is an apples-to-apples comparison.

    ## 3. Model Design

    Historical Simulation is transparent and distribution-free, but it reacts
    slowly when volatility regimes change. KDE-weighted Historical Simulation
    smooths the empirical tail and gives more importance to recent returns.
    GARCH-t and GJR-GARCH-t provide interpretable time-varying volatility
    benchmarks with Student-t innovations. The MLP quantile-regression model is
    included as a neural benchmark trained with pinball loss, not as an assumed
    winner.

    ## 4. Main Backtesting Results

    The aligned failure rates are:

    - GARCH(1,1)-t: 0.0168, 0.0651, 0.1187 at the 1%, 5%, and 10% VaR levels.
- GJR-GARCH(1,1)-t: 0.0146, 0.0654, 0.1115 at the 1%, 5%, and 10% VaR levels.
- Gaussian KDE Weighted HS: 0.0129, 0.0459, 0.0898 at the 1%, 5%, and 10% VaR levels.
- MLP Quantile: 0.0047, 0.0332, 0.2115 at the 1%, 5%, and 10% VaR levels.
- Weighted Historical Simulation: 0.0168, 0.0530, 0.1019 at the 1%, 5%, and 10% VaR levels.

    The lowest pinball losses on the aligned sample are: 1% VaR -
    GJR-GARCH(1,1)-t (0.000305); 5% VaR - GJR-GARCH(1,1)-t (0.001150); 10% VaR - GJR-GARCH(1,1)-t (0.001874).
    These loss results should be interpreted together with coverage and
    independence diagnostics.

    ## 5. Comparative Interpretation

    The results support a conservative interpretation. Historical methods are
    easy to audit but can struggle during regime changes. GARCH-type models add
    useful volatility structure but do not perfectly solve unconditional
    coverage. The direct MLP benchmark is flexible, but its tail behaviour is
    unstable under limited rolling-window tail observations. Therefore, the
    main contribution of the mini program is a reproducible and statistically
    disciplined VaR comparison, not a claim that neural methods automatically
    dominate classical risk models.

    ## 6. Appendix: Exploratory Model C Extension

    I also preserve an exploratory Model C extension in the archive and discuss
    it here as technical evidence, not as the main result. Model C uses a
    GARCH-based risk anchor and a neural correction layer. It is interesting
    because it shows how a neural model can be constrained by financial risk
    structure rather than asked to learn the full VaR level from scratch.

    This extension should be read with two caveats. First, its evaluation sample
    is shorter than the main aligned comparison. Second, some regularization
    choices were selected after inspecting out-of-sample failure-rate behaviour.
    For that reason, Model C is presented as an exploratory extension and not
    as the primary ranking basis.

    ## 7. Conclusion

    This unified project demonstrates statistical modelling, Python
    implementation, financial risk understanding, and critical comparison. The
    final submission is deliberately concise and reproducible, while the archive
    preserves the deeper experimental work for reference.

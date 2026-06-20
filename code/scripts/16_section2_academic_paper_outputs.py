from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller

from utils import PROJECT_ROOT, REPORT_DIR, load_spy_data


OUT_DIR = PROJECT_ROOT / "outputs" / "section2_academic"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
PDF_DIR = PROJECT_ROOT / "outputs" / "pdf"
DOWNLOADS = Path.home() / "Downloads"

EN_MD = REPORT_DIR / "section_2_data_features_var_backtesting_framework_en.md"
ZH_MD = REPORT_DIR / "section_2_data_features_var_backtesting_framework_zh_preview.md"
CAPTIONS_EN = REPORT_DIR / "section_2_academic_figure_captions_en.md"
CAPTIONS_ZH = REPORT_DIR / "section_2_academic_figure_captions_zh.md"
PDF_SOURCE = REPORT_DIR / "section_2_data_features_var_backtesting_framework_pdf_source_en.md"
PDF_OUTPUT = PDF_DIR / "section_2_data_features_var_backtesting_framework_en.pdf"


BLUE = "#1F4E79"
ORANGE = "#B35C1E"
GREEN = "#2D6A4F"
PURPLE = "#6A4C93"
GRAY = "#4D4D4D"
LIGHT_GRAY = "#D9DDE7"
VERY_LIGHT = "#F7F7F7"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def set_academic_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.7,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def format_percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    formatter = mticker.PercentFormatter(1.0, decimals=0)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def add_zero_line(ax: plt.Axes) -> None:
    ax.axhline(0, color="#333333", linewidth=0.6)


def annualized_vol(x: pd.Series) -> pd.Series:
    return np.sqrt(x.clip(lower=0) * 252.0)


def compute_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = df["log_ret"].dropna()
    rows = []
    for col in ["log_ret", "rv5", "bv"]:
        x = df[col].dropna()
        rows.append(
            {
                "variable": col,
                "n": int(x.size),
                "mean": x.mean(),
                "sd": x.std(),
                "min": x.min(),
                "p01": x.quantile(0.01),
                "p05": x.quantile(0.05),
                "p10": x.quantile(0.10),
                "median": x.median(),
                "p90": x.quantile(0.90),
                "p95": x.quantile(0.95),
                "p99": x.quantile(0.99),
                "max": x.max(),
                "skewness": stats.skew(x, bias=False),
                "excess_kurtosis": stats.kurtosis(x, fisher=True, bias=False),
            }
        )
    desc = pd.DataFrame(rows)

    jb = stats.jarque_bera(r)
    tests = pd.DataFrame(
        [
            {
                "test": "Jarque-Bera",
                "series": "log_ret",
                "statistic": float(jb.statistic),
                "p_value": float(jb.pvalue),
            },
            {
                "test": "ADF",
                "series": "log_ret",
                "statistic": float(adfuller(r, autolag="AIC")[0]),
                "p_value": float(adfuller(r, autolag="AIC")[1]),
            },
            {
                "test": "ADF",
                "series": "squared_log_ret",
                "statistic": float(adfuller(r**2, autolag="AIC")[0]),
                "p_value": float(adfuller(r**2, autolag="AIC")[1]),
            },
        ]
    )

    window_design = pd.DataFrame(
        [
            {
                "window": w,
                "forecast_start": df.index[w].strftime("%Y-%m-%d"),
                "forecast_end": df.index[-1].strftime("%Y-%m-%d"),
                "out_of_sample_n": len(df) - w,
                "expected_1pct": (len(df) - w) * 0.01,
                "expected_5pct": (len(df) - w) * 0.05,
                "expected_10pct": (len(df) - w) * 0.10,
            }
            for w in [250, 500, 1000]
        ]
    )

    regime_df = df.copy()
    regime_df["volatility_regime"] = pd.qcut(
        regime_df["rv5"],
        q=[0, 1 / 3, 2 / 3, 1],
        labels=["Low", "Medium", "High"],
    )
    regime = (
        regime_df.groupby("volatility_regime", observed=False)
        .agg(
            n=("log_ret", "size"),
            mean_return=("log_ret", "mean"),
            sd_return=("log_ret", "std"),
            p01_return=("log_ret", lambda x: x.quantile(0.01)),
            p05_return=("log_ret", lambda x: x.quantile(0.05)),
            mean_rv5=("rv5", "mean"),
            mean_bv=("bv", "mean"),
        )
        .reset_index()
    )

    corr = df[["log_ret", "rv5", "bv"]].corr()

    tables = {
        "descriptive_statistics": desc,
        "distribution_tests": tests,
        "window_design": window_design,
        "volatility_regime_summary": regime,
        "correlation_matrix": corr,
    }
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=name == "correlation_matrix")
    return tables


def fig_returns_and_volatility(df: pd.DataFrame) -> None:
    r = df["log_ret"]
    q01 = r.quantile(0.01)
    rolling_vol = r.rolling(60).std() * math.sqrt(252)

    fig, axes = plt.subplots(2, 1, figsize=(6.85, 4.6), sharex=True)
    axes[0].plot(df.index, r, color=BLUE, linewidth=0.45)
    axes[0].scatter(df.index[r <= q01], r[r <= q01], s=8, color=ORANGE, zorder=3)
    axes[0].axhline(q01, color=ORANGE, linewidth=0.7, linestyle="--")
    add_zero_line(axes[0])
    axes[0].set_ylabel("Log return")
    format_percent_axis(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].plot(df.index, rolling_vol, color=GRAY, linewidth=0.65)
    axes[1].fill_between(df.index, 0, rolling_vol, color=LIGHT_GRAY, alpha=0.45)
    axes[1].set_ylabel("Annualized volatility")
    format_percent_axis(axes[1])
    format_date_axis(axes[1])
    panel_label(axes[1], "(b)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.8)

    savefig(fig, "fig2_1_returns_and_rolling_volatility")


def fig_distribution_and_qq(df: pd.DataFrame) -> None:
    r = df["log_ret"].dropna()
    mu, sigma = r.mean(), r.std()
    grid = np.linspace(r.quantile(0.001), r.quantile(0.999), 600)
    normal_pdf = stats.norm.pdf(grid, loc=mu, scale=sigma)

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9))
    axes[0].hist(r, bins=85, density=True, color="#E8EEF7", edgecolor=BLUE, linewidth=0.35)
    axes[0].plot(grid, normal_pdf, color=GRAY, linewidth=0.8)
    for q, color in [(0.01, ORANGE), (0.05, GREEN), (0.10, PURPLE)]:
        axes[0].axvline(r.quantile(q), color=color, linewidth=0.7, linestyle="--")
    axes[0].set_xlabel("Log return")
    axes[0].set_ylabel("Density")
    format_percent_axis(axes[0], "x")
    panel_label(axes[0], "(a)")

    sorted_r = np.sort(r.to_numpy())
    probs = (np.arange(1, len(sorted_r) + 1) - 0.5) / len(sorted_r)
    theo = stats.norm.ppf(probs, loc=mu, scale=sigma)
    tail = probs <= 0.05
    axes[1].scatter(theo[~tail], sorted_r[~tail], s=5, color=BLUE, alpha=0.35, edgecolor="none")
    axes[1].scatter(theo[tail], sorted_r[tail], s=6, color=ORANGE, alpha=0.75, edgecolor="none")
    lo = min(theo.min(), sorted_r.min())
    hi = max(theo.max(), sorted_r.max())
    axes[1].plot([lo, hi], [lo, hi], color=GRAY, linewidth=0.7, linestyle="--")
    axes[1].set_xlabel("Normal quantile")
    axes[1].set_ylabel("Empirical quantile")
    format_percent_axis(axes[1], "x")
    format_percent_axis(axes[1], "y")
    panel_label(axes[1], "(b)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=LIGHT_GRAY, linewidth=0.4, alpha=0.7)

    savefig(fig, "fig2_2_distribution_and_qq")


def fig_autocorrelation(df: pd.DataFrame) -> None:
    r = df["log_ret"].dropna()
    series = {
        "Returns": r,
        "Squared returns": r**2,
        "Absolute returns": r.abs(),
    }
    lags = np.arange(1, 41)
    conf = 1.96 / math.sqrt(len(r))

    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.55), sharey=True)
    for ax, (label, values), color, panel in zip(
        axes,
        series.items(),
        [BLUE, ORANGE, GREEN],
        ["(a)", "(b)", "(c)"],
    ):
        values_acf = acf(values, nlags=40, fft=True)[1:]
        ax.axhspan(-conf, conf, color=VERY_LIGHT, zorder=0)
        ax.axhline(0, color=GRAY, linewidth=0.6)
        ax.vlines(lags, 0, values_acf, color=color, linewidth=0.65)
        ax.set_title(label, fontsize=8.5, pad=3)
        ax.set_xlabel("Lag")
        panel_label(ax, panel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.35, alpha=0.7)
    axes[0].set_ylabel("ACF")
    savefig(fig, "fig2_3_autocorrelation_diagnostics")


def fig_volatility_proxies(df: pd.DataFrame) -> None:
    rv = annualized_vol(df["rv5"]).rolling(5).mean()
    bv = annualized_vol(df["bv"]).rolling(5).mean()
    trimmed = pd.DataFrame({"rv5": annualized_vol(df["rv5"]), "bv": annualized_vol(df["bv"])}).dropna()
    x_max = max(trimmed["rv5"].quantile(0.995), trimmed["bv"].quantile(0.995))

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9))
    axes[0].plot(df.index, rv, color=BLUE, linewidth=0.6, label="rv5")
    axes[0].plot(df.index, bv, color=ORANGE, linewidth=0.6, linestyle="--", label="bv")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("Annualized volatility")
    format_percent_axis(axes[0])
    format_date_axis(axes[0])
    panel_label(axes[0], "(a)")

    axes[1].hexbin(trimmed["bv"], trimmed["rv5"], gridsize=34, mincnt=1, cmap="Greys")
    axes[1].plot([0, x_max], [0, x_max], color=ORANGE, linewidth=0.7, linestyle="--")
    axes[1].set_xlim(0, x_max)
    axes[1].set_ylim(0, x_max)
    axes[1].set_xlabel("Bipower variation")
    axes[1].set_ylabel("Realized volatility")
    format_percent_axis(axes[1], "x")
    format_percent_axis(axes[1], "y")
    panel_label(axes[1], "(b)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=LIGHT_GRAY, linewidth=0.35, alpha=0.7)
    savefig(fig, "fig2_4_realized_volatility_proxies")


def fig_regime_and_windows(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    regime_df = df.copy()
    regime_df["Volatility regime"] = pd.qcut(regime_df["rv5"], q=[0, 1 / 3, 2 / 3, 1], labels=["Low", "Medium", "High"])
    data = [regime_df.loc[regime_df["Volatility regime"] == label, "log_ret"].dropna() for label in ["Low", "Medium", "High"]]
    window = tables["window_design"]

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.9))
    box = axes[0].boxplot(
        data,
        tick_labels=["Low", "Medium", "High"],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        boxprops={"linewidth": 0.7},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
    )
    for patch, color in zip(box["boxes"], ["#DCEAF7", "#EAEAEA", "#F4D8C8"]):
        patch.set_facecolor(color)
    axes[0].set_xlabel("Volatility regime")
    axes[0].set_ylabel("Log return")
    format_percent_axis(axes[0])
    panel_label(axes[0], "(a)")

    x = np.arange(len(window))
    axes[1].bar(x, window["out_of_sample_n"], color="#E6E6E6", edgecolor=GRAY, linewidth=0.7)
    axes[1].set_xticks(x, [f"W={int(w)}" for w in window["window"]])
    axes[1].set_ylabel("Out-of-sample forecasts")
    for xi, value in zip(x, window["out_of_sample_n"]):
        axes[1].text(xi, value + 60, f"{int(value):,}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylim(0, window["out_of_sample_n"].max() * 1.16)
    panel_label(axes[1], "(b)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.35, alpha=0.7)
    savefig(fig, "fig2_5_volatility_regimes_and_windows")


def generate_figures(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    fig_returns_and_volatility(df)
    fig_distribution_and_qq(df)
    fig_autocorrelation(df)
    fig_volatility_proxies(df)
    fig_regime_and_windows(df, tables)


def write_captions() -> None:
    en = """# Section 2 academic figure captions

Figure 2.1. SPY daily returns and rolling volatility. Panel (a) plots daily log returns and marks observations below the empirical 1% quantile. Panel (b) reports the 60-trading-day annualized rolling volatility.

Figure 2.2. Empirical distribution and normal Q-Q plot of SPY daily log returns. Panel (a) compares the empirical density with a fitted Gaussian density and marks the empirical 1%, 5%, and 10% lower-tail quantiles. Panel (b) compares empirical quantiles with the fitted Gaussian benchmark.

Figure 2.3. Autocorrelation diagnostics. Panels (a), (b), and (c) report autocorrelation functions for raw returns, squared returns, and absolute returns up to 40 trading-day lags.

Figure 2.4. Realized volatility proxies. Panel (a) plots annualized rv5 and bv after five-day smoothing. Panel (b) reports the joint distribution of annualized rv5 and bv using a hexbin density plot.

Figure 2.5. Volatility regimes and rolling-window design. Panel (a) compares return distributions across rv5-based volatility regimes. Panel (b) reports the out-of-sample forecast counts for W = 250, 500, and 1000.
"""
    zh = """# 第二章学术图表标题与说明

图2.1 SPY日收益率与滚动波动率。图(a)报告日对数收益率，并标记低于经验1%分位数的观测值；图(b)报告60个交易日滚动年化波动率。

图2.2 SPY日对数收益率的经验分布与正态Q-Q图。图(a)比较经验密度和拟合正态密度，并标记经验1%、5%和10%下尾分位数；图(b)比较样本经验分位数与正态分布基准。

图2.3 自相关诊断。图(a)、图(b)和图(c)分别报告原始收益率、平方收益率和绝对收益率在1至40阶滞后的自相关函数。

图2.4 已实现波动率代理变量。图(a)报告五日平滑后的年化rv5和bv；图(b)使用hexbin密度图展示年化rv5和bv的联合分布。

图2.5 波动状态与滚动窗口设计。图(a)比较基于rv5划分的不同波动状态下的收益率分布；图(b)报告W=250、500和1000时的样本外预测数量。
"""
    CAPTIONS_EN.write_text(en, encoding="utf-8")
    CAPTIONS_ZH.write_text(zh, encoding="utf-8")


def write_chapters(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    desc = tables["descriptive_statistics"].set_index("variable")
    tests = tables["distribution_tests"].set_index(["test", "series"])
    regime = tables["volatility_regime_summary"].set_index("volatility_regime")
    corr = tables["correlation_matrix"]
    window = tables["window_design"].set_index("window")

    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    n = len(df)
    mean_ret = desc.loc["log_ret", "mean"]
    sd_ret = desc.loc["log_ret", "sd"]
    skew = desc.loc["log_ret", "skewness"]
    kurt = desc.loc["log_ret", "excess_kurtosis"]
    jb_p = tests.loc[("Jarque-Bera", "log_ret"), "p_value"]
    q01 = desc.loc["log_ret", "p01"]
    q05 = desc.loc["log_ret", "p05"]
    q10 = desc.loc["log_ret", "p10"]
    low_p05 = regime.loc["Low", "p05_return"]
    high_p05 = regime.loc["High", "p05_return"]
    rv_bv_corr = corr.loc["rv5", "bv"]

    en = f"""# Section 2. Data Characteristics and the VaR Backtesting Framework

## 2.1 Data and variables

This section documents the empirical data, motivates the model choices used in the following chapters, and defines a common out-of-sample VaR backtesting design. The dataset contains {n:,} daily observations for SPY from {start} to {end}. The target variable is the daily log return, denoted by `log_ret`. The dataset also contains two high-frequency volatility measures, `rv5` and `bv`, which are used to describe market conditions and to provide volatility-related inputs for the neural quantile model in Section 5.

The role of this section is not merely descriptive. The distributional and time-series properties of the data determine whether the later modelling choices are justified. Heavy tails motivate non-Gaussian and nonparametric VaR methods. Volatility clustering motivates conditional-variance models. The availability of realized-volatility proxies motivates the inclusion of nonlinear machine-learning models with volatility features.

## 2.2 Distributional properties of returns

The sample mean of `log_ret` is {mean_ret:.6f}, while the sample standard deviation is {sd_ret:.6f}. The return distribution is not close to Gaussian: the sample skewness is {skew:.3f}, and the excess kurtosis is {kurt:.3f}. The Jarque-Bera normality test has a p-value of {jb_p:.4g}, rejecting normality at conventional levels. This evidence is directly relevant for VaR because a Gaussian benchmark may understate the probability of extreme losses.

Figure 2.1(a) shows that large negative returns are clustered in stress periods rather than being evenly distributed through time. Figure 2.1(b) reports the corresponding rolling volatility, which rises sharply in the same stress episodes.

![Figure 2.1. SPY daily returns and rolling volatility.](../outputs/section2_academic/figures/fig2_1_returns_and_rolling_volatility.png)

Figure 2.2 provides the distributional evidence more directly. The empirical lower-tail quantiles at the 1%, 5%, and 10% levels are {q01:.6f}, {q05:.6f}, and {q10:.6f}, respectively. These empirical quantiles provide the basic nonparametric benchmark for Historical Simulation in Section 3, but they also show why tail estimation is difficult, especially at the 1% level.

![Figure 2.2. Empirical distribution and normal Q-Q plot of SPY daily log returns.](../outputs/section2_academic/figures/fig2_2_distribution_and_qq.png)

## 2.3 Volatility clustering and volatility proxies

The rolling-volatility evidence in Figure 2.1(b) shows that volatility is clearly state-dependent: calm periods alternate with stress periods, and elevated volatility tends to persist. This pattern motivates the GARCH family in Section 4, because a static unconditional quantile cannot fully account for changing conditional risk.

Figure 2.4 examines `rv5` and `bv`. The correlation between the two volatility proxies is {rv_bv_corr:.3f}, indicating that they contain closely related information about market volatility. However, the two measures are not identical, especially during extreme trading days. This provides a reason to retain both variables as inputs for the neural quantile model rather than relying only on lagged returns.

![Figure 2.4. Realized volatility proxies.](../outputs/section2_academic/figures/fig2_4_realized_volatility_proxies.png)

## 2.4 Autocorrelation evidence and modelling implications

Figure 2.3 compares the autocorrelation functions of returns, squared returns, and absolute returns. Raw returns show weak serial dependence, suggesting limited predictability in the conditional mean. In contrast, squared and absolute returns display stronger persistence, which is consistent with volatility clustering. This distinction is important: the empirical task is not to forecast the direction of SPY returns, but to forecast the conditional lower tail of the return distribution.

![Figure 2.3. Autocorrelation diagnostics for returns, squared returns, and absolute returns.](../outputs/section2_academic/figures/fig2_3_autocorrelation_diagnostics.png)

The volatility-regime summary reinforces this point. The 5% return quantile is {low_p05:.6f} in the low-volatility regime, compared with {high_p05:.6f} in the high-volatility regime. Therefore, the same nominal VaR level corresponds to very different return thresholds across market states.

![Figure 2.5. Volatility regimes and rolling-window design.](../outputs/section2_academic/figures/fig2_5_volatility_regimes_and_windows.png)

## 2.5 Out-of-sample VaR forecasting design

The forecast target is the one-day-ahead lower-tail VaR of `log_ret` at the 1%, 5%, and 10% levels. For a tail probability alpha, VaR is interpreted as the conditional quantile satisfying Pr(r_{{t+1}} < VaR_{{alpha,t+1}} | F_t) = alpha.

To match Sections 3 and 4, the empirical analysis compares three rolling-window lengths: W = 250, 500, and 1000 trading days. These windows correspond approximately to one, two, and four trading years. For a forecast made at time t, the model uses only the information set F_t(W) = {{r_{{t-W+1}}, ..., r_t}} and the corresponding volatility variables observed inside the same window. The window then rolls forward one day at a time.

The resulting out-of-sample forecast counts are {int(window.loc[250, "out_of_sample_n"]):,} for W = 250, {int(window.loc[500, "out_of_sample_n"]):,} for W = 500, and {int(window.loc[1000, "out_of_sample_n"]):,} for W = 1000. Sections 3 and 4 retain all three windows in the empirical comparison, while placing emphasis on W = 1000 because it provides more tail observations for crisis-versus-calm analysis.

## 2.6 Backtesting criteria

The backtesting procedure is common across all models. A VaR violation occurs when the realized return is below the forecasted VaR. The failure rate compares the observed violation frequency with the nominal tail probability. The Kupiec unconditional coverage test evaluates whether the violation rate is statistically consistent with the target probability. The Christoffersen independence and conditional coverage tests examine whether violations are serially clustered. The duration test evaluates the spacing between violations, and the Lopez loss provides a loss-based comparison of VaR forecasts.

This common framework makes the following chapters directly comparable. Section 3 evaluates nonparametric Historical Simulation methods, Section 4 evaluates GARCH-type conditional-volatility models, and Section 5 evaluates neural quantile regression.
"""

    zh = f"""# 第二章 数据特征分析与VaR回测框架

## 2.1 数据与变量

本章说明实证数据，给出后续模型选择的经验依据，并建立统一的样本外VaR回测框架。样本为SPY日度数据，区间为{start}至{end}，共{n:,}个交易日。预测目标变量为日对数收益率`log_ret`。数据还包含两个高频波动率指标`rv5`和`bv`，用于描述市场波动状态，并作为第五章神经网络分位数模型的波动率输入。

本章并不是单纯的数据描述。数据的分布特征和时间序列特征决定后续模型是否有必要。厚尾特征支持非正态和非参数VaR方法，波动聚集支持条件方差模型，而已实现波动率代理变量的存在则支持引入带有波动率特征的非线性机器学习模型。

## 2.2 收益率分布特征

`log_ret`的样本均值为{mean_ret:.6f}，样本标准差为{sd_ret:.6f}。收益率分布明显偏离正态分布：样本偏度为{skew:.3f}，超额峰度为{kurt:.3f}。Jarque-Bera正态性检验的p值为{jb_p:.4g}，在常用显著性水平下拒绝正态分布假设。这一点对VaR尤其重要，因为正态分布基准可能低估极端损失概率。

图2.1(a)表明，大幅负收益并不是均匀分布在整个样本期内，而是集中出现在市场压力时期。图2.1(b)报告对应的滚动波动率，可以看到波动率在相同压力阶段明显上升。

![图2.1 SPY日收益率与滚动波动率。](../outputs/section2_academic/figures/fig2_1_returns_and_rolling_volatility.png)

图2.2进一步从分布角度提供证据。1%、5%和10%经验下尾分位数分别为{q01:.6f}、{q05:.6f}和{q10:.6f}。这些经验分位数构成第三章历史模拟法的非参数基准，但也说明尾部估计本身存在困难，尤其是1%极端尾部。

![图2.2 SPY日对数收益率的经验分布与正态Q-Q图。](../outputs/section2_academic/figures/fig2_2_distribution_and_qq.png)

## 2.3 波动聚集与波动率代理变量

图2.1(b)中的滚动波动率结果表明，波动率具有明显状态依赖性：平稳时期与压力时期交替出现，高波动状态具有持续性。这一现象支持第四章使用GARCH类模型，因为静态无条件分位数无法充分刻画随时间变化的条件风险。

图2.4考察`rv5`和`bv`。两个波动率代理变量的相关系数为{rv_bv_corr:.3f}，说明二者都包含市场波动信息。但它们并不完全相同，尤其在极端交易日可能出现差异。因此，在神经网络分位数模型中同时保留二者，比只使用滞后收益率更有信息基础。

![图2.4 已实现波动率代理变量。](../outputs/section2_academic/figures/fig2_4_realized_volatility_proxies.png)

## 2.4 自相关证据与建模含义

图2.3比较了原始收益率、平方收益率和绝对收益率的自相关函数。原始收益率的序列相关性较弱，说明条件均值可预测性有限；平方收益率和绝对收益率则表现出更强持续性，符合波动聚集特征。这一区分很关键：本文不是预测SPY收益方向，而是预测收益率分布的条件下尾。

![图2.3 自相关诊断。](../outputs/section2_academic/figures/fig2_3_autocorrelation_diagnostics.png)

波动状态分组也支持这一点。低波动状态下5%收益率分位数为{low_p05:.6f}，高波动状态下则为{high_p05:.6f}。因此，相同名义VaR水平在不同市场状态下对应完全不同的收益率阈值。

![图2.5 波动状态与滚动窗口设计。](../outputs/section2_academic/figures/fig2_5_volatility_regimes_and_windows.png)

## 2.5 样本外VaR预测设计

本文预测目标为`log_ret`的一日 ahead 下尾VaR，尾部概率为1%、5%和10%。对给定尾部概率alpha，VaR可理解为满足Pr(r_{{t+1}} < VaR_{{alpha,t+1}} | F_t) = alpha的条件分位数。

为匹配第三章和第四章，本文比较W = 250、500和1000三种滚动窗口，分别约等于一、两和四个交易年。在时间t进行预测时，模型只能使用信息集F_t(W) = {{r_{{t-W+1}}, ..., r_t}}以及同一窗口内的波动率变量。之后窗口逐日向前滚动。

三种窗口对应的样本外预测数量分别为：W = 250时{int(window.loc[250, "out_of_sample_n"]):,}个，W = 500时{int(window.loc[500, "out_of_sample_n"]):,}个，W = 1000时{int(window.loc[1000, "out_of_sample_n"]):,}个。第三章和第四章保留三种窗口的比较，并重点解释W = 1000，因为该窗口在危机期和平稳期比较中提供更多尾部观测。图2.5同时给出波动状态分组和不同滚动窗口下的样本外预测数量。

## 2.6 回测评价标准

所有模型使用统一回测程序。当实际收益率低于预测VaR时，记为一次VaR违约。failure rate比较实际违约频率与名义尾部概率。Kupiec无条件覆盖检验考察违约率是否与目标概率一致。Christoffersen独立性检验和条件覆盖检验考察违约是否存在时间聚集。duration test从违约间隔角度评价模型是否持续失效。Lopez loss则提供损失函数意义下的VaR预测比较。

因此，后续章节具有可比性：第三章评价非参数历史模拟方法，第四章评价GARCH类条件波动率模型，第五章评价神经网络分位数回归模型。
"""

    EN_MD.write_text(en, encoding="utf-8")
    ZH_MD.write_text(zh, encoding="utf-8")


def build_pdf_source() -> None:
    md = EN_MD.read_text(encoding="utf-8")
    md = md.replace("# Section 2. Data Characteristics and the VaR Backtesting Framework", "## Section 2. Data Characteristics and the VaR Backtesting Framework", 1)
    PDF_SOURCE.write_text(md, encoding="utf-8")


def build_pdf() -> None:
    import importlib.util

    exporter_path = PROJECT_ROOT / "scripts" / "11_export_section3_latex_pdf.py"
    spec = importlib.util.spec_from_file_location("section3_exporter", exporter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load exporter from {exporter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build_pdf(PDF_SOURCE, PDF_OUTPUT, chinese=False)
    if DOWNLOADS.exists():
        shutil.copy2(PDF_OUTPUT, DOWNLOADS / PDF_OUTPUT.name)


def main() -> None:
    ensure_dirs()
    set_academic_style()
    df = load_spy_data().sort_index()
    tables = compute_tables(df)
    generate_figures(df, tables)
    write_captions()
    write_chapters(df, tables)
    build_pdf_source()
    build_pdf()
    print(f"Academic figures: {FIG_DIR}")
    print(f"Tables: {TABLE_DIR}")
    print(f"English chapter: {EN_MD}")
    print(f"Chinese preview: {ZH_MD}")
    print(f"PDF: {PDF_OUTPUT}")
    if DOWNLOADS.exists():
        print(f"Copied PDF to: {DOWNLOADS / PDF_OUTPUT.name}")


if __name__ == "__main__":
    main()

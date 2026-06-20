from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller

from utils import PROJECT_ROOT, REPORT_DIR, REQUIRED_COLUMNS, load_spy_data


SECTION2_DIR = PROJECT_ROOT / "outputs" / "section2"
SECTION2_FIG_DIR = SECTION2_DIR / "figures"
SECTION2_TABLE_DIR = SECTION2_DIR / "tables"


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

COLORS = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
}


def ensure_section2_dirs() -> None:
    SECTION2_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SECTION2_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def use_publication_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "text.color": TOKENS["ink"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman", "Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.75,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.05,
            "patch.linewidth": 0.9,
        }
    )


def add_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top", fontsize=11.5, fontweight="semibold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def format_date_axis(ax: plt.Axes, max_ticks: int = 7) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(length=0)


def export_figure(fig: plt.Figure, filename: str) -> None:
    png_path = SECTION2_FIG_DIR / f"{filename}.png"
    svg_path = SECTION2_FIG_DIR / f"{filename}.svg"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def annualized_vol(series: pd.Series) -> pd.Series:
    return np.sqrt(series.clip(lower=0) * 252.0)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def compute_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    log_ret = df["log_ret"].dropna()
    rv5 = df["rv5"].dropna()
    bv = df["bv"].dropna()

    overview = pd.DataFrame(
        [
            ["sample_start", df.index.min().strftime("%Y-%m-%d")],
            ["sample_end", df.index.max().strftime("%Y-%m-%d")],
            ["observations", len(df)],
            ["variables", ", ".join(REQUIRED_COLUMNS)],
            ["missing_log_ret", int(df["log_ret"].isna().sum())],
            ["missing_rv5", int(df["rv5"].isna().sum())],
            ["missing_bv", int(df["bv"].isna().sum())],
            ["forecast_target", "one-day-ahead lower-tail VaR for log_ret"],
            ["var_levels", "1%, 5%, 10%"],
        ],
        columns=["item", "value"],
    )

    rows = []
    for col in REQUIRED_COLUMNS:
        x = df[col].dropna()
        rows.append(
            {
                "variable": col,
                "count": int(x.count()),
                "mean": x.mean(),
                "std": x.std(),
                "min": x.min(),
                "q01": x.quantile(0.01),
                "q05": x.quantile(0.05),
                "q10": x.quantile(0.10),
                "median": x.median(),
                "q90": x.quantile(0.90),
                "q95": x.quantile(0.95),
                "q99": x.quantile(0.99),
                "max": x.max(),
                "skewness": stats.skew(x, bias=False),
                "excess_kurtosis": stats.kurtosis(x, fisher=True, bias=False),
            }
        )
    descriptive = pd.DataFrame(rows)

    jb_stat, jb_p = stats.jarque_bera(log_ret)
    adf_ret = adfuller(log_ret, autolag="AIC")
    adf_sq = adfuller(log_ret**2, autolag="AIC")
    tests = pd.DataFrame(
        [
            {
                "test": "Jarque-Bera normality",
                "series": "log_ret",
                "statistic": float(jb_stat),
                "p_value": float(jb_p),
                "interpretation": "reject normality when p_value < 0.05",
            },
            {
                "test": "ADF unit-root",
                "series": "log_ret",
                "statistic": float(adf_ret[0]),
                "p_value": float(adf_ret[1]),
                "interpretation": "stationary return series when p_value < 0.05",
            },
            {
                "test": "ADF unit-root",
                "series": "log_ret_squared",
                "statistic": float(adf_sq[0]),
                "p_value": float(adf_sq[1]),
                "interpretation": "stationary squared-return series when p_value < 0.05",
            },
        ]
    )

    tail = pd.DataFrame(
        [
            {"alpha": "1%", "empirical_quantile": log_ret.quantile(0.01), "observed_tail_count": int((log_ret <= log_ret.quantile(0.01)).sum())},
            {"alpha": "5%", "empirical_quantile": log_ret.quantile(0.05), "observed_tail_count": int((log_ret <= log_ret.quantile(0.05)).sum())},
            {"alpha": "10%", "empirical_quantile": log_ret.quantile(0.10), "observed_tail_count": int((log_ret <= log_ret.quantile(0.10)).sum())},
        ]
    )

    regime = df.copy()
    regime["rv5_regime"] = pd.qcut(regime["rv5"], q=[0.0, 0.33, 0.66, 1.0], labels=["Low volatility", "Medium volatility", "High volatility"])
    regime_summary = (
        regime.groupby("rv5_regime", observed=False)
        .agg(
            observations=("log_ret", "size"),
            mean_return=("log_ret", "mean"),
            return_std=("log_ret", "std"),
            q01_return=("log_ret", lambda x: x.quantile(0.01)),
            q05_return=("log_ret", lambda x: x.quantile(0.05)),
            mean_rv5=("rv5", "mean"),
            mean_bv=("bv", "mean"),
        )
        .reset_index()
    )

    correlations = df[REQUIRED_COLUMNS].corr()

    worst = df.nsmallest(12, "log_ret")[["log_ret", "rv5", "bv"]].copy()
    worst.insert(0, "date", worst.index.strftime("%Y-%m-%d"))

    window_rows = []
    for window in [250, 500, 1000]:
        start_idx = window
        forecast_start = df.index[start_idx].strftime("%Y-%m-%d")
        forecast_end = df.index[-1].strftime("%Y-%m-%d")
        window_rows.append(
            {
                "window_size": window,
                "interpretation": f"approximately {window // 250} trading year(s)",
                "forecast_start": forecast_start,
                "forecast_end": forecast_end,
                "out_of_sample_forecasts": len(df) - window,
                "expected_1pct_violations": (len(df) - window) * 0.01,
                "expected_5pct_violations": (len(df) - window) * 0.05,
                "expected_10pct_violations": (len(df) - window) * 0.10,
            }
        )
    window_design = pd.DataFrame(window_rows)

    tables = {
        "section2_dataset_overview": overview,
        "section2_descriptive_statistics": descriptive,
        "section2_distribution_stationarity_tests": tests,
        "section2_tail_quantiles": tail,
        "section2_volatility_regime_summary": regime_summary,
        "section2_correlation_matrix": correlations,
        "section2_worst_return_days": worst,
        "section2_window_design": window_design,
    }
    for name, table in tables.items():
        table.to_csv(SECTION2_TABLE_DIR / f"{name}.csv", index=(name == "section2_correlation_matrix"))
    return tables


def plot_returns_with_tails(df: pd.DataFrame) -> None:
    q01 = df["log_ret"].quantile(0.01)
    tail = df[df["log_ret"] <= q01]
    start = df.index.min().strftime("%Y")
    end = df.index.max().strftime("%Y")
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.plot(df.index, df["log_ret"], color=COLORS["blue"]["mid"], linewidth=0.65, label="Daily log return")
    ax.scatter(tail.index, tail["log_ret"], s=13, color=COLORS["orange"]["mid"], edgecolor=COLORS["orange"]["dark"], linewidth=0.3, label="Bottom 1% days", zorder=3)
    ax.axhline(q01, color=COLORS["orange"]["dark"], linestyle=":", linewidth=1.0, label=f"Empirical 1% quantile ({pct(q01)})")
    ax.axhline(0, color=NEUTRAL["dark"], linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    format_date_axis(ax)
    clean_axes(ax)
    ax.set_xlabel("")
    ax.set_ylabel("Log return")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=3, borderaxespad=0)
    add_header(fig, ax, "Time series of SPY daily log returns and lower-tail observations", f"Daily log returns, {start}-{end}; marked observations fall below the empirical 1% quantile.")
    export_figure(fig, "fig2_1_log_returns_tail_events")


def plot_realized_volatility(df: pd.DataFrame) -> None:
    plot_df = pd.DataFrame(
        {
            "rv5_ann": annualized_vol(df["rv5"]),
            "bv_ann": annualized_vol(df["bv"]),
        },
        index=df.index,
    ).rolling(5).mean()

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.plot(plot_df.index, plot_df["rv5_ann"], color=COLORS["blue"]["mid"], linewidth=0.9, label="Realized volatility (rv5)")
    ax.plot(plot_df.index, plot_df["bv_ann"], color=COLORS["gold"]["mid"], linewidth=0.9, linestyle="--", label="Bipower variation (bv)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    format_date_axis(ax)
    clean_axes(ax)
    ax.set_xlabel("")
    ax.set_ylabel("Annualized volatility")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Time variation in realized volatility and bipower variation", "Five-day moving averages of annualized volatility proxies derived from rv5 and bv.")
    export_figure(fig, "fig2_2_realized_volatility_and_bv")


def plot_return_distribution(df: pd.DataFrame) -> None:
    x = df["log_ret"].dropna()
    mu, sigma = x.mean(), x.std()
    grid = np.linspace(x.quantile(0.001), x.quantile(0.999), 600)
    normal_pdf = stats.norm.pdf(grid, loc=mu, scale=sigma)
    q01, q05, q10 = x.quantile([0.01, 0.05, 0.10])

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(x, bins=90, density=True, color=COLORS["blue"]["light"], edgecolor=COLORS["blue"]["dark"], linewidth=0.45, alpha=0.92, label="Empirical density")
    ax.plot(grid, normal_pdf, color=NEUTRAL["dark"], linewidth=1.1, label="Fitted normal density")
    for q, label, color in [(q01, "1%", COLORS["orange"]["dark"]), (q05, "5%", COLORS["gold"]["dark"]), (q10, "10%", COLORS["pink"]["dark"])]:
        ax.axvline(q, color=color, linestyle=":", linewidth=1.0)
        ax.text(q, ax.get_ylim()[1] * 0.86, label, ha="center", va="top", fontsize=8, color=color)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    clean_axes(ax)
    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Empirical distribution of SPY daily log returns", "Histogram with fitted Gaussian density and empirical lower-tail quantiles used for VaR analysis.")
    export_figure(fig, "fig2_3_return_distribution_normal_overlay")


def plot_qq(df: pd.DataFrame) -> None:
    x = np.sort(df["log_ret"].dropna().to_numpy())
    n = len(x)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probs, loc=x.mean(), scale=x.std())
    low_tail = probs <= 0.05

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(theoretical[~low_tail], x[~low_tail], s=8, color=COLORS["blue"]["base"], edgecolor="none", alpha=0.35, label="Central observations")
    ax.scatter(theoretical[low_tail], x[low_tail], s=13, color=COLORS["orange"]["mid"], edgecolor=COLORS["orange"]["dark"], linewidth=0.2, alpha=0.8, label="Lower 5% tail")
    lo = min(theoretical.min(), x.min())
    hi = max(theoretical.max(), x.max())
    ax.plot([lo, hi], [lo, hi], color=NEUTRAL["dark"], linestyle=":", linewidth=1.0, label="Gaussian reference")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    clean_axes(ax)
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Empirical return quantile")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=2, borderaxespad=0)
    add_header(fig, ax, "Normal Q-Q plot of SPY daily log returns", "The Gaussian reference distribution uses the sample mean and standard deviation of daily log returns.")
    export_figure(fig, "fig2_4_qq_plot_lower_tail")


def plot_rolling_volatility(df: pd.DataFrame) -> None:
    rolling_vol = df["log_ret"].rolling(60).std() * np.sqrt(252)
    threshold = rolling_vol.quantile(0.90)
    high_vol = rolling_vol >= threshold

    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.fill_between(df.index, 0, rolling_vol, where=high_vol.to_numpy(), color=COLORS["orange"]["xlight"], alpha=0.95, label="Top 10% rolling-volatility regime")
    ax.plot(df.index, rolling_vol, color=COLORS["blue"]["mid"], linewidth=0.9, label="60-day rolling volatility")
    ax.axhline(threshold, color=COLORS["orange"]["dark"], linestyle=":", linewidth=1.0, label=f"90th percentile ({pct(threshold)})")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    format_date_axis(ax)
    clean_axes(ax)
    ax.set_xlabel("")
    ax.set_ylabel("Annualized volatility")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=3, borderaxespad=0)
    add_header(fig, ax, "Sixty-day rolling volatility of SPY daily log returns", "Annualized standard deviation computed from a 60-trading-day rolling window.")
    export_figure(fig, "fig2_5_rolling_volatility_regimes")


def plot_acf_panels(df: pd.DataFrame) -> None:
    values = df["log_ret"].dropna()
    acf_ret = acf(values, nlags=40, fft=True)[1:]
    acf_sq = acf(values**2, nlags=40, fft=True)[1:]
    lags = np.arange(1, 41)
    conf = 1.96 / np.sqrt(len(values))

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.5), sharex=True)
    for ax, vals, title, color in [
        (axes[0], acf_ret, "Returns", COLORS["blue"]["mid"]),
        (axes[1], acf_sq, "Squared returns", COLORS["orange"]["mid"]),
    ]:
        ax.axhspan(-conf, conf, color=NEUTRAL["xlight"], zorder=0)
        ax.axhline(0, color=NEUTRAL["dark"], linewidth=0.8)
        ax.vlines(lags, 0, vals, color=color, linewidth=1.0)
        ax.scatter(lags, vals, s=12, color=color, edgecolor=COLORS["blue"]["dark"] if title == "Returns" else COLORS["orange"]["dark"], linewidth=0.25, zorder=3)
        clean_axes(ax)
        ax.set_ylabel(title)
    axes[1].set_xlabel("Lag")
    add_header(fig, axes[0], "Autocorrelation functions of returns and squared returns", "ACF up to 40 trading-day lags; shaded band is an approximate 95% white-noise interval.")
    export_figure(fig, "fig2_6_acf_returns_and_squared_returns")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[REQUIRED_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "section2_corr",
        [COLORS["orange"]["light"], TOKENS["panel"], COLORS["blue"]["mid"]],
    )
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(np.arange(len(corr.columns)), corr.columns)
    ax.set_yticks(np.arange(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9, color=TOKENS["ink"])
    ax.grid(False)
    clean_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, length=0, colors=TOKENS["muted"])
    add_header(fig, ax, "Correlation matrix of return and volatility variables", "Pearson correlations among daily log returns, realized volatility, and bipower variation.")
    export_figure(fig, "fig2_7_correlation_heatmap")


def plot_rv_bv_relationship(df: pd.DataFrame) -> None:
    plot_df = pd.DataFrame(
        {
            "rv5_ann": annualized_vol(df["rv5"]),
            "bv_ann": annualized_vol(df["bv"]),
            "abs_ret": df["log_ret"].abs(),
        }
    ).dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "section2_hex",
        [TOKENS["panel"], COLORS["blue"]["xlight"], COLORS["blue"]["base"], COLORS["blue"]["mid"]],
    )
    hb = ax.hexbin(plot_df["bv_ann"], plot_df["rv5_ann"], gridsize=36, mincnt=1, cmap=cmap, linewidths=0.2, edgecolors=TOKENS["panel"])
    lo = min(plot_df["bv_ann"].min(), plot_df["rv5_ann"].min())
    hi = max(plot_df["bv_ann"].quantile(0.995), plot_df["rv5_ann"].quantile(0.995))
    ax.plot([lo, hi], [lo, hi], color=NEUTRAL["dark"], linestyle=":", linewidth=1.0, label="45-degree reference")
    ax.set_xlim(left=0, right=hi)
    ax.set_ylim(bottom=0, top=hi)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    clean_axes(ax)
    ax.set_xlabel("Bipower variation, annualized")
    ax.set_ylabel("Realized volatility, annualized")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncol=1, borderaxespad=0)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Trading days", color=TOKENS["muted"], fontsize=8)
    cbar.ax.tick_params(labelsize=8, length=0, colors=TOKENS["muted"])
    add_header(fig, ax, "Relationship between realized volatility and bipower variation", "Hexbin density of annualized rv5 and bv; values above the 99.5th percentile are trimmed only for axis readability.")
    export_figure(fig, "fig2_8_rv5_bv_hexbin")


def plot_monthly_volatility_heatmap(df: pd.DataFrame) -> None:
    monthly = df["log_ret"].resample("ME").std() * np.sqrt(252)
    matrix = monthly.to_frame("vol")
    matrix["year"] = matrix.index.year
    matrix["month"] = matrix.index.month
    pivot = matrix.pivot(index="year", columns="month", values="vol")

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "section2_monthly_vol",
        [TOKENS["panel"], COLORS["blue"]["xlight"], COLORS["blue"]["light"], COLORS["orange"]["base"], COLORS["orange"]["dark"]],
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index.astype(str))
    ax.grid(False)
    clean_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.025)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    cbar.ax.tick_params(labelsize=8, length=0, colors=TOKENS["muted"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_header(fig, ax, "Monthly realized volatility by calendar year and month", "Annualized monthly standard deviation of daily log returns; darker cells indicate higher realized volatility.")
    export_figure(fig, "fig2_9_monthly_volatility_heatmap")


def generate_figures(df: pd.DataFrame) -> None:
    plot_returns_with_tails(df)
    plot_realized_volatility(df)
    plot_return_distribution(df)
    plot_qq(df)
    plot_rolling_volatility(df)
    plot_acf_panels(df)
    plot_correlation_heatmap(df)
    plot_rv_bv_relationship(df)
    plot_monthly_volatility_heatmap(df)


def write_chapter_draft(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> Path:
    desc = tables["section2_descriptive_statistics"].set_index("variable")
    tests = tables["section2_distribution_stationarity_tests"].set_index("test")
    tail = tables["section2_tail_quantiles"].set_index("alpha")
    regime = tables["section2_volatility_regime_summary"].set_index("rv5_regime")
    corr = tables["section2_correlation_matrix"]
    window_design = tables["section2_window_design"].set_index("window_size")

    start = df.index.min().strftime("%Y-%m-%d")
    end = df.index.max().strftime("%Y-%m-%d")
    n = len(df)
    mean_ret = desc.loc["log_ret", "mean"]
    std_ret = desc.loc["log_ret", "std"]
    skew_ret = desc.loc["log_ret", "skewness"]
    kurt_ret = desc.loc["log_ret", "excess_kurtosis"]
    jb_p = tests.loc["Jarque-Bera normality", "p_value"]
    q01 = tail.loc["1%", "empirical_quantile"]
    q05 = tail.loc["5%", "empirical_quantile"]
    q10 = tail.loc["10%", "empirical_quantile"]
    low_q05 = regime.loc["Low volatility", "q05_return"]
    high_q05 = regime.loc["High volatility", "q05_return"]
    rv_bv_corr = corr.loc["rv5", "bv"]
    w250_n = int(window_design.loc[250, "out_of_sample_forecasts"])
    w500_n = int(window_design.loc[500, "out_of_sample_forecasts"])
    w1000_n = int(window_design.loc[1000, "out_of_sample_forecasts"])
    w250_start = window_design.loc[250, "forecast_start"]
    w500_start = window_design.loc[500, "forecast_start"]
    w1000_start = window_design.loc[1000, "forecast_start"]

    text = f"""# 第二章 数据特征分析与 VaR 回测框架

## 2.1 数据来源与变量定义

本章首先对实证数据进行描述性统计和探索性分析，并在此基础上给出后续 VaR 预测模型的统一实证框架。本文使用 SPY 的日度收益率和高频波动率指标作为实证样本。样本区间为 {start} 至 {end}，共 {n:,} 个交易日。核心变量包括日对数收益率 `log_ret`、五分钟 realized volatility 指标 `rv5`，以及 bipower variation 指标 `bv`。其中 `log_ret` 是后续 VaR 预测的目标变量，`rv5` 和 `bv` 则用于刻画市场条件和波动状态，并为第五章的神经网络分位数模型提供可解释的波动率输入。

表 `section2_dataset_overview.csv` 汇总了样本范围、观测数量、缺失情况和后续 VaR 预测设定。样本中三个核心变量均可直接用于滚动窗口实证分析，因此后续模型可以在统一的数据口径下进行比较。需要强调的是，本章的探索性分析并非单纯的数据可视化，而是服务于后续模型选择：收益率分布特征决定是否需要非正态和非参数方法，波动聚集决定是否需要条件异方差模型，而高频波动率变量则决定神经网络模型是否具备额外信息输入。

## 2.2 收益率分布特征

表 `section2_descriptive_statistics.csv` 显示，SPY 日对数收益率的样本均值为 {mean_ret:.6f}，标准差为 {std_ret:.6f}。从偏度和峰度看，收益率分布并不服从简单的正态假设：偏度为 {skew_ret:.3f}，超额峰度为 {kurt_ret:.3f}。Jarque-Bera 正态性检验的 p 值为 {jb_p:.4g}，在常用显著性水平下拒绝正态分布假设。这说明若直接使用正态分布假设估计 VaR，可能低估极端损失发生的概率。

图 2-1 展示了日收益率时间序列，并标记低于经验 1% 分位数的极端下跌日。可以看到，尾部损失并非均匀分布，而是集中出现在市场压力阶段。图 2-3 和图 2-4 分别从直方图和 Q-Q 图角度进一步说明收益率存在尖峰厚尾和下尾偏离。因此，本文不能只依赖正态线性模型，而需要比较非参数历史模拟、厚尾 GARCH 和神经网络分位数模型。

从 VaR 角度看，经验 1%、5% 和 10% 分位数分别为 {q01:.6f}、{q05:.6f} 和 {q10:.6f}。这些分位数为第三章历史模拟法提供了最直接的非参数基准，也说明了极端尾部样本数量有限，尤其是 1% VaR 的估计更容易受到窗口长度和市场状态变化影响。

## 2.3 波动聚集与市场状态

图 2-2 报告了 annualized `rv5` 与 `bv` 的时间序列，图 2-5 报告了 60 日滚动年化波动率。两个图共同说明，SPY 的波动率存在明显的状态转换和持续性：平稳阶段的波动率较低，而金融危机和其他市场压力阶段会形成显著的高波动区间。这一事实直接支持第四章引入 GARCH 类模型，因为 GARCH 模型的核心就是用条件方差动态刻画波动聚集。若不考虑条件波动率变化，模型在高波动阶段容易给出过于乐观的风险阈值。

表 `section2_volatility_regime_summary.csv` 将样本按 `rv5` 分成低、中、高三个波动状态。低波动状态下 5% 收益率分位数为 {low_q05:.6f}，高波动状态下 5% 收益率分位数为 {high_q05:.6f}，说明相同置信水平下的风险阈值会随市场状态显著改变。因此，固定窗口经验分位数虽然透明，但可能在状态切换时反应不足；这也是后续比较加权历史模拟、GARCH-t 和神经网络模型的原因。

## 2.4 自相关、波动代理变量与建模含义

图 2-6 比较了原始收益率和平方收益率的自相关函数。原始收益率自相关整体较弱，说明均值预测空间有限；但平方收益率的自相关更强，反映出波动率具有持续性。这个结果与金融时间序列的典型经验事实一致：收益率方向难以预测，但风险水平和波动状态可以建模。

图 2-7 和图 2-8 进一步考察了 `log_ret`、`rv5` 和 `bv` 之间的关系。`rv5` 与 `bv` 的相关系数为 {rv_bv_corr:.3f}，说明两个高频波动变量包含高度重叠的市场波动信息，但在跳跃或极端交易日也可能出现差异。第五章神经网络模型可以利用这类变量，在不预设线性条件方差方程的情况下学习尾部分位数。

## 2.5 VaR 预测与回测框架

基于上述数据事实，本文采用统一的滚动窗口 VaR 预测框架。预测目标是 `log_ret` 的一日 ahead 下尾 VaR，尾部概率设为 1%、5% 和 10%。为了避免 look-ahead bias，每个模型在预测日之前的信息集上估计，并只使用预测日前已经可观测的数据。对给定尾部概率 alpha，VaR 可写为条件下尾分位数：

$$
\Pr(r_{{t+1}} < \mathrm{{VaR}}_{{\alpha,t+1}}\mid \mathcal{{F}}_t)=\alpha.
$$

为与第三章历史模拟法和第四章 GARCH 类模型保持一致，本文比较 W = 250、500 和 1000 三种滚动窗口，分别近似对应一、两和四个交易年。设第 t 日的日对数收益率为 r_t，在预测 r_{{t+1}} 的 VaR 时，长度为 W 的信息集定义为：

$$
\mathcal{{F}}_t(W)=\left\{{r_{{t-W+1}},r_{{t-W+2}},\ldots,r_t\right\}}.
$$

模型只能使用该窗口内的历史收益率及同一窗口内的 `rv5`、`bv` 等变量。窗口随后逐日向前滚动，形成样本外 VaR 预测序列。三种窗口对应的样本外预测期分别为：W = 250 时从 {w250_start} 至 {end}，共 {w250_n:,} 个预测日；W = 500 时从 {w500_start} 至 {end}，共 {w500_n:,} 个预测日；W = 1000 时从 {w1000_start} 至 {end}，共 {w1000_n:,} 个预测日。第三章和第四章在报告结果时保留三种窗口的比较，并重点解释 W = 1000 的结果，因为该窗口在极端尾部 VaR 估计中提供更多尾部观测，适合进行危机期与平稳期的子样本比较。

对于神经网络分位数模型，滚动训练集同样遵循上述时间顺序约束；对于 GARCH 类模型，参数估计和条件方差预测也仅基于滚动窗口内信息。由此，第三章、第四章和第五章的模型结果具有可比性。

后续章节比较三类模型：

1. 第三章使用历史模拟法及其改进，包括普通历史模拟、时间加权历史模拟和 KDE 平滑加权历史模拟。
2. 第四章使用 GARCH 类模型，重点考察 GARCH(1,1)-t 和 GJR-GARCH(1,1)-t 对厚尾和非对称波动的刻画能力。
3. 第五章使用神经网络分位数回归，将滞后收益率和高频波动率变量输入 MLP，并通过 pinball loss 直接预测 VaR 分位数。

模型评价使用统一回测框架。首先，定义 VaR 违约指示变量：

$$
I_{{t+1}}=\mathbf{{1}}\left(r_{{t+1}}<\widehat{{\mathrm{{VaR}}}}_{{\alpha,t+1}}\right).
$$

failure rate 用于比较实际 VaR 违约比例是否接近名义尾部概率。Kupiec 无条件覆盖检验用于检验 VaR 违约率是否显著偏离理论水平。Christoffersen 独立性检验和条件覆盖检验用于判断 VaR 违约是否存在聚集现象。duration test 从违约间隔角度检验风险预测是否存在持续性失效，Lopez loss 则提供损失函数意义下的模型比较标准。第二章的可视化和统计检验表明，SPY 收益率同时具有厚尾、波动聚集和市场状态转换特征，因此第三章、第四章和第五章分别引入历史模拟、GARCH 类模型和神经网络模型具有明确的金融统计动机。

## 2.6 图表清单

- 图 2-1：`outputs/section2/figures/fig2_1_log_returns_tail_events.png`
- 图 2-2：`outputs/section2/figures/fig2_2_realized_volatility_and_bv.png`
- 图 2-3：`outputs/section2/figures/fig2_3_return_distribution_normal_overlay.png`
- 图 2-4：`outputs/section2/figures/fig2_4_qq_plot_lower_tail.png`
- 图 2-5：`outputs/section2/figures/fig2_5_rolling_volatility_regimes.png`
- 图 2-6：`outputs/section2/figures/fig2_6_acf_returns_and_squared_returns.png`
- 图 2-7：`outputs/section2/figures/fig2_7_correlation_heatmap.png`
- 图 2-8：`outputs/section2/figures/fig2_8_rv5_bv_hexbin.png`
- 图 2-9：`outputs/section2/figures/fig2_9_monthly_volatility_heatmap.png`
"""

    path = REPORT_DIR / "section_2_data_visualization_empirical_analysis_zh.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_plotting_notes() -> Path:
    text = dedent(
        """\
        # Section 2 plotting style notes

        The figures in Section 2 use a static academic-publication workflow:

        - Matplotlib is the rendering engine because it is already installed in the project environment.
        - The style system follows a restrained grammar-of-graphics approach associated with popular Python/R plotting libraries: fixed semantic tokens, explicit palette maps, quiet grids, visible axis anchors, and neutral descriptive titles/subtitles.
        - PNG files are exported at 320 dpi for direct insertion into the thesis; SVG files are exported alongside them for later vector editing.
        - The palette is intentionally limited to blue, gold, orange, olive, pink, and neutrals. This avoids default rainbow colors and keeps the chapter visually consistent.
        - Every figure states the metric scope in the subtitle so the chart remains interpretable after being copied into Word, LaTeX, or PowerPoint.
        - Figure titles are descriptive rather than promotional or conclusion-led; substantive interpretation belongs in the surrounding thesis text.
        """
    )
    path = REPORT_DIR / "section_2_plotting_style_notes.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_figure_captions() -> Path:
    text = """# 第二章图表标题与说明

图 2-1 SPY日对数收益率及下尾观测值时间序列

注：蓝色线表示SPY日对数收益率，橙色点表示低于经验1%分位数的观测值。该图用于说明样本期内极端下跌事件的时间分布及其聚集特征。

图 2-2 已实现波动率与双幂变差的时间变化

注：图中变量为由`rv5`和`bv`计算得到的年化波动率代理变量，并使用5日移动平均进行平滑处理。该图用于观察高频波动率指标在不同市场阶段的变化。

图 2-3 SPY日对数收益率的经验分布

注：柱状图表示日对数收益率的经验密度，实线表示基于样本均值和标准差拟合的正态密度，虚线表示经验1%、5%和10%下尾分位数。

图 2-4 SPY日对数收益率的正态Q-Q图

注：横轴为理论正态分位数，纵轴为样本经验分位数。下尾观测点相对参考线的偏离用于说明正态分布假设在尾部风险刻画上的局限。

图 2-5 SPY日对数收益率的60日滚动波动率

注：滚动波动率按60个交易日窗口计算并进行年化处理。阴影区域表示滚动波动率处于样本90%分位数以上的高波动状态。

图 2-6 收益率与平方收益率的自相关函数

注：图中报告1至40阶滞后的自相关系数，阴影区域为近似95%白噪声置信区间。平方收益率自相关用于检验波动聚集现象。

图 2-7 收益率与波动率变量的相关系数矩阵

注：矩阵报告`log_ret`、`rv5`和`bv`之间的Pearson相关系数，用于说明高频波动率变量之间的信息重叠程度。

图 2-8 已实现波动率与双幂变差的关系

注：图中使用hexbin密度图展示年化`rv5`和`bv`之间的联合分布，并给出45度参考线。坐标轴为提高可读性，对99.5%分位数以上的极端值作显示范围截断。

图 2-9 按年份和月份划分的月度已实现波动率

注：每个单元格表示对应年月内日对数收益率标准差的年化值，颜色越深表示该月市场波动越高。
"""
    path = REPORT_DIR / "section_2_figure_captions_zh.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    ensure_section2_dirs()
    use_publication_theme()
    df = load_spy_data()
    df = df.sort_index()

    tables = compute_tables(df)
    generate_figures(df)
    chapter_path = write_chapter_draft(df, tables)
    notes_path = write_plotting_notes()
    captions_path = write_figure_captions()

    print(f"Section 2 tables saved to: {SECTION2_TABLE_DIR}")
    print(f"Section 2 figures saved to: {SECTION2_FIG_DIR}")
    print(f"Section 2 chapter draft saved to: {chapter_path}")
    print(f"Plotting notes saved to: {notes_path}")
    print(f"Figure captions saved to: {captions_path}")


if __name__ == "__main__":
    main()

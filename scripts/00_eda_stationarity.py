"""EDA and stationarity checks for Ghana TB GBD data.

This script creates:
- Time-series plots for all modeling targets
- Sex-stratified trend plots
- Condensed age-group trend plots
- ADF/KPSS stationarity tests for d = 0, 1, 2
- Recommended differencing order d for each modeling series

Run:
   python scripts/00_eda_stationarity.py
"""

from __future__ import annotations

# Warnings are used to silence known statsmodels lookup-table warnings during
# stationarity testing.
import warnings
# Path gives robust file paths relative to the repository root.
from pathlib import Path

# Matplotlib, seaborn, numpy, and pandas are the core EDA stack used throughout
# this script.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
# ACF/PACF plots help inspect serial dependence and possible AR/MA structure.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# InterpolationWarning appears when KPSS p-values are outside lookup bounds.
from statsmodels.tools.sm_exceptions import InterpolationWarning
# seasonal_decompose is used here as an exploratory trend/cycle decomposition.
from statsmodels.tsa.seasonal import seasonal_decompose
# ADF and KPSS provide complementary stationarity tests.
from statsmodels.tsa.stattools import adfuller, kpss


# Resolve the repository root from this script's location so the script works
# whether it is run from the project root or another working directory.
ROOT = Path(__file__).resolve().parents[1]
# Raw IHME GBD export used for stratified EDA by sex and age group.
RAW_PATH = ROOT / "data" / "TB_df" / "IHME-GBD_2023_DATA-42460a02-1.csv"
# Wide modeling table created by scripts/01_prepare_and_explore.py.
WIDE_PATH = ROOT / "data" / "processed" / "ghana_tb_modeling_wide.csv"
# Folder for CSV diagnostics and summary outputs.
TABLE_DIR = ROOT / "outputs" / "tables"
# Folder for EDA figure outputs.
EDA_FIGURE_DIR = ROOT / "outputs" / "figures" / "eda"

# All series available in the wide modeling table. Stationarity tests are run
# for all of these so secondary endpoints remain documented.
MODELING_SERIES = [
    "incidence_age_standardized_rate",
    "mortality_to_incidence_age_standardized_ratio",
    "incidence_all_age_rate",
    "incidence_all_age_number",
    "mortality_age_standardized_rate",
    "mortality_all_age_rate",
    "mortality_all_age_number",
]

PRIMARY_ANALYSIS_SERIES = [
    "incidence_age_standardized_rate",
    "mortality_age_standardized_rate",
    "mortality_to_incidence_age_standardized_ratio",
]

SERIES_LABELS = {
    "incidence_age_standardized_rate": "Incidence age-standardized rate",
    "mortality_to_incidence_age_standardized_ratio": "Mortality-to-incidence ratio",
    "incidence_all_age_rate": "Incidence all-age rate",
    "incidence_all_age_number": "Incidence all-age number",
    "mortality_age_standardized_rate": "Mortality age-standardized rate",
    "mortality_all_age_rate": "Mortality all-age rate",
    "mortality_all_age_number": "Mortality all-age number",
}

PRIMARY_SERIES_COLORS = {
    "incidence_age_standardized_rate": "#0072B2",
    "mortality_age_standardized_rate": "#D55E00",
    "mortality_to_incidence_age_standardized_ratio": "#009E73",
}

SEX_COLORS = {
    "Both": "#4D4D4D",
    "Female": "#CC79A7",
    "Male": "#0072B2",
}

AGE_COLORS = {
    "<5 years": "#0072B2",
    "5-14 years": "#D55E00",
    "15-49 years": "#009E73",
    "50-69 years": "#CC79A7",
    "70+ years": "#E69F00",
    "All ages": "#332288",
    "Age-standardized": "#4D4D4D",
}

EDA_FIGSIZE_1X3 = (25, 12)
EDA_LINEWIDTH = 6.0
EDA_MARKERSIZE = 10.0
EDA_LEGEND_PROPS = {"weight": "bold", "size": 35}


def series_y_label(series: str) -> str:
    if "ratio" in series:
        return "Mortality / incidence"
    if "rate" in series:
        return "Rate per 100,000"
    return "Number"


def set_eda_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.titlesize": 30,
            "axes.titleweight": "bold",
            "axes.labelsize": 26,
            "axes.labelweight": "bold",
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 24,
        },
    )


def style_primary_axis(ax) -> None:
    ax.grid(True, which="major", color="#8F8F8F", linewidth=1.8, alpha=0.9)
    ax.margins(x=0.02)
    ax.tick_params(axis="both", width=1.8, length=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)


def style_facet_grid(g) -> None:
    for ax in g.axes.flat:
        style_primary_axis(ax)
        ax.title.set_fontsize(34)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(30)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontsize(30)
        ax.yaxis.label.set_fontweight("bold")
        ax.tick_params(labelsize=26)


def bold_figure_legend(fig) -> None:
    for legend in fig.legends:
        legend.get_title().set_fontweight("bold")
        legend.get_title().set_fontsize(EDA_LEGEND_PROPS["size"])
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontsize(EDA_LEGEND_PROPS["size"])


def plot_primary_line(ax, years, values, series: str, label: str = "Observed") -> None:
    ax.plot(
        years,
        values,
        color=PRIMARY_SERIES_COLORS[series],
        marker="o",
        markersize=EDA_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        linewidth=EDA_LINEWIDTH,
        label=label,
        zorder=3,
    )

# Mapping from processed column names back to GBD measure, age, and metric
# labels. This makes stationarity output tables easier to interpret.
TARGET_MAP = {
    "incidence_age_standardized_rate": ("Incidence", "Age-standardized", "Rate"),
    "mortality_to_incidence_age_standardized_ratio": (
        "Mortality-to-incidence ratio",
        "Age-standardized",
        "Ratio",
    ),
    "incidence_all_age_rate": ("Incidence", "All ages", "Rate"),
    "incidence_all_age_number": ("Incidence", "All ages", "Number"),
    "mortality_age_standardized_rate": ("Deaths", "Age-standardized", "Rate"),
    "mortality_all_age_rate": ("Deaths", "All ages", "Rate"),
    "mortality_all_age_number": ("Deaths", "All ages", "Number"),
}

# Condensed age groups used for readable stratified plots. The raw data contain
# many overlapping age definitions, so the EDA uses a smaller interpretable set.
CONDENSED_AGES = [
    "<5 years",
    "5-14 years",
    "15-49 years",
    "50-69 years",
    "70+ years",
    "All ages",
    "Age-standardized",
]


def ensure_dirs() -> None:
    # Create output directories if they do not already exist.
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    EDA_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def safe_adf(values: np.ndarray) -> float:
    # The ADF test has null hypothesis "unit root / nonstationary".
    # Small p-values therefore support stationarity.
    try:
        # autolag="AIC" lets statsmodels choose the lag length for the test.
        return float(adfuller(values, autolag="AIC")[1])
    except Exception:
        # Return NaN instead of crashing if the test cannot be computed for a
        # short or numerically awkward series.
        return float("nan")


def safe_kpss(values: np.ndarray) -> float:
    # The KPSS test has null hypothesis "stationary". Large p-values therefore
    # support stationarity.
    try:
        with warnings.catch_warnings():
            # KPSS often warns when the statistic is outside tabulated p-value
            # bounds; the returned boundary p-value is still useful for EDA.
            warnings.simplefilter("ignore", InterpolationWarning)
            return float(kpss(values, regression="c", nlags="auto")[1])
    except Exception:
        # Return NaN instead of crashing if KPSS fails.
        return float("nan")


def difference_values(values: np.ndarray, d: int) -> np.ndarray:
    # Start with the original level series.
    differenced = np.asarray(values, dtype=float)
    # Apply differencing d times. d=0 leaves the series unchanged.
    for _ in range(d):
        differenced = np.diff(differenced)
    return differenced


def stationarity_by_d(values: np.ndarray, max_d: int = 3) -> list[dict[str, float | int | bool | str]]:
    # Store one diagnostic row per differencing order.
    rows = []
    # Check d=0, d=1, and d=2 by default.
    for d in range(max_d + 1):
        # Test the series after applying d differences.
        tested = difference_values(values, d)
        # ADF p-value: stationary if <= 0.05.
        adf_p = safe_adf(tested)
        # KPSS p-value: stationary if >= 0.05.
        kpss_p = safe_kpss(tested)
        # Convert the p-values into boolean stationarity decisions.
        adf_stationary = bool(np.isfinite(adf_p) and adf_p <= 0.05)
        kpss_stationary = bool(np.isfinite(kpss_p) and kpss_p >= 0.05)
        # Record whether both tests agree or only one supports stationarity.
        if adf_stationary and kpss_stationary:
            decision = "stationary_by_both_tests"
        elif adf_stationary:
            decision = "stationary_by_adf_only"
        elif kpss_stationary:
            decision = "stationary_by_kpss_only"
        else:
            decision = "nonstationary"
        rows.append(
            {
                "d": d,
                "n_tested": len(tested),
                "adf_p": adf_p,
                "kpss_p": kpss_p,
                "adf_stationary": adf_stationary,
                "kpss_stationary": kpss_stationary,
                "decision": decision,
            }
        )
    return rows


def choose_recommended_d(test_rows: list[dict[str, float | int | bool | str]]) -> tuple[int, str]:
    # Best case: choose the smallest d where ADF and KPSS agree.
    for row in test_rows:
        if row["adf_stationary"] and row["kpss_stationary"]:
            return int(row["d"]), "smallest_d_stationary_by_both_adf_and_kpss"
    # If the tests disagree, choose the smallest d supported by at least one
    # test. This is a pragmatic fallback for short annual series.
    for row in test_rows:
        if row["adf_stationary"] or row["kpss_stationary"]:
            return int(row["d"]), "smallest_d_stationary_by_at_least_one_test"
    # If no test supports stationarity, choose the maximum tested d.
    return int(test_rows[-1]["d"]), "max_d_fallback_no_test_agreement"


def plot_modeling_series(wide: pd.DataFrame) -> None:
    set_eda_theme()
    fig, axes = plt.subplots(1, 3, figsize=EDA_FIGSIZE_1X3, sharex=True)

    for ax, series in zip(axes, PRIMARY_ANALYSIS_SERIES):
        plot_primary_line(ax, wide["year"], wide[series].astype(float), series)
        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        style_primary_axis(ax)

    fig.suptitle("Trends in Primary Tuberculosis Endpoints in Ghana, 1990-2023", fontsize=38, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(EDA_FIGURE_DIR / "primary_modeling_series_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_primary_age_standardized_series(wide: pd.DataFrame) -> None:
    set_eda_theme()
    # Plot the three primary endpoints: incidence, mortality, and their ratio.
    # A side-by-side layout makes the three primary endpoints easy to compare.
    fig, axes = plt.subplots(1, 3, figsize=EDA_FIGSIZE_1X3, sharex=True)
    # Draw one panel for each endpoint.
    for ax, series in zip(axes, PRIMARY_ANALYSIS_SERIES):
        values = wide[series].astype(float)
        plot_primary_line(ax, wide["year"], values, series)
        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        style_primary_axis(ax)
    fig.suptitle("Trend of primary tuberculosis endpoints in Ghana (1990-2023)", fontsize=38, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(EDA_FIGURE_DIR / "Trend_of_TB_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_primary_rolling_statistics(wide: pd.DataFrame, window: int = 5) -> None:
    set_eda_theme()
    # A rolling mean and rolling standard deviation make nonstationary level
    # shifts easier to see in a short annual series.
    fig, axes = plt.subplots(1, 3, figsize=EDA_FIGSIZE_1X3, sharex=True)

    for ax, series in zip(axes, PRIMARY_ANALYSIS_SERIES):
        values = wide[series].astype(float)
        rolling_mean = values.rolling(window=window, min_periods=window).mean()
        rolling_std = values.rolling(window=window, min_periods=window).std()

        ax.plot(
            wide["year"],
            values,
            marker="o",
            markersize=EDA_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=EDA_LINEWIDTH,
            color="#555555",
            label="Observed",
            zorder=3,
        )
        ax.plot(
            wide["year"],
            rolling_mean,
            linewidth=EDA_LINEWIDTH,
            color=PRIMARY_SERIES_COLORS[series],
            label=f"Trailing {window}-year rolling mean",
            zorder=4,
        )
        ax.fill_between(
            wide["year"],
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            color=PRIMARY_SERIES_COLORS[series],
            alpha=0.20,
            label=f"Trailing {window}-year mean ± 1 SD",
            zorder=2,
        )
        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        style_primary_axis(ax)

    fig.suptitle(
        f"Observed primary TB series with {window}-year rolling mean and variability band",
        fontsize=38,
        fontweight="bold",
        y=0.99,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        prop=EDA_LEGEND_PROPS,
        columnspacing=1.6,
        handlelength=3.0,
        handletextpad=0.7,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.91))
    fig.savefig(EDA_FIGURE_DIR / "primary_series_rolling_statistics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_primary_age_standardized_bar_comparison(wide: pd.DataFrame) -> None:
    set_eda_theme()
    # Bar charts are shown separately because the ratio is on a different scale
    # from the rates.
    fig, axes = plt.subplots(1, 3, figsize=EDA_FIGSIZE_1X3, sharex=True)

    for ax, series in zip(axes, PRIMARY_ANALYSIS_SERIES):
        sns.barplot(
            data=wide,
            x="year",
            y=series,
            color=PRIMARY_SERIES_COLORS[series],
            edgecolor="white",
            linewidth=1.0,
            ax=ax,
        )
        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        for index, label in enumerate(ax.get_xticklabels()):
            label.set_visible(index % 4 == 0)
            label.set_rotation(45)
            label.set_horizontalalignment("right")
        style_primary_axis(ax)

    fig.suptitle("Primary tuberculosis endpoints in Ghana, 1990-2023", fontsize=38, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(EDA_FIGURE_DIR / "primary_series_bar_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mortality_to_incidence_ratio(wide: pd.DataFrame) -> None:
    set_eda_theme()
    # The mortality-to-incidence ratio compares death burden against incident
    # disease burden using the two primary age-standardized rates.
    ratio = (
        wide["mortality_age_standardized_rate"].astype(float)
        / wide["incidence_age_standardized_rate"].astype(float)
    )

    # Create a compact one-panel figure for the ratio trend.
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot the ratio as a line so the direction of relative mortality burden is
    # easy to follow over time.
    ax.plot(
        wide["year"],
        ratio,
        marker="o",
        markersize=EDA_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        linewidth=EDA_LINEWIDTH,
        color=PRIMARY_SERIES_COLORS["mortality_to_incidence_age_standardized_ratio"],
        label="Observed ratio",
        zorder=3,
    )

    # Add a dashed fitted trend line to summarize the long-run direction.
    sns.regplot(
        x=wide["year"],
        y=ratio,
        scatter=False,
        ci=None,
        color="#555555",
        line_kws={"linestyle": "--", "linewidth": EDA_LINEWIDTH, "alpha": 0.85},
        ax=ax,
    )

    observed_handle = Line2D(
        [],
        [],
        color=PRIMARY_SERIES_COLORS["mortality_to_incidence_age_standardized_ratio"],
        marker="o",
        markeredgecolor="white",
        markeredgewidth=1.2,
        linewidth=EDA_LINEWIDTH,
        label="Observed ratio",
    )
    trend_handle = Line2D([], [], color="#555555", linestyle="--", linewidth=EDA_LINEWIDTH, alpha=0.85, label="Trend")

    # Use clear labels that state this is a ratio, not a rate per 100,000.
    ax.set_title("Mortality-to-incidence ratio for age-standardized TB rates in Ghana")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mortality rate / incidence rate")
    style_primary_axis(ax)
    ax.legend(
        handles=[observed_handle, trend_handle],
        loc="upper right",
        frameon=False,
        prop=EDA_LEGEND_PROPS,
        handlelength=3.0,
        handletextpad=0.7,
    )

    # Save the ratio trend as a manuscript-friendly PNG.
    fig.tight_layout()
    fig.savefig(
        EDA_FIGURE_DIR / "mortality_to_incidence_ratio_trend.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_sex_stratified(raw: pd.DataFrame) -> None:
    # Keep incidence and deaths, rate metric, and national all-age or
    # age-standardized age definitions. This gives a compact sex comparison.
    subset = raw[
        raw["measure_name"].isin(["Incidence", "Deaths"])
        & raw["metric_name"].eq("Rate")
        & raw["age_name"].isin(["All ages", "Age-standardized"])
    ].copy()
    # Combine measure and age definition into a single facet label.
    subset["panel"] = subset["measure_name"] + " - " + subset["age_name"]

    # Plot male, female, and both-sex rates across time.
    g = sns.relplot(
        data=subset,
        x="year",
        y="val",
        hue="sex_name",
        col="panel",
        col_wrap=2,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4,
        aspect=1.5,
    )
    g.set_axis_labels("Year", "Rate per 100,000")
    g.set_titles("{col_name}")
    # Place the legend in the upper-right corner of the figure.
    sns.move_legend(g, "upper right", frameon=False, title="Sex")
    g.figure.suptitle("Ghana TB rates by sex, 1990-2023", y=1.02)
    g.despine(left=True, bottom=True)
    g.figure.tight_layout(rect=(0, 0, 1, 0.95))
    g.figure.savefig(EDA_FIGURE_DIR / "sex_stratified_rates_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def plot_primary_age_standardized_by_sex(raw: pd.DataFrame) -> None:
    set_eda_theme()
    # Focus on the primary age-standardized endpoints but show sex strata.
    subset = raw[
        raw["measure_name"].isin(["Incidence", "Deaths"])
        & raw["metric_name"].eq("Rate")
        & raw["age_name"].eq("Age-standardized")
    ].copy()

    rate_wide = (
        subset.pivot_table(
            index=["year", "sex_name"],
            columns="measure_name",
            values="val",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"Deaths": "mortality", "Incidence": "incidence"})
    )
    rate_wide["ratio"] = rate_wide["mortality"] / rate_wide["incidence"]

    fig, axes = plt.subplots(1, 3, figsize=EDA_FIGSIZE_1X3, sharex=True)
    panels = [
        ("incidence", "Incidence age-standardized rate", "Rate per 100,000"),
        ("mortality", "Mortality age-standardized rate", "Rate per 100,000"),
        ("ratio", "Mortality-to-incidence ratio", "Mortality / incidence"),
    ]
    handles = labels = None
    for ax, (column, title, ylabel) in zip(axes, panels):
        for sex_name, group in rate_wide.groupby("sex_name"):
            ax.plot(
                group["year"],
                group[column],
                color=SEX_COLORS.get(sex_name, "#555555"),
                marker="o",
                markersize=EDA_MARKERSIZE,
                markeredgecolor="white",
                markeredgewidth=1.2,
                linewidth=EDA_LINEWIDTH,
                label=sex_name,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        style_primary_axis(ax)
        handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="Sex",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            frameon=False,
            prop=EDA_LEGEND_PROPS,
            columnspacing=1.6,
            handlelength=3.0,
            handletextpad=0.7,
        )
    fig.suptitle(
        "Ghana TB primary age-standardized endpoints by sex, 1990-2023",
        fontsize=38,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.91))
    fig.savefig(EDA_FIGURE_DIR / "primary_age_standardized_rates_by_sex.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_primary_acf_pacf(wide: pd.DataFrame) -> None:
    # ACF/PACF diagnostics help assess how much each year depends on prior
    # years and can inform ARIMA p/q choices.
    # Limit lags to keep plots readable and statistically sensible for only
    # 34 annual observations.
    max_lags = 12

    for series in PRIMARY_ANALYSIS_SERIES:
        # Level series: the original modeling endpoint.
        values = wide[series].to_numpy(dtype=float)
        # First difference: annual change in the endpoint.
        differenced = np.diff(values)
        # statsmodels PACF requires lag count to be less than half the sample
        # size, so the lag count is bounded dynamically.
        level_lags = min(max_lags, len(values) // 2 - 1)
        diff_lags = min(max_lags, len(differenced) // 2 - 1)

        # Four panels: ACF/PACF for levels and ACF/PACF for first differences.
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # ACF of the original level series shows persistence/trend dependence.
        plot_acf(values, lags=level_lags, ax=axes[0, 0], zero=False)
        axes[0, 0].set_title("ACF: level series")
        # PACF of levels helps identify direct autoregressive structure.
        plot_pacf(values, lags=level_lags, ax=axes[0, 1], zero=False, method="ywm")
        axes[0, 1].set_title("PACF: level series")
        # ACF after first differencing shows serial dependence in annual changes.
        plot_acf(differenced, lags=diff_lags, ax=axes[1, 0], zero=False)
        axes[1, 0].set_title("ACF: first difference")
        # PACF after first differencing helps diagnose AR structure in changes.
        plot_pacf(differenced, lags=diff_lags, ax=axes[1, 1], zero=False, method="ywm")
        axes[1, 1].set_title("PACF: first difference")

        # Apply consistent x-axis labeling.
        for ax in axes.ravel():
            ax.set_xlabel("Lag")
            ax.margins(x=0.02)

        fig.suptitle(f"{SERIES_LABELS[series]} autocorrelation diagnostics", y=1.02)
        fig.tight_layout()
        fig.savefig(EDA_FIGURE_DIR / f"acf_pacf_{series}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_primary_decomposition(wide: pd.DataFrame) -> None:
    # Annual data do not have monthly/quarterly seasonality. This decomposition
    # is therefore labeled as an exploratory 5-year cycle, not true seasonality.
    # Collect decomposition components for export to CSV.
    decomposition_rows = []
    # Five years gives a compact visual cycle for annual public-health data.
    period = 5

    for series in PRIMARY_ANALYSIS_SERIES:
        # Create a year-indexed Series for statsmodels decomposition.
        ts = pd.Series(
            wide[series].to_numpy(dtype=float),
            index=pd.Index(wide["year"].astype(int), name="year"),
            name=series,
        )
        # Additive decomposition is appropriate because the series are rates and
        # changes are interpreted on the original scale.
        decomposition = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")

        # Save all decomposition components to a long CSV-friendly table.
        components = pd.DataFrame(
            {
                "year": ts.index,
                "series": series,
                "observed": decomposition.observed.to_numpy(),
                "trend": decomposition.trend.to_numpy(),
                "cycle_period5": decomposition.seasonal.to_numpy(),
                "remainder": decomposition.resid.to_numpy(),
            }
        )
        decomposition_rows.append(components)

        # Four stacked panels show observed value, estimated trend, cyclical
        # component, and residual/remainder.
        fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
        axes[0].plot(ts.index, decomposition.observed, marker="o", linewidth=2.0)
        axes[0].set_title("Observed")
        axes[1].plot(ts.index, decomposition.trend, color="tab:blue", linewidth=2.0)
        axes[1].set_title("Estimated trend")
        axes[2].plot(ts.index, decomposition.seasonal, color="tab:green", linewidth=2.0)
        axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axes[2].set_title("Exploratory 5-year cycle component")
        axes[3].plot(ts.index, decomposition.resid, color="tab:red", linewidth=2.0)
        axes[3].axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axes[3].set_title("Remainder")

        # Apply shared formatting to all decomposition panels.
        for ax in axes:
            ax.set_ylabel(series_y_label(series))
            ax.margins(x=0.02)
        axes[-1].set_xlabel("Year")
        fig.suptitle(f"{SERIES_LABELS[series]} decomposition (period = {period} years)", y=1.01)
        fig.tight_layout()
        fig.savefig(EDA_FIGURE_DIR / f"decomposition_{series}_period5.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Export decomposition values so they can be inspected or reported in tables.
    pd.concat(decomposition_rows, ignore_index=True).to_csv(
        TABLE_DIR / "primary_series_decomposition_period5.csv",
        index=False,
    )


def plot_condensed_age_groups(raw: pd.DataFrame) -> None:
    set_eda_theme()
    # Use Both sexes and the rate metric to compare age patterns without
    # doubling lines by sex.
    subset = raw[
        raw["measure_name"].isin(["Incidence", "Deaths"])
        & raw["metric_name"].eq("Rate")
        & raw["sex_name"].eq("Both")
        & raw["age_name"].isin(CONDENSED_AGES)
    ].copy()
    # Preserve a logical age order in the legend and plot.
    subset["age_name"] = pd.Categorical(subset["age_name"], categories=CONDENSED_AGES, ordered=True)
    wide = (
        subset.pivot_table(
            index=["year", "age_name"],
            columns="measure_name",
            values="val",
            aggfunc="first",
            observed=False,
        )
        .reset_index()
        .rename(columns={"Deaths": "Mortality", "Incidence": "Incidence"})
    )
    wide["Mortality-to-incidence ratio"] = wide["Mortality"] / wide["Incidence"]
    endpoint_order = ["Incidence", "Mortality", "Mortality-to-incidence ratio"]
    long = wide.melt(
        id_vars=["year", "age_name"],
        value_vars=endpoint_order,
        var_name="endpoint",
        value_name="value",
    )
    long["endpoint"] = pd.Categorical(long["endpoint"], categories=endpoint_order, ordered=True)

    # Facet by endpoint so incidence, mortality, and the ratio have separate y scales.
    g = sns.relplot(
        data=long,
        x="year",
        y="value",
        hue="age_name",
        hue_order=CONDENSED_AGES,
        palette=AGE_COLORS,
        col="endpoint",
        col_order=endpoint_order,
        kind="line",
        marker="o",
        linewidth=EDA_LINEWIDTH,
        markersize=EDA_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        facet_kws={"sharey": False},
        height=12,
        aspect=1.25,
    )
    g.set_axis_labels("Year", "Value")
    g.set_titles("{col_name}")
    style_facet_grid(g)
    # Keep the age-group legend outside the plot area.
    sns.move_legend(g, "center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Age group")
    bold_figure_legend(g.figure)
    g.figure.suptitle(
        "Ghana TB endpoints by age group, Both sexes, 1990-2023",
        fontsize=52,
        fontweight="bold",
        y=1.02,
    )
    # Leave space for the legend and title.
    g.figure.subplots_adjust(right=0.72, top=0.88)
    g.figure.savefig(EDA_FIGURE_DIR / "condensed_age_group_rates_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def plot_sex_by_condensed_age(raw: pd.DataFrame) -> None:
    set_eda_theme()
    # Compare male and female trends within condensed age bands. "Both" is
    # excluded here because the focus is sex differences.
    subset = raw[
        raw["measure_name"].isin(["Incidence", "Deaths"])
        & raw["metric_name"].eq("Rate")
        & raw["sex_name"].isin(["Male", "Female"])
        & raw["age_name"].isin(CONDENSED_AGES[:-2])
    ].copy()
    # Exclude All ages and Age-standardized from this age-band panel and retain
    # the intended order.
    age_order = CONDENSED_AGES[:-2]
    subset["age_name"] = pd.Categorical(subset["age_name"], categories=age_order, ordered=True)
    wide = (
        subset.pivot_table(
            index=["year", "sex_name", "age_name"],
            columns="measure_name",
            values="val",
            aggfunc="first",
            observed=False,
        )
        .reset_index()
        .rename(columns={"Deaths": "Mortality", "Incidence": "Incidence"})
    )
    wide["Mortality-to-incidence ratio"] = wide["Mortality"] / wide["Incidence"]
    endpoint_order = ["Incidence", "Mortality", "Mortality-to-incidence ratio"]
    long = wide.melt(
        id_vars=["year", "sex_name", "age_name"],
        value_vars=endpoint_order,
        var_name="endpoint",
        value_name="value",
    )
    long["endpoint"] = pd.Categorical(long["endpoint"], categories=endpoint_order, ordered=True)

    # Grid layout: rows are endpoint types, columns are condensed age groups.
    g = sns.relplot(
        data=long,
        x="year",
        y="value",
        hue="sex_name",
        hue_order=["Female", "Male"],
        palette=SEX_COLORS,
        row="endpoint",
        row_order=endpoint_order,
        col="age_name",
        col_order=age_order,
        kind="line",
        marker="o",
        linewidth=EDA_LINEWIDTH,
        markersize=EDA_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=1.2,
        facet_kws={"sharey": False},
        height=8.0,
        aspect=1.25,
    )
    g.set_axis_labels("Year", "Value")
    g.set_titles("{row_name} | {col_name}")
    style_facet_grid(g)
    # Place the sex legend in the upper-right corner of the figure.
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, title="Sex")
    bold_figure_legend(g.figure)
    g.figure.suptitle(
        "Ghana TB endpoints by sex and age group, 1990-2023",
        fontsize=52,
        fontweight="bold",
        y=1.03,
    )
    g.figure.tight_layout(rect=(0, 0, 1, 0.91))
    g.figure.savefig(EDA_FIGURE_DIR / "sex_by_condensed_age_rates_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def run_stationarity(wide: pd.DataFrame) -> None:
    # These rows store every test result for every d and every modeling series.
    diagnostics = []
    # These rows store only the recommended differencing order per series.
    recommendations = []
    for series in PRIMARY_ANALYSIS_SERIES:
        # Pull one univariate series from the wide modeling table.
        values = wide[series].to_numpy(dtype=float)
        # Run ADF/KPSS after d=0, d=1, and d=2 transformations.
        rows = stationarity_by_d(values, max_d=2)
        # Choose a recommended d from the test outcomes.
        recommended_d, rule = choose_recommended_d(rows)
        # Recover readable GBD labels for the output table.
        measure, age, metric = TARGET_MAP[series]
        for row in rows:
            # Add contextual labels to each stationarity-test row.
            diagnostics.append(
                {
                    "series": series,
                    "measure": measure,
                    "age": age,
                    "metric": metric,
                    **row,
                }
            )
        # Pull the test row corresponding to the selected d so its p-values can
        # be reported next to the recommendation.
        selected_row = next(row for row in rows if row["d"] == recommended_d)
        recommendations.append(
            {
                "series": series,
                "measure": measure,
                "age": age,
                "metric": metric,
                "recommended_d": recommended_d,
                "selection_rule": rule,
                "selected_adf_p": selected_row["adf_p"],
                "selected_kpss_p": selected_row["kpss_p"],
                "selected_decision": selected_row["decision"],
            }
        )

    # Full diagnostic table: every series x every differencing order.
    pd.DataFrame(diagnostics).to_csv(TABLE_DIR / "stationarity_tests_by_d.csv", index=False)
    # Compact table: one recommended d per series.
    pd.DataFrame(recommendations).to_csv(TABLE_DIR / "recommended_d_by_series.csv", index=False)


def main() -> None:
    # Ensure output folders exist before any plot/table writing starts.
    ensure_dirs()
    # Set a consistent visual theme for all EDA plots.
    sns.set_theme(style="whitegrid", context="talk")
    # Raw data are needed for sex- and age-stratified plots.
    raw = pd.read_csv(RAW_PATH)
    # Wide processed data are needed for modeling-series and stationarity plots.
    wide = pd.read_csv(WIDE_PATH)

    # Plot the three primary modeling series.
    plot_modeling_series(wide)
    # Plot the three primary age-standardized endpoints.
    plot_primary_age_standardized_series(wide)
    # Show rolling mean and variability bands for the three primary series.
    plot_primary_rolling_statistics(wide)
    # Plot bar comparisons of the three primary endpoints to visually compare their levels and trends.
    plot_primary_age_standardized_bar_comparison(wide)
    # Plot the ratio of mortality to incidence rates. This ratio provides insight into the relationship between disease burden and death burden over time.
    plot_mortality_to_incidence_ratio(wide)
    # Plot sex-stratified all-age and age-standardized rates.
    plot_sex_stratified(raw)
    # Plot primary age-standardized endpoints by sex.
    plot_primary_age_standardized_by_sex(raw)
    # Plot ACF/PACF diagnostics for the three primary series.
    plot_primary_acf_pacf(wide)
    # Plot exploratory trend/cycle decomposition for the three primary series.
    plot_primary_decomposition(wide)
    # Plot condensed age-group trends for Both sexes.
    plot_condensed_age_groups(raw)
    # Plot sex differences within condensed age groups.
    plot_sex_by_condensed_age(raw)
    # Save ADF/KPSS stationarity tests and recommended differencing orders.
    run_stationarity(wide)

    # Print the main output locations for the user.
    print(f"Saved EDA figures to {EDA_FIGURE_DIR}")
    print(f"Saved stationarity tests to {TABLE_DIR / 'stationarity_tests_by_d.csv'}")
    print(f"Saved recommended differencing orders to {TABLE_DIR / 'recommended_d_by_series.csv'}")


if __name__ == "__main__":
    main()

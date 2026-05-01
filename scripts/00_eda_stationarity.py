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

# The manuscript's primary endpoints: age-standardized incidence and mortality
# rates for Both sexes.
PRIMARY_SERIES = [
    "incidence_age_standardized_rate",
    "mortality_age_standardized_rate",
]

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
    # Human-readable labels for figure titles.
    labels = {
        "incidence_age_standardized_rate": "Incidence age-standardized rate",
        "incidence_all_age_rate": "Incidence all-age rate",
        "incidence_all_age_number": "Incidence all-age number",
        "mortality_age_standardized_rate": "Mortality age-standardized rate",
        "mortality_all_age_rate": "Mortality all-age rate",
        "mortality_all_age_number": "Mortality all-age number",
    }
    # Convert the wide table to long format so seaborn can facet by series.
    long = wide.melt(id_vars="year", value_vars=MODELING_SERIES, var_name="series", value_name="value")
    # Attach readable labels to each processed series name.
    long["label"] = long["series"].map(labels)

    # Plot all prepared modeling series in a 2-column faceted layout. sharey=False
    # is important because rates and counts are on different scales.
    g = sns.relplot(
        data=long,
        x="year",
        y="value",
        col="label",
        col_wrap=2,
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4,
        aspect=1.5,
    )
    # Set common axis labels for all facets.
    g.set_axis_labels("Year", "Value")
    # Use each series label as the facet title.
    g.set_titles("{col_name}")
    # Add one title for the whole figure.
    g.figure.suptitle("Ghana TB modeling series, 1990-2023", y=1.02)
    # Tighten layout and save at print-friendly resolution.
    g.figure.tight_layout()
    g.figure.savefig(EDA_FIGURE_DIR / "modeling_series_over_time.png", dpi=300, bbox_inches="tight")
    # Close the figure to avoid memory buildup when the script creates many
    # plots.
    plt.close(g.figure)


def plot_primary_age_standardized_series(wide: pd.DataFrame) -> None:
    # Only the two primary rate endpoints are plotted here.
    labels = {
        "incidence_age_standardized_rate": "Incidence age-standardized rate",
        "mortality_age_standardized_rate": "Mortality age-standardized rate",
    }
    # A side-by-side layout makes incidence and mortality trends easy to compare.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    # Loop over the two primary rate series and draw one panel for each.
    for ax, series in zip(axes, PRIMARY_SERIES):
        ax.plot(wide["year"], wide[series], marker="o", linewidth=2.4)
        ax.set_title(labels[series])
        ax.set_xlabel("Year")
        ax.set_ylabel("Rate per 100,000")
        ax.margins(x=0.02)
                # Keep only the left and bottom axes so each panel has an L-shaped frame.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Make ticks appear only on the visible axes.
        ax.tick_params(top=False, right=False)

        # Optional: slightly thicken the visible L-shaped axes.
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
    fig.suptitle("Trend of Tuberculosis incidence and mortality rates in Ghana (1990-2023)", y=1.02)
    fig.tight_layout()
    fig.savefig(EDA_FIGURE_DIR / "Trend_of_TB_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_primary_age_standardized_bar_comparison(wide: pd.DataFrame) -> None:
    # Keep the bar comparison limited to the two primary rate endpoints so both bars
    # use the same "rate per 100,000" unit.
    labels = {
        "incidence_age_standardized_rate": "Incidence",
        "mortality_age_standardized_rate": "Mortality",
    }

    # Convert the wide table to long format so seaborn can draw grouped bars
    # with one bar for incidence and one bar for mortality within each year.
    long = wide.melt(
        id_vars="year",
        value_vars=PRIMARY_SERIES,
        var_name="series",
        value_name="rate",
    )

    # Replace machine-readable column names with short legend labels.
    long["endpoint"] = long["series"].map(labels)

    # A wide figure is used because the annual data include 34 years, and each
    # year has two bars.
    fig, ax = plt.subplots(figsize=(20, 7))

    # Draw grouped bars: x-axis is year, bar color identifies incidence versus
    # mortality.
    sns.barplot(
        data=long,
        x="year",
        y="rate",
        hue="endpoint",
        palette={"Incidence": "tab:blue", "Mortality": "tab:red"},
        ax=ax,
    )

    # Use a concise manuscript-style title and axis labels.
    ax.set_title("Age-standardized tuberculosis incidence and mortality rates in Ghana")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate per 100,000")

    # Show every other year label so the x-axis remains readable.
    for index, label in enumerate(ax.get_xticklabels()):
        label.set_visible(index % 2 == 0)
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # Keep only the left and bottom axes so the bar chart also has an L-shaped
    # frame.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(top=False, right=False)

    # Move the legend outside the plotting area so it does not cover bars.
    sns.move_legend(
        ax,
        "center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Endpoint",
    )

    # Reserve right-side space for the external legend and save the figure.
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(
        EDA_FIGURE_DIR / "primary_age_standardized_incidence_mortality_bar_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_mortality_to_incidence_ratio(wide: pd.DataFrame) -> None:
    # The mortality-to-incidence ratio compares death burden against incident
    # disease burden using the two primary age-standardized rates.
    ratio = (
        wide["mortality_age_standardized_rate"].astype(float)
        / wide["incidence_age_standardized_rate"].astype(float)
    )

    # Create a compact one-panel figure for the ratio trend.
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the ratio as a line so the direction of relative mortality burden is
    # easy to follow over time.
    ax.plot(
        wide["year"],
        ratio,
        marker="o",
        linewidth=2.4,
        color="tab:purple",
        label="Observed ratio",
    )

    # Add a dashed fitted trend line to summarize the long-run direction.
    sns.regplot(
        x=wide["year"],
        y=ratio,
        scatter=False,
        ci=None,
        color="black",
        line_kws={"linestyle": "--", "linewidth": 1.8, "alpha": 0.75},
        ax=ax,
    )

    # Use clear labels that state this is a ratio, not a rate per 100,000.
    ax.set_title("Mortality-to-incidence ratio for age-standardized TB rates in Ghana")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mortality rate / incidence rate")
    ax.margins(x=0.02)

    # Keep only the left and bottom axes for an L-shaped frame.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(top=False, right=False)

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
    # Move the legend outside the plot area so it does not cover lines.
    sns.move_legend(g, "center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Sex")
    g.figure.suptitle("Ghana TB rates by sex, 1990-2023", y=1.02)
    # Reserve right-side whitespace for the external legend.
    g.despine(left=True, bottom=True)
    g.figure.subplots_adjust(right=0.84, top=0.9)
    g.figure.savefig(EDA_FIGURE_DIR / "sex_stratified_rates_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def plot_primary_age_standardized_by_sex(raw: pd.DataFrame) -> None:
    # Focus on the primary age-standardized rate definition but show sex strata.
    subset = raw[
        raw["measure_name"].isin(["Incidence", "Deaths"])
        & raw["metric_name"].eq("Rate")
        & raw["age_name"].eq("Age-standardized")
    ].copy()

    # One panel for incidence and one panel for deaths.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    # Save legend handles from seaborn so one shared legend can be placed
    # outside the axes.
    handles = labels = None
    for ax, measure in zip(axes, ["Incidence", "Deaths"]):
        # Subset to one measure for this panel.
        panel = subset[subset["measure_name"] == measure]
        # Draw sex-specific lines.
        sns.lineplot(data=panel, x="year", y="val", hue="sex_name", marker="o", linewidth=2.2, ax=ax)
        # Grab handles before removing the per-axis legend.
        handles, labels = ax.get_legend_handles_labels()
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title(f"{measure} age-standardized rate")
        ax.set_xlabel("Year")
        ax.set_ylabel("Rate per 100,000")
        ax.margins(x=0.02)
    if handles and labels:
        # Add one shared legend outside the figure.
        fig.legend(handles, labels, title="Sex", loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False)
    fig.suptitle("Primary Ghana TB age-standardized rates by sex, 1990-2023", y=1.02)
    # Reserve right-side margin for the external legend.
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    fig.savefig(EDA_FIGURE_DIR / "primary_age_standardized_rates_by_sex.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_primary_acf_pacf(wide: pd.DataFrame) -> None:
    # ACF/PACF diagnostics help assess how much each year depends on prior
    # years and can inform ARIMA p/q choices.
    labels = {
        "incidence_age_standardized_rate": "Incidence age-standardized rate",
        "mortality_age_standardized_rate": "Mortality age-standardized rate",
    }
    # Limit lags to keep plots readable and statistically sensible for only
    # 34 annual observations.
    max_lags = 12

    for series in PRIMARY_SERIES:
        # Level series: the original age-standardized rate.
        values = wide[series].to_numpy(dtype=float)
        # First difference: annual change in the rate.
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

        fig.suptitle(f"{labels[series]} autocorrelation diagnostics", y=1.02)
        fig.tight_layout()
        fig.savefig(EDA_FIGURE_DIR / f"acf_pacf_{series}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_primary_decomposition(wide: pd.DataFrame) -> None:
    # Annual data do not have monthly/quarterly seasonality. This decomposition
    # is therefore labeled as an exploratory 5-year cycle, not true seasonality.
    labels = {
        "incidence_age_standardized_rate": "Incidence age-standardized rate",
        "mortality_age_standardized_rate": "Mortality age-standardized rate",
    }
    # Collect decomposition components for export to CSV.
    decomposition_rows = []
    # Five years gives a compact visual cycle for annual public-health data.
    period = 5

    for series in PRIMARY_SERIES:
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

        # Four stacked panels show observed rate, estimated trend, cyclical
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
            ax.set_ylabel("Rate")
            ax.margins(x=0.02)
        axes[-1].set_xlabel("Year")
        fig.suptitle(f"{labels[series]} decomposition (period = {period} years)", y=1.01)
        fig.tight_layout()
        fig.savefig(EDA_FIGURE_DIR / f"decomposition_{series}_period5.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Export decomposition values so they can be inspected or reported in tables.
    pd.concat(decomposition_rows, ignore_index=True).to_csv(
        TABLE_DIR / "primary_age_standardized_decomposition_period5.csv",
        index=False,
    )


def plot_condensed_age_groups(raw: pd.DataFrame) -> None:
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

    # Facet by measure so incidence and mortality have separate y scales.
    g = sns.relplot(
        data=subset,
        x="year",
        y="val",
        hue="age_name",
        col="measure_name",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=5,
        aspect=1.35,
    )
    g.set_axis_labels("Year", "Rate per 100,000")
    g.set_titles("{col_name}")
    # Keep the age-group legend outside the plot area.
    sns.move_legend(g, "center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Age group")
    g.figure.suptitle("Ghana TB rates by condensed age group, Both sexes, 1990-2023", y=1.04)
    # Leave space for the legend and title.
    g.figure.subplots_adjust(right=0.78, top=0.86)
    g.figure.savefig(EDA_FIGURE_DIR / "condensed_age_group_rates_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def plot_sex_by_condensed_age(raw: pd.DataFrame) -> None:
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
    subset["age_name"] = pd.Categorical(subset["age_name"], categories=CONDENSED_AGES[:-2], ordered=True)

    # Grid layout: rows are measure types, columns are condensed age groups.
    g = sns.relplot(
        data=subset,
        x="year",
        y="val",
        hue="sex_name",
        row="measure_name",
        col="age_name",
        kind="line",
        facet_kws={"sharey": False},
        height=3,
        aspect=1.25,
    )
    g.set_axis_labels("Year", "Rate per 100,000")
    g.set_titles("{row_name} | {col_name}")
    # Place the sex legend outside the plot grid.
    sns.move_legend(g, "center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Sex")
    g.figure.suptitle("Ghana TB rates by sex and condensed age group, 1990-2023", y=1.02)
    # Reserve space for the external legend and title.
    g.figure.subplots_adjust(right=0.9, top=0.88)
    g.figure.savefig(EDA_FIGURE_DIR / "sex_by_condensed_age_rates_over_time.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def run_stationarity(wide: pd.DataFrame) -> None:
    # These rows store every test result for every d and every modeling series.
    diagnostics = []
    # These rows store only the recommended differencing order per series.
    recommendations = []
    for series in MODELING_SERIES:
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

    # Plot all prepared modeling series.
    #plot_modeling_series(wide)
    # Plot the two primary age-standardized rate endpoints.
    plot_primary_age_standardized_series(wide)
    # Plot a bar comparison of the two primary age-standardized rates to visually compare their levels and trends.
    plot_primary_age_standardized_bar_comparison(wide)
    # Plot the ratio of mortality to incidence rates. This ratio provides insight into the relationship between disease burden and death burden over time.
    plot_mortality_to_incidence_ratio(wide)
    # Plot sex-stratified all-age and age-standardized rates.
    plot_sex_stratified(raw)
    # Plot primary age-standardized rates by sex.
    plot_primary_age_standardized_by_sex(raw)
    # Plot ACF/PACF diagnostics for primary endpoints.
    plot_primary_acf_pacf(wide)
    # Plot exploratory trend/cycle decomposition for primary endpoints.
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

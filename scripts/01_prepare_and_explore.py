"""Prepare Ghana TB GBD data and generate first trend-analysis outputs.

The raw IHME export is long-form and contains many overlapping age groups,
sexes, and metrics. This script creates a modeling-ready annual table using
Both sexes and the national all-age / age-standardized series needed for
forecasting incidence and mortality from 1990 to 2023.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "TB_df" / "IHME-GBD_2023_DATA-42460a02-1.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures" / "trend"

TARGETS = {
    ("Incidence", "Age-standardized", "Rate"): "incidence_age_standardized_rate",
    ("Incidence", "All ages", "Rate"): "incidence_all_age_rate",
    ("Incidence", "All ages", "Number"): "incidence_all_age_number",
    ("Deaths", "Age-standardized", "Rate"): "mortality_age_standardized_rate",
    ("Deaths", "All ages", "Rate"): "mortality_all_age_rate",
    ("Deaths", "All ages", "Number"): "mortality_all_age_number",
}

DERIVED_RATIO_SERIES = "mortality_to_incidence_age_standardized_ratio"


def ensure_dirs() -> None:
    for directory in (PROCESSED_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    expected = {
        "measure_name",
        "location_name",
        "sex_name",
        "age_name",
        "cause_name",
        "metric_name",
        "year",
        "val",
        "lower",
        "upper",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing required columns: {sorted(missing)}")
    return df


def build_modeling_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_keys = set(TARGETS)
    filtered = df[
        (df["location_name"] == "Ghana")
        & (df["cause_name"] == "Tuberculosis")
        & (df["sex_name"] == "Both")
        & df[["measure_name", "age_name", "metric_name"]]
        .apply(tuple, axis=1)
        .isin(target_keys)
    ].copy()

    duplicate_count = filtered.duplicated(
        ["measure_name", "age_name", "metric_name", "year"]
    ).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate target-year rows")

    filtered["series"] = filtered[["measure_name", "age_name", "metric_name"]].apply(
        lambda row: TARGETS[tuple(row)], axis=1
    )

    modeling = filtered.pivot(index="year", columns="series", values="val").reset_index()
    lower = (
        filtered.pivot(index="year", columns="series", values="lower")
        .add_suffix("_lower")
        .reset_index()
    )
    upper = (
        filtered.pivot(index="year", columns="series", values="upper")
        .add_suffix("_upper")
        .reset_index()
    )

    modeling = modeling.merge(lower, on="year").merge(upper, on="year")
    modeling = modeling.sort_values("year").reset_index(drop=True)

    modeling[DERIVED_RATIO_SERIES] = (
        modeling["mortality_age_standardized_rate"] / modeling["incidence_age_standardized_rate"]
    )
    modeling[f"{DERIVED_RATIO_SERIES}_lower"] = (
        modeling["mortality_age_standardized_rate_lower"] / modeling["incidence_age_standardized_rate_upper"]
    )
    modeling[f"{DERIVED_RATIO_SERIES}_upper"] = (
        modeling["mortality_age_standardized_rate_upper"] / modeling["incidence_age_standardized_rate_lower"]
    )

    ratio_long = modeling[
        [
            "year",
            DERIVED_RATIO_SERIES,
            f"{DERIVED_RATIO_SERIES}_upper",
            f"{DERIVED_RATIO_SERIES}_lower",
        ]
    ].copy()
    ratio_long = ratio_long.rename(
        columns={
            DERIVED_RATIO_SERIES: "val",
            f"{DERIVED_RATIO_SERIES}_upper": "upper",
            f"{DERIVED_RATIO_SERIES}_lower": "lower",
        }
    )
    ratio_long["population_group_id"] = 1
    ratio_long["population_group_name"] = "All Population"
    ratio_long["measure_id"] = pd.NA
    ratio_long["measure_name"] = "Mortality-to-incidence ratio"
    ratio_long["location_id"] = 207
    ratio_long["location_name"] = "Ghana"
    ratio_long["sex_id"] = 3
    ratio_long["sex_name"] = "Both"
    ratio_long["age_id"] = 27
    ratio_long["age_name"] = "Age-standardized"
    ratio_long["cause_id"] = 297
    ratio_long["cause_name"] = "Tuberculosis"
    ratio_long["metric_id"] = pd.NA
    ratio_long["metric_name"] = "Ratio"
    ratio_long["series"] = DERIVED_RATIO_SERIES

    long_columns = list(filtered.columns)
    filtered = pd.concat([filtered, ratio_long[long_columns]], ignore_index=True)

    expected_years = set(range(1990, 2024))
    actual_years = set(modeling["year"])
    if actual_years != expected_years:
        raise ValueError(f"Unexpected year coverage. Missing: {sorted(expected_years - actual_years)}")

    return filtered.sort_values(["series", "year"]), modeling


def make_summary_tables(modeling: pd.DataFrame) -> None:
    series_cols = [col for col in modeling.columns if not col.endswith(("_lower", "_upper"))]
    series_cols.remove("year")

    rows = []
    for col in series_cols:
        series = modeling[["year", col]].dropna().sort_values("year")
        first = series.iloc[0]
        last = series.iloc[-1]
        pct_change = ((last[col] / first[col]) - 1) * 100
        annual_change = ((last[col] / first[col]) ** (1 / (last["year"] - first["year"])) - 1) * 100
        min_row = series.loc[series[col].idxmin()]
        max_row = series.loc[series[col].idxmax()]
        rows.append(
            {
                "series": col,
                "value_1990": first[col],
                "value_2023": last[col],
                "percent_change_1990_2023": pct_change,
                "average_annual_percent_change": annual_change,
                "minimum_year": int(min_row["year"]),
                "minimum_value": min_row[col],
                "maximum_year": int(max_row["year"]),
                "maximum_value": max_row[col],
            }
        )

    trend_summary = pd.DataFrame(rows)
    trend_summary.to_csv(TABLE_DIR / "trend_summary_1990_2023.csv", index=False)

    yoy = modeling[["year", *series_cols]].copy()
    for col in series_cols:
        yoy[col] = yoy[col].pct_change() * 100
    yoy.to_csv(TABLE_DIR / "year_over_year_percent_change.csv", index=False)

    diagnostics = []
    for col in series_cols:
        values = modeling[col].to_numpy()
        diffs = modeling[col].diff().dropna().to_numpy()
        diagnostics.append(
            {
                "series": col,
                "n_observations": len(values),
                "increasing_years": int((modeling[col].diff() > 0).sum()),
                "decreasing_years": int((modeling[col].diff() < 0).sum()),
                "adf_level_p": safe_adf(values),
                "kpss_level_p": safe_kpss(values),
                "adf_first_difference_p": safe_adf(diffs),
                "kpss_first_difference_p": safe_kpss(diffs),
            }
        )
    pd.DataFrame(diagnostics).to_csv(TABLE_DIR / "time_series_diagnostics.csv", index=False)


def safe_adf(values) -> float:
    try:
        return float(adfuller(values, autolag="AIC")[1])
    except Exception:
        return float("nan")


def safe_kpss(values) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            return float(kpss(values, regression="c", nlags="auto")[1])
    except Exception:
        return float("nan")


def plot_trends(modeling: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    plot_specs = [
        (
            "incidence_rates_trend.png",
            "Tuberculosis incidence rates in Ghana, 1990-2023",
            ["incidence_age_standardized_rate", "incidence_all_age_rate"],
            "Rate per 100,000",
        ),
        (
            "mortality_rates_trend.png",
            "Tuberculosis mortality rates in Ghana, 1990-2023",
            ["mortality_age_standardized_rate", "mortality_all_age_rate"],
            "Rate per 100,000",
        ),
        (
            "incidence_mortality_numbers_trend.png",
            "Tuberculosis incidence and mortality counts in Ghana, 1990-2023",
            ["incidence_all_age_number", "mortality_all_age_number"],
            "Number",
        ),
    ]

    labels = {
        "incidence_age_standardized_rate": "Incidence, age-standardized rate",
        "incidence_all_age_rate": "Incidence, all-age rate",
        "mortality_age_standardized_rate": "Mortality, age-standardized rate",
        "mortality_all_age_rate": "Mortality, all-age rate",
        "incidence_all_age_number": "Incidence, all-age number",
        "mortality_all_age_number": "Mortality, all-age number",
    }

    for filename, title, columns, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(12, 7))
        for col in columns:
            ax.plot(modeling["year"], modeling[col], marker="o", linewidth=2.4, label=labels[col])
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.margins(x=0.02)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / filename, dpi=300)
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    raw = load_raw_data()
    long_targets, modeling = build_modeling_table(raw)

    long_targets.to_csv(PROCESSED_DIR / "ghana_tb_modeling_long.csv", index=False)
    modeling.to_csv(PROCESSED_DIR / "ghana_tb_modeling_wide.csv", index=False)

    make_summary_tables(modeling)
    plot_trends(modeling)

    print(f"Saved modeling data to {PROCESSED_DIR}")
    print(f"Saved tables to {TABLE_DIR}")
    print(f"Saved figures to {FIGURE_DIR}")
    print(f"Wide modeling shape: {modeling.shape[0]} rows x {modeling.shape[1]} columns")


if __name__ == "__main__":
    main()

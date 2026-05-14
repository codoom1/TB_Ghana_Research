"""Propagate GBD uncertainty intervals through selected forecast models.

Run after `04_forecast_best_models.py`:

    python scripts/05_forecast_uncertainty.py --n-sim 100 --epochs 50
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

import model_arima
import model_arima_lstm
import model_causal_tcn
import model_ets
import model_lstm
import model_multistep_lstm
from model_baselines import drift_forecast, naive_forecast
from model_common import (
    FORECAST_DIR,
    FORECAST_END,
    FORECAST_START,
    MODEL_OUTPUT_DIR,
    PRIMARY_SERIES,
    SERIES,
    TABLE_DIR,
    UNCERTAINTY_FIGURE_DIR,
    ensure_output_dirs,
    load_modeling_data,
)


SERIES_LABELS = {
    "incidence_age_standardized_rate": "Incidence age-standardized rate",
    "mortality_to_incidence_age_standardized_ratio": "Mortality-to-incidence ratio",
    "incidence_all_age_rate": "Incidence all-age rate",
    "incidence_all_age_number": "Incidence all-age number",
    "mortality_age_standardized_rate": "Mortality age-standardized rate",
    "mortality_all_age_rate": "Mortality all-age rate",
    "mortality_all_age_number": "Mortality all-age number",
}

PLOT_TITLE_SIZE = 30
PLOT_AXIS_LABEL_SIZE = 22
PLOT_TICK_SIZE = 18
PLOT_LEGEND_SIZE = 30
PLOT_LINEWIDTH = 7.0
PLOT_MARKERSIZE = 10.5
PLOT_MARKEREDGEWIDTH = 1.2
PLOT_OBSERVED_COLOR = "#3F3F3F"
PLOT_FORECAST_COLOR = "#0072B2"
PLOT_GBD_FILL = "#56B4E9"
PLOT_COMBINED_FILL = "#E69F00"
PLOT_GBD_HATCH = "///"
PLOT_COMBINED_HATCH = "\\\\"


def series_y_label(series: str) -> str:
    if "ratio" in series:
        return "Mortality / incidence"
    if "rate" in series:
        return "Rate per 100,000"
    return "Number"


def model_registry(epochs: int):
    return {
        "Naive": lambda train, steps, **_: naive_forecast(train, steps),
        "Drift": lambda train, steps, **_: drift_forecast(train, steps),
        "ARIMA": lambda train, steps, **kwargs: model_arima.forecast(train, steps, **kwargs),
        "ETS": lambda train, steps, **_: model_ets.forecast(train, steps),
        "LSTM": lambda train, steps, **_: model_lstm.forecast(train, steps, epochs=epochs),
        "Causal_TCN": lambda train, steps, **_: model_causal_tcn.forecast(train, steps, epochs=epochs),
        "ARIMA_LSTM": lambda train, steps, **kwargs: model_arima_lstm.forecast(train, steps, epochs=epochs, **kwargs),
        "Multistep_LSTM": lambda train, steps, **_: model_multistep_lstm.forecast(train, steps, epochs=epochs),
    }


def sample_pert(
    lower: np.ndarray,
    mode: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    shape: float = 4.0, ## The standard PERT distribution uses a shape of 4, which gives a reasonable balance of central concentration and tail weight. Adjust as needed for more or less variability.
) -> np.ndarray:
    # A scaled beta-PERT distribution respects the GBD lower and upper bounds
    # while centering simulated histories around the reported point estimate.
    lower = np.asarray(lower, dtype=float)
    mode = np.asarray(mode, dtype=float)
    upper = np.asarray(upper, dtype=float)
    width = upper - lower
    sampled = mode.copy()
    valid = np.isfinite(lower) & np.isfinite(mode) & np.isfinite(upper) & (width > 0)
    if not np.any(valid):
        return sampled

    relative_mode = np.clip((mode[valid] - lower[valid]) / width[valid], 1e-6, 1 - 1e-6)
    alpha = 1 + shape * relative_mode
    beta = 1 + shape * (1 - relative_mode)
    sampled[valid] = lower[valid] + rng.beta(alpha, beta) * width[valid]
    sampled = np.maximum(sampled, 0)
    return sampled


def load_holdout_residuals(
    prediction_path=MODEL_OUTPUT_DIR / "all_model_test_predictions_2018_2023.csv",
) -> dict[tuple[str, str], np.ndarray]:
    """Load centered holdout residuals by series and model.

    Residuals are defined as actual minus predicted values on the 2018-2023
    holdout period. Centering keeps the combined simulations around the point
    forecast while using the empirical holdout error spread.
    """
    if not prediction_path.exists():
        return {}

    predictions = pd.read_csv(prediction_path)
    required = {"series", "model", "actual", "predicted"}
    if not required.issubset(predictions.columns):
        return {}

    residual_lookup = {}
    predictions = predictions.copy()
    predictions["residual"] = predictions["actual"].astype(float) - predictions["predicted"].astype(float)
    for (series, model_name), group in predictions.groupby(["series", "model"]):
        residuals = group["residual"].to_numpy(dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals):
            residual_lookup[(str(series), str(model_name))] = residuals - np.mean(residuals)
    return residual_lookup


def sample_model_error(
    residuals: np.ndarray | None,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap centered holdout residuals and scale modestly by horizon."""
    if residuals is None:
        return np.zeros(steps, dtype=float)

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0 or np.isclose(np.std(residuals), 0):
        return np.zeros(steps, dtype=float)

    sampled = rng.choice(residuals, size=steps, replace=True)
    holdout_horizon_mean = np.mean(np.arange(1, len(residuals) + 1, dtype=float))
    horizon_scale = np.sqrt(np.arange(1, steps + 1, dtype=float) / holdout_horizon_mean)
    return sampled * horizon_scale


def simulate_forecast_intervals(
    data: pd.DataFrame,
    best_models: pd.DataFrame,
    n_sim: int,
    epochs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = model_registry(epochs)
    steps = FORECAST_END - FORECAST_START + 1
    forecast_years = list(range(FORECAST_START, FORECAST_END + 1))
    rng = np.random.default_rng(seed)
    residual_lookup = load_holdout_residuals()
    simulation_rows = []
    summary_rows = []

    for _, selected in tqdm(
        best_models.iterrows(),
        total=len(best_models),
        desc="Simulating forecast uncertainty",
        unit="series",
    ):
        series = str(selected["series"])
        model_name = str(selected["model"])
        lower_col = f"{series}_lower"
        upper_col = f"{series}_upper"

        if model_name not in models:
            raise ValueError(f"Unknown model '{model_name}' for series '{series}'")
        if lower_col not in data.columns or upper_col not in data.columns:
            raise ValueError(f"Missing uncertainty columns for series '{series}'")

        mode = data[series].to_numpy(dtype=float)
        lower = data[lower_col].to_numpy(dtype=float)
        upper = data[upper_col].to_numpy(dtype=float)
        runner = models[model_name]
        model_residuals = residual_lookup.get((series, model_name))
        gbd_paths = []
        combined_paths = []

        for sim_id in range(n_sim):
            sampled_history = sample_pert(lower, mode, upper, rng)
            try:
                model_kwargs = {}
                if model_name in {"ARIMA", "ARIMA_LSTM"}:
                    arima_selected_d, arima_d_reason = model_arima.select_d(sampled_history)
                    model_kwargs = {
                        "selected_d": arima_selected_d,
                        "d_reason": f"{arima_d_reason}; source=simulated_history",
                    }
                result = runner(sampled_history, steps, **model_kwargs)
                gbd_forecast = np.maximum(np.asarray(result.predictions, dtype=float), 0)
            except Exception:
                gbd_forecast = np.repeat(np.nan, steps)

            model_error = sample_model_error(model_residuals, steps, rng)
            combined_forecast = np.maximum(gbd_forecast + model_error, 0)

            gbd_paths.append(gbd_forecast)
            combined_paths.append(combined_forecast)
            for year, gbd_value, combined_value, error_value in zip(
                forecast_years,
                gbd_forecast,
                combined_forecast,
                model_error,
            ):
                simulation_rows.append(
                    {
                        "series": series,
                        "model": model_name,
                        "simulation": sim_id + 1,
                        "year": year,
                        "gbd_input_forecast": float(gbd_value) if np.isfinite(gbd_value) else np.nan,
                        "model_error": float(error_value) if np.isfinite(error_value) else np.nan,
                        "combined_forecast": float(combined_value) if np.isfinite(combined_value) else np.nan,
                    }
                )

        gbd_array = np.asarray(gbd_paths, dtype=float)
        combined_array = np.asarray(combined_paths, dtype=float)
        gbd_lower_q = np.nanpercentile(gbd_array, 2.5, axis=0)
        gbd_median_q = np.nanpercentile(gbd_array, 50, axis=0)
        gbd_upper_q = np.nanpercentile(gbd_array, 97.5, axis=0)
        gbd_mean_q = np.nanmean(gbd_array, axis=0)
        combined_lower_q = np.nanpercentile(combined_array, 2.5, axis=0)
        combined_median_q = np.nanpercentile(combined_array, 50, axis=0)
        combined_upper_q = np.nanpercentile(combined_array, 97.5, axis=0)
        combined_mean_q = np.nanmean(combined_array, axis=0)

        for (
            year,
            gbd_lower_value,
            gbd_median_value,
            gbd_upper_value,
            gbd_mean_value,
            combined_lower_value,
            combined_median_value,
            combined_upper_value,
            combined_mean_value,
        ) in zip(
            forecast_years,
            gbd_lower_q,
            gbd_median_q,
            gbd_upper_q,
            gbd_mean_q,
            combined_lower_q,
            combined_median_q,
            combined_upper_q,
            combined_mean_q,
        ):
            summary_rows.append(
                {
                    "series": series,
                    "selected_model": model_name,
                    "year": year,
                    "gbd_input_uncertainty_lower": float(gbd_lower_value),
                    "gbd_input_uncertainty_median": float(gbd_median_value),
                    "gbd_input_uncertainty_upper": float(gbd_upper_value),
                    "gbd_input_uncertainty_mean": float(gbd_mean_value),
                    "combined_uncertainty_lower": float(combined_lower_value),
                    "combined_uncertainty_median": float(combined_median_value),
                    "combined_uncertainty_upper": float(combined_upper_value),
                    "combined_uncertainty_mean": float(combined_mean_value),
                    # Backward-compatible names for the original GBD-input-only interval.
                    "forecast_uncertainty_lower": float(gbd_lower_value),
                    "forecast_uncertainty_median": float(gbd_median_value),
                    "forecast_uncertainty_upper": float(gbd_upper_value),
                    "forecast_uncertainty_mean": float(gbd_mean_value),
                    "n_simulations": n_sim,
                    "n_holdout_residuals": int(0 if model_residuals is None else len(model_residuals)),
                }
            )

    return pd.DataFrame(simulation_rows), pd.DataFrame(summary_rows)


def merge_with_point_forecasts(intervals: pd.DataFrame) -> pd.DataFrame:
    point_path = FORECAST_DIR / "best_model_forecasts_2024_2030.csv"
    point = pd.read_csv(point_path)
    merged = intervals.merge(
        point[["series", "year", "forecast", "selected_model"]],
        on=["series", "year", "selected_model"],
        how="left",
    )
    merged = merged.rename(columns={"forecast": "point_forecast"})
    return merged[[
            "series",
            "selected_model",
            "year",
            "point_forecast",
            "gbd_input_uncertainty_lower",
            "gbd_input_uncertainty_median",
            "gbd_input_uncertainty_upper",
            "gbd_input_uncertainty_mean",
            "combined_uncertainty_lower",
            "combined_uncertainty_median",
            "combined_uncertainty_upper",
            "combined_uncertainty_mean",
            "forecast_uncertainty_lower",
            "forecast_uncertainty_median",
            "forecast_uncertainty_upper",
            "forecast_uncertainty_mean",
            "n_simulations",
            "n_holdout_residuals",
        ]
        ].sort_values(["series", "year"])


def plot_uncertainty_forecasts(data: pd.DataFrame, intervals: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plot_series = [
        "incidence_age_standardized_rate",
        "mortality_age_standardized_rate",
        "mortality_to_incidence_age_standardized_ratio",
    ]
    plot_series = [series for series in plot_series if series in set(intervals["series"])]
    # Primary analysis has three endpoints, so this gives the requested 1x3
    # layout in the intended order. If --all-series is used later, the same code
    # wraps into rows of 3.
    ncols = min(3, max(1, len(plot_series)))
    nrows = int(np.ceil(len(plot_series) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5 * ncols, 13.8 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, series in zip(axes, plot_series):
        fc = intervals[intervals["series"] == series]
        model_name = fc["selected_model"].iloc[0]
        lower_col = f"{series}_lower"
        upper_col = f"{series}_upper"

        years_obs = data["year"].to_numpy(dtype=float)
        observed = data[series].to_numpy(dtype=float)
        observed_lower = data[lower_col].to_numpy(dtype=float)
        observed_upper = data[upper_col].to_numpy(dtype=float)
        years_fc = fc["year"].to_numpy(dtype=float)

        ax.fill_between(
            years_obs,
            observed_lower,
            observed_upper,
            color=PLOT_GBD_FILL,
            alpha=0.20,
            hatch=PLOT_GBD_HATCH,
            edgecolor=PLOT_GBD_FILL,
            label="GBD uncertainty",
            zorder=1,
        )
        ax.plot(
            years_obs,
            observed,
            color=PLOT_OBSERVED_COLOR,
            marker="o",
            markersize=PLOT_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            linewidth=PLOT_LINEWIDTH,
            label="Observed",
            zorder=3,
        )
        ax.fill_between(
            years_fc,
            fc["combined_uncertainty_lower"].to_numpy(dtype=float),
            fc["combined_uncertainty_upper"].to_numpy(dtype=float),
            color=PLOT_COMBINED_FILL,
            alpha=0.22,
            hatch=PLOT_COMBINED_HATCH,
            edgecolor=PLOT_COMBINED_FILL,
            label="Combined uncertainty",
            zorder=1,
        )
        ax.fill_between(
            years_fc,
            fc["gbd_input_uncertainty_lower"].to_numpy(dtype=float),
            fc["gbd_input_uncertainty_upper"].to_numpy(dtype=float),
            color=PLOT_FORECAST_COLOR,
            alpha=0.18,
            label="GBD-input forecast uncertainty",
            zorder=2,
        )
        ax.plot(
            years_fc,
            fc["point_forecast"],
            color=PLOT_FORECAST_COLOR,
            marker="o",
            linestyle="--",
            markersize=PLOT_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            linewidth=PLOT_LINEWIDTH,
            label="Point forecast",
            zorder=4,
        )
        ax.axvline(2023, color="black", linewidth=1.4, alpha=0.65)
        ax.set_title(f"{SERIES_LABELS[series]}\nBest model: {model_name}")
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        ax.title.set_fontsize(PLOT_TITLE_SIZE)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(PLOT_AXIS_LABEL_SIZE)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontsize(PLOT_AXIS_LABEL_SIZE)
        ax.yaxis.label.set_fontweight("bold")
        ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE, width=1.4, length=6)
        ax.margins(x=0.02)

    for ax in axes[len(plot_series) :]:
        ax.axis("off")

    observed_handle = plt.Line2D(
        [0],
        [0],
        color=PLOT_OBSERVED_COLOR,
        marker="o",
        linewidth=PLOT_LINEWIDTH,
        markersize=PLOT_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=PLOT_MARKEREDGEWIDTH,
        label="Observed",
    )
    gbd_handle = plt.Line2D(
        [0],
        [0],
        color=PLOT_FORECAST_COLOR,
        marker="o",
        linewidth=PLOT_LINEWIDTH,
        linestyle="--",
        markersize=PLOT_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=PLOT_MARKEREDGEWIDTH,
        label="Point forecast",
    )
    gbd_band_handle = Patch(
        facecolor=PLOT_GBD_FILL,
        edgecolor=PLOT_GBD_FILL,
        hatch=PLOT_GBD_HATCH,
        alpha=0.20,
        label="GBD uncertainty",
    )
    combined_handle = Patch(
        facecolor=PLOT_COMBINED_FILL,
        edgecolor=PLOT_COMBINED_FILL,
        hatch=PLOT_COMBINED_HATCH,
        alpha=0.22,
        label="Combined uncertainty",
    )
    fig.legend(
        [observed_handle, gbd_handle, gbd_band_handle, combined_handle],
        ["Observed", "Point forecast", "GBD uncertainty", "Combined uncertainty"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        prop={"size": PLOT_LEGEND_SIZE, "weight": "bold"},
        columnspacing=1.4,
        handlelength=2.4,
        handletextpad=0.6,
    )
    fig.suptitle(
        "Best-model forecasts with GBD-input and combined uncertainty, 2024-2030",
        y=0.995,
        fontsize=32,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.87))
    fig.savefig(UNCERTAINTY_FIGURE_DIR / "best_model_forecasts_2024_2030_with_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sim", type=int, default=100, help="Number of uncertainty simulations per series.")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for neural models during each simulation.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for uncertainty simulation.")
    parser.add_argument(
        "--best-model-path",
        default=str(TABLE_DIR / "best_models_by_series.csv"),
        help="Best-model table produced by the model evaluation script.",
    )
    parser.add_argument(
        "--all-series",
        action="store_true",
        help="Simulate uncertainty for all selected series. By default, uses primary endpoints.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    data = load_modeling_data()
    best = pd.read_csv(args.best_model_path)
    if not args.all_series:
        best = best[best["series"].isin(PRIMARY_SERIES)].copy()

    simulations, intervals = simulate_forecast_intervals(
        data=data,
        best_models=best,
        n_sim=args.n_sim,
        epochs=args.epochs,
        seed=args.seed,
    )
    intervals = merge_with_point_forecasts(intervals)

    simulations.to_csv(FORECAST_DIR / "forecast_uncertainty_simulations_2024_2030.csv", index=False)
    intervals.to_csv(FORECAST_DIR / "best_model_forecasts_2024_2030_with_uncertainty.csv", index=False)
    intervals.to_csv(TABLE_DIR / "best_model_forecasts_2024_2030_with_uncertainty.csv", index=False)
    plot_uncertainty_forecasts(data, intervals)

    print(f"Saved simulation draws: {FORECAST_DIR / 'forecast_uncertainty_simulations_2024_2030.csv'}")
    print(f"Saved forecast intervals: {FORECAST_DIR / 'best_model_forecasts_2024_2030_with_uncertainty.csv'}")
    print(f"Saved table copy: {TABLE_DIR / 'best_model_forecasts_2024_2030_with_uncertainty.csv'}")
    print(f"Saved figure: {UNCERTAINTY_FIGURE_DIR / 'best_model_forecasts_2024_2030_with_uncertainty.png'}")
    endpoints = intervals[intervals["year"].isin([FORECAST_START, FORECAST_END])]
    print("\nForecast intervals at endpoints:")
    print(
        endpoints[
            [
                "series",
                "year",
                "point_forecast",
                "gbd_input_uncertainty_lower",
                "gbd_input_uncertainty_upper",
                "combined_uncertainty_lower",
                "combined_uncertainty_upper",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

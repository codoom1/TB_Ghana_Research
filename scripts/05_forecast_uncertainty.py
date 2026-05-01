"""Propagate GBD uncertainty intervals through selected forecast models.

Run after `04_forecast_best_models.py`:

    python scripts/05_forecast_uncertainty.py --n-sim 100 --epochs 50
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
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


def series_y_label(series: str) -> str:
    if "ratio" in series:
        return "Mortality / incidence"
    if "rate" in series:
        return "Rate per 100,000"
    return "Number"


def model_registry(epochs: int):
    return {
        "Naive": lambda train, steps: naive_forecast(train, steps),
        "Drift": lambda train, steps: drift_forecast(train, steps),
        "ARIMA": lambda train, steps: model_arima.forecast(train, steps),
        "ETS": lambda train, steps: model_ets.forecast(train, steps),
        "LSTM": lambda train, steps: model_lstm.forecast(train, steps, epochs=epochs),
        "Causal_TCN": lambda train, steps: model_causal_tcn.forecast(train, steps, epochs=epochs),
        "ARIMA_LSTM": lambda train, steps: model_arima_lstm.forecast(train, steps, epochs=epochs),
        "Multistep_LSTM": lambda train, steps: model_multistep_lstm.forecast(train, steps, epochs=epochs),
    }


def sample_pert(
    lower: np.ndarray,
    mode: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    shape: float = 4.0,
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
        simulated_paths = []

        for sim_id in range(n_sim):
            sampled_history = sample_pert(lower, mode, upper, rng)
            try:
                result = runner(sampled_history, steps)
                forecast = np.maximum(np.asarray(result.predictions, dtype=float), 0)
            except Exception:
                forecast = np.repeat(np.nan, steps)
            simulated_paths.append(forecast)
            for year, value in zip(forecast_years, forecast):
                simulation_rows.append(
                    {
                        "series": series,
                        "model": model_name,
                        "simulation": sim_id + 1,
                        "year": year,
                        "forecast": float(value) if np.isfinite(value) else np.nan,
                    }
                )

        simulated_array = np.asarray(simulated_paths, dtype=float)
        lower_q = np.nanpercentile(simulated_array, 2.5, axis=0)
        median_q = np.nanpercentile(simulated_array, 50, axis=0)
        upper_q = np.nanpercentile(simulated_array, 97.5, axis=0)
        mean_q = np.nanmean(simulated_array, axis=0)

        for year, lower_value, median_value, upper_value, mean_value in zip(
            forecast_years,
            lower_q,
            median_q,
            upper_q,
            mean_q,
        ):
            summary_rows.append(
                {
                    "series": series,
                    "selected_model": model_name,
                    "year": year,
                    "forecast_uncertainty_lower": float(lower_value),
                    "forecast_uncertainty_median": float(median_value),
                    "forecast_uncertainty_upper": float(upper_value),
                    "forecast_uncertainty_mean": float(mean_value),
                    "n_simulations": n_sim,
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
            "forecast_uncertainty_lower",
            "forecast_uncertainty_median",
            "forecast_uncertainty_upper",
            "forecast_uncertainty_mean",
            "n_simulations",
        ]
        ].sort_values(["series", "year"])


def plot_uncertainty_forecasts(data: pd.DataFrame, intervals: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plot_series = [series for series in SERIES if series in set(intervals["series"])]
    # Primary analysis has three endpoints, so this gives the requested 1x3
    # layout. If --all-series is used later, the same code wraps into rows of 3.
    ncols = min(3, max(1, len(plot_series)))
    nrows = int(np.ceil(len(plot_series) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5 * ncols, 5.8 * nrows), squeeze=False)
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

        ax.fill_between(years_obs, observed_lower, observed_upper, color="gray", alpha=0.18, label="GBD uncertainty")
        ax.plot(years_obs, observed, color="black", marker="o", linewidth=2.2, label="Observed")
        ax.fill_between(
            years_fc,
            fc["forecast_uncertainty_lower"].to_numpy(dtype=float),
            fc["forecast_uncertainty_upper"].to_numpy(dtype=float),
            color="tab:blue",
            alpha=0.22,
            label="Forecast uncertainty",
        )
        ax.plot(
            years_fc,
            fc["point_forecast"],
            color="tab:blue",
            marker="o",
            linestyle="--",
            linewidth=2.2,
            label="Point forecast",
        )
        ax.axvline(2023, color="black", linewidth=1, alpha=0.55)
        ax.set_title(f"{SERIES_LABELS[series]}\nBest model: {model_name}")
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        ax.margins(x=0.02)

    for ax in axes[len(plot_series) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False)
    fig.suptitle("Best-model forecasts with propagated GBD uncertainty, 2024-2030", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.84, 0.95))
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
                "forecast_uncertainty_lower",
                "forecast_uncertainty_upper",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

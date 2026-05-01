"""Refit selected models on 1990-2023 and forecast Ghana TB burden to 2030.

Run after `03_train_evaluate_all_models.py`:

    /opt/homebrew/anaconda3/envs/deepposture/bin/python scripts/04_forecast_best_models.py
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
    FORECAST_FIGURE_DIR,
    FORECAST_START,
    PRIMARY_SERIES,
    SERIES,
    TABLE_DIR,
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


def plot_forecasts(data: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    for old_path in FORECAST_FIGURE_DIR.glob("best_model_*_forecast_2030.png"):
        old_path.unlink()

    plot_series = [series for series in SERIES if series in set(forecasts["series"])]
    ncols = 2 if len(plot_series) > 1 else 1
    nrows = int(np.ceil(len(plot_series) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, series in zip(axes, plot_series):
        fc = forecasts[forecasts["series"] == series]
        if fc.empty:
            ax.axis("off")
            continue
        model_name = fc["selected_model"].iloc[0]

        ax.plot(data["year"], data[series], marker="o", linewidth=2.4, label="Observed")
        ax.plot(fc["year"], fc["forecast"], marker="o", linestyle="--", linewidth=2.4, label=f"Forecast ({model_name})")
        ax.axvline(2023, color="black", linewidth=1, alpha=0.5)
        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        ax.margins(x=0.02)

    for ax in axes[len(plot_series) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), ncol=1, frameon=False)
    fig.suptitle("Best-model forecasts for Ghana TB burden, 2024-2030", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(FORECAST_FIGURE_DIR / "best_model_forecasts_2024_2030_subplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def forecast_all_models(
    data: pd.DataFrame,
    models: dict,
    selected_series: list[str],
    steps: int,
    forecast_years: list[int],
) -> pd.DataFrame:
    rows = []
    for series in tqdm(selected_series, desc="Forecasting all models", unit="series"):
        train = data[series].to_numpy(dtype=float)
        for model_name, runner in models.items():
            result = runner(train, steps)
            predictions = np.maximum(np.asarray(result.predictions, dtype=float), 0)
            for year, prediction in zip(forecast_years, predictions):
                rows.append(
                    {
                        "series": series,
                        "model": model_name,
                        "year": year,
                        "forecast": float(prediction),
                        "model_params": result.params,
                    }
                )
    return pd.DataFrame(rows).sort_values(["series", "model", "year"])


def plot_all_model_forecasts(data: pd.DataFrame, all_forecasts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    model_order = list(model_registry(epochs=1).keys())

    for series in [series for series in SERIES if series in set(all_forecasts["series"])]:
        fig, axes = plt.subplots(2, 4, figsize=(24, 11), squeeze=False)
        axes = axes.ravel()
        observed = data[["year", series]]

        for ax, model_name in zip(axes, model_order):
            fc = all_forecasts[
                (all_forecasts["series"] == series)
                & (all_forecasts["model"] == model_name)
            ]
            ax.plot(observed["year"], observed[series], color="black", marker="o", linewidth=2.1, label="Observed")
            ax.plot(fc["year"], fc["forecast"], marker="o", linestyle="--", linewidth=2.1, label="Forecast")
            ax.axvline(2023, color="black", linewidth=1, alpha=0.5)
            ax.set_title(model_name)
            ax.set_xlabel("Year")
            ax.set_ylabel(series_y_label(series))
            ax.margins(x=0.02)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False)
        fig.suptitle(f"{SERIES_LABELS[series]}: model-specific forecasts, 2024-2030", y=0.995)
        fig.tight_layout(rect=(0, 0, 0.92, 0.95))
        fig.savefig(FORECAST_FIGURE_DIR / f"all_model_forecasts_{series}_2x4.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mortality_to_incidence_ratio_by_model(data: pd.DataFrame, all_forecasts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    model_order = list(model_registry(epochs=1).keys())
    incidence_series = "incidence_age_standardized_rate"
    mortality_series = "mortality_age_standardized_rate"
    required_series = {incidence_series, mortality_series}

    if not required_series.issubset(set(all_forecasts["series"])):
        return

    observed = pd.DataFrame(
        {
            "year": data["year"],
            "ratio": data[mortality_series].astype(float) / data[incidence_series].astype(float),
        }
    )

    forecast_wide = (
        all_forecasts[all_forecasts["series"].isin(required_series)]
        .pivot_table(index=["model", "year"], columns="series", values="forecast", aggfunc="first")
        .reset_index()
    )
    forecast_wide["ratio"] = (
        forecast_wide[mortality_series].astype(float)
        / forecast_wide[incidence_series].astype(float).replace(0, np.nan)
    )

    fig, axes = plt.subplots(2, 4, figsize=(24, 11), squeeze=False)
    axes = axes.ravel()

    for ax, model_name in zip(axes, model_order):
        model_ratio = forecast_wide[forecast_wide["model"] == model_name]
        ax.plot(observed["year"], observed["ratio"], color="black", marker="o", linewidth=2.1, label="Observed")
        ax.plot(
            model_ratio["year"],
            model_ratio["ratio"],
            marker="o",
            linestyle="--",
            linewidth=2.1,
            label="Forecast ratio",
        )
        ax.axvline(2023, color="black", linewidth=1, alpha=0.5)
        ax.set_title(model_name)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mortality / incidence")
        ax.margins(x=0.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False)
    fig.suptitle("Mortality-to-incidence ratio implied by model forecasts, 2024-2030", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    fig.savefig(FORECAST_FIGURE_DIR / "mortality_to_incidence_ratio_forecasts_by_model_2x4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250, help="Epochs for each PyTorch model.")
    parser.add_argument(
        "--best-model-path",
        default=str(TABLE_DIR / "best_models_by_series.csv"),
        help="CSV produced by the training/evaluation script.",
    )
    parser.add_argument(
        "--all-series",
        action="store_true",
        help="Forecast all series present in the best-model table. By default, forecasts primary age-standardized endpoints.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    data = load_modeling_data()
    best = pd.read_csv(args.best_model_path)
    if not args.all_series:
        best = best[best["series"].isin(PRIMARY_SERIES)].copy()
    models = model_registry(args.epochs)
    steps = FORECAST_END - FORECAST_START + 1
    forecast_years = list(range(FORECAST_START, FORECAST_END + 1))
    selected_series = [series for series in SERIES if series in set(best["series"])]

    rows = []
    for _, selected in tqdm(best.iterrows(), total=len(best), desc="Forecasting best models", unit="series"):
        series = selected["series"]
        model_name = selected["model"]
        if series not in SERIES:
            continue
        if model_name not in models:
            raise ValueError(f"Unknown selected model '{model_name}' for series '{series}'")

        train = data[series].to_numpy(dtype=float)
        result = models[model_name](train, steps)
        predictions = np.maximum(np.asarray(result.predictions, dtype=float), 0)

        for year, prediction in zip(forecast_years, predictions):
            rows.append(
                {
                    "series": series,
                    "year": year,
                    "forecast": float(prediction),
                    "selected_model": model_name,
                    "model_params": result.params,
                    "selection_rmse": float(selected["rmse"]),
                    "selection_mape": float(selected["mape"]),
                }
            )

    forecasts = pd.DataFrame(rows).sort_values(["series", "year"])
    forecasts.to_csv(FORECAST_DIR / "best_model_forecasts_2024_2030.csv", index=False)
    forecasts.to_csv(TABLE_DIR / "best_model_forecasts_2024_2030.csv", index=False)
    plot_forecasts(data, forecasts)

    all_model_forecasts = forecast_all_models(data, models, selected_series, steps, forecast_years)
    all_model_forecasts.to_csv(FORECAST_DIR / "all_model_forecasts_2024_2030.csv", index=False)
    all_model_forecasts.to_csv(TABLE_DIR / "all_model_forecasts_2024_2030.csv", index=False)
    plot_all_model_forecasts(data, all_model_forecasts)
    plot_mortality_to_incidence_ratio_by_model(data, all_model_forecasts)

    print(f"Saved forecasts: {FORECAST_DIR / 'best_model_forecasts_2024_2030.csv'}")
    print(f"Saved all-model forecasts: {FORECAST_DIR / 'all_model_forecasts_2024_2030.csv'}")
    print(f"Saved table copy: {TABLE_DIR / 'best_model_forecasts_2024_2030.csv'}")
    print(f"Saved figure: {FORECAST_FIGURE_DIR / 'best_model_forecasts_2024_2030_subplots.png'}")
    print(f"Saved all-model 2x4 figures: {FORECAST_FIGURE_DIR / 'all_model_forecasts_<series>_2x4.png'}")
    print(f"Saved ratio forecast figure: {FORECAST_FIGURE_DIR / 'mortality_to_incidence_ratio_forecasts_by_model_2x4.png'}")
    print("\nForecast endpoints:")
    endpoints = forecasts[forecasts["year"].isin([FORECAST_START, FORECAST_END])]
    print(endpoints.pivot(index="series", columns="year", values="forecast").round(3).to_string())


if __name__ == "__main__":
    main()

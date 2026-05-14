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

MODEL_COLORS = {
    # Okabe-Ito inspired high-contrast colors that remain distinguishable in print.
    "Naive": "#0072B2",
    "Drift": "#D55E00",
    "ARIMA": "#009E73",
    "ETS": "#CC79A7",
    "LSTM": "#E69F00",
    "Causal_TCN": "#56B4E9",
    "ARIMA_LSTM": "#F0E442",
    "Multistep_LSTM": "#000000",
}

PLOT_TITLE_SIZE = 26
PLOT_AXIS_LABEL_SIZE = 22
PLOT_TICK_SIZE = 18
PLOT_LEGEND_SIZE = 20
PLOT_LINEWIDTH = 4.6
PLOT_MARKERSIZE = 10.5
PLOT_MARKEREDGEWIDTH = 1.4
PLOT_OBSERVED_COLOR = "#3F3F3F"
PLOT_FORECAST_COLORS = {
    "incidence_age_standardized_rate": "#0072B2",
    "mortality_to_incidence_age_standardized_ratio": "#009E73",
    "incidence_all_age_rate": "#56B4E9",
    "incidence_all_age_number": "#E69F00",
    "mortality_age_standardized_rate": "#D55E00",
    "mortality_all_age_rate": "#CC79A7",
    "mortality_all_age_number": "#F0E442",
}


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


def plot_forecasts(data: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    for old_path in FORECAST_FIGURE_DIR.glob("best_model_*_forecast_2030.png"):
        old_path.unlink()

    plot_series = [series for series in SERIES if series in set(forecasts["series"])]
    ncols = min(3, max(1, len(plot_series)))
    nrows = 1 if len(plot_series) else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5 * ncols, 7.2), squeeze=False)
    axes = axes.ravel()
    for ax, series in zip(axes, plot_series):
        fc = forecasts[forecasts["series"] == series]
        if fc.empty:
            ax.axis("off")
            continue
        model_name = fc["selected_model"].iloc[0]

        ax.plot(
            data["year"],
            data[series],
            color=PLOT_OBSERVED_COLOR,
            marker="o",
            markersize=PLOT_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            linewidth=PLOT_LINEWIDTH,
            label="Observed",
        )
        ax.plot(
            fc["year"],
            fc["forecast"],
            color=PLOT_FORECAST_COLORS.get(series, "#0072B2"),
            marker="o",
            markersize=PLOT_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            linestyle="--",
            linewidth=PLOT_LINEWIDTH,
            label=f"Forecast ({model_name})",
        )
        ax.axvline(2023, color="black", linewidth=1.4, alpha=0.55)
        ax.set_title(SERIES_LABELS[series])
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
    forecast_handle = plt.Line2D(
        [0],
        [0],
        color="#0072B2",
        marker="o",
        linewidth=PLOT_LINEWIDTH,
        linestyle="--",
        markersize=PLOT_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=PLOT_MARKEREDGEWIDTH,
        label="Predicted",
    )
    fig.legend(
        [observed_handle, forecast_handle],
        ["Observed", "Predicted"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        prop={"size": PLOT_LEGEND_SIZE, "weight": "bold"},
    )
    fig.suptitle(
        "Best-model forecasts for Ghana TB burden, 2024-2030",
        y=0.995,
        fontsize=32,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.88))
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
        arima_selected_d, arima_d_reason = model_arima.select_d(train)
        for model_name, runner in models.items():
            model_kwargs = {}
            if model_name in {"ARIMA", "ARIMA_LSTM"}:
                model_kwargs = {
                    "selected_d": arima_selected_d,
                    "d_reason": f"{arima_d_reason}; source=full_observed_period",
                }
            result = runner(train, steps, **model_kwargs)
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
        fig, axes = plt.subplots(2, 4, figsize=(26, 12.5), squeeze=False)
        axes = axes.ravel()
        observed = data[["year", series]]

        for ax, model_name in zip(axes, model_order):
            fc = all_forecasts[
                (all_forecasts["series"] == series)
                & (all_forecasts["model"] == model_name)
            ]
            forecast_color = MODEL_COLORS[model_name]
            ax.plot(
                observed["year"],
                observed[series],
                color="#555555",
                marker="o",
                markersize=PLOT_MARKERSIZE,
                markeredgecolor="white",
                markeredgewidth=PLOT_MARKEREDGEWIDTH,
                linewidth=PLOT_LINEWIDTH,
                label="Observed",
                zorder=2,
            )
            ax.plot(
                fc["year"],
                fc["forecast"],
                color=forecast_color,
                marker="o",
                markersize=PLOT_MARKERSIZE,
                markeredgecolor="white",
                markeredgewidth=PLOT_MARKEREDGEWIDTH,
                linestyle="--",
                linewidth=PLOT_LINEWIDTH,
                label=model_name,
                zorder=3,
            )
            ax.axvline(2023, color="#222222", linewidth=1.5, alpha=0.7)
            ax.set_title(model_name)
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

        observed_handle = plt.Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            label="Observed",
        )
        forecast_handle = plt.Line2D(
            [0],
            [0],
            color="#222222",
            marker="o",
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            linestyle="--",
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            label="Predicted",
        )
        fig.legend(
            [observed_handle, forecast_handle],
            ["Observed", "Predicted"],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            frameon=False,
            ncol=2,
            prop={"size": PLOT_LEGEND_SIZE, "weight": "bold"},
        )
        fig.suptitle(
            f"{SERIES_LABELS[series]}: model-specific forecasts, 2024-2030",
            y=0.995,
            fontsize=32,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.10, 1, 0.93))
        fig.savefig(FORECAST_FIGURE_DIR / f"all_model_forecasts_{series}_2x4.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_mortality_to_incidence_ratio_by_model(data: pd.DataFrame, all_forecasts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    model_order = list(model_registry(epochs=1).keys())
    ratio_series = "mortality_to_incidence_age_standardized_ratio"

    if ratio_series not in set(all_forecasts["series"]):
        return

    observed = pd.DataFrame({"year": data["year"], "ratio": data[ratio_series].astype(float)})
    forecast_wide = all_forecasts[all_forecasts["series"] == ratio_series].copy()
    forecast_wide["ratio"] = forecast_wide["forecast"].astype(float)

    fig, axes = plt.subplots(2, 4, figsize=(26, 12.5), squeeze=False)
    axes = axes.ravel()

    for ax, model_name in zip(axes, model_order):
        model_ratio = forecast_wide[forecast_wide["model"] == model_name]
        ax.plot(
            observed["year"],
            observed["ratio"],
            color="black",
            marker="o",
            markersize=PLOT_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            linewidth=PLOT_LINEWIDTH,
            label="Observed",
            zorder=2,
        )
        ax.plot(
            model_ratio["year"],
            model_ratio["ratio"],
            color=MODEL_COLORS.get(model_name, "#222222"),
            marker="o",
            linestyle="--",
            markersize=PLOT_MARKERSIZE,
            markeredgecolor="white",
            markeredgewidth=PLOT_MARKEREDGEWIDTH,
            linewidth=PLOT_LINEWIDTH,
            label="Predicted",
            zorder=3,
        )
        ax.axvline(2023, color="black", linewidth=1.6, alpha=0.65)
        ax.set_title(model_name)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mortality / incidence")
        ax.title.set_fontsize(PLOT_TITLE_SIZE)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(PLOT_AXIS_LABEL_SIZE)
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontsize(PLOT_AXIS_LABEL_SIZE)
        ax.yaxis.label.set_fontweight("bold")
        ax.tick_params(axis="both", labelsize=PLOT_TICK_SIZE, width=1.4, length=6)
        ax.margins(x=0.02)

    observed_handle = plt.Line2D(
        [0],
        [0],
        color="black",
        marker="o",
        linewidth=PLOT_LINEWIDTH,
        markersize=PLOT_MARKERSIZE,
        markeredgecolor="white",
        markeredgewidth=PLOT_MARKEREDGEWIDTH,
        label="Observed",
    )
    predicted_handle = plt.Line2D(
        [0],
        [0],
        color="#222222",
        marker="o",
        markeredgecolor="white",
        markeredgewidth=PLOT_MARKEREDGEWIDTH,
        linestyle="--",
        linewidth=PLOT_LINEWIDTH,
        markersize=PLOT_MARKERSIZE,
        label="Predicted",
    )
    fig.legend(
        [observed_handle, predicted_handle],
        ["Observed", "Predicted"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        prop={"size": PLOT_LEGEND_SIZE, "weight": "bold"},
    )
    fig.suptitle(
        "Mortality-to-incidence ratio implied by model forecasts, 2024-2030",
        y=0.995,
        fontsize=32,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(FORECAST_FIGURE_DIR / "mortality_to_incidence_ratio_forecasts_by_model_2x4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="Maximum epochs for each PyTorch model.")
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
        model_kwargs = {}
        if model_name in {"ARIMA", "ARIMA_LSTM"}:
            arima_selected_d, arima_d_reason = model_arima.select_d(train)
            model_kwargs = {
                "selected_d": arima_selected_d,
                "d_reason": f"{arima_d_reason}; source=full_observed_period",
            }
        result = models[model_name](train, steps, **model_kwargs)
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

"""Fit classical forecasting baselines for Ghana TB burden series.

Models:
- Naive: last observed value
- Drift: straight-line random walk with drift
- ARIMA: small AIC-selected grid using statsmodels
- ETS: exponential smoothing variants

The script evaluates models on 2018-2023 and then refits the best model on
1990-2023 to forecast 2024-2030.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ghana_tb_modeling_wide.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures" / "forecasts"

FORECAST_START = 2024
FORECAST_END = 2030
TEST_START = 2018

SERIES = [
    "incidence_age_standardized_rate",
    "incidence_all_age_rate",
    "incidence_all_age_number",
    "mortality_age_standardized_rate",
    "mortality_all_age_rate",
    "mortality_all_age_number",
]


@dataclass(frozen=True)
class ModelResult:
    name: str
    params: str
    predictions: np.ndarray


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error / actual)) * 100),
    }


def naive_forecast(train: np.ndarray, steps: int) -> ModelResult:
    return ModelResult("Naive", "last_observation", np.repeat(train[-1], steps))


def drift_forecast(train: np.ndarray, steps: int) -> ModelResult:
    drift = (train[-1] - train[0]) / (len(train) - 1)
    horizon = np.arange(1, steps + 1)
    return ModelResult("Drift", "linear_drift", train[-1] + horizon * drift)


def arima_forecast(train: np.ndarray, steps: int) -> ModelResult:
    best_aic = np.inf
    best_order = None
    best_fit = None
    candidate_orders = [
        (0, 1, 0),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ValueWarning)
        warnings.simplefilter("ignore", UserWarning)
        for order in candidate_orders:
            try:
                fit = ARIMA(
                    train,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method_kwargs={"maxiter": 75})
            except Exception:
                continue
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = fit.aic
                best_order = order
                best_fit = fit

    if best_fit is None or best_order is None:
        return naive_forecast(train, steps)

    forecast = np.asarray(best_fit.forecast(steps=steps), dtype=float)
    return ModelResult("ARIMA", f"order={best_order}; aic={best_aic:.2f}", forecast)


def ets_forecast(train: np.ndarray, steps: int) -> ModelResult:
    candidates = [
        {"trend": None, "damped_trend": False},
        {"trend": "add", "damped_trend": False},
        {"trend": "add", "damped_trend": True},
    ]
    best_aic = np.inf
    best_fit = None
    best_params = ""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for params in candidates:
            try:
                fit = ExponentialSmoothing(
                    train,
                    trend=params["trend"],
                    damped_trend=params["damped_trend"],
                    seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True)
            except Exception:
                continue
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit
                best_params = f"trend={params['trend']}; damped={params['damped_trend']}; aic={best_aic:.2f}"

    if best_fit is None:
        return naive_forecast(train, steps)

    forecast = np.asarray(best_fit.forecast(steps), dtype=float)
    return ModelResult("ETS", best_params, forecast)


def evaluate_series(series_name: str, values: pd.Series) -> tuple[pd.DataFrame, str]:
    train = values[values.index < TEST_START].to_numpy(dtype=float)
    test = values[values.index >= TEST_START].to_numpy(dtype=float)
    steps = len(test)

    model_results = [
        naive_forecast(train, steps),
        drift_forecast(train, steps),
        arima_forecast(train, steps),
        ets_forecast(train, steps),
    ]

    rows = []
    for result in model_results:
        row = {
            "series": series_name,
            "model": result.name,
            "params": result.params,
            "train_end": TEST_START - 1,
            "test_start": TEST_START,
            "test_end": int(values.index.max()),
        }
        row.update(metrics(test, result.predictions))
        rows.append(row)

    evaluation = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    best_model = str(evaluation.loc[0, "model"])
    return evaluation, best_model


def forecast_full_series(series_name: str, values: pd.Series, model_name: str) -> pd.DataFrame:
    train = values.to_numpy(dtype=float)
    steps = FORECAST_END - FORECAST_START + 1

    if model_name == "ARIMA":
        result = arima_forecast(train, steps)
    elif model_name == "ETS":
        result = ets_forecast(train, steps)
    elif model_name == "Drift":
        result = drift_forecast(train, steps)
    else:
        result = naive_forecast(train, steps)

    return pd.DataFrame(
        {
            "series": series_name,
            "year": range(FORECAST_START, FORECAST_END + 1),
            "forecast": np.maximum(result.predictions, 0),
            "selected_model": result.name,
            "model_params": result.params,
        }
    )


def plot_forecasts(data: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    labels = {
        "incidence_age_standardized_rate": "Incidence age-standardized rate",
        "incidence_all_age_rate": "Incidence all-age rate",
        "incidence_all_age_number": "Incidence all-age number",
        "mortality_age_standardized_rate": "Mortality age-standardized rate",
        "mortality_all_age_rate": "Mortality all-age rate",
        "mortality_all_age_number": "Mortality all-age number",
    }

    for series_name in SERIES:
        old_path = FIGURE_DIR / f"{series_name}_forecast_2030.png"
        if old_path.exists():
            old_path.unlink()

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    axes = axes.ravel()
    for ax, series_name in zip(axes, SERIES):
        ax.plot(data["year"], data[series_name], marker="o", linewidth=2.4, label="Observed")
        fc = forecasts[forecasts["series"] == series_name]
        model_name = fc["selected_model"].iloc[0]
        ax.plot(fc["year"], fc["forecast"], marker="o", linewidth=2.4, linestyle="--", label=f"Forecast ({model_name})")
        ax.axvline(2023, color="black", linewidth=1, alpha=0.5)
        ax.set_title(labels[series_name])
        ax.set_xlabel("Year")
        ax.set_ylabel("Rate per 100,000" if "rate" in series_name else "Number")
        ax.margins(x=0.02)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="center left", bbox_to_anchor=(0.99, 0.5), ncol=1, frameon=False)
    fig.suptitle("Classical best-model forecasts for Ghana TB burden, 2024-2030", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(FIGURE_DIR / "classical_forecasts_2024_2030_subplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    data = data.set_index("year")

    evaluations = []
    forecasts = []
    for series_name in SERIES:
        evaluation, best_model = evaluate_series(series_name, data[series_name])
        evaluations.append(evaluation)
        forecasts.append(forecast_full_series(series_name, data[series_name], best_model))

    evaluation_table = pd.concat(evaluations, ignore_index=True)
    forecast_table = pd.concat(forecasts, ignore_index=True)

    evaluation_table.to_csv(TABLE_DIR / "classical_model_evaluation_2018_2023.csv", index=False)
    forecast_table.to_csv(TABLE_DIR / "classical_forecasts_2024_2030.csv", index=False)

    plot_forecasts(data.reset_index(), forecast_table)

    print(f"Saved model evaluation to {TABLE_DIR / 'classical_model_evaluation_2018_2023.csv'}")
    print(f"Saved forecasts to {TABLE_DIR / 'classical_forecasts_2024_2030.csv'}")
    print(f"Saved forecast figure to {FIGURE_DIR / 'classical_forecasts_2024_2030_subplots.png'}")


if __name__ == "__main__":
    main()

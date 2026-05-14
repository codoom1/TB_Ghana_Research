"""Train and evaluate all Ghana TB forecasting models.

Run with the PyTorch environment:

    /opt/homebrew/anaconda3/envs/deepposture/bin/python scripts/03_train_evaluate_all_models.py
"""

from __future__ import annotations

import argparse
import time

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
    EVALUATION_FIGURE_DIR,
    MODEL_OUTPUT_DIR,
    PRIMARY_SERIES,
    SERIES,
    TABLE_DIR,
    TEST_START,
    ensure_output_dirs,
    load_modeling_data,
    metric_dict,
    selected_model_table,
    split_train_test,
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

CLASSICAL_MODELS = ["Naive", "Drift", "ARIMA", "ETS"]
DEEP_LEARNING_MODELS = ["LSTM", "Causal_TCN", "ARIMA_LSTM", "Multistep_LSTM"]
MODEL_COLORS = {
    # High-contrast print-friendly model colors.
    "Naive": "#0072B2",
    "Drift": "#D55E00",
    "ARIMA": "#009E73",
    "ETS": "#CC79A7",
    "LSTM": "#E69F00",
    "Causal_TCN": "#56B4E9",
    "ARIMA_LSTM": "#B79F00",
    "Multistep_LSTM": "#332288",
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


def plot_model_test_predictions(data: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """Plot observed test-set predictions for every model in one comparison figure."""
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.titlesize": 40,
            "axes.titleweight": "bold",
            "axes.labelsize": 40,
            "axes.labelweight": "bold",
            "xtick.labelsize": 40,
            "ytick.labelsize": 40,
            "legend.fontsize": 40,
        },
    )
    predicted_series = set(predictions["series"])
    plot_series = [series for series in PRIMARY_SERIES if series in predicted_series]
    plot_series.extend(series for series in SERIES if series in predicted_series and series not in plot_series)
    ncols = min(3, max(1, len(plot_series)))
    nrows = int(np.ceil(len(plot_series) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8 * ncols, 18.2 * nrows), squeeze=False)
    axes = axes.ravel()
    model_order = list(model_registry(epochs=1).keys())
    plotted_models = [model for model in model_order if model in set(predictions["model"])]

    for ax, series in zip(axes, plot_series):
        observed = data[["year", series]]
        ax.plot(
            observed["year"],
            observed[series],
            color="#4D4D4D",
            marker="o",
            markersize=8.8,
            linewidth=5.0,
            label="Observed",
            zorder=2,
        )
        ax.axvline(TEST_START, color="#111111", linestyle=":", linewidth=3.0, alpha=0.95)

        series_predictions = predictions[predictions["series"] == series]
        for model_name in plotted_models:
            group = series_predictions[series_predictions["model"] == model_name]
            if group.empty:
                continue
            ax.plot(
                group["year"],
                group["predicted"],
                marker="o",
                linestyle="--",
                linewidth=6.4,
                markersize=10.6,
                markeredgecolor="white",
                markeredgewidth=1.3,
                color=MODEL_COLORS[model_name],
                label=model_name,
                zorder=3,
            )

        ax.set_title(SERIES_LABELS[series])
        ax.set_xlabel("Year")
        ax.set_ylabel(series_y_label(series))
        ax.margins(x=0.02)
        ax.grid(True, which="major", color="#8F8F8F", linewidth=1.8, alpha=0.9)
        ax.tick_params(axis="both", width=1.8, length=8)

    for ax in axes[len(plot_series) :]:
        ax.axis("off")

    observed_handle = plt.Line2D(
        [0],
        [0],
        color="#4D4D4D",
        marker="o",
        linewidth=5.0,
        markersize=8.8,
        label="Observed",
    )
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model_name],
            marker="o",
            markeredgecolor="white",
            markeredgewidth=1.3,
            linestyle="--",
            linewidth=5.4,
            markersize=10.6,
            label=model_name,
        )
        for model_name in plotted_models
    ]
    fig.legend(
    [observed_handle, *model_handles],
    ["Observed", *plotted_models],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=5,
    frameon=False,
    columnspacing=1.9,
    handlelength=3.3,
    handletextpad=0.75,
    prop={"weight": "bold", "size": 40},
)
    fig.suptitle("Model comparison on the 2018-2023 test period", fontsize=40, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0.16, 1, 0.91))
    fig.savefig(EVALUATION_FIGURE_DIR / "all_model_test_predictions_subplots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_barplot(
    evaluation: pd.DataFrame,
    models: list[str],
    filename: str,
    title: str,
) -> None:
    subset = evaluation[evaluation["model"].isin(models)].copy()
    if subset.empty:
        return

    subset["series_label"] = subset["series"].map(SERIES_LABELS)
    subset["model"] = pd.Categorical(subset["model"], categories=models, ordered=True)
    subset = subset.sort_values(["series_label", "model"])

    long = subset.melt(
        id_vars=["series", "series_label", "model"],
        value_vars=["mae", "rmse", "mape"],
        var_name="metric",
        value_name="metric_value",
    )
    long["metric"] = long["metric"].map(
        {
            "mae": "MAE",
            "rmse": "RMSE",
            "mape": "MAPE (%)",
        }
    )
    metric_order = ["MAE", "RMSE", "MAPE (%)"]
    long["metric"] = pd.Categorical(long["metric"], categories=metric_order, ordered=True)

    sns.set_theme(style="whitegrid", context="talk")
    n_series = subset["series"].nunique()
    fig, axes = plt.subplots(
        3,
        n_series,
        figsize=(7.5 * n_series, 15),
        squeeze=False,
        sharey=False,
    )
    palette = sns.color_palette("viridis", len(models))

    for col_idx, (series, series_group) in enumerate(subset.groupby("series", sort=False)):
        for row_idx, metric in enumerate(metric_order):
            ax = axes[row_idx, col_idx]
            group = long[(long["series"] == series) & (long["metric"] == metric)]
            sns.barplot(data=group, x="model", y="metric_value", ax=ax, palette=palette, hue="model", legend=False)

            metric_col = "mape" if metric == "MAPE (%)" else metric.lower()
            best_idx = series_group[metric_col].idxmin()
            best_model = series_group.loc[best_idx, "model"]
            for patch, model_name in zip(ax.patches, group["model"]):
                if model_name == best_model:
                    patch.set_edgecolor("black")
                    patch.set_linewidth(2.4)

            if row_idx == 0:
                ax.set_title(SERIES_LABELS[series])
            else:
                ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=35)
            for label in ax.get_xticklabels():
                label.set_ha("right")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(EVALUATION_FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparisons(evaluation: pd.DataFrame) -> None:
    plot_error_barplot(
        evaluation,
        CLASSICAL_MODELS,
        "evaluation_errors_classical_models.png",
        "Classical model error comparison, 2018-2023 test period",
    )
    plot_error_barplot(
        evaluation,
        DEEP_LEARNING_MODELS,
        "evaluation_errors_deep_learning_models.png",
        "Deep learning model error comparison, 2018-2023 test period",
    )
    plot_error_barplot(
        evaluation,
        CLASSICAL_MODELS + DEEP_LEARNING_MODELS,
        "evaluation_errors_all_models.png",
        "All model error comparison, 2018-2023 test period",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300, help="Maximum epochs for each PyTorch model.")
    parser.add_argument(
        "--all-series",
        action="store_true",
        help="Evaluate all seven modeling series instead of the three default age-standardized endpoints.",
    )
    parser.add_argument(
        "--series",
        nargs="*",
        default=None,
        choices=SERIES,
        help="Optional subset of target series to evaluate. Overrides the primary/default selection.",
    )
    args = parser.parse_args()
    selected_series = args.series if args.series is not None else (SERIES if args.all_series else PRIMARY_SERIES)

    ensure_output_dirs()
    data = load_modeling_data()
    models = model_registry(args.epochs)
    total_runs = len(selected_series) * len(models)

    evaluation_rows = []
    prediction_rows = []

    progress = tqdm(total=total_runs, desc="Training/evaluating models", unit="model")
    for series in selected_series:
        train, test, test_years = split_train_test(data, series)
        steps = len(test)
        arima_selected_d, arima_d_reason = model_arima.select_d(train)
        for model_name, runner in models.items():
            start = time.perf_counter()
            try:
                tqdm.write(f"Training model {model_name} for series {series}...")
                model_kwargs = {}
                if model_name in {"ARIMA", "ARIMA_LSTM"}:
                    model_kwargs = {
                        "selected_d": arima_selected_d,
                        "d_reason": f"{arima_d_reason}; source=training_period",
                    }
                if model_name in {"LSTM", "Causal_TCN", "ARIMA_LSTM", "Multistep_LSTM"}:
                    model_kwargs["progress_callback"] = lambda message, _series=series, _model=model_name: tqdm.write(
                        f"{_series} | {_model}: {message}"
                    )
                result = runner(train, steps, **model_kwargs)
                predictions = np.maximum(np.asarray(result.predictions, dtype=float), 0)
                status = "ok"
                error = ""
            except Exception as exc:
                predictions = np.repeat(np.nan, steps)
                result = None
                status = "failed"
                error = repr(exc)

            elapsed = time.perf_counter() - start
            if status == "ok":
                tqdm.write(f"{model_name} done training for series {series} in {elapsed:.1f}s")
            else:
                tqdm.write(f"Training failed for model {model_name} on series {series}: {error}")
            progress.set_postfix(series=series[:18], model=model_name, status=status)
            progress.update(1)

            if status == "ok":
                row = {
                    "series": series,
                    "model": model_name,
                    "params": result.params,
                    "train_start": int(data["year"].min()),
                    "train_end": TEST_START - 1,
                    "test_start": TEST_START,
                    "test_end": int(data["year"].max()),
                    "runtime_seconds": elapsed,
                    "status": status,
                    "error": error,
                }
                row.update(metric_dict(test, predictions))
            else:
                row = {
                    "series": series,
                    "model": model_name,
                    "params": "",
                    "train_start": int(data["year"].min()),
                    "train_end": TEST_START - 1,
                    "test_start": TEST_START,
                    "test_end": int(data["year"].max()),
                    "runtime_seconds": elapsed,
                    "status": status,
                    "error": error,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                }
            evaluation_rows.append(row)

            for year, actual, predicted in zip(test_years, test, predictions):
                prediction_rows.append(
                    {
                        "series": series,
                        "model": model_name,
                        "year": int(year),
                        "actual": float(actual),
                        "predicted": float(predicted) if np.isfinite(predicted) else np.nan,
                    }
                )

    progress.close()

    evaluation = pd.DataFrame(evaluation_rows).sort_values(["series", "rmse", "mae", "mape"])
    predictions = pd.DataFrame(prediction_rows)
    best_models = selected_model_table(evaluation.dropna(subset=["rmse"]))

    evaluation.to_csv(TABLE_DIR / "all_model_evaluation_2018_2023.csv", index=False)
    predictions.to_csv(MODEL_OUTPUT_DIR / "all_model_test_predictions_2018_2023.csv", index=False)
    best_models.to_csv(TABLE_DIR / "best_models_by_series.csv", index=False)
    plot_model_test_predictions(data, predictions)
    plot_error_comparisons(evaluation)

    print(f"Saved evaluation: {TABLE_DIR / 'all_model_evaluation_2018_2023.csv'}")
    print(f"Saved test predictions: {MODEL_OUTPUT_DIR / 'all_model_test_predictions_2018_2023.csv'}")
    print(f"Saved best-model table: {TABLE_DIR / 'best_models_by_series.csv'}")
    print(f"Saved model-comparison figure: {EVALUATION_FIGURE_DIR / 'all_model_test_predictions_subplots.png'}")
    print(f"Saved error barplots: {EVALUATION_FIGURE_DIR / 'evaluation_errors_*.png'}")
    print("\nBest models by RMSE:")
    print(best_models[["series", "model", "mae", "rmse", "mape"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

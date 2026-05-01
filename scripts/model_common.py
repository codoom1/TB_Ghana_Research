"""Shared utilities for Ghana TB forecasting models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd


# Resolve paths relative to the repository root so scripts can be run from any
# working directory.
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ghana_tb_modeling_wide.csv"
OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model_outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
FIGURE_DIR = OUTPUT_DIR / "figures"
TREND_FIGURE_DIR = FIGURE_DIR / "trend"
EVALUATION_FIGURE_DIR = FIGURE_DIR / "evaluation"
FORECAST_FIGURE_DIR = FIGURE_DIR / "forecasts"
UNCERTAINTY_FIGURE_DIR = FIGURE_DIR / "uncertainty"

# Temporal split and final forecast horizon used consistently across models.
TEST_START = 2018
FORECAST_START = 2024
FORECAST_END = 2030

# All series available in the prepared wide dataset. The full seven-series
# analysis can be requested with --all-series.
SERIES = [
    "incidence_age_standardized_rate",
    "mortality_to_incidence_age_standardized_ratio",
    "incidence_all_age_rate",
    "incidence_all_age_number",
    "mortality_age_standardized_rate",
    "mortality_all_age_rate",
    "mortality_all_age_number",
]

# Primary endpoints for the main manuscript analysis.
PRIMARY_SERIES = [
    "incidence_age_standardized_rate",
    "mortality_age_standardized_rate",
    "mortality_to_incidence_age_standardized_ratio",
]


@dataclass
class ForecastResult:
    """Standard return object used by every model script."""

    # Human-readable model name used in output tables and plots.
    model: str
    # Text description of fitted hyperparameters or fallback behavior.
    params: str
    # Forecasted values on the original TB scale, one value per future step.
    predictions: np.ndarray


@dataclass
class DifferenceTransform:
    """First-difference transform with inverse forecast reconstruction."""

    # Last observed level is needed to convert predicted differences back to
    # original incidence/mortality rates.
    last_level: float

    @classmethod
    def fit_transform(cls, values: np.ndarray) -> tuple["DifferenceTransform", np.ndarray]:
        # Convert user input to a numeric array so np.diff behaves predictably.
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            raise ValueError("At least two observations are required for differencing.")
        # Store the last level and return first differences: y_t - y_(t-1).
        return cls(last_level=float(values[-1])), np.diff(values)

    def inverse_forecast(self, predicted_differences: np.ndarray) -> np.ndarray:
        # Forecasted levels are the final observed level plus cumulative future
        # changes. This is how neural forecasts return to the original scale.
        predicted_differences = np.asarray(predicted_differences, dtype=float)
        return self.last_level + np.cumsum(predicted_differences)


def ensure_output_dirs() -> None:
    # Create all output folders used by the training and forecasting scripts.
    for directory in (
        TABLE_DIR,
        MODEL_OUTPUT_DIR,
        FORECAST_DIR,
        FIGURE_DIR,
        TREND_FIGURE_DIR,
        EVALUATION_FIGURE_DIR,
        FORECAST_FIGURE_DIR,
        UNCERTAINTY_FIGURE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_modeling_data() -> pd.DataFrame:
    # The modeling table is wide: one row per year and one column per target
    # series. This format is convenient for univariate model loops.
    data = pd.read_csv(DATA_PATH)
    if "year" not in data.columns:
        raise ValueError("Expected a year column in the modeling data.")
    # Fail early if the data preparation step did not create expected columns.
    missing = [series for series in SERIES if series not in data.columns]
    if missing:
        raise ValueError(f"Missing expected series columns: {missing}")
    # Keep chronological order explicit.
    return data.sort_values("year").reset_index(drop=True)


def split_train_test(data: pd.DataFrame, series: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Temporal holdout: train on years before TEST_START and evaluate on later
    # years. No shuffling is used for time-series forecasting.
    train = data.loc[data["year"] < TEST_START, series].to_numpy(dtype=float)
    test = data.loc[data["year"] >= TEST_START, series].to_numpy(dtype=float)
    test_years = data.loc[data["year"] >= TEST_START, "year"].to_numpy(dtype=int)
    return train, test, test_years


def metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    # Convert to numeric arrays so vectorized metric calculations are stable.
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(np.abs(error / actual)) * 100),
    }


def make_supervised(values: np.ndarray, seq_len: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    # Convert a single time series into sliding-window supervised examples.
    # Example with seq_len=5, horizon=1:
    # X=[z1,z2,z3,z4,z5], y=z6.
    x, y = [], []
    values = np.asarray(values, dtype=float)
    # Last valid window must leave enough observations for the requested target
    # horizon.
    max_start = len(values) - seq_len - horizon + 1
    for start in range(max_start):
        end = start + seq_len
        # Historical lag window used as model input.
        x.append(values[start:end])
        # Target is either one future step or a vector of future steps.
        target = values[end : end + horizon]
        y.append(target[0] if horizon == 1 else target)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def choose_seq_len(values: np.ndarray, requested: int = 5, horizon: int = 1) -> int:
    # Short annual series cannot support large lag windows. Keep at least a few
    # supervised samples after accounting for the forecast horizon.
    max_len = max(2, len(values) - horizon - 2)
    return int(min(requested, max_len))


def set_global_seed(seed: int = 42) -> None:
    # Set seeds for reproducible neural-network initialization and optimization.
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        # CUDA may not be present, but this keeps the function portable.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Classical models should still run if torch is unavailable.
        pass


def selected_model_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    # Select the best model for each series by RMSE, with MAE/MAPE as tie-breaks.
    ordered = evaluation.sort_values(["series", "rmse", "mae", "mape"]).reset_index(drop=True)
    return ordered.groupby("series", as_index=False).first()

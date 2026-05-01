"""Simple baseline models for annual Ghana TB forecasting."""

from __future__ import annotations

import numpy as np

from model_common import ForecastResult


def naive_forecast(train: np.ndarray, steps: int) -> ForecastResult:
    # Naive persistence forecast: every future value equals the last observed
    # training value. This is the minimum benchmark a useful model should beat.
    return ForecastResult("Naive", "last_observation", np.repeat(train[-1], steps))


def drift_forecast(train: np.ndarray, steps: int) -> ForecastResult:
    # Estimate the average annual change between the first and last training
    # observations.
    drift = (train[-1] - train[0]) / (len(train) - 1)
    # Forecast horizons are 1, 2, ..., steps years ahead.
    horizon = np.arange(1, steps + 1)
    # Extend the last observed value by the historical average annual change.
    return ForecastResult("Drift", "linear_drift", train[-1] + horizon * drift)

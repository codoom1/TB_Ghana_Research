"""ETS competing model for annual Ghana TB forecasting."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from model_common import ForecastResult


MODEL_NAME = "ETS"


def forecast(train: np.ndarray, steps: int) -> ForecastResult:
    # ETS candidates are deliberately simple because the annual GBD series are
    # short. Seasonal components are not used for annual data.
    candidates = [
        {"trend": None, "damped_trend": False},
        {"trend": "add", "damped_trend": False},
        {"trend": "add", "damped_trend": True},
    ]
    # Track the model with the smallest Akaike Information Criterion.
    best_aic = np.inf
    best_fit = None
    best_params = ""

    with warnings.catch_warnings():
        # statsmodels may warn on small samples or difficult optimization; failed
        # candidates are skipped below.
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for params in candidates:
            try:
                # Fit one ETS variant. initialization_method="estimated" lets
                # statsmodels estimate the initial level/trend from data.
                fit = ExponentialSmoothing(
                    train,
                    trend=params["trend"],
                    damped_trend=params["damped_trend"],
                    seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True)
            except Exception:
                # Keep the grid search robust if one specification fails.
                continue
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = float(fit.aic)
                best_fit = fit
                best_params = f"trend={params['trend']}; damped={params['damped_trend']}; aic={best_aic:.2f}"

    if best_fit is None:
        # Conservative fallback: if no ETS candidate fits, use persistence.
        prediction = np.repeat(train[-1], steps)
        return ForecastResult(MODEL_NAME, "fallback=last_observation", prediction)

    # Forecast the requested number of future annual values.
    prediction = np.asarray(best_fit.forecast(steps), dtype=float)
    return ForecastResult(MODEL_NAME, best_params, prediction)

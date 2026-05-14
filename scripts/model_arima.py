"""ARIMA model for annual Ghana TB forecasting."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from model_common import ForecastResult


MODEL_NAME = "ARIMA"

def _safe_adf(values: np.ndarray) -> float:
    # ADF null hypothesis: the series has a unit root. Small p-values support
    # stationarity. Return NaN rather than failing the whole model run.
    try:
        return float(adfuller(values, autolag="AIC")[1])
    except Exception:
        return float("nan")


def _safe_kpss(values: np.ndarray) -> float:
    # KPSS null hypothesis: the series is stationary. Large p-values support
    # stationarity. Warnings are common for lookup-table boundary p-values.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(values, regression="c", nlags="auto")[1])
    except Exception:
        return float("nan")


def difference_values(values: np.ndarray, d: int) -> np.ndarray:
    # Apply d rounds of differencing. d=0 returns the original series.
    differenced = np.asarray(values, dtype=float)
    for _ in range(d):
        differenced = np.diff(differenced)
    return differenced


def select_d(values: np.ndarray, max_d: int = 2) -> tuple[int, str]:
    """Choose the smallest d that is supported by ADF and KPSS evidence."""
    candidates = []
    for d in range(max_d + 1):
        # Test the original series, first difference, and second difference.
        differenced = difference_values(values, d)
        adf_p = _safe_adf(differenced)
        kpss_p = _safe_kpss(differenced)
        # ADF passes when p <= 0.05; KPSS passes when p >= 0.05.
        adf_pass = np.isfinite(adf_p) and adf_p <= 0.05
        kpss_pass = np.isfinite(kpss_p) and kpss_p >= 0.05
        candidates.append((d, adf_p, kpss_p, adf_pass, kpss_pass))
        # Prefer the smallest d where both tests agree.
        if adf_pass and kpss_pass:
            reason = f"adf_p={adf_p:.4f}; kpss_p={kpss_p:.4f}; rule=adf_and_kpss"
            return d, reason

    # If the two tests do not agree, use the smallest d supported by either one.
    for d, adf_p, kpss_p, adf_pass, kpss_pass in candidates:
        if adf_pass or kpss_pass:
            passed = "adf" if adf_pass else "kpss"
            reason = f"adf_p={adf_p:.4f}; kpss_p={kpss_p:.4f}; rule={passed}_only"
            return d, reason

    # Last fallback keeps the ARIMA search moving if tests are inconclusive.
    d, adf_p, kpss_p, _, _ = candidates[-1]
    return d, f"adf_p={adf_p:.4f}; kpss_p={kpss_p:.4f}; rule=max_d_fallback"


def candidate_orders_for_d(d: int) -> list[tuple[int, int, int]]:
    # Search a small low-order ARIMA grid around the data-driven differencing
    # order. This is intentionally conservative for only 34 annual observations.
    return [
        (0, d, 0),
        (1, d, 0),
        (0, d, 1),
        (1, d, 1),
        (2, d, 0),
        (0, d, 2),
    ]


def forecast(
    train: np.ndarray,
    steps: int,
    selected_d: int | None = None,
    d_reason: str | None = None,
) -> ForecastResult:
    # Initialize model-selection trackers.
    best_aic = np.inf
    best_order = None
    best_fit = None
    # Use the differencing order supplied by the orchestration script. If a
    # caller does not supply one, select it from the available training data.
    if selected_d is None:
        selected_d, d_reason = select_d(train)
    else:
        d_reason = d_reason or "provided_by_caller"

    with warnings.catch_warnings():
        # Individual ARIMA candidates may not converge on small samples.
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ValueWarning)
        warnings.simplefilter("ignore", UserWarning)
        for order in candidate_orders_for_d(selected_d):
            try:
                # Fit one candidate ARIMA(p,d,q) model.
                fit = ARIMA(
                    train,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method_kwargs={"maxiter": 75})
            except Exception:
                # Skip failed specifications and continue searching.
                continue
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = float(fit.aic)
                best_order = order
                best_fit = fit

    if best_fit is None or best_order is None:
        # If all ARIMA candidates fail, fall back to the naive forecast.
        prediction = np.repeat(train[-1], steps)
        return ForecastResult(MODEL_NAME, f"selected_d={selected_d}; {d_reason}; fallback=last_observation", prediction)

    # Produce future values on the original scale; statsmodels handles the
    # integration back from differenced space internally.
    prediction = np.asarray(best_fit.forecast(steps=steps), dtype=float)
    return ForecastResult(MODEL_NAME, f"selected_d={selected_d}; {d_reason}; order={best_order}; aic={best_aic:.2f}", prediction)

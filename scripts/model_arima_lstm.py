"""Hybrid ARIMA + LSTM residual model for annual Ghana TB forecasting."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from torch import nn

from model_arima import candidate_orders_for_d, select_d
from model_common import ForecastResult, choose_seq_len, make_supervised, set_global_seed


MODEL_NAME = "ARIMA_LSTM"


class ResidualLSTM(nn.Module):
    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        # Residual sequence has one feature at each time step: standardized
        # ARIMA residual.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        # Predict one next residual correction.
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq_len x 1.
        output, _ = self.lstm(x)
        # Final hidden state summarizes the residual lag window.
        return self.linear(output[:, -1, :]).squeeze(-1)


def _device() -> torch.device:
    # Use Apple MPS when available, otherwise CPU.
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _best_arima(train: np.ndarray, selected_d: int | None = None, d_reason: str | None = None):
    # This helper repeats the ARIMA selection logic so the hybrid model can use
    # the same data-driven differencing and AIC model choice.
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
        # Failed or nonconverged ARIMA candidates are skipped.
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ValueWarning)
        warnings.simplefilter("ignore", UserWarning)
        for order in candidate_orders_for_d(selected_d):
            try:
                # Fit one low-order ARIMA candidate.
                fit = ARIMA(
                    train,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(method_kwargs={"maxiter": 75})
            except Exception:
                continue
            # Retain the candidate with the lowest AIC.
            if np.isfinite(fit.aic) and fit.aic < best_aic:
                best_aic = float(fit.aic)
                best_order = order
                best_fit = fit
    return best_fit, best_order, best_aic, selected_d, d_reason


def forecast(
    train: np.ndarray,
    steps: int,
    selected_d: int | None = None,
    d_reason: str | None = None,
    seq_len: int = 5,
    epochs: int = 250,
    lr: float = 0.01,
    hidden_size: int = 12,
    seed: int = 42,
) -> ForecastResult:
    # Set seeds before neural residual training.
    set_global_seed(seed)
    train = np.asarray(train, dtype=float)
    # Step 1: fit the best ARIMA model to capture linear trend/autocorrelation.
    arima_fit, order, aic, selected_d, d_reason = _best_arima(train, selected_d=selected_d, d_reason=d_reason)
    if arima_fit is None or order is None:
        # If ARIMA cannot be fit, the hybrid has no baseline component.
        prediction = np.repeat(train[-1], steps)
        return ForecastResult(MODEL_NAME, f"selected_d={selected_d}; {d_reason}; fallback=last_observation", prediction)

    # ARIMA provides the baseline future path.
    arima_forecast = np.asarray(arima_fit.forecast(steps=steps), dtype=float)
    # Residuals are the part ARIMA did not explain. The LSTM tries to learn
    # structure in this leftover component.
    residuals = np.asarray(arima_fit.resid, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    # Keep residual lag window feasible for the number of residual observations.
    seq_len = choose_seq_len(residuals, seq_len, horizon=1)

    # Standardize residuals before LSTM training.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(residuals.reshape(-1, 1)).ravel()
    # Build supervised residual windows: X=past residuals, y=next residual.
    x, y = make_supervised(scaled, seq_len=seq_len, horizon=1)
    if len(x) < 4 or np.isclose(np.std(residuals), 0):
        # If residuals are too few or nearly constant, use ARIMA alone.
        params = f"selected_d={selected_d}; {d_reason}; order={order}; aic={aic:.2f}; residual_model=fallback_zero"
        return ForecastResult(MODEL_NAME, params, arima_forecast)

    device = _device()
    # Tensor shape: batch x seq_len x 1.
    x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # Initialize residual LSTM and optimizer.
    model = ResidualLSTM(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        # Fit the residual correction model.
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    # Recursively forecast residual corrections into the future.
    history = list(scaled)
    residual_forecast = []
    with torch.no_grad():
        for _ in range(steps):
            x_input = torch.tensor(
                np.asarray(history[-seq_len:], dtype=float)[None, :, None],
                dtype=torch.float32,
                device=device,
            )
            pred = float(model(x_input).cpu().numpy()[0])
            # Predicted residual becomes part of the next lag window.
            history.append(pred)
            residual_forecast.append(pred)

    # Convert residual predictions back to the residual scale.
    residual_forecast = scaler.inverse_transform(np.asarray(residual_forecast).reshape(-1, 1)).ravel()
    # Hybrid forecast = ARIMA baseline + predicted residual correction.
    prediction = arima_forecast + residual_forecast
    params = (
        f"selected_d={selected_d}; {d_reason}; order={order}; aic={aic:.2f}; residual_seq_len={seq_len}; "
        f"hidden_size={hidden_size}; epochs={epochs}; lr={lr}; device={device.type}"
    )
    return ForecastResult(MODEL_NAME, params, prediction)

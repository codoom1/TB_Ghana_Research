"""Hybrid ARIMA + LSTM residual model for annual Ghana TB forecasting."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable
import warnings

import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def _fit_fold(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=False,
    )

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    stagnant_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x.to(device)), val_y.to(device)).item()

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def _fit_full_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    batch_size: int,
    epochs: int,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=False,
    )

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model


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
    epochs: int = 300,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 10,
    n_splits: int = 10,
    hidden_size: int = 12,
    seed: int = 42,
    progress_callback: Callable[[str], None] | None = None,
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
    x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    fold_best_epochs: list[int] = []
    effective_splits = min(n_splits, len(x_tensor) - 1)
    if effective_splits >= 2:
        splitter = TimeSeriesSplit(n_splits=effective_splits)
        split_indices = np.arange(len(x_tensor))
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(split_indices), start=1):
            if progress_callback is not None:
                progress_callback(
                    f"Training model ARIMA_LSTM, fold {fold_idx}/{effective_splits}: training on {len(train_idx)} residual windows, validating on {len(val_idx)} residual windows"
                )
            fold_model = ResidualLSTM(hidden_size=hidden_size).to(device)
            _, best_epoch = _fit_fold(
                fold_model,
                x_tensor[train_idx],
                y_tensor[train_idx],
                x_tensor[val_idx],
                y_tensor[val_idx],
                device=device,
                lr=lr,
                batch_size=batch_size,
                max_epochs=epochs,
                patience=patience,
            )
            fold_best_epochs.append(best_epoch)
            if progress_callback is not None:
                progress_callback(f"ARIMA_LSTM fold {fold_idx}/{effective_splits} done (best_epoch={best_epoch})")

    final_epochs = int(np.median(fold_best_epochs)) if fold_best_epochs else epochs
    final_epochs = max(1, min(epochs, final_epochs))

    # Refit the residual correction model on all windows.
    if progress_callback is not None:
        progress_callback(f"Training model ARIMA_LSTM on all residual windows for {final_epochs} epochs")
    model = ResidualLSTM(hidden_size=hidden_size).to(device)
    model = _fit_full_model(
        model,
        x_tensor,
        y_tensor,
        device=device,
        lr=lr,
        batch_size=batch_size,
        epochs=final_epochs,
    )

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
        f"hidden_size={hidden_size}; max_epochs={epochs}; final_epochs={final_epochs}; "
        f"batch_size={batch_size}; patience={patience}; n_splits={effective_splits}; lr={lr}; device={device.type}"
    )
    if progress_callback is not None:
        progress_callback("ARIMA_LSTM training complete")
    return ForecastResult(MODEL_NAME, params, prediction)

"""Recursive one-step LSTM model for annual Ghana TB forecasting."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_common import (
    DifferenceTransform,
    ForecastResult,
    choose_seq_len,
    make_supervised,
    set_global_seed,
)


MODEL_NAME = "LSTM"


class OneStepLSTM(nn.Module):
    def __init__(self, hidden_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        # input_size=1 because each time step contains one value: the
        # standardized first difference of the TB series.
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Use a small nonlinear head after the stacked LSTM layers.
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq_len x 1.
        output, _ = self.lstm(x)
        # Use only the final time step because it summarizes the lag window.
        return self.head(output[:, -1, :]).squeeze(-1)


def _device() -> torch.device:
    # Prefer Apple Silicon acceleration when available; otherwise use CPU.
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


def forecast(
    train: np.ndarray,
    steps: int,
    seq_len: int = 5,
    epochs: int = 300,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 10,
    n_splits: int = 10,
    hidden_size: int = 16,
    seed: int = 42,
    progress_callback: Callable[[str], None] | None = None,
) -> ForecastResult:
    # Make neural training reproducible across runs.
    set_global_seed(seed)
    train = np.asarray(train, dtype=float)
    # Train neural models on annual changes instead of nonstationary levels.
    difference_transform, train_differences = DifferenceTransform.fit_transform(train)
    # Keep lag window feasible for the short annual series.
    seq_len = choose_seq_len(train_differences, seq_len, horizon=1)

    # Standardize differences so neural optimization is numerically stable.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_differences.reshape(-1, 1)).ravel()
    # Build supervised windows: X=[past seq_len changes], y=next change.
    x, y = make_supervised(scaled, seq_len=seq_len, horizon=1)
    if len(x) < 4:
        # If too few windows exist, avoid fitting an unstable neural model.
        prediction = np.repeat(train[-1], steps)
        return ForecastResult(MODEL_NAME, "fallback=last_observation", prediction)

    device = _device()
    # PyTorch LSTM input shape is batch x seq_len x features.
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
                    f"Training model LSTM (2-layer + ReLU), fold {fold_idx}/{effective_splits}: training on {len(train_idx)} windows, validating on {len(val_idx)} windows"
                )
            fold_model = OneStepLSTM(hidden_size=hidden_size, num_layers=2).to(device)
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
                progress_callback(f"LSTM fold {fold_idx}/{effective_splits} done (best_epoch={best_epoch})")

    final_epochs = int(np.median(fold_best_epochs)) if fold_best_epochs else epochs
    final_epochs = max(1, min(epochs, final_epochs))

    # Refit the model on all available training windows before forecasting.
    if progress_callback is not None:
        progress_callback(f"Training model LSTM (2-layer + ReLU) on all windows for {final_epochs} epochs")
    model = OneStepLSTM(hidden_size=hidden_size, num_layers=2).to(device)
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
    # Recursive forecasting starts from the standardized training history.
    history = list(scaled)
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            # Feed the most recent lag window, including previous predictions
            # after the first forecast step.
            x_input = torch.tensor(
                np.asarray(history[-seq_len:], dtype=float)[None, :, None],
                dtype=torch.float32,
                device=device,
            )
            pred = float(model(x_input).cpu().numpy()[0])
            # Append the predicted standardized difference so it can be used in
            # the next recursive step.
            history.append(pred)
            predictions.append(pred)

    # Convert standardized predicted differences back to raw differences.
    predicted_differences = scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()
    # Reconstruct forecasted levels from the final observed training level.
    predictions = difference_transform.inverse_forecast(predicted_differences)
    params = (
        f"transform=first_difference; seq_len={seq_len}; hidden_size={hidden_size}; num_layers=2; relu_head=True; "
        f"max_epochs={epochs}; final_epochs={final_epochs}; batch_size={batch_size}; "
        f"patience={patience}; n_splits={effective_splits}; lr={lr}; device={device.type}"
    )
    if progress_callback is not None:
        progress_callback("LSTM training complete")
    return ForecastResult(MODEL_NAME, params, predictions)

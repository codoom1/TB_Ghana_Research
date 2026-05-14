"""Direct multistep LSTM model for annual Ghana TB forecasting."""

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


MODEL_NAME = "Multistep_LSTM"


class MultiStepLSTM(nn.Module):
    def __init__(self, horizon: int, hidden_size: int = 16, num_layers: int = 2) -> None:
        super().__init__()
        # Each input time step has one feature: the standardized first
        # difference of the annual TB series.
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Directly output one value for each forecast horizon step.
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq_len x 1.
        output, _ = self.lstm(x)
        # Final hidden state maps to a vector of future differences.
        return self.head(output[:, -1, :])


def _device() -> torch.device:
    # Use Apple MPS if available; otherwise train on CPU.
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
    # Fix random seeds so neural results are as reproducible as possible.
    set_global_seed(seed)
    train = np.asarray(train, dtype=float)
    # Difference the level series before neural training.
    difference_transform, train_differences = DifferenceTransform.fit_transform(train)
    # The direct target length is "steps", so lag length must leave enough
    # complete supervised windows.
    seq_len = choose_seq_len(train_differences, seq_len, horizon=steps)

    # Standardize the differenced series for stable gradient descent.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_differences.reshape(-1, 1)).ravel()
    # Build windows where each target is a vector of "steps" future changes.
    x, y = make_supervised(scaled, seq_len=seq_len, horizon=steps)
    if len(x) < 4:
        # Too few windows for a direct multistep neural model; use persistence.
        prediction = np.repeat(train[-1], steps)
        return ForecastResult(MODEL_NAME, "fallback=last_observation", prediction)

    device = _device()
    # X shape: batch x seq_len x 1. y shape: batch x steps.
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
                    f"Training model Multistep_LSTM, fold {fold_idx}/{effective_splits}: training on {len(train_idx)} windows, validating on {len(val_idx)} windows"
                )
            fold_model = MultiStepLSTM(horizon=steps, hidden_size=hidden_size, num_layers=2).to(device)
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
                progress_callback(f"Multistep_LSTM fold {fold_idx}/{effective_splits} done (best_epoch={best_epoch})")

    final_epochs = int(np.median(fold_best_epochs)) if fold_best_epochs else epochs
    final_epochs = max(1, min(epochs, final_epochs))

    # Refit on all available training windows before final forecasting.
    if progress_callback is not None:
        progress_callback(f"Training model Multistep_LSTM on all windows for {final_epochs} epochs")
    model = MultiStepLSTM(horizon=steps, hidden_size=hidden_size, num_layers=2).to(device)
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
    with torch.no_grad():
        # Final forecast uses the last observed lag window only once and returns
        # the entire future path.
        x_input = torch.tensor(
            np.asarray(scaled[-seq_len:], dtype=float)[None, :, None],
            dtype=torch.float32,
            device=device,
        )
        predictions = model(x_input).cpu().numpy().ravel()

    # Convert standardized predicted differences back to raw annual changes.
    predicted_differences = scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    # Accumulate predicted changes from the final observed level.
    predictions = difference_transform.inverse_forecast(predicted_differences)
    params = (
        f"transform=first_difference; seq_len={seq_len}; horizon={steps}; "
        f"hidden_size={hidden_size}; num_layers=2; relu_head=True; max_epochs={epochs}; final_epochs={final_epochs}; "
        f"batch_size={batch_size}; patience={patience}; n_splits={effective_splits}; lr={lr}; device={device.type}"
    )
    if progress_callback is not None:
        progress_callback("Multistep_LSTM training complete")
    return ForecastResult(MODEL_NAME, params, predictions)

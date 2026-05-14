"""Causal temporal convolutional network for annual Ghana TB forecasting."""

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


MODEL_NAME = "Causal_TCN"


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left padding only."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        # Left padding preserves sequence length while ensuring the convolution
        # never sees future values.
        self.left_padding = (kernel_size - 1) * dilation
        # Dilation spaces out kernel taps, increasing the historical receptive
        # field without adding many parameters.
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only on the left: [past padding | observed sequence].
        x = nn.functional.pad(x, (self.left_padding, 0))
        return self.conv(x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        # Two causal convolutions form one residual block. Dropout is small
        # because the dataset is short.
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection helps optimization by preserving the input signal.
        return x + self.net(x)


class CausalTCN(nn.Module):
    def __init__(
        self,
        channels: int = 16,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        # Convert the single input channel into a richer channel representation.
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        # Stack residual blocks with increasing dilation to see multiple lag
        # distances while preserving causal direction.
        self.blocks = nn.Sequential(
            *[
                ResidualTCNBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        # Map final time-step channels to one predicted next difference.
        self.output = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Incoming shape from make_supervised: batch x seq_len x 1.
        # Conv1d expects: batch x channels x seq_len.
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.blocks(x)
        # Use the representation at the last observed time step to predict the
        # next annual change.
        last_step = x[:, :, -1]
        return self.output(last_step).squeeze(-1)


def _device() -> torch.device:
    # Prefer Apple MPS acceleration if PyTorch exposes it.
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
    seq_len: int = 8,
    epochs: int = 300,
    lr: float = 0.001,
    batch_size: int = 16,
    patience: int = 10,
    n_splits: int = 5,
    channels: int = 16,
    kernel_size: int = 3,
    dropout: float = 0.10,
    seed: int = 42,
    progress_callback: Callable[[str], None] | None = None,
) -> ForecastResult:
    # Reproducible initialization and training.
    set_global_seed(seed)
    train = np.asarray(train, dtype=float)
    # Work on first differences to reduce trend/nonstationarity.
    difference_transform, train_differences = DifferenceTransform.fit_transform(train)
    # Keep the lag window valid for the short annual series.
    seq_len = choose_seq_len(train_differences, seq_len, horizon=1)
    # Kernel cannot be longer than the available input window.
    kernel_size = min(kernel_size, seq_len)

    # Standardize differenced values before neural training.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_differences.reshape(-1, 1)).ravel()
    # Build one-step supervised windows.
    x, y = make_supervised(scaled, seq_len=seq_len, horizon=1)
    if len(x) < 4:
        # Avoid fitting a neural model with too few examples.
        prediction = np.repeat(train[-1], steps)
        return ForecastResult(MODEL_NAME, "fallback=last_observation", prediction)

    device = _device()
    # Tensor shape before model transpose: batch x seq_len x 1.
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
                    f"Training model Causal_TCN, fold {fold_idx}/{effective_splits}: training on {len(train_idx)} windows, validating on {len(val_idx)} windows"
                )
            fold_model = CausalTCN(
                channels=channels,
                kernel_size=kernel_size,
                dilations=(1, 2, 4),
                dropout=dropout,
            ).to(device)
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
                progress_callback(f"Causal_TCN fold {fold_idx}/{effective_splits} done (best_epoch={best_epoch})")

    final_epochs = int(np.median(fold_best_epochs)) if fold_best_epochs else epochs
    final_epochs = max(1, min(epochs, final_epochs))

    # Refit on all available training windows before forecasting.
    if progress_callback is not None:
        progress_callback(f"Training model Causal_TCN on all windows for {final_epochs} epochs")
    model = CausalTCN(
        channels=channels,
        kernel_size=kernel_size,
        dilations=(1, 2, 4),
        dropout=dropout,
    ).to(device)
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
    # Recursive forecast: append each predicted standardized difference to the
    # history before predicting the next year.
    history = list(scaled)
    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            x_input = torch.tensor(
                np.asarray(history[-seq_len:], dtype=float)[None, :, None],
                dtype=torch.float32,
                device=device,
            )
            pred = float(model(x_input).cpu().numpy()[0])
            history.append(pred)
            predictions.append(pred)

    # Return from standardized differences to original annual changes.
    predicted_differences = scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()
    # Reconstruct incidence/mortality levels from predicted changes.
    predictions = difference_transform.inverse_forecast(predicted_differences)
    params = (
        f"transform=first_difference; architecture=causal_tcn; seq_len={seq_len}; "
        f"channels={channels}; kernel_size={kernel_size}; dilations=(1,2,4); "
        f"dropout={dropout}; max_epochs={epochs}; final_epochs={final_epochs}; "
        f"batch_size={batch_size}; patience={patience}; n_splits={effective_splits}; "
        f"lr={lr}; device={device.type}"
    )
    if progress_callback is not None:
        progress_callback("Causal_TCN training complete")
    return ForecastResult(MODEL_NAME, params, predictions)

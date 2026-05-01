"""Causal temporal convolutional network for annual Ghana TB forecasting."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

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
        channels: int = 12,
        kernel_size: int = 2,
        dilations: tuple[int, ...] = (1, 2),
        dropout: float = 0.05,
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


def forecast(
    train: np.ndarray,
    steps: int,
    seq_len: int = 5,
    epochs: int = 250,
    lr: float = 0.01,
    channels: int = 12,
    kernel_size: int = 2,
    dropout: float = 0.05,
    seed: int = 42,
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
    x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # Initialize the causal TCN. Dilations are fixed small values for the short
    # annual series.
    model = CausalTCN(
        channels=channels,
        kernel_size=kernel_size,
        dilations=(1, 2),
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        # Full-batch training is acceptable because the supervised dataset is
        # very small.
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

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
        f"channels={channels}; kernel_size={kernel_size}; dilations=(1,2); "
        f"dropout={dropout}; epochs={epochs}; lr={lr}; device={device.type}"
    )
    return ForecastResult(MODEL_NAME, params, predictions)

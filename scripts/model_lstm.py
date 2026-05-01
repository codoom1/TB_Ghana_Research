"""Recursive one-step LSTM model for annual Ghana TB forecasting."""

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


MODEL_NAME = "LSTM"


class OneStepLSTM(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        # input_size=1 because each time step contains one value: the
        # standardized first difference of the TB series.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        # Map the final hidden state to one predicted next difference.
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq_len x 1.
        output, _ = self.lstm(x)
        # Use only the final time step because it summarizes the lag window.
        return self.linear(output[:, -1, :]).squeeze(-1)


def _device() -> torch.device:
    # Prefer Apple Silicon acceleration when available; otherwise use CPU.
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def forecast(
    train: np.ndarray,
    steps: int,
    seq_len: int = 5,
    epochs: int = 250,
    lr: float = 0.01,
    hidden_size: int = 16,
    seed: int = 42,
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
    x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # Initialize the model and optimizer.
    model = OneStepLSTM(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        # Standard PyTorch training step: clear gradients, forward pass,
        # compute loss, backpropagate, and update parameters.
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

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
        f"transform=first_difference; seq_len={seq_len}; hidden_size={hidden_size}; "
        f"epochs={epochs}; lr={lr}; device={device.type}"
    )
    return ForecastResult(MODEL_NAME, params, predictions)

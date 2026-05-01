"""Direct multistep LSTM model for annual Ghana TB forecasting."""

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


MODEL_NAME = "Multistep_LSTM"


class MultiStepLSTM(nn.Module):
    def __init__(self, horizon: int, hidden_size: int = 16) -> None:
        super().__init__()
        # Each input time step has one feature: the standardized first
        # difference of the annual TB series.
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        # Directly output one value for each forecast horizon step.
        self.linear = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: batch x seq_len x 1.
        output, _ = self.lstm(x)
        # Final hidden state maps to a vector of future differences.
        return self.linear(output[:, -1, :])


def _device() -> torch.device:
    # Use Apple MPS if available; otherwise train on CPU.
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def forecast(
    train: np.ndarray,
    steps: int,
    seq_len: int = 5,
    epochs: int = 300,
    lr: float = 0.01,
    hidden_size: int = 16,
    seed: int = 42,
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
    x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # Initialize model and optimizer.
    model = MultiStepLSTM(horizon=steps, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        # Optimize all horizon outputs jointly with mean squared error.
        optimizer.zero_grad()
        loss = loss_fn(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

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
        f"hidden_size={hidden_size}; epochs={epochs}; lr={lr}; device={device.type}"
    )
    return ForecastResult(MODEL_NAME, params, predictions)

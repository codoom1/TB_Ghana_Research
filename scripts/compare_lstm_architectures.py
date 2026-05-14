"""Compare LSTM architecture and differencing choices for Ghana TB series.

This script uses the same temporal holdout as the rest of the project:
train on years before 2018 and score on 2018-2023. It compares:

* a one-layer LSTM vs a two-layer LSTM + ReLU head
* first-differenced inputs vs raw-level inputs

Training uses TimeSeriesSplit on the supervised windows to pick an epoch
budget, then refits on all available training windows before forecasting.

Run with the PyTorch environment:

    /opt/homebrew/anaconda3/envs/deepposture/bin/python scripts/compare_lstm_architectures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model_common import (
    DifferenceTransform,
    PRIMARY_SERIES,
    TABLE_DIR,
    choose_seq_len,
    ensure_output_dirs,
    load_modeling_data,
    make_supervised,
    metric_dict,
    set_global_seed,
    split_train_test,
)


MODEL_NAME_SINGLE = "LSTM_1Layer"
MODEL_NAME_DEEP = "LSTM_2Layer_ReLU"
TRANSFORM_DIFFERENCE = "difference"
TRANSFORM_LEVEL = "level"


class OneLayerLSTM(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.linear(output[:, -1, :]).squeeze(-1)


class TwoLayerLSTMReLU(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _prepare_training_signal(train: np.ndarray, transform_kind: str) -> tuple[DifferenceTransform | None, StandardScaler, np.ndarray]:
    train = np.asarray(train, dtype=float)
    if transform_kind == TRANSFORM_DIFFERENCE:
        transform, signal = DifferenceTransform.fit_transform(train)
    elif transform_kind == TRANSFORM_LEVEL:
        transform, signal = None, train
    else:
        raise ValueError(f"Unknown transform kind: {transform_kind}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(signal.reshape(-1, 1)).ravel()
    return transform, scaler, scaled


def _fit_model(
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
) -> int:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=False,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
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
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stagnant_epochs = 0
        else:
            stagnant_epochs += 1
            if stagnant_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch


def _select_epoch_budget(
    model_factory: Callable[[int], nn.Module],
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    *,
    device: torch.device,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    n_splits: int,
    hidden_size: int,
) -> int:
    effective_splits = min(n_splits, len(x_tensor) - 1)
    if effective_splits < 2:
        return max_epochs

    splitter = TimeSeriesSplit(n_splits=effective_splits)
    split_indices = np.arange(len(x_tensor))
    fold_best_epochs: list[int] = []

    for train_idx, val_idx in splitter.split(split_indices):
        fold_model = model_factory(hidden_size).to(device)
        best_epoch = _fit_fold(
            fold_model,
            x_tensor[train_idx],
            y_tensor[train_idx],
            x_tensor[val_idx],
            y_tensor[val_idx],
            device=device,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )
        fold_best_epochs.append(best_epoch)

    if not fold_best_epochs:
        return max_epochs
    return max(1, min(max_epochs, int(np.median(fold_best_epochs))))


def _forecast_with_model(
    train: np.ndarray,
    steps: int,
    model_factory: Callable[[int], nn.Module],
    *,
    transform_kind: str,
    seq_len: int,
    epochs: int,
    lr: float,
    batch_size: int,
    n_splits: int,
    patience: int,
    hidden_size: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    set_global_seed(seed)
    train = np.asarray(train, dtype=float)
    transform, scaler, scaled = _prepare_training_signal(train, transform_kind)
    seq_len = choose_seq_len(scaled, seq_len, horizon=1)

    x, y = make_supervised(scaled, seq_len=seq_len, horizon=1)
    if len(x) < 4:
        return np.repeat(train[-1], steps), "fallback=last_observation"

    device = _device()
    x_tensor = torch.tensor(x[:, :, None], dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    final_epochs = _select_epoch_budget(
        model_factory,
        x_tensor,
        y_tensor,
        device=device,
        lr=lr,
        batch_size=batch_size,
        max_epochs=epochs,
        patience=patience,
        n_splits=n_splits,
        hidden_size=hidden_size,
    )

    model = model_factory(hidden_size).to(device)
    model = _fit_model(
        model,
        x_tensor,
        y_tensor,
        device=device,
        lr=lr,
        batch_size=batch_size,
        epochs=final_epochs,
    )

    history = list(scaled)
    predictions = []
    model.eval()
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

    predicted_differences = scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()
    if transform_kind == TRANSFORM_DIFFERENCE:
        forecast = transform.inverse_forecast(predicted_differences)
        transform_label = "d1"
    else:
        forecast = predicted_differences
        transform_label = "none"
    params = (
        f"transform={transform_label}; seq_len={seq_len}; hidden_size={hidden_size}; "
        f"max_epochs={epochs}; final_epochs={final_epochs}; batch_size={batch_size}; "
        f"n_splits={n_splits}; patience={patience}; lr={lr}; device={device.type}"
    )
    return forecast, params


def compare_series(
    data: pd.DataFrame,
    series_name: str,
    *,
    seq_len: int,
    epochs: int,
    lr: float,
    batch_size: int,
    n_splits: int,
    patience: int,
    hidden_size: int,
    seed: int,
) -> pd.DataFrame:
    train, test, test_years = split_train_test(data, series_name)
    steps = len(test)

    configurations = [
        (
            MODEL_NAME_SINGLE,
            OneLayerLSTM,
            f"layers=1; activation=none; hidden_size={hidden_size}",
        ),
        (
            MODEL_NAME_DEEP,
            TwoLayerLSTMReLU,
            f"layers=2; activation=relu; hidden_size={hidden_size}",
        ),
    ]
    transforms = [
        (TRANSFORM_DIFFERENCE, "d1"),
        (TRANSFORM_LEVEL, "none"),
    ]

    rows = []
    for transform_kind, transform_label in transforms:
        for model_name, model_factory, architecture_note in configurations:
            predictions, params = _forecast_with_model(
                train,
                steps,
                model_factory,
                transform_kind=transform_kind,
                seq_len=seq_len,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                n_splits=n_splits,
                patience=patience,
                hidden_size=hidden_size,
                seed=seed,
            )
            row = {
                "series": series_name,
                "model": model_name,
                "transform": transform_label,
                "architecture": architecture_note,
                "params": params,
                "train_end": 2017,
                "test_start": 2018,
                "test_end": int(test_years.max()),
            }
            row.update(metric_dict(test, predictions))
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["series", "transform", "rmse", "mae"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one-layer/two-layer LSTM variants with and without first differencing.")
    parser.add_argument("--all-series", action="store_true", help="Compare every modeled series instead of only the primary series.")
    parser.add_argument("--seq-len", type=int, default=5, help="Requested lag window length.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for each model.")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--folds", type=int, default=5, help="TimeSeriesSplit folds used to choose an epoch budget.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience within each fold.")
    parser.add_argument("--hidden-size", type=int, default=16, help="LSTM hidden size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(TABLE_DIR / "lstm_architecture_transform_comparison.csv"),
        help="Where to save the comparison table.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    data = load_modeling_data()
    selected_series = PRIMARY_SERIES if not args.all_series else [series for series in data.columns if series != "year"]

    results = []
    for series_name in selected_series:
        results.append(
            compare_series(
                data,
                series_name,
                seq_len=args.seq_len,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                n_splits=args.folds,
                patience=args.patience,
                hidden_size=args.hidden_size,
                seed=args.seed,
            )
        )

    comparison = pd.concat(results, ignore_index=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    summary = comparison[["series", "transform", "model", "mae", "rmse", "mape"]].sort_values(["series", "transform", "rmse", "mae"])
    print(summary.to_string(index=False))
    print(f"\nSaved comparison table to {output_path}")


if __name__ == "__main__":
    main()
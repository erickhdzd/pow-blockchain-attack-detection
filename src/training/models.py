"""PyTorch neural architectures for tabular and sequential classification."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Feed-forward baseline over one telemetry observation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNN1DClassifier(nn.Module):
    """Temporal 1D-CNN over telemetry windows."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden_dim, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.transpose(1, 2)))


class RecurrentClassifier(nn.Module):
    """Shared LSTM/GRU sequence classifier."""

    def __init__(self, kind: str, input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = recurrent(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :])


class CNNLSTMClassifier(nn.Module):
    """Convolutional feature extraction followed by temporal LSTM modeling."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        output, _ = self.lstm(features)
        return self.head(output[:, -1, :])


class TransformerClassifier(nn.Module):
    """Transformer encoder with learned positional embeddings."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.projection(x) + self.position[:, : x.shape[1], :]
        encoded = self.encoder(encoded)
        return self.head(self.norm(encoded[:, -1, :]))


def build_neural_model(
    name: str,
    input_dim: int,
    sequence_length: int,
    settings: dict[str, Any],
) -> nn.Module:
    """Construct one supported neural classifier."""

    hidden = int(settings.get("hidden_dim", 128))
    dropout = float(settings.get("dropout", 0.2))
    if name == "mlp":
        return MLPClassifier(input_dim, hidden, dropout)
    if name == "cnn1d":
        return CNN1DClassifier(input_dim, hidden, dropout)
    if name in {"lstm", "gru"}:
        return RecurrentClassifier(
            name,
            input_dim,
            hidden,
            int(settings.get("num_layers", 2)),
            dropout,
        )
    if name == "cnn_lstm":
        return CNNLSTMClassifier(input_dim, hidden, dropout)
    if name == "transformer":
        return TransformerClassifier(
            input_dim=input_dim,
            sequence_length=sequence_length,
            d_model=int(settings.get("d_model", 128)),
            nhead=int(settings.get("nhead", 4)),
            num_layers=int(settings.get("num_layers", 3)),
            dim_feedforward=int(settings.get("dim_feedforward", 256)),
            dropout=dropout,
        )
    raise ValueError(f"Unknown neural model: {name!r}.")


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

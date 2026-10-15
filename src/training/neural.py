"""Deterministic PyTorch training loops with early stopping."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(slots=True)
class NeuralFitResult:
    """Best model state and runtime diagnostics."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_loss: float
    training_seconds: float
    peak_gpu_memory_mb: float


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda with an explicit CUDA availability check."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def fit_neural(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
) -> NeuralFitResult:
    """Fit a classifier and restore the lowest-validation-loss state."""

    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    counts = np.bincount(train_y, minlength=4).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    train_loader = _loader(train_x, train_y, batch_size, True, seed, num_workers)
    validation_loader = _loader(
        validation_x, validation_y, batch_size, False, seed, num_workers
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        validation_loss = _mean_loss(model, validation_loader, criterion, device)
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    seconds = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    model.load_state_dict(best_state)
    return NeuralFitResult(best_state, best_epoch, best_loss, seconds, float(peak))


def _mean_loss(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(features), labels)
            total += float(loss.item()) * len(labels)
            count += len(labels)
    return total / max(count, 1)


def predict_neural(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, float]:
    """Return probabilities and per-sample inference latency in milliseconds."""

    dummy_y = np.zeros(len(x), dtype=np.int64)
    loader = _loader(x, dummy_y, batch_size, False, 0, num_workers)
    model.eval()
    outputs: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device, non_blocking=True))
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), (elapsed * 1000.0 / max(len(x), 1))

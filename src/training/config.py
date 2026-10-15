# ruff: noqa: TRY004
"""Validated training configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Store validated experiment-wide training settings."""

    seeds: tuple[int, ...]
    device: str
    num_workers: int
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    sequence_length: int
    sequence_stride: int
    gradient_clip_norm: float
    save_models: bool
    dataset_root: Path
    results_root: Path
    tasks: dict[str, dict[str, Any]]
    classical_models: dict[str, dict[str, Any]]
    neural_models: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("Training seeds must be non-empty and non-negative.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        for name, value in {
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid optimizer hyperparameters.")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive.")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Validate and normalize a YAML mapping with string keys."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings.")
    return {str(key): item for key, item in value.items()}


def _require_nested_mapping(value: Any, name: str) -> dict[str, dict[str, Any]]:
    """Validate and normalize a nested YAML mapping with string keys."""

    outer = _require_mapping(value, name)
    nested: dict[str, dict[str, Any]] = {}
    for key, item in outer.items():
        nested[key] = _require_mapping(item, f"{name}.{key}")
    return nested


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load and validate the canonical YAML training configuration."""

    source = Path(path)
    raw = _require_mapping(
        yaml.safe_load(source.read_text(encoding="utf-8")),
        "training configuration",
    )
    training = _require_mapping(raw.get("training"), "training")
    paths = _require_mapping(raw.get("paths"), "paths")

    return TrainingConfig(
        seeds=tuple(int(seed) for seed in training["seeds"]),
        device=str(training["device"]),
        num_workers=int(training["num_workers"]),
        batch_size=int(training["batch_size"]),
        max_epochs=int(training["max_epochs"]),
        patience=int(training["patience"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        sequence_length=int(training["sequence_length"]),
        sequence_stride=int(training["sequence_stride"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        save_models=bool(training.get("save_models", False)),
        dataset_root=Path(paths["dataset_root"]),
        results_root=Path(paths["results_root"]),
        tasks=_require_nested_mapping(raw.get("tasks"), "tasks"),
        classical_models=_require_nested_mapping(
            raw.get("classical_models"),
            "classical_models",
        ),
        neural_models=_require_nested_mapping(
            raw.get("neural_models"),
            "neural_models",
        ),
    )

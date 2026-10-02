"""Leakage-safe tabular and sequential dataset loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class SplitData:
    """Aligned features, labels, and metadata for one split."""

    x: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]


@dataclass(slots=True)
class DatasetBundle:
    """Training, validation, and test splits with a train-fitted scaler."""

    train: SplitData
    validation: SplitData
    test: SplitData
    scaler: StandardScaler


def _read_split(root: Path, split: str, x_prefix: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    x = pd.read_csv(root / "splits" / f"{x_prefix}_{split}.csv")
    y = pd.read_csv(root / "splits" / f"y_{split}.csv")["attack_label"]
    metadata = pd.read_csv(root / "splits" / f"metadata_{split}.csv")
    if not (len(x) == len(y) == len(metadata)):
        raise ValueError(f"Misaligned rows in split {split!r}.")
    return x, y, metadata


def load_dataset_bundle(
    dataset_root: str | Path,
    x_prefix: str,
    drop_features: tuple[str, ...] = (),
) -> DatasetBundle:
    """Load splits, optionally ablate features, and fit scaling on training only."""

    root = Path(dataset_root)
    raw: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {
        split: _read_split(root, split, x_prefix)
        for split in ("train", "validation", "test")
    }
    if drop_features:
        missing = [
            feature
            for feature in drop_features
            if any(feature not in raw[split][0].columns for split in raw)
        ]
        if missing:
            raise ValueError(f"Cannot drop missing features: {sorted(set(missing))}.")
        raw = {
            split: (x.drop(columns=list(drop_features)), y, metadata)
            for split, (x, y, metadata) in raw.items()
        }

    feature_names = tuple(raw["train"][0].columns)
    if any(tuple(raw[split][0].columns) != feature_names for split in raw):
        raise ValueError("Feature schemas differ across splits.")
    scaler = StandardScaler()
    train_x = scaler.fit_transform(raw["train"][0].to_numpy(dtype=np.float32))

    def make(split: str, values: np.ndarray) -> SplitData:
        _, y, metadata = raw[split]
        return SplitData(
            x=np.asarray(values, dtype=np.float32),
            y=y.to_numpy(dtype=np.int64),
            metadata=metadata.reset_index(drop=True),
            feature_names=feature_names,
        )

    return DatasetBundle(
        train=make("train", train_x),
        validation=make(
            "validation",
            scaler.transform(raw["validation"][0].to_numpy(dtype=np.float32)),
        ),
        test=make(
            "test",
            scaler.transform(raw["test"][0].to_numpy(dtype=np.float32)),
        ),
        scaler=scaler,
    )


@dataclass(slots=True)
class SequenceData:
    """Contiguous windows that never cross simulation boundaries."""

    x: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


@dataclass(slots=True)
class AlignedSplitViews:
    """Endpoint and sequence representations of the same logical samples."""

    endpoints: SplitData
    sequences: SequenceData


@dataclass(slots=True)
class AlignedDatasetViews:
    """Aligned endpoint/sequence views for train, validation, and test."""

    train: AlignedSplitViews
    validation: AlignedSplitViews
    test: AlignedSplitViews


def _sequence_index_windows(data: SplitData, length: int, stride: int) -> list[list[int]]:
    if length <= 0 or stride <= 0:
        raise ValueError("Sequence length and stride must be positive.")
    windows: list[list[int]] = []
    simulation_ids = data.metadata["simulation_id"].astype(str).to_numpy()
    for simulation_id in pd.unique(simulation_ids):
        indices = np.flatnonzero(simulation_ids == simulation_id)
        if len(indices) < length:
            continue
        for start in range(0, len(indices) - length + 1, stride):
            selected_array = indices[start : start + length]
            if not np.all(np.diff(selected_array) == 1):
                raise ValueError("Simulation rows must be contiguous within a split.")
            windows.append([int(index) for index in selected_array])
    if not windows:
        raise ValueError("No sequences could be created.")
    return windows


def make_aligned_split_views(data: SplitData, length: int, stride: int) -> AlignedSplitViews:
    """Build tabular endpoints and temporal windows from identical sample endpoints."""

    index_windows = _sequence_index_windows(data, length, stride)
    endpoint_indices: list[int] = [window[-1] for window in index_windows]
    endpoint_metadata: pd.DataFrame = (
        data.metadata.iloc[endpoint_indices, :]
        .copy()
        .reset_index(drop=True)
    )
    endpoint_labels = np.take(
        data.y,
        endpoint_indices,
        axis=0,
    ).astype(np.int64, copy=True)
    endpoint_features = np.take(
        data.x,
        endpoint_indices,
        axis=0,
    ).astype(np.float32, copy=True)

    endpoints = SplitData(
        x=endpoint_features,
        y=endpoint_labels.copy(),
        metadata=endpoint_metadata.copy(),
        feature_names=data.feature_names,
    )
    sequences = SequenceData(
        x=np.stack(
            [np.take(data.x, window, axis=0) for window in index_windows],
            axis=0,
        ).astype(np.float32, copy=False),
        y=endpoint_labels,
        metadata=endpoint_metadata.copy(),
    )
    return AlignedSplitViews(endpoints=endpoints, sequences=sequences)


def make_aligned_dataset_views(
    bundle: DatasetBundle,
    length: int,
    stride: int,
) -> AlignedDatasetViews:
    """Create common logical samples for every model family in every split."""

    return AlignedDatasetViews(
        train=make_aligned_split_views(bundle.train, length, stride),
        validation=make_aligned_split_views(bundle.validation, length, stride),
        test=make_aligned_split_views(bundle.test, length, stride),
    )


def make_endpoint_view(data: SplitData, length: int, stride: int) -> SplitData:
    """Return final-timestep rows for the same windows used by sequence models."""

    return make_aligned_split_views(data, length, stride).endpoints


def make_sequences(data: SplitData, length: int, stride: int) -> SequenceData:
    """Create fixed-length windows labelled at the final timestep."""

    return make_aligned_split_views(data, length, stride).sequences

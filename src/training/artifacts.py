"""Canonical result serialization and paper-ready figures."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_result_directories(root: Path) -> dict[str, Path]:
    """Create the single canonical results directory tree."""

    directories = {
        name: root / name
        for name in ("metrics", "tables", "figures", "logs", "models", "manifests")
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def save_confusion_matrix(matrix: np.ndarray, path: Path, title: str) -> None:
    """Save a compact confusion-matrix figure."""

    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(matrix)
    fig.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(4))
    axis.set_yticks(range(4))
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(rows: list[dict[str, Any]], path: Path) -> None:
    """Write aggregate run metrics with stable column ordering."""

    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_runs(summary_path: Path, output_path: Path) -> pd.DataFrame:
    """Aggregate repeated seeds into mean and sample standard deviation."""

    frame = pd.read_csv(summary_path)
    metric_columns: list[str] = [
        column
        for column in map(str, frame.columns)
        if column not in {"task", "model", "seed"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    aggregations: dict[str, list[str]] = {
        column: ["mean", "std"] for column in metric_columns
    }
    aggregated = frame.groupby(["task", "model"], as_index=True).agg(aggregations)
    aggregated.columns = [f"{name}_{stat}" for name, stat in aggregated.columns]
    aggregated = aggregated.reset_index()
    aggregated.to_csv(output_path, index=False)
    return aggregated


def sha256_file(path: Path) -> str:
    """Hash a file using SHA-256."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON metadata."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")






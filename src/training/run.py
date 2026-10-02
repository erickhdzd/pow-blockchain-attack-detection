"""Command-line training entry point for all paper experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.utils.class_weight import compute_sample_weight

from src.training.artifacts import (
    aggregate_runs,
    ensure_result_directories,
    save_confusion_matrix,
    sha256_file,
    write_manifest,
    write_summary_table,
)
from src.training.classical import build_classical_model, predict_probabilities
from src.training.config import TrainingConfig, load_training_config
from src.training.data import AlignedDatasetViews, load_dataset_bundle, make_aligned_dataset_views
from src.training.metrics import MetricResult, detection_delay_blocks, evaluate_multiclass
from src.training.models import build_neural_model, parameter_count
from src.training.neural import fit_neural, predict_neural, resolve_device
from src.training.reproducibility import seed_everything

RunOutput = tuple[dict[str, Any], MetricResult]


def _sample_id_sha256(metadata: pd.DataFrame) -> str:
    """Hash ordered evaluation sample identifiers for cross-model auditability."""

    if "sample_id" not in metadata.columns:
        raise ValueError("Metadata must contain sample_id for evaluation auditing.")
    payload = "\n".join(metadata["sample_id"].astype(str)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_counts(views: AlignedDatasetViews) -> dict[str, int]:
    return {
        "train_samples": len(views.train.endpoints.y),
        "validation_samples": len(views.validation.endpoints.y),
        "test_samples": len(views.test.endpoints.y),
    }


def _run_classical(
    config: TrainingConfig,
    views: AlignedDatasetViews,
    task: str,
    model_name: str,
    settings: dict[str, Any],
    seed: int,
    directories: dict[str, Path],
) -> RunOutput:
    seed_everything(seed)
    model = build_classical_model(model_name, seed, settings)
    started = time.perf_counter()
    if model_name == "xgboost":
        sample_weight = compute_sample_weight("balanced", views.train.endpoints.y)
        model.fit(views.train.endpoints.x, views.train.endpoints.y, sample_weight=sample_weight)
    else:
        model.fit(views.train.endpoints.x, views.train.endpoints.y)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = predict_probabilities(model, views.test.endpoints.x)
    inference_ms = (time.perf_counter() - started) * 1000.0 / len(views.test.endpoints.y)
    predictions = probabilities.argmax(axis=1)
    metrics = evaluate_multiclass(views.test.endpoints.y, predictions, probabilities)
    delay = detection_delay_blocks(views.test.endpoints.metadata, predictions)
    if config.save_models:
        stem = f"{task}__{model_name}__seed_{seed}"
        joblib.dump(model, directories["models"] / f"{stem}.joblib")
    summary = {
        "task": task,
        "model": model_name,
        "seed": seed,
        **metrics.aggregate,
        **delay,
        "training_seconds": training_seconds,
        "inference_ms_per_sample": inference_ms,
        "parameter_count": 0,
        "peak_gpu_memory_mb": 0.0,
        "best_epoch": 0,
        **_sample_counts(views),
        "evaluation_sample_id_sha256": _sample_id_sha256(views.test.endpoints.metadata),
    }
    return summary, metrics


def _run_neural(
    config: TrainingConfig,
    views: AlignedDatasetViews,
    task: str,
    model_name: str,
    settings: dict[str, Any],
    seed: int,
    directories: dict[str, Path],
    profile: str,
) -> RunOutput:
    seed_everything(seed)
    is_mlp = model_name == "mlp"
    if is_mlp:
        train_x, train_y = views.train.endpoints.x, views.train.endpoints.y
        validation_x, validation_y = views.validation.endpoints.x, views.validation.endpoints.y
        test_x, test_y = views.test.endpoints.x, views.test.endpoints.y
        test_metadata = views.test.endpoints.metadata
    else:
        train_x, train_y = views.train.sequences.x, views.train.sequences.y
        validation_x, validation_y = views.validation.sequences.x, views.validation.sequences.y
        test_x, test_y = views.test.sequences.x, views.test.sequences.y
        test_metadata = views.test.sequences.metadata
    device = resolve_device(config.device)
    model = build_neural_model(
        model_name,
        views.train.endpoints.x.shape[1],
        config.sequence_length,
        settings,
    )
    epochs = min(config.max_epochs, 2) if profile == "smoke" else config.max_epochs
    patience = min(config.patience, 1) if profile == "smoke" else config.patience
    fit = fit_neural(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_epochs=epochs,
        patience=patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        seed=seed,
    )
    probabilities, inference_ms = predict_neural(
        model,
        test_x,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    predictions = probabilities.argmax(axis=1)
    metrics = evaluate_multiclass(test_y, predictions, probabilities)
    delay = detection_delay_blocks(test_metadata, predictions)
    if config.save_models:
        stem = f"{task}__{model_name}__seed_{seed}"
        torch.save(fit.state_dict, directories["models"] / f"{stem}.pt")
    summary = {
        "task": task,
        "model": model_name,
        "seed": seed,
        **metrics.aggregate,
        **delay,
        "training_seconds": fit.training_seconds,
        "inference_ms_per_sample": inference_ms,
        "parameter_count": parameter_count(model),
        "peak_gpu_memory_mb": fit.peak_gpu_memory_mb,
        "best_epoch": fit.best_epoch,
        **_sample_counts(views),
        "evaluation_sample_id_sha256": _sample_id_sha256(test_metadata),
    }
    return summary, metrics


def _record_run(
    summary: dict[str, Any],
    metrics: MetricResult,
    summaries: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    confusion_totals: dict[tuple[str, str], np.ndarray],
) -> None:
    summaries.append(summary)
    for raw_row in metrics.per_class.to_dict(orient="records"):
        row = {str(key): value for key, value in raw_row.items()}
        per_class_rows.append(
            {
                "task": summary["task"],
                "model": summary["model"],
                "seed": summary["seed"],
                **row,
            }
        )
    key = (str(summary["task"]), str(summary["model"]))
    if key not in confusion_totals:
        confusion_totals[key] = np.zeros((4, 4), dtype=np.int64)
    confusion_totals[key] += metrics.confusion


def run_experiments(config_path: str | Path, profile: str) -> Path:
    """Run the configured experiment matrix and return the aggregate table path."""

    config = load_training_config(config_path)
    directories = ensure_result_directories(config.results_root)
    seeds = config.seeds[:1] if profile == "smoke" else config.seeds
    tasks = {"general": config.tasks["general"]} if profile == "smoke" else config.tasks
    summaries: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_totals: dict[tuple[str, str], np.ndarray] = {}
    endpoint_audit: dict[str, dict[str, Any]] = {}
    for task, task_settings in tasks.items():
        drop_features = tuple(
            str(feature)
            for feature in task_settings.get("drop_features", ())
        )
        bundle = load_dataset_bundle(
            config.dataset_root,
            str(task_settings["x_prefix"]),
            drop_features=drop_features,
        )
        views = make_aligned_dataset_views(
            bundle,
            config.sequence_length,
            config.sequence_stride,
        )
        endpoint_audit[task] = {
            **_sample_counts(views),
            "feature_count": len(views.train.endpoints.feature_names),
            "drop_features": list(drop_features),
            "test_sample_id_sha256": _sample_id_sha256(views.test.endpoints.metadata),
            "test_class_counts": {
                str(label): int(np.sum(views.test.endpoints.y == label))
                for label in range(4)
            },
        }
        for name, settings in config.classical_models.items():
            if not bool(settings.get("enabled", True)):
                continue
            if profile == "smoke" and name != "logistic_regression":
                continue
            for seed in seeds:
                summary, metrics = _run_classical(
                    config,
                    views,
                    task,
                    name,
                    settings,
                    seed,
                    directories,
                )
                _record_run(summary, metrics, summaries, per_class_rows, confusion_totals)
        for name, settings in config.neural_models.items():
            if not bool(settings.get("enabled", True)):
                continue
            if profile == "smoke" and name not in {"mlp", "transformer"}:
                continue
            for seed in seeds:
                summary, metrics = _run_neural(
                    config,
                    views,
                    task,
                    name,
                    settings,
                    seed,
                    directories,
                    profile,
                )
                _record_run(summary, metrics, summaries, per_class_rows, confusion_totals)
    summary_path = directories["metrics"] / "run_metrics.csv"
    write_summary_table(summaries, summary_path)
    pd.DataFrame(per_class_rows).to_csv(
        directories["metrics"] / "per_class_metrics.csv",
        index=False,
    )
    for (task, model), matrix in confusion_totals.items():
        save_confusion_matrix(
            matrix,
            directories["figures"] / f"{task}__{model}__confusion.png",
            f"{task}: {model}",
        )
    aggregate_path = directories["tables"] / "model_comparison.csv"
    aggregate_runs(summary_path, aggregate_path)
    dataset_manifest = config.dataset_root / "manifests" / "dataset_manifest.json"
    manifest = {
        "profile": profile,
        "config_sha256": sha256_file(Path(config_path)),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "runs": len(summaries),
        "aligned_endpoints": endpoint_audit,
    }
    write_manifest(directories["manifests"] / "training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Aggregate table: {aggregate_path}")
    return aggregate_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    run_experiments(args.config, args.profile)


if __name__ == "__main__":
    main()

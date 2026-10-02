"""Robust statistical anomaly baseline for PoW attack detection.

The baseline models NORMAL training endpoints using robust univariate
location and scale estimates. Validation data are used only to select the
attack-detection threshold. Test data are not accessed during calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.training.data import SplitData, make_aligned_split_views

FEATURES = (
    "height_std",
    "consensus_ratio",
    "latest_block_interval_seconds",
    "mean_recent_block_interval_seconds",
    "std_recent_block_interval_seconds",
    "unique_miner_count",
    "dominant_miner_ratio",
    "miner_hhi",
)

MAD_NORMALIZATION = 1.4826


def load_aligned_split(
    root: Path,
    split: str,
    length: int,
    stride: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if split not in {"train", "validation"}:
        raise ValueError(
            "Calibration may access only train and validation."
        )

    x = pd.read_csv(root / f"x_early_{split}.csv")
    y = (
        pd.read_csv(root / f"y_{split}.csv")
        .iloc[:, 0]
        .to_numpy(dtype=np.int64)
    )
    metadata = pd.read_csv(
        root / f"metadata_{split}.csv"
    )

    missing = set(FEATURES) - set(x.columns)
    if missing:
        raise ValueError(
            f"{split} is missing features: {sorted(missing)}"
        )

    if not (len(x) == len(y) == len(metadata)):
        raise ValueError(
            f"{split}: X/y/metadata length mismatch."
        )

    selected = pd.DataFrame(
        x.loc[:, list(FEATURES)].to_numpy(dtype=float),
        columns=list(FEATURES),
        index=x.index,
    )

    split_data = SplitData(
        x=selected.to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        y=y,
        metadata=metadata,
        feature_names=tuple(FEATURES),
    )

    aligned = make_aligned_split_views(
        split_data,
        length,
        stride,
    ).endpoints

    aligned_x = pd.DataFrame(
        aligned.x,
        columns=list(FEATURES),
    )

    return aligned_x, aligned.y, aligned.metadata


def fit_normal_reference(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normal = x_train.loc[y_train == 0]

    if normal.empty:
        raise ValueError("Training data contain no NORMAL endpoints.")

    median = normal.median(axis=0).to_numpy(dtype=float)
    mad = (
        (normal - normal.median(axis=0))
        .abs()
        .median(axis=0)
        .to_numpy(dtype=float)
    )
    scale = MAD_NORMALIZATION * mad

    if not np.isfinite(median).all():
        raise ValueError("Non-finite NORMAL medians.")
    if not np.isfinite(scale).all():
        raise ValueError("Non-finite NORMAL scales.")
    if np.any(scale <= 0.0):
        bad = [
            FEATURES[index]
            for index in np.flatnonzero(scale <= 0.0)
        ]
        raise ValueError(
            f"Non-positive robust scales: {bad}"
        )

    return median, scale


def anomaly_scores(
    x: pd.DataFrame,
    median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = x.loc[:, FEATURES].to_numpy(dtype=float)
    z = (values - median) / scale
    scores = np.sqrt(np.mean(np.square(z), axis=1))

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly scores.")

    return scores


def binary_metrics(
    y_true_multiclass: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = (y_true_multiclass != 0).astype(np.int64)
    y_pred = (scores >= threshold).astype(np.int64)

    return {
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique_scores = np.unique(scores)

    if unique_scores.size < 2:
        raise ValueError(
            "Validation anomaly scores contain fewer than two values."
        )

    return np.concatenate(
        (
            np.array(
                [
                    np.nextafter(
                        unique_scores[0],
                        -np.inf,
                    )
                ]
            ),
            (
                unique_scores[:-1]
                + unique_scores[1:]
            )
            / 2.0,
            np.array(
                [
                    np.nextafter(
                        unique_scores[-1],
                        np.inf,
                    )
                ]
            ),
        )
    )


def select_threshold(
    y_validation: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None

    for threshold_value in candidate_thresholds(scores):
        threshold: float = threshold_value.item()
        metrics = binary_metrics(
            y_validation,
            scores,
            threshold,
        )

        if best_metrics is None or best_threshold is None:
            best_threshold = threshold
            best_metrics = metrics
            continue

        candidate_key = (
            metrics["f1"],
            metrics["mcc"],
            metrics["balanced_accuracy"],
            -threshold,
        )
        best_key = (
            best_metrics["f1"],
            best_metrics["mcc"],
            best_metrics["balanced_accuracy"],
            -best_threshold,
        )

        if candidate_key > best_key:
            best_threshold = threshold
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Threshold selection failed.")

    return best_threshold, best_metrics


def calibrate(
    data_root: Path,
    output_path: Path,
    length: int,
    stride: int,
) -> None:
    x_train, y_train, train_metadata = load_aligned_split(
        data_root,
        "train",
        length,
        stride,
    )
    (
        x_validation,
        y_validation,
        validation_metadata,
    ) = load_aligned_split(
        data_root,
        "validation",
        length,
        stride,
    )

    print(
        "Aligned endpoints: "
        f"train={len(y_train)}, "
        f"validation={len(y_validation)}"
    )

    if len(y_train) != 9540:
        raise ValueError(
            f"Expected 9540 train endpoints, got {len(y_train)}."
        )
    if len(y_validation) != 1980:
        raise ValueError(
            "Expected 1980 validation endpoints, "
            f"got {len(y_validation)}."
        )

    if train_metadata["simulation_id"].nunique() != 212:
        raise ValueError(
            "Unexpected train simulation count."
        )
    if validation_metadata["simulation_id"].nunique() != 44:
        raise ValueError(
            "Unexpected validation simulation count."
        )

    median, scale = fit_normal_reference(
        x_train,
        y_train,
    )

    validation_scores = anomaly_scores(
        x_validation,
        median,
        scale,
    )

    threshold, metrics = select_threshold(
        y_validation,
        validation_scores,
    )

    artifact = {
        "method": "robust_statistical_anomaly_baseline",
        "task": "binary_attack_detection",
        "normal_label": 0,
        "attack_definition": "multiclass_label != 0",
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "location": "normal_train_median",
        "scale": "1.4826_times_normal_train_mad",
        "score": "root_mean_squared_robust_z",
        "threshold_source": "validation",
        "threshold_objective": "attack_f1",
        "alignment": {
            "sequence_length": length,
            "stride": stride,
            "endpoint": "final_timestep",
            "train_endpoint_count": int(len(y_train)),
            "validation_endpoint_count": int(
                len(y_validation)
            ),
            "train_simulation_count": int(
                train_metadata["simulation_id"].nunique()
            ),
            "validation_simulation_count": int(
                validation_metadata[
                    "simulation_id"
                ].nunique()
            ),
        },
        "threshold_tiebreakers": [
            "mcc",
            "balanced_accuracy",
            "lower_threshold",
        ],
        "threshold": threshold,
        "validation_metrics": metrics,
        "normal_train_count": int(
            np.count_nonzero(y_train == 0)
        ),
        "validation_count": int(len(y_validation)),
        "validation_attack_count": int(
            np.count_nonzero(y_validation != 0)
        ),
        "normal_median": {
            feature: float(value)
            for feature, value in zip(
                FEATURES,
                median,
                strict=True,
            )
        },
        "normal_scale": {
            feature: float(value)
            for feature, value in zip(
                FEATURES,
                scale,
                strict=True,
            )
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("Calibration completed.")
    print(f"Threshold: {threshold:.12f}")
    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Artifact: {output_path}")



def load_test_aligned_split(
    root: Path,
    features: tuple[str, ...],
    length: int,
    stride: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load test endpoints without exposing test to calibration."""

    x = pd.read_csv(root / "x_early_test.csv")
    y = (
        pd.read_csv(root / "y_test.csv")
        .iloc[:, 0]
        .to_numpy(dtype=np.int64)
    )
    metadata = pd.read_csv(
        root / "metadata_test.csv"
    )

    missing = set(features) - set(x.columns)
    if missing:
        raise ValueError(
            f"test is missing features: {sorted(missing)}"
        )

    if not (len(x) == len(y) == len(metadata)):
        raise ValueError(
            "test: X/y/metadata length mismatch."
        )

    selected = pd.DataFrame(
        x.loc[:, list(features)].to_numpy(dtype=float),
        columns=list(features),
        index=x.index,
    )

    split_data = SplitData(
        x=selected.to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        y=y,
        metadata=metadata,
        feature_names=features,
    )

    aligned = make_aligned_split_views(
        split_data,
        length,
        stride,
    ).endpoints

    aligned_x = pd.DataFrame(
        aligned.x,
        columns=list(features),
    )

    return aligned_x, aligned.y, aligned.metadata


def evaluate_frozen(
    data_root: Path,
    calibration_path: Path,
    output_path: Path,
) -> None:
    """Evaluate test once using an immutable calibration artifact."""

    artifact = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )

    if artifact.get("method") != (
        "robust_statistical_anomaly_baseline"
    ):
        raise ValueError(
            "Unexpected calibration method."
        )

    if artifact.get("threshold_source") != "validation":
        raise ValueError(
            "Frozen threshold was not selected on validation."
        )

    features_value = artifact.get("features")
    if not isinstance(features_value, list):
        raise ValueError(
            "Calibration artifact has invalid features."
        )

    features = tuple(
        str(feature)
        for feature in features_value
    )

    if features != FEATURES:
        raise ValueError(
            "Frozen feature schema does not match "
            "the canonical baseline."
        )

    alignment = artifact.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError(
            "Calibration artifact has no alignment metadata."
        )

    length_value = alignment.get("sequence_length")
    stride_value = alignment.get("stride")

    if not isinstance(length_value, int):
        raise ValueError(
            "Invalid frozen sequence length."
        )
    if not isinstance(stride_value, int):
        raise ValueError(
            "Invalid frozen stride."
        )

    threshold_value = artifact.get("threshold")
    if not isinstance(threshold_value, (int, float)):
        raise ValueError(
            "Invalid frozen threshold."
        )
    threshold = float(threshold_value)

    median_values = artifact.get("normal_median")
    scale_values = artifact.get("normal_scale")

    if not isinstance(median_values, dict):
        raise ValueError(
            "Invalid frozen NORMAL medians."
        )
    if not isinstance(scale_values, dict):
        raise ValueError(
            "Invalid frozen NORMAL scales."
        )

    try:
        median = np.asarray(
            [
                float(median_values[feature])
                for feature in features
            ],
            dtype=float,
        )
        scale = np.asarray(
            [
                float(scale_values[feature])
                for feature in features
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid frozen feature statistics."
        ) from exc

    if not np.isfinite(median).all():
        raise ValueError(
            "Frozen medians contain non-finite values."
        )
    if not np.isfinite(scale).all():
        raise ValueError(
            "Frozen scales contain non-finite values."
        )
    if np.any(scale <= 0.0):
        raise ValueError(
            "Frozen scales must be positive."
        )
    if not np.isfinite(threshold):
        raise ValueError(
            "Frozen threshold must be finite."
        )

    x_test, y_test, test_metadata = (
        load_test_aligned_split(
            data_root,
            features,
            length_value,
            stride_value,
        )
    )

    if len(y_test) != 1980:
        raise ValueError(
            f"Expected 1980 test endpoints, got {len(y_test)}."
        )

    if test_metadata["simulation_id"].nunique() != 44:
        raise ValueError(
            "Unexpected test simulation count."
        )

    expected_counts = {
        0: 1882,
        1: 69,
        2: 16,
        3: 13,
    }
    observed_counts = {
        int(label): int(count)
        for label, count in zip(
            *np.unique(y_test, return_counts=True),
            strict=True,
        )
    }

    if observed_counts != expected_counts:
        raise ValueError(
            "Unexpected test label distribution: "
            f"{observed_counts}"
        )

    scores = anomaly_scores(
        x_test,
        median,
        scale,
    )
    metrics = binary_metrics(
        y_test,
        scores,
        threshold,
    )

    y_binary = (y_test != 0).astype(np.int64)
    predictions = (
        scores >= threshold
    ).astype(np.int64)

    tp = int(
        np.count_nonzero(
            (y_binary == 1) & (predictions == 1)
        )
    )
    fp = int(
        np.count_nonzero(
            (y_binary == 0) & (predictions == 1)
        )
    )
    tn = int(
        np.count_nonzero(
            (y_binary == 0) & (predictions == 0)
        )
    )
    fn = int(
        np.count_nonzero(
            (y_binary == 1) & (predictions == 0)
        )
    )

    result = {
        "method": artifact["method"],
        "task": "binary_attack_detection",
        "calibration_artifact": str(calibration_path),
        "threshold": threshold,
        "threshold_source": "validation",
        "features": list(features),
        "feature_count": len(features),
        "sequence_length": length_value,
        "stride": stride_value,
        "test_endpoint_count": int(len(y_test)),
        "test_simulation_count": int(
            test_metadata["simulation_id"].nunique()
        ),
        "test_multiclass_counts": observed_counts,
        "test_binary_counts": {
            "normal": int(
                np.count_nonzero(y_binary == 0)
            ),
            "attack": int(
                np.count_nonzero(y_binary == 1)
            ),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "metrics": metrics,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print("Frozen test evaluation completed.")
    print(
        "Aligned test endpoints: "
        f"{len(y_test)}"
    )
    print(
        "Test label counts: "
        f"{observed_counts}"
    )
    print(
        "Binary confusion matrix: "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )
    print("Test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"Artifact: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Fit NORMAL reference and select threshold.",
    )
    calibrate_parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/splits"),
    )
    calibrate_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/statistical_baseline/"
            "calibration.json"
        ),
    )
    calibrate_parser.add_argument(
        "--length",
        type=int,
        default=16,
    )
    calibrate_parser.add_argument(
        "--stride",
        type=int,
        default=4,
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate frozen calibration on test.",
    )
    evaluate_parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/splits"),
    )
    evaluate_parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "results/statistical_baseline/"
            "calibration_aligned_frozen.json"
        ),
    )
    evaluate_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/statistical_baseline/"
            "test_evaluation.json"
        ),
    )

    args = parser.parse_args()

    if args.command == "calibrate":
        calibrate(
            args.data_root,
            args.output,
            args.length,
            args.stride,
        )
        return

    if args.command == "evaluate":
        evaluate_frozen(
            args.data_root,
            args.calibration,
            args.output,
        )
        return

    raise RuntimeError(
        f"Unsupported command: {args.command}"
    )


if __name__ == "__main__":
    main()

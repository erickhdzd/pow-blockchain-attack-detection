"""Multiclass metrics and detection-delay evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ("normal", "double_spend", "selfish_mining", "majority_attack")
ATTACK_NAME_TO_ID = {name.upper(): index for index, name in enumerate(CLASS_NAMES)}


@dataclass(slots=True, frozen=True)
class MetricResult:
    """Aggregate and per-class evaluation outputs."""

    aggregate: dict[str, float]
    per_class: pd.DataFrame
    confusion: np.ndarray


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> MetricResult:
    """Compute stable four-class metrics from predictions and probabilities."""

    labels = np.arange(4)
    precision_raw, recall_raw, f1_raw, support_raw = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    precision = np.asarray(precision_raw)
    recall = np.asarray(recall_raw)
    f1 = np.asarray(f1_raw)
    support = np.asarray(support_raw)
    y_bin = np.asarray(label_binarize(y_true, classes=labels))
    aggregate = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_roc_auc": float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr")),
        "macro_pr_auc": float(_macro_pr_auc(y_bin, probabilities)),
    }
    rows = []
    for label, name in enumerate(CLASS_NAMES):
        truth_positive = y_true == label
        pred_positive = y_pred == label
        false_positive = np.logical_and(~truth_positive, pred_positive).sum()
        true_negative = np.logical_and(~truth_positive, ~pred_positive).sum()
        denominator = false_positive + true_negative
        rows.append(
            {
                "class_id": label,
                "class_name": name,
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
                "false_positive_rate": float(false_positive / denominator) if denominator else 0.0,
                "detection_rate": float(recall[label]),
            }
        )
    return MetricResult(
        aggregate=aggregate,
        per_class=pd.DataFrame(rows),
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )


def _macro_pr_auc(y_bin: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_bin, probabilities, average="macro"))


def detection_delay_blocks(metadata: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    """Measure first correct attack detection after configured attack onset."""

    frame = metadata.copy()
    frame["prediction"] = y_pred
    delays: list[int] = []
    detected = 0
    eligible = 0
    for _, group in frame.groupby("simulation_id", sort=False):
        configured = _attack_type_id(group["configured_attack_type"].iloc[0])
        if configured == 0:
            continue
        eligible += 1
        start = int(group["attack_start_step"].iloc[0])
        active = group[group["step"] >= start]
        correct = active[active["prediction"] == configured]
        if correct.empty:
            continue
        detected += 1
        delays.append(int(correct["step"].iloc[0]) - start)
    return {
        "attack_scenarios": float(eligible),
        "detected_attack_scenarios": float(detected),
        "scenario_detection_rate": float(detected / eligible) if eligible else 0.0,
        "mean_detection_delay_blocks": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay_blocks": float(np.median(delays)) if delays else float("nan"),
    }


def _attack_type_id(value: object) -> int:
    """Normalize numeric or textual attack metadata to the stable class ID."""

    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in ATTACK_NAME_TO_ID:
            return ATTACK_NAME_TO_ID[normalized]
        try:
            return int(normalized)
        except ValueError as exc:
            raise ValueError(f"Unknown attack type metadata: {value!r}.") from exc

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)

    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return int(numeric_value)

    raise ValueError("Unknown attack type metadata value.")



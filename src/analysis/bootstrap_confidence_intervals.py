"""Stratified simulation-level bootstrap confidence intervals.

The bootstrap resamples complete simulations with replacement within each
configured attack scenario. All aligned endpoints belonging to a selected
simulation are retained together, preserving within-simulation dependence.

For each bootstrap replicate, the same resampled simulations are evaluated
for every training seed. Metrics are first computed independently for each
seed and then averaged across seeds. Percentile confidence intervals therefore
quantify test-simulation sampling uncertainty around mean multi-seed
performance without treating training seeds or overlapping endpoints as
independent observations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

LABELS = (0, 1, 2, 3)
ATTACK_LABELS = (1, 2, 3)
CLASS_NAMES = {
    0: "normal",
    1: "double_spend",
    2: "selfish_mining",
    3: "majority_attack",
}
SEEDS = (42, 123, 2026, 2027, 2028)


@dataclass(frozen=True)
class Experiment:
    task: str
    model: str
    prediction_root: Path


EXPERIMENTS = (
    Experiment(
        task="general",
        model="gru",
        prediction_root=Path("results/bootstrap_general_gru/predictions"),
    ),
    Experiment(
        task="early",
        model="cnn_lstm",
        prediction_root=Path("results/bootstrap_early_cnn_lstm/predictions"),
    ),
)


def load_scenario_strata(path: Path) -> dict[str, tuple[str, ...]]:
    scenarios = pd.read_csv(str(path))

    required = {"simulation_id", "attack_type", "split"}
    missing = required - set(scenarios.columns)
    if missing:
        raise ValueError(f"Scenario manifest is missing columns: {sorted(missing)}")

    test = scenarios.loc[scenarios["split"] == "test"].copy()
    if test["simulation_id"].duplicated().any():
        raise ValueError("Duplicate test simulation IDs in scenario manifest.")

    strata: dict[str, tuple[str, ...]] = {}
    for attack_type, group in test.groupby("attack_type", sort=True):
        ids = tuple(sorted(group["simulation_id"].astype(str)))
        strata[str(attack_type)] = ids

    if len(strata) != 4:
        raise ValueError(f"Expected four scenario strata, found {len(strata)}.")

    if any(len(ids) != 11 for ids in strata.values()):
        raise ValueError(
            "Expected exactly 11 test simulations in every scenario stratum."
        )

    return strata


def load_predictions(experiment: Experiment) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}

    required = {
        "simulation_id",
        "step",
        "y_true",
        "y_pred",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
    }

    reference_identity: pd.DataFrame | None = None

    for seed in SEEDS:
        path = (
            experiment.prediction_root
            / f"{experiment.task}__{experiment.model}__seed_{seed}.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(str(path))
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        frame["simulation_id"] = frame["simulation_id"].astype(str)

        identity = frame[["simulation_id", "step", "y_true"]].reset_index(drop=True)

        if reference_identity is None:
            reference_identity = identity
        elif not identity.equals(reference_identity):
            raise ValueError(f"Evaluation endpoints differ across seeds: {path}")

        frames[seed] = frame

    return frames


def validate_prediction_strata(
    frames: dict[int, pd.DataFrame],
    strata: dict[str, tuple[str, ...]],
) -> None:
    expected_ids = {
        simulation_id
        for simulation_ids in strata.values()
        for simulation_id in simulation_ids
    }

    for seed, frame in frames.items():
        actual_ids = set(frame["simulation_id"].unique())
        if actual_ids != expected_ids:
            raise ValueError(
                f"Seed {seed}: prediction simulations do not match "
                "the scenario manifest."
            )


def macro_pr_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    binary = np.column_stack([(y_true == label).astype(np.int64) for label in LABELS])

    scores = [
        float(
            average_precision_score(
                binary[:, label],
                probabilities[:, label],
            )
        )
        for label in LABELS
    ]
    return float(np.mean(scores))


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=np.int64)
    y_pred = frame["y_pred"].to_numpy(dtype=np.int64)
    probabilities = frame[["prob_0", "prob_1", "prob_2", "prob_3"]].to_numpy(
        dtype=float
    )

    result = {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_pr_auc": macro_pr_auc(y_true, probabilities),
    }

    for label in ATTACK_LABELS:
        name = CLASS_NAMES[label]
        result[f"{name}_precision"] = float(
            precision_score(
                y_true,
                y_pred,
                labels=[label],
                average="macro",
                zero_division=0,
            )
        )
        result[f"{name}_recall"] = float(
            recall_score(
                y_true,
                y_pred,
                labels=[label],
                average="macro",
                zero_division=0,
            )
        )
        result[f"{name}_f1"] = float(
            f1_score(
                y_true,
                y_pred,
                labels=[label],
                average="macro",
                zero_division=0,
            )
        )

    return result


def simulation_groups(
    frames: dict[int, pd.DataFrame],
) -> dict[int, dict[str, pd.DataFrame]]:
    result: dict[int, dict[str, pd.DataFrame]] = {}

    for seed, frame in frames.items():
        seed_groups: dict[str, pd.DataFrame] = {}

        for simulation_id, group in frame.groupby(
            "simulation_id",
            sort=False,
        ):
            seed_groups[str(simulation_id)] = group.reset_index(drop=True)

        result[seed] = seed_groups

    return result


def draw_simulations(
    rng: np.random.Generator,
    strata: dict[str, tuple[str, ...]],
) -> list[str]:
    selected: list[str] = []

    for attack_type in sorted(strata):
        ids = np.asarray(strata[attack_type], dtype=object)
        draw = rng.choice(ids, size=len(ids), replace=True)
        selected.extend(str(value) for value in draw)

    return selected


def materialize_bootstrap_frame(
    groups: dict[str, pd.DataFrame],
    selected_ids: list[str],
) -> pd.DataFrame:
    return pd.concat(
        [groups[simulation_id] for simulation_id in selected_ids],
        ignore_index=True,
    )


def original_multi_seed_metrics(
    frames: dict[int, pd.DataFrame],
) -> dict[str, float]:
    per_seed_metrics = [compute_metrics(frames[seed]) for seed in SEEDS]
    metric_names = tuple(per_seed_metrics[0])

    return {
        metric: float(
            np.mean([seed_metrics[metric] for seed_metrics in per_seed_metrics])
        )
        for metric in metric_names
    }


def bootstrap_experiment(
    experiment: Experiment,
    strata: dict[str, tuple[str, ...]],
    replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = load_predictions(experiment)
    validate_prediction_strata(frames, strata)
    groups = simulation_groups(frames)

    point = original_multi_seed_metrics(frames)
    metric_names = tuple(point)

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_values = {
        metric: np.empty(replicates, dtype=float) for metric in metric_names
    }

    for replicate in range(replicates):
        selected_ids = draw_simulations(rng, strata)

        seed_metrics = []
        for seed in SEEDS:
            sampled = materialize_bootstrap_frame(
                groups[seed],
                selected_ids,
            )
            seed_metrics.append(compute_metrics(sampled))

        for metric in metric_names:
            bootstrap_values[metric][replicate] = float(
                np.mean([values[metric] for values in seed_metrics])
            )

    rows = []
    for metric in metric_names:
        values = bootstrap_values[metric]
        lower, upper = np.percentile(values, [2.5, 97.5])

        rows.append(
            {
                "task": experiment.task,
                "model": experiment.model,
                "metric": metric,
                "point_estimate": point[metric],
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": bootstrap_seed,
            }
        )

    replicates_frame = pd.DataFrame(
        {
            "replicate": np.arange(replicates, dtype=int),
            **bootstrap_values,
        }
    )
    replicates_frame.insert(0, "model", experiment.model)
    replicates_frame.insert(0, "task", experiment.task)

    return pd.DataFrame(rows), replicates_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        default=Path("data/manifests/scenarios.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/bootstrap_statistics"),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )
    args = parser.parse_args()

    if args.replicates < 1:
        parser.error("--replicates must be positive.")

    strata = load_scenario_strata(args.scenario_manifest)

    all_intervals: list[pd.DataFrame] = []
    all_replicates: list[pd.DataFrame] = []

    for experiment in EXPERIMENTS:
        experiment_intervals, experiment_replicates = bootstrap_experiment(
            experiment,
            strata,
            args.replicates,
            args.seed,
        )
        all_intervals.append(experiment_intervals)
        all_replicates.append(experiment_replicates)

    args.output_root.mkdir(parents=True, exist_ok=True)

    intervals: pd.DataFrame = pd.concat(
        all_intervals,
        ignore_index=True,
    )
    replicates: pd.DataFrame = pd.concat(
        all_replicates,
        ignore_index=True,
    )

    intervals.to_csv(
        args.output_root / "confidence_intervals.csv",
        index=False,
    )
    replicates.to_csv(
        args.output_root / "bootstrap_replicates.csv",
        index=False,
    )

    audit = {
        "method": "stratified_simulation_cluster_bootstrap",
        "confidence_interval": "percentile_95",
        "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "training_seeds": list(SEEDS),
        "scenario_strata": {
            attack_type: list(simulation_ids)
            for attack_type, simulation_ids in strata.items()
        },
        "simulations_per_stratum": {
            attack_type: len(simulation_ids)
            for attack_type, simulation_ids in strata.items()
        },
        "total_test_simulations": sum(
            len(simulation_ids) for simulation_ids in strata.values()
        ),
        "endpoint_resampling": False,
        "seed_aggregation": "metric_then_mean_across_training_seeds",
    }

    (args.output_root / "bootstrap_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    display = intervals.copy()
    for column in (
        "point_estimate",
        "ci_lower_95",
        "ci_upper_95",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4f}")

    print(display.to_string(index=False))
    print(f"\nResults written to: {args.output_root}")


if __name__ == "__main__":
    main()

"""Leakage-safe simulation-level dataset splitting."""

from __future__ import annotations

from collections import defaultdict
from random import Random

from src.dataset.config import SplitRatios
from src.dataset.scenario import ScenarioSpec

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


def assign_simulation_splits(
    scenarios: tuple[ScenarioSpec, ...],
    ratios: SplitRatios,
    seed: int,
) -> dict[str, str]:
    """Assign whole simulations to stratified, mutually exclusive splits."""

    grouped: dict[int, list[ScenarioSpec]] = defaultdict(list)
    for scenario in scenarios:
        grouped[int(scenario.attack_type)].append(scenario)

    assignments: dict[str, str] = {}

    for attack_value in sorted(grouped):
        group = list(grouped[attack_value])
        rng = Random(seed + attack_value * 7919)
        rng.shuffle(group)
        train_count, validation_count, test_count = _split_counts(
            len(group),
            ratios,
        )

        boundaries = (
            ("train", 0, train_count),
            (
                "validation",
                train_count,
                train_count + validation_count,
            ),
            (
                "test",
                train_count + validation_count,
                train_count + validation_count + test_count,
            ),
        )

        for split_name, start, end in boundaries:
            for scenario in group[start:end]:
                assignments[scenario.simulation_id] = split_name

    if len(assignments) != len(scenarios):
        raise RuntimeError("Every simulation must be assigned to exactly one split.")

    return assignments


def _split_counts(total: int, ratios: SplitRatios) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(
            "At least three simulations per class are required for splitting."
        )

    validation = max(1, round(total * ratios.validation))
    test = max(1, round(total * ratios.test))
    train = total - validation - test

    if train < 1:
        train = 1
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1

    if train + validation + test != total:
        raise RuntimeError("Internal split allocation error.")

    return train, validation, test

"""Experiment-plan validation and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True, frozen=True)
class FloatRange:
    """Inclusive floating-point sampling range."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not isfinite(self.minimum) or not isfinite(self.maximum):
            raise ValueError("Float range boundaries must be finite.")
        if self.minimum > self.maximum:
            raise ValueError("Float range minimum must not exceed maximum.")


@dataclass(slots=True, frozen=True)
class IntRange:
    """Inclusive integer sampling range."""

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum <= 0:
            raise ValueError("Integer range minimum must be positive.")
        if self.minimum > self.maximum:
            raise ValueError("Integer range minimum must not exceed maximum.")


@dataclass(slots=True, frozen=True)
class NetworkProfile:
    """Bounds for one network operating regime."""

    name: str
    latency_ms: FloatRange
    latency_std_ms: FloatRange
    packet_loss: FloatRange

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Network profile name must not be empty.")
        if self.latency_ms.minimum < 0.0:
            raise ValueError("Network latency must be non-negative.")
        if self.latency_std_ms.minimum < 0.0:
            raise ValueError("Network latency deviation must be non-negative.")
        if not 0.0 <= self.packet_loss.minimum <= self.packet_loss.maximum <= 1.0:
            raise ValueError("Packet-loss range must lie within [0, 1].")


@dataclass(slots=True, frozen=True)
class SplitRatios:
    """Simulation-level train, validation, and test ratios."""

    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Split ratios must be finite and positive.")
        if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Split ratios must sum to exactly 1.0.")


@dataclass(slots=True, frozen=True)
class ExperimentPlan:
    """Validated configuration for reproducible dataset generation."""

    base_seed: int
    scenarios_per_class: int
    duration_blocks: int
    block_interval_seconds: float
    normal_window_length: IntRange
    double_spend_window_length: IntRange
    selfish_window_length: IntRange
    majority_window_length: IntRange
    attack_start_margin: int
    node_count: IntRange
    miner_count: IntRange
    double_spend_q: FloatRange
    selfish_q: FloatRange
    selfish_gamma: FloatRange
    majority_q: FloatRange
    confirmation_targets: tuple[int, ...]
    miner_concentration_alphas: tuple[tuple[str, float], ...]
    network_profiles: tuple[NetworkProfile, ...]
    splits: SplitRatios
    output_root: str

    def __post_init__(self) -> None:
        if self.base_seed < 0:
            raise ValueError("Base seed must be non-negative.")
        if self.scenarios_per_class < 3:
            raise ValueError("At least three scenarios per class are required.")
        if self.duration_blocks <= 0:
            raise ValueError("Simulation duration must be positive.")
        if not isfinite(self.block_interval_seconds) or self.block_interval_seconds <= 0.0:
            raise ValueError("Block interval must be finite and positive.")
        if self.attack_start_margin < 0:
            raise ValueError("Attack start margin must be non-negative.")
        window_ranges = (
            self.normal_window_length,
            self.double_spend_window_length,
            self.selfish_window_length,
            self.majority_window_length,
        )
        if any(
            window.maximum + 2 * self.attack_start_margin > self.duration_blocks
            for window in window_ranges
        ):
            raise ValueError("Attack windows and margins must fit within the simulation.")
        if not 0.0 < self.double_spend_q.minimum <= self.double_spend_q.maximum < 0.5:
            raise ValueError("Double-spend q range must lie within (0, 0.5).")
        if not 0.0 < self.selfish_q.minimum <= self.selfish_q.maximum < 0.5:
            raise ValueError("Selfish-mining q range must lie within (0, 0.5).")
        if not 0.0 <= self.selfish_gamma.minimum <= self.selfish_gamma.maximum <= 1.0:
            raise ValueError("Selfish-mining gamma range must lie within [0, 1].")
        if not 0.5 < self.majority_q.minimum <= self.majority_q.maximum <= 1.0:
            raise ValueError("Majority q range must lie within (0.5, 1].")
        if not self.confirmation_targets or any(value <= 0 for value in self.confirmation_targets):
            raise ValueError("Confirmation targets must contain positive integers.")
        if not self.miner_concentration_alphas:
            raise ValueError("At least one miner-concentration regime is required.")
        names = [name for name, _ in self.miner_concentration_alphas]
        if len(set(names)) != len(names):
            raise ValueError("Miner-concentration regime names must be unique.")
        if any(not name or not isfinite(alpha) or alpha <= 0.0 for name, alpha in self.miner_concentration_alphas):
            raise ValueError("Miner-concentration alphas must be finite and positive.")
        if not self.network_profiles:
            raise ValueError("At least one network profile is required.")
        profile_names = [profile.name for profile in self.network_profiles]
        if len(set(profile_names)) != len(profile_names):
            raise ValueError("Network profile names must be unique.")
        if not self.output_root:
            raise ValueError("Output root must not be empty.")


def _float_range(mapping: dict[str, Any], key: str) -> FloatRange:
    values = mapping[key]
    return FloatRange(float(values[0]), float(values[1]))


def _int_range(mapping: dict[str, Any], key: str) -> IntRange:
    values = mapping[key]
    return IntRange(int(values[0]), int(values[1]))


def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    """Load and validate one project-root-relative YAML experiment plan."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    experiment = data["experiment"]
    scenarios = data["scenarios"]
    attacks = data["attacks"]
    network = data["network_profiles"]
    mining = data["mining"]
    split_data = data["splits"]

    profiles = tuple(
        NetworkProfile(
            name=name,
            latency_ms=_float_range(values, "base_latency_ms"),
            latency_std_ms=_float_range(values, "latency_std_ms"),
            packet_loss=_float_range(values, "packet_loss_probability"),
        )
        for name, values in network.items()
    )

    concentration = tuple(
        (name, float(alpha))
        for name, alpha in mining["concentration_alpha"].items()
    )

    return ExperimentPlan(
        base_seed=int(experiment["base_seed"]),
        scenarios_per_class=int(experiment["scenarios_per_class"]),
        duration_blocks=int(experiment["duration_blocks"]),
        block_interval_seconds=float(experiment["block_interval_seconds"]),
        normal_window_length=_int_range(
            scenarios["attack_window_length_blocks"],
            "normal",
        ),
        double_spend_window_length=_int_range(
            scenarios["attack_window_length_blocks"],
            "double_spend",
        ),
        selfish_window_length=_int_range(
            scenarios["attack_window_length_blocks"],
            "selfish_mining",
        ),
        majority_window_length=_int_range(
            scenarios["attack_window_length_blocks"],
            "majority_attack",
        ),
        attack_start_margin=int(scenarios["attack_start_margin_blocks"]),
        node_count=_int_range(scenarios, "num_nodes"),
        miner_count=_int_range(scenarios, "num_miners"),
        double_spend_q=_float_range(attacks["double_spend"], "attacker_hash_share"),
        selfish_q=_float_range(attacks["selfish_mining"], "attacker_hash_share"),
        selfish_gamma=_float_range(attacks["selfish_mining"], "gamma"),
        majority_q=_float_range(attacks["majority_attack"], "attacker_hash_share"),
        confirmation_targets=tuple(int(value) for value in attacks["double_spend"]["target_confirmations"]),
        miner_concentration_alphas=concentration,
        network_profiles=profiles,
        splits=SplitRatios(
            train=float(split_data["train"]),
            validation=float(split_data["validation"]),
            test=float(split_data["test"]),
        ),
        output_root=str(data["output"]["root"]),
    )

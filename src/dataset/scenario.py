"""Deterministic generation of heterogeneous simulation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from random import Random

from src.dataset.config import ExperimentPlan
from src.simulator.attacks import AttackType


@dataclass(slots=True, frozen=True)
class ScenarioSpec:
    """Store every sampled parameter required to reproduce one simulation."""

    simulation_id: str
    scenario_id: str
    seed: int
    attack_type: AttackType
    duration_blocks: int
    block_interval_seconds: float
    attack_start_step: int
    attack_end_step: int
    num_nodes: int
    num_miners: int
    public_miner_hash_powers: tuple[float, ...]
    miner_concentration: str
    network_profile: str
    base_latency_ms: float
    latency_std_ms: float
    packet_loss_probability: float
    double_spend_q: float
    double_spend_target_confirmations: int
    selfish_q: float
    gamma: float
    majority_q: float

    def __post_init__(self) -> None:
        if not self.simulation_id or not self.scenario_id:
            raise ValueError("Scenario identifiers must not be empty.")
        if self.seed < 0:
            raise ValueError("Scenario seed must be non-negative.")
        if not 1 <= self.attack_start_step <= self.attack_end_step <= self.duration_blocks:
            raise ValueError("Attack window must lie within the simulation.")
        if self.num_nodes <= 0 or self.num_miners <= 0:
            raise ValueError("Node and miner counts must be positive.")
        if len(self.public_miner_hash_powers) != self.num_miners:
            raise ValueError("Miner hash-power count must equal num_miners.")
        if any(value <= 0.0 for value in self.public_miner_hash_powers):
            raise ValueError("Every public miner must have positive hash power.")
        if not isclose(sum(self.public_miner_hash_powers), 1.0, abs_tol=1e-12):
            raise ValueError("Normalized public miner hash powers must sum to 1.0.")

    @property
    def attacker_hash_share(self) -> float | None:
        """Return the attack-specific q value, or None for normal scenarios."""

        if self.attack_type is AttackType.DOUBLE_SPEND:
            return self.double_spend_q
        if self.attack_type is AttackType.SELFISH_MINING:
            return self.selfish_q
        if self.attack_type is AttackType.MAJORITY_ATTACK:
            return self.majority_q
        return None


class ScenarioGenerator:
    """Generate balanced, randomized, reproducible simulation scenarios."""

    _ATTACK_TYPES = (
        AttackType.NORMAL,
        AttackType.DOUBLE_SPEND,
        AttackType.SELFISH_MINING,
        AttackType.MAJORITY_ATTACK,
    )

    def __init__(self, plan: ExperimentPlan) -> None:
        self._plan = plan

    def generate(self) -> tuple[ScenarioSpec, ...]:
        """Generate a balanced scenario collection in stable order."""

        scenarios: list[ScenarioSpec] = []
        ordinal = 0

        for attack_type in self._ATTACK_TYPES:
            for class_index in range(self._plan.scenarios_per_class):
                seed = self._plan.base_seed + ordinal * 1009
                rng = Random(seed)
                scenarios.append(
                    self._sample_scenario(
                        attack_type=attack_type,
                        class_index=class_index,
                        ordinal=ordinal,
                        seed=seed,
                        rng=rng,
                    )
                )
                ordinal += 1

        return tuple(scenarios)

    def _sample_scenario(
        self,
        attack_type: AttackType,
        class_index: int,
        ordinal: int,
        seed: int,
        rng: Random,
    ) -> ScenarioSpec:
        window_range = self._window_range(attack_type)
        window_length = rng.randint(
            window_range.minimum,
            window_range.maximum,
        )
        minimum_start = self._plan.attack_start_margin + 1
        maximum_start = (
            self._plan.duration_blocks
            - self._plan.attack_start_margin
            - window_length
            + 1
        )
        attack_start = rng.randint(minimum_start, maximum_start)
        attack_end = attack_start + window_length - 1

        num_nodes = rng.randint(
            self._plan.node_count.minimum,
            self._plan.node_count.maximum,
        )
        num_miners = rng.randint(
            self._plan.miner_count.minimum,
            self._plan.miner_count.maximum,
        )

        concentration_name, alpha = rng.choice(
            self._plan.miner_concentration_alphas
        )
        miner_hash_powers = self._sample_miner_hash_powers(
            rng,
            num_miners,
            alpha,
        )

        profile = rng.choice(self._plan.network_profiles)

        return ScenarioSpec(
            simulation_id=f"sim_{ordinal:06d}",
            scenario_id=f"{attack_type.name.lower()}_{class_index:04d}",
            seed=seed,
            attack_type=attack_type,
            duration_blocks=self._plan.duration_blocks,
            block_interval_seconds=self._plan.block_interval_seconds,
            attack_start_step=attack_start,
            attack_end_step=attack_end,
            num_nodes=num_nodes,
            num_miners=num_miners,
            public_miner_hash_powers=miner_hash_powers,
            miner_concentration=concentration_name,
            network_profile=profile.name,
            base_latency_ms=rng.uniform(
                profile.latency_ms.minimum,
                profile.latency_ms.maximum,
            ),
            latency_std_ms=rng.uniform(
                profile.latency_std_ms.minimum,
                profile.latency_std_ms.maximum,
            ),
            packet_loss_probability=rng.uniform(
                profile.packet_loss.minimum,
                profile.packet_loss.maximum,
            ),
            double_spend_q=rng.uniform(
                self._plan.double_spend_q.minimum,
                self._plan.double_spend_q.maximum,
            ),
            double_spend_target_confirmations=rng.choice(
                self._plan.confirmation_targets
            ),
            selfish_q=rng.uniform(
                self._plan.selfish_q.minimum,
                self._plan.selfish_q.maximum,
            ),
            gamma=rng.uniform(
                self._plan.selfish_gamma.minimum,
                self._plan.selfish_gamma.maximum,
            ),
            majority_q=rng.uniform(
                self._plan.majority_q.minimum,
                self._plan.majority_q.maximum,
            ),
        )


    def _window_range(self, attack_type: AttackType):
        if attack_type is AttackType.NORMAL:
            return self._plan.normal_window_length
        if attack_type is AttackType.DOUBLE_SPEND:
            return self._plan.double_spend_window_length
        if attack_type is AttackType.SELFISH_MINING:
            return self._plan.selfish_window_length
        if attack_type is AttackType.MAJORITY_ATTACK:
            return self._plan.majority_window_length
        raise ValueError(f"Unsupported attack type: {attack_type!r}.")

    @staticmethod
    def _sample_miner_hash_powers(
        rng: Random,
        count: int,
        alpha: float,
    ) -> tuple[float, ...]:
        weights = [rng.gammavariate(alpha, 1.0) for _ in range(count)]
        total = sum(weights)
        normalized = [value / total for value in weights]
        normalized[-1] += 1.0 - sum(normalized)
        return tuple(normalized)

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from random import Random

from src.simulator.attacks import (
    AttackConfiguration,
    AttackType,
)
from src.simulator.blockchain import Block
from src.simulator.chain_observer import (
    CanonicalChainObserver,
    ChainObservation,
)
from src.simulator.double_spend_scenario import (
    DoubleSpendMiningSide,
    DoubleSpendScenario,
    DoubleSpendScenarioState,
    DoubleSpendScenarioStatus,
)
from src.simulator.majority_attack import (
    MajorityAttack,
    MajorityAttackStatus,
    MajorityMiningSide,
)
from src.simulator.mining import Miner, MiningPool
from src.simulator.network import Network, Node
from src.simulator.observable_features import (
    ObservableFeatureExtractor,
    ObservableFeatureVector,
)
from src.simulator.selfish_mining import (
    SelfishMiningAttack,
    SelfishMiningSide,
    SelfishMiningStatus,
)
from src.simulator.telemetry import (
    TelemetryCollector,
    TelemetrySnapshot,
)


@dataclass(slots=True, frozen=True)
class SimulationParameters:
    """Store validated parameters required by the simulation engine."""

    duration_blocks: int
    block_interval_seconds: float
    selfish_attacker_hash_share: float
    seed: int

    def __post_init__(self) -> None:
        if self.duration_blocks <= 0:
            raise ValueError(
                "Simulation duration must be positive."
            )

        if (not isfinite(self.block_interval_seconds)
            or self.block_interval_seconds <= 0.0
        ):
            raise ValueError(
                "Mean block interval must be a finite positive value."
            )

        if not 0.0 < self.selfish_attacker_hash_share < 0.5:
            raise ValueError(
                "Selfish-mining attacker hash share must be in the interval "
                "(0, 0.5)."
            )

        if self.seed < 0:
            raise ValueError(
                "Simulation random seed must be non-negative."
            )


@dataclass(slots=True, frozen=True)
class SimulationStep:
    """Store one synchronized observable simulation sample."""

    step: int
    timestamp: float
    canonical_height: int
    attack_label: AttackType
    telemetry: TelemetrySnapshot
    chain_observation: ChainObservation
    features: ObservableFeatureVector

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError(
                "Simulation step must be positive."
            )

        if self.timestamp < 0.0:
            raise ValueError(
                "Simulation timestamp must be non-negative."
            )

        if self.canonical_height < 0:
            raise ValueError(
                "Canonical chain height must be non-negative."
            )

        if self.telemetry.timestamp != self.timestamp:
            raise ValueError(
                "Telemetry timestamp must equal simulation timestamp."
            )

        if self.features.timestamp != self.timestamp:
            raise ValueError(
                "Feature timestamp must equal simulation timestamp."
            )

        if self.chain_observation.height != self.canonical_height:
            raise ValueError(
                "Chain observation height must equal canonical height."
            )

        if self.telemetry.reference_height != self.canonical_height:
            raise ValueError(
                "Telemetry reference height must equal canonical height."
            )

        if self.features.canonical_height != self.canonical_height:
            raise ValueError(
                "Feature canonical height must equal simulation height."
            )

        if self.features.reference_height != self.canonical_height:
            raise ValueError(
                "Feature reference height must equal simulation height."
            )


class SimulationEngine:
    """Coordinate reproducible blockchain simulation and feature extraction."""

    def __init__(
        self,
        network: Network,
        attack_configuration: AttackConfiguration,
        parameters: SimulationParameters,
        public_miners: Sequence[Miner],
    ) -> None:
        if not network.nodes:
            raise ValueError(
                "Simulation network must contain at least one node."
            )

        if not public_miners:
            raise ValueError(
                "Simulation must contain at least one public miner."
            )

        self._network = network
        self._attack_configuration = attack_configuration
        self._parameters = parameters

        self._rng = Random(parameters.seed)

        self._clock_rng = Random(
            self._derived_seed(4)
        )
        self._double_spend_rng = Random(
            self._derived_seed(5)
        )
        self._selfish_mining_rng = Random(
            self._derived_seed(6)
        )
        self._majority_attack_rng = Random(
            self._derived_seed(7)
        )

        self._mining_pool = MiningPool(
            public_miners,
            self._derived_seed(3),
        )

        self._steps: list[SimulationStep] = []

        self._double_spend_scenario: DoubleSpendScenario | None = None
        self._selfish_attack: SelfishMiningAttack | None = None
        self._majority_attack: MajorityAttack | None = None

        self._chain_observer = CanonicalChainObserver()
        self._previous_attack_label = AttackType.NORMAL

    @property
    def steps(self) -> tuple[SimulationStep, ...]:
        """Return completed simulation steps as an immutable sequence."""

        return tuple(self._steps)

    @property
    def public_miners(self) -> tuple[Miner, ...]:
        """Return the configured public miners."""

        return self._mining_pool.miners

    @property
    def double_spend_status(self) -> DoubleSpendScenarioStatus | None:
        """Return internal double-spend outcome metadata when available."""

        if self._double_spend_scenario is None:
            return None

        return self._double_spend_scenario.status()

    @property
    def selfish_mining_status(self) -> SelfishMiningStatus | None:
        """Return internal selfish-mining outcome metadata when available."""

        if self._selfish_attack is None:
            return None

        return self._selfish_attack.status()

    @property
    def majority_attack_status(self) -> MajorityAttackStatus | None:
        """Return internal majority-attack outcome metadata when available."""

        if self._majority_attack is None:
            return None

        return self._majority_attack.status()

    def run(self) -> tuple[SimulationStep, ...]:
        """Run the deterministic simulation exactly once."""

        if self._steps:
            raise RuntimeError(
                "Simulation engine can only be run once."
            )

        reference_node = self._reference_node()

        self._chain_observer.reset()
        self._chain_observer.observe(reference_node)

        timestamp = 0.0

        for step in range(
                1,
                self._parameters.duration_blocks + 1,
        ):
            timestamp += self._sample_next_block_interval()
            self._advance_network(
                timestamp
            )

            attack_label = (
                self._attack_configuration.label_at(step)
            )

            if (
                self._previous_attack_label is AttackType.DOUBLE_SPEND
                and attack_label is not AttackType.DOUBLE_SPEND
            ):
                self._finalize_double_spend_attempt()

            if (
                self._previous_attack_label is AttackType.SELFISH_MINING
                and attack_label is not AttackType.SELFISH_MINING
            ):
                self._finalize_selfish_mining_attempt()

            if (
                self._previous_attack_label is AttackType.MAJORITY_ATTACK
                and attack_label is not AttackType.MAJORITY_ATTACK
            ):
                self._finalize_majority_attack_attempt()

            self._execute_step(
                step=step,
                timestamp=timestamp,
                attack_label=attack_label,
            )

            telemetry = TelemetryCollector.collect(
                network=self._network,
                timestamp=timestamp,
            )

            chain_observation = (
                self._chain_observer.observe(
                    reference_node
                )
            )

            features = ObservableFeatureExtractor.extract(
                telemetry=telemetry,
                chain_observation=chain_observation,
            )

            self._steps.append(
                SimulationStep(
                    step=step,
                    timestamp=timestamp,
                    canonical_height=reference_node.height,
                    attack_label=attack_label,
                    telemetry=telemetry,
                    chain_observation=chain_observation,
                    features=features,
                )
            )

            self._previous_attack_label = attack_label

        if self._previous_attack_label is AttackType.DOUBLE_SPEND:
            self._finalize_double_spend_attempt()
        elif self._previous_attack_label is AttackType.SELFISH_MINING:
            self._finalize_selfish_mining_attempt()
        elif self._previous_attack_label is AttackType.MAJORITY_ATTACK:
            self._finalize_majority_attack_attempt()

        return self.steps

    def _advance_network(
        self,
        timestamp: float,
    ) -> None:
        """Process due deliveries and reconcile previously observed chains."""

        self._network.deliver_until(
            timestamp
        )

        self._network.reconcile_from(
            self._reference_node().node_id
        )

    def _execute_step(
        self,
        step: int,
        timestamp: float,
        attack_label: AttackType,
    ) -> None:
        """Execute exactly one configured simulation step."""

        if attack_label is AttackType.NORMAL:
            self._mine_honest_block(timestamp)
            return

        if attack_label is AttackType.DOUBLE_SPEND:
            self._execute_double_spend_step(timestamp)
            return

        if attack_label is AttackType.SELFISH_MINING:
            self._execute_selfish_mining_step(timestamp)
            return

        if attack_label is AttackType.MAJORITY_ATTACK:
            self._execute_majority_attack_step(timestamp)
            return

        raise ValueError(
            f"Unsupported attack type at simulation step {step}: "
            f"{attack_label!r}."
        )

    def _mine_honest_block(
        self,
        timestamp: float,
    ) -> Block:
        """Mine and propagate one public canonical block."""

        reference_node = self._reference_node()

        block = self._mining_pool.create_block(
            reference_node,
            timestamp,
        )

        self._network.broadcast_block(
            source_node_id=reference_node.node_id,
            block=block,
            sent_at=timestamp,
        )

        self._network.deliver_until(
            timestamp
        )

        return block

    def _execute_double_spend_step(
        self,
        timestamp: float,
    ) -> None:
        """Execute one stochastic double-spend PoW mining event."""

        scenario = self._double_spend_scenario

        if scenario is None:
            scenario = DoubleSpendScenario(
                victim_node=self._reference_node(),
                parameters=(
                    self._attack_configuration.double_spend
                ),
                attacker_id="double_spend_attacker",
                sender="attacker_wallet",
                victim_receiver="merchant_wallet",
                attacker_receiver="attacker_recovery_wallet",
                spent_output_id="double_spend_utxo",
            )
            scenario.start(timestamp)
            self._double_spend_scenario = scenario

        if scenario.state in {
            DoubleSpendScenarioState.SUCCEEDED,
            DoubleSpendScenarioState.FAILED,
        }:
            self._mine_honest_block(timestamp)
            return

        mining_side = self._select_double_spend_mining_side()
        public_miner_id: str | None = None

        if mining_side is DoubleSpendMiningSide.PUBLIC:
            public_miner_id = self._mining_pool.select_miner().miner_id

        block = scenario.process_mining_event(
            timestamp=timestamp,
            mining_side=mining_side,
            public_miner_id=public_miner_id,
        )

        if mining_side is DoubleSpendMiningSide.PUBLIC:
            reference_node = self._reference_node()
            self._network.broadcast_block(
                source_node_id=reference_node.node_id,
                block=block,
                sent_at=timestamp,
            )
            self._network.deliver_until(timestamp)

        if scenario.can_publish_successfully():
            scenario.publish()

    def _select_double_spend_mining_side(
        self,
    ) -> DoubleSpendMiningSide:
        """Select the winner of one double-spend PoW event."""

        attacker_hash_share = (
            self._attack_configuration.double_spend.attacker_hash_share
        )

        if self._double_spend_rng.random() < attacker_hash_share:
            return DoubleSpendMiningSide.ATTACKER

        return DoubleSpendMiningSide.PUBLIC

    def _finalize_double_spend_attempt(self) -> None:
        """Finalize an unfinished double-spend attempt as failed."""

        scenario = self._double_spend_scenario

        if scenario is None:
            return

        if scenario.state in {
            DoubleSpendScenarioState.SUCCEEDED,
            DoubleSpendScenarioState.FAILED,
        }:
            return

        scenario.fail()

    def _execute_selfish_mining_step(
        self,
        timestamp: float,
    ) -> None:
        """Execute one stochastic selfish-mining PoW event."""

        selfish_attack = self._selfish_attack

        if selfish_attack is None:
            selfish_attack = SelfishMiningAttack(
                victim_node=self._reference_node(),
                parameters=self._attack_configuration.selfish_mining,
                attacker_id="selfish_attacker",
                seed=self._derived_seed(1),
            )
            selfish_attack.start()
            self._selfish_attack = selfish_attack

        mining_side = self._select_selfish_mining_side()
        public_miner_id: str | None = None
        if mining_side is SelfishMiningSide.PUBLIC:
            public_miner_id = self._mining_pool.select_miner().miner_id

        block = selfish_attack.process_mining_event(
            timestamp=timestamp,
            mining_side=mining_side,
            public_miner_id=public_miner_id,
        )

        if mining_side is SelfishMiningSide.PUBLIC:
            self._broadcast_current_reference_tip(timestamp, block)

    def _select_selfish_mining_side(self) -> SelfishMiningSide:
        """Select the winner of one selfish-mining PoW event."""

        q = self._attack_configuration.selfish_mining.attacker_hash_share
        if self._selfish_mining_rng.random() < q:
            return SelfishMiningSide.ATTACKER
        return SelfishMiningSide.PUBLIC

    def _finalize_selfish_mining_attempt(self) -> None:
        """Finish an active selfish-mining window without forced success."""

        if self._selfish_attack is not None:
            self._selfish_attack.finish()

    def _execute_majority_attack_step(
        self,
        timestamp: float,
    ) -> None:
        """Execute one stochastic majority-attack PoW event."""

        majority_attack = self._majority_attack

        if majority_attack is None:
            majority_attack = MajorityAttack(
                victim_node=self._reference_node(),
                parameters=self._attack_configuration.majority_attack,
                attacker_id="majority_attacker",
            )
            majority_attack.start()
            self._majority_attack = majority_attack

        mining_side = self._select_majority_mining_side()
        public_miner_id: str | None = None
        if mining_side is MajorityMiningSide.PUBLIC:
            public_miner_id = self._mining_pool.select_miner().miner_id

        block = majority_attack.process_mining_event(
            timestamp=timestamp,
            mining_side=mining_side,
            public_miner_id=public_miner_id,
        )

        if mining_side is MajorityMiningSide.PUBLIC:
            self._broadcast_current_reference_tip(timestamp, block)

    def _select_majority_mining_side(self) -> MajorityMiningSide:
        """Select the winner of one majority-attack PoW event."""

        q = self._attack_configuration.majority_attack.attacker_hash_share
        if self._majority_attack_rng.random() < q:
            return MajorityMiningSide.ATTACKER
        return MajorityMiningSide.PUBLIC

    def _finalize_majority_attack_attempt(self) -> None:
        """Finish an active majority-attack window without forced success."""

        if self._majority_attack is not None:
            self._majority_attack.finish()

    def _broadcast_current_reference_tip(
        self,
        timestamp: float,
        event_block: Block,
    ) -> None:
        """Broadcast a public event block when it remains canonical."""

        reference_node = self._reference_node()
        if reference_node.tip_hash != event_block.block_hash:
            return
        self._network.broadcast_block(
            source_node_id=reference_node.node_id,
            block=event_block,
            sent_at=timestamp,
        )
        self._network.deliver_until(timestamp)

    def _reference_node(self) -> Node:
        """Return the node used for canonical-chain observation."""

        return self._network.nodes[0]

    def _sample_next_block_interval(
            self,
    ) -> float:
        """Sample one strictly positive exponential PoW block interval."""

        rate = (
                1.0
                / self._parameters.block_interval_seconds
        )

        interval = self._clock_rng.expovariate(
            rate
        )

        while interval <= 0.0:
            interval = self._clock_rng.expovariate(
                rate
            )

        return interval

    def _derived_seed(
        self,
        offset: int,
    ) -> int:
        """Derive deterministic component seeds from the simulation seed."""

        return self._parameters.seed + offset
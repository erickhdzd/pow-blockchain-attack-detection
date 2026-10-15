from __future__ import annotations

from collections import Counter
from itertools import pairwise
from math import isfinite
from pathlib import Path

import pytest

from src.simulator.attacks import (
    AttackConfiguration,
    AttackType,
    AttackWindow,
    DoubleSpendParameters,
    MajorityAttackParameters,
    SelfishMiningParameters,
)
from src.simulator.double_spend_scenario import (
    DoubleSpendMiningSide,
    DoubleSpendScenario,
    DoubleSpendScenarioState,
)
from src.simulator.engine import SimulationEngine, SimulationParameters
from src.simulator.mining import Miner
from src.simulator.network import Network, NetworkParameters, Node


def public_miners() -> tuple[Miner, ...]:
    return (
        Miner("miner_a", 0.40),
        Miner("miner_b", 0.25),
        Miner("miner_c", 0.20),
        Miner("miner_d", 0.10),
        Miner("miner_e", 0.05),
    )


def attack_configuration(
    *,
    attack_type: AttackType = AttackType.DOUBLE_SPEND,
    start: int = 3,
    end: int = 30,
    q: float = 0.30,
    confirmations: int = 2,
) -> AttackConfiguration:
    return AttackConfiguration(
        attack_type=attack_type,
        window=AttackWindow(start, end),
        double_spend=DoubleSpendParameters(
            transaction_value=1.0,
            target_confirmations=confirmations,
            attacker_hash_share=q,
        ),
        selfish_mining=SelfishMiningParameters(gamma=0.50),
        majority_attack=MajorityAttackParameters(attacker_hash_share=0.51),
    )


def make_network(
    *,
    seed: int = 42,
    node_count: int = 1,
    latency_ms: float = 0.0,
    latency_std_ms: float = 0.0,
    packet_loss: float = 0.0,
) -> Network:
    return Network(
        nodes=[Node(f"node_{index}") for index in range(node_count)],
        parameters=NetworkParameters(
            base_latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            packet_loss_probability=packet_loss,
        ),
        seed=seed,
    )


def make_engine(
    *,
    seed: int = 42,
    q: float = 0.30,
    confirmations: int = 2,
    blocks: int = 40,
    start: int = 3,
    end: int = 30,
    node_count: int = 1,
    latency_ms: float = 0.0,
    latency_std_ms: float = 0.0,
    packet_loss: float = 0.0,
    mean_interval: float = 600.0,
    attack_type: AttackType = AttackType.DOUBLE_SPEND,
) -> tuple[Network, SimulationEngine]:
    simulation_network = make_network(
        seed=seed,
        node_count=node_count,
        latency_ms=latency_ms,
        latency_std_ms=latency_std_ms,
        packet_loss=packet_loss,
    )
    simulation_engine = SimulationEngine(
        network=simulation_network,
        attack_configuration=attack_configuration(
            attack_type=attack_type,
            start=start,
            end=end,
            q=q,
            confirmations=confirmations,
        ),
        parameters=SimulationParameters(
            duration_blocks=blocks,
            block_interval_seconds=mean_interval,
            selfish_attacker_hash_share=0.30,
            seed=seed,
        ),
        public_miners=public_miners(),
    )
    return simulation_network, simulation_engine


def make_scenario(
    *,
    q: float = 0.30,
    confirmations: int = 2,
) -> tuple[Node, DoubleSpendScenario]:
    node = Node("victim")
    scenario = DoubleSpendScenario(
        victim_node=node,
        parameters=DoubleSpendParameters(
            transaction_value=1.0,
            target_confirmations=confirmations,
            attacker_hash_share=q,
        ),
        attacker_id="double_spend_attacker",
        sender="attacker_wallet",
        victim_receiver="merchant_wallet",
        attacker_receiver="attacker_recovery_wallet",
        spent_output_id="double_spend_utxo",
    )
    return node, scenario


def run_signature(seed: int, q: float = 0.30) -> tuple[object, ...]:
    network, engine = make_engine(seed=seed, q=q)
    steps = engine.run()
    reference = network.nodes[0]
    status = engine.double_spend_status
    return (
        tuple(block.block_hash for block in reference.blockchain.chain),
        tuple(block.miner_id for block in reference.blockchain.chain),
        tuple(step.timestamp for step in steps),
        tuple(int(step.attack_label) for step in steps),
        tuple(step.features.as_numeric_tuple() for step in steps),
        status,
    )


def outcome(seed: int, q: float) -> bool:
    _, engine = make_engine(seed=seed, q=q)
    engine.run()
    status = engine.double_spend_status
    assert status is not None
    return status.successful


def test_01_parameter_default_and_explicit_q() -> None:
    default = DoubleSpendParameters(1.0, 6)
    explicit = DoubleSpendParameters(1.0, 6, 0.45)
    assert default.attacker_hash_share == 0.30
    assert explicit.attacker_hash_share == 0.45


@pytest.mark.parametrize(
    "invalid_q",
    (0.0, -0.1, 0.5, 0.51, 1.0, float("inf"), float("-inf"), float("nan")),
)
def test_02_invalid_q_rejected(invalid_q: float) -> None:
    with pytest.raises(ValueError):
        DoubleSpendParameters(1.0, 6, invalid_q)


def test_03_config_contains_double_spend_q() -> None:
    config_text = Path("configs/config.yaml").read_text(encoding="utf-8")
    assert "double_spend:" in config_text
    assert "attacker_hash_share: 0.30" in config_text


def test_04_mining_side_is_explicit() -> None:
    assert DoubleSpendMiningSide.PUBLIC is not DoubleSpendMiningSide.ATTACKER


def test_05_scenario_starts_without_mining() -> None:
    node, scenario = make_scenario()
    scenario.start(1.0)
    assert node.height == 0
    assert scenario.state is DoubleSpendScenarioState.PAYMENT_PENDING
    assert scenario.status().private_branch_length == 0


def test_06_public_event_mines_exactly_one_public_block() -> None:
    node, scenario = make_scenario()
    scenario.start(1.0)
    block = scenario.process_mining_event(
        timestamp=2.0,
        mining_side=DoubleSpendMiningSide.PUBLIC,
        public_miner_id="miner_a",
    )
    assert node.height == 1
    assert block.miner_id == "miner_a"
    assert scenario.status().private_branch_length == 0
    assert node.blockchain.validate()


def test_07_attacker_event_mines_exactly_one_private_block() -> None:
    node, scenario = make_scenario()
    scenario.start(1.0)
    block = scenario.process_mining_event(
        timestamp=2.0,
        mining_side=DoubleSpendMiningSide.ATTACKER,
    )
    assert node.height == 0
    assert block.miner_id == "double_spend_attacker"
    assert scenario.status().private_branch_length == 1
    assert node.blockchain.validate()


def test_08_public_event_requires_real_miner_id() -> None:
    _, scenario = make_scenario()
    scenario.start(1.0)
    with pytest.raises(ValueError):
        scenario.process_mining_event(
            timestamp=2.0,
            mining_side=DoubleSpendMiningSide.PUBLIC,
        )


def test_09_failed_scenario_is_terminal_and_idempotent() -> None:
    _, scenario = make_scenario()
    scenario.start(1.0)
    scenario.fail()
    scenario.fail()
    assert scenario.state is DoubleSpendScenarioState.FAILED
    with pytest.raises(RuntimeError):
        scenario.process_mining_event(
            timestamp=2.0,
            mining_side=DoubleSpendMiningSide.ATTACKER,
        )


def test_10_success_requires_confirmation_and_longer_private_branch() -> None:
    _, scenario = make_scenario(confirmations=1)
    scenario.start(1.0)
    scenario.process_mining_event(
        timestamp=2.0,
        mining_side=DoubleSpendMiningSide.PUBLIC,
        public_miner_id="miner_a",
    )
    scenario.process_mining_event(3.0, DoubleSpendMiningSide.ATTACKER)
    assert not scenario.can_publish_successfully()
    scenario.process_mining_event(4.0, DoubleSpendMiningSide.ATTACKER)
    assert scenario.can_publish_successfully()
    result = scenario.publish()
    assert result.adopted
    assert scenario.state is DoubleSpendScenarioState.SUCCEEDED
    assert scenario.status().successful


def test_11_successful_scenario_cannot_be_failed_afterward() -> None:
    _, scenario = make_scenario(confirmations=1)
    scenario.start(1.0)
    scenario.process_mining_event(2.0, DoubleSpendMiningSide.PUBLIC, "miner_a")
    scenario.process_mining_event(3.0, DoubleSpendMiningSide.ATTACKER)
    scenario.process_mining_event(4.0, DoubleSpendMiningSide.ATTACKER)
    scenario.publish()
    with pytest.raises(RuntimeError):
        scenario.fail()


def test_12_engine_runs_one_stochastic_event_per_step() -> None:
    network, engine = make_engine(seed=100, q=0.10, blocks=40)
    steps = engine.run()
    status = engine.double_spend_status
    assert len(steps) == 40
    assert status is not None
    assert network.nodes[0].blockchain.validate()
    assert status.state in {
        DoubleSpendScenarioState.SUCCEEDED,
        DoubleSpendScenarioState.FAILED,
    }


def test_13_attack_window_labels_are_unchanged() -> None:
    _, engine = make_engine(seed=42, start=5, end=12, blocks=20)
    steps = engine.run()
    labels = [step.attack_label for step in steps]
    assert labels[:4] == [AttackType.NORMAL] * 4
    assert labels[4:12] == [AttackType.DOUBLE_SPEND] * 8
    assert labels[12:] == [AttackType.NORMAL] * 8


def test_14_window_expiration_marks_unfinished_attack_failed() -> None:
    _, engine = make_engine(
        seed=42,
        q=0.10,
        confirmations=6,
        blocks=10,
        start=2,
        end=3,
    )
    engine.run()
    status = engine.double_spend_status
    assert status is not None
    assert status.state is DoubleSpendScenarioState.FAILED
    assert not status.successful


def test_15_simulation_end_marks_unfinished_attack_failed() -> None:
    _, engine = make_engine(
        seed=42,
        q=0.10,
        confirmations=6,
        blocks=3,
        start=1,
        end=3,
    )
    engine.run()
    status = engine.double_spend_status
    assert status is not None
    assert status.state is DoubleSpendScenarioState.FAILED


def test_16_known_seed_produces_real_success() -> None:
    _, engine = make_engine(seed=103, q=0.30)
    engine.run()
    status = engine.double_spend_status
    assert status is not None
    assert status.state is DoubleSpendScenarioState.SUCCEEDED
    assert status.successful
    assert status.publication_attempted


def test_17_known_seed_produces_real_failure_at_same_q() -> None:
    _, engine = make_engine(seed=100, q=0.30)
    engine.run()
    status = engine.double_spend_status
    assert status is not None
    assert status.state is DoubleSpendScenarioState.FAILED
    assert not status.successful


def test_18_same_seed_is_bitwise_reproducible() -> None:
    assert run_signature(2026) == run_signature(2026)


def test_19_different_seeds_change_simulation() -> None:
    assert run_signature(2026) != run_signature(2027)


def test_20_clock_rng_is_independent_of_double_spend_q() -> None:
    _, low_engine = make_engine(seed=2026, q=0.10)
    _, high_engine = make_engine(seed=2026, q=0.45)
    low_steps = low_engine.run()
    high_steps = high_engine.run()
    assert tuple(step.timestamp for step in low_steps) == tuple(
        step.timestamp for step in high_steps
    )


def test_21_stochastic_pow_clock_remains_positive_and_nonfixed() -> None:
    _, engine = make_engine(seed=2028, blocks=200)
    steps = engine.run()
    timestamps = [step.timestamp for step in steps]
    intervals = [timestamps[0]] + [
        current - previous for previous, current in pairwise(timestamps)
    ]
    assert all(isfinite(value) and value > 0.0 for value in intervals)
    assert len({round(value, 6) for value in intervals}) > 190


def test_22_q_selector_tracks_configured_probability() -> None:
    _, low_engine = make_engine(seed=2028, q=0.10)
    _, high_engine = make_engine(seed=2028, q=0.45)
    low = Counter(low_engine._select_double_spend_mining_side() for _ in range(10_000))
    high = Counter(
        high_engine._select_double_spend_mining_side() for _ in range(10_000)
    )
    low_rate = low[DoubleSpendMiningSide.ATTACKER] / 10_000
    high_rate = high[DoubleSpendMiningSide.ATTACKER] / 10_000
    assert 0.08 < low_rate < 0.12
    assert 0.42 < high_rate < 0.48
    assert high_rate > low_rate


def test_23_higher_q_increases_success_rate_over_many_seeds() -> None:
    seeds = range(100, 200)
    low_successes = sum(outcome(seed, 0.10) for seed in seeds)
    high_successes = sum(outcome(seed, 0.45) for seed in seeds)
    assert low_successes < high_successes
    assert high_successes - low_successes >= 20


def test_24_same_q_contains_both_success_and_failure() -> None:
    outcomes = [outcome(seed, 0.30) for seed in range(100, 150)]
    assert any(outcomes)
    assert not all(outcomes)


def test_25_no_artificial_honest_miner_in_scientific_run() -> None:
    network, engine = make_engine(seed=103, q=0.30, blocks=60, end=45)
    engine.run()
    miner_ids = {block.miner_id for block in network.nodes[0].blockchain.chain[1:]}
    assert "honest" not in miner_ids


def test_26_canonical_public_miners_are_pool_members_or_attacker() -> None:
    network, engine = make_engine(seed=103, q=0.30, blocks=60, end=45)
    engine.run()
    allowed = {miner.miner_id for miner in public_miners()} | {"double_spend_attacker"}
    observed = {block.miner_id for block in network.nodes[0].blockchain.chain[1:]}
    assert observed <= allowed


def test_27_successful_attack_generates_observable_reorganization() -> None:
    _, engine = make_engine(seed=103, q=0.30)
    steps = engine.run()
    assert any(step.chain_observation.reorganization_detected for step in steps)
    assert any(step.chain_observation.reorganization_depth > 0 for step in steps)


def test_28_private_ground_truth_does_not_leak_into_features() -> None:
    _, engine = make_engine(seed=103, q=0.30)
    steps = engine.run()
    forbidden = {
        "attacker_hash_share",
        "attack_successful",
        "successful",
        "private_branch",
        "private_branch_length",
        "private_branch_ahead",
        "mining_side",
        "attack_phase",
        "publication_attempted",
    }
    feature_names = set(steps[0].features.as_dict())
    assert forbidden.isdisjoint(feature_names)
    assert len(feature_names) == 23


def test_29_feature_and_telemetry_timestamps_remain_synchronized() -> None:
    _, engine = make_engine(seed=103, q=0.30)
    steps = engine.run()
    for step in steps:
        assert step.timestamp == step.telemetry.timestamp == step.features.timestamp
        assert step.canonical_height == step.features.canonical_height
        assert step.canonical_height == step.features.reference_height


def test_30_multinode_latency_and_loss_run_is_valid() -> None:
    network, engine = make_engine(
        seed=2027,
        q=0.40,
        blocks=100,
        start=10,
        end=70,
        node_count=5,
        latency_ms=120.0,
        latency_std_ms=30.0,
        packet_loss=0.05,
    )
    steps = engine.run()
    assert len(steps) == 100
    assert all(node.blockchain.validate() for node in network.nodes)
    assert all(len(step.features.as_dict()) == 23 for step in steps)


def test_31_multinode_run_is_reproducible() -> None:
    def signature() -> tuple[object, ...]:
        network, engine = make_engine(
            seed=2028,
            q=0.35,
            blocks=120,
            start=20,
            end=90,
            node_count=5,
            latency_ms=100.0,
            latency_std_ms=20.0,
            packet_loss=0.10,
        )
        steps = engine.run()
        return (
            tuple(block.block_hash for block in network.nodes[0].blockchain.chain),
            tuple(step.timestamp for step in steps),
            tuple(step.features.as_numeric_tuple() for step in steps),
            engine.double_spend_status,
        )

    assert signature() == signature()


def test_32_normal_mode_still_runs() -> None:
    network, engine = make_engine(
        seed=42,
        blocks=30,
        attack_type=AttackType.NORMAL,
    )
    steps = engine.run()
    assert len(steps) == 30
    assert all(step.attack_label is AttackType.NORMAL for step in steps)
    assert network.nodes[0].blockchain.validate()
    assert engine.double_spend_status is None


def test_33_engine_can_only_run_once() -> None:
    _, engine = make_engine(seed=42)
    engine.run()
    with pytest.raises(RuntimeError):
        engine.run()


def test_34_attack_type_values_remain_stable() -> None:
    assert int(AttackType.NORMAL) == 0
    assert int(AttackType.DOUBLE_SPEND) == 1
    assert int(AttackType.SELFISH_MINING) == 2
    assert int(AttackType.MAJORITY_ATTACK) == 3

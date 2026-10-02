from __future__ import annotations

from math import isfinite

import pytest

from src.simulator.attacks import (
    AttackConfiguration,
    AttackType,
    AttackWindow,
    DoubleSpendParameters,
    MajorityAttackParameters,
    SelfishMiningParameters,
)
from src.simulator.engine import SimulationEngine, SimulationParameters
from src.simulator.majority_attack import MajorityAttack, MajorityAttackState, MajorityMiningSide
from src.simulator.mining import Miner
from src.simulator.network import Network, NetworkParameters, Node
from src.simulator.selfish_mining import SelfishMiningAttack, SelfishMiningSide, SelfishMiningState


def miners() -> tuple[Miner, ...]:
    return (
        Miner("miner_a", 0.40), Miner("miner_b", 0.25), Miner("miner_c", 0.20),
        Miner("miner_d", 0.10), Miner("miner_e", 0.05),
    )


def make_engine(
    attack_type: AttackType,
    *, seed: int = 42, blocks: int = 60, start: int = 1, end: int = 50,
    selfish_q: float = 0.30, gamma: float = 0.50, majority_q: float = 0.60,
    nodes: int = 1, latency_ms: float = 0.0, latency_std_ms: float = 0.0,
    packet_loss: float = 0.0,
) -> tuple[Network, SimulationEngine]:
    net = Network(
        [Node(f"node_{i}") for i in range(nodes)],
        NetworkParameters(latency_ms, latency_std_ms, packet_loss), seed,
    )
    cfg = AttackConfiguration(
        attack_type, AttackWindow(start, end), DoubleSpendParameters(1.0, 2, 0.30),
        SelfishMiningParameters(gamma, selfish_q), MajorityAttackParameters(majority_q),
    )
    eng = SimulationEngine(
        net, cfg, SimulationParameters(blocks, 600.0, selfish_q, seed), miners()
    )
    return net, eng


def signature(attack_type: AttackType, seed: int) -> tuple:
    net, eng = make_engine(attack_type, seed=seed, nodes=3, latency_ms=100.0,
                           latency_std_ms=20.0, packet_loss=0.03)
    steps = eng.run()
    return (
        tuple(b.block_hash for b in net.nodes[0].blockchain.chain),
        tuple(s.timestamp for s in steps),
        tuple(s.features.as_numeric_tuple() for s in steps),
    )


def test_01_selfish_parameters_include_q() -> None:
    p = SelfishMiningParameters(0.5, 0.3)
    assert p.gamma == 0.5 and p.attacker_hash_share == 0.3


@pytest.mark.parametrize("q", [0.0, -0.1, 0.5, 0.8, float("inf"), float("nan")])
def test_02_invalid_selfish_q_rejected(q: float) -> None:
    with pytest.raises(ValueError):
        SelfishMiningParameters(0.5, q)


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("inf"), float("nan")])
def test_03_invalid_gamma_rejected(gamma: float) -> None:
    with pytest.raises(ValueError):
        SelfishMiningParameters(gamma, 0.3)


def test_04_selfish_side_values_are_stable() -> None:
    assert int(SelfishMiningSide.PUBLIC) == 0
    assert int(SelfishMiningSide.ATTACKER) == 1


def test_05_selfish_attacker_event_only_grows_private_branch() -> None:
    node = Node("victim")
    attack = SelfishMiningAttack(node, SelfishMiningParameters(0.5, 0.3), "attacker", 1)
    attack.start()
    attack.process_mining_event(1.0, SelfishMiningSide.ATTACKER)
    assert node.height == 0 and attack.private_branch.height == 1


def test_06_selfish_public_event_requires_real_miner() -> None:
    attack = SelfishMiningAttack(Node("victim"), SelfishMiningParameters(0.5, 0.3), "attacker", 1)
    attack.start()
    with pytest.raises(ValueError):
        attack.process_mining_event(1.0, SelfishMiningSide.PUBLIC)


def test_07_selfish_one_block_lead_enters_race_after_public_block() -> None:
    attack = SelfishMiningAttack(Node("victim"), SelfishMiningParameters(0.5, 0.3), "attacker", 1)
    attack.start()
    attack.process_mining_event(1.0, SelfishMiningSide.ATTACKER)
    attack.process_mining_event(2.0, SelfishMiningSide.PUBLIC, "miner_a")
    assert attack.state is SelfishMiningState.RACE


def test_08_gamma_one_wins_public_supported_race() -> None:
    node = Node("victim")
    attack = SelfishMiningAttack(node, SelfishMiningParameters(1.0, 0.3), "attacker", 1)
    attack.start()
    attack.process_mining_event(1.0, SelfishMiningSide.ATTACKER)
    attack.process_mining_event(2.0, SelfishMiningSide.PUBLIC, "miner_a")
    attack.process_mining_event(3.0, SelfishMiningSide.PUBLIC, "miner_b")
    status = attack.status()
    assert status.race_wins == 1 and status.reorganizations == 1


def test_09_gamma_zero_loses_public_supported_race() -> None:
    node = Node("victim")
    attack = SelfishMiningAttack(node, SelfishMiningParameters(0.0, 0.3), "attacker", 1)
    attack.start()
    attack.process_mining_event(1.0, SelfishMiningSide.ATTACKER)
    attack.process_mining_event(2.0, SelfishMiningSide.PUBLIC, "miner_a")
    attack.process_mining_event(3.0, SelfishMiningSide.PUBLIC, "miner_b")
    status = attack.status()
    assert status.race_losses == 1 and status.reorganizations == 0


def test_10_selfish_finish_is_terminal() -> None:
    attack = SelfishMiningAttack(Node("victim"), SelfishMiningParameters(0.5, 0.3), "attacker", 1)
    attack.start()
    attack.finish()
    assert attack.state is SelfishMiningState.FINISHED
    with pytest.raises(RuntimeError):
        attack.process_mining_event(1.0, SelfishMiningSide.ATTACKER)


def test_11_majority_parameter_boundaries() -> None:
    assert MajorityAttackParameters(0.500001).attacker_hash_share > 0.5
    assert MajorityAttackParameters(1.0).attacker_hash_share == 1.0


@pytest.mark.parametrize("q", [0.0, 0.5, 1.01, float("inf"), float("nan")])
def test_12_invalid_majority_q_rejected(q: float) -> None:
    with pytest.raises(ValueError):
        MajorityAttackParameters(q)


def test_13_majority_side_values_are_stable() -> None:
    assert int(MajorityMiningSide.PUBLIC) == 0
    assert int(MajorityMiningSide.ATTACKER) == 1


def test_14_majority_attacker_event_only_grows_private_branch() -> None:
    node = Node("victim")
    attack = MajorityAttack(node, MajorityAttackParameters(0.6), "attacker")
    attack.start()
    attack.process_mining_event(1.0, MajorityMiningSide.ATTACKER)
    assert node.height == 0 and attack.attacker_branch.height == 1


def test_15_majority_public_event_requires_real_miner() -> None:
    attack = MajorityAttack(Node("victim"), MajorityAttackParameters(0.6), "attacker")
    attack.start()
    with pytest.raises(ValueError):
        attack.process_mining_event(1.0, MajorityMiningSide.PUBLIC)


def test_16_majority_public_then_superior_private_branch_reorganizes() -> None:
    node = Node("victim")
    attack = MajorityAttack(node, MajorityAttackParameters(0.6), "attacker")
    attack.start()
    attack.process_mining_event(1.0, MajorityMiningSide.PUBLIC, "miner_a")
    attack.process_mining_event(2.0, MajorityMiningSide.ATTACKER)
    attack.process_mining_event(3.0, MajorityMiningSide.ATTACKER)
    status = attack.status()
    assert status.reorganizations == 1 and status.successful


def test_17_majority_full_hash_power_can_publish_controlled_chain() -> None:
    node = Node("victim")
    attack = MajorityAttack(node, MajorityAttackParameters(1.0), "attacker")
    attack.start()
    attack.process_mining_event(1.0, MajorityMiningSide.ATTACKER)
    attack.process_mining_event(2.0, MajorityMiningSide.ATTACKER)
    assert attack.status().blocks_published == 2 and node.height == 2


def test_18_majority_finish_is_terminal() -> None:
    attack = MajorityAttack(Node("victim"), MajorityAttackParameters(0.6), "attacker")
    attack.start()
    attack.finish()
    assert attack.state is MajorityAttackState.FINISHED
    with pytest.raises(RuntimeError):
        attack.process_mining_event(1.0, MajorityMiningSide.ATTACKER)


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_19_engine_attack_labels_respect_window(attack_type: AttackType) -> None:
    _, eng = make_engine(attack_type, blocks=30, start=5, end=20)
    labels = [s.attack_label for s in eng.run()]
    assert labels[:4] == [AttackType.NORMAL] * 4
    assert labels[4:20] == [attack_type] * 16
    assert labels[20:] == [AttackType.NORMAL] * 10


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_20_attack_window_is_finalized(attack_type: AttackType) -> None:
    _, eng = make_engine(attack_type, blocks=30, start=2, end=10)
    eng.run()
    if attack_type is AttackType.SELFISH_MINING:
        status = eng.selfish_mining_status
        assert status is not None
        assert status.finished
    else:
        status = eng.majority_attack_status
        assert status is not None
        assert status.finished


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_21_same_seed_is_bitwise_reproducible(attack_type: AttackType) -> None:
    assert signature(attack_type, 2026) == signature(attack_type, 2026)


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_22_different_seeds_change_run(attack_type: AttackType) -> None:
    assert signature(attack_type, 2026) != signature(attack_type, 2027)


def test_23_selfish_selector_tracks_q() -> None:
    _, low = make_engine(AttackType.SELFISH_MINING, selfish_q=0.10)
    _, high = make_engine(AttackType.SELFISH_MINING, selfish_q=0.45)
    low_rate = sum(low._select_selfish_mining_side() is SelfishMiningSide.ATTACKER for _ in range(10000)) / 10000
    high_rate = sum(high._select_selfish_mining_side() is SelfishMiningSide.ATTACKER for _ in range(10000)) / 10000
    assert 0.08 < low_rate < 0.12 and 0.42 < high_rate < 0.48


def test_24_majority_selector_tracks_q() -> None:
    _, low = make_engine(AttackType.MAJORITY_ATTACK, majority_q=0.51)
    _, high = make_engine(AttackType.MAJORITY_ATTACK, majority_q=0.80)
    low_rate = sum(low._select_majority_mining_side() is MajorityMiningSide.ATTACKER for _ in range(10000)) / 10000
    high_rate = sum(high._select_majority_mining_side() is MajorityMiningSide.ATTACKER for _ in range(10000)) / 10000
    assert 0.49 < low_rate < 0.53 and 0.78 < high_rate < 0.82


def selfish_mean_reorg(q: float, gamma: float) -> float:
    values = []
    for seed in range(40):
        _, eng = make_engine(AttackType.SELFISH_MINING, seed=seed, blocks=50, end=45,
                             selfish_q=q, gamma=gamma)
        eng.run()
        status = eng.selfish_mining_status
        assert status is not None
        values.append(status.reorganizations)
    return sum(values) / len(values)


def test_25_higher_selfish_q_increases_strategic_reorganizations() -> None:
    assert selfish_mean_reorg(0.40, 0.5) > selfish_mean_reorg(0.10, 0.5)


def test_26_higher_gamma_increases_selfish_race_success() -> None:
    assert selfish_mean_reorg(0.30, 1.0) > selfish_mean_reorg(0.30, 0.0)


def test_27_higher_majority_q_increases_published_control() -> None:
    means = []
    for q in (0.51, 0.70):
        values = []
        for seed in range(40):
            _, eng = make_engine(AttackType.MAJORITY_ATTACK, seed=seed, blocks=50, end=45, majority_q=q)
            eng.run()
            status = eng.majority_attack_status
            assert status is not None
            values.append(status.blocks_published)
        means.append(sum(values) / len(values))
    assert means[1] > means[0]


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_28_no_synthetic_honest_miner(attack_type: AttackType) -> None:
    net, eng = make_engine(attack_type, seed=123, blocks=80, end=70)
    eng.run()
    assert "honest" not in {b.miner_id for b in net.nodes[0].blockchain.chain[1:]}


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_29_canonical_miners_are_known(attack_type: AttackType) -> None:
    net, eng = make_engine(attack_type, seed=2028, blocks=80, end=70)
    eng.run()
    allowed = {m.miner_id for m in miners()} | {"selfish_attacker", "majority_attacker"}
    assert {b.miner_id for b in net.nodes[0].blockchain.chain[1:]} <= allowed


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_30_private_ground_truth_does_not_leak_into_features(attack_type: AttackType) -> None:
    _, eng = make_engine(attack_type)
    steps = eng.run()
    forbidden = {"attacker_hash_share", "gamma", "private_lead", "attacker_lead", "successful",
                 "attack_successful", "attack_phase", "mining_side"}
    assert forbidden.isdisjoint(steps[0].features.as_dict())


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_31_multinode_latency_loss_run_is_valid(attack_type: AttackType) -> None:
    net, eng = make_engine(attack_type, seed=2027, blocks=100, end=80, nodes=5,
                           latency_ms=120.0, latency_std_ms=30.0, packet_loss=0.05)
    steps = eng.run()
    assert len(steps) == 100 and net.nodes[0].blockchain.validate()
    assert all(len(s.features.as_dict()) == 23 for s in steps)


@pytest.mark.parametrize("attack_type", [AttackType.SELFISH_MINING, AttackType.MAJORITY_ATTACK])
def test_32_timestamps_are_positive_nonfixed_and_synchronized(attack_type: AttackType) -> None:
    _, eng = make_engine(attack_type, blocks=100, end=90)
    steps = eng.run()
    ts = [s.timestamp for s in steps]
    assert all(isfinite(t) for t in ts)
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert len({round(b-a, 6) for a, b in zip(ts, ts[1:])}) > 90
    assert all(s.timestamp == s.telemetry.timestamp == s.features.timestamp for s in steps)


def test_33_double_spend_regression_still_runs() -> None:
    net, eng = make_engine(AttackType.DOUBLE_SPEND, seed=42, blocks=40, end=30)
    assert len(eng.run()) == 40 and net.nodes[0].blockchain.validate()


def test_34_normal_regression_still_runs() -> None:
    net, eng = make_engine(AttackType.NORMAL, seed=42, blocks=40)
    assert len(eng.run()) == 40 and net.nodes[0].height == 40


from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.dataset.builder import DatasetBuilder
from src.dataset.config import (
    ExperimentPlan,
    FloatRange,
    IntRange,
    NetworkProfile,
    SplitRatios,
    load_experiment_plan,
)
from src.dataset.scenario import ScenarioGenerator
from src.dataset.schema import (
    EARLY_DETECTION_FEATURE_COLUMNS,
    FORBIDDEN_MODEL_COLUMNS,
    METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OBSERVABLE_COLUMNS,
    TARGET_COLUMNS,
)
from src.dataset.splitting import assign_simulation_splits
from src.simulator.attacks import AttackType


def make_plan(seed: int = 2027, scenarios_per_class: int = 4) -> ExperimentPlan:
    return ExperimentPlan(
        base_seed=seed,
        scenarios_per_class=scenarios_per_class,
        duration_blocks=32,
        block_interval_seconds=600.0,
        normal_window_length=IntRange(8, 12),
        double_spend_window_length=IntRange(8, 12),
        selfish_window_length=IntRange(2, 8),
        majority_window_length=IntRange(1, 6),
        attack_start_margin=4,
        node_count=IntRange(3, 5),
        miner_count=IntRange(2, 4),
        double_spend_q=FloatRange(0.10, 0.45),
        selfish_q=FloatRange(0.10, 0.45),
        selfish_gamma=FloatRange(0.0, 1.0),
        majority_q=FloatRange(0.51, 0.70),
        confirmation_targets=(2, 3, 4),
        miner_concentration_alphas=(
            ("decentralized", 8.0),
            ("moderate", 2.0),
            ("concentrated", 0.5),
        ),
        network_profiles=(
            NetworkProfile(
                "low",
                FloatRange(30.0, 80.0),
                FloatRange(5.0, 15.0),
                FloatRange(0.0, 0.002),
            ),
            NetworkProfile(
                "high",
                FloatRange(180.0, 250.0),
                FloatRange(30.0, 60.0),
                FloatRange(0.005, 0.02),
            ),
        ),
        splits=SplitRatios(0.5, 0.25, 0.25),
        output_root="data",
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_01_model_features_exclude_forbidden_columns() -> None:
    assert not set(MODEL_FEATURE_COLUMNS) & FORBIDDEN_MODEL_COLUMNS
    assert "timestamp" not in MODEL_FEATURE_COLUMNS
    assert "canonical_height" not in MODEL_FEATURE_COLUMNS
    assert "reference_height" not in MODEL_FEATURE_COLUMNS


def test_02_early_detection_excludes_post_reorg_signals() -> None:
    for column in (
        "blocks_removed",
        "common_ancestor_height",
        "reorganization_detected",
        "reorganization_depth",
    ):
        assert column not in EARLY_DETECTION_FEATURE_COLUMNS


def test_03_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        SplitRatios(0.7, 0.2, 0.2)


def test_04_default_experiment_yaml_loads() -> None:
    plan = load_experiment_plan("configs/experiment.yaml")
    assert plan.scenarios_per_class == 75
    assert plan.duration_blocks == 192
    assert plan.double_spend_q.maximum < 0.5
    assert plan.majority_q.minimum > 0.5


def test_05_scenario_collection_is_balanced() -> None:
    scenarios = ScenarioGenerator(make_plan()).generate()
    counts = {attack_type: 0 for attack_type in AttackType}
    for scenario in scenarios:
        counts[scenario.attack_type] += 1
    assert set(counts.values()) == {4}


def test_06_scenario_generation_is_exactly_reproducible() -> None:
    first = ScenarioGenerator(make_plan()).generate()
    second = ScenarioGenerator(make_plan()).generate()
    assert first == second


def test_07_different_base_seed_changes_scenarios() -> None:
    assert ScenarioGenerator(make_plan(1)).generate() != ScenarioGenerator(make_plan(2)).generate()


def test_08_attack_windows_are_randomized_and_valid() -> None:
    scenarios = ScenarioGenerator(make_plan(scenarios_per_class=12)).generate()
    windows = {(scenario.attack_start_step, scenario.attack_end_step) for scenario in scenarios}
    assert len(windows) > 1
    for scenario in scenarios:
        assert 1 <= scenario.attack_start_step <= scenario.attack_end_step <= scenario.duration_blocks


def test_09_attack_parameter_ranges_are_respected() -> None:
    plan = make_plan(scenarios_per_class=12)
    for scenario in ScenarioGenerator(plan).generate():
        assert plan.double_spend_q.minimum <= scenario.double_spend_q <= plan.double_spend_q.maximum
        assert plan.selfish_q.minimum <= scenario.selfish_q <= plan.selfish_q.maximum
        assert plan.selfish_gamma.minimum <= scenario.gamma <= plan.selfish_gamma.maximum
        assert plan.majority_q.minimum <= scenario.majority_q <= plan.majority_q.maximum


def test_10_public_miner_hash_shares_are_positive_and_normalized() -> None:
    for scenario in ScenarioGenerator(make_plan()).generate():
        assert all(value > 0.0 for value in scenario.public_miner_hash_powers)
        assert sum(scenario.public_miner_hash_powers) == pytest.approx(1.0, abs=1e-12)


def test_11_scenarios_vary_network_and_miner_regimes() -> None:
    scenarios = ScenarioGenerator(make_plan(scenarios_per_class=20)).generate()
    assert len({scenario.network_profile for scenario in scenarios}) > 1
    assert len({scenario.miner_concentration for scenario in scenarios}) > 1


def test_12_split_assignment_is_simulation_level_and_complete() -> None:
    scenarios = ScenarioGenerator(make_plan()).generate()
    assignments = assign_simulation_splits(scenarios, make_plan().splits, 99)
    assert set(assignments) == {scenario.simulation_id for scenario in scenarios}
    assert set(assignments.values()) == {"train", "validation", "test"}


def test_13_split_assignment_is_stratified_by_attack_class() -> None:
    scenarios = ScenarioGenerator(make_plan()).generate()
    assignments = assign_simulation_splits(scenarios, make_plan().splits, 99)
    for attack_type in AttackType:
        splits = {
            assignments[scenario.simulation_id]
            for scenario in scenarios
            if scenario.attack_type is attack_type
        }
        assert splits == {"train", "validation", "test"}


@pytest.fixture(scope="module")
def built_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("dataset_build")
    result = DatasetBuilder(make_plan()).build(root)
    assert result.scenario_count == 16
    assert result.sample_count == 16 * 32
    return root


def test_14_builder_creates_expected_output_files(built_dataset: Path) -> None:
    expected = {
        "raw/blockchain_telemetry.csv",
        "manifests/scenarios.csv",
        "manifests/schema.json",
        "manifests/dataset_manifest.json",
    }
    for split in ("train", "validation", "test"):
        expected.update(
            {
                f"splits/x_{split}.csv",
                f"splits/x_early_{split}.csv",
                f"splits/metadata_{split}.csv",
                f"splits/y_{split}.csv",
            }
        )
    assert expected <= {
        str(path.relative_to(built_dataset))
        for path in built_dataset.rglob("*")
        if path.is_file()
    }


def test_15_model_x_header_is_exact_and_leakage_safe(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "splits/x_train.csv")
    assert rows
    assert tuple(rows[0]) == MODEL_FEATURE_COLUMNS
    assert not set(rows[0]) & FORBIDDEN_MODEL_COLUMNS


def test_16_early_x_header_is_exact(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "splits/x_early_train.csv")
    assert rows
    assert tuple(rows[0]) == EARLY_DETECTION_FEATURE_COLUMNS


def test_17_y_contains_only_target(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "splits/y_train.csv")
    assert rows
    assert tuple(rows[0]) == TARGET_COLUMNS
    assert {int(row["attack_label"]) for row in rows} <= {0, 1, 2, 3}


def test_18_metadata_contains_ground_truth_but_is_separate(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "splits/metadata_train.csv")
    assert rows
    assert tuple(rows[0]) == METADATA_COLUMNS
    assert "attack_successful" in rows[0]
    assert "seed" in rows[0]
    assert "configured_attack_type" in rows[0]


def test_19_raw_dataset_preserves_all_observables(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "raw/blockchain_telemetry.csv")
    assert rows
    assert tuple(rows[0]) == (*METADATA_COLUMNS, *OBSERVABLE_COLUMNS, *TARGET_COLUMNS)


def test_20_split_files_have_aligned_row_counts(built_dataset: Path) -> None:
    for split in ("train", "validation", "test"):
        counts = {
            len(read_rows(built_dataset / f"splits/{kind}_{split}.csv"))
            for kind in ("x", "x_early", "metadata", "y")
        }
        assert len(counts) == 1


def test_21_no_simulation_crosses_split_boundaries(built_dataset: Path) -> None:
    seen: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for row in read_rows(built_dataset / f"splits/metadata_{split}.csv"):
            simulation_id = row["simulation_id"]
            assert simulation_id not in seen or seen[simulation_id] == split
            seen[simulation_id] = split


def test_22_attack_scenarios_contain_pre_active_and_post_phases(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "raw/blockchain_telemetry.csv")
    by_simulation: dict[str, set[str]] = {}
    attack_type: dict[str, str] = {}
    for row in rows:
        simulation_id = row["simulation_id"]
        by_simulation.setdefault(simulation_id, set()).add(row["attack_phase"])
        attack_type[simulation_id] = row["configured_attack_type"]
    for simulation_id, phases in by_simulation.items():
        if attack_type[simulation_id] != "NORMAL":
            assert {"pre_attack", "active", "post_attack"} <= phases


def test_23_scenario_outcome_is_constant_within_simulation(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "raw/blockchain_telemetry.csv")
    outcomes: dict[str, set[str]] = {}
    for row in rows:
        outcomes.setdefault(row["simulation_id"], set()).add(row["attack_successful"])
    assert all(len(values) == 1 for values in outcomes.values())


def test_24_scenario_manifest_has_one_row_per_simulation(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "manifests/scenarios.csv")
    assert len(rows) == 16
    assert len({row["simulation_id"] for row in rows}) == 16


def test_25_manifest_hashes_match_files(built_dataset: Path) -> None:
    manifest = json.loads((built_dataset / "manifests/dataset_manifest.json").read_text())
    for relative_path, expected_hash in manifest["sha256"].items():
        assert sha256(built_dataset / relative_path) == expected_hash


def test_26_schema_manifest_matches_python_constants(built_dataset: Path) -> None:
    schema = json.loads((built_dataset / "manifests/schema.json").read_text())
    assert tuple(schema["observable_columns"]) == OBSERVABLE_COLUMNS
    assert tuple(schema["model_feature_columns"]) == MODEL_FEATURE_COLUMNS
    assert tuple(schema["early_detection_feature_columns"]) == EARLY_DETECTION_FEATURE_COLUMNS


def test_27_raw_sample_ids_are_unique(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "raw/blockchain_telemetry.csv")
    sample_ids = [row["sample_id"] for row in rows]
    assert len(sample_ids) == len(set(sample_ids))


def test_28_normal_scenarios_have_no_attacker_hash_share(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "raw/blockchain_telemetry.csv")
    normal_rows = [row for row in rows if row["configured_attack_type"] == "NORMAL"]
    assert normal_rows
    assert all(row["attacker_hash_share"] == "" for row in normal_rows)


def test_29_gamma_is_metadata_only_for_selfish_mining(built_dataset: Path) -> None:
    rows = read_rows(built_dataset / "raw/blockchain_telemetry.csv")
    for row in rows:
        if row["configured_attack_type"] == "SELFISH_MINING":
            assert row["gamma"] != ""
        else:
            assert row["gamma"] == ""


def test_30_complete_build_is_bitwise_reproducible(tmp_path: Path) -> None:
    plan = make_plan(scenarios_per_class=3)
    first = tmp_path / "first"
    second = tmp_path / "second"
    DatasetBuilder(plan).build(first)
    DatasetBuilder(plan).build(second)

    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    for relative_path in first_files:
        assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()

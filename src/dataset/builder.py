"""Reproducible construction of raw and leakage-safe ML datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.dataset.config import ExperimentPlan
from src.dataset.scenario import ScenarioGenerator, ScenarioSpec
from src.dataset.schema import (
    EARLY_DETECTION_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OBSERVABLE_COLUMNS,
    TARGET_COLUMNS,
)
from src.dataset.splitting import SPLIT_NAMES, assign_simulation_splits
from src.simulator.attacks import (
    AttackConfiguration,
    AttackType,
    AttackWindow,
    DoubleSpendParameters,
    MajorityAttackParameters,
    SelfishMiningParameters,
)
from src.simulator.engine import SimulationEngine, SimulationParameters
from src.simulator.mining import Miner
from src.simulator.network import Network, NetworkParameters, Node


@dataclass(slots=True, frozen=True)
class DatasetBuildResult:
    """Summarize one completed deterministic dataset build."""

    output_root: Path
    scenario_count: int
    sample_count: int
    split_sample_counts: dict[str, int]
    manifest_path: Path


class DatasetBuilder:
    """Run simulations and serialize raw, model, target, and metadata files."""

    def __init__(self, plan: ExperimentPlan) -> None:
        self._plan = plan

    def build(self, output_root: str | Path | None = None) -> DatasetBuildResult:
        """Build the complete dataset from deterministic scenario specifications."""

        root = Path(output_root if output_root is not None else self._plan.output_root)
        raw_dir = root / "raw"
        splits_dir = root / "splits"
        manifests_dir = root / "manifests"

        for directory in (raw_dir, splits_dir, manifests_dir):
            directory.mkdir(parents=True, exist_ok=True)

        scenarios = ScenarioGenerator(self._plan).generate()
        assignments = assign_simulation_splits(
            scenarios,
            self._plan.splits,
            self._plan.base_seed + 404,
        )

        raw_path = raw_dir / "blockchain_telemetry.csv"
        scenario_manifest_path = manifests_dir / "scenarios.csv"
        split_sample_counts = {name: 0 for name in SPLIT_NAMES}
        sample_count = 0

        with ExitStack() as stack:
            raw_handle = stack.enter_context(raw_path.open("w", encoding="utf-8", newline=""))
            raw_columns = (*METADATA_COLUMNS, *OBSERVABLE_COLUMNS, *TARGET_COLUMNS)
            raw_writer = csv.DictWriter(raw_handle, fieldnames=raw_columns)
            raw_writer.writeheader()

            writers = self._open_split_writers(stack, splits_dir)
            scenario_rows: list[dict[str, Any]] = []

            for scenario in scenarios:
                split = assignments[scenario.simulation_id]
                steps, successful = self._run_scenario(scenario)

                for simulation_step in steps:
                    sample_id = f"{scenario.simulation_id}_{simulation_step.step:05d}"
                    feature_dict = simulation_step.features.as_dict()
                    metadata = self._metadata_row(
                        scenario=scenario,
                        split=split,
                        step=simulation_step.step,
                        sample_id=sample_id,
                        successful=successful,
                    )
                    target = {"attack_label": int(simulation_step.attack_label)}

                    raw_row = {
                        **metadata,
                        **feature_dict,
                        **target,
                    }
                    raw_writer.writerow(raw_row)

                    writers[split]["x"].writerow(
                        {column: feature_dict[column] for column in MODEL_FEATURE_COLUMNS}
                    )
                    writers[split]["x_early"].writerow(
                        {
                            column: feature_dict[column]
                            for column in EARLY_DETECTION_FEATURE_COLUMNS
                        }
                    )
                    writers[split]["metadata"].writerow(metadata)
                    writers[split]["y"].writerow(target)

                    split_sample_counts[split] += 1
                    sample_count += 1

                scenario_rows.append(
                    self._scenario_manifest_row(
                        scenario,
                        split,
                        successful,
                        len(steps),
                    )
                )

        self._write_scenario_manifest(scenario_manifest_path, scenario_rows)
        schema_path = manifests_dir / "schema.json"
        self._write_schema(schema_path)

        manifest_path = manifests_dir / "dataset_manifest.json"
        self._write_dataset_manifest(
            manifest_path=manifest_path,
            root=root,
            scenario_count=len(scenarios),
            sample_count=sample_count,
            split_sample_counts=split_sample_counts,
        )

        return DatasetBuildResult(
            output_root=root,
            scenario_count=len(scenarios),
            sample_count=sample_count,
            split_sample_counts=split_sample_counts,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _open_split_writers(
        stack: ExitStack,
        splits_dir: Path,
    ) -> dict[str, dict[str, csv.DictWriter]]:
        writers: dict[str, dict[str, csv.DictWriter]] = {}
        definitions = {
            "x": MODEL_FEATURE_COLUMNS,
            "x_early": EARLY_DETECTION_FEATURE_COLUMNS,
            "metadata": METADATA_COLUMNS,
            "y": TARGET_COLUMNS,
        }

        for split in SPLIT_NAMES:
            writers[split] = {}
            for kind, columns in definitions.items():
                path = splits_dir / f"{kind}_{split}.csv"
                handle = stack.enter_context(
                    open(path, "w", encoding="utf-8", newline="")
                )
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writers[split][kind] = writer

        return writers

    def _run_scenario(
        self,
        scenario: ScenarioSpec,
    ) -> tuple[tuple[Any, ...], bool]:
        nodes = [Node(f"node_{index:03d}") for index in range(scenario.num_nodes)]
        network = Network(
            nodes=nodes,
            parameters=NetworkParameters(
                base_latency_ms=scenario.base_latency_ms,
                latency_std_ms=scenario.latency_std_ms,
                packet_loss_probability=scenario.packet_loss_probability,
            ),
            seed=scenario.seed + 101,
        )
        miners = tuple(
            Miner(
                miner_id=f"miner_{index:03d}",
                hash_power=hash_power,
            )
            for index, hash_power in enumerate(scenario.public_miner_hash_powers)
        )
        attack_configuration = AttackConfiguration(
            attack_type=scenario.attack_type,
            window=AttackWindow(
                scenario.attack_start_step,
                scenario.attack_end_step,
            ),
            double_spend=DoubleSpendParameters(
                transaction_value=1.0,
                target_confirmations=scenario.double_spend_target_confirmations,
                attacker_hash_share=scenario.double_spend_q,
            ),
            selfish_mining=SelfishMiningParameters(
                gamma=scenario.gamma,
                attacker_hash_share=scenario.selfish_q,
            ),
            majority_attack=MajorityAttackParameters(
                attacker_hash_share=scenario.majority_q,
            ),
        )
        engine = SimulationEngine(
            network=network,
            attack_configuration=attack_configuration,
            parameters=SimulationParameters(
                duration_blocks=scenario.duration_blocks,
                block_interval_seconds=scenario.block_interval_seconds,
                selfish_attacker_hash_share=scenario.selfish_q,
                seed=scenario.seed,
            ),
            public_miners=miners,
        )
        steps = engine.run()
        return steps, self._attack_successful(engine, scenario.attack_type)

    @staticmethod
    def _attack_successful(engine: SimulationEngine, attack_type: AttackType) -> bool:
        if attack_type is AttackType.NORMAL:
            return False
        if attack_type is AttackType.DOUBLE_SPEND:
            status = engine.double_spend_status
        elif attack_type is AttackType.SELFISH_MINING:
            status = engine.selfish_mining_status
        elif attack_type is AttackType.MAJORITY_ATTACK:
            status = engine.majority_attack_status
        else:
            raise ValueError(f"Unsupported attack type: {attack_type!r}.")
        if status is None:
            return False
        return bool(status.successful)

    @staticmethod
    def _metadata_row(
        scenario: ScenarioSpec,
        split: str,
        step: int,
        sample_id: str,
        successful: bool,
    ) -> dict[str, Any]:
        active = (
            scenario.attack_type is not AttackType.NORMAL
            and scenario.attack_start_step <= step <= scenario.attack_end_step
        )
        if scenario.attack_type is AttackType.NORMAL:
            phase = "normal"
        elif step < scenario.attack_start_step:
            phase = "pre_attack"
        elif step <= scenario.attack_end_step:
            phase = "active"
        else:
            phase = "post_attack"

        return {
            "sample_id": sample_id,
            "simulation_id": scenario.simulation_id,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "step": step,
            "configured_attack_type": scenario.attack_type.name,
            "attack_active": active,
            "attack_successful": successful,
            "attack_phase": phase,
            "attack_start_step": scenario.attack_start_step,
            "attack_end_step": scenario.attack_end_step,
            "attacker_hash_share": (
                "" if scenario.attacker_hash_share is None else scenario.attacker_hash_share
            ),
            "gamma": (
                scenario.gamma
                if scenario.attack_type is AttackType.SELFISH_MINING
                else ""
            ),
            "network_profile": scenario.network_profile,
            "num_nodes": scenario.num_nodes,
            "num_miners": scenario.num_miners,
            "base_latency_ms": scenario.base_latency_ms,
            "latency_std_ms": scenario.latency_std_ms,
            "packet_loss_probability": scenario.packet_loss_probability,
            "miner_concentration": scenario.miner_concentration,
            "split": split,
        }

    @staticmethod
    def _scenario_manifest_row(
        scenario: ScenarioSpec,
        split: str,
        successful: bool,
        sample_count: int,
    ) -> dict[str, Any]:
        row = asdict(scenario)
        row["attack_type"] = scenario.attack_type.name
        row["public_miner_hash_powers"] = json.dumps(
            scenario.public_miner_hash_powers,
            separators=(",", ":"),
        )
        row["split"] = split
        row["attack_successful"] = successful
        row["sample_count"] = sample_count
        return row

    @staticmethod
    def _write_scenario_manifest(
        path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        if not rows:
            raise RuntimeError("Scenario manifest cannot be empty.")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_schema(path: Path) -> None:
        payload = {
            "observable_columns": OBSERVABLE_COLUMNS,
            "model_feature_columns": MODEL_FEATURE_COLUMNS,
            "early_detection_feature_columns": EARLY_DETECTION_FEATURE_COLUMNS,
            "metadata_columns": METADATA_COLUMNS,
            "target_columns": TARGET_COLUMNS,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_dataset_manifest(
        manifest_path: Path,
        root: Path,
        scenario_count: int,
        sample_count: int,
        split_sample_counts: dict[str, int],
    ) -> None:
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path != manifest_path
        )
        hashes = {
            str(path.relative_to(root)): DatasetBuilder._sha256(path)
            for path in files
        }
        payload = {
            "scenario_count": scenario_count,
            "sample_count": sample_count,
            "split_sample_counts": split_sample_counts,
            "sha256": hashes,
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


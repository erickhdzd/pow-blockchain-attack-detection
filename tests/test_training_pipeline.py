from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.training.artifacts import aggregate_runs, sha256_file, write_manifest, write_summary_table
from src.training.classical import build_classical_model, predict_probabilities
from src.training.config import TrainingConfig, load_training_config
from src.training.data import (
    SplitData,
    load_dataset_bundle,
    load_external_split,
    make_aligned_dataset_views,
    make_aligned_split_views,
    make_endpoint_view,
    make_sequences,
)
from src.training.metrics import CLASS_NAMES, detection_delay_blocks, evaluate_multiclass
from src.training.models import build_neural_model, parameter_count
from src.training.neural import fit_neural, predict_neural, resolve_device
from src.training.reproducibility import seed_everything
from src.training.run import run_experiments


def _metadata(simulations: int, rows_per_simulation: int) -> pd.DataFrame:
    rows = []
    for simulation in range(simulations):
        configured = simulation % 4
        for step in range(1, rows_per_simulation + 1):
            rows.append(
                {
                    "sample_id": f"sim_{simulation:03d}_{step:05d}",
                    "simulation_id": f"sim_{simulation:03d}",
                    "scenario_id": f"scenario_{simulation:03d}",
                    "seed": simulation,
                    "step": step,
                    "configured_attack_type": configured,
                    "attack_active": configured != 0 and 3 <= step <= 6,
                    "attack_successful": configured != 0,
                    "attack_phase": "active" if configured != 0 and 3 <= step <= 6 else "pre_attack",
                    "attack_start_step": 3,
                    "attack_end_step": 6,
                    "attacker_hash_share": 0.3 if configured else np.nan,
                    "gamma": 0.5 if configured == 2 else np.nan,
                    "network_profile": "medium",
                    "num_nodes": 10,
                    "num_miners": 5,
                    "base_latency_ms": 100.0,
                    "latency_std_ms": 20.0,
                    "packet_loss_probability": 0.0,
                    "miner_concentration": "moderate",
                    "split": "train",
                }
            )
    return pd.DataFrame(rows)


def _write_dataset(root: Path) -> None:
    split_sizes = {"train": 8, "validation": 4, "test": 4}
    rng = np.random.default_rng(7)
    feature_names = ["f0", "f1", "f2", "f3"]
    for split, simulations in split_sizes.items():
        metadata = _metadata(simulations, 8)
        metadata["split"] = split
        y = metadata["configured_attack_type"].to_numpy(dtype=np.int64)
        x = rng.normal(size=(len(metadata), len(feature_names))).astype(np.float32)
        x[:, 0] += y
        pd.DataFrame(x, columns=feature_names).to_csv(root / "splits" / f"x_{split}.csv", index=False)
        pd.DataFrame(x[:, :3], columns=feature_names[:3]).to_csv(
            root / "splits" / f"x_early_{split}.csv",
            index=False,
        )
        pd.DataFrame({"attack_label": y}).to_csv(root / "splits" / f"y_{split}.csv", index=False)
        metadata.to_csv(root / "splits" / f"metadata_{split}.csv", index=False)
    (root / "manifests").mkdir(exist_ok=True)
    (root / "manifests" / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "splits").mkdir(parents=True)
    _write_dataset(root)
    return root


def test_01_training_yaml_loads() -> None:
    config = load_training_config("configs/training.yaml")
    assert config.seeds == (42, 123, 2026, 2027, 2028)
    assert config.sequence_length == 16
    assert config.save_models is False


def test_02_invalid_device_rejected() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(
            seeds=(1,),
            device="tpu",
            num_workers=0,
            batch_size=1,
            max_epochs=1,
            patience=1,
            learning_rate=0.1,
            weight_decay=0.0,
            sequence_length=2,
            sequence_stride=1,
            gradient_clip_norm=1.0,
            save_models=False,
            dataset_root=Path("data"),
            results_root=Path("results"),
            tasks={},
            classical_models={},
            neural_models={},
        )


@pytest.mark.parametrize("field", ["batch_size", "max_epochs", "patience", "sequence_length", "sequence_stride"])
def test_03_nonpositive_training_dimensions_rejected(field: str) -> None:
    values = {
        "seeds": (1,),
        "device": "cpu",
        "num_workers": 0,
        "batch_size": 1,
        "max_epochs": 1,
        "patience": 1,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "sequence_length": 2,
        "sequence_stride": 1,
        "gradient_clip_norm": 1.0,
        "save_models": False,
        "dataset_root": Path("data"),
        "results_root": Path("results"),
        "tasks": {},
        "classical_models": {},
        "neural_models": {},
        field: 0,
    }
    with pytest.raises(ValueError):
        TrainingConfig(**values)


def test_04_bundle_scaler_is_fit_on_train_only(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x")
    assert isinstance(bundle.scaler, StandardScaler)
    assert np.allclose(bundle.train.x.mean(axis=0), 0.0, atol=1e-6)


def test_05_bundle_preserves_feature_names(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x")
    assert bundle.train.feature_names == ("f0", "f1", "f2", "f3")


def test_06_early_bundle_has_distinct_schema(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x_early")
    assert bundle.train.x.shape[1] == 3


def test_07_sequences_do_not_cross_simulations(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x")
    sequences = make_sequences(bundle.train, length=4, stride=2)
    assert sequences.x.shape[1:] == (4, 4)
    assert len(sequences.metadata) == len(sequences.y)
    assert sequences.metadata["step"].isin({4, 6, 8}).all()


def test_08_invalid_sequence_dimensions_rejected(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x")
    with pytest.raises(ValueError):
        make_sequences(bundle.train, length=0, stride=1)


def test_09_sequence_label_matches_last_timestep() -> None:
    metadata = _metadata(1, 5)
    data = SplitData(
        x=np.arange(10, dtype=np.float32).reshape(5, 2),
        y=np.asarray([0, 1, 2, 3, 1]),
        metadata=metadata,
        feature_names=("a", "b"),
    )
    sequences = make_sequences(data, 3, 1)
    assert sequences.y.tolist() == [2, 3, 1]


def test_10_class_names_are_stable() -> None:
    assert CLASS_NAMES == ("normal", "double_spend", "selfish_mining", "majority_attack")


def test_11_perfect_metrics_equal_one() -> None:
    y = np.asarray([0, 1, 2, 3] * 3)
    probabilities = np.eye(4)[y]
    result = evaluate_multiclass(y, y, probabilities)
    assert result.aggregate["macro_f1"] == pytest.approx(1.0)
    assert result.aggregate["mcc"] == pytest.approx(1.0)


def test_12_confusion_matrix_is_four_by_four() -> None:
    y = np.asarray([0, 1, 2, 3])
    result = evaluate_multiclass(y, y, np.eye(4))
    assert result.confusion.shape == (4, 4)


def test_13_detection_delay_uses_attack_start() -> None:
    metadata = _metadata(4, 8)
    predictions = np.zeros(len(metadata), dtype=np.int64)
    for index, row in metadata.iterrows():
        configured = int(row["configured_attack_type"])
        if configured and int(row["step"]) >= 5:
            predictions[index] = configured
    delay = detection_delay_blocks(metadata, predictions)
    assert delay["detected_attack_scenarios"] == 3.0
    assert delay["mean_detection_delay_blocks"] == pytest.approx(2.0)


def test_14_detection_delay_accepts_textual_attack_names() -> None:
    metadata = _metadata(4, 8)
    names = {0: "NORMAL", 1: "DOUBLE_SPEND", 2: "SELFISH_MINING", 3: "MAJORITY_ATTACK"}
    metadata["configured_attack_type"] = metadata["configured_attack_type"].map(names)
    predictions = np.zeros(len(metadata), dtype=np.int64)
    for index, row in metadata.iterrows():
        attack_id = {value: key for key, value in names.items()}[row["configured_attack_type"]]
        if attack_id and int(row["step"]) >= 5:
            predictions[index] = attack_id
    delay = detection_delay_blocks(metadata, predictions)
    assert delay["detected_attack_scenarios"] == 3.0
    assert delay["mean_detection_delay_blocks"] == pytest.approx(2.0)


def test_15_logistic_regression_outputs_four_probabilities() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 4))
    y = np.tile(np.arange(4), 20)
    model = build_classical_model("logistic_regression", 42, {})
    model.fit(x, y)
    probabilities = predict_probabilities(model, x[:5])
    assert probabilities.shape == (5, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "hist_gradient_boosting"])
def test_15_supported_classical_models_build(name: str) -> None:
    assert build_classical_model(name, 42, {}) is not None


def test_16_unknown_classical_model_rejected() -> None:
    with pytest.raises(ValueError):
        build_classical_model("unknown", 42, {})


@pytest.mark.parametrize("name", ["mlp", "cnn1d", "lstm", "gru", "cnn_lstm", "transformer"])
def test_17_neural_forward_shape(name: str) -> None:
    settings = {"hidden_dim": 16, "num_layers": 1, "d_model": 16, "nhead": 4, "dim_feedforward": 32, "dropout": 0.0}
    model = build_neural_model(name, 5, 8, settings)
    x = torch.randn(3, 5) if name == "mlp" else torch.randn(3, 8, 5)
    assert model(x).shape == (3, 4)


def test_18_parameter_count_is_positive() -> None:
    model = build_neural_model("mlp", 5, 8, {"hidden_dim": 8, "dropout": 0.0})
    assert parameter_count(model) > 0


def test_19_unknown_neural_model_rejected() -> None:
    with pytest.raises(ValueError):
        build_neural_model("unknown", 5, 8, {})


def test_20_cpu_device_resolves() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_21_cuda_request_fails_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this runtime.")
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_22_seed_everything_repeats_numpy_draws() -> None:
    seed_everything(123)
    first = np.random.random(4)
    seed_everything(123)
    second = np.random.random(4)
    assert np.array_equal(first, second)


def test_23_neural_fit_and_predict_cpu() -> None:
    seed_everything(3)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    y = np.tile(np.arange(4), 16).astype(np.int64)
    model = build_neural_model("mlp", 4, 4, {"hidden_dim": 8, "dropout": 0.0})
    result = fit_neural(
        model,
        x,
        y,
        x,
        y,
        device=torch.device("cpu"),
        batch_size=16,
        num_workers=0,
        max_epochs=1,
        patience=1,
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        seed=3,
    )
    probabilities, latency = predict_neural(
        model,
        x[:8],
        device=torch.device("cpu"),
        batch_size=4,
        num_workers=0,
    )
    assert result.best_epoch == 1
    assert probabilities.shape == (8, 4)
    assert latency >= 0.0


def test_24_summary_and_aggregation_are_canonical(tmp_path: Path) -> None:
    summary = tmp_path / "runs.csv"
    aggregate = tmp_path / "aggregate.csv"
    rows = [
        {"task": "general", "model": "mlp", "seed": 1, "macro_f1": 0.5},
        {"task": "general", "model": "mlp", "seed": 2, "macro_f1": 0.7},
    ]
    write_summary_table(rows, summary)
    table = aggregate_runs(summary, aggregate)
    assert table["macro_f1_mean"].iloc[0] == pytest.approx(0.6)
    assert aggregate.is_file()


def test_25_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)


def test_26_manifest_is_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, {"z": 1, "a": 2})
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["a", "z"]



def test_27_endpoint_view_matches_sequence_endpoints(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x")
    aligned = make_aligned_split_views(bundle.test, length=4, stride=2)
    assert len(aligned.endpoints.y) == len(aligned.sequences.y)
    assert np.array_equal(aligned.endpoints.y, aligned.sequences.y)
    assert aligned.endpoints.metadata["sample_id"].tolist() == aligned.sequences.metadata[
        "sample_id"
    ].tolist()


def test_28_tabular_endpoint_features_match_final_sequence_timestep(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x")
    aligned = make_aligned_split_views(bundle.test, length=4, stride=2)
    assert np.array_equal(aligned.endpoints.x, aligned.sequences.x[:, -1, :])


def test_29_all_splits_have_common_endpoint_and_sequence_counts(tiny_dataset: Path) -> None:
    bundle = load_dataset_bundle(tiny_dataset, "x")
    views = make_aligned_dataset_views(bundle, length=4, stride=2)
    for split in (views.train, views.validation, views.test):
        assert len(split.endpoints.y) == len(split.sequences.y)
        assert split.endpoints.metadata["sample_id"].tolist() == split.sequences.metadata[
            "sample_id"
        ].tolist()


def test_30_general_and_early_views_share_identical_logical_endpoints(tiny_dataset: Path) -> None:
    general = load_dataset_bundle(tiny_dataset, "x")
    early = load_dataset_bundle(tiny_dataset, "x_early")
    for general_split, early_split in (
        (general.train, early.train),
        (general.validation, early.validation),
        (general.test, early.test),
    ):
        general_endpoints = make_endpoint_view(general_split, length=4, stride=2)
        early_endpoints = make_endpoint_view(early_split, length=4, stride=2)
        assert np.array_equal(general_endpoints.y, early_endpoints.y)
        assert general_endpoints.metadata["sample_id"].tolist() == early_endpoints.metadata[
            "sample_id"
        ].tolist()


def test_31_smoke_run_records_identical_evaluation_support_for_all_models(
    tiny_dataset: Path,
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        f"""training:
  base_seed: 2027
  seeds: [42]
  device: "cpu"
  num_workers: 0
  batch_size: 16
  max_epochs: 1
  patience: 1
  learning_rate: 0.001
  weight_decay: 0.0
  sequence_length: 4
  sequence_stride: 2
  gradient_clip_norm: 1.0
  save_models: false

tasks:
  general:
    x_prefix: "x"

classical_models:
  logistic_regression:
    enabled: true

neural_models:
  mlp:
    enabled: true
    hidden_dim: 8
    dropout: 0.0
  transformer:
    enabled: true
    d_model: 8
    nhead: 2
    num_layers: 1
    dim_feedforward: 16
    dropout: 0.0

paths:
  dataset_root: "{tiny_dataset.as_posix()}"
  results_root: "{results_root.as_posix()}"
""",
        encoding="utf-8",
    )
    run_experiments(config_path, "smoke")
    runs = pd.read_csv(results_root / "metrics" / "run_metrics.csv")
    assert len(runs) == 3
    assert runs["test_samples"].nunique() == 1
    assert runs["train_samples"].nunique() == 1
    assert runs["validation_samples"].nunique() == 1
    assert runs["evaluation_sample_id_sha256"].nunique() == 1

    manifest = json.loads(
        (results_root / "manifests" / "training_manifest.json").read_text(encoding="utf-8")
    )
    audit = manifest["aligned_endpoints"]["general"]
    assert audit["test_samples"] == int(runs["test_samples"].iloc[0])
    assert audit["test_sample_id_sha256"] == runs["evaluation_sample_id_sha256"].iloc[0]


def test_32_feature_ablation_preserves_samples_and_removes_only_requested_features(
    tiny_dataset: Path,
) -> None:
    """Feature ablation must change only the requested input dimensions."""

    for split in ("train", "validation", "test"):
        x_path = tiny_dataset / "splits" / f"x_{split}.csv"
        x = pd.read_csv(x_path)
        x["dominant_miner_ratio"] = np.linspace(0.1, 0.9, len(x))
        x["miner_hhi"] = np.linspace(0.2, 0.8, len(x))
        x.to_csv(x_path, index=False)

    full = load_dataset_bundle(tiny_dataset, "x")
    ablated = load_dataset_bundle(
        tiny_dataset,
        "x",
        drop_features=("dominant_miner_ratio", "miner_hhi"),
    )

    removed = {"dominant_miner_ratio", "miner_hhi"}

    assert set(full.train.feature_names) - set(ablated.train.feature_names) == removed
    assert len(ablated.train.feature_names) == len(full.train.feature_names) - len(removed)

    for full_split, ablated_split in (
        (full.train, ablated.train),
        (full.validation, ablated.validation),
        (full.test, ablated.test),
    ):
        assert len(full_split.y) == len(ablated_split.y)
        assert np.array_equal(full_split.y, ablated_split.y)
        assert full_split.metadata.equals(ablated_split.metadata)


def test_33_external_split_uses_id_scaler_without_refitting(
    tiny_dataset: Path,
) -> None:
    """External evaluation must reuse the scaler fitted on ID training data."""

    bundle = load_dataset_bundle(tiny_dataset, "x")

    external = load_external_split(
        tiny_dataset,
        "x",
        "test",
        bundle.scaler,
        bundle.train.feature_names,
    )

    raw_test = pd.read_csv(tiny_dataset / "splits" / "x_test.csv")
    expected = bundle.scaler.transform(
        raw_test.to_numpy(dtype=np.float32)
    ).astype(np.float32)

    assert np.allclose(external.x, expected)
    assert external.feature_names == bundle.train.feature_names
    assert np.array_equal(
        external.y,
        pd.read_csv(tiny_dataset / "splits" / "y_test.csv")[
            "attack_label"
        ].to_numpy(dtype=np.int64),
    )

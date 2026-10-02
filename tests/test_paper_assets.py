from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.generate_paper_assets import (
    EXPECTED_MODELS,
    EXPECTED_SEEDS,
    build_general_vs_early,
    build_pairwise_tests,
    build_top_class_table,
    top_models,
    validate_inputs,
)


def _load_results(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = pd.read_csv(project_root / "results" / "metrics" / "run_metrics.csv")
    per_class = pd.read_csv(project_root / "results" / "metrics" / "per_class_metrics.csv")
    return run, per_class


def test_frozen_run_grid_is_complete(project_root: Path = Path(".")) -> None:
    run, per_class = _load_results(project_root)
    audit = validate_inputs(run, per_class)
    assert audit["run_count"] == 100
    assert audit["seeds"] == list(EXPECTED_SEEDS)
    assert len(audit["models"]) == len(EXPECTED_MODELS)


def test_all_runs_share_one_evaluation_hash(project_root: Path = Path(".")) -> None:
    run, per_class = _load_results(project_root)
    audit = validate_inputs(run, per_class)
    assert len(run["evaluation_sample_id_sha256"].unique()) == 1
    assert audit["evaluation_sample_id_sha256"] == run["evaluation_sample_id_sha256"].iloc[0]


def test_aligned_split_sizes_are_constant(project_root: Path = Path(".")) -> None:
    run, per_class = _load_results(project_root)
    audit = validate_inputs(run, per_class)
    assert audit["train_samples"] == 9540
    assert audit["validation_samples"] == 1980
    assert audit["test_samples"] == 1980


def test_test_class_supports_sum_to_test_size(project_root: Path = Path(".")) -> None:
    run, per_class = _load_results(project_root)
    audit = validate_inputs(run, per_class)
    assert sum(audit["test_class_counts"].values()) == audit["test_samples"]


def test_macro_f1_winners_match_frozen_results(project_root: Path = Path(".")) -> None:
    run, _ = _load_results(project_root)
    winners = top_models(run)
    assert winners == {"general": "gru", "early": "cnn_lstm"}


def test_top_class_table_has_four_classes_per_task(project_root: Path = Path(".")) -> None:
    run, per_class = _load_results(project_root)
    winners = top_models(run)
    table = build_top_class_table(per_class, winners)
    assert len(table) == 8
    assert table.groupby("task")["class_id"].nunique().to_dict() == {"early": 4, "general": 4}


def test_general_vs_early_table_is_model_complete(project_root: Path = Path(".")) -> None:
    run, _ = _load_results(project_root)
    table = build_general_vs_early(run)
    assert set(table["model"]) == set(EXPECTED_MODELS)
    assert len(table) == len(EXPECTED_MODELS)
    assert table["macro_f1_wilcoxon_p_holm"].between(0.0, 1.0).all()


def test_pairwise_statistics_are_complete(project_root: Path = Path(".")) -> None:
    run, _ = _load_results(project_root)
    winners = top_models(run)
    table = build_pairwise_tests(run, winners)
    assert len(table) == 18
    assert table["p_holm"].between(0.0, 1.0).all()

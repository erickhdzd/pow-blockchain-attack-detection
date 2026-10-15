from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Buffer, Iterable
from pathlib import Path
from typing import TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EXPECTED_TASKS = ("general", "early")
EXPECTED_MODELS = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "xgboost",
    "mlp",
    "cnn1d",
    "lstm",
    "gru",
    "cnn_lstm",
    "transformer",
)
EXPECTED_SEEDS = (42, 123, 2026, 2027, 2028)
EXPECTED_CLASSES = {
    0: "normal",
    1: "double_spend",
    2: "selfish_mining",
    3: "majority_attack",
}
PRIMARY_METRICS = (
    "macro_f1",
    "macro_pr_auc",
    "mcc",
    "balanced_accuracy",
    "scenario_detection_rate",
)
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "cnn1d": "1D-CNN",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN-LSTM",
    "transformer": "Transformer",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(cast(Buffer, chunk))
    return digest.hexdigest()


def sha256_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(cast(Buffer, value.encode("utf-8")))
        digest.update(cast(Buffer, b"\n"))
    return digest.hexdigest()


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving input order."""
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, raw_index in enumerate(order):
        index = int(raw_index)
        candidate = (m - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.allclose(delta, 0.0):
        return 0.0, 1.0
    result = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


class AuditResult(TypedDict):
    run_count: int
    tasks: list[str]
    models: list[str]
    seeds: list[int]
    evaluation_sample_id_sha256: str
    train_samples: int
    validation_samples: int
    test_samples: int
    test_class_counts: dict[str, int]


def validate_inputs(run: pd.DataFrame, per_class: pd.DataFrame) -> AuditResult:
    required_run = {
        "task",
        "model",
        "seed",
        "macro_f1",
        "macro_pr_auc",
        "mcc",
        "balanced_accuracy",
        "scenario_detection_rate",
        "mean_detection_delay_blocks",
        "training_seconds",
        "inference_ms_per_sample",
        "parameter_count",
        "peak_gpu_memory_mb",
        "train_samples",
        "validation_samples",
        "test_samples",
        "evaluation_sample_id_sha256",
    }
    required_class = {
        "task",
        "model",
        "seed",
        "class_id",
        "class_name",
        "precision",
        "recall",
        "f1",
        "support",
        "false_positive_rate",
        "detection_rate",
    }
    missing_run = required_run - set(run.columns)
    missing_class = required_class - set(per_class.columns)
    if missing_run or missing_class:
        raise ValueError(
            f"Missing required columns: run={sorted(missing_run)}, per_class={sorted(missing_class)}"
        )

    if len(run) != 100:
        raise ValueError(f"Expected 100 runs, found {len(run)}.")
    if run[["task", "model", "seed"]].duplicated().any():
        raise ValueError("Duplicate task/model/seed combinations were found.")
    if run.isna().any().any():
        raise ValueError("run_metrics.csv contains missing values.")
    if per_class.isna().any().any():
        raise ValueError("per_class_metrics.csv contains missing values.")

    tasks = tuple(sorted(run["task"].unique()))
    if set(tasks) != set(EXPECTED_TASKS):
        raise ValueError(f"Unexpected tasks: {tasks}.")
    if set(run["model"].unique()) != set(EXPECTED_MODELS):
        raise ValueError("The model set does not match the frozen experiment.")
    if {int(seed) for seed in run["seed"].unique()} != set(EXPECTED_SEEDS):
        raise ValueError("The seed set does not match the frozen experiment.")

    expected_pairs = {
        (task, model, seed)
        for task in EXPECTED_TASKS
        for model in EXPECTED_MODELS
        for seed in EXPECTED_SEEDS
    }
    observed_pairs = set(
        run[["task", "model", "seed"]].itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs:
        raise ValueError(
            "The run grid is incomplete or contains unexpected combinations."
        )

    hashes = run["evaluation_sample_id_sha256"].unique().tolist()
    if len(hashes) != 1:
        raise ValueError(
            "Models/tasks do not share one common evaluation endpoint hash."
        )

    split_sizes = {}
    for field in ("train_samples", "validation_samples", "test_samples"):
        values = sorted(int(value) for value in run[field].unique())
        if len(values) != 1:
            raise ValueError(f"{field} is not identical across all runs: {values}")
        split_sizes[field] = values[0]

    class_supports: dict[int, int] = {}
    for class_id, expected_name in EXPECTED_CLASSES.items():
        subset = per_class[per_class["class_id"] == class_id]
        if subset.empty:
            raise ValueError(f"Missing class {class_id}.")
        if set(subset["class_name"].unique()) != {expected_name}:
            raise ValueError(f"Unexpected name for class {class_id}.")
        supports = sorted(int(value) for value in subset["support"].unique())
        if len(supports) != 1:
            raise ValueError(f"Support differs for class {class_id}: {supports}")
        class_supports[class_id] = supports[0]
    if sum(class_supports.values()) != split_sizes["test_samples"]:
        raise ValueError("Per-class supports do not sum to the aligned test size.")

    return {
        "run_count": len(run),
        "tasks": list(EXPECTED_TASKS),
        "models": list(EXPECTED_MODELS),
        "seeds": list(EXPECTED_SEEDS),
        "evaluation_sample_id_sha256": hashes[0],
        "train_samples": split_sizes["train_samples"],
        "validation_samples": split_sizes["validation_samples"],
        "test_samples": split_sizes["test_samples"],
        "test_class_counts": {str(k): v for k, v in class_supports.items()},
    }


def aggregate_model_table(run: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "mcc",
        "macro_roc_auc",
        "macro_pr_auc",
        "scenario_detection_rate",
        "mean_detection_delay_blocks",
        "training_seconds",
        "inference_ms_per_sample",
        "parameter_count",
        "peak_gpu_memory_mb",
    ]
    available: list[str] = [metric for metric in metrics if metric in run.columns]
    aggregations: dict[str, list[str]] = {
        metric: ["mean", "std"] for metric in available
    }
    agg = run.groupby(["task", "model"], sort=True).agg(aggregations)
    flattened_columns: list[str] = []
    for column in agg.columns:
        if isinstance(column, tuple) and len(column) == 2:
            metric, stat = column
            flattened_columns.append(f"{metric!s}_{stat!s}")
        else:
            flattened_columns.append(str(column))
    agg.columns = flattened_columns
    table = agg.reset_index()
    table.insert(2, "display_name", table["model"].map(DISPLAY_NAMES))
    return table.sort_values(
        ["task", "macro_f1_mean"], ascending=[True, False]
    ).reset_index(drop=True)


def top_models(run: pd.DataFrame) -> dict[str, str]:
    means = run.groupby(["task", "model"], as_index=False)["macro_f1"].mean()
    return {
        task: str(
            means[means["task"] == task]
            .sort_values("macro_f1", ascending=False)
            .iloc[0]["model"]
        )
        for task in EXPECTED_TASKS
    }


def build_top_class_table(
    per_class: pd.DataFrame, winners: dict[str, str]
) -> pd.DataFrame:
    rows = []
    for task, model in winners.items():
        subset = per_class[(per_class["task"] == task) & (per_class["model"] == model)]
        agg = subset.groupby(["class_id", "class_name"], as_index=False).agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            false_positive_rate_mean=("false_positive_rate", "mean"),
            detection_rate_mean=("detection_rate", "mean"),
            support=("support", "first"),
        )
        agg.insert(0, "task", task)
        agg.insert(1, "model", model)
        agg.insert(2, "display_name", DISPLAY_NAMES[model])
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def build_general_vs_early(run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in EXPECTED_MODELS:
        general = run[(run["task"] == "general") & (run["model"] == model)].sort_values(
            "seed"
        )
        early = run[(run["task"] == "early") & (run["model"] == model)].sort_values(
            "seed"
        )
        if general["seed"].tolist() != early["seed"].tolist():
            raise ValueError(f"Seed alignment failed for {model}.")
        row: dict[str, object] = {"model": model, "display_name": DISPLAY_NAMES[model]}
        for metric in PRIMARY_METRICS:
            row[f"general_{metric}_mean"] = float(general[metric].mean())
            row[f"early_{metric}_mean"] = float(early[metric].mean())
            row[f"early_minus_general_{metric}"] = float(
                early[metric].mean() - general[metric].mean()
            )
        stat, p_value = safe_wilcoxon(
            early["macro_f1"].to_numpy(), general["macro_f1"].to_numpy()
        )
        row["macro_f1_wilcoxon_statistic"] = stat
        row["macro_f1_wilcoxon_p_value"] = p_value
        rows.append(row)
    result = pd.DataFrame(rows)
    result["macro_f1_wilcoxon_p_holm"] = holm_adjust(
        result["macro_f1_wilcoxon_p_value"].tolist()
    )
    result["statistical_note"] = (
        "Exploratory paired test across five seeds; interpret cautiously."
    )
    return result.sort_values("early_macro_f1_mean", ascending=False).reset_index(
        drop=True
    )


def build_pairwise_tests(run: pd.DataFrame, winners: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in EXPECTED_TASKS:
        winner = winners[task]
        winner_runs = run[(run["task"] == task) & (run["model"] == winner)].sort_values(
            "seed"
        )
        task_rows = []
        for model in EXPECTED_MODELS:
            if model == winner:
                continue
            other = run[(run["task"] == task) & (run["model"] == model)].sort_values(
                "seed"
            )
            stat, p_value = safe_wilcoxon(
                winner_runs["macro_f1"].to_numpy(), other["macro_f1"].to_numpy()
            )
            task_rows.append(
                {
                    "task": task,
                    "best_model": winner,
                    "best_display_name": DISPLAY_NAMES[winner],
                    "comparison_model": model,
                    "comparison_display_name": DISPLAY_NAMES[model],
                    "best_macro_f1_mean": float(winner_runs["macro_f1"].mean()),
                    "comparison_macro_f1_mean": float(other["macro_f1"].mean()),
                    "mean_difference": float(
                        winner_runs["macro_f1"].mean() - other["macro_f1"].mean()
                    ),
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                }
            )
        adjusted = holm_adjust([cast(float, row["p_value"]) for row in task_rows])
        for row, p_holm in zip(task_rows, adjusted, strict=True):
            row["p_holm"] = p_holm
            row["statistical_note"] = (
                "Exploratory paired comparison across five seeds; Holm-corrected within task."
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_efficiency_table(run: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "macro_f1",
        "macro_pr_auc",
        "mcc",
        "training_seconds",
        "inference_ms_per_sample",
        "parameter_count",
        "peak_gpu_memory_mb",
    ]
    aggregations: dict[str, str] = {column: "mean" for column in columns}
    table = run.groupby(["task", "model"], as_index=False).agg(aggregations)
    table.insert(2, "display_name", table["model"].map(DISPLAY_NAMES))
    return table.sort_values(["task", "macro_f1"], ascending=[True, False]).reset_index(
        drop=True
    )


def save_macro_f1_figure(run: pd.DataFrame, output: Path) -> None:
    summary = (
        run.groupby(["model", "task"])["macro_f1"].agg(["mean", "std"]).reset_index()
    )
    models = sorted(
        EXPECTED_MODELS,
        key=lambda m: run[(run["task"] == "general") & (run["model"] == m)][
            "macro_f1"
        ].mean(),
        reverse=True,
    )
    x = np.arange(len(models), dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    offsets = {"general": -0.14, "early": 0.14}
    markers = {"general": "o", "early": "s"}
    for task in EXPECTED_TASKS:
        means = []
        stds = []
        for model in models:
            row = summary[(summary["model"] == model) & (summary["task"] == task)].iloc[
                0
            ]
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
        ax.errorbar(
            x + offsets[task],
            means,
            yerr=stds,
            fmt=markers[task],
            capsize=3,
            label=task.capitalize(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=35, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Model")
    ax.set_title("Aligned-endpoint performance: General vs Early")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_stability_figure(run: pd.DataFrame, output: Path) -> None:
    order = sorted(
        EXPECTED_MODELS,
        key=lambda m: run[run["model"] == m]["macro_f1"].mean(),
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    positions = np.arange(len(order), dtype=float)
    for task, offset, marker in (("general", -0.13, "o"), ("early", 0.13, "s")):
        for i, model in enumerate(order):
            values = (
                run[(run["task"] == task) & (run["model"] == model)]
                .sort_values("seed")["macro_f1"]
                .to_numpy()
            )
            ax.scatter(
                np.full(len(values), positions[i] + offset),
                values,
                marker=marker,
                s=24,
                alpha=0.8,
            )
        task_means = [
            run[(run["task"] == task) & (run["model"] == m)]["macro_f1"].mean()
            for m in order
        ]
        ax.plot(
            positions + offset,
            task_means,
            marker=marker,
            linewidth=1.0,
            label=task.capitalize(),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in order], rotation=35, ha="right")
    ax.set_ylabel("Macro-F1 across seeds")
    ax.set_xlabel("Model")
    ax.set_title("Five-seed stability on common evaluation endpoints")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_class_figure(
    per_class: pd.DataFrame, winners: dict[str, str], output: Path
) -> None:
    labels = ["Normal", "Double Spend", "Selfish Mining", "Majority / 51%"]
    x = np.arange(4, dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    offsets = {"general": -0.14, "early": 0.14}
    markers = {"general": "o", "early": "s"}
    for task in EXPECTED_TASKS:
        model = winners[task]
        subset = per_class[(per_class["task"] == task) & (per_class["model"] == model)]
        summary = (
            subset.groupby("class_id")["f1"].agg(["mean", "std"]).reindex(range(4))
        )
        ax.errorbar(
            x + offsets[task],
            summary["mean"].to_numpy(),
            yerr=summary["std"].to_numpy(),
            fmt=markers[task],
            capsize=3,
            label=f"{task.capitalize()}: {DISPLAY_NAMES[model]}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Per-class F1")
    ax.set_title("Best Macro-F1 model in each task")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_efficiency_figure(run: pd.DataFrame, output: Path) -> None:
    summary = run.groupby(["task", "model"], as_index=False).agg(
        macro_f1=("macro_f1", "mean"),
        inference_ms_per_sample=("inference_ms_per_sample", "mean"),
        parameter_count=("parameter_count", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    for task, marker in (("general", "o"), ("early", "s")):
        subset = summary[summary["task"] == task]
        x = np.maximum(subset["inference_ms_per_sample"].to_numpy(dtype=float), 1e-6)
        y = subset["macro_f1"].to_numpy(dtype=float)
        sizes = 20.0 + 80.0 * np.sqrt(
            np.maximum(subset["parameter_count"].to_numpy(dtype=float), 0.0)
            / max(float(summary["parameter_count"].max()), 1.0)
        )
        ax.scatter(x, y, s=sizes, marker=marker, alpha=0.75, label=task.capitalize())
        for xx, yy, model in zip(x, y, subset["model"], strict=True):
            if model in {"gru", "cnn_lstm", "lstm", "transformer", "random_forest"}:
                ax.annotate(
                    DISPLAY_NAMES[model],
                    (xx, yy),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency (ms/sample, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Performance-efficiency trade-off")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_latex(
    df: pd.DataFrame, path: Path, float_format: str = "%.4f"
) -> None:
    path.write_text(
        df.to_latex(index=False, escape=True, float_format=float_format),
        encoding="utf-8",
    )


def write_readme(path: Path, winners: dict[str, str], audit: AuditResult) -> None:
    text = f"""# Paper assets\n\nThis directory was generated automatically from the frozen full-training outputs.\nNo model was retrained and no metric was entered manually.\n\n## Validated experiment\n\n- Runs: {audit["run_count"]}\n- Tasks: {", ".join(audit["tasks"])}\n- Models: 10\n- Seeds: {", ".join(str(seed) for seed in audit["seeds"])}\n- Train endpoints: {audit["train_samples"]}\n- Validation endpoints: {audit["validation_samples"]}\n- Test endpoints: {audit["test_samples"]}\n- Evaluation sample SHA-256: `{audit["evaluation_sample_id_sha256"]}`\n- General Macro-F1 winner: {DISPLAY_NAMES[winners["general"]]}\n- Early Macro-F1 winner: {DISPLAY_NAMES[winners["early"]]}\n\n## Statistical interpretation\n\nWilcoxon signed-rank tests are paired by seed and Holm-adjusted within each comparison family.\nBecause there are only five seeds, these tests are exploratory and should not be treated as strong inferential evidence.\nEffect magnitudes, seed stability, class-level metrics, and confidence limitations should be discussed alongside p-values.\n\n## Files\n\n- `tables/model_comparison_paper.csv`: model-level mean and standard deviation metrics.\n- `tables/top_models_per_class.csv`: class-level metrics for the best Macro-F1 model in each task.\n- `tables/general_vs_early.csv`: paired General/Early comparison.\n- `tables/efficiency.csv`: performance and efficiency summary.\n- `statistics/best_vs_others_macro_f1.csv`: exploratory paired Wilcoxon tests against each task winner.\n- `figures/macro_f1_general_vs_early.png`: main aligned-endpoint performance figure.\n- `figures/macro_f1_seed_stability.png`: five-seed stability figure.\n- `figures/top_models_per_class_f1.png`: per-class F1 for the two task winners.\n- `figures/performance_efficiency.png`: performance/latency/parameter trade-off.\n- `figures/general__gru__confusion.png` and `figures/early__cnn_lstm__confusion.png`: selected aggregate confusion matrices from training outputs.\n- `manifest.json`: source and generated-file hashes.\n"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible paper assets from frozen training results."
    )
    parser.add_argument(
        "--results-root", default="results", help="Training results root directory."
    )
    parser.add_argument(
        "--output-root",
        default="results/paper_assets",
        help="Output directory for paper assets.",
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
        help="Frozen training configuration path.",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="data/manifests/dataset_manifest.json",
        help="Dataset manifest path.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    statistics_dir = output_root / "statistics"
    for directory in (tables_dir, figures_dir, statistics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_path = results_root / "metrics" / "run_metrics.csv"
    per_class_path = results_root / "metrics" / "per_class_metrics.csv"
    training_manifest_path = results_root / "manifests" / "training_manifest.json"
    source_paths = [
        run_path,
        per_class_path,
        training_manifest_path,
        Path(args.config),
        Path(args.dataset_manifest),
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    run = pd.read_csv(str(run_path))
    per_class = pd.read_csv(str(per_class_path))
    audit = validate_inputs(run, per_class)
    winners = top_models(run)

    model_table = aggregate_model_table(run)
    top_class_table = build_top_class_table(per_class, winners)
    general_vs_early = build_general_vs_early(run)
    pairwise = build_pairwise_tests(run, winners)
    efficiency = build_efficiency_table(run)

    model_table.to_csv(tables_dir / "model_comparison_paper.csv", index=False)
    top_class_table.to_csv(tables_dir / "top_models_per_class.csv", index=False)
    general_vs_early.to_csv(tables_dir / "general_vs_early.csv", index=False)
    efficiency.to_csv(tables_dir / "efficiency.csv", index=False)
    pairwise.to_csv(statistics_dir / "best_vs_others_macro_f1.csv", index=False)

    key_columns = [
        "task",
        "display_name",
        "macro_f1_mean",
        "macro_f1_std",
        "macro_pr_auc_mean",
        "macro_pr_auc_std",
        "mcc_mean",
        "mcc_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "scenario_detection_rate_mean",
        "scenario_detection_rate_std",
        "mean_detection_delay_blocks_mean",
        "training_seconds_mean",
        "inference_ms_per_sample_mean",
        "parameter_count_mean",
        "peak_gpu_memory_mb_mean",
    ]
    dataframe_to_latex(
        model_table[key_columns], tables_dir / "model_comparison_paper.tex"
    )
    dataframe_to_latex(top_class_table, tables_dir / "top_models_per_class.tex")

    save_macro_f1_figure(run, figures_dir / "macro_f1_general_vs_early.png")
    save_stability_figure(run, figures_dir / "macro_f1_seed_stability.png")
    save_per_class_figure(
        per_class, winners, figures_dir / "top_models_per_class_f1.png"
    )
    save_efficiency_figure(run, figures_dir / "performance_efficiency.png")

    selected_confusions = {
        "general": winners["general"],
        "early": winners["early"],
    }
    for task, model in selected_confusions.items():
        source = results_root / "figures" / f"{task}__{model}__confusion.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, figures_dir / source.name)

    audit_path = output_root / "audit.json"
    audit_payload = {
        **audit,
        "macro_f1_winners": winners,
        "source_grid_sha256": sha256_values(
            "|".join(
                (
                    str(task),
                    str(model),
                    str(seed),
                    str(evaluation_hash),
                )
            )
            for task, model, seed, evaluation_hash in (
                run.sort_values(["task", "model", "seed"])
                .loc[:, ["task", "model", "seed", "evaluation_sample_id_sha256"]]
                .itertuples(index=False, name=None)
            )
        ),
    }
    audit_path.write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_readme(output_root / "README.md", winners, audit)

    generated_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "generator": "generate_paper_assets.py",
        "sources": {str(path): sha256_file(path) for path in source_paths},
        "generated_files": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in generated_files
        },
        "audit": audit_payload,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(audit_payload, indent=2, sort_keys=True))
    print(f"Paper assets: {output_root}")


if __name__ == "__main__":
    main()

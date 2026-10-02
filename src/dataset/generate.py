"""Command-line entry point for reproducible dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.builder import DatasetBuilder
from src.dataset.config import load_experiment_plan


def main() -> None:
    """Generate the configured experimental dataset."""

    parser = argparse.ArgumentParser(
        description="Generate reproducible PoW attack-detection datasets."
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Project-root-relative experiment YAML path.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for the dataset output root.",
    )
    args = parser.parse_args()

    plan = load_experiment_plan(Path(args.config))
    result = DatasetBuilder(plan).build(args.output_root)

    print(f"Scenarios: {result.scenario_count}")
    print(f"Samples: {result.sample_count}")
    for split, count in result.split_sample_counts.items():
        print(f"{split}: {count}")
    print(f"Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()

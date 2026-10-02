"""Verify runtime dependencies and the generated dataset before training."""
from __future__ import annotations

import json
import platform

import numpy
import pandas
import sklearn
import torch
import xgboost
import yaml

from src.training.config import load_training_config


def main() -> None:
    config = load_training_config("configs/training.yaml")
    required = [
        config.dataset_root / "splits" / "x_train.csv",
        config.dataset_root / "splits" / "x_early_train.csv",
        config.dataset_root / "splits" / "y_train.csv",
        config.dataset_root / "splits" / "metadata_train.csv",
        config.dataset_root / "manifests" / "dataset_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing dataset files: " + ", ".join(missing))
    payload = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "pyyaml": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "dataset": "ready",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Classical machine-learning baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_classical_model(name: str, seed: int, settings: dict[str, Any]) -> Any:
    """Construct one seeded baseline classifier."""

    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(settings.get("n_estimators", 300)),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            class_weight="balanced",
            learning_rate=0.08,
            random_state=seed,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is required for the xgboost baseline.") from exc
        return XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            n_estimators=int(settings.get("n_estimators", 300)),
            max_depth=int(settings.get("max_depth", 6)),
            learning_rate=float(settings.get("learning_rate", 0.05)),
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classical model: {name!r}.")


def predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    """Return four-class probability estimates."""

    probabilities = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probabilities.shape[1] != 4:
        raise ValueError("Expected four-class probability output.")
    return probabilities

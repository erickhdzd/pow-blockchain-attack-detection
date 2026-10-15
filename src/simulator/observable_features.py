from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.simulator.chain_observer import ChainObservation
from src.simulator.telemetry import TelemetrySnapshot


@dataclass(slots=True, frozen=True)
class ObservableFeatureVector:
    """Store model-safe features derived only from observable blockchain state."""

    timestamp: float
    canonical_height: int

    mean_height: float
    min_height: int
    max_height: int
    height_std: float
    height_range: int

    unique_tip_count: int
    consensus_ratio: float

    reference_height: int

    tip_age_seconds: float
    latest_block_interval_seconds: float
    mean_recent_block_interval_seconds: float
    std_recent_block_interval_seconds: float

    unique_miner_count: int
    dominant_miner_ratio: float
    miner_hhi: float

    blocks_added: int
    blocks_removed: int
    common_ancestor_height: int
    reorganization_detected: bool
    reorganization_depth: int
    tip_changed: bool

    def __post_init__(self) -> None:
        self._validate_non_negative_finite(
            self.timestamp,
            "Feature timestamp",
        )

        if self.canonical_height < 0:
            raise ValueError("Canonical height must be non-negative.")

        self._validate_non_negative_finite(
            self.mean_height,
            "Mean height",
        )

        if self.min_height < 0:
            raise ValueError("Minimum height must be non-negative.")

        if self.max_height < 0:
            raise ValueError("Maximum height must be non-negative.")

        if self.min_height > self.max_height:
            raise ValueError("Minimum height must not exceed maximum height.")

        self._validate_non_negative_finite(
            self.height_std,
            "Height standard deviation",
        )

        if self.height_range < 0:
            raise ValueError("Height range must be non-negative.")

        if self.height_range != self.max_height - self.min_height:
            raise ValueError(
                "Height range must equal maximum height minus minimum height."
            )

        if self.unique_tip_count <= 0:
            raise ValueError("Unique tip count must be positive.")

        self._validate_probability(
            self.consensus_ratio,
            "Consensus ratio",
        )

        if self.reference_height < 0:
            raise ValueError("Reference height must be non-negative.")

        if not self.min_height <= self.reference_height <= self.max_height:
            raise ValueError(
                "Reference height must be within the network height range."
            )

        if self.canonical_height != self.reference_height:
            raise ValueError("Canonical height must equal reference height.")

        self._validate_non_negative_finite(
            self.tip_age_seconds,
            "Tip age",
        )

        self._validate_non_negative_finite(
            self.latest_block_interval_seconds,
            "Latest block interval",
        )

        self._validate_non_negative_finite(
            self.mean_recent_block_interval_seconds,
            "Mean recent block interval",
        )

        self._validate_non_negative_finite(
            self.std_recent_block_interval_seconds,
            "Recent block-interval standard deviation",
        )

        if self.unique_miner_count < 0:
            raise ValueError("Unique miner count must be non-negative.")

        self._validate_probability(
            self.dominant_miner_ratio,
            "Dominant miner ratio",
        )

        self._validate_probability(
            self.miner_hhi,
            "Miner HHI",
        )

        if self.canonical_height == 0:
            if self.unique_miner_count != 0:
                raise ValueError("Genesis-only feature vectors must have zero miners.")

            if self.dominant_miner_ratio != 0.0:
                raise ValueError(
                    "Genesis-only feature vectors must have zero dominant miner ratio."
                )

            if self.miner_hhi != 0.0:
                raise ValueError(
                    "Genesis-only feature vectors must have zero miner HHI."
                )

        elif self.unique_miner_count == 0:
            raise ValueError(
                "Non-genesis feature vectors must contain at least one miner."
            )

        if self.blocks_added < 0:
            raise ValueError("Added block count must be non-negative.")

        if self.blocks_removed < 0:
            raise ValueError("Removed block count must be non-negative.")

        if self.common_ancestor_height < 0:
            raise ValueError("Common ancestor height must be non-negative.")

        if self.common_ancestor_height > self.canonical_height:
            raise ValueError("Common ancestor height must not exceed canonical height.")

        if self.reorganization_depth < 0:
            raise ValueError("Reorganization depth must be non-negative.")

        if self.reorganization_detected and self.reorganization_depth == 0:
            raise ValueError("Detected reorganizations must have positive depth.")

        if not self.reorganization_detected and self.reorganization_depth != 0:
            raise ValueError("Non-reorganization feature vectors must have zero depth.")

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return features using a stable serialization order."""

        return {
            "timestamp": self.timestamp,
            "canonical_height": self.canonical_height,
            "mean_height": self.mean_height,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "height_std": self.height_std,
            "height_range": self.height_range,
            "unique_tip_count": self.unique_tip_count,
            "consensus_ratio": self.consensus_ratio,
            "reference_height": self.reference_height,
            "tip_age_seconds": self.tip_age_seconds,
            "latest_block_interval_seconds": self.latest_block_interval_seconds,
            "mean_recent_block_interval_seconds": self.mean_recent_block_interval_seconds,
            "std_recent_block_interval_seconds": self.std_recent_block_interval_seconds,
            "unique_miner_count": self.unique_miner_count,
            "dominant_miner_ratio": self.dominant_miner_ratio,
            "miner_hhi": self.miner_hhi,
            "blocks_added": self.blocks_added,
            "blocks_removed": self.blocks_removed,
            "common_ancestor_height": self.common_ancestor_height,
            "reorganization_detected": self.reorganization_detected,
            "reorganization_depth": self.reorganization_depth,
            "tip_changed": self.tip_changed,
        }

    def as_numeric_tuple(self) -> tuple[float, ...]:
        """Return the feature vector as deterministic numeric model input."""

        return tuple(float(value) for value in self.as_dict().values())

    @staticmethod
    def _validate_non_negative_finite(
        value: float,
        field_name: str,
    ) -> None:
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{field_name} must be a finite non-negative value.")

    @staticmethod
    def _validate_probability(
        value: float,
        field_name: str,
    ) -> None:
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be a finite value in [0, 1].")


class ObservableFeatureExtractor:
    """Create model-safe features from public simulation observations."""

    @staticmethod
    def extract(
        telemetry: TelemetrySnapshot,
        chain_observation: ChainObservation,
    ) -> ObservableFeatureVector:
        """Combine public network telemetry and canonical-chain observations."""

        if chain_observation.height != telemetry.reference_height:
            raise ValueError(
                "Observed canonical height must equal telemetry reference height."
            )

        return ObservableFeatureVector(
            timestamp=telemetry.timestamp,
            canonical_height=chain_observation.height,
            mean_height=telemetry.mean_height,
            min_height=telemetry.min_height,
            max_height=telemetry.max_height,
            height_std=telemetry.height_std,
            height_range=(telemetry.max_height - telemetry.min_height),
            unique_tip_count=telemetry.unique_tip_count,
            consensus_ratio=telemetry.consensus_ratio,
            reference_height=telemetry.reference_height,
            tip_age_seconds=telemetry.tip_age_seconds,
            latest_block_interval_seconds=telemetry.latest_block_interval_seconds,
            mean_recent_block_interval_seconds=telemetry.mean_recent_block_interval_seconds,
            std_recent_block_interval_seconds=telemetry.std_recent_block_interval_seconds,
            unique_miner_count=telemetry.unique_miner_count,
            dominant_miner_ratio=telemetry.dominant_miner_ratio,
            miner_hhi=telemetry.miner_hhi,
            blocks_added=chain_observation.blocks_added,
            blocks_removed=chain_observation.blocks_removed,
            common_ancestor_height=chain_observation.common_ancestor_height,
            reorganization_detected=chain_observation.reorganization_detected,
            reorganization_depth=chain_observation.reorganization_depth,
            tip_changed=chain_observation.tip_changed,
        )

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from statistics import fmean, pstdev

from src.simulator.blockchain import Block
from src.simulator.network import Network


@dataclass(slots=True, frozen=True)
class TelemetrySnapshot:
    """Store publicly observable blockchain and network telemetry."""

    timestamp: float
    mean_height: float
    min_height: int
    max_height: int
    height_std: float
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

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(
                "Telemetry timestamp must be a finite non-negative value."
            )

        if not isfinite(self.mean_height) or self.mean_height < 0.0:
            raise ValueError(
                "Mean height must be a finite non-negative value."
            )

        if self.min_height < 0:
            raise ValueError("Minimum height must be non-negative.")

        if self.max_height < 0:
            raise ValueError("Maximum height must be non-negative.")

        if self.min_height > self.max_height:
            raise ValueError(
                "Minimum height must not exceed maximum height."
            )

        if not isfinite(self.height_std) or self.height_std < 0.0:
            raise ValueError(
                "Height standard deviation must be a finite "
                "non-negative value."
            )

        if self.unique_tip_count <= 0:
            raise ValueError("Unique tip count must be positive.")

        if (
            not isfinite(self.consensus_ratio)
            or not 0.0 <= self.consensus_ratio <= 1.0
        ):
            raise ValueError(
                "Consensus ratio must be a finite value in [0, 1]."
            )

        if self.reference_height < 0:
            raise ValueError(
                "Reference height must be non-negative."
            )

        if not self.min_height <= self.reference_height <= self.max_height:
            raise ValueError(
                "Reference height must be within the network height range."
            )

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
            raise ValueError(
                "Unique miner count must be non-negative."
            )

        self._validate_probability(
            self.dominant_miner_ratio,
            "Dominant miner ratio",
        )
        self._validate_probability(
            self.miner_hhi,
            "Miner HHI",
        )

        if self.reference_height == 0:
            if self.unique_miner_count != 0:
                raise ValueError(
                    "Genesis-only telemetry must have zero observed miners."
                )

            if self.dominant_miner_ratio != 0.0:
                raise ValueError(
                    "Genesis-only telemetry must have zero dominant "
                    "miner ratio."
                )

            if self.miner_hhi != 0.0:
                raise ValueError(
                    "Genesis-only telemetry must have zero miner HHI."
                )
        elif self.unique_miner_count == 0:
            raise ValueError(
                "Non-genesis telemetry must contain at least one miner."
            )

    @staticmethod
    def _validate_non_negative_finite(
        value: float,
        field_name: str,
    ) -> None:
        if not isfinite(value) or value < 0.0:
            raise ValueError(
                f"{field_name} must be a finite non-negative value."
            )

    @staticmethod
    def _validate_probability(
        value: float,
        field_name: str,
    ) -> None:
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{field_name} must be a finite value in [0, 1]."
            )


class TelemetryCollector:
    """Collect model-safe telemetry from publicly observable blockchain state."""

    @staticmethod
    def collect(
        network: Network,
        timestamp: float,
        miner_window_blocks: int = 20,
    ) -> TelemetrySnapshot:
        """Collect one deterministic telemetry snapshot."""

        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "Telemetry timestamp must be a finite non-negative value."
            )

        if miner_window_blocks <= 0:
            raise ValueError(
                "Miner observation window must be positive."
            )

        nodes = network.nodes

        if not nodes:
            raise ValueError(
                "Telemetry collection requires at least one network node."
            )

        heights = [
            node.height
            for node in nodes
        ]

        tip_hashes = [
            node.tip_hash
            for node in nodes
        ]

        mean_height = float(fmean(heights))
        min_height = min(heights)
        max_height = max(heights)

        height_std = (
            float(pstdev(heights))
            if len(heights) > 1
            else 0.0
        )

        tip_counts = Counter(tip_hashes)
        unique_tip_count = len(tip_counts)
        consensus_ratio = max(tip_counts.values()) / len(nodes)

        reference_node = nodes[0]
        reference_chain = reference_node.blockchain.chain

        (
            tip_age_seconds,
            latest_block_interval_seconds,
            mean_recent_block_interval_seconds,
            std_recent_block_interval_seconds,
        ) = TelemetryCollector._block_time_statistics(
            reference_chain,
            timestamp,
            miner_window_blocks,
        )

        (
            unique_miner_count,
            dominant_miner_ratio,
            miner_hhi,
        ) = TelemetryCollector._miner_statistics(
            reference_chain,
            miner_window_blocks,
        )

        return TelemetrySnapshot(
            timestamp=timestamp,
            mean_height=mean_height,
            min_height=min_height,
            max_height=max_height,
            height_std=height_std,
            unique_tip_count=unique_tip_count,
            consensus_ratio=consensus_ratio,
            reference_height=reference_node.height,
            tip_age_seconds=tip_age_seconds,
            latest_block_interval_seconds=(
                latest_block_interval_seconds
            ),
            mean_recent_block_interval_seconds=(
                mean_recent_block_interval_seconds
            ),
            std_recent_block_interval_seconds=(
                std_recent_block_interval_seconds
            ),
            unique_miner_count=unique_miner_count,
            dominant_miner_ratio=dominant_miner_ratio,
            miner_hhi=miner_hhi,
        )

    @staticmethod
    def _block_time_statistics(
            chain: Sequence[Block],
            observation_timestamp: float,
            window_blocks: int,
    ) -> tuple[float, float, float, float]:
        """Calculate observable timing statistics from the canonical chain."""

        if not chain:
            raise RuntimeError(
                "Canonical blockchain must contain a genesis block."
            )

        tip_timestamp = chain[-1].timestamp

        if tip_timestamp > observation_timestamp:
            raise ValueError(
                "Canonical tip timestamp must not exceed observation time."
            )

        tip_age_seconds = observation_timestamp - tip_timestamp

        if len(chain) == 1:
            return (
                tip_age_seconds,
                0.0,
                0.0,
                0.0,
            )

        recent_blocks = chain[-(window_blocks + 1):]

        intervals: list[float] = []

        for previous, current in zip(
            recent_blocks,
            recent_blocks[1:],
            strict=False,
        ):
            interval = current.timestamp - previous.timestamp

            if not isfinite(interval) or interval < 0.0:
                raise ValueError(
                    "Canonical block timestamps must be finite and "
                    "non-decreasing."
                )

            intervals.append(interval)

        latest_interval = intervals[-1]
        mean_interval = float(fmean(intervals))
        interval_std = (
            float(pstdev(intervals))
            if len(intervals) > 1
            else 0.0
        )

        return (
            tip_age_seconds,
            latest_interval,
            mean_interval,
            interval_std,
        )

    @staticmethod
    def _miner_statistics(
            chain: Sequence[Block],
            window_blocks: int,
    ) -> tuple[int, float, float]:
        """Calculate recent public miner-concentration statistics."""

        if len(chain) == 1:
            return 0, 0.0, 0.0

        recent_blocks = chain[1:][-window_blocks:]

        miner_counts = Counter(
            block.miner_id
            for block in recent_blocks
        )

        total_blocks = len(recent_blocks)

        if total_blocks == 0:
            return 0, 0.0, 0.0

        unique_miner_count = len(miner_counts)
        dominant_miner_ratio = (
            max(miner_counts.values()) / total_blocks
        )

        miner_hhi = sum(
            (count / total_blocks) ** 2
            for count in miner_counts.values()
        )

        return (
            unique_miner_count,
            dominant_miner_ratio,
            miner_hhi,
        )
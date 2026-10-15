from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from random import Random

from src.simulator.blockchain import Block
from src.simulator.network import Node


@dataclass(slots=True, frozen=True)
class Miner:
    """Represent a miner with a positive amount of relative hash power."""

    miner_id: str
    hash_power: float

    def __post_init__(self) -> None:
        if not self.miner_id:
            raise ValueError("Miner identifier must not be empty.")

        if not isfinite(self.hash_power) or self.hash_power <= 0.0:
            raise ValueError("Miner hash power must be a finite positive value.")


class MiningPool:
    """Select miners proportionally to hash power using seeded randomness."""

    def __init__(
        self,
        miners: Sequence[Miner],
        seed: int,
    ) -> None:
        if seed < 0:
            raise ValueError("Mining-pool random seed must be non-negative.")

        if not miners:
            raise ValueError("Mining pool must contain at least one miner.")

        miner_tuple = tuple(miners)

        miner_ids = [miner.miner_id for miner in miner_tuple]

        if len(set(miner_ids)) != len(miner_ids):
            raise ValueError("Mining pool miner identifiers must be unique.")

        total_hash_power = sum(miner.hash_power for miner in miner_tuple)

        if not isfinite(total_hash_power) or total_hash_power <= 0.0:
            raise ValueError(
                "Total mining-pool hash power must be finite and positive."
            )

        cumulative_hash_power: list[float] = []
        running_total = 0.0

        for miner in miner_tuple:
            running_total += miner.hash_power
            cumulative_hash_power.append(running_total)

        self._miners = miner_tuple
        self._seed = seed
        self._rng = Random(seed)
        self._total_hash_power = total_hash_power
        self._cumulative_hash_power = tuple(cumulative_hash_power)

    @property
    def miners(self) -> tuple[Miner, ...]:
        """Return miners as an immutable sequence."""

        return self._miners

    @property
    def seed(self) -> int:
        """Return the random seed used by this mining pool."""

        return self._seed

    @property
    def total_hash_power(self) -> float:
        """Return the sum of all configured miner hash power."""

        return self._total_hash_power

    def normalized_hash_shares(self) -> dict[str, float]:
        """Return each miner's normalized fraction of total hash power."""

        return {
            miner.miner_id: (miner.hash_power / self._total_hash_power)
            for miner in self._miners
        }

    def select_miner(self) -> Miner:
        """Select one miner proportionally to configured hash power."""

        target = self._rng.random() * self._total_hash_power

        for miner, cumulative_power in zip(
            self._miners,
            self._cumulative_hash_power,
            strict=True,
        ):
            if target < cumulative_power:
                return miner

        return self._miners[-1]

    def create_block(
        self,
        node: Node,
        timestamp: float,
    ) -> Block:
        """Create and finalize a block mined on the node's current tip."""

        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("Block timestamp must be a finite non-negative value.")

        miner = self.select_miner()

        block = Block(
            height=node.height + 1,
            previous_hash=node.tip_hash,
            miner_id=miner.miner_id,
            timestamp=timestamp,
        )
        block.finalize()

        return block

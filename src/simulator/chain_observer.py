from __future__ import annotations

from dataclasses import dataclass

from src.simulator.network import Node


@dataclass(slots=True, frozen=True)
class ChainObservation:
    """Describe publicly observable canonical-chain changes."""

    height: int
    blocks_added: int
    blocks_removed: int
    common_ancestor_height: int
    reorganization_detected: bool
    reorganization_depth: int
    tip_changed: bool

    def __post_init__(self) -> None:
        if self.height < 0:
            raise ValueError("Observed chain height must be non-negative.")

        if self.blocks_added < 0:
            raise ValueError("Added block count must be non-negative.")

        if self.blocks_removed < 0:
            raise ValueError("Removed block count must be non-negative.")

        if self.common_ancestor_height < 0:
            raise ValueError(
                "Common ancestor height must be non-negative."
            )

        if self.common_ancestor_height > self.height:
            raise ValueError(
                "Common ancestor height must not exceed observed chain height."
            )

        if self.reorganization_depth < 0:
            raise ValueError(
                "Reorganization depth must be non-negative."
            )

        if self.reorganization_detected and self.reorganization_depth == 0:
            raise ValueError(
                "Detected reorganizations must have positive depth."
            )

        if (
            not self.reorganization_detected
            and self.reorganization_depth != 0
        ):
            raise ValueError(
                "Non-reorganization observations must have zero depth."
            )


class CanonicalChainObserver:
    """Track observable changes to one node's canonical blockchain."""

    def __init__(self) -> None:
        self._previous_hashes: tuple[str, ...] | None = None

    @property
    def initialized(self) -> bool:
        """Return whether at least one chain has already been observed."""

        return self._previous_hashes is not None

    def observe(self, node: Node) -> ChainObservation:
        """Observe the node's canonical chain and describe changes since last call."""

        current_hashes = tuple(
            block.block_hash
            for block in node.blockchain.chain
        )

        if not current_hashes:
            raise RuntimeError("Canonical blockchain must not be empty.")

        previous_hashes = self._previous_hashes

        if previous_hashes is None:
            self._previous_hashes = current_hashes

            return ChainObservation(
                height=node.height,
                blocks_added=0,
                blocks_removed=0,
                common_ancestor_height=node.height,
                reorganization_detected=False,
                reorganization_depth=0,
                tip_changed=False,
            )

        common_prefix_length = self._common_prefix_length(
            previous_hashes,
            current_hashes,
        )

        blocks_removed = len(previous_hashes) - common_prefix_length
        blocks_added = len(current_hashes) - common_prefix_length

        reorganization_detected = blocks_removed > 0
        reorganization_depth = (
            blocks_removed
            if reorganization_detected
            else 0
        )

        tip_changed = previous_hashes[-1] != current_hashes[-1]

        common_ancestor_height = common_prefix_length - 1

        if common_ancestor_height < 0:
            raise RuntimeError(
                "Observed chains do not share the canonical genesis block."
            )

        observation = ChainObservation(
            height=node.height,
            blocks_added=blocks_added,
            blocks_removed=blocks_removed,
            common_ancestor_height=common_ancestor_height,
            reorganization_detected=reorganization_detected,
            reorganization_depth=reorganization_depth,
            tip_changed=tip_changed,
        )

        self._previous_hashes = current_hashes

        return observation

    def reset(self) -> None:
        """Forget previously observed canonical-chain state."""

        self._previous_hashes = None

    @staticmethod
    def _common_prefix_length(
        first: tuple[str, ...],
        second: tuple[str, ...],
    ) -> int:
        """Return the number of identical leading hashes in two chains."""

        common_length = 0

        for first_hash, second_hash in zip(first, second, strict=False):
            if first_hash != second_hash:
                break

            common_length += 1

        return common_length
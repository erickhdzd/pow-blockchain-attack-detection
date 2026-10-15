from __future__ import annotations

from dataclasses import dataclass

from src.simulator.blockchain import Blockchain
from src.simulator.forks import BlockchainBranch
from src.simulator.network import Node


@dataclass(slots=True, frozen=True)
class ReorganizationResult:
    """Describe the outcome of a canonical-chain adoption attempt."""

    adopted: bool
    old_height: int
    new_height: int
    fork_height: int
    removed_blocks: int
    added_blocks: int

    def __post_init__(self) -> None:
        if self.old_height < 0:
            raise ValueError("Old chain height must be non-negative.")

        if self.new_height < 0:
            raise ValueError("New chain height must be non-negative.")

        if self.fork_height < 0:
            raise ValueError("Fork height must be non-negative.")

        if self.fork_height > self.old_height:
            raise ValueError(
                "Fork height must not exceed the old canonical-chain height."
            )

        if self.removed_blocks < 0:
            raise ValueError("Removed block count must be non-negative.")

        if self.added_blocks < 0:
            raise ValueError("Added block count must be non-negative.")


class ChainSelector:
    """Apply longest-chain selection to competing proof-of-work branches."""

    @staticmethod
    def adopt_if_longer(
        node: Node,
        branch: BlockchainBranch,
    ) -> ReorganizationResult:
        """
        Adopt a competing branch when it is strictly longer than the local chain.

        The branch must originate from a block already present in the node's
        current canonical chain.
        """

        if not branch.validate():
            raise ValueError("Cannot adopt an invalid blockchain branch.")

        current_chain = node.blockchain.chain
        old_height = node.height

        parent_height = branch.parent_block.height

        if parent_height > old_height:
            raise ValueError(
                "Branch parent height exceeds the node's canonical-chain height."
            )

        canonical_parent = current_chain[parent_height]

        if canonical_parent.block_hash != branch.parent_block.block_hash:
            raise ValueError("Branch parent is not part of the node's canonical chain.")

        if branch.height <= old_height:
            return ReorganizationResult(
                adopted=False,
                old_height=old_height,
                new_height=old_height,
                fork_height=parent_height,
                removed_blocks=0,
                added_blocks=0,
            )

        replacement_chain = Blockchain()

        for block in current_chain[1 : parent_height + 1]:
            replacement_chain.add_block(block)

        for block in branch.blocks:
            replacement_chain.add_block(block)

        if not replacement_chain.validate():
            raise ValueError("Candidate replacement chain failed canonical validation.")

        removed_blocks = old_height - parent_height
        added_blocks = branch.length

        node.blockchain = replacement_chain

        return ReorganizationResult(
            adopted=True,
            old_height=old_height,
            new_height=node.height,
            fork_height=parent_height,
            removed_blocks=removed_blocks,
            added_blocks=added_blocks,
        )

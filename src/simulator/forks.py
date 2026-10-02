from __future__ import annotations

from dataclasses import dataclass, field

from src.simulator.blockchain import Block


@dataclass(slots=True)
class BlockchainBranch:
    """Represent a blockchain branch extending from a common parent block."""

    branch_id: str
    parent_block: Block
    blocks: list[Block] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("Branch ID must not be empty.")

        if not self.parent_block.block_hash:
            raise ValueError(
                "Branch parent block must be finalized before creating a branch."
            )

        self._validate_existing_blocks()

    @property
    def length(self) -> int:
        """Return the number of blocks added after the common parent."""

        return len(self.blocks)

    @property
    def height(self) -> int:
        """Return the height of the current branch tip."""

        if not self.blocks:
            return self.parent_block.height

        return self.blocks[-1].height

    @property
    def tip(self) -> Block:
        """Return the current branch tip."""

        if not self.blocks:
            return self.parent_block

        return self.blocks[-1]

    @property
    def tip_hash(self) -> str:
        """Return the hash of the current branch tip."""

        return self.tip.block_hash

    def add_block(self, block: Block) -> None:
        """Append a valid block that directly extends the current branch."""

        expected_height = self.height + 1

        if block.height != expected_height:
            raise ValueError(
                f"Invalid branch block height: expected {expected_height}, "
                f"received {block.height}."
            )

        if block.previous_hash != self.tip_hash:
            raise ValueError(
                "Invalid branch previous hash: the block does not extend "
                "the current branch tip."
            )

        if block.timestamp < self.tip.timestamp:
            raise ValueError(
                "Invalid branch block timestamp: it must not precede "
                "the current branch tip."
            )

        if not block.is_hash_valid():
            raise ValueError(
                "Invalid branch block hash: the stored hash does not match "
                "the block contents."
            )

        self.blocks.append(block)

    def validate(self) -> bool:
        """Return whether the complete branch is internally consistent."""

        if not self.parent_block.is_hash_valid():
            return False

        previous_block = self.parent_block

        for current_block in self.blocks:
            if current_block.height != previous_block.height + 1:
                return False

            if current_block.previous_hash != previous_block.block_hash:
                return False

            if current_block.timestamp < previous_block.timestamp:
                return False

            if not current_block.is_hash_valid():
                return False

            previous_block = current_block

        return True

    def _validate_existing_blocks(self) -> None:
        """Validate blocks supplied during branch construction."""

        if not self.validate():
            raise ValueError(
                "Existing branch blocks do not form a valid extension "
                "of the parent block."
            )


class ForkManager:
    """Manage competing blockchain branches sharing a common parent."""

    def __init__(self, parent_block: Block) -> None:
        if not parent_block.block_hash:
            raise ValueError(
                "Fork parent block must be finalized before creating "
                "a fork manager."
            )

        if not parent_block.is_hash_valid():
            raise ValueError("Fork parent block hash is invalid.")

        self._parent_block = parent_block
        self._branches: dict[str, BlockchainBranch] = {}

    @property
    def parent_block(self) -> Block:
        """Return the common parent block of all managed branches."""

        return self._parent_block

    @property
    def branches(self) -> tuple[BlockchainBranch, ...]:
        """Return all currently managed branches."""

        return tuple(self._branches.values())

    def create_branch(self, branch_id: str) -> BlockchainBranch:
        """Create and register an empty branch."""

        if not branch_id:
            raise ValueError("Branch ID must not be empty.")

        if branch_id in self._branches:
            raise ValueError(f"Branch ID already exists: {branch_id}.")

        branch = BlockchainBranch(
            branch_id=branch_id,
            parent_block=self._parent_block,
        )

        self._branches[branch_id] = branch

        return branch

    def get_branch(self, branch_id: str) -> BlockchainBranch:
        """Return a registered branch by ID."""

        try:
            return self._branches[branch_id]
        except KeyError as exc:
            raise KeyError(f"Unknown branch ID: {branch_id}.") from exc

    def longest_branches(self) -> tuple[BlockchainBranch, ...]:
        """Return every branch tied for the greatest current height."""

        if not self._branches:
            return ()

        greatest_height = max(branch.height for branch in self._branches.values())

        return tuple(
            branch
            for branch in self._branches.values()
            if branch.height == greatest_height
        )

    def has_competing_tips(self) -> bool:
        """Return whether multiple distinct branch tips currently exist."""

        if len(self._branches) < 2:
            return False

        tip_hashes = {
            branch.tip_hash
            for branch in self._branches.values()
        }

        return len(tip_hashes) > 1
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite


@dataclass(slots=True, frozen=True)
class Transaction:
    """Represent a simplified blockchain transaction."""

    tx_id: str
    sender: str
    receiver: str
    amount: float
    timestamp: float
    spent_output_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tx_id:
            raise ValueError("Transaction ID must not be empty.")

        if not self.sender:
            raise ValueError("Transaction sender must not be empty.")

        if not self.receiver:
            raise ValueError("Transaction receiver must not be empty.")

        if not isfinite(self.amount) or self.amount <= 0.0:
            raise ValueError("Transaction amount must be a finite positive value.")

        if not isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError(
                "Transaction timestamp must be a finite non-negative value."
            )


@dataclass(slots=True)
class Block:
    """Represent a simplified block in a proof-of-work blockchain."""

    height: int
    previous_hash: str
    miner_id: str
    timestamp: float
    transactions: list[Transaction] = field(default_factory=list)
    nonce: int = 0
    block_hash: str = ""

    def __post_init__(self) -> None:
        if self.height < 0:
            raise ValueError("Block height must be non-negative.")

        if not self.previous_hash:
            raise ValueError("Previous block hash must not be empty.")

        if not self.miner_id:
            raise ValueError("Miner ID must not be empty.")

        if not isfinite(self.timestamp) or self.timestamp < 0.0:
            raise ValueError("Block timestamp must be a finite non-negative value.")

        if self.nonce < 0:
            raise ValueError("Block nonce must be non-negative.")

    def compute_hash(self) -> str:
        """Compute the deterministic SHA-256 hash of the block contents."""

        transaction_ids = ",".join(
            transaction.tx_id for transaction in self.transactions
        )

        payload = (
            f"{self.height}|"
            f"{self.previous_hash}|"
            f"{self.miner_id}|"
            f"{self.timestamp:.9f}|"
            f"{self.nonce}|"
            f"{transaction_ids}"
        )

        return sha256(payload.encode("utf-8")).hexdigest()

    def finalize(self) -> None:
        """Compute and store the current block hash."""

        self.block_hash = self.compute_hash()

    def is_hash_valid(self) -> bool:
        """Return whether the stored hash matches the block contents."""

        return bool(self.block_hash) and self.block_hash == self.compute_hash()


class Blockchain:
    """Maintain a single canonical blockchain branch."""

    GENESIS_PREVIOUS_HASH = "0" * 64

    def __init__(self) -> None:
        self._chain: list[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        genesis_block = Block(
            height=0,
            previous_hash=self.GENESIS_PREVIOUS_HASH,
            miner_id="genesis",
            timestamp=0.0,
        )
        genesis_block.finalize()
        self._chain.append(genesis_block)

    @property
    def chain(self) -> tuple[Block, ...]:
        """Return an immutable view of the current canonical chain."""

        return tuple(self._chain)

    @property
    def tip(self) -> Block:
        """Return the current canonical chain tip."""

        return self._chain[-1]

    @property
    def height(self) -> int:
        """Return the height of the current canonical chain."""

        return self.tip.height

    def add_block(self, block: Block) -> None:
        """Append a valid block that directly extends the current chain tip."""

        expected_height = self.height + 1

        if block.height != expected_height:
            raise ValueError(
                f"Invalid block height: expected {expected_height}, "
                f"received {block.height}."
            )

        if block.previous_hash != self.tip.block_hash:
            raise ValueError(
                "Invalid previous hash: the block does not extend the current "
                "canonical chain."
            )

        if block.timestamp < self.tip.timestamp:
            raise ValueError(
                "Invalid block timestamp: it must not precede the current chain tip."
            )

        if not block.is_hash_valid():
            raise ValueError(
                "Invalid block hash: the stored hash does not match the block contents."
            )

        self._chain.append(block)

    def validate(self) -> bool:
        """Validate the complete canonical chain."""

        if not self._chain:
            return False

        genesis_block = self._chain[0]

        if genesis_block.height != 0:
            return False

        if genesis_block.previous_hash != self.GENESIS_PREVIOUS_HASH:
            return False

        if genesis_block.miner_id != "genesis":
            return False

        if genesis_block.timestamp != 0.0:
            return False

        if not genesis_block.is_hash_valid():
            return False

        for previous_block, current_block in zip(
            self._chain,
            self._chain[1:],
            strict=False,
        ):
            if current_block.height != previous_block.height + 1:
                return False

            if current_block.previous_hash != previous_block.block_hash:
                return False

            if current_block.timestamp < previous_block.timestamp:
                return False

            if not current_block.is_hash_valid():
                return False

        return True
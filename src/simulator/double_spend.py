from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from src.simulator.attacks import DoubleSpendParameters
from src.simulator.blockchain import Block, Transaction
from src.simulator.consensus import ChainSelector, ReorganizationResult
from src.simulator.forks import BlockchainBranch
from src.simulator.network import Node


@dataclass(slots=True, frozen=True)
class DoubleSpendTransactionPair:
    """Represent two conflicting transactions spending the same output."""

    payment_transaction: Transaction
    conflicting_transaction: Transaction

    def __post_init__(self) -> None:
        payment_output = self.payment_transaction.spent_output_id
        conflicting_output = self.conflicting_transaction.spent_output_id

        if payment_output is None:
            raise ValueError(
                "Payment transaction must reference a spent output."
            )

        if conflicting_output is None:
            raise ValueError(
                "Conflicting transaction must reference a spent output."
            )

        if payment_output != conflicting_output:
            raise ValueError(
                "Double-spend transactions must spend the same output."
            )

        if self.payment_transaction.tx_id == self.conflicting_transaction.tx_id:
            raise ValueError(
                "Double-spend transactions must have different transaction IDs."
            )


@dataclass(slots=True, frozen=True)
class DoubleSpendStatus:
    """Describe the internal state of one double-spending attempt."""

    public_confirmations: int
    private_branch_length: int
    target_confirmations_reached: bool
    private_branch_ahead: bool
    successful: bool

    def __post_init__(self) -> None:
        if self.public_confirmations < 0:
            raise ValueError("Public confirmation count must be non-negative.")

        if self.private_branch_length < 0:
            raise ValueError("Private branch length must be non-negative.")


class DoubleSpendAttack:
    """Model a private-chain double-spending attempt."""

    def __init__(
        self,
        victim_node: Node,
        parameters: DoubleSpendParameters,
        attacker_id: str,
    ) -> None:
        if not attacker_id:
            raise ValueError("Double-spend attacker ID must not be empty.")

        self._victim_node = victim_node
        self._parameters = parameters
        self._attacker_id = attacker_id

        self._fork_parent: Block | None = None
        self._private_branch: BlockchainBranch | None = None
        self._transactions: DoubleSpendTransactionPair | None = None
        self._payment_block_height: int | None = None
        self._successful = False

    @property
    def started(self) -> bool:
        """Return whether the attack has been initialized."""

        return self._private_branch is not None

    @property
    def successful(self) -> bool:
        """Return whether the attack has successfully reorganized the victim."""

        return self._successful

    @property
    def private_branch(self) -> BlockchainBranch:
        """Return the attacker's private branch."""

        if self._private_branch is None:
            raise RuntimeError("Double-spend attack has not been started.")

        return self._private_branch

    @property
    def transactions(self) -> DoubleSpendTransactionPair:
        """Return the conflicting transaction pair."""

        if self._transactions is None:
            raise RuntimeError("Double-spend attack has not been started.")

        return self._transactions

    def start(
            self,
            sender: str,
            victim_receiver: str,
            attacker_receiver: str,
            spent_output_id: str,
            timestamp: float,
    ) -> DoubleSpendTransactionPair:
        """Initialize conflicting transactions and the private attack branch."""

        if self.started:
            raise RuntimeError("Double-spend attack has already been started.")

        if not sender:
            raise ValueError("Double-spend transaction sender must not be empty.")

        if not victim_receiver:
            raise ValueError(
                "Double-spend victim receiver must not be empty."
            )

        if not attacker_receiver:
            raise ValueError(
                "Double-spend attacker receiver must not be empty."
            )

        if not spent_output_id:
            raise ValueError("Spent output ID must not be empty.")

        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "Double-spend timestamp must be a finite non-negative value."
            )

        fork_parent = self._victim_node.blockchain.tip

        payment_transaction = Transaction(
            tx_id=self._make_transaction_id(
                role="payment",
                sender=sender,
                receiver=victim_receiver,
                spent_output_id=spent_output_id,
                timestamp=timestamp,
            ),
            sender=sender,
            receiver=victim_receiver,
            amount=self._parameters.transaction_value,
            timestamp=timestamp,
            spent_output_id=spent_output_id,
        )

        conflicting_transaction = Transaction(
            tx_id=self._make_transaction_id(
                role="conflict",
                sender=sender,
                receiver=attacker_receiver,
                spent_output_id=spent_output_id,
                timestamp=timestamp,
            ),
            sender=sender,
            receiver=attacker_receiver,
            amount=self._parameters.transaction_value,
            timestamp=timestamp,
            spent_output_id=spent_output_id,
        )

        transaction_pair = DoubleSpendTransactionPair(
            payment_transaction=payment_transaction,
            conflicting_transaction=conflicting_transaction,
        )

        private_branch = BlockchainBranch(
            branch_id=f"double_spend_{self._attacker_id}",
            parent_block=fork_parent,
        )

        self._fork_parent = fork_parent
        self._transactions = transaction_pair
        self._private_branch = private_branch

        return transaction_pair

    def register_payment_block(self, block_height: int) -> None:
        """Record the canonical block height containing the victim payment."""

        if not self.started:
            raise RuntimeError("Double-spend attack has not been started.")

        if block_height <= self.private_branch.parent_block.height:
            raise ValueError(
                "Payment block height must be above the fork-parent height."
            )

        if block_height > self._victim_node.height:
            raise ValueError(
                "Payment block height exceeds the victim's canonical-chain height."
            )

        block = self._victim_node.blockchain.chain[block_height]

        payment_tx_id = self.transactions.payment_transaction.tx_id

        if not any(
            transaction.tx_id == payment_tx_id
            for transaction in block.transactions
        ):
            raise ValueError(
                "Registered payment block does not contain the payment transaction."
            )

        self._payment_block_height = block_height

    def add_private_block(self, block: Block) -> None:
        """Add one withheld attacker block to the private branch."""

        if not self.started:
            raise RuntimeError("Double-spend attack has not been started.")

        if not any(
            transaction.tx_id == self.transactions.conflicting_transaction.tx_id
            for transaction in block.transactions
        ) and self.private_branch.length == 0:
            raise ValueError(
                "The first private block must contain the conflicting transaction."
            )

        self.private_branch.add_block(block)

    def public_confirmations(self) -> int:
        """Return the current number of confirmations of the victim payment."""

        if self._payment_block_height is None:
            return 0

        if self._payment_block_height > self._victim_node.height:
            return 0

        payment_block = self._victim_node.blockchain.chain[
            self._payment_block_height
        ]

        payment_tx_id = self.transactions.payment_transaction.tx_id

        if not any(
            transaction.tx_id == payment_tx_id
            for transaction in payment_block.transactions
        ):
            return 0

        return self._victim_node.height - self._payment_block_height + 1

    def status(self) -> DoubleSpendStatus:
        """Return the current internal status of the attack."""

        confirmations = self.public_confirmations()

        return DoubleSpendStatus(
            public_confirmations=confirmations,
            private_branch_length=self.private_branch.length,
            target_confirmations_reached=(
                confirmations >= self._parameters.target_confirmations
            ),
            private_branch_ahead=(
                self.private_branch.height > self._victim_node.height
            ),
            successful=self._successful,
        )

    def publish_private_branch(self) -> ReorganizationResult:
        """Publish the private branch and attempt canonical-chain adoption."""

        if not self.started:
            raise RuntimeError("Double-spend attack has not been started.")

        result = ChainSelector.adopt_if_longer(
            node=self._victim_node,
            branch=self.private_branch,
        )

        if result.adopted:
            self._successful = (
                self.public_confirmations() == 0
                and self._canonical_conflicting_transaction_exists()
            )

        return result

    def _canonical_conflicting_transaction_exists(self) -> bool:
        """Return whether the conflicting transaction is now canonical."""

        conflicting_tx_id = self.transactions.conflicting_transaction.tx_id

        return any(
            transaction.tx_id == conflicting_tx_id
            for block in self._victim_node.blockchain.chain
            for transaction in block.transactions
        )

    @staticmethod
    def _make_transaction_id(
        role: str,
        sender: str,
        receiver: str,
        spent_output_id: str,
        timestamp: float,
    ) -> str:
        """Create a deterministic transaction identifier."""

        payload = (
            f"{role}|"
            f"{sender}|"
            f"{receiver}|"
            f"{spent_output_id}|"
            f"{timestamp:.9f}"
        )

        return sha256(payload.encode("utf-8")).hexdigest()
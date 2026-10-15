from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from src.simulator.attacks import DoubleSpendParameters
from src.simulator.blockchain import Block
from src.simulator.consensus import ReorganizationResult
from src.simulator.double_spend import (
    DoubleSpendAttack,
    DoubleSpendTransactionPair,
)
from src.simulator.network import Node


class DoubleSpendMiningSide(Enum):
    """Identify the winner of one double-spend mining event."""

    PUBLIC = auto()
    ATTACKER = auto()


class DoubleSpendScenarioState(Enum):
    """Represent the internal lifecycle of a double-spending scenario."""

    INACTIVE = auto()
    PAYMENT_PENDING = auto()
    CONFIRMING = auto()
    TARGET_REACHED = auto()
    PRIVATE_BRANCH_AHEAD = auto()
    SUCCEEDED = auto()
    FAILED = auto()


@dataclass(slots=True, frozen=True)
class DoubleSpendScenarioStatus:
    """Describe the internal state of a double-spending scenario."""

    state: DoubleSpendScenarioState
    public_confirmations: int
    target_confirmations: int
    private_branch_length: int
    private_branch_ahead: bool
    publication_attempted: bool
    successful: bool

    def __post_init__(self) -> None:
        if self.public_confirmations < 0:
            raise ValueError("Public confirmation count must be non-negative.")

        if self.target_confirmations <= 0:
            raise ValueError("Target confirmation count must be positive.")

        if self.private_branch_length < 0:
            raise ValueError("Private branch length must be non-negative.")


class DoubleSpendScenario:
    """Coordinate the lifecycle of one stochastic double-spending attempt."""

    def __init__(
        self,
        victim_node: Node,
        parameters: DoubleSpendParameters,
        attacker_id: str,
        sender: str,
        victim_receiver: str,
        attacker_receiver: str,
        spent_output_id: str,
    ) -> None:
        if not attacker_id:
            raise ValueError("Double-spend attacker ID must not be empty.")

        if not sender:
            raise ValueError("Double-spend sender must not be empty.")

        if not victim_receiver:
            raise ValueError("Double-spend victim receiver must not be empty.")

        if not attacker_receiver:
            raise ValueError("Double-spend attacker receiver must not be empty.")

        if not spent_output_id:
            raise ValueError("Double-spend spent-output ID must not be empty.")

        self._victim_node = victim_node
        self._parameters = parameters
        self._attacker_id = attacker_id
        self._sender = sender
        self._victim_receiver = victim_receiver
        self._attacker_receiver = attacker_receiver
        self._spent_output_id = spent_output_id

        self._attack = DoubleSpendAttack(
            victim_node=victim_node,
            parameters=parameters,
            attacker_id=attacker_id,
        )

        self._state = DoubleSpendScenarioState.INACTIVE
        self._transactions: DoubleSpendTransactionPair | None = None
        self._publication_attempted = False

    @property
    def started(self) -> bool:
        """Return whether the scenario has been initialized."""

        return self._transactions is not None

    @property
    def state(self) -> DoubleSpendScenarioState:
        """Return the current scenario state."""

        return self._state

    @property
    def attack(self) -> DoubleSpendAttack:
        """Return the underlying double-spending attack object."""

        return self._attack

    @property
    def transactions(self) -> DoubleSpendTransactionPair:
        """Return the conflicting transaction pair."""

        if self._transactions is None:
            raise RuntimeError("Double-spend scenario has not been started.")

        return self._transactions

    def start(self, timestamp: float) -> DoubleSpendTransactionPair:
        """Initialize the conflicting transactions."""

        if self.started:
            raise RuntimeError("Double-spend scenario has already been started.")

        transactions = self._attack.start(
            sender=self._sender,
            victim_receiver=self._victim_receiver,
            attacker_receiver=self._attacker_receiver,
            spent_output_id=self._spent_output_id,
            timestamp=timestamp,
        )

        self._transactions = transactions
        self._state = DoubleSpendScenarioState.PAYMENT_PENDING

        return transactions

    def mine_payment_block(
        self,
        timestamp: float,
        miner_id: str = "honest",
    ) -> Block:
        """Mine the public block containing the victim payment."""

        if not self.started:
            raise RuntimeError("Double-spend scenario has not been started.")

        if self._state is not DoubleSpendScenarioState.PAYMENT_PENDING:
            raise RuntimeError(
                "Victim payment can only be mined while payment is pending."
            )

        if not miner_id:
            raise ValueError("Payment-block miner ID must not be empty.")

        block = Block(
            height=self._victim_node.height + 1,
            previous_hash=self._victim_node.tip_hash,
            miner_id=miner_id,
            timestamp=timestamp,
            transactions=[self.transactions.payment_transaction],
        )
        block.finalize()

        if not self._victim_node.receive_block(block):
            raise RuntimeError("Victim node rejected the public payment block.")

        self._attack.register_payment_block(block.height)
        self._update_state()

        return block

    def mine_public_confirmation(
        self,
        timestamp: float,
        miner_id: str = "honest",
    ) -> Block:
        """Mine one additional public confirmation block."""

        if not self.started:
            raise RuntimeError("Double-spend scenario has not been started.")

        if self._state not in {
            DoubleSpendScenarioState.CONFIRMING,
            DoubleSpendScenarioState.TARGET_REACHED,
            DoubleSpendScenarioState.PRIVATE_BRANCH_AHEAD,
        }:
            raise RuntimeError(
                "Public confirmations require a canonical victim payment."
            )

        if not miner_id:
            raise ValueError("Confirmation-block miner ID must not be empty.")

        block = Block(
            height=self._victim_node.height + 1,
            previous_hash=self._victim_node.tip_hash,
            miner_id=miner_id,
            timestamp=timestamp,
        )
        block.finalize()

        if not self._victim_node.receive_block(block):
            raise RuntimeError("Victim node rejected the public confirmation block.")

        self._update_state()

        return block

    def mine_private_block(self, timestamp: float) -> Block:
        """Mine one withheld attacker block on the conflicting private branch."""

        if not self.started:
            raise RuntimeError("Double-spend scenario has not been started.")

        if self._state in {
            DoubleSpendScenarioState.SUCCEEDED,
            DoubleSpendScenarioState.FAILED,
        }:
            raise RuntimeError(
                "Cannot mine private blocks after the scenario has terminated."
            )

        transactions = (
            [self.transactions.conflicting_transaction]
            if self._attack.private_branch.length == 0
            else []
        )

        block = Block(
            height=self._attack.private_branch.height + 1,
            previous_hash=self._attack.private_branch.tip_hash,
            miner_id=self._attacker_id,
            timestamp=timestamp,
            transactions=transactions,
        )
        block.finalize()

        self._attack.add_private_block(block)
        self._update_state()

        return block

    def process_mining_event(
        self,
        timestamp: float,
        mining_side: DoubleSpendMiningSide,
        public_miner_id: str | None = None,
    ) -> Block:
        """Process exactly one public or attacker mining event."""

        if not self.started:
            raise RuntimeError("Double-spend scenario must be started before mining.")

        if self._state in {
            DoubleSpendScenarioState.SUCCEEDED,
            DoubleSpendScenarioState.FAILED,
        }:
            raise RuntimeError("Double-spend scenario has already finished.")

        if mining_side is DoubleSpendMiningSide.PUBLIC:
            if not public_miner_id:
                raise ValueError(
                    "Public miner identifier must be provided for public mining."
                )

            if self._state is DoubleSpendScenarioState.PAYMENT_PENDING:
                block = self.mine_payment_block(
                    timestamp=timestamp,
                    miner_id=public_miner_id,
                )
            else:
                block = self.mine_public_confirmation(
                    timestamp=timestamp,
                    miner_id=public_miner_id,
                )
        elif mining_side is DoubleSpendMiningSide.ATTACKER:
            block = self.mine_private_block(timestamp)
        else:
            raise ValueError("Unsupported double-spend mining side.")

        return block

    def fail(self) -> None:
        """Finalize an unfinished attack attempt as failed."""

        if not self.started:
            raise RuntimeError(
                "Double-spend scenario must be started before it can fail."
            )

        if self._state is DoubleSpendScenarioState.SUCCEEDED:
            raise RuntimeError(
                "A successful double-spend scenario cannot be marked as failed."
            )

        if self._state is DoubleSpendScenarioState.FAILED:
            return

        self._state = DoubleSpendScenarioState.FAILED

    def can_publish_successfully(self) -> bool:
        """
        Return whether publication can satisfy the modeled victim-acceptance attack.

        A successful modeled attack requires the payment to have reached the
        victim's confirmation threshold and the private branch to be strictly
        longer than the public chain.
        """

        if not self.started:
            return False

        attack_status = self._attack.status()

        return (
            attack_status.target_confirmations_reached
            and attack_status.private_branch_ahead
        )

    def publish(self) -> ReorganizationResult:
        """Publish the private branch after the victim acceptance threshold."""

        if not self.started:
            raise RuntimeError("Double-spend scenario has not been started.")

        if self._publication_attempted:
            raise RuntimeError("Private branch publication has already been attempted.")

        if not self.can_publish_successfully():
            raise RuntimeError(
                "Private branch cannot yet produce a confirmed double-spend success."
            )

        self._publication_attempted = True

        result = self._attack.publish_private_branch()

        if result.adopted and self._attack.successful:
            self._state = DoubleSpendScenarioState.SUCCEEDED
        else:
            self._state = DoubleSpendScenarioState.FAILED

        return result

    def status(self) -> DoubleSpendScenarioStatus:
        """Return the current internal scenario status."""

        attack_status = self._attack.status() if self.started else None

        if attack_status is None:
            return DoubleSpendScenarioStatus(
                state=self._state,
                public_confirmations=0,
                target_confirmations=self._parameters.target_confirmations,
                private_branch_length=0,
                private_branch_ahead=False,
                publication_attempted=self._publication_attempted,
                successful=False,
            )

        return DoubleSpendScenarioStatus(
            state=self._state,
            public_confirmations=attack_status.public_confirmations,
            target_confirmations=self._parameters.target_confirmations,
            private_branch_length=attack_status.private_branch_length,
            private_branch_ahead=attack_status.private_branch_ahead,
            publication_attempted=self._publication_attempted,
            successful=attack_status.successful,
        )

    def _update_state(self) -> None:
        """Update the scenario state from the underlying attack state."""

        attack_status = self._attack.status()

        if attack_status.successful:
            self._state = DoubleSpendScenarioState.SUCCEEDED
        elif (
            attack_status.target_confirmations_reached
            and attack_status.private_branch_ahead
        ):
            self._state = DoubleSpendScenarioState.PRIVATE_BRANCH_AHEAD
        elif attack_status.target_confirmations_reached:
            self._state = DoubleSpendScenarioState.TARGET_REACHED
        elif attack_status.public_confirmations > 0:
            self._state = DoubleSpendScenarioState.CONFIRMING
        else:
            self._state = DoubleSpendScenarioState.PAYMENT_PENDING

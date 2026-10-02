from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from random import Random

from src.simulator.attacks import SelfishMiningParameters
from src.simulator.blockchain import Block
from src.simulator.consensus import ChainSelector
from src.simulator.forks import BlockchainBranch
from src.simulator.network import Node


class SelfishMiningSide(IntEnum):
    """Identify the winner of one selfish-mining PoW event."""

    PUBLIC = 0
    ATTACKER = 1


class SelfishMiningState(Enum):
    """Represent the internal state of a selfish-mining strategy."""

    INACTIVE = auto()
    PRIVATE_LEAD = auto()
    RACE = auto()
    PUBLISHED = auto()
    FINISHED = auto()


@dataclass(slots=True, frozen=True)
class SelfishMiningStatus:
    """Describe internal ground truth for one selfish-mining scenario."""

    state: SelfishMiningState
    private_height: int
    public_height: int
    private_lead: int
    private_blocks_mined: int
    public_blocks_mined: int
    blocks_published: int
    reorganizations: int
    race_wins: int
    race_losses: int
    finished: bool

    @property
    def successful(self) -> bool:
        """Return whether strategic publication changed the canonical chain."""

        return self.reorganizations > 0


class SelfishMiningAttack:
    """Model repeated block-withholding cycles under a selfish-mining policy."""

    def __init__(
        self,
        victim_node: Node,
        parameters: SelfishMiningParameters,
        attacker_id: str,
        seed: int,
    ) -> None:
        if not attacker_id:
            raise ValueError("Selfish-mining attacker ID must not be empty.")
        if seed < 0:
            raise ValueError("Selfish-mining random seed must be non-negative.")

        self._victim_node = victim_node
        self._parameters = parameters
        self._attacker_id = attacker_id
        self._race_rng = Random(seed)
        self._private_branch: BlockchainBranch | None = None
        self._state = SelfishMiningState.INACTIVE
        self._private_blocks_mined = 0
        self._public_blocks_mined = 0
        self._blocks_published = 0
        self._reorganizations = 0
        self._race_wins = 0
        self._race_losses = 0
        self._finished = False

    @property
    def started(self) -> bool:
        return self._private_branch is not None

    @property
    def state(self) -> SelfishMiningState:
        return self._state

    @property
    def private_branch(self) -> BlockchainBranch:
        if self._private_branch is None:
            raise RuntimeError("Selfish-mining attack has not been started.")
        return self._private_branch

    @property
    def private_lead(self) -> int:
        if not self.started:
            return 0
        return self.private_branch.height - self._victim_node.height

    def start(self) -> None:
        if self.started:
            raise RuntimeError("Selfish-mining attack has already been started.")
        self._reset_private_branch()

    def process_mining_event(
        self,
        timestamp: float,
        mining_side: SelfishMiningSide,
        public_miner_id: str | None = None,
    ) -> Block:
        """Process exactly one PoW winner and apply the selfish strategy."""

        self._require_active()
        if self._state is SelfishMiningState.PUBLISHED:
            self._reset_private_branch()

        if mining_side is SelfishMiningSide.ATTACKER:
            was_race = self._state is SelfishMiningState.RACE
            block = self._mine_private_block(timestamp, self._attacker_id)
            if was_race:
                self._race_wins += 1
                self._publish_if_longer()
            return block

        if mining_side is not SelfishMiningSide.PUBLIC:
            raise ValueError("Unsupported selfish-mining side.")
        if not public_miner_id:
            raise ValueError("Public miner identifier is required.")

        if self._state is SelfishMiningState.RACE:
            if self._race_rng.random() < self._parameters.gamma:
                block = self._mine_private_block(timestamp, public_miner_id)
                self._public_blocks_mined += 1
                self._race_wins += 1
                self._publish_if_longer()
                return block

            block = self._mine_public_block(timestamp, public_miner_id)
            self._race_losses += 1
            self._reset_private_branch()
            return block

        lead_before = self.private_lead
        block = self._mine_public_block(timestamp, public_miner_id)

        if lead_before == 1 and self.private_branch.length > 0:
            self._state = SelfishMiningState.RACE
        elif lead_before >= 2:
            self._publish_if_longer()
        else:
            self._reset_private_branch()

        return block

    def finish(self) -> None:
        """Finish the attack window without forcing private publication."""

        if not self.started:
            return
        self._finished = True
        self._state = SelfishMiningState.FINISHED

    def status(self) -> SelfishMiningStatus:
        private_height = (
            self.private_branch.height if self.started else self._victim_node.height
        )
        return SelfishMiningStatus(
            state=self._state,
            private_height=private_height,
            public_height=self._victim_node.height,
            private_lead=private_height - self._victim_node.height if self.started else 0,
            private_blocks_mined=self._private_blocks_mined,
            public_blocks_mined=self._public_blocks_mined,
            blocks_published=self._blocks_published,
            reorganizations=self._reorganizations,
            race_wins=self._race_wins,
            race_losses=self._race_losses,
            finished=self._finished,
        )

    def _mine_private_block(self, timestamp: float, miner_id: str) -> Block:
        block = Block(
            height=self.private_branch.height + 1,
            previous_hash=self.private_branch.tip_hash,
            miner_id=miner_id,
            timestamp=timestamp,
        )
        block.finalize()
        self.private_branch.add_block(block)
        if miner_id == self._attacker_id:
            self._private_blocks_mined += 1
        self._state = SelfishMiningState.PRIVATE_LEAD
        return block

    def _mine_public_block(self, timestamp: float, miner_id: str) -> Block:
        block = Block(
            height=self._victim_node.height + 1,
            previous_hash=self._victim_node.tip_hash,
            miner_id=miner_id,
            timestamp=timestamp,
        )
        block.finalize()
        if not self._victim_node.receive_block(block):
            raise RuntimeError("Public selfish-mining block was not accepted.")
        self._public_blocks_mined += 1
        return block

    def _publish_if_longer(self) -> bool:
        result = ChainSelector.adopt_if_longer(self._victim_node, self.private_branch)
        if not result.adopted:
            return False
        self._blocks_published += result.added_blocks
        if result.removed_blocks > 0:
            self._reorganizations += 1
        self._state = SelfishMiningState.PUBLISHED
        return True

    def _reset_private_branch(self) -> None:
        self._private_branch = BlockchainBranch(
            branch_id=f"selfish_{self._attacker_id}",
            parent_block=self._victim_node.blockchain.tip,
        )
        self._state = SelfishMiningState.PRIVATE_LEAD

    def _require_active(self) -> None:
        if not self.started:
            raise RuntimeError("Selfish-mining attack has not been started.")
        if self._finished:
            raise RuntimeError("Selfish-mining attack has already finished.")

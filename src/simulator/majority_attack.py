from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum, auto

from src.simulator.attacks import MajorityAttackParameters
from src.simulator.blockchain import Block
from src.simulator.consensus import ChainSelector
from src.simulator.forks import BlockchainBranch
from src.simulator.network import Node


class MajorityMiningSide(IntEnum):
    """Identify the winner of one majority-attack PoW event."""

    PUBLIC = 0
    ATTACKER = 1


class MajorityAttackState(Enum):
    """Represent the internal state of a majority-hash-power attack."""

    INACTIVE = auto()
    COMPETING = auto()
    AHEAD = auto()
    PUBLISHED = auto()
    FINISHED = auto()


@dataclass(slots=True, frozen=True)
class MajorityAttackStatus:
    """Describe internal ground truth for one majority-attack scenario."""

    state: MajorityAttackState
    attacker_height: int
    public_height: int
    attacker_lead: int
    attacker_blocks_mined: int
    honest_blocks_mined: int
    blocks_published: int
    reorganizations: int
    finished: bool

    @property
    def successful(self) -> bool:
        return self.blocks_published > 0


class MajorityAttack:
    """Model a majority coalition privately competing with the public chain."""

    def __init__(
        self,
        victim_node: Node,
        parameters: MajorityAttackParameters,
        attacker_id: str,
    ) -> None:
        if not attacker_id:
            raise ValueError("Majority-attack attacker ID must not be empty.")
        self._victim_node = victim_node
        self._parameters = parameters
        self._attacker_id = attacker_id
        self._attacker_branch: BlockchainBranch | None = None
        self._state = MajorityAttackState.INACTIVE
        self._attacker_blocks_mined = 0
        self._honest_blocks_mined = 0
        self._blocks_published = 0
        self._reorganizations = 0
        self._public_blocks_in_cycle = 0
        self._finished = False

    @property
    def started(self) -> bool:
        return self._attacker_branch is not None

    @property
    def state(self) -> MajorityAttackState:
        return self._state

    @property
    def attacker_branch(self) -> BlockchainBranch:
        if self._attacker_branch is None:
            raise RuntimeError("Majority attack has not been started.")
        return self._attacker_branch

    @property
    def attacker_lead(self) -> int:
        if not self.started:
            return 0
        return self.attacker_branch.height - self._victim_node.height

    def start(self) -> None:
        if self.started:
            raise RuntimeError("Majority attack has already been started.")
        self._reset_attacker_branch()

    def process_mining_event(
        self,
        timestamp: float,
        mining_side: MajorityMiningSide,
        public_miner_id: str | None = None,
    ) -> Block:
        """Process exactly one attacker or public PoW winner."""

        self._require_active()
        if self._state is MajorityAttackState.PUBLISHED:
            self._reset_attacker_branch()

        if mining_side is MajorityMiningSide.ATTACKER:
            block = Block(
                height=self.attacker_branch.height + 1,
                previous_hash=self.attacker_branch.tip_hash,
                miner_id=self._attacker_id,
                timestamp=timestamp,
            )
            block.finalize()
            self.attacker_branch.add_block(block)
            self._attacker_blocks_mined += 1
            self._update_state()
            if (
                self.attacker_lead >= 2
                or (self._public_blocks_in_cycle > 0 and self.attacker_lead > 0)
            ):
                self._publish_if_longer()
            return block

        if mining_side is not MajorityMiningSide.PUBLIC:
            raise ValueError("Unsupported majority-attack mining side.")
        if not public_miner_id:
            raise ValueError("Public miner identifier is required.")

        block = Block(
            height=self._victim_node.height + 1,
            previous_hash=self._victim_node.tip_hash,
            miner_id=public_miner_id,
            timestamp=timestamp,
        )
        block.finalize()
        if not self._victim_node.receive_block(block):
            raise RuntimeError("Public majority-attack block was not accepted.")
        self._honest_blocks_mined += 1
        self._public_blocks_in_cycle += 1
        self._update_state()
        return block

    def finish(self) -> None:
        """Finish the attack window without forcing private publication."""

        if not self.started:
            return
        self._finished = True
        self._state = MajorityAttackState.FINISHED

    def status(self) -> MajorityAttackStatus:
        attacker_height = (
            self.attacker_branch.height if self.started else self._victim_node.height
        )
        return MajorityAttackStatus(
            state=self._state,
            attacker_height=attacker_height,
            public_height=self._victim_node.height,
            attacker_lead=attacker_height - self._victim_node.height if self.started else 0,
            attacker_blocks_mined=self._attacker_blocks_mined,
            honest_blocks_mined=self._honest_blocks_mined,
            blocks_published=self._blocks_published,
            reorganizations=self._reorganizations,
            finished=self._finished,
        )

    def _publish_if_longer(self) -> bool:
        result = ChainSelector.adopt_if_longer(self._victim_node, self.attacker_branch)
        if not result.adopted:
            return False
        self._blocks_published += result.added_blocks
        if result.removed_blocks > 0:
            self._reorganizations += 1
        self._state = MajorityAttackState.PUBLISHED
        return True

    def _reset_attacker_branch(self) -> None:
        self._attacker_branch = BlockchainBranch(
            branch_id=f"majority_{self._attacker_id}",
            parent_block=self._victim_node.blockchain.tip,
        )
        self._public_blocks_in_cycle = 0
        self._state = MajorityAttackState.COMPETING

    def _update_state(self) -> None:
        self._state = (
            MajorityAttackState.AHEAD
            if self.attacker_lead > 0
            else MajorityAttackState.COMPETING
        )

    def _require_active(self) -> None:
        if not self.started:
            raise RuntimeError("Majority attack has not been started.")
        if self._finished:
            raise RuntimeError("Majority attack has already finished.")

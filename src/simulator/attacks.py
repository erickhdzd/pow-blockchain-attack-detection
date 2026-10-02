from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

class AttackType(IntEnum):
    """Define ground-truth classes used by the blockchain attack simulator."""

    NORMAL = 0
    DOUBLE_SPEND = 1
    SELFISH_MINING = 2
    MAJORITY_ATTACK = 3


@dataclass(slots=True, frozen=True)
class AttackWindow:
    """Define the inclusive block interval in which an attack is active."""

    start_block: int
    end_block: int

    def __post_init__(self) -> None:
        if self.start_block < 0:
            raise ValueError("Attack start block must be non-negative.")

        if self.end_block < self.start_block:
            raise ValueError(
                "Attack end block must not precede the attack start block."
            )

    def contains(self, block_height: int) -> bool:
        """Return whether a block height lies inside the attack window."""

        if block_height < 0:
            raise ValueError("Block height must be non-negative.")

        return self.start_block <= block_height <= self.end_block


@dataclass(slots=True, frozen=True)
class DoubleSpendParameters:
    """Store validated parameters for a double-spending attack."""

    transaction_value: float
    target_confirmations: int
    attacker_hash_share: float = 0.30

    def __post_init__(self) -> None:
        if (
            not isfinite(self.transaction_value)
            or self.transaction_value <= 0.0
        ):
            raise ValueError(
                "Double-spend transaction value must be a finite positive "
                "value."
            )

        if self.target_confirmations <= 0:
            raise ValueError(
                "Double-spend target confirmations must be positive."
            )

        if (
            not isfinite(self.attacker_hash_share)
            or not 0.0 < self.attacker_hash_share < 0.5
        ):
            raise ValueError(
                "Double-spend attacker hash share must be a finite value "
                "in the interval (0, 0.5)."
            )

@dataclass(slots=True, frozen=True)
class SelfishMiningParameters:
    """Store validated parameters for a selfish-mining attack."""

    gamma: float
    attacker_hash_share: float = 0.30

    def __post_init__(self) -> None:
        if not isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError(
                "Selfish-mining gamma must be a finite value between 0 and 1."
            )

        if (
            not isfinite(self.attacker_hash_share)
            or not 0.0 < self.attacker_hash_share < 0.5
        ):
            raise ValueError(
                "Selfish-mining attacker hash share must be a finite value "
                "in the interval (0, 0.5)."
            )


@dataclass(slots=True, frozen=True)
class MajorityAttackParameters:
    """Store internal parameters for a simulated majority attack."""

    attacker_hash_share: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.attacker_hash_share)
            or not 0.5 < self.attacker_hash_share <= 1.0
        ):
            raise ValueError(
                "Majority-attack hash share must be finite and greater than "
                "0.5 and at most 1.0."
            )


@dataclass(slots=True, frozen=True)
class AttackConfiguration:
    """Store validated ground-truth configuration for one simulation scenario."""

    attack_type: AttackType
    window: AttackWindow
    double_spend: DoubleSpendParameters
    selfish_mining: SelfishMiningParameters
    majority_attack: MajorityAttackParameters

    def is_active(self, block_height: int) -> bool:
        """Return whether the configured attack is active at a block height."""

        if self.attack_type is AttackType.NORMAL:
            return False

        return self.window.contains(block_height)

    def label_at(self, block_height: int) -> AttackType:
        """Return the ground-truth class at a given block height."""

        if self.is_active(block_height):
            return self.attack_type

        return AttackType.NORMAL
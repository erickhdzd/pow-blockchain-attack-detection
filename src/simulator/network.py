from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from random import Random

from src.simulator.blockchain import Block, Blockchain


@dataclass(slots=True, frozen=True)
class NetworkParameters:
    """Store validated public-network propagation parameters."""

    base_latency_ms: float
    latency_std_ms: float
    packet_loss_probability: float

    def __post_init__(self) -> None:
        if not isfinite(self.base_latency_ms) or self.base_latency_ms < 0.0:
            raise ValueError(
                "Base network latency must be a finite non-negative value."
            )

        if not isfinite(self.latency_std_ms) or self.latency_std_ms < 0.0:
            raise ValueError(
                "Network latency standard deviation must be a finite "
                "non-negative value."
            )

        if (
            not isfinite(self.packet_loss_probability)
            or not 0.0 <= self.packet_loss_probability <= 1.0
        ):
            raise ValueError(
                "Packet loss probability must be a finite value in [0, 1]."
            )


@dataclass(slots=True, frozen=True)
class BlockDelivery:
    """Describe the observable result of one block-delivery attempt."""

    source_node_id: str
    target_node_id: str
    block_hash: str
    sent_at: float
    delivery_time: float
    latency_ms: float
    accepted: bool

    def __post_init__(self) -> None:
        if not self.source_node_id:
            raise ValueError("Source node identifier must not be empty.")

        if not self.target_node_id:
            raise ValueError("Target node identifier must not be empty.")

        if self.source_node_id == self.target_node_id:
            raise ValueError("Source and target node identifiers must differ.")

        if not self.block_hash:
            raise ValueError("Delivered block hash must not be empty.")

        if not isfinite(self.sent_at) or self.sent_at < 0.0:
            raise ValueError("Delivery send time must be a finite non-negative value.")

        if not isfinite(self.delivery_time) or self.delivery_time < self.sent_at:
            raise ValueError("Delivery time must be finite and not precede send time.")

        if not isfinite(self.latency_ms) or self.latency_ms < 0.0:
            raise ValueError("Delivery latency must be a finite non-negative value.")


@dataclass(order=True, slots=True)
class _PendingDelivery:
    """Store one internally scheduled block delivery."""

    delivery_time: float
    sequence_number: int
    source_node_id: str = field(compare=False)
    target_node_id: str = field(compare=False)
    sent_at: float = field(compare=False)
    latency_ms: float = field(compare=False)
    block: Block = field(compare=False)


class Node:
    """Represent a network node with its own canonical blockchain."""

    def __init__(
        self,
        node_id: str,
    ) -> None:
        if not node_id:
            raise ValueError("Node identifier must not be empty.")

        self._node_id = node_id
        self._blockchain = Blockchain()

    @property
    def node_id(self) -> str:
        """Return the node identifier."""

        return self._node_id

    @property
    def blockchain(self) -> Blockchain:
        """Return the node's local canonical blockchain."""

        return self._blockchain

    @blockchain.setter
    def blockchain(
        self,
        blockchain: Blockchain,
    ) -> None:
        """Replace the local canonical blockchain with a valid chain."""

        if not isinstance(blockchain, Blockchain):
            raise TypeError("Node blockchain must be a Blockchain instance.")

        if not blockchain.validate():
            raise ValueError("Node blockchain replacement must be valid.")

        self._blockchain = blockchain

    @property
    def height(self) -> int:
        """Return the local canonical-chain height."""

        return self._blockchain.height

    @property
    def tip_hash(self) -> str:
        """Return the local canonical tip hash."""

        return self._blockchain.tip.block_hash

    def receive_block(
        self,
        block: Block,
    ) -> bool:
        """Accept a valid direct extension of the local canonical chain."""

        if block.height != self.height + 1:
            return False

        if block.previous_hash != self.tip_hash:
            return False

        self._blockchain.add_block(block)

        return True

    def adopt_longer_chain(
        self,
        chain: Sequence[Block],
    ) -> bool:
        """Adopt a valid strictly longer canonical chain."""

        candidate_blocks = tuple(chain)

        if not candidate_blocks:
            raise ValueError("Candidate chain must contain at least the genesis block.")

        if len(candidate_blocks) <= len(self._blockchain.chain):
            return False

        candidate = Blockchain()

        if candidate_blocks[0].block_hash != candidate.chain[0].block_hash:
            return False

        try:
            for block in candidate_blocks[1:]:
                candidate.add_block(block)
        except ValueError:
            return False

        if not candidate.validate():
            return False

        self._blockchain = candidate

        return True


class Network:
    """Model reproducible delayed block propagation between blockchain nodes."""

    def __init__(
        self,
        nodes: list[Node],
        parameters: NetworkParameters,
        seed: int,
    ) -> None:
        if not nodes:
            raise ValueError("Network must contain at least one node.")

        if seed < 0:
            raise ValueError("Network random seed must be non-negative.")

        node_tuple = tuple(nodes)

        node_ids = [node.node_id for node in node_tuple]

        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Network node identifiers must be unique.")

        self._nodes = node_tuple
        self._nodes_by_id = {node.node_id: node for node in node_tuple}

        self._parameters = parameters
        self._seed = seed
        self._rng = Random(seed)

        self._pending_deliveries: list[_PendingDelivery] = []
        self._sequence_number = 0

        self._scheduled_delivery_count = 0
        self._dropped_delivery_count = 0
        self._completed_delivery_count = 0

        self._last_delivery_time = 0.0

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Return network nodes as an immutable sequence."""

        return self._nodes

    @property
    def parameters(self) -> NetworkParameters:
        """Return immutable network propagation parameters."""

        return self._parameters

    @property
    def seed(self) -> int:
        """Return the network random seed."""

        return self._seed

    @property
    def pending_delivery_count(self) -> int:
        """Return the number of currently scheduled deliveries."""

        return len(self._pending_deliveries)

    @property
    def scheduled_delivery_count(self) -> int:
        """Return the total number of successfully scheduled deliveries."""

        return self._scheduled_delivery_count

    @property
    def dropped_delivery_count(self) -> int:
        """Return the total number of deliveries dropped before scheduling."""

        return self._dropped_delivery_count

    @property
    def completed_delivery_count(self) -> int:
        """Return the total number of processed delivery attempts."""

        return self._completed_delivery_count

    @property
    def next_delivery_time(self) -> float | None:
        """Return the next scheduled delivery time, if one exists."""

        if not self._pending_deliveries:
            return None

        return self._pending_deliveries[0].delivery_time

    def get_node(
        self,
        node_id: str,
    ) -> Node:
        """Return a network node by identifier."""

        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown network node: {node_id!r}.") from exc

    def broadcast_block(
        self,
        source_node_id: str,
        block: Block,
        sent_at: float,
    ) -> int:
        """Accept a block at its source and schedule propagation to peers."""

        if not isfinite(sent_at) or sent_at < 0.0:
            raise ValueError("Broadcast time must be a finite non-negative value.")

        if sent_at < self._last_delivery_time:
            raise ValueError(
                "Broadcast time must not precede already processed network time."
            )

        source_node = self.get_node(source_node_id)

        if source_node.tip_hash != block.block_hash:
            accepted = source_node.receive_block(block)

            if not accepted:
                raise ValueError(
                    "Source node could not accept the block being broadcast."
                )

        scheduled = 0

        for target_node in self._nodes:
            if target_node.node_id == source_node_id:
                continue

            if self._packet_is_dropped():
                self._dropped_delivery_count += 1
                continue

            latency_ms = self._sample_latency_ms()

            delivery_time = sent_at + latency_ms / 1000.0

            pending = _PendingDelivery(
                delivery_time=delivery_time,
                sequence_number=self._sequence_number,
                source_node_id=source_node_id,
                target_node_id=target_node.node_id,
                sent_at=sent_at,
                latency_ms=latency_ms,
                block=block,
            )

            heapq.heappush(
                self._pending_deliveries,
                pending,
            )

            self._sequence_number += 1
            self._scheduled_delivery_count += 1
            scheduled += 1

        return scheduled

    def deliver_until(
        self,
        timestamp: float,
    ) -> tuple[BlockDelivery, ...]:
        """Process every scheduled delivery due at or before timestamp."""

        if not isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(
                "Network delivery time must be a finite non-negative value."
            )

        if timestamp < self._last_delivery_time:
            raise ValueError("Network delivery time must not move backwards.")

        results: list[BlockDelivery] = []

        while (
            self._pending_deliveries
            and self._pending_deliveries[0].delivery_time <= timestamp
        ):
            pending = heapq.heappop(self._pending_deliveries)

            target_node = self.get_node(pending.target_node_id)

            accepted = target_node.receive_block(pending.block)

            results.append(
                BlockDelivery(
                    source_node_id=pending.source_node_id,
                    target_node_id=pending.target_node_id,
                    block_hash=pending.block.block_hash,
                    sent_at=pending.sent_at,
                    delivery_time=pending.delivery_time,
                    latency_ms=pending.latency_ms,
                    accepted=accepted,
                )
            )

            self._completed_delivery_count += 1

        self._last_delivery_time = timestamp

        return tuple(results)

    def flush(
        self,
    ) -> tuple[BlockDelivery, ...]:
        """Process all currently pending deliveries."""

        if not self._pending_deliveries:
            return ()

        final_delivery_time = max(
            pending.delivery_time for pending in self._pending_deliveries
        )

        return self.deliver_until(final_delivery_time)

    def catch_up_node(
        self,
        target_node_id: str,
        source_node_id: str,
    ) -> bool:
        """Catch up a lagging node when its chain is a source-chain prefix."""

        if target_node_id == source_node_id:
            raise ValueError("Catch-up source and target nodes must differ.")

        source_node = self.get_node(source_node_id)
        target_node = self.get_node(target_node_id)

        source_chain = source_node.blockchain.chain
        target_chain = target_node.blockchain.chain

        if len(target_chain) > len(source_chain):
            return False

        source_prefix = source_chain[: len(target_chain)]

        if tuple(block.block_hash for block in source_prefix) != tuple(
            block.block_hash for block in target_chain
        ):
            return False

        for block in source_chain[len(target_chain) :]:
            if not target_node.receive_block(block):
                return False

        return True

    def synchronize_all_from(
        self,
        source_node_id: str,
    ) -> dict[str, bool]:
        """Attempt prefix-safe catch-up of every peer from one source node."""

        self.get_node(source_node_id)

        results: dict[str, bool] = {}

        for node in self._nodes:
            if node.node_id == source_node_id:
                continue

            results[node.node_id] = self.catch_up_node(
                target_node_id=node.node_id,
                source_node_id=source_node_id,
            )

        return results

    def reconcile_from(
        self,
        source_node_id: str,
    ) -> dict[str, bool]:
        """Reconcile peers with a strictly longer canonical source chain."""

        source_node = self.get_node(source_node_id)

        results: dict[str, bool] = {}

        for target_node in self._nodes:
            if target_node.node_id == source_node_id:
                continue

            if (
                target_node.height == source_node.height
                and target_node.tip_hash == source_node.tip_hash
            ):
                results[target_node.node_id] = True
                continue

            if self.catch_up_node(
                target_node_id=target_node.node_id,
                source_node_id=source_node_id,
            ):
                results[target_node.node_id] = True
                continue

            results[target_node.node_id] = target_node.adopt_longer_chain(
                source_node.blockchain.chain
            )

        return results

    def _sample_latency_ms(
        self,
    ) -> float:
        """Sample one non-negative Gaussian propagation latency."""

        latency_ms = self._rng.gauss(
            self._parameters.base_latency_ms,
            self._parameters.latency_std_ms,
        )

        return max(
            0.0,
            latency_ms,
        )

    def _packet_is_dropped(
        self,
    ) -> bool:
        """Return whether the next peer delivery is dropped."""

        return self._rng.random() < self._parameters.packet_loss_probability

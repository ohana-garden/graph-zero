"""
Graph Zero Canonical Mutation Log

Append-only, hash-chained. The Replay Theorem depends on this:
any two nodes with the same log produce the same state.

Nothing is deleted. Nothing is modified. Entries only append.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from graph_zero.core.mutation import (
    CanonicalMutation, MutationEnvelope, ImpactVector, GateError
)


@dataclass
class StateRoot:
    """A commitment to graph state at a specific log position."""
    root_hash: str
    at_entry_index: int
    at_entry_hash: str


class CanonicalLog:
    """The community's permanent, append-only mutation history.

    Every accepted change is recorded here in order.
    The hash chain ensures tamper detection.
    """

    def __init__(self, community_id: str):
        self.community_id = community_id
        self.entries: list[CanonicalMutation] = []
        self._seq_tracker: dict[tuple[bytes, bytes], int] = {}  # (author, device) -> last seq

    @property
    def length(self) -> int:
        return len(self.entries)

    @property
    def head_hash(self) -> str:
        """Hash of the most recent entry, or empty string for empty log."""
        if not self.entries:
            return ""
        return self.entries[-1].canonical_hash

    @property
    def state_root(self) -> StateRoot:
        """Current state root commitment."""
        if not self.entries:
            return StateRoot(root_hash=self._compute_state_root(), at_entry_index=-1, at_entry_hash="")
        return StateRoot(
            root_hash=self._compute_state_root(),
            at_entry_index=len(self.entries) - 1,
            at_entry_hash=self.entries[-1].canonical_hash
        )

    def _compute_state_root(self) -> str:
        """Deterministic state root from the log.

        For now, this is a hash of all canonical hashes in order.
        In production with FalkorDB, this would be computed from
        the materialized graph (sorted node_hashes || sorted edge_hashes).
        """
        h = hashlib.sha256()
        h.update(self.community_id.encode())
        for entry in self.entries:
            h.update(entry.canonical_hash.encode())
        return h.hexdigest()

    def check_seq(self, author_key: bytes, device_id: bytes, seq: int) -> bool:
        """Check if this sequence number is valid (monotonically increasing).
        Returns True if valid, False if replay."""
        key = (author_key, device_id)
        last_seq = self._seq_tracker.get(key, -1)
        return seq > last_seq

    def append(self, envelope: MutationEnvelope, impact_vector: ImpactVector,
               write_set: set[str],
               conflict_ref: Optional[str] = None,
               quarantine_flags: Optional[list[str]] = None) -> CanonicalMutation:
        """Append an accepted mutation to the log.

        This should ONLY be called after the gate has accepted the mutation.
        """
        # Update sequence tracker
        key = (envelope.author_key, envelope.device_id)
        self._seq_tracker[key] = envelope.seq

        # Build canonical mutation
        canonical = CanonicalMutation(
            envelope=envelope,
            write_set=write_set,
            impact_vector=impact_vector,
            conflict_ref=conflict_ref,
            quarantine_flags=quarantine_flags or [],
            canonical_prev=self.head_hash,
        )
        canonical.canonical_hash = canonical.compute_hash()
        self.entries.append(canonical)
        return canonical

    def verify_chain(self) -> bool:
        """Verify the hash chain integrity.
        Returns True if every entry's prev_hash matches the previous entry's hash."""
        for i, entry in enumerate(self.entries):
            if i == 0:
                if entry.canonical_prev != "":
                    return False
            else:
                if entry.canonical_prev != self.entries[i - 1].canonical_hash:
                    return False
            # Recompute hash and verify
            expected_hash = entry.compute_hash()
            if entry.canonical_hash != expected_hash:
                return False
        return True

    def get_entry(self, index: int) -> Optional[CanonicalMutation]:
        """Get a specific entry by index."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def entries_since(self, after_hash: str) -> list[CanonicalMutation]:
        """Get all entries after a given hash. For sync."""
        if after_hash == "":
            return list(self.entries)
        for i, entry in enumerate(self.entries):
            if entry.canonical_hash == after_hash:
                return list(self.entries[i + 1:])
        return []  # hash not found — diverged

    def find_common_prefix(self, other_head_hash: str) -> Optional[int]:
        """Find the index where this log and another diverge.
        Returns None if no common prefix found."""
        for i, entry in enumerate(self.entries):
            if entry.canonical_hash == other_head_hash:
                return i
        return None

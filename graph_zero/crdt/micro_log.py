"""
Graph Zero Micro-Log and Merge Algorithm

When a device goes offline, it accumulates mutations in a micro-log.
On reconnect, the merge algorithm processes each entry through the gate.

The merge is deterministic: any node processing the same micro-log
against the same state produces the same result.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

from graph_zero.core.identity import AgentIdentity
from graph_zero.core.mutation import (
    MutationEnvelope, CanonicalMutation, GateResult, GateError
)
from graph_zero.core.log import CanonicalLog
from graph_zero.core.gate import Gate, CommunityState


@dataclass
class MicroLog:
    """A sealed package of mutations created while offline."""
    community_id: str
    author_key: bytes
    device_id: bytes
    entries: list[MutationEnvelope]
    micro_sig: bytes           # author signs hash of all entries

    @staticmethod
    def create(identity: AgentIdentity, community_id: str, device_id: bytes,
               entries: list[MutationEnvelope]) -> 'MicroLog':
        """Create a signed micro-log."""
        # Compute hash of all entries
        h = hashlib.sha256()
        for e in entries:
            h.update(e.mutation_hash)
        all_hash = h.digest()

        return MicroLog(
            community_id=community_id,
            author_key=identity.public_key,
            device_id=device_id,
            entries=entries,
            micro_sig=identity.sign(all_hash)
        )

    def verify_sig(self) -> bool:
        """Verify the micro-log signature."""
        h = hashlib.sha256()
        for e in self.entries:
            h.update(e.mutation_hash)
        all_hash = h.digest()
        return AgentIdentity.verify(self.author_key, all_hash, self.micro_sig)

    def verify_ordering(self) -> bool:
        """Verify entries have strictly increasing seq."""
        if not self.entries:
            return True
        for i in range(1, len(self.entries)):
            if self.entries[i].seq <= self.entries[i - 1].seq:
                return False
        return True


# ============================================================
# Merge Result
# ============================================================

@dataclass
class MergeEntry:
    """Result of merging one micro-log entry."""
    envelope: MutationEnvelope
    accepted: bool
    canonical: Optional[CanonicalMutation] = None
    error: Optional[GateError] = None
    error_detail: str = ""
    conflict_created: bool = False


@dataclass
class MergeResult:
    """Result of merging an entire micro-log."""
    success: bool
    entries: list[MergeEntry]
    accepted_count: int = 0
    rejected_count: int = 0
    error: str = ""


# ============================================================
# Sync State Machine
# ============================================================

class SyncState(Enum):
    """States in the sync state machine."""
    IDLE = auto()
    REQUEST_PREFIX = auto()
    SEND_MISSING = auto()
    VALIDATE_CHAIN = auto()
    APPLY_ENTRIES = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class SyncSession:
    """Tracks state of a sync session."""
    state: SyncState = SyncState.IDLE
    common_prefix_index: Optional[int] = None
    entries_to_apply: list[MutationEnvelope] = field(default_factory=list)
    applied_count: int = 0
    error_detail: str = ""


# ============================================================
# Merge Function
# ============================================================

def merge_micro_log(gate: Gate, log: CanonicalLog,
                    micro_log: MicroLog) -> MergeResult:
    """Merge a micro-log into the canonical log.

    Deterministic: same inputs → same outputs.
    Replayable: any node can recompute.
    Bounded: per-entry checks only.
    """
    from graph_zero.core.gate import process_mutation

    # Validate micro-log integrity
    if micro_log.community_id != log.community_id:
        return MergeResult(success=False, entries=[],
                           error="Community ID mismatch")

    if not micro_log.verify_sig():
        return MergeResult(success=False, entries=[],
                           error="Invalid micro-log signature")

    if not micro_log.verify_ordering():
        return MergeResult(success=False, entries=[],
                           error="Micro-log entries not in strictly increasing seq order")

    # Process each entry through the gate
    results = []
    accepted = 0
    rejected = 0

    for entry in micro_log.entries:
        result, canonical = process_mutation(gate, log, entry)

        merge_entry = MergeEntry(
            envelope=entry,
            accepted=result.accepted,
            canonical=canonical,
            error=result.error,
            error_detail=result.error_detail,
        )

        if result.accepted:
            accepted += 1
        else:
            rejected += 1

        results.append(merge_entry)

    return MergeResult(
        success=True,
        entries=results,
        accepted_count=accepted,
        rejected_count=rejected,
    )


# ============================================================
# Sync Protocol (simplified)
# ============================================================

def sync_logs(source: CanonicalLog, target: CanonicalLog,
              gate: Gate) -> SyncSession:
    """Sync from source to target.

    Follows the state machine from Section 6.6:
    IDLE → REQUEST_PREFIX → SEND_MISSING → VALIDATE_CHAIN → APPLY_ENTRIES → COMPLETE
    """
    from graph_zero.core.gate import process_mutation

    session = SyncSession(state=SyncState.REQUEST_PREFIX)

    # REQUEST_PREFIX: find common log position
    if target.length == 0:
        session.common_prefix_index = -1
    else:
        idx = source.find_common_prefix(target.head_hash)
        if idx is None:
            # No common prefix — catastrophic divergence
            session.state = SyncState.ERROR
            session.error_detail = "No common prefix — logs have diverged"
            return session
        session.common_prefix_index = idx

    # SEND_MISSING: get entries target doesn't have
    session.state = SyncState.SEND_MISSING
    missing = source.entries_since(
        target.head_hash if target.length > 0 else ""
    )
    session.entries_to_apply = [e.envelope for e in missing]

    # VALIDATE_CHAIN: verify hash chain of missing entries
    session.state = SyncState.VALIDATE_CHAIN
    for i, entry in enumerate(missing):
        if i == 0:
            expected_prev = target.head_hash if target.length > 0 else ""
        else:
            expected_prev = missing[i - 1].canonical_hash
        if entry.canonical_prev != expected_prev:
            session.state = SyncState.ERROR
            session.error_detail = f"Hash chain broken at index {i}"
            return session

    # APPLY_ENTRIES: apply each entry through the gate
    session.state = SyncState.APPLY_ENTRIES
    for envelope in session.entries_to_apply:
        result, _ = process_mutation(gate, target, envelope)
        if result.accepted:
            session.applied_count += 1
        # Note: rejections during sync are logged, not fatal
        # (the source already accepted them, so rejection means state divergence)

    # COMPLETE: verify state roots match
    session.state = SyncState.COMPLETE
    if source.state_root.root_hash != target.state_root.root_hash:
        # This SHOULD NOT happen if the Replay Theorem holds
        session.state = SyncState.ERROR
        session.error_detail = "State roots diverge after sync — Replay Theorem violated"

    return session

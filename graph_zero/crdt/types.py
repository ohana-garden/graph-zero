"""
Graph Zero CRDT Types

Conflict-free Replicated Data Types that guarantee offline changes
merge correctly regardless of arrival order.

These are the math that makes 85% of community life flow freely
even when devices are offline for days.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ============================================================
# G-Set (Grow-Only Set)
# ============================================================

class GSet:
    """Grow-only set. Items can be added, never removed.
    Merge = set union.

    Used for: messages, acknowledgments, check-ins, capability
    declarations, sensor readings.
    """

    def __init__(self, items: Optional[set] = None):
        self._items: set = items or set()

    def add(self, item: Any) -> None:
        """Add an item (idempotent)."""
        if isinstance(item, dict):
            item = tuple(sorted(item.items()))
        self._items.add(item)

    def contains(self, item: Any) -> bool:
        if isinstance(item, dict):
            item = tuple(sorted(item.items()))
        return item in self._items

    @property
    def items(self) -> frozenset:
        return frozenset(self._items)

    def merge(self, other: 'GSet') -> 'GSet':
        """Merge two G-Sets. Commutative, associative, idempotent."""
        return GSet(self._items | other._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other) -> bool:
        if not isinstance(other, GSet):
            return False
        return self._items == other._items


# ============================================================
# LWW-Register (Last-Writer-Wins Register)
# ============================================================

@dataclass
class LWWRegister:
    """Last-Writer-Wins Register with Hybrid Logical Clock.
    Only the latest value matters.
    Merge = choose value with greatest (hlc, author_tiebreak).

    Used for: status, location — fields where the latest value
    fully supersedes prior values.
    """
    value: Any = None
    hlc: int = 0
    author: str = ""  # tiebreak: lexicographic on author key hash

    def set(self, value: Any, hlc: int, author: str) -> None:
        """Set the register value."""
        if hlc > self.hlc or (hlc == self.hlc and author > self.author):
            self.value = value
            self.hlc = hlc
            self.author = author

    def merge(self, other: 'LWWRegister') -> 'LWWRegister':
        """Merge two LWW registers. Commutative, associative, idempotent."""
        if other.hlc > self.hlc or (other.hlc == self.hlc and other.author > self.author):
            return LWWRegister(value=other.value, hlc=other.hlc, author=other.author)
        return LWWRegister(value=self.value, hlc=self.hlc, author=self.author)

    def __eq__(self, other) -> bool:
        if not isinstance(other, LWWRegister):
            return False
        return self.value == other.value and self.hlc == other.hlc


# ============================================================
# MV-Register (Multi-Value Register)
# ============================================================

@dataclass(frozen=True)
class MVValue:
    """A single value in an MV-Register with metadata."""
    value: Any
    hlc: int
    author: str

    def __hash__(self):
        return hash((str(self.value), self.hlc, self.author))


class MVRegister:
    """Multi-Value Register. Retains ALL concurrent values.
    Collapse only via explicit Resolve.

    Used for: surplus declarations, need declarations, availability
    — fields where concurrent offline values are ALL potentially
    valid and discarding one could affect who gets help.

    Compaction rules:
    - Same-author auto-collapse: if all concurrent values from same
      author and causally ordered, keep only the latest.
    - Cardinality cap: max 5 concurrent values per key.
    - Staleness expiry: values older than 30 days auto-archived.
    """

    MAX_CONCURRENT = 5
    STALENESS_MS = 30 * 24 * 60 * 60 * 1000  # 30 days in ms

    def __init__(self, values: Optional[set] = None):
        self._values: set[MVValue] = values or set()

    def set(self, value: Any, hlc: int, author: str) -> None:
        """Add a concurrent value."""
        mv = MVValue(value=value, hlc=hlc, author=author)
        self._values.add(mv)
        self._compact(hlc)

    def _compact(self, current_hlc: int) -> None:
        """Apply compaction rules."""
        # Same-author auto-collapse: keep only latest per author
        by_author: dict[str, list[MVValue]] = {}
        for v in self._values:
            by_author.setdefault(v.author, []).append(v)

        compacted = set()
        for author, vals in by_author.items():
            if len(vals) > 1:
                # Keep only the latest from this author
                latest = max(vals, key=lambda x: x.hlc)
                compacted.add(latest)
            else:
                compacted.add(vals[0])

        # Staleness expiry
        compacted = {v for v in compacted if (current_hlc - v.hlc) < self.STALENESS_MS}

        # Cardinality cap: evict oldest if > MAX_CONCURRENT
        if len(compacted) > self.MAX_CONCURRENT:
            sorted_vals = sorted(compacted, key=lambda x: x.hlc, reverse=True)
            compacted = set(sorted_vals[:self.MAX_CONCURRENT])

        self._values = compacted

    @property
    def values(self) -> frozenset[MVValue]:
        return frozenset(self._values)

    @property
    def current_values(self) -> list[Any]:
        """Get all current values (for display)."""
        return [v.value for v in sorted(self._values, key=lambda x: x.hlc, reverse=True)]

    def resolve(self, chosen_value: Any) -> None:
        """Resolve to a single value (requires external Class B mutation)."""
        chosen = [v for v in self._values if v.value == chosen_value]
        if chosen:
            self._values = {chosen[0]}

    def merge(self, other: 'MVRegister') -> 'MVRegister':
        """Merge two MV-Registers. Union of all values, then compact."""
        merged = MVRegister(self._values | other._values)
        now = max((v.hlc for v in merged._values), default=0)
        merged._compact(now)
        return merged

    def __len__(self) -> int:
        return len(self._values)


# ============================================================
# PN-Counter (Positive-Negative Counter)
# ============================================================

class PNCounter:
    """Positive-Negative Counter for bounded offline spending.

    Used for: offline Kala tip budgets.
    Merge: component-wise max of positive/negative vectors.

    Each device tracks its own positive (credits) and negative (debits).
    The counter value = sum(positives) - sum(negatives).
    """

    def __init__(self):
        self._positive: dict[str, float] = {}  # device_id -> total credits
        self._negative: dict[str, float] = {}  # device_id -> total debits

    def increment(self, device_id: str, amount: float) -> None:
        """Add credits (positive)."""
        self._positive[device_id] = self._positive.get(device_id, 0) + amount

    def decrement(self, device_id: str, amount: float) -> None:
        """Add debits (negative)."""
        self._negative[device_id] = self._negative.get(device_id, 0) + amount

    @property
    def value(self) -> float:
        """Current counter value."""
        return sum(self._positive.values()) - sum(self._negative.values())

    def merge(self, other: 'PNCounter') -> 'PNCounter':
        """Merge two PN-Counters. Component-wise max."""
        result = PNCounter()
        all_devices = set(self._positive.keys()) | set(other._positive.keys())
        for d in all_devices:
            result._positive[d] = max(
                self._positive.get(d, 0),
                other._positive.get(d, 0)
            )
        all_devices = set(self._negative.keys()) | set(other._negative.keys())
        for d in all_devices:
            result._negative[d] = max(
                self._negative.get(d, 0),
                other._negative.get(d, 0)
            )
        return result


# ============================================================
# Budget Pool (for offline Kala spending)
# ============================================================

@dataclass
class BudgetPool:
    """Pre-allocated offline spending budget.

    When a device goes offline, it reserves a spending pool.
    Spending beyond the pool → gate REJECT on reconnect.
    """
    device_id: str
    agent_key_hash: str
    allocated: float           # pre-split budget amount
    spent: float = 0.0         # running total of offline spending
    created_at: int = 0        # when the pool was allocated

    @property
    def remaining(self) -> float:
        return max(0, self.allocated - self.spent)

    def spend(self, amount: float) -> bool:
        """Try to spend from the pool. Returns False if insufficient."""
        if amount > self.remaining:
            return False
        self.spent += amount
        return True

    @staticmethod
    def allocate(agent_key_hash: str, device_id: str,
                 balance: float, rate: float = 0.1, minimum: float = 10.0) -> 'BudgetPool':
        """Allocate a budget pool. Default: 10% of balance or 10 Kala, whichever is greater."""
        allocated = max(balance * rate, minimum)
        allocated = min(allocated, balance)  # can't allocate more than balance
        return BudgetPool(
            device_id=device_id,
            agent_key_hash=agent_key_hash,
            allocated=allocated,
            created_at=int(time.time() * 1000)
        )

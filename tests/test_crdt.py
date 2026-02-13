"""Tests for CRDT types and micro-log merge."""

import sys, os
sys.path.insert(0, '/home/claude')

from graph_zero.crdt.types import GSet, LWWRegister, MVRegister, PNCounter, BudgetPool
from graph_zero.crdt.micro_log import MicroLog, merge_micro_log, SyncState, sync_logs
from graph_zero.core.identity import generate_identity
from graph_zero.core.mutation import (
    MutationEnvelope, MutationPayload, MutationType, CRDTType
)
from graph_zero.core.log import CanonicalLog
from graph_zero.core.gate import Gate, CommunityState, process_mutation


# ============================================================
# G-Set Tests
# ============================================================

def test_gset_basic():
    a = GSet()
    a.add("msg1")
    a.add("msg2")
    a.add("msg2")  # idempotent
    assert len(a) == 2
    assert a.contains("msg1")
    print("  ✓ gset_basic")

def test_gset_merge():
    a = GSet()
    a.add("msg1")
    a.add("msg2")
    b = GSet()
    b.add("msg2")
    b.add("msg3")
    merged = a.merge(b)
    assert len(merged) == 3
    # Commutativity
    assert a.merge(b) == b.merge(a)
    # Idempotency
    assert a.merge(a) == a
    print("  ✓ gset_merge (commutative, idempotent)")


# ============================================================
# LWW-Register Tests
# ============================================================

def test_lww_basic():
    r = LWWRegister()
    r.set("at home", hlc=100, author="alice")
    assert r.value == "at home"
    r.set("at garden", hlc=200, author="alice")
    assert r.value == "at garden"
    # Old timestamp doesn't overwrite
    r.set("at store", hlc=50, author="alice")
    assert r.value == "at garden"
    print("  ✓ lww_basic")

def test_lww_merge():
    a = LWWRegister(value="at home", hlc=100, author="alice")
    b = LWWRegister(value="at garden", hlc=200, author="alice")
    merged = a.merge(b)
    assert merged.value == "at garden"
    # Commutativity
    assert a.merge(b).value == b.merge(a).value
    print("  ✓ lww_merge (commutative)")

def test_lww_tiebreak():
    """Same HLC → lexicographic tiebreak on author."""
    a = LWWRegister(value="val_a", hlc=100, author="alice")
    b = LWWRegister(value="val_b", hlc=100, author="bob")
    merged = a.merge(b)
    assert merged.value == "val_b"  # bob > alice lexicographically
    print("  ✓ lww_tiebreak")


# ============================================================
# MV-Register Tests
# ============================================================

def test_mv_concurrent():
    """Concurrent values from different authors are both kept."""
    r = MVRegister()
    r.set("10 breadfruit", hlc=100, author="aunty")
    r.set("5 taro", hlc=100, author="uncle")
    assert len(r) == 2
    vals = r.current_values
    assert "10 breadfruit" in vals
    assert "5 taro" in vals
    print("  ✓ mv_concurrent")

def test_mv_same_author_collapse():
    """Same author's multiple values collapse to latest."""
    r = MVRegister()
    r.set("10 breadfruit", hlc=100, author="aunty")
    r.set("8 breadfruit", hlc=200, author="aunty")
    assert len(r) == 1
    assert r.current_values == ["8 breadfruit"]
    print("  ✓ mv_same_author_collapse")

def test_mv_cardinality_cap():
    """Max 5 concurrent values."""
    r = MVRegister()
    for i in range(7):
        r.set(f"val_{i}", hlc=100 + i, author=f"author_{i}")
    assert len(r) <= MVRegister.MAX_CONCURRENT
    print("  ✓ mv_cardinality_cap")

def test_mv_merge():
    a = MVRegister()
    a.set("breadfruit", hlc=100, author="aunty")
    b = MVRegister()
    b.set("taro", hlc=100, author="uncle")
    merged = a.merge(b)
    assert len(merged) == 2
    # Commutativity
    assert len(a.merge(b)) == len(b.merge(a))
    print("  ✓ mv_merge (commutative)")

def test_mv_resolve():
    """Explicit resolve collapses to chosen value."""
    r = MVRegister()
    r.set("breadfruit", hlc=100, author="aunty")
    r.set("taro", hlc=100, author="uncle")
    assert len(r) == 2
    r.resolve("taro")
    assert len(r) == 1
    assert r.current_values == ["taro"]
    print("  ✓ mv_resolve")


# ============================================================
# PN-Counter Tests
# ============================================================

def test_pn_counter():
    c = PNCounter()
    c.increment("device_a", 100)
    assert c.value == 100
    c.decrement("device_a", 30)
    assert c.value == 70
    c.decrement("device_b", 10)  # different device
    assert c.value == 60
    print("  ✓ pn_counter")

def test_pn_merge():
    a = PNCounter()
    a.increment("dev1", 100)
    a.decrement("dev1", 20)

    b = PNCounter()
    b.increment("dev1", 80)   # a saw 100, b saw 80 — max wins
    b.decrement("dev2", 10)

    merged = a.merge(b)
    # Positive: dev1 = max(100, 80) = 100
    # Negative: dev1 = max(20, 0) = 20, dev2 = max(0, 10) = 10
    assert merged.value == 100 - 20 - 10  # 70
    print("  ✓ pn_merge")


# ============================================================
# Budget Pool Tests
# ============================================================

def test_budget_pool():
    pool = BudgetPool.allocate("agent_abc", "dev1", balance=100.0)
    assert pool.allocated == 10.0  # 10% of 100
    assert pool.remaining == 10.0
    assert pool.spend(5.0)
    assert pool.remaining == 5.0
    assert not pool.spend(6.0)  # exceeds remaining
    assert pool.remaining == 5.0
    print("  ✓ budget_pool")

def test_budget_pool_minimum():
    pool = BudgetPool.allocate("agent_abc", "dev1", balance=50.0)
    assert pool.allocated == 10.0  # minimum is 10, 10% of 50 = 5 < 10
    print("  ✓ budget_pool_minimum")


# ============================================================
# Micro-Log Merge Tests
# ============================================================

def make_community_and_agent():
    state = CommunityState(community_id="test_community")
    log = CanonicalLog(community_id="test_community")
    gate = Gate(state, log)
    identity = generate_identity()
    device_id = os.urandom(16)
    key_hash = identity.key_hash
    state.registered_devices[key_hash] = {device_id}
    state.kala_balances[key_hash] = 100.0
    return state, log, gate, identity, device_id

def make_msg(identity, device_id, seq, content="hello"):
    payload = MutationPayload(
        mutation_type=MutationType.MESSAGE,
        data={"content": content},
        write_set={"message"},
        crdt_type=CRDTType.G_SET,
    )
    return MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=seq, base_root="", payload=payload,
    )

def test_micro_log_merge():
    """Offline micro-log merges correctly."""
    state, log, gate, identity, device_id = make_community_and_agent()

    # Create a micro-log (as if device was offline)
    entries = [make_msg(identity, device_id, seq=i, content=f"offline msg {i}")
               for i in range(1, 6)]
    ml = MicroLog.create(identity, "test_community", device_id, entries)

    assert ml.verify_sig()
    assert ml.verify_ordering()

    result = merge_micro_log(gate, log, ml)
    assert result.success
    assert result.accepted_count == 5
    assert result.rejected_count == 0
    assert log.length == 5
    assert log.verify_chain()
    print("  ✓ micro_log_merge (5 entries)")

def test_micro_log_bad_sig():
    """Tampered micro-log is rejected."""
    state, log, gate, identity, device_id = make_community_and_agent()
    entries = [make_msg(identity, device_id, seq=1)]
    ml = MicroLog(
        community_id="test_community",
        author_key=identity.public_key,
        device_id=device_id,
        entries=entries,
        micro_sig=b'\x00' * 64  # bad sig
    )
    result = merge_micro_log(gate, log, ml)
    assert not result.success
    assert "signature" in result.error.lower()
    print("  ✓ micro_log_bad_sig")

def test_micro_log_bad_ordering():
    """Out-of-order micro-log is rejected."""
    state, log, gate, identity, device_id = make_community_and_agent()
    entries = [
        make_msg(identity, device_id, seq=3),
        make_msg(identity, device_id, seq=1),  # out of order!
    ]
    ml = MicroLog.create(identity, "test_community", device_id, entries)
    result = merge_micro_log(gate, log, ml)
    assert not result.success
    assert "order" in result.error.lower()
    print("  ✓ micro_log_bad_ordering")


# ============================================================
# Sync Protocol Tests
# ============================================================

def test_sync_empty_target():
    """Sync from populated source to empty target."""
    state1, log1, gate1, identity, device_id = make_community_and_agent()

    # Add entries to source
    for seq in range(1, 6):
        m = make_msg(identity, device_id, seq)
        process_mutation(gate1, log1, m)

    # Create empty target with same state config
    state2 = CommunityState(community_id="test_community")
    state2.registered_devices[identity.key_hash] = {device_id}
    state2.kala_balances[identity.key_hash] = 100.0
    log2 = CanonicalLog(community_id="test_community")
    gate2 = Gate(state2, log2)

    session = sync_logs(log1, log2, gate2)
    assert session.state == SyncState.COMPLETE
    assert session.applied_count == 5
    assert log2.length == 5
    # Replay theorem: same state roots
    assert log1.state_root.root_hash == log2.state_root.root_hash
    print("  ✓ sync_empty_target")

def test_sync_partial():
    """Sync target that has some entries already."""
    state, log1, gate1, identity, device_id = make_community_and_agent()

    # Build 10 mutations
    mutations = [make_msg(identity, device_id, seq=i) for i in range(1, 11)]

    # Source has all 10
    for m in mutations:
        process_mutation(gate1, log1, m)

    # Target has first 5
    state2 = CommunityState(community_id="test_community")
    state2.registered_devices[identity.key_hash] = {device_id}
    state2.kala_balances[identity.key_hash] = 100.0
    log2 = CanonicalLog(community_id="test_community")
    gate2 = Gate(state2, log2)
    for m in mutations[:5]:
        process_mutation(gate2, log2, m)

    session = sync_logs(log1, log2, gate2)
    assert session.state == SyncState.COMPLETE
    assert session.applied_count == 5  # only the missing 5
    assert log2.length == 10
    assert log1.state_root.root_hash == log2.state_root.root_hash
    print("  ✓ sync_partial")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing CRDT types...\n")

    print("G-Set:")
    test_gset_basic()
    test_gset_merge()

    print("\nLWW-Register:")
    test_lww_basic()
    test_lww_merge()
    test_lww_tiebreak()

    print("\nMV-Register:")
    test_mv_concurrent()
    test_mv_same_author_collapse()
    test_mv_cardinality_cap()
    test_mv_merge()
    test_mv_resolve()

    print("\nPN-Counter:")
    test_pn_counter()
    test_pn_merge()

    print("\nBudget Pool:")
    test_budget_pool()
    test_budget_pool_minimum()

    print("\nMicro-Log Merge:")
    test_micro_log_merge()
    test_micro_log_bad_sig()
    test_micro_log_bad_ordering()

    print("\nSync Protocol:")
    test_sync_empty_target()
    test_sync_partial()

    print("\n" + "=" * 50)
    print("ALL CRDT TESTS PASSED ✓")
    print("=" * 50)

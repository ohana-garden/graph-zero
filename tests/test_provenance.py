"""Tests for provenance taint detection, authority closure, earned terrain, conflicts."""

import sys, time
sys.path.insert(0, '/home/claude')

from graph_zero.graph.backend import PropertyGraph
from graph_zero.graph.schema import (
    NT, ET, bootstrap_community, create_agent, add_terrain_node, connect_terrain
)
from graph_zero.provenance.provenance import (
    is_tainted, check_authority_closure, get_provenance_chain,
    promote_external_claim, create_conflict_set, resolve_conflict,
    create_challenge, get_open_conflicts, get_quarantine_flags,
    Contender, ConflictStatus
)

def make_graph():
    g = PropertyGraph()
    bootstrap_community(g, "comm1", "Test")
    return g

# ============================================================
# Taint Tests
# ============================================================

def test_bedrock_not_tainted():
    g = make_graph()
    add_terrain_node(g, "t1", "Sacred text passage", "bedrock", provenance_type="BEDROCK")
    assert not is_tainted(g, "t1")
    assert check_authority_closure(g, "t1")
    print("  ✓ bedrock_not_tainted")

def test_witness_not_tainted():
    g = make_graph()
    add_terrain_node(g, "t1", "Community knowledge", "community", provenance_type="WITNESS")
    assert not is_tainted(g, "t1")
    print("  ✓ witness_not_tainted")

def test_cross_verified_not_tainted():
    g = make_graph()
    add_terrain_node(g, "t1", "Plant is taro", "earned", provenance_type="CROSS_VERIFIED")
    assert not is_tainted(g, "t1")
    print("  ✓ cross_verified_not_tainted")

def test_inference_is_tainted():
    g = make_graph()
    add_terrain_node(g, "t1", "AI guess about something", "earned", provenance_type="INFERENCE")
    assert is_tainted(g, "t1")
    assert not check_authority_closure(g, "t1")
    print("  ✓ inference_is_tainted")

def test_source_unverified_is_tainted():
    g = make_graph()
    g.add_node("claim1", NT.EXTERNAL_CLAIM, {
        "source_text": "Some Wikipedia article",
        "provenance_type": "SOURCE_UNVERIFIED",
    })
    assert is_tainted(g, "claim1")
    print("  ✓ source_unverified_is_tainted")

def test_promoted_claim_not_tainted():
    """External claim promoted through verification is NOT tainted."""
    g = make_graph()

    # External claim (tainted)
    g.add_node("claim1", NT.EXTERNAL_CLAIM, {
        "source_text": "Taro grows in pH 5.5-6.5",
        "provenance_type": "SOURCE_UNVERIFIED",
    })
    assert is_tainted(g, "claim1")

    # Human witness verifies it
    g.add_node("wv1", NT.WITNESS_VERIFICATION, {
        "external_claim_id": "claim1",
        "verifier_key": "human_witness_001",
        "provenance_type": "WITNESS",
        "verified_at": int(time.time() * 1000),
    })
    g.add_edge("wv1", "claim1", ET.VERIFIES)

    # Promote to terrain
    terrain = promote_external_claim(
        g, "claim1", "wv1", "terrain_taro_ph",
        "Taro grows best in pH 5.5-6.5"
    )
    assert terrain is not None
    assert not is_tainted(g, "terrain_taro_ph")
    assert check_authority_closure(g, "terrain_taro_ph")
    print("  ✓ promoted_claim_not_tainted")

def test_cross_verified_promotion():
    """Claim verified by 2+ AI models (knowledge, not wisdom)."""
    g = make_graph()

    g.add_node("claim1", NT.EXTERNAL_CLAIM, {
        "source_text": "This plant is taro",
        "provenance_type": "SOURCE_UNVERIFIED",
    })

    g.add_node("cv1", NT.CROSS_VERIFICATION, {
        "target_claim_id": "claim1",
        "models": ["claude", "llama"],
        "providers": ["anthropic", "meta"],
        "agreement_score": 0.95,
        "provenance_type": "CROSS_VERIFIED",
    })
    g.add_edge("cv1", "claim1", ET.VERIFIES)

    terrain = promote_external_claim(
        g, "claim1", "cv1", "terrain_taro_id",
        "This plant is taro (cross-verified)"
    )
    assert terrain is not None
    assert not is_tainted(g, "terrain_taro_id")
    print("  ✓ cross_verified_promotion")

def test_provenance_chain():
    g = make_graph()
    add_terrain_node(g, "t_bedrock", "Sacred passage", "bedrock", provenance_type="BEDROCK")
    add_terrain_node(g, "t_community", "Local practice", "community", provenance_type="WITNESS")
    connect_terrain(g, "t_bedrock", "t_community", 0.8, "WITNESS")

    chain = get_provenance_chain(g, "t_community")
    assert len(chain) >= 1
    assert chain[0]["entity_id"] == "t_community"
    assert not chain[0]["tainted"]
    print("  ✓ provenance_chain")

def test_promote_requires_verification():
    """Can't promote without proper verification node."""
    g = make_graph()
    g.add_node("claim1", NT.EXTERNAL_CLAIM, {"provenance_type": "SOURCE_UNVERIFIED"})
    g.add_node("fake_verifier", NT.TERRAIN_NODE, {"provenance_type": "INFERENCE"})

    result = promote_external_claim(g, "claim1", "fake_verifier", "t1", "text")
    assert result is None
    print("  ✓ promote_requires_verification")

def test_promote_requires_external_claim():
    """Can't promote a non-ExternalClaim."""
    g = make_graph()
    add_terrain_node(g, "t1", "Already terrain", "bedrock", provenance_type="BEDROCK")
    g.add_node("wv1", NT.WITNESS_VERIFICATION, {"provenance_type": "WITNESS"})

    result = promote_external_claim(g, "t1", "wv1", "t2", "text")
    assert result is None
    print("  ✓ promote_requires_external_claim")

# ============================================================
# Conflict Tests
# ============================================================

def test_create_conflict():
    g = make_graph()
    contenders = [
        Contender("hash_a", "alice", 100, "root_1"),
        Contender("hash_b", "bob", 101, "root_1"),
    ]
    cs = create_conflict_set(g, "comm1", "task:123/assignment", contenders, ["assignee"])
    assert cs.get("status") == "OPEN"
    assert cs.get("contender_count") == 2
    assert cs.get("object_key") == "task:123/assignment"

    open_cs = get_open_conflicts(g, "comm1")
    assert len(open_cs) == 1
    print("  ✓ create_conflict")

def test_resolve_conflict():
    g = make_graph()
    contenders = [
        Contender("hash_a", "alice", 100, "root_1"),
        Contender("hash_b", "bob", 101, "root_1"),
    ]
    cs = create_conflict_set(g, "comm1", "task:456", contenders, ["assignee"])

    # Resolve in favor of alice
    ok = resolve_conflict(g, cs.id, "hash_a", "steward_001", "Alice was first")
    assert ok
    assert cs.get("status") == "RESOLVED"
    assert cs.get("chosen_mutation") == "hash_a"

    # Can't resolve again
    ok2 = resolve_conflict(g, cs.id, "hash_b", "steward_001")
    assert not ok2

    # No more open conflicts
    assert len(get_open_conflicts(g, "comm1")) == 0
    print("  ✓ resolve_conflict")

def test_challenge_creates_quarantine():
    g = make_graph()
    flag = create_challenge(g, "mutation_hash_xyz", "challenger_001", "CONDITION_A", "proof")
    assert flag.get("target_mutation") == "mutation_hash_xyz"
    assert flag.get("status") == "ASSERTED_VALID"
    assert flag.get("attached_by") == "challenger_001"

    flags = get_quarantine_flags(g, "mutation_hash_xyz")
    assert len(flags) == 1
    print("  ✓ challenge_creates_quarantine")

def test_multiple_conflicts():
    g = make_graph()
    for i in range(5):
        contenders = [
            Contender(f"hash_{i}_a", "alice", 100 + i, "root"),
            Contender(f"hash_{i}_b", "bob", 100 + i, "root"),
        ]
        create_conflict_set(g, "comm1", f"obj_{i}", contenders, ["field"])

    open_cs = get_open_conflicts(g, "comm1")
    assert len(open_cs) == 5

    # Resolve 3 of them
    for cs in open_cs[:3]:
        resolve_conflict(g, cs.id, cs.get("contenders")[0]["mutation_hash"], "steward")

    assert len(get_open_conflicts(g, "comm1")) == 2
    print("  ✓ multiple_conflicts (5 created, 3 resolved, 2 remain)")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing provenance and conflict resolution...\n")

    print("Taint Detection:")
    test_bedrock_not_tainted()
    test_witness_not_tainted()
    test_cross_verified_not_tainted()
    test_inference_is_tainted()
    test_source_unverified_is_tainted()

    print("\nEarned Terrain Protocol:")
    test_promoted_claim_not_tainted()
    test_cross_verified_promotion()
    test_provenance_chain()
    test_promote_requires_verification()
    test_promote_requires_external_claim()

    print("\nConflict Resolution:")
    test_create_conflict()
    test_resolve_conflict()
    test_challenge_creates_quarantine()
    test_multiple_conflicts()

    print("\n" + "=" * 50)
    print("ALL PROVENANCE TESTS PASSED ✓")
    print("=" * 50)

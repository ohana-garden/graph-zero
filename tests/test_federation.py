"""Tests for federation: snapshots, clone protocol, replication, visiting agents."""

import sys, time
sys.path.insert(0, '/home/claude')

from graph_zero.graph.backend import PropertyGraph
from graph_zero.graph.schema import (
    NT, ET, bootstrap_community, create_agent,
    add_terrain_node, connect_terrain
)
from graph_zero.federation.federation import (
    SnapshotFormat, Snapshot, create_snapshot, restore_snapshot,
    CloneEnvelope, create_clone_envelope, admit_visiting_agent,
    TRUST_DAMPENING_FACTOR,
    ReplicationMode, LogReplicator,
)


def make_community(cid="comm1", name="Lower Puna"):
    g = PropertyGraph()
    bootstrap_community(g, cid, name)
    return g


def make_populated_community():
    g = make_community()
    create_agent(g, "comm1", "kawika", "Kawika", "human")
    create_agent(g, "comm1", "leilani", "Leilani", "human")
    g.update_node("kawika", trust_ceiling=0.8, public_key="kawika_pub_key")
    g.update_node("leilani", trust_ceiling=0.6)
    # Add skills
    g.add_node("skill_herbs", NT.SKILL, {"name": "Medicinal Herbs"})
    g.add_edge("kawika", "skill_herbs", ET.HAS_SKILL)
    # Add terrain
    add_terrain_node(g, "t1", "Taro cultivation basics", "community",
                     provenance_type="WITNESS")
    add_terrain_node(g, "t2", "Breadfruit storage methods", "earned",
                     provenance_type="CROSS_VERIFIED")
    connect_terrain(g, "t1", "t2", 0.7, "WITNESS")
    return g


# ============================================================
# Snapshot Tests
# ============================================================

def test_snapshot_full():
    g = make_populated_community()
    snap = create_snapshot(g, "comm1", "admin_key")
    assert snap.manifest.node_count > 0
    assert snap.manifest.edge_count > 0
    assert snap.manifest.format == SnapshotFormat.FULL
    assert snap.verify()
    print(f"  ✓ snapshot_full ({snap.manifest.node_count} nodes, {snap.manifest.edge_count} edges)")

def test_snapshot_terrain_only():
    g = make_populated_community()
    snap = create_snapshot(g, "comm1", "admin_key", SnapshotFormat.TERRAIN_ONLY)
    node_types = {n["node_type"] for n in snap.nodes}
    assert NT.TERRAIN_NODE in node_types or NT.VIRTUE_ANCHOR in node_types
    assert NT.VESSEL_ANCHOR not in node_types
    assert snap.verify()
    print(f"  ✓ snapshot_terrain_only ({snap.manifest.node_count} nodes)")

def test_snapshot_agents_only():
    g = make_populated_community()
    snap = create_snapshot(g, "comm1", "admin_key", SnapshotFormat.AGENTS_ONLY)
    node_types = {n["node_type"] for n in snap.nodes}
    assert NT.VESSEL_ANCHOR in node_types
    assert NT.TERRAIN_NODE not in node_types
    assert snap.verify()
    print(f"  ✓ snapshot_agents_only ({snap.manifest.node_count} nodes)")

def test_snapshot_selective():
    g = make_populated_community()
    snap = create_snapshot(g, "comm1", "admin_key", SnapshotFormat.SELECTIVE,
                           included_types=[NT.CONFLICT_SET])
    # No conflicts exist, should be empty
    assert snap.manifest.node_count == 0
    assert snap.verify()
    print("  ✓ snapshot_selective")

def test_snapshot_restore():
    """Snapshot → restore → identical graph."""
    g = make_populated_community()
    snap = create_snapshot(g, "comm1", "admin_key")
    restored = restore_snapshot(snap)
    assert restored.node_count == g.node_count
    assert restored.edge_count == g.edge_count
    # State hashes match
    assert restored.compute_state_hash() == g.compute_state_hash()
    print("  ✓ snapshot_restore (state hash matches)")

def test_snapshot_tamper_detection():
    """Tampering with snapshot nodes should fail verification."""
    g = make_populated_community()
    snap = create_snapshot(g, "comm1", "admin_key")
    assert snap.verify()
    # Tamper
    if snap.nodes:
        snap.nodes[0]["id"] = "TAMPERED"
    assert not snap.verify()
    print("  ✓ snapshot_tamper_detection")

def test_snapshot_deterministic():
    """Same graph → same snapshot hash."""
    g = make_populated_community()
    s1 = create_snapshot(g, "comm1", "admin_key")
    s2 = create_snapshot(g, "comm1", "admin_key")
    # Manifests have different timestamps but same content hash
    assert s1.compute_hash() == s2.compute_hash()
    print("  ✓ snapshot_deterministic")


# ============================================================
# Clone Protocol Tests
# ============================================================

def test_create_clone():
    g = make_populated_community()
    env = create_clone_envelope(g, "kawika", "comm1", validity_days=90)
    assert env is not None
    assert env.home_vessel_id == "kawika"
    assert env.trust_ceiling_home == 0.8
    assert abs(env.trust_ceiling_dampened - 0.8 * TRUST_DAMPENING_FACTOR) < 0.001
    assert "Medicinal Herbs" in env.skills
    assert not env.is_expired()
    print(f"  ✓ create_clone (trust {env.trust_ceiling_home} → {env.trust_ceiling_dampened})")

def test_clone_nonexistent():
    g = make_populated_community()
    env = create_clone_envelope(g, "nobody", "comm1")
    assert env is None
    print("  ✓ clone_nonexistent")

def test_clone_expiry():
    g = make_populated_community()
    env = create_clone_envelope(g, "kawika", "comm1", validity_days=90)
    assert not env.is_expired()
    # Simulate expiry
    far_future = int(time.time() * 1000) + (200 * 86400000)
    assert env.is_expired(now=far_future)
    print("  ✓ clone_expiry")

def test_clone_hash_deterministic():
    g = make_populated_community()
    env = create_clone_envelope(g, "kawika", "comm1")
    h1 = env.compute_hash()
    h2 = env.compute_hash()
    assert h1 == h2
    print("  ✓ clone_hash_deterministic")


# ============================================================
# Visiting Agent Tests
# ============================================================

def test_admit_visitor():
    home_g = make_populated_community()
    env = create_clone_envelope(home_g, "kawika", "comm1")

    # Foreign community
    foreign_g = make_community("comm2", "Hilo Community")

    vid = admit_visiting_agent(foreign_g, "comm2", env)
    assert vid is not None

    visitor = foreign_g.get_node(vid)
    assert visitor is not None
    assert visitor.get("type") == "visiting"
    assert visitor.get("trust_ceiling") == env.trust_ceiling_dampened
    assert visitor.get("trust_ceiling") < env.trust_ceiling_home

    # Has position
    pos_edges = foreign_g.get_incoming(vid, ET.PART_OF_VESSEL)
    assert len(pos_edges) >= 1

    # Has clone reference
    clone_edges = foreign_g.get_outgoing(vid, ET.CLONE_OF)
    assert len(clone_edges) == 1
    print(f"  ✓ admit_visitor (dampened trust={visitor.get('trust_ceiling'):.2f})")

def test_admit_expired_clone():
    home_g = make_populated_community()
    env = create_clone_envelope(home_g, "kawika", "comm1", validity_days=1)
    # Force expiry
    env.expires_at = int(time.time() * 1000) - 1000

    foreign_g = make_community("comm2", "Hilo")
    vid = admit_visiting_agent(foreign_g, "comm2", env)
    assert vid is None
    print("  ✓ admit_expired_clone (rejected)")

def test_visiting_trust_dampened():
    """Visiting trust is always less than home trust."""
    home_g = make_populated_community()
    env = create_clone_envelope(home_g, "kawika", "comm1")
    foreign_g = make_community("comm2", "Hilo")
    vid = admit_visiting_agent(foreign_g, "comm2", env)
    visitor = foreign_g.get_node(vid)
    assert visitor.get("trust_ceiling") == env.trust_ceiling_home * TRUST_DAMPENING_FACTOR
    assert visitor.get("trust_ceiling") < env.trust_ceiling_home
    print("  ✓ visiting_trust_dampened")


# ============================================================
# Log Replication Tests
# ============================================================

def test_establish_link():
    repl = LogReplicator()
    link = repl.establish_link("comm1", "comm2", ReplicationMode.TERRAIN_SYNC)
    assert link.active
    assert link.sync_count == 0
    links = repl.get_active_links("comm1")
    assert len(links) == 1
    print("  ✓ establish_link")

def test_dissolve_link():
    repl = LogReplicator()
    link = repl.establish_link("comm1", "comm2", ReplicationMode.TERRAIN_SYNC)
    repl.dissolve_link(link.link_id)
    assert len(repl.get_active_links("comm1")) == 0
    print("  ✓ dissolve_link")

def test_terrain_replication():
    """Replicate terrain from comm1 to comm2."""
    g1 = make_populated_community()
    g2 = make_community("comm2", "Hilo")

    repl = LogReplicator()
    link = repl.establish_link("comm1", "comm2", ReplicationMode.TERRAIN_SYNC)

    batch = repl.prepare_batch(g1, link)
    assert len(batch.nodes) > 0
    # Should only contain terrain + virtue nodes
    node_types = {n["node_type"] for n in batch.nodes}
    assert NT.VESSEL_ANCHOR not in node_types  # no agents
    assert batch.batch_hash != ""

    result = repl.apply_batch(batch, g2)
    assert result["applied_nodes"] > 0
    assert "error" not in result

    # Federated nodes have prefix
    fed_nodes = [n for n in g2.get_nodes_by_type(NT.TERRAIN_NODE)
                 if n.id.startswith("fed:comm1:")]
    assert len(fed_nodes) >= 2  # t1 and t2
    assert fed_nodes[0].get("_federated_from") == "comm1"
    print(f"  ✓ terrain_replication ({result['applied_nodes']} nodes, {result['applied_edges']} edges)")

def test_replication_hash_check():
    """Tampered batch should be rejected."""
    g1 = make_populated_community()
    g2 = make_community("comm2", "Hilo")

    repl = LogReplicator()
    link = repl.establish_link("comm1", "comm2", ReplicationMode.TERRAIN_SYNC)
    batch = repl.prepare_batch(g1, link)

    # Tamper with batch
    if batch.nodes:
        batch.nodes[0]["id"] = "EVIL_NODE"

    result = repl.apply_batch(batch, g2)
    assert "error" in result
    assert result["applied_nodes"] == 0
    print("  ✓ replication_hash_check (tamper rejected)")

def test_full_mirror():
    """Full mirror replicates everything."""
    g1 = make_populated_community()
    g2 = PropertyGraph()

    repl = LogReplicator()
    link = repl.establish_link("comm1", "comm2", ReplicationMode.FULL_MIRROR)
    batch = repl.prepare_batch(g1, link)

    assert len(batch.nodes) == g1.node_count
    result = repl.apply_batch(batch, g2)
    assert result["applied_nodes"] == g1.node_count
    print(f"  ✓ full_mirror ({result['applied_nodes']} nodes)")

def test_sync_count_tracking():
    """Sync count increments after each batch."""
    g1 = make_populated_community()
    g2 = make_community("comm2", "Hilo")

    repl = LogReplicator()
    link = repl.establish_link("comm1", "comm2", ReplicationMode.TERRAIN_SYNC)

    for i in range(3):
        batch = repl.prepare_batch(g1, link)
        repl.apply_batch(batch, g2)

    updated_link = repl.get_active_links("comm1")[0]
    assert updated_link.sync_count == 3
    assert updated_link.last_sync_at > 0
    print("  ✓ sync_count_tracking")


# ============================================================
# End-to-End Federation Scenario
# ============================================================

def test_e2e_federation():
    """Complete scenario: two communities, terrain sharing, agent visiting."""
    # Community 1: Lower Puna (taro knowledge)
    g1 = make_community("puna", "Lower Puna")
    create_agent(g1, "puna", "kawika", "Kawika", "human")
    g1.update_node("kawika", trust_ceiling=0.8, public_key="kpub")
    g1.add_node("skill_taro", NT.SKILL, {"name": "Taro Cultivation"})
    g1.add_edge("kawika", "skill_taro", ET.HAS_SKILL)
    add_terrain_node(g1, "t_taro", "Taro grows in pH 5.5-6.5", "community",
                     provenance_type="WITNESS")

    # Community 2: Hilo (breadfruit knowledge)
    g2 = make_community("hilo", "Hilo")
    create_agent(g2, "hilo", "leilani", "Leilani", "human")
    g2.update_node("leilani", trust_ceiling=0.7)
    add_terrain_node(g2, "t_bread", "Breadfruit stores 3 days in shade", "community",
                     provenance_type="WITNESS")

    # 1. Establish terrain sharing
    repl = LogReplicator()
    link_12 = repl.establish_link("puna", "hilo", ReplicationMode.TERRAIN_SYNC)
    link_21 = repl.establish_link("hilo", "puna", ReplicationMode.TERRAIN_SYNC)

    # 2. Sync terrain both ways
    batch_to_hilo = repl.prepare_batch(g1, link_12)
    result1 = repl.apply_batch(batch_to_hilo, g2)
    assert result1["applied_nodes"] > 0

    batch_to_puna = repl.prepare_batch(g2, link_21)
    result2 = repl.apply_batch(batch_to_puna, g1)
    assert result2["applied_nodes"] > 0

    # Hilo now has Puna's taro knowledge
    fed_taro = [n for n in g2.get_nodes_by_type(NT.TERRAIN_NODE)
                if "taro" in n.get("source_text", "").lower()
                and n.get("_federated_from") == "puna"]
    assert len(fed_taro) >= 1

    # Puna now has Hilo's breadfruit knowledge
    fed_bread = [n for n in g1.get_nodes_by_type(NT.TERRAIN_NODE)
                 if "breadfruit" in n.get("source_text", "").lower()
                 and n.get("_federated_from") == "hilo"]
    assert len(fed_bread) >= 1

    # 3. Kawika visits Hilo
    env = create_clone_envelope(g1, "kawika", "puna")
    vid = admit_visiting_agent(g2, "hilo", env)
    visitor = g2.get_node(vid)
    assert visitor.get("trust_ceiling") < 0.8  # dampened
    assert visitor.get("type") == "visiting"

    # 4. Create snapshot of Hilo (now has federated content + visitor)
    snap = create_snapshot(g2, "hilo", "admin")
    assert snap.verify()
    assert snap.manifest.node_count > 0

    print(f"  ✓ e2e_federation:")
    print(f"    terrain shared: puna→hilo {result1['applied_nodes']}n, hilo→puna {result2['applied_nodes']}n")
    print(f"    kawika visiting hilo: trust {visitor.get('trust_ceiling'):.2f}")
    print(f"    hilo snapshot: {snap.manifest.node_count} nodes, verified")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing federation...\n")

    print("Snapshots:")
    test_snapshot_full()
    test_snapshot_terrain_only()
    test_snapshot_agents_only()
    test_snapshot_selective()
    test_snapshot_restore()
    test_snapshot_tamper_detection()
    test_snapshot_deterministic()

    print("\nClone Protocol:")
    test_create_clone()
    test_clone_nonexistent()
    test_clone_expiry()
    test_clone_hash_deterministic()

    print("\nVisiting Agents:")
    test_admit_visitor()
    test_admit_expired_clone()
    test_visiting_trust_dampened()

    print("\nLog Replication:")
    test_establish_link()
    test_dissolve_link()
    test_terrain_replication()
    test_replication_hash_check()
    test_full_mirror()
    test_sync_count_tracking()

    print("\nEnd-to-End:")
    test_e2e_federation()

    print("\n" + "=" * 50)
    print("ALL FEDERATION TESTS PASSED ✓")
    print("=" * 50)

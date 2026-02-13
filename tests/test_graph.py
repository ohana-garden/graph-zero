"""Tests for Graph Zero graph layer: backend, schema, traversal, trust, vitals."""

import sys, time
sys.path.insert(0, '/home/claude')

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import (
    NT, ET, VERIFIED_PROVENANCE,
    assemble_constellation, traverse_terrain, find_entry_points,
    compute_trust_flow, compute_vital_signs,
    bootstrap_community, create_agent, add_terrain_node,
    connect_terrain, record_interaction, _cosine_similarity, _gini,
    Constellation, TraversalResult, TrustFlowResult, VitalSigns,
)


# ============================================================
# Backend Tests
# ============================================================

def test_add_get_node():
    g = PropertyGraph()
    n = g.add_node("n1", "Person", {"name": "Alice"})
    assert g.has_node("n1")
    assert g.get_node("n1").get("name") == "Alice"
    assert g.node_count == 1
    print("  ✓ add_get_node")

def test_add_edge():
    g = PropertyGraph()
    g.add_node("a", "Person")
    g.add_node("b", "Person")
    e = g.add_edge("a", "b", "KNOWS", {"since": 2020})
    assert e is not None
    assert g.edge_count == 1
    assert len(g.get_outgoing("a")) == 1
    assert len(g.get_incoming("b")) == 1
    assert g.add_edge("a", "nonexistent", "KNOWS") is None
    print("  ✓ add_edge")

def test_find_nodes():
    g = PropertyGraph()
    g.add_node("a", "Person", {"role": "admin"})
    g.add_node("b", "Person", {"role": "member"})
    g.add_node("c", "Person", {"role": "admin"})
    admins = g.find_nodes("Person", role="admin")
    assert len(admins) == 2
    print("  ✓ find_nodes")

def test_get_neighbors():
    g = PropertyGraph()
    g.add_node("a", "Person")
    g.add_node("b", "Person")
    g.add_node("c", "Person")
    g.add_edge("a", "b", "KNOWS")
    g.add_edge("a", "c", "KNOWS")
    neighbors = g.get_neighbors("a", "KNOWS", direction="out")
    assert len(neighbors) == 2
    print("  ✓ get_neighbors")

def test_remove_node():
    g = PropertyGraph()
    g.add_node("a", "Person")
    g.add_node("b", "Person")
    g.add_edge("a", "b", "KNOWS")
    assert g.remove_node("a")
    assert not g.has_node("a")
    assert g.edge_count == 0
    print("  ✓ remove_node (cascades edges)")

def test_traverse_bfs():
    g = PropertyGraph()
    for i in range(5):
        g.add_node(f"n{i}", "Node")
    g.add_edge("n0", "n1", "LINK")
    g.add_edge("n1", "n2", "LINK")
    g.add_edge("n2", "n3", "LINK")
    g.add_edge("n3", "n4", "LINK")
    results = g.traverse("n0", "LINK", max_depth=3)
    ids = [r[0].id for r in results]
    assert "n1" in ids
    assert "n2" in ids
    assert "n3" in ids
    assert "n4" not in ids
    print("  ✓ traverse_bfs (depth bounded)")

def test_find_paths():
    g = PropertyGraph()
    g.add_node("a", "N")
    g.add_node("b", "N")
    g.add_node("c", "N")
    g.add_node("d", "N")
    g.add_edge("a", "b", "E")
    g.add_edge("b", "d", "E")
    g.add_edge("a", "c", "E")
    g.add_edge("c", "d", "E")
    paths = g.find_paths("a", "d", "E")
    assert len(paths) == 2
    print("  ✓ find_paths (2 paths)")

def test_state_hash_deterministic():
    g1 = PropertyGraph()
    g2 = PropertyGraph()
    for g in [g1, g2]:
        g.add_node("a", "Person", {"name": "Alice"})
        g.add_node("b", "Person", {"name": "Bob"})
        g.add_edge("a", "b", "KNOWS", edge_id="e1")
    assert g1.compute_state_hash() == g2.compute_state_hash()
    print("  ✓ state_hash_deterministic")


# ============================================================
# Bootstrap Tests
# ============================================================

def test_bootstrap_community():
    g = PropertyGraph()
    comm = bootstrap_community(g, "comm1", "Lower Puna")
    assert g.has_node("comm1")
    virtues = g.get_nodes_by_type(NT.VIRTUE_ANCHOR)
    assert len(virtues) == 9
    months = g.get_nodes_by_type(NT.BADI_MONTH)
    assert len(months) == 19
    policy = g.get_nodes_by_type(NT.POLICY_CONFIG)
    assert len(policy) == 1
    coupling_edges = []
    for v in virtues:
        coupling_edges.extend(g.get_outgoing(v.id, ET.COUPLES_WITH))
    assert len(coupling_edges) == 4
    print("  ✓ bootstrap_community (9 virtues, 19 months, 4 couplings)")

def test_create_agent():
    g = PropertyGraph()
    bootstrap_community(g, "comm1", "Lower Puna")
    c = create_agent(g, "comm1", "vessel_aunty", "Aunty Mele",
                     agent_type="human", initial_kala=200.0)
    assert c is not None
    assert c.name == "Aunty Mele"
    assert c.agent_type == "human"
    assert c.kala_balance == 200.0
    assert c.moral_position == [0.5] * 9
    assert c.community is not None
    assert c.community.id == "comm1"
    print("  ✓ create_agent (full constellation)")

def test_assemble_constellation():
    g = PropertyGraph()
    bootstrap_community(g, "comm1", "Test Community")
    create_agent(g, "comm1", "v1", "Agent One")
    c = assemble_constellation(g, "v1")
    assert c is not None
    assert c.position is not None
    assert c.momentum is not None
    assert c.kala is not None
    print("  ✓ assemble_constellation")


# ============================================================
# Terrain Tests
# ============================================================

def build_terrain(g):
    add_terrain_node(g, "t_water", "Water is essential for taro cultivation",
                     "bedrock", embedding=[1.0, 0.0, 0.5])
    add_terrain_node(g, "t_soil", "Volcanic soil provides key nutrients",
                     "bedrock", embedding=[0.8, 0.2, 0.4])
    add_terrain_node(g, "t_rotation", "Crop rotation improves yield by 30%",
                     "community", embedding=[0.9, 0.1, 0.6],
                     provenance_type="WITNESS")
    add_terrain_node(g, "t_compost", "Composting banana leaves works well here",
                     "earned", embedding=[0.7, 0.3, 0.5],
                     provenance_type="EMPIRICAL")
    add_terrain_node(g, "t_rumor", "Magic crystals help plants grow",
                     "earned", embedding=[0.1, 0.9, 0.1],
                     provenance_type="INFERENCE")
    connect_terrain(g, "t_water", "t_soil", provenance_type="BEDROCK")
    connect_terrain(g, "t_soil", "t_rotation", provenance_type="WITNESS")
    connect_terrain(g, "t_rotation", "t_compost", provenance_type="EMPIRICAL")
    connect_terrain(g, "t_compost", "t_rumor", provenance_type="INFERENCE")

def test_terrain_traversal():
    g = PropertyGraph()
    build_terrain(g)
    results = traverse_terrain(g, ["t_water"], max_depth=5)
    ids = [r.node.id for r in results]
    assert "t_soil" in ids
    assert "t_rotation" in ids
    assert "t_compost" in ids
    assert "t_rumor" not in ids
    print("  ✓ terrain_traversal (filters tainted provenance)")

def test_find_entry_points():
    g = PropertyGraph()
    build_terrain(g)
    entry_ids = find_entry_points(g, [1.0, 0.0, 0.5], threshold=0.9)
    assert "t_water" in entry_ids
    print("  ✓ find_entry_points (cosine similarity)")

def test_terrain_layers():
    g = PropertyGraph()
    build_terrain(g)
    results = traverse_terrain(g, ["t_water"], max_depth=5)
    layers = {r.node.id: r.layer for r in results}
    assert layers.get("t_soil") == "bedrock"
    assert layers.get("t_rotation") == "community"
    assert layers.get("t_compost") == "earned"
    print("  ✓ terrain_layers")


# ============================================================
# Trust Flow Tests
# ============================================================

def test_trust_flow_basic():
    g = PropertyGraph()
    bootstrap_community(g, "c1", "Trust Test")
    create_agent(g, "c1", "human1", "Alice", agent_type="human")
    create_agent(g, "c1", "ai1", "Bot", agent_type="ai")
    record_interaction(g, "human1", "ai1", "conversation", "helped with garden")
    record_interaction(g, "human1", "ai1", "verification", "confirmed taro info")
    result = compute_trust_flow(g, "ai1")
    assert result.trust_ceiling > 0
    assert result.path_count > 0
    print(f"  ✓ trust_flow_basic (ceiling={result.trust_ceiling:.3f})")

def test_trust_flow_no_anchor():
    g = PropertyGraph()
    bootstrap_community(g, "c1", "Trust Test")
    create_agent(g, "c1", "orphan", "Orphan Bot", agent_type="ai")
    result = compute_trust_flow(g, "orphan")
    assert result.trust_ceiling == 0.0
    print("  ✓ trust_flow_no_anchor (zero trust)")

def test_trust_flow_chain():
    g = PropertyGraph()
    bootstrap_community(g, "c1", "Trust Test")
    create_agent(g, "c1", "h1", "Human", agent_type="human")
    create_agent(g, "c1", "a1", "Agent1", agent_type="ai")
    create_agent(g, "c1", "a2", "Agent2", agent_type="ai")
    record_interaction(g, "h1", "a1", "conversation")
    record_interaction(g, "a1", "a2", "collaboration")
    result_a1 = compute_trust_flow(g, "a1")
    result_a2 = compute_trust_flow(g, "a2")
    assert result_a1.trust_ceiling >= result_a2.trust_ceiling
    print(f"  ✓ trust_flow_chain (a1={result_a1.trust_ceiling:.3f} >= a2={result_a2.trust_ceiling:.3f})")


# ============================================================
# Vital Signs Tests
# ============================================================

def test_vital_signs():
    g = PropertyGraph()
    bootstrap_community(g, "c1", "Vital Test")
    create_agent(g, "c1", "h1", "Alice", agent_type="human", initial_kala=500.0)
    create_agent(g, "c1", "h2", "Bob", agent_type="human", initial_kala=50.0)
    build_terrain(g)
    g.add_node("conflict1", NT.CONFLICT_SET, {"status": "OPEN"})
    signs = compute_vital_signs(g, "c1")
    assert signs.active_agents == 2
    assert signs.open_conflicts == 1
    assert signs.kala_concentration > 0
    assert len(signs.moral_variance) == 9
    print(f"  ✓ vital_signs (agents={signs.active_agents}, conflicts={signs.open_conflicts}, gini={signs.kala_concentration:.3f})")

def test_gini_coefficient():
    assert _gini([100, 100, 100]) < 0.01
    assert _gini([0, 0, 300]) > 0.5
    assert _gini([]) == 0.0
    print("  ✓ gini_coefficient")

def test_cosine_similarity():
    assert abs(_cosine_similarity([1, 0], [1, 0]) - 1.0) < 1e-6
    assert abs(_cosine_similarity([1, 0], [0, 1]) - 0.0) < 1e-6
    assert abs(_cosine_similarity([1, 0], [-1, 0]) - (-1.0)) < 1e-6
    assert _cosine_similarity([], []) == 0.0
    print("  ✓ cosine_similarity")


# ============================================================
# Integration: Grandmother's Question
# ============================================================

def test_grandmothers_question():
    g = PropertyGraph()
    bootstrap_community(g, "lower_puna", "Lower Puna Community")

    grandma = create_agent(g, "lower_puna", "grandma", "Tutu", agent_type="human", initial_kala=100)
    aunty = create_agent(g, "lower_puna", "aunty", "Aunty Mele", agent_type="human", initial_kala=50)
    uncle = create_agent(g, "lower_puna", "uncle", "Uncle Kai", agent_type="human", initial_kala=75)
    garden_bot = create_agent(g, "lower_puna", "garden_bot", "Garden Agent", agent_type="ai")
    med_bot = create_agent(g, "lower_puna", "med_bot", "Med Agent", agent_type="ai")

    record_interaction(g, "grandma", "garden_bot", "daily_checkin")
    record_interaction(g, "grandma", "garden_bot", "surplus_coordination")
    record_interaction(g, "uncle", "garden_bot", "surplus_declaration")
    record_interaction(g, "aunty", "med_bot", "prescription_request")
    record_interaction(g, "grandma", "med_bot", "health_inquiry")

    add_terrain_node(g, "taro_growing", "Taro grows in lo'i (wetland patches)",
                     "bedrock", embedding=[0.9, 0.1, 0.5])
    add_terrain_node(g, "insulin_info", "Insulin requires refrigeration below 46°F",
                     "bedrock", embedding=[0.1, 0.9, 0.5])
    add_terrain_node(g, "uncle_surplus", "Uncle Kai has surplus taro this week",
                     "earned", embedding=[0.8, 0.2, 0.4], provenance_type="WITNESS")
    add_terrain_node(g, "med_quest", "Med-QUEST covers insulin for qualifying residents",
                     "community", embedding=[0.2, 0.8, 0.6], provenance_type="WITNESS")

    connect_terrain(g, "taro_growing", "uncle_surplus", provenance_type="WITNESS")
    connect_terrain(g, "insulin_info", "med_quest", provenance_type="WITNESS")

    taro_results = traverse_terrain(g, ["taro_growing"], max_depth=3)
    assert any(r.node.id == "uncle_surplus" for r in taro_results)

    med_results = traverse_terrain(g, ["insulin_info"], max_depth=3)
    assert any(r.node.id == "med_quest" for r in med_results)

    garden_trust = compute_trust_flow(g, "garden_bot")
    assert garden_trust.trust_ceiling > 0

    med_trust = compute_trust_flow(g, "med_bot")
    assert med_trust.trust_ceiling > 0

    signs = compute_vital_signs(g, "lower_puna")
    assert signs.active_agents == 5

    print("  ✓ grandmothers_question (end-to-end: terrain → trust → vitals)")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing Graph layer...\n")

    print("Backend:")
    test_add_get_node()
    test_add_edge()
    test_find_nodes()
    test_get_neighbors()
    test_remove_node()
    test_traverse_bfs()
    test_find_paths()
    test_state_hash_deterministic()

    print("\nBootstrap:")
    test_bootstrap_community()
    test_create_agent()
    test_assemble_constellation()

    print("\nTerrain:")
    test_terrain_traversal()
    test_find_entry_points()
    test_terrain_layers()

    print("\nTrust Flow:")
    test_trust_flow_basic()
    test_trust_flow_no_anchor()
    test_trust_flow_chain()

    print("\nVital Signs:")
    test_vital_signs()
    test_gini_coefficient()
    test_cosine_similarity()

    print("\nIntegration:")
    test_grandmothers_question()

    print("\n" + "=" * 50)
    print("ALL GRAPH TESTS PASSED ✓")
    print("=" * 50)

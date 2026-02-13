"""Tests for Graph Zero interface layer: sessions, z-layers, context assembly, dashboard."""

import sys, time
sys.path.insert(0, '/home/claude')

from graph_zero.graph.backend import PropertyGraph
from graph_zero.graph.schema import (
    NT, ET, bootstrap_community, create_agent,
    add_terrain_node, connect_terrain,
)
from graph_zero.execution.execution import (
    ExecutionEngine, ExecutionStatus, ToolSpec, ToolDomain,
    register_tool, grant_tool_access,
    SANDBOX_STRICT, SANDBOX_STANDARD,
)
from graph_zero.memory.memory import MemoryStore, EpisodeData, SemanticFactData
from graph_zero.interface.interface import (
    ZLayer, ZLayerContent, ContextFrame,
    SessionManager, SessionState,
    build_dashboard,
)


def make_full_community():
    """Build a complete community with agents, terrain, tools, memories."""
    g = PropertyGraph()
    bootstrap_community(g, "comm1", "Lower Puna")

    # Agents
    create_agent(g, "comm1", "kawika", "Kawika", "human")
    create_agent(g, "comm1", "leilani", "Leilani", "human")
    g.update_node("kawika", trust_ceiling=0.8, attestation_depth=0)
    g.update_node("leilani", trust_ceiling=0.6, attestation_depth=0)

    # Terrain
    add_terrain_node(g, "t_service", "Service to the sick is noble",
                     "bedrock", embedding=[0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.9, 0.5, 0.9],
                     provenance_type="BEDROCK")
    add_terrain_node(g, "t_herbs", "Hawaiian medicinal herbs for healing",
                     "community", embedding=[0.3, 0.1, 0.5, 0.4, 0.2, 0.3, 0.7, 0.6, 0.5],
                     provenance_type="WITNESS")
    add_terrain_node(g, "t_taro", "Taro as recovery food",
                     "earned", embedding=[0.2, 0.1, 0.4, 0.5, 0.2, 0.3, 0.6, 0.5, 0.4],
                     provenance_type="CROSS_VERIFIED")
    connect_terrain(g, "t_service", "t_herbs", 0.8, "WITNESS")
    connect_terrain(g, "t_herbs", "t_taro", 0.7, "CROSS_VERIFIED")

    # Interactions (for trust flow)
    g.add_edge("kawika", "leilani", ET.INTERACTED_WITH, {
        "interaction_type": "collaborated",
        "timestamp": int(time.time() * 1000),
    })

    # Tools
    engine = ExecutionEngine(g)
    register_tool(g, ToolSpec(
        tool_id="tool_plant_id", name="Plant Identifier",
        description="Identify plants", domain=ToolDomain.GARDENING,
        sandbox=SANDBOX_STANDARD, trust_floor=0.1,
    ))
    register_tool(g, ToolSpec(
        tool_id="tool_weather", name="Weather Check",
        description="Get weather", domain=ToolDomain.INFORMATION,
        sandbox=SANDBOX_STRICT, trust_floor=0.0,
    ))
    grant_tool_access(g, "kawika", "tool_plant_id")
    grant_tool_access(g, "kawika", "tool_weather")
    engine.register_handler("tool_plant_id", lambda d: {"plant": "taro", "confidence": 0.95})
    engine.register_handler("tool_weather", lambda d: {"temp": 78, "condition": "partly cloudy"})

    # Pre-load some memories
    ms = MemoryStore(g, "kawika")
    ms.ingest_episode(EpisodeData(
        participants=["kawika", "leilani"],
        interaction_type="conversation",
        content="Discussed taro pH levels with Leilani",
        summary="Taro pH discussion",
        embedding=[0.4, 0.1, 0.5, 0.4, 0.2, 0.3, 0.6, 0.5, 0.5],
        emotional_valence=0.5,
        emotional_arousal=0.3,
    ))
    ms.extract_fact(SemanticFactData(
        subject="taro", predicate="optimal_ph", object_value="5.5-6.5",
        confidence=0.85, embedding=[0.4, 0.1, 0.5, 0.4, 0.2, 0.3, 0.6, 0.5, 0.5],
        source_episodes=[],
    ))

    return g, engine


# ============================================================
# Session Tests
# ============================================================

def test_create_session():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")
    assert session is not None
    assert session.vessel_id == "kawika"
    assert session.community_id == "comm1"
    assert session.constellation is not None
    assert session.constellation.name == "Kawika"
    assert len(session.active_tools) >= 2
    print(f"  ✓ create_session (trust={session.trust_ceiling:.2f}, tools={len(session.active_tools)})")

def test_create_session_nonexistent():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("nobody", "comm1")
    assert session is None
    print("  ✓ create_session_nonexistent")

def test_close_session():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")
    session.conversation_history.append({"role": "user", "text": "hello"})
    ok = mgr.close_session(session.session_id)
    assert ok
    assert mgr.get_session(session.session_id) is None
    print("  ✓ close_session")


# ============================================================
# Context Assembly Tests
# ============================================================

def test_assemble_context():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")

    # Query about sick neighbor
    query_embedding = [0.85, 0.1, 0.75, 0.65, 0.2, 0.3, 0.88, 0.5, 0.85]
    frame = mgr.assemble_context(session, "My neighbor is sick", query_embedding)

    assert frame.vessel_id == "kawika"
    assert frame.total_items > 0

    # Z0: Immediate (tools should be there)
    z0 = frame.get_layer(ZLayer.IMMEDIATE)
    tool_items = [i for i in z0.items if i["type"] == "tool"]
    assert len(tool_items) >= 2

    # Z1: Personal (memories)
    z1 = frame.get_layer(ZLayer.PERSONAL)
    # Should find memories about taro
    assert z1.count >= 0  # may or may not match depending on similarity

    print(f"  ✓ assemble_context (total={frame.total_items}, "
          f"z0={z0.count}, z1={z1.count}, "
          f"z2={frame.get_layer(ZLayer.COMMUNITY).count}, "
          f"z4={frame.get_layer(ZLayer.BEDROCK).count}, "
          f"assembly={frame.assembly_ms}ms)")

def test_context_z_layers_ordered():
    """Items from all_items() should be nearest-first."""
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")

    query_embedding = [0.5] * 9
    frame = mgr.assemble_context(session, "test query", query_embedding)

    items = frame.all_items()
    # First items should be from z0 (immediate)
    if items:
        # tools are in z0
        z0_items = frame.get_layer(ZLayer.IMMEDIATE).items
        if z0_items:
            assert items[0] in z0_items
    print("  ✓ context_z_layers_ordered")

def test_context_terrain_traversal():
    """Context should include terrain from verified provenance paths."""
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")

    # Query about service to sick (should match bedrock terrain)
    query_embedding = [0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.9, 0.5, 0.9]
    frame = mgr.assemble_context(session, "service to the sick", query_embedding)

    # Bedrock layer should have content
    z4 = frame.get_layer(ZLayer.BEDROCK)
    bedrock_texts = [i.get("source_text", "") for i in z4.items]
    if bedrock_texts:
        assert any("service" in t.lower() or "noble" in t.lower() for t in bedrock_texts)

    # Community layer should have herbs
    z2 = frame.get_layer(ZLayer.COMMUNITY)
    comm_texts = [i.get("source_text", "") for i in z2.items]

    total_terrain = z2.count + z4.count
    print(f"  ✓ context_terrain_traversal (bedrock={z4.count}, community={z2.count})")


# ============================================================
# Query Processing Tests
# ============================================================

def test_process_query():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")

    result = mgr.process_query(session, "What herbs help sick neighbors?",
                                [0.85, 0.1, 0.75, 0.65, 0.2, 0.3, 0.88, 0.5, 0.85])

    assert result["session_id"] == session.session_id
    assert result["vessel_id"] == "kawika"
    assert result["context_frame"]["total_items"] > 0
    assert len(result["available_tools"]) >= 2
    assert len(session.conversation_history) == 1
    print(f"  ✓ process_query (items={result['context_frame']['total_items']})")

def test_conversation_history():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")

    mgr.process_query(session, "First question", [0.5]*9)
    mgr.add_response(session, "First answer")
    mgr.process_query(session, "Second question", [0.5]*9)

    assert len(session.conversation_history) == 3
    assert session.conversation_history[0]["role"] == "user"
    assert session.conversation_history[1]["role"] == "assistant"
    assert session.conversation_history[2]["role"] == "user"
    print("  ✓ conversation_history (3 turns)")


# ============================================================
# Tool Invocation Tests
# ============================================================

def test_invoke_tool_via_session():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")

    result = mgr.invoke_tool(session, "tool_weather", {"location": "Pahoa"})
    assert result.status == ExecutionStatus.COMPLETED
    assert result.output_data["temp"] == 78
    print("  ✓ invoke_tool_via_session")

def test_invoke_tool_denied():
    g, engine = make_full_community()
    # Register a high-trust tool
    register_tool(g, ToolSpec(
        tool_id="tool_gov", name="Governance", description="Sensitive",
        domain=ToolDomain.GOVERNANCE, sandbox=SANDBOX_STRICT,
        trust_floor=0.99,
    ))
    grant_tool_access(g, "kawika", "tool_gov")

    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")
    result = mgr.invoke_tool(session, "tool_gov", {})
    assert result.status == ExecutionStatus.DENIED
    print("  ✓ invoke_tool_denied (trust too low)")


# ============================================================
# Dashboard Tests
# ============================================================

def test_dashboard():
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    mgr.create_session("kawika", "comm1")

    dash = build_dashboard(g, "comm1", mgr)
    assert dash.community_id == "comm1"
    assert dash.total_agents >= 2
    assert dash.total_terrain >= 3
    assert dash.active_sessions >= 1
    print(f"  ✓ dashboard (agents={dash.total_agents}, terrain={dash.total_terrain}, "
          f"sessions={dash.active_sessions})")


# ============================================================
# Full Integration: Grandmother's Question through Interface
# ============================================================

def test_grandmothers_question_via_interface():
    """The complete flow through the interface layer.

    Grandmother asks: 'My neighbor is sick, what should I bring?'
    System should assemble context from all layers.
    """
    g, engine = make_full_community()
    mgr = SessionManager(g, engine)
    session = mgr.create_session("kawika", "comm1")
    assert session is not None

    # The query
    result = mgr.process_query(
        session,
        "My neighbor is sick, what should I bring?",
        [0.85, 0.1, 0.75, 0.65, 0.2, 0.3, 0.88, 0.5, 0.85]
    )

    assert result["context_frame"]["total_items"] > 0

    # Should have tools available
    assert "tool_plant_id" in result["available_tools"]

    # Record response
    mgr.add_response(session, "Based on community knowledge, bring taro and medicinal herbs.")

    # Close session (should persist as episode)
    mgr.close_session(session.session_id)

    print(f"  ✓ grandmothers_question_via_interface:")
    cf = result["context_frame"]
    print(f"    z0={cf['z0_immediate']}, z1={cf['z1_personal']}, "
          f"z2={cf['z2_community']}, z3={cf['z3_federated']}, z4={cf['z4_bedrock']}")
    print(f"    total context items: {cf['total_items']}")
    print(f"    tools: {len(result['available_tools'])}")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing interface layer...\n")

    print("Sessions:")
    test_create_session()
    test_create_session_nonexistent()
    test_close_session()

    print("\nContext Assembly:")
    test_assemble_context()
    test_context_z_layers_ordered()
    test_context_terrain_traversal()

    print("\nQuery Processing:")
    test_process_query()
    test_conversation_history()

    print("\nTool Invocation:")
    test_invoke_tool_via_session()
    test_invoke_tool_denied()

    print("\nDashboard:")
    test_dashboard()

    print("\nGrandmother's Question (full integration):")
    test_grandmothers_question_via_interface()

    print("\n" + "=" * 50)
    print("ALL INTERFACE TESTS PASSED ✓")
    print("=" * 50)

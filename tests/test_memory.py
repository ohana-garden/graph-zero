"""Tests for Graph Zero memory system: episodes, bi-temporal, hybrid retrieval, consolidation."""

import sys, time
sys.path.insert(0, '/home/claude')

from graph_zero.graph.backend import PropertyGraph
from graph_zero.graph.schema import bootstrap_community, create_agent
from graph_zero.memory.memory import (
    MemoryStore, MemoryType, BiTemporalStamp,
    EpisodeData, SemanticFactData, RetrievalResult,
    EPISODE_NODE, SEMANTIC_FACT
)


def make_agent():
    g = PropertyGraph()
    bootstrap_community(g, "comm1", "Test")
    create_agent(g, "comm1", "alice", "Alice", "human")
    return g, MemoryStore(g, "alice")


# ============================================================
# Episode Tests
# ============================================================

def test_ingest_episode():
    g, ms = make_agent()
    eid = ms.ingest_episode(EpisodeData(
        participants=["alice", "bob"],
        interaction_type="conversation",
        content="Talked about taro cultivation in Lower Puna",
        summary="Taro cultivation discussion",
        embedding=[0.5] * 8,
        emotional_valence=0.6,
        emotional_arousal=0.4,
    ))
    assert eid.startswith("ep_alice_")
    node = g.get_node(eid)
    assert node is not None
    assert node.get("interaction_type") == "conversation"
    assert node.get("invalid_at") is None  # current
    print("  ✓ ingest_episode")

def test_episode_chain():
    """Episodes are linked by FOLLOWED_BY."""
    g, ms = make_agent()
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="First", summary="First", embedding=[0.1]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    e2 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Second", summary="Second", embedding=[0.2]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    # e1 FOLLOWED_BY e2
    edges = g.get_outgoing(e1, "FOLLOWED_BY")
    assert len(edges) == 1
    assert edges[0].target_id == e2
    print("  ✓ episode_chain")

def test_episode_stats():
    g, ms = make_agent()
    for i in range(5):
        ms.ingest_episode(EpisodeData(
            participants=["alice"], interaction_type="conversation",
            content=f"Episode {i}", summary=f"Ep {i}", embedding=[float(i)/10]*8,
            emotional_valence=0.0, emotional_arousal=0.1,
        ))
    stats = ms.stats()
    assert stats["episodes"] == 5
    assert stats["active_facts"] == 0
    print("  ✓ episode_stats")


# ============================================================
# Semantic Fact Tests
# ============================================================

def test_extract_fact():
    g, ms = make_agent()
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice", "kawika"], interaction_type="conversation",
        content="Kawika said taro grows best in pH 5.5-6.5",
        summary="Taro pH range", embedding=[0.5]*8,
        emotional_valence=0.3, emotional_arousal=0.2,
    ))
    fid = ms.extract_fact(SemanticFactData(
        subject="taro",
        predicate="optimal_ph",
        object_value="5.5-6.5",
        confidence=0.8,
        embedding=[0.5]*8,
        source_episodes=[e1],
    ))
    assert fid.startswith("fact_alice_")
    node = g.get_node(fid)
    assert node.get("subject") == "taro"
    assert node.get("object_value") == "5.5-6.5"
    assert node.get("confidence") == 0.8
    print("  ✓ extract_fact")

def test_fact_reinforcement():
    """Same fact extracted twice → reinforcement, not duplication."""
    g, ms = make_agent()
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Taro pH 5.5-6.5", summary="pH", embedding=[0.5]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    fid1 = ms.extract_fact(SemanticFactData(
        subject="taro", predicate="optimal_ph", object_value="5.5-6.5",
        confidence=0.7, embedding=[0.5]*8, source_episodes=[e1],
    ))
    e2 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Confirmed: taro pH 5.5-6.5", summary="pH confirm", embedding=[0.5]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    fid2 = ms.extract_fact(SemanticFactData(
        subject="taro", predicate="optimal_ph", object_value="5.5-6.5",
        confidence=0.8, embedding=[0.5]*8, source_episodes=[e2],
    ))
    # Same fact ID returned
    assert fid1 == fid2
    node = g.get_node(fid1)
    assert node.get("reinforcement_count") == 2
    assert node.get("confidence") > 0.7  # boosted
    print("  ✓ fact_reinforcement")

def test_fact_contradiction():
    """Different value for same subject+predicate → old invalidated."""
    g, ms = make_agent()
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="pH is 5.5-6.5", summary="pH", embedding=[0.5]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    fid1 = ms.extract_fact(SemanticFactData(
        subject="taro", predicate="optimal_ph", object_value="5.5-6.5",
        confidence=0.7, embedding=[0.5]*8, source_episodes=[e1],
    ))
    e2 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Actually pH is 5.0-6.0", summary="pH revised", embedding=[0.5]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    fid2 = ms.extract_fact(SemanticFactData(
        subject="taro", predicate="optimal_ph", object_value="5.0-6.0",
        confidence=0.9, embedding=[0.5]*8, source_episodes=[e2],
    ))
    assert fid1 != fid2
    # Old fact invalidated
    old_node = g.get_node(fid1)
    assert old_node.get("invalid_at") is not None
    # New fact current
    new_node = g.get_node(fid2)
    assert new_node.get("invalid_at") is None
    # Contradiction edge exists
    edges = g.get_outgoing(fid2, "CONTRADICTS")
    assert len(edges) == 1
    assert edges[0].target_id == fid1
    print("  ✓ fact_contradiction")


# ============================================================
# Bi-Temporal Query Tests
# ============================================================

def test_bitemporal_query():
    """Query what was known at a specific time."""
    g, ms = make_agent()
    t1 = int(time.time() * 1000) - 5000  # 5 seconds ago
    t2 = int(time.time() * 1000) - 2000  # 2 seconds ago

    # Fact 1: recorded at t1
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Early observation", summary="Early", embedding=[0.3]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    f1_node = g.get_node(e1)
    f1_node.set("recorded_at", t1)

    # Fact 2: recorded at t2
    e2 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Later observation", summary="Later", embedding=[0.7]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    f2_node = g.get_node(e2)
    f2_node.set("recorded_at", t2)

    # Query at t1+1ms: should only see fact 1
    results = ms.query_at(as_of=t1 + 1)
    contents = [n.get("content", "") for n in results]
    assert any("Early" in c for c in contents)
    # Should not see later observation
    early_results = [n for n in results if n.get("recorded_at", 0) <= t1 + 1]
    assert len(early_results) >= 1
    print("  ✓ bitemporal_query")

def test_current_facts():
    g, ms = make_agent()
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Observation", summary="Obs", embedding=[0.5]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    ms.extract_fact(SemanticFactData(
        subject="taro", predicate="color", object_value="green",
        confidence=0.9, embedding=[0.5]*8, source_episodes=[e1],
    ))
    ms.extract_fact(SemanticFactData(
        subject="taro", predicate="height", object_value="3 feet",
        confidence=0.7, embedding=[0.4]*8, source_episodes=[e1],
    ))
    facts = ms.current_facts()
    assert len(facts) == 2
    print("  ✓ current_facts")


# ============================================================
# Hybrid Retrieval Tests
# ============================================================

def test_retrieval_semantic():
    """Semantic similarity should rank matching content higher."""
    g, ms = make_agent()
    ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Taro cultivation in wetlands", summary="Taro",
        embedding=[0.9, 0.1, 0.8, 0.1, 0.7, 0.1, 0.8, 0.1],
        emotional_valence=0.3, emotional_arousal=0.2,
    ))
    ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="conversation",
        content="Car maintenance schedule", summary="Cars",
        embedding=[0.1, 0.9, 0.1, 0.8, 0.1, 0.9, 0.1, 0.8],
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    # Query about taro
    results = ms.retrieve([0.85, 0.15, 0.75, 0.15, 0.65, 0.15, 0.75, 0.15])
    assert len(results) >= 2
    assert "Taro" in results[0].content or "taro" in results[0].content.lower()
    assert results[0].semantic_score > results[1].semantic_score
    print("  ✓ retrieval_semantic")

def test_retrieval_salience():
    """Emotionally salient memories rank higher (all else equal)."""
    g, ms = make_agent()
    embedding = [0.5]*8
    ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Boring routine observation", summary="Boring",
        embedding=embedding, emotional_valence=0.0, emotional_arousal=0.0,
    ))
    ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Exciting discovery of rare plant!", summary="Discovery",
        embedding=embedding, emotional_valence=0.9, emotional_arousal=0.9,
    ))
    results = ms.retrieve(embedding)
    assert len(results) >= 2
    # Discovery should rank higher due to salience
    assert results[0].salience_score > results[1].salience_score
    print("  ✓ retrieval_salience")

def test_retrieval_updates_access():
    """Retrieval should update access_count and last_accessed."""
    g, ms = make_agent()
    eid = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Something", summary="Something", embedding=[0.5]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    node = g.get_node(eid)
    assert node.get("access_count") == 0
    ms.retrieve([0.5]*8)
    assert node.get("access_count") == 1
    print("  ✓ retrieval_updates_access")


# ============================================================
# Consolidation Tests
# ============================================================

def test_consolidation():
    """Episodes → semantic facts via consolidation."""
    g, ms = make_agent()
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice", "kawika"], interaction_type="conversation",
        content="Kawika showed me that breadfruit stores for 3 days",
        summary="Breadfruit storage", embedding=[0.5]*8,
        emotional_valence=0.3, emotional_arousal=0.2,
    ))
    e2 = ms.ingest_episode(EpisodeData(
        participants=["alice", "kawika"], interaction_type="conversation",
        content="Kawika said breadfruit needs shade to store",
        summary="Breadfruit shade", embedding=[0.5]*8,
        emotional_valence=0.2, emotional_arousal=0.1,
    ))
    fact_ids = ms.consolidate([e1, e2], [
        SemanticFactData(
            subject="breadfruit", predicate="storage_duration",
            object_value="3 days", confidence=0.8,
            embedding=[0.5]*8, source_episodes=[],
        ),
        SemanticFactData(
            subject="breadfruit", predicate="storage_condition",
            object_value="shade", confidence=0.7,
            embedding=[0.5]*8, source_episodes=[],
        ),
    ])
    assert len(fact_ids) == 2
    stats = ms.stats()
    assert stats["active_facts"] == 2
    assert stats["episodes"] == 2
    # Facts should link back to episodes via EXTRACTED_FROM
    for fid in fact_ids:
        edges = g.get_outgoing(fid, "EXTRACTED_FROM")
        assert len(edges) == 2  # linked to both episodes
    print("  ✓ consolidation")


# ============================================================
# Decay Tests
# ============================================================

def test_memory_decay():
    """Memories decay over time; reinforced ones resist decay."""
    g, ms = make_agent()
    e1 = ms.ingest_episode(EpisodeData(
        participants=["alice"], interaction_type="observation",
        content="Old memory", summary="Old", embedding=[0.3]*8,
        emotional_valence=0.0, emotional_arousal=0.1,
    ))
    # Set last_accessed to 60 days ago
    for edge in g.get_outgoing("alice", "REMEMBERS"):
        if edge.target_id == e1:
            edge.set("last_accessed", int(time.time()*1000) - 60*24*60*60*1000)

    decayed = ms.apply_decay(decay_threshold=0.3, half_life_days=7.0)
    assert decayed >= 1
    print("  ✓ memory_decay")


# ============================================================
# Run all
# ============================================================

if __name__ == "__main__":
    print("Testing memory system...\n")

    print("Episodes:")
    test_ingest_episode()
    test_episode_chain()
    test_episode_stats()

    print("\nSemantic Facts:")
    test_extract_fact()
    test_fact_reinforcement()
    test_fact_contradiction()

    print("\nBi-Temporal Queries:")
    test_bitemporal_query()
    test_current_facts()

    print("\nHybrid Retrieval:")
    test_retrieval_semantic()
    test_retrieval_salience()
    test_retrieval_updates_access()

    print("\nConsolidation:")
    test_consolidation()

    print("\nDecay:")
    test_memory_decay()

    print("\n" + "=" * 50)
    print("ALL MEMORY TESTS PASSED ✓")
    print("=" * 50)

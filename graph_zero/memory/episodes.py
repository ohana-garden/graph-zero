"""
Graph Zero Memory System

Episodes: Temporal sequences of interactions, stored as graph structures.
Bi-temporal: Every fact has (valid_at, recorded_at, invalid_at).
Hybrid retrieval: Combines embedding similarity, graph proximity, and recency.
Consolidation: Frequent patterns compress into durable memories.

Based on Graphiti-style episodic memory adapted for the moral geometry framework.
"""

import math
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import NT, ET, _cosine_similarity


# ============================================================
# Temporal Primitives
# ============================================================

@dataclass
class BiTemporal:
    """Bi-temporal timestamp pair.

    valid_at: When the fact became true in the real world.
    recorded_at: When we learned about it (always monotonic).
    invalid_at: When the fact stopped being true (None = still valid).
    """
    valid_at: int       # ms since epoch
    recorded_at: int    # ms since epoch
    invalid_at: Optional[int] = None  # ms since epoch, None = current

    @property
    def is_current(self) -> bool:
        return self.invalid_at is None

    def invalidate(self, at: Optional[int] = None) -> None:
        self.invalid_at = at or int(time.time() * 1000)


def now_ms() -> int:
    return int(time.time() * 1000)


# ============================================================
# Episode — a sequence of interactions
# ============================================================

class EpisodeStatus(Enum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    CONSOLIDATED = "CONSOLIDATED"


@dataclass
class EpisodeEntry:
    """A single entry in an episode."""
    entry_id: str
    actor_id: str
    action: str
    content: str
    embedding: Optional[list[float]] = None
    timestamp: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = now_ms()


@dataclass
class Episode:
    """A temporal sequence of related interactions."""
    episode_id: str
    community_id: str
    participants: list[str]
    entries: list[EpisodeEntry] = field(default_factory=list)
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    started_at: int = 0
    closed_at: Optional[int] = None
    summary_embedding: Optional[list[float]] = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.started_at == 0:
            self.started_at = now_ms()

    def add_entry(self, entry: EpisodeEntry) -> None:
        self.entries.append(entry)

    def close(self) -> None:
        self.status = EpisodeStatus.CLOSED
        self.closed_at = now_ms()


# ============================================================
# Memory Store — graph-backed episodic memory
# ============================================================

class MemoryStore:
    """Graph-backed episodic memory with bi-temporal queries.

    Stores episodes as Episode nodes with FOLLOWED_BY chains,
    individual memories as Memory nodes linked to agents via REMEMBERS.
    """

    def __init__(self, graph: PropertyGraph):
        self.graph = graph

    # --------------------------------------------------------
    # Episode Ingestion
    # --------------------------------------------------------

    def ingest_episode(self, episode: Episode) -> Node:
        """Ingest a complete episode into the graph.

        Creates:
        - Episode node
        - Memory nodes for each entry
        - FOLLOWED_BY chain between entries
        - REMEMBERS edges from participants to each memory
        - PART_OF edges from memories to episode
        """
        g = self.graph
        ts = now_ms()

        # Episode node
        ep_node = g.add_node(episode.episode_id, NT.EPISODE, {
            "community_id": episode.community_id,
            "participants": episode.participants,
            "status": episode.status.value,
            "started_at": episode.started_at,
            "closed_at": episode.closed_at,
            "entry_count": len(episode.entries),
            "tags": episode.tags,
            "summary_embedding": episode.summary_embedding,
            "valid_at": episode.started_at,
            "recorded_at": ts,
            "invalid_at": None,
        })

        prev_mem_id = None
        for i, entry in enumerate(episode.entries):
            mem_id = f"mem_{episode.episode_id}_{i}"
            mem_node = g.add_node(mem_id, NT.MEMORY, {
                "actor_id": entry.actor_id,
                "action": entry.action,
                "content": entry.content,
                "embedding": entry.embedding,
                "episode_id": episode.episode_id,
                "sequence": i,
                "timestamp": entry.timestamp,
                "valid_at": entry.timestamp,
                "recorded_at": ts,
                "invalid_at": None,
                "access_count": 0,
                "last_accessed": ts,
                "strength": 1.0,
                "metadata": entry.metadata,
            })

            # PART_OF episode
            g.add_edge(mem_id, episode.episode_id, ET.PART_OF)

            # FOLLOWED_BY chain
            if prev_mem_id:
                g.add_edge(prev_mem_id, mem_id, ET.FOLLOWED_BY, {
                    "delta_ms": entry.timestamp - episode.entries[i-1].timestamp,
                })
            prev_mem_id = mem_id

            # REMEMBERS from each participant
            for pid in episode.participants:
                if g.has_node(pid):
                    g.add_edge(pid, mem_id, ET.REMEMBERS, {
                        "last_accessed": ts,
                        "access_count": 0,
                        "strength": 1.0,
                    })

        return ep_node

    # --------------------------------------------------------
    # Memory Creation (single facts)
    # --------------------------------------------------------

    def store_memory(self, memory_id: str, agent_id: str,
                     content: str, embedding: Optional[list[float]] = None,
                     valid_at: Optional[int] = None,
                     metadata: Optional[dict] = None) -> Node:
        """Store a single memory fact for an agent."""
        g = self.graph
        ts = now_ms()

        mem = g.add_node(memory_id, NT.MEMORY, {
            "actor_id": agent_id,
            "action": "remember",
            "content": content,
            "embedding": embedding,
            "valid_at": valid_at or ts,
            "recorded_at": ts,
            "invalid_at": None,
            "access_count": 0,
            "last_accessed": ts,
            "strength": 1.0,
            "metadata": metadata or {},
        })

        if g.has_node(agent_id):
            g.add_edge(agent_id, memory_id, ET.REMEMBERS, {
                "last_accessed": ts,
                "access_count": 0,
                "strength": 1.0,
            })

        return mem

    # --------------------------------------------------------
    # Bi-Temporal Queries
    # --------------------------------------------------------

    def query_at(self, agent_id: str,
                 as_of: Optional[int] = None,
                 valid_at: Optional[int] = None) -> list[Node]:
        """Bi-temporal query: what did we know at time X about time Y?

        as_of: Knowledge time — what was recorded by this point.
        valid_at: Reality time — what was true at this point.

        Both default to "now" if not specified.
        """
        g = self.graph
        ts = now_ms()
        as_of = as_of or ts
        valid_at = valid_at or ts

        results = []
        for edge in g.get_outgoing(agent_id, ET.REMEMBERS):
            mem = g.get_node(edge.target_id)
            if not mem or mem.node_type != NT.MEMORY:
                continue

            # Knowledge-time filter: was this recorded by as_of?
            recorded = mem.get("recorded_at", 0)
            if recorded > as_of:
                continue

            # Reality-time filter: was this valid at valid_at?
            mem_valid = mem.get("valid_at", 0)
            mem_invalid = mem.get("invalid_at")
            if mem_valid > valid_at:
                continue
            if mem_invalid is not None and mem_invalid <= valid_at:
                continue

            results.append(mem)

        return results

    def invalidate_memory(self, memory_id: str,
                          at: Optional[int] = None) -> bool:
        """Mark a memory as no longer valid (soft delete)."""
        g = self.graph
        mem = g.get_node(memory_id)
        if not mem or mem.node_type != NT.MEMORY:
            return False
        mem.set("invalid_at", at or now_ms())
        return True

    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------

    def retrieve(self, agent_id: str, query_embedding: list[float],
                 limit: int = 10,
                 recency_weight: float = 0.3,
                 similarity_weight: float = 0.5,
                 strength_weight: float = 0.2,
                 as_of: Optional[int] = None) -> list[tuple[Node, float]]:
        """Hybrid retrieval combining embedding similarity, recency, and strength.

        Score = similarity_weight * cosine_sim
              + recency_weight * recency_score
              + strength_weight * strength

        Returns list of (memory_node, score) sorted by score descending.
        """
        g = self.graph
        ts = as_of or now_ms()

        # Get all current memories for this agent
        candidates = self.query_at(agent_id, as_of=ts, valid_at=ts)

        scored = []
        for mem in candidates:
            mem_embedding = mem.get("embedding")
            if not mem_embedding:
                continue

            # Cosine similarity
            sim = _cosine_similarity(query_embedding, mem_embedding)

            # Recency: exponential decay, half-life = 7 days
            last_accessed = mem.get("last_accessed", 0)
            age_days = max(0, (ts - last_accessed)) / (24 * 60 * 60 * 1000)
            recency = math.exp(-0.693 * age_days / 7.0)  # half-life 7 days

            # Strength (from consolidation / access count)
            strength = mem.get("strength", 0.5)

            score = (similarity_weight * max(0, sim)
                     + recency_weight * recency
                     + strength_weight * strength)

            scored.append((mem, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update access counts for retrieved memories
        for mem, _ in scored[:limit]:
            count = mem.get("access_count", 0) + 1
            mem.set("access_count", count)
            mem.set("last_accessed", ts)
            # Strengthen on access (diminishing returns)
            current_strength = mem.get("strength", 0.5)
            mem.set("strength", min(1.0, current_strength + 0.05 * (1 - current_strength)))

        return scored[:limit]

    # --------------------------------------------------------
    # Graph-Proximity Retrieval
    # --------------------------------------------------------

    def retrieve_related(self, memory_id: str, max_depth: int = 3,
                         limit: int = 10) -> list[tuple[Node, int]]:
        """Find memories related by graph proximity.

        Walks FOLLOWED_BY (temporal), RELATES_TO (semantic),
        and shared episode membership.
        """
        g = self.graph
        results = []
        seen = {memory_id}

        # Walk FOLLOWED_BY chain (temporal neighbors)
        for edge_type in [ET.FOLLOWED_BY, ET.RELATES_TO]:
            traversed = g.traverse(memory_id, edge_type, max_depth=max_depth)
            for node, depth, _ in traversed:
                if node.id not in seen and node.node_type == NT.MEMORY:
                    if node.get("invalid_at") is None:
                        results.append((node, depth))
                        seen.add(node.id)

        # Same-episode memories
        mem = g.get_node(memory_id)
        if mem:
            ep_id = mem.get("episode_id")
            if ep_id:
                for edge in g.get_incoming(ep_id, ET.PART_OF):
                    node = g.get_node(edge.source_id)
                    if node and node.id not in seen and node.node_type == NT.MEMORY:
                        if node.get("invalid_at") is None:
                            results.append((node, 1))
                            seen.add(node.id)

        results.sort(key=lambda x: x[1])
        return results[:limit]

    # --------------------------------------------------------
    # Consolidation — compress repeated patterns
    # --------------------------------------------------------

    def consolidate(self, agent_id: str,
                    similarity_threshold: float = 0.85,
                    min_count: int = 3) -> list[Node]:
        """Find clusters of similar memories and consolidate them.

        If N memories are very similar (cosine > threshold),
        create a single consolidated memory with boosted strength.
        Original memories are invalidated, not deleted.

        Returns list of new consolidated memory nodes.
        """
        g = self.graph
        ts = now_ms()
        current = self.query_at(agent_id, as_of=ts, valid_at=ts)

        # Only consider memories with embeddings
        with_emb = [(m, m.get("embedding")) for m in current if m.get("embedding")]

        # Find clusters via greedy clustering
        used = set()
        clusters = []
        for i, (mem_i, emb_i) in enumerate(with_emb):
            if mem_i.id in used:
                continue
            cluster = [mem_i]
            for j, (mem_j, emb_j) in enumerate(with_emb):
                if i == j or mem_j.id in used:
                    continue
                sim = _cosine_similarity(emb_i, emb_j)
                if sim >= similarity_threshold:
                    cluster.append(mem_j)

            if len(cluster) >= min_count:
                clusters.append(cluster)
                for m in cluster:
                    used.add(m.id)

        # Create consolidated memories
        consolidated = []
        for cluster in clusters:
            # Use the most-accessed memory's content as representative
            representative = max(cluster, key=lambda m: m.get("access_count", 0))
            embs = [m.get("embedding") for m in cluster if m.get("embedding")]

            # Average embedding
            if embs:
                avg_emb = [sum(e[d] for e in embs) / len(embs) for d in range(len(embs[0]))]
            else:
                avg_emb = None

            # Total strength is boosted by cluster size
            total_access = sum(m.get("access_count", 0) for m in cluster)
            strength = min(1.0, 0.5 + 0.1 * len(cluster))

            con_id = f"con_{agent_id}_{hashlib.sha256(representative.id.encode()).hexdigest()[:8]}"
            con_mem = self.store_memory(
                con_id, agent_id,
                content=f"[consolidated x{len(cluster)}] {representative.get('content', '')}",
                embedding=avg_emb,
                valid_at=min(m.get("valid_at", ts) for m in cluster),
                metadata={
                    "consolidated_from": [m.id for m in cluster],
                    "cluster_size": len(cluster),
                    "total_access_count": total_access,
                },
            )
            con_mem.set("strength", strength)
            con_mem.set("access_count", total_access)
            consolidated.append(con_mem)

            # Invalidate originals
            for m in cluster:
                m.set("invalid_at", ts)

        return consolidated

    # --------------------------------------------------------
    # Decay — weaken old unused memories
    # --------------------------------------------------------

    def apply_decay(self, agent_id: str,
                    half_life_days: float = 30.0,
                    min_strength: float = 0.01) -> int:
        """Apply temporal decay to memory strengths.

        Memories lose strength exponentially based on time since last access.
        Returns count of memories that fell below min_strength (candidates for pruning).
        """
        g = self.graph
        ts = now_ms()
        weak_count = 0

        for edge in g.get_outgoing(agent_id, ET.REMEMBERS):
            mem = g.get_node(edge.target_id)
            if not mem or mem.node_type != NT.MEMORY:
                continue
            if mem.get("invalid_at") is not None:
                continue

            last_accessed = mem.get("last_accessed", ts)
            age_days = max(0, (ts - last_accessed)) / (24 * 60 * 60 * 1000)
            decay_factor = math.exp(-0.693 * age_days / half_life_days)

            current_strength = mem.get("strength", 1.0)
            new_strength = current_strength * decay_factor

            mem.set("strength", new_strength)
            edge.set("strength", new_strength)

            if new_strength < min_strength:
                weak_count += 1

        return weak_count

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    def memory_count(self, agent_id: str, include_invalid: bool = False) -> int:
        """Count memories for an agent."""
        g = self.graph
        count = 0
        for edge in g.get_outgoing(agent_id, ET.REMEMBERS):
            mem = g.get_node(edge.target_id)
            if mem and mem.node_type == NT.MEMORY:
                if include_invalid or mem.get("invalid_at") is None:
                    count += 1
        return count

    def episode_count(self, community_id: Optional[str] = None) -> int:
        episodes = self.graph.get_nodes_by_type(NT.EPISODE)
        if community_id:
            episodes = [e for e in episodes if e.get("community_id") == community_id]
        return len(episodes)

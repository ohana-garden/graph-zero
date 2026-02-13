"""
Graph Zero Memory System

Inspired by Graphiti but built on Graph Zero's property graph.
Three memory types:
  - Episodic: raw interaction records with timestamps
  - Semantic: consolidated facts extracted from episodes
  - Procedural: learned skills and patterns

Bi-temporal model:
  - valid_at: when the fact was true in the world
  - recorded_at: when the system learned it
  - invalid_at: when the fact stopped being true (None = still valid)

Hybrid retrieval combines:
  1. Semantic similarity (embedding cosine distance)
  2. Graph proximity (hops from query context)
  3. Recency weighting (exponential decay)
  4. Emotional salience (from Hume EVI integration)
"""

import math
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import NT, ET, _cosine_similarity


# ============================================================
# Memory Node Types (extend NT)
# ============================================================

# These extend NT but we define as strings for the in-memory backend
MEMORY_NODE = "Memory"
EPISODE_NODE = "Episode"
SEMANTIC_FACT = "SemanticFact"
PROCEDURAL_SKILL = "ProceduralSkill"
MEMORY_CONTEXT = "MemoryContext"


# ============================================================
# Memory Edge Types (extend ET)
# ============================================================

REMEMBERS = "REMEMBERS"
PART_OF = "PART_OF"
FOLLOWED_BY = "FOLLOWED_BY"
EXTRACTED_FROM = "EXTRACTED_FROM"
RELATED_TO = "RELATES_TO"
CONTRADICTS = "CONTRADICTS"
SUPERSEDES_MEMORY = "SUPERSEDES_MEMORY"
REINFORCED_BY = "REINFORCED_BY"


# ============================================================
# Core Data Structures
# ============================================================

class MemoryType(Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class BiTemporalStamp:
    """Bi-temporal timestamp for memory validity tracking."""
    valid_at: int          # when the fact was true (ms since epoch)
    recorded_at: int       # when the system recorded it (ms since epoch)
    invalid_at: Optional[int] = None  # when it stopped being true (None = current)

    @property
    def is_current(self) -> bool:
        return self.invalid_at is None

    def invalidate(self, at: Optional[int] = None) -> None:
        self.invalid_at = at or int(time.time() * 1000)


@dataclass
class EpisodeData:
    """Raw interaction data for episodic memory."""
    participants: list[str]      # vessel_ids
    interaction_type: str        # conversation, observation, collaboration, etc.
    content: str                 # raw text content
    summary: str                 # compressed summary
    embedding: list[float]       # vector embedding of content
    emotional_valence: float     # -1.0 to 1.0 from Hume EVI
    emotional_arousal: float     # 0.0 to 1.0
    location_context: str = ""   # optional location
    tool_context: list[str] = field(default_factory=list)  # tools used


@dataclass
class SemanticFactData:
    """Extracted fact from one or more episodes."""
    subject: str
    predicate: str
    object_value: str
    confidence: float            # 0.0 to 1.0
    embedding: list[float]
    source_episodes: list[str]   # episode IDs it was extracted from
    reinforcement_count: int = 1


@dataclass
class RetrievalResult:
    """A single result from hybrid retrieval."""
    memory_id: str
    memory_type: MemoryType
    content: str
    score: float                  # combined retrieval score
    semantic_score: float
    recency_score: float
    proximity_score: float
    salience_score: float
    bi_temporal: BiTemporalStamp
    node: Node


# ============================================================
# Memory Store
# ============================================================

class MemoryStore:
    """Graph-backed memory system for a single agent.

    Wraps the property graph with memory-specific operations.
    In production, backed by FalkorDB + Graphiti patterns.
    """

    def __init__(self, graph: PropertyGraph, vessel_id: str):
        self.graph = graph
        self.vessel_id = vessel_id
        self._episode_counter = 0
        self._fact_counter = 0

    # --------------------------------------------------------
    # Episode Ingestion
    # --------------------------------------------------------

    def ingest_episode(self, data: EpisodeData,
                       valid_at: Optional[int] = None) -> str:
        """Ingest a raw interaction as episodic memory.

        Returns the episode node ID.
        """
        now = int(time.time() * 1000)
        self._episode_counter += 1
        episode_id = f"ep_{self.vessel_id}_{self._episode_counter}"

        self.graph.add_node(episode_id, EPISODE_NODE, {
            "vessel_id": self.vessel_id,
            "participants": data.participants,
            "interaction_type": data.interaction_type,
            "content": data.content,
            "summary": data.summary,
            "embedding": data.embedding,
            "emotional_valence": data.emotional_valence,
            "emotional_arousal": data.emotional_arousal,
            "location_context": data.location_context,
            "tool_context": data.tool_context,
            "valid_at": valid_at or now,
            "recorded_at": now,
            "invalid_at": None,
            "access_count": 0,
            "last_accessed": now,
        })

        # Link agent REMEMBERS episode
        self.graph.add_edge(self.vessel_id, episode_id, REMEMBERS, {
            "strength": 1.0,
            "last_accessed": now,
        })

        # Link to other participants
        for pid in data.participants:
            if pid != self.vessel_id and self.graph.has_node(pid):
                self.graph.add_edge(episode_id, pid, "INVOLVES", {
                    "role": "participant",
                })

        # Link to previous episode (temporal chain)
        prev_episodes = self._get_recent_episodes(limit=1, before=episode_id)
        if prev_episodes:
            self.graph.add_edge(prev_episodes[0].id, episode_id, FOLLOWED_BY, {
                "gap_ms": now - prev_episodes[0].get("recorded_at", 0),
            })

        return episode_id

    # --------------------------------------------------------
    # Semantic Fact Extraction
    # --------------------------------------------------------

    def extract_fact(self, data: SemanticFactData,
                     valid_at: Optional[int] = None) -> str:
        """Extract and store a semantic fact from episodes.

        Checks for contradictions and reinforcement with existing facts.
        Returns fact node ID.
        """
        now = int(time.time() * 1000)
        self._fact_counter += 1
        fact_id = f"fact_{self.vessel_id}_{self._fact_counter}"

        # Check for existing facts about same subject+predicate
        existing = self._find_matching_facts(data.subject, data.predicate)
        contradicted_ids = []

        for ex_node in existing:
            ex_obj = ex_node.get("object_value", "")
            if ex_obj == data.object_value:
                # Reinforcement — same fact seen again
                count = ex_node.get("reinforcement_count", 1) + 1
                ex_node.set("reinforcement_count", count)
                ex_node.set("confidence", min(1.0, data.confidence + 0.1 * (count - 1)))
                # Link reinforcement
                for ep_id in data.source_episodes:
                    if self.graph.has_node(ep_id):
                        self.graph.add_edge(ep_id, ex_node.id, REINFORCED_BY)
                return ex_node.id
            else:
                # Contradiction — different object for same subject+predicate
                # Invalidate old fact, mark for linking after node creation
                ex_node.set("invalid_at", now)
                contradicted_ids.append(ex_node.id)

        self.graph.add_node(fact_id, SEMANTIC_FACT, {
            "vessel_id": self.vessel_id,
            "subject": data.subject,
            "predicate": data.predicate,
            "object_value": data.object_value,
            "confidence": data.confidence,
            "embedding": data.embedding,
            "source_episodes": data.source_episodes,
            "reinforcement_count": data.reinforcement_count,
            "valid_at": valid_at or now,
            "recorded_at": now,
            "invalid_at": None,
        })

        # Link to source episodes
        for ep_id in data.source_episodes:
            if self.graph.has_node(ep_id):
                self.graph.add_edge(fact_id, ep_id, EXTRACTED_FROM)

        # Link contradiction edges (deferred until node exists)
        for old_id in contradicted_ids:
            self.graph.add_edge(fact_id, old_id, CONTRADICTS, {
                "detected_at": now,
            })

        # Link agent REMEMBERS fact
        self.graph.add_edge(self.vessel_id, fact_id, REMEMBERS, {
            "strength": data.confidence,
            "last_accessed": now,
        })

        return fact_id

    # --------------------------------------------------------
    # Bi-Temporal Queries
    # --------------------------------------------------------

    def query_at(self, as_of: int, valid_at: Optional[int] = None) -> list[Node]:
        """Bi-temporal query: what did the agent know at time `as_of`
        about the world at time `valid_at`?

        as_of: recorded_at cutoff (what we knew by then)
        valid_at: if set, filter to facts valid at that world-time
        """
        results = []
        for edge in self.graph.get_outgoing(self.vessel_id, REMEMBERS):
            node = self.graph.get_node(edge.target_id)
            if not node:
                continue

            recorded = node.get("recorded_at", 0)
            if recorded > as_of:
                continue  # wasn't known yet

            if valid_at is not None:
                fact_valid = node.get("valid_at", 0)
                fact_invalid = node.get("invalid_at")
                if fact_valid > valid_at:
                    continue  # wasn't true yet
                if fact_invalid is not None and fact_invalid <= valid_at:
                    continue  # was already invalidated

            results.append(node)
        return results

    def current_facts(self) -> list[Node]:
        """Get all currently valid semantic facts."""
        results = []
        for node in self.graph.get_nodes_by_type(SEMANTIC_FACT):
            if (node.get("vessel_id") == self.vessel_id
                    and node.get("invalid_at") is None):
                results.append(node)
        return results

    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------

    def retrieve(self, query_embedding: list[float],
                 context_node_ids: Optional[list[str]] = None,
                 limit: int = 10,
                 recency_half_life_days: float = 7.0,
                 weights: Optional[dict[str, float]] = None) -> list[RetrievalResult]:
        """Hybrid retrieval combining 4 signals.

        1. Semantic similarity (cosine of embeddings)
        2. Recency (exponential decay from last_accessed)
        3. Graph proximity (hops from context nodes)
        4. Emotional salience (arousal * |valence|)

        Weights default to: semantic=0.4, recency=0.25, proximity=0.2, salience=0.15
        """
        w = weights or {
            "semantic": 0.4,
            "recency": 0.25,
            "proximity": 0.2,
            "salience": 0.15,
        }

        now = int(time.time() * 1000)
        half_life_ms = recency_half_life_days * 24 * 60 * 60 * 1000

        # Compute proximity scores if context nodes given
        proximity_map: dict[str, float] = {}
        if context_node_ids:
            for ctx_id in context_node_ids:
                traversed = self.graph.traverse(ctx_id, RELATED_TO, max_depth=3,
                                                direction="both")
                for node, depth, _ in traversed:
                    score = 1.0 / (1.0 + depth)
                    proximity_map[node.id] = max(proximity_map.get(node.id, 0), score)

        candidates = []
        for edge in self.graph.get_outgoing(self.vessel_id, REMEMBERS):
            node = self.graph.get_node(edge.target_id)
            if not node:
                continue
            # Skip invalidated
            if node.get("invalid_at") is not None:
                continue

            node_embedding = node.get("embedding", [])

            # 1. Semantic similarity
            sem_score = _cosine_similarity(query_embedding, node_embedding) if node_embedding else 0.0
            sem_score = max(0.0, sem_score)  # clamp negatives

            # 2. Recency
            last_accessed = node.get("last_accessed", node.get("recorded_at", 0))
            age_ms = max(1, now - last_accessed)
            recency_score = math.exp(-0.693 * age_ms / half_life_ms)  # ln(2) ≈ 0.693

            # 3. Proximity
            prox_score = proximity_map.get(node.id, 0.0)

            # 4. Emotional salience
            valence = abs(node.get("emotional_valence", 0.0))
            arousal = node.get("emotional_arousal", 0.0)
            salience_score = valence * arousal

            # Combined score
            combined = (w["semantic"] * sem_score +
                        w["recency"] * recency_score +
                        w["proximity"] * prox_score +
                        w["salience"] * salience_score)

            # Determine memory type
            if node.node_type == EPISODE_NODE:
                mem_type = MemoryType.EPISODIC
                content = node.get("summary", node.get("content", ""))
            elif node.node_type == SEMANTIC_FACT:
                mem_type = MemoryType.SEMANTIC
                content = f"{node.get('subject')} {node.get('predicate')} {node.get('object_value')}"
            elif node.node_type == PROCEDURAL_SKILL:
                mem_type = MemoryType.PROCEDURAL
                content = node.get("description", "")
            else:
                mem_type = MemoryType.EPISODIC
                content = node.get("content", node.get("source_text", ""))

            candidates.append(RetrievalResult(
                memory_id=node.id,
                memory_type=mem_type,
                content=content,
                score=combined,
                semantic_score=sem_score,
                recency_score=recency_score,
                proximity_score=prox_score,
                salience_score=salience_score,
                bi_temporal=BiTemporalStamp(
                    valid_at=node.get("valid_at", 0),
                    recorded_at=node.get("recorded_at", 0),
                    invalid_at=node.get("invalid_at"),
                ),
                node=node,
            ))

        # Sort by combined score, return top N
        candidates.sort(key=lambda r: r.score, reverse=True)

        # Update access counts for retrieved memories
        for result in candidates[:limit]:
            node = result.node
            node.set("access_count", node.get("access_count", 0) + 1)
            node.set("last_accessed", now)

        return candidates[:limit]

    # --------------------------------------------------------
    # Memory Consolidation
    # --------------------------------------------------------

    def consolidate(self, episode_ids: list[str],
                    facts: list[SemanticFactData]) -> list[str]:
        """Consolidate episodes into semantic facts.

        This is the memory compression step:
        raw episodes → extracted facts with provenance links.
        """
        fact_ids = []
        for fact_data in facts:
            fact_data.source_episodes = episode_ids
            fid = self.extract_fact(fact_data)
            fact_ids.append(fid)
        return fact_ids

    # --------------------------------------------------------
    # Memory Decay
    # --------------------------------------------------------

    def apply_decay(self, decay_threshold: float = 0.01,
                    half_life_days: float = 30.0) -> int:
        """Apply memory decay: weaken REMEMBERS edges over time.

        Memories below threshold are not deleted (append-only)
        but their REMEMBERS strength drops, reducing retrieval priority.

        Returns number of decayed memories.
        """
        now = int(time.time() * 1000)
        half_life_ms = half_life_days * 24 * 60 * 60 * 1000
        decayed = 0

        for edge in self.graph.get_outgoing(self.vessel_id, REMEMBERS):
            last_accessed = edge.get("last_accessed", 0)
            age_ms = max(1, now - last_accessed)
            current_strength = edge.get("strength", 1.0)

            # Decay based on time since last access
            decay_factor = math.exp(-0.693 * age_ms / half_life_ms)
            new_strength = current_strength * decay_factor

            # Reinforced facts decay slower
            node = self.graph.get_node(edge.target_id)
            if node and node.node_type == SEMANTIC_FACT:
                reinforcement = node.get("reinforcement_count", 1)
                new_strength *= (1.0 + 0.1 * math.log(reinforcement))

            edge.set("strength", max(decay_threshold, new_strength))
            if new_strength <= decay_threshold:
                decayed += 1

        return decayed

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _get_recent_episodes(self, limit: int = 5,
                             before: Optional[str] = None) -> list[Node]:
        """Get recent episode nodes, ordered by recorded_at desc."""
        episodes = []
        for edge in self.graph.get_outgoing(self.vessel_id, REMEMBERS):
            node = self.graph.get_node(edge.target_id)
            if node and node.node_type == EPISODE_NODE:
                if before and node.id >= before:
                    continue
                episodes.append(node)
        episodes.sort(key=lambda n: n.get("recorded_at", 0), reverse=True)
        return episodes[:limit]

    def _find_matching_facts(self, subject: str, predicate: str) -> list[Node]:
        """Find existing semantic facts with same subject+predicate."""
        results = []
        for node in self.graph.get_nodes_by_type(SEMANTIC_FACT):
            if (node.get("vessel_id") == self.vessel_id
                    and node.get("subject") == subject
                    and node.get("predicate") == predicate
                    and node.get("invalid_at") is None):
                results.append(node)
        return results

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    def stats(self) -> dict:
        """Memory statistics for this agent."""
        episodes = 0
        facts = 0
        invalidated = 0
        for edge in self.graph.get_outgoing(self.vessel_id, REMEMBERS):
            node = self.graph.get_node(edge.target_id)
            if not node:
                continue
            if node.node_type == EPISODE_NODE:
                episodes += 1
            elif node.node_type == SEMANTIC_FACT:
                if node.get("invalid_at") is not None:
                    invalidated += 1
                else:
                    facts += 1
        return {
            "episodes": episodes,
            "active_facts": facts,
            "invalidated_facts": invalidated,
            "total": episodes + facts + invalidated,
        }

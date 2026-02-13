"""
Graph Zero Interface Layer

The entry point that ties everything together.
Handles:
  - Session management (agent login → constellation loaded)
  - Context-aware entry (query → terrain → relevant memories + tools)
  - Z-layer rendering (what to show at each depth level)
  - API surface for PWA/Hume EVI integration

Z-Layers (from nearest to farthest):
  0. Immediate — current conversation, active tools
  1. Personal — agent's memories, recent episodes, skills
  2. Community — local terrain, nearby agents, open conflicts
  3. Federated — shared terrain from other communities
  4. Bedrock — sacred/foundational texts, invariant principles
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from graph_zero.graph.backend import PropertyGraph, Node
from graph_zero.graph.schema import (
    NT, ET, assemble_constellation, Constellation,
    traverse_terrain, find_entry_points, compute_trust_flow,
    compute_vital_signs, VitalSigns,
)
from graph_zero.memory.memory import (
    MemoryStore, MemoryType, EpisodeData, RetrievalResult,
)
from graph_zero.execution.execution import (
    ExecutionEngine, ExecutionRequest, ExecutionResult, ExecutionStatus,
)


# ============================================================
# Z-Layer System
# ============================================================

class ZLayer:
    IMMEDIATE = 0
    PERSONAL = 1
    COMMUNITY = 2
    FEDERATED = 3
    BEDROCK = 4


@dataclass
class ZLayerContent:
    """Content at a specific z-layer depth."""
    layer: int
    label: str
    items: list[dict] = field(default_factory=list)
    count: int = 0


@dataclass
class ContextFrame:
    """A complete context frame assembled from all z-layers.

    This is what an agent "sees" at any given moment.
    """
    vessel_id: str
    timestamp: int
    layers: dict[int, ZLayerContent] = field(default_factory=dict)
    total_items: int = 0
    assembly_ms: int = 0

    def get_layer(self, z: int) -> ZLayerContent:
        return self.layers.get(z, ZLayerContent(layer=z, label="empty"))

    def all_items(self) -> list[dict]:
        """Flatten all items across layers, nearest first."""
        items = []
        for z in sorted(self.layers.keys()):
            items.extend(self.layers[z].items)
        return items


# ============================================================
# Session — an agent's active interaction context
# ============================================================

@dataclass
class SessionState:
    """Active session for an agent."""
    session_id: str
    vessel_id: str
    community_id: str
    constellation: Optional[Constellation]
    memory_store: MemoryStore
    trust_ceiling: float
    started_at: int
    last_active: int
    conversation_history: list[dict] = field(default_factory=list)
    active_tools: list[str] = field(default_factory=list)
    context_frame: Optional[ContextFrame] = None


class SessionManager:
    """Manages agent sessions and context assembly."""

    def __init__(self, graph: PropertyGraph, engine: ExecutionEngine):
        self.graph = graph
        self.engine = engine
        self._sessions: dict[str, SessionState] = {}
        self._session_counter = 0

    def create_session(self, vessel_id: str,
                       community_id: str) -> Optional[SessionState]:
        """Initialize a session for an agent."""
        constellation = assemble_constellation(self.graph, vessel_id)
        if not constellation:
            return None

        # Compute trust
        trust_result = compute_trust_flow(self.graph, vessel_id)

        # Create memory store
        memory = MemoryStore(self.graph, vessel_id)

        # Get available tools
        available = self.engine.get_available_tools(
            vessel_id, trust_result.trust_ceiling)
        tool_ids = [t.id for t in available]

        now = int(time.time() * 1000)
        self._session_counter += 1
        session_id = f"session_{vessel_id}_{self._session_counter}"

        session = SessionState(
            session_id=session_id,
            vessel_id=vessel_id,
            community_id=community_id,
            constellation=constellation,
            memory_store=memory,
            trust_ceiling=trust_result.trust_ceiling,
            started_at=now,
            last_active=now,
            active_tools=tool_ids,
        )

        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        """Close a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            # Record conversation as episode if non-empty
            if session.conversation_history:
                session.memory_store.ingest_episode(EpisodeData(
                    participants=[session.vessel_id],
                    interaction_type="session",
                    content=str(session.conversation_history[-3:]),
                    summary=f"Session with {len(session.conversation_history)} turns",
                    embedding=[0.5] * 8,
                    emotional_valence=0.0,
                    emotional_arousal=0.1,
                ))
            return True
        return False

    # --------------------------------------------------------
    # Context Assembly
    # --------------------------------------------------------

    def assemble_context(self, session: SessionState,
                         query_text: str,
                         query_embedding: list[float]) -> ContextFrame:
        """Assemble a complete context frame for a query.

        Walks all z-layers from nearest to farthest.
        """
        start = time.time()
        now = int(time.time() * 1000)
        frame = ContextFrame(vessel_id=session.vessel_id, timestamp=now)

        # Z0: Immediate — conversation history + active tools
        z0 = ZLayerContent(layer=ZLayer.IMMEDIATE, label="Immediate")
        for turn in session.conversation_history[-5:]:
            z0.items.append({"type": "conversation", **turn})
        for tid in session.active_tools:
            tool = self.graph.get_node(tid)
            if tool:
                z0.items.append({
                    "type": "tool",
                    "tool_id": tid,
                    "name": tool.get("name"),
                    "domain": tool.get("domain"),
                })
        z0.count = len(z0.items)
        frame.layers[ZLayer.IMMEDIATE] = z0

        # Z1: Personal — memories relevant to query
        z1 = ZLayerContent(layer=ZLayer.PERSONAL, label="Personal")
        memories = session.memory_store.retrieve(query_embedding, limit=5)
        for mem in memories:
            z1.items.append({
                "type": "memory",
                "memory_type": mem.memory_type.value,
                "content": mem.content,
                "score": mem.score,
                "memory_id": mem.memory_id,
            })
        z1.count = len(z1.items)
        frame.layers[ZLayer.PERSONAL] = z1

        # Z2: Community — terrain traversal with local provenance
        z2 = ZLayerContent(layer=ZLayer.COMMUNITY, label="Community")
        entry_ids = find_entry_points(self.graph, query_embedding, threshold=0.5, limit=3)
        terrain_results = traverse_terrain(self.graph, entry_ids, max_depth=4, limit=5)
        for tr in terrain_results:
            if tr.node.get("layer") in ("community", "earned"):
                z2.items.append({
                    "type": "terrain",
                    "source_text": tr.source_text,
                    "layer": tr.layer,
                    "authority": tr.authority_weight,
                    "path_weight": tr.path_weight,
                    "depth": tr.depth,
                })
        z2.count = len(z2.items)
        frame.layers[ZLayer.COMMUNITY] = z2

        # Z3: Federated — terrain from other communities
        z3 = ZLayerContent(layer=ZLayer.FEDERATED, label="Federated")
        for tr in terrain_results:
            node = tr.node
            if node.get("_federated_from"):
                z3.items.append({
                    "type": "federated_terrain",
                    "source_text": tr.source_text,
                    "from_community": node.get("_federated_from"),
                    "authority": tr.authority_weight,
                })
        # Also check explicitly federated nodes
        for node in self.graph.get_nodes_by_type(NT.TERRAIN_NODE):
            if (node.get("_federated_from") and
                    node.id not in {i.get("node_id") for i in z3.items}):
                embedding = node.get("embedding", [])
                if embedding:
                    from graph_zero.graph.schema import _cosine_similarity
                    sim = _cosine_similarity(query_embedding, embedding)
                    if sim > 0.5:
                        z3.items.append({
                            "type": "federated_terrain",
                            "source_text": node.get("source_text", ""),
                            "from_community": node.get("_federated_from"),
                            "similarity": sim,
                        })
        z3.count = len(z3.items)
        frame.layers[ZLayer.FEDERATED] = z3

        # Z4: Bedrock — foundational terrain
        z4 = ZLayerContent(layer=ZLayer.BEDROCK, label="Bedrock")
        for tr in terrain_results:
            if tr.node.get("layer") == "bedrock":
                z4.items.append({
                    "type": "bedrock",
                    "source_text": tr.source_text,
                    "authority": tr.authority_weight,
                })
        z4.count = len(z4.items)
        frame.layers[ZLayer.BEDROCK] = z4

        # Totals
        frame.total_items = sum(lc.count for lc in frame.layers.values())
        frame.assembly_ms = int((time.time() - start) * 1000)
        session.context_frame = frame
        session.last_active = now

        return frame

    # --------------------------------------------------------
    # Query Processing
    # --------------------------------------------------------

    def process_query(self, session: SessionState,
                      query_text: str,
                      query_embedding: list[float]) -> dict:
        """Process a user query through the full pipeline.

        1. Assemble context from all z-layers
        2. Add to conversation history
        3. Return structured response context

        The actual LLM call happens outside this function.
        This provides the context the LLM needs.
        """
        # Record the query
        session.conversation_history.append({
            "role": "user",
            "text": query_text,
            "timestamp": int(time.time() * 1000),
        })

        # Assemble context
        frame = self.assemble_context(session, query_text, query_embedding)

        return {
            "session_id": session.session_id,
            "vessel_id": session.vessel_id,
            "trust_ceiling": session.trust_ceiling,
            "context_frame": {
                "total_items": frame.total_items,
                "assembly_ms": frame.assembly_ms,
                "z0_immediate": frame.get_layer(ZLayer.IMMEDIATE).count,
                "z1_personal": frame.get_layer(ZLayer.PERSONAL).count,
                "z2_community": frame.get_layer(ZLayer.COMMUNITY).count,
                "z3_federated": frame.get_layer(ZLayer.FEDERATED).count,
                "z4_bedrock": frame.get_layer(ZLayer.BEDROCK).count,
            },
            "available_tools": session.active_tools,
            "items": frame.all_items(),
        }

    def add_response(self, session: SessionState,
                     response_text: str) -> None:
        """Record an assistant response in conversation history."""
        session.conversation_history.append({
            "role": "assistant",
            "text": response_text,
            "timestamp": int(time.time() * 1000),
        })

    # --------------------------------------------------------
    # Tool Invocation (convenience wrapper)
    # --------------------------------------------------------

    def invoke_tool(self, session: SessionState,
                    tool_id: str, input_data: dict,
                    context: str = "") -> ExecutionResult:
        """Invoke a tool within a session context."""
        req = self.engine.create_request(
            session.vessel_id, tool_id, input_data, context)
        return self.engine.execute(req, session.trust_ceiling)


# ============================================================
# Community Dashboard — the Navigator's view
# ============================================================

@dataclass
class DashboardView:
    """What a community steward sees."""
    community_id: str
    vital_signs: VitalSigns
    active_sessions: int
    visiting_agents: int
    open_conflicts: int
    federation_links: int
    recent_terrain_count: int
    total_agents: int
    total_terrain: int


def build_dashboard(graph: PropertyGraph,
                    community_id: str,
                    session_manager: Optional[SessionManager] = None) -> DashboardView:
    """Build a community dashboard view."""
    signs = compute_vital_signs(graph, community_id)

    active_sessions = 0
    if session_manager:
        active_sessions = sum(
            1 for s in session_manager._sessions.values()
            if s.community_id == community_id
        )

    visiting = sum(1 for n in graph.get_nodes_by_type(NT.VESSEL_ANCHOR)
                   if n.get("type") == "visiting")

    total_agents = sum(1 for n in graph.get_nodes_by_type(NT.VESSEL_ANCHOR)
                       if n.get("active", False))

    total_terrain = len(graph.get_nodes_by_type(NT.TERRAIN_NODE))

    fed_links = sum(1 for n in graph.get_nodes_by_type(NT.VESSEL_ANCHOR)
                    if n.get("type") == "visiting")

    return DashboardView(
        community_id=community_id,
        vital_signs=signs,
        active_sessions=active_sessions,
        visiting_agents=visiting,
        open_conflicts=signs.open_conflicts,
        federation_links=fed_links,
        recent_terrain_count=signs.terrain_additions_30d,
        total_agents=total_agents,
        total_terrain=total_terrain,
    )

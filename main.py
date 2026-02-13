"""
Graph Zero API Service

FastAPI wrapper around Graph Zero, deployable on Railway.
Connects to FalkorDB for persistent storage (falls back to in-memory).
"""

import os
import time
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Graph Zero imports
from graph_zero.graph.backend import PropertyGraph
from graph_zero.graph.schema import (
    NT, ET, bootstrap_community, create_agent, assemble_constellation,
    add_terrain_node, connect_terrain, traverse_terrain, find_entry_points,
    compute_trust_flow, compute_vital_signs, record_interaction,
    VIRTUE_NAMES,
)
from graph_zero.moral.geometry import (
    PhaseState, score_action, project_position, all_constraints_satisfied,
    check_velocity, DEFAULT_COUPLINGS, NUM_VIRTUES,
)
from graph_zero.provenance.provenance import (
    is_tainted, check_authority_closure, get_provenance_chain,
    promote_external_claim, create_conflict_set, resolve_conflict,
    create_challenge, get_open_conflicts, Contender,
)
from graph_zero.memory.memory import (
    MemoryStore, EpisodeData, SemanticFactData,
)
from graph_zero.execution.execution import (
    ExecutionEngine, ToolSpec, ToolDomain, SandboxProfile, SandboxLevel,
    register_tool, grant_tool_access,
)
from graph_zero.federation.federation import (
    create_snapshot, restore_snapshot, SnapshotFormat,
    create_clone_envelope, admit_visiting_agent, TRUST_DAMPENING_FACTOR,
    LogReplicator, ReplicationMode,
)
from graph_zero.interface.interface import (
    SessionManager, build_dashboard, ZLayer,
)


# ============================================================
# Global State
# ============================================================

graph: PropertyGraph = None
engine: ExecutionEngine = None
session_mgr: SessionManager = None
replicator: LogReplicator = None
COMMUNITY_ID = os.getenv("COMMUNITY_ID", "lower_puna")
COMMUNITY_NAME = os.getenv("COMMUNITY_NAME", "Lower Puna Community")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bootstrap community on startup."""
    global graph, engine, session_mgr, replicator
    graph = PropertyGraph()
    bootstrap_community(graph, COMMUNITY_ID, COMMUNITY_NAME)
    engine = ExecutionEngine(graph)
    session_mgr = SessionManager(graph, engine)
    replicator = LogReplicator()
    print(f"Graph Zero booted: {COMMUNITY_NAME} ({COMMUNITY_ID})")
    print(f"  Virtues: {len(graph.get_nodes_by_type(NT.VIRTUE_ANCHOR))}")
    print(f"  Nodes: {graph.node_count}, Edges: {graph.edge_count}")
    yield
    print("Graph Zero shutting down")


app = FastAPI(
    title="Graph Zero",
    description="Graph-native agent framework with moral geometry",
    version="0.6.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================

class AgentCreate(BaseModel):
    vessel_id: str
    name: str
    agent_type: str = "human"
    initial_position: Optional[list[float]] = None
    initial_kala: float = 100.0

class TerrainAdd(BaseModel):
    node_id: str
    source_text: str
    layer: str = "community"
    embedding: Optional[list[float]] = None
    virtue_scores: Optional[list[float]] = None
    provenance_type: str = "WITNESS"

class TerrainConnect(BaseModel):
    source_id: str
    target_id: str
    weight: float = 1.0
    provenance_type: str = "BEDROCK"

class TerrainQuery(BaseModel):
    entry_point_ids: Optional[list[str]] = None
    query_embedding: Optional[list[float]] = None
    max_depth: int = 5
    limit: int = 10
    threshold: float = 0.5

class MoralScoreRequest(BaseModel):
    vessel_id: str
    action_impacts: list[float]

class EpisodeIngest(BaseModel):
    vessel_id: str
    participants: list[str]
    interaction_type: str
    content: str
    summary: str
    embedding: list[float] = Field(default_factory=lambda: [0.5]*8)
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0

class FactExtract(BaseModel):
    vessel_id: str
    subject: str
    predicate: str
    object_value: str
    confidence: float = 0.8
    embedding: list[float] = Field(default_factory=lambda: [0.5]*8)
    source_episodes: list[str] = Field(default_factory=list)

class MemoryQuery(BaseModel):
    vessel_id: str
    query_embedding: list[float]
    limit: int = 10

class SessionCreate(BaseModel):
    vessel_id: str

class QueryProcess(BaseModel):
    session_id: str
    query_text: str
    query_embedding: list[float]

class InteractionRecord(BaseModel):
    agent_a: str
    agent_b: str
    interaction_type: str
    context: str = ""

class SnapshotCreate(BaseModel):
    format: str = "full"
    creator_key: str = "admin"

class ConflictCreate(BaseModel):
    object_key: str
    contenders: list[dict]
    contested_fields: list[str]

class ConflictResolve(BaseModel):
    conflict_id: str
    chosen_mutation_hash: str
    resolver_key: str
    justification: str = ""


# ============================================================
# Health
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "Graph Zero",
        "version": "0.6.0",
        "community": COMMUNITY_NAME,
        "community_id": COMMUNITY_ID,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "status": "operational",
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "nodes": graph.node_count, "edges": graph.edge_count}


# ============================================================
# Agent Endpoints
# ============================================================

@app.post("/agents")
async def create_agent_endpoint(req: AgentCreate):
    c = create_agent(graph, COMMUNITY_ID, req.vessel_id, req.name,
                     req.agent_type, req.initial_position, req.initial_kala)
    if not c:
        raise HTTPException(400, "Failed to create agent")
    return {
        "vessel_id": c.vessel_id,
        "name": c.name,
        "type": c.agent_type,
        "kala_balance": c.kala_balance,
        "moral_position": c.moral_position,
    }

@app.get("/agents/{vessel_id}")
async def get_agent(vessel_id: str):
    c = assemble_constellation(graph, vessel_id)
    if not c:
        raise HTTPException(404, "Agent not found")
    return {
        "vessel_id": c.vessel_id,
        "name": c.name,
        "type": c.agent_type,
        "trust_ceiling": c.trust_ceiling,
        "kala_balance": c.kala_balance,
        "moral_position": c.moral_position,
        "tools": [t.get("name") for t in c.tools],
        "skills": [s.get("name") for s in c.skills],
        "memories": len(c.memories),
    }

@app.get("/agents/{vessel_id}/trust")
async def get_trust(vessel_id: str):
    result = compute_trust_flow(graph, vessel_id)
    return {
        "vessel_id": result.vessel_id,
        "trust_ceiling": result.trust_ceiling,
        "path_count": result.path_count,
        "interaction_diversity": result.interaction_diversity,
    }

@app.post("/agents/interact")
async def interact(req: InteractionRecord):
    edge = record_interaction(graph, req.agent_a, req.agent_b,
                              req.interaction_type, req.context)
    if not edge:
        raise HTTPException(400, "Failed to record interaction")
    return {"status": "recorded", "edge_id": edge.id}


# ============================================================
# Terrain Endpoints
# ============================================================

@app.post("/terrain")
async def add_terrain(req: TerrainAdd):
    node = add_terrain_node(graph, req.node_id, req.source_text, req.layer,
                            req.embedding, req.virtue_scores, req.provenance_type)
    return {"node_id": node.id, "layer": req.layer, "provenance": req.provenance_type}

@app.post("/terrain/connect")
async def connect_terrain_endpoint(req: TerrainConnect):
    edge = connect_terrain(graph, req.source_id, req.target_id,
                           req.weight, req.provenance_type)
    if not edge:
        raise HTTPException(400, "Failed to connect terrain nodes")
    return {"edge_id": edge.id, "source": req.source_id, "target": req.target_id}

@app.post("/terrain/query")
async def query_terrain(req: TerrainQuery):
    entry_ids = req.entry_point_ids or []
    if req.query_embedding and not entry_ids:
        entry_ids = find_entry_points(graph, req.query_embedding,
                                      req.threshold, limit=5)
    results = traverse_terrain(graph, entry_ids, req.max_depth, req.limit)
    return {
        "entry_points": entry_ids,
        "results": [{
            "node_id": r.node.id,
            "source_text": r.source_text,
            "layer": r.layer,
            "authority_weight": r.authority_weight,
            "depth": r.depth,
            "path_weight": r.path_weight,
            "provenance_types": r.provenance_types,
        } for r in results],
    }


# ============================================================
# Moral Geometry Endpoints
# ============================================================

@app.post("/moral/score")
async def moral_score(req: MoralScoreRequest):
    if len(req.action_impacts) != NUM_VIRTUES:
        raise HTTPException(400, f"Need exactly {NUM_VIRTUES} impacts")
    c = assemble_constellation(graph, req.vessel_id)
    if not c:
        raise HTTPException(404, "Agent not found")
    state = PhaseState(position=c.moral_position, momentum=[0.0]*NUM_VIRTUES)
    result = score_action(state, req.action_impacts)
    return {
        "total_valence": result.total_valence,
        "virtue_impacts": result.virtue_impacts,
        "constraint_violations": result.constraint_violations,
        "projected_delta": result.projected_delta,
    }

@app.post("/moral/project")
async def moral_project(desired: list[float]):
    if len(desired) != NUM_VIRTUES:
        raise HTTPException(400, f"Need exactly {NUM_VIRTUES} values")
    projected = project_position(desired)
    return {
        "projected": projected,
        "constraints_satisfied": all_constraints_satisfied(projected),
    }


# ============================================================
# Provenance Endpoints
# ============================================================

@app.get("/provenance/{entity_id}/taint")
async def check_taint(entity_id: str):
    return {
        "entity_id": entity_id,
        "tainted": is_tainted(graph, entity_id),
        "citable": check_authority_closure(graph, entity_id),
    }

@app.get("/provenance/{entity_id}/chain")
async def provenance_chain(entity_id: str):
    chain = get_provenance_chain(graph, entity_id)
    return {"entity_id": entity_id, "chain": chain}


# ============================================================
# Memory Endpoints
# ============================================================

@app.post("/memory/episode")
async def ingest_episode(req: EpisodeIngest):
    ms = MemoryStore(graph, req.vessel_id)
    eid = ms.ingest_episode(EpisodeData(
        participants=req.participants,
        interaction_type=req.interaction_type,
        content=req.content,
        summary=req.summary,
        embedding=req.embedding,
        emotional_valence=req.emotional_valence,
        emotional_arousal=req.emotional_arousal,
    ))
    return {"episode_id": eid}

@app.post("/memory/fact")
async def extract_fact(req: FactExtract):
    ms = MemoryStore(graph, req.vessel_id)
    fid = ms.extract_fact(SemanticFactData(
        subject=req.subject,
        predicate=req.predicate,
        object_value=req.object_value,
        confidence=req.confidence,
        embedding=req.embedding,
        source_episodes=req.source_episodes,
    ))
    return {"fact_id": fid}

@app.post("/memory/retrieve")
async def retrieve_memories(req: MemoryQuery):
    ms = MemoryStore(graph, req.vessel_id)
    results = ms.retrieve(req.query_embedding, limit=req.limit)
    return {
        "results": [{
            "memory_id": r.memory_id,
            "type": r.memory_type.value,
            "content": r.content,
            "score": round(r.score, 4),
            "semantic": round(r.semantic_score, 4),
            "recency": round(r.recency_score, 4),
        } for r in results],
    }

@app.get("/memory/{vessel_id}/stats")
async def memory_stats(vessel_id: str):
    ms = MemoryStore(graph, vessel_id)
    return ms.stats()


# ============================================================
# Session / Context Endpoints
# ============================================================

@app.post("/session")
async def create_session(req: SessionCreate):
    session = session_mgr.create_session(req.vessel_id, COMMUNITY_ID)
    if not session:
        raise HTTPException(404, "Agent not found")
    return {
        "session_id": session.session_id,
        "vessel_id": session.vessel_id,
        "trust_ceiling": session.trust_ceiling,
        "tools": session.active_tools,
    }

@app.post("/session/query")
async def process_query(req: QueryProcess):
    session = session_mgr.get_session(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    result = session_mgr.process_query(session, req.query_text, req.query_embedding)
    return result

@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    ok = session_mgr.close_session(session_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"status": "closed"}


# ============================================================
# Community Endpoints
# ============================================================

@app.get("/community/dashboard")
async def dashboard():
    dash = build_dashboard(graph, COMMUNITY_ID, session_mgr)
    return {
        "community_id": dash.community_id,
        "total_agents": dash.total_agents,
        "total_terrain": dash.total_terrain,
        "active_sessions": dash.active_sessions,
        "visiting_agents": dash.visiting_agents,
        "open_conflicts": dash.open_conflicts,
        "vital_signs": {
            "terrain_additions_30d": dash.vital_signs.terrain_additions_30d,
            "active_agents": dash.vital_signs.active_agents,
            "moral_variance": {str(k): round(v, 4) for k, v in dash.vital_signs.moral_variance.items()},
            "kala_concentration": round(dash.vital_signs.kala_concentration, 4),
        },
    }

@app.get("/community/vitals")
async def vitals():
    signs = compute_vital_signs(graph, COMMUNITY_ID)
    return {
        "terrain_30d": signs.terrain_additions_30d,
        "active_agents": signs.active_agents,
        "open_conflicts": signs.open_conflicts,
        "kala_gini": round(signs.kala_concentration, 4),
        "moral_variance": {str(k): round(v, 4) for k, v in signs.moral_variance.items()},
    }


# ============================================================
# Conflict Endpoints
# ============================================================

@app.get("/conflicts")
async def list_conflicts():
    conflicts = get_open_conflicts(graph, COMMUNITY_ID)
    return {"open_conflicts": [{
        "id": c.id,
        "object_key": c.get("object_key"),
        "contender_count": c.get("contender_count"),
        "status": c.get("status"),
    } for c in conflicts]}

@app.post("/conflicts")
async def create_conflict(req: ConflictCreate):
    contenders = [Contender(**c) for c in req.contenders]
    cs = create_conflict_set(graph, COMMUNITY_ID, req.object_key,
                             contenders, req.contested_fields)
    return {"conflict_id": cs.id, "status": cs.get("status")}

@app.post("/conflicts/resolve")
async def resolve(req: ConflictResolve):
    ok = resolve_conflict(graph, req.conflict_id, req.chosen_mutation_hash,
                          req.resolver_key, req.justification)
    if not ok:
        raise HTTPException(400, "Resolution failed")
    return {"status": "resolved"}


# ============================================================
# Snapshot Endpoints
# ============================================================

@app.post("/snapshot")
async def create_snap(req: SnapshotCreate):
    fmt = getattr(SnapshotFormat, req.format.upper(), SnapshotFormat.FULL)
    snap = create_snapshot(graph, COMMUNITY_ID, req.creator_key, fmt)
    return {
        "snapshot_id": snap.manifest.snapshot_id,
        "format": snap.manifest.format.value,
        "nodes": snap.manifest.node_count,
        "edges": snap.manifest.edge_count,
        "state_root": snap.manifest.state_root,
        "verified": snap.verify(),
    }

@app.get("/graph/stats")
async def graph_stats():
    return {
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "state_hash": graph.compute_state_hash(),
        "node_types": {nt: len(graph.get_nodes_by_type(nt))
                       for nt in set(n.node_type for n in graph._nodes.values())},
    }


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

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

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Graph Zero imports
from graph_zero.graph.backend import PropertyGraph

try:
    from graph_zero.graph.falkordb_backend import FalkorPropertyGraph
    HAS_FALKORDB = True
except ImportError:
    HAS_FALKORDB = False

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

# MCP imports
try:
    from mcp.server.sse import SseServerTransport
    from graph_zero.mcp.server import create_mcp_server
    from starlette.routing import Route, Mount
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


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

    falkor_host = os.getenv("FALKORDB_HOST")
    falkor_port = int(os.getenv("FALKORDB_PORT", "6379"))
    falkor_pass = os.getenv("FALKORDB_PASSWORD", "")
    graph_name = os.getenv("FALKORDB_GRAPH", "graph_zero")

    if falkor_host and HAS_FALKORDB:
        try:
            graph = FalkorPropertyGraph(
                host=falkor_host, port=falkor_port,
                password=falkor_pass, graph_name=graph_name)
            backend_name = f"FalkorDB ({falkor_host}:{falkor_port}/{graph_name})"
        except Exception as e:
            print(f"FalkorDB connection failed: {e}, falling back to in-memory")
            graph = PropertyGraph()
            backend_name = "In-Memory (FalkorDB failed)"
    else:
        graph = PropertyGraph()
        backend_name = "In-Memory"

    # Only bootstrap if graph is empty
    if graph.node_count == 0:
        bootstrap_community(graph, COMMUNITY_ID, COMMUNITY_NAME)
        print(f"Bootstrapped: {COMMUNITY_NAME}")
    else:
        print(f"Graph already populated: {graph.node_count} nodes, {graph.edge_count} edges")

    engine = ExecutionEngine(graph)
    session_mgr = SessionManager(graph, engine)
    replicator = LogReplicator()

    # MCP server setup
    if HAS_MCP:
        mcp_server = create_mcp_server(
            lambda: graph, lambda: engine, lambda: session_mgr,
            COMMUNITY_ID, COMMUNITY_NAME)
        sse = SseServerTransport("/mcp/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await mcp_server.run(
                    streams[0], streams[1], mcp_server.create_initialization_options()
                )

        async def handle_messages(request):
            await sse.handle_post_message(request.scope, request.receive, request._send)

        from starlette.routing import Route, Mount
        app.routes.append(Route("/mcp/sse", endpoint=handle_sse))
        app.routes.append(Route("/mcp/messages/", endpoint=handle_messages, methods=["POST"]))

        print(f"  MCP: enabled at /mcp/sse")
    else:
        print(f"  MCP: disabled (mcp package not installed)")

    print(f"Graph Zero booted: {COMMUNITY_NAME} ({COMMUNITY_ID})")
    print(f"  Backend: {backend_name}")
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
    allow_credentials=False,
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
    graph_name = os.getenv("FALKORDB_GRAPH", "graph_zero")
    backend = f"falkordb/{graph_name}" if (HAS_FALKORDB and isinstance(graph, FalkorPropertyGraph)) else "in-memory"
    return {
        "service": "Graph Zero",
        "version": "0.6.0",
        "community": COMMUNITY_NAME,
        "community_id": COMMUNITY_ID,
        "backend": backend,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "status": "operational",
    }

@app.get("/health")
async def health():
    mcp_status = "enabled" if HAS_MCP else "disabled"
    return {"status": "healthy", "nodes": graph.node_count, "edges": graph.edge_count,
            "mcp": mcp_status}

@app.get("/mcp/info")
async def mcp_info():
    """MCP server discovery endpoint for Agent Zero and other clients."""
    return {
        "name": "graph-zero",
        "version": "0.6.0",
        "description": "Graph-native agent framework with moral geometry",
        "mcp_enabled": HAS_MCP,
        "sse_endpoint": "/mcp/sse",
        "messages_endpoint": "/mcp/messages/",
        "tools": 20,
        "community_id": COMMUNITY_ID,
        "community_name": COMMUNITY_NAME,
        "connection_url": "https://graph-zero-production-2a4d.up.railway.app/mcp/sse",
    }

@app.get("/falkordb/discover")
async def falkordb_discover():
    """Discover all graphs and data in the FalkorDB instance."""
    import redis
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    
    try:
        r = redis.Redis(host=host, port=port, password=pw, decode_responses=True)
        info = r.info("server")
        
        graphs_raw = r.execute_command("GRAPH.LIST")
        graphs = [g if isinstance(g, str) else g.decode() for g in (graphs_raw or [])]
        
        result = {"redis_version": info.get("redis_version"), "graphs": {}}
        
        for gname in graphs:
            ginfo = {"name": gname}
            try:
                nr = r.execute_command("GRAPH.QUERY", gname, "MATCH (n) RETURN count(n)")
                ginfo["nodes"] = nr[1][0][0] if nr and nr[1] else 0
                
                er = r.execute_command("GRAPH.QUERY", gname, "MATCH ()-[rel]->() RETURN count(rel)")
                ginfo["edges"] = er[1][0][0] if er and er[1] else 0
                
                lr = r.execute_command("GRAPH.QUERY", gname, "CALL db.labels()")
                ginfo["labels"] = [row[0] for row in (lr[1] if lr else [])]
                
                rr = r.execute_command("GRAPH.QUERY", gname, "CALL db.relationshipTypes()")
                ginfo["rel_types"] = [row[0] for row in (rr[1] if rr else [])]
                
                # Sample: count per label
                label_counts = {}
                for label in ginfo["labels"][:30]:
                    try:
                        cr = r.execute_command("GRAPH.QUERY", gname, f"MATCH (n:`{label}`) RETURN count(n)")
                        label_counts[label] = cr[1][0][0] if cr and cr[1] else 0
                    except:
                        label_counts[label] = "?"
                ginfo["label_counts"] = label_counts
                
                # Sample: check for embedding properties
                try:
                    emb_check = r.execute_command("GRAPH.QUERY", gname, 
                        "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) LIMIT 1")
                    ginfo["nodes_with_embeddings"] = emb_check[1][0][0] if emb_check and emb_check[1] else 0
                except:
                    ginfo["nodes_with_embeddings"] = "unknown"
                    
            except Exception as e:
                ginfo["error"] = str(e)
            
            result["graphs"][gname] = ginfo
        
        # Also check memory usage
        mem = r.info("memory")
        result["memory_used_mb"] = round(mem.get("used_memory", 0) / 1024 / 1024, 1)
        result["memory_peak_mb"] = round(mem.get("used_memory_peak", 0) / 1024 / 1024, 1)
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/falkordb/query")
async def falkordb_raw_query(request: Request):
    """Run a raw Cypher query against any graph. For schema discovery."""
    body = await request.json()
    graph_name = body.get("graph", "FalkorDB")
    query = body.get("query", "RETURN 1")
    
    import redis
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    
    try:
        r = redis.Redis(host=host, port=port, password=pw, decode_responses=False)
        raw = r.execute_command("GRAPH.QUERY", graph_name, query)
        
        # Parse FalkorDB result format: [headers, rows, stats]
        headers = [h.decode() if isinstance(h, bytes) else str(h) for h in (raw[0] if raw else [])]
        rows = []
        for row in (raw[1] if len(raw) > 1 else []):
            parsed_row = []
            for cell in row:
                if isinstance(cell, bytes):
                    parsed_row.append(cell.decode())
                elif isinstance(cell, list):
                    # Could be a node or relationship
                    parsed_row.append(str(cell)[:500])
                else:
                    parsed_row.append(cell)
            rows.append(parsed_row)
        
        stats = []
        if len(raw) > 2:
            for s in raw[2]:
                stats.append(s.decode() if isinstance(s, bytes) else str(s))
        
        return {"headers": headers, "rows": rows[:100], "row_count": len(rows), "stats": stats}
    except Exception as e:
        return {"error": str(e)}


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

@app.get("/agents")
async def list_agents():
    """List all vessel anchors."""
    anchors = graph.get_nodes_by_type(NT.VESSEL_ANCHOR)
    agents = []
    for a in anchors[:50]:
        agents.append({
            "vessel_id": a.id,
            "name": a.get("name") or a.id,
            "type": a.get("type") or a.get("clone_type") or "agent",
            "role": a.get("role", ""),
        })
    seen = set()
    unique = []
    for ag in agents:
        if ag["vessel_id"] not in seen:
            seen.add(ag["vessel_id"])
            unique.append(ag)
    return {"agents": unique, "total": len(unique)}

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
    
    # Embedding-based search: find nearest terrain nodes by cosine similarity
    if req.query_embedding and not entry_ids:
        entry_ids = find_entry_points(graph, req.query_embedding,
                                      req.threshold, limit=req.limit)
    
    # Try graph traversal first
    results = traverse_terrain(graph, entry_ids, req.max_depth, req.limit)
    
    # If traversal found nothing but we have entry points, return the entry nodes directly
    # (migrated terrain may not have inter-node edges yet)
    if not results and entry_ids:
        for eid in entry_ids[:req.limit]:
            node = graph.get_node(eid)
            if node and node.node_type == "TerrainNode":
                from graph_zero.graph.schema import TraversalResult
                results.append(TraversalResult(
                    node=node,
                    source_text=node.get("source_text", ""),
                    layer=node.get("layer", "community"),
                    authority_weight=float(node.get("authority_weight", 0.5)),
                    depth=0,
                    path_weight=1.0,
                    provenance_types=[node.get("provenance", "UNKNOWN")],
                ))
    
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
            "source_work": r.node.get("source_work", ""),
            "author": r.node.get("author", ""),
        } for r in results],
    }

@app.post("/terrain/search")
async def terrain_search(req: TerrainQuery):
    """Direct embedding similarity search across all terrain.
    
    Unlike /terrain/query which does graph traversal, this does flat
    cosine similarity search — works even without inter-terrain edges.
    Returns top-N most similar terrain nodes.
    """
    if not req.query_embedding:
        raise HTTPException(400, "query_embedding required for search")
    
    # Use FalkorDB direct query for vector similarity
    if HAS_FALKORDB and isinstance(graph, FalkorPropertyGraph):
        from falkordb import FalkorDB as FDB
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        pw = os.getenv("FALKORDB_PASSWORD", "")
        db = FDB(host=host, port=port, password=pw)
        g = db.select_graph(os.getenv("FALKORDB_GRAPH", "graph_zero"))
        
        # Compute cosine similarity in Cypher
        # FalkorDB doesn't have native vector search, so we pull candidates
        # and compute similarity in Python
        emb = req.query_embedding
        limit = min(req.limit, 50)
        
        result = g.query(
            f"MATCH (n:TerrainNode) WHERE n.embedding IS NOT NULL AND size(n.embedding) > 0 "
            f"RETURN n._id, n.source_text, n.source_work, n.author, n.layer, "
            f"n.authority_weight, n.embedding LIMIT 500")
        
        import math
        def cosine_sim(a, b):
            if len(a) != len(b) or not a:
                return 0.0
            dot = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)
        
        scored = []
        for row in result.result_set:
            nid, text, work, author, layer, aw, node_emb = row
            if node_emb and len(node_emb) == len(emb):
                sim = cosine_sim(emb, node_emb)
                if sim >= (req.threshold or 0.0):
                    scored.append({
                        "node_id": nid, "source_text": text or "",
                        "source_work": work or "", "author": author or "",
                        "layer": layer or "community",
                        "authority_weight": float(aw) if aw else 0.5,
                        "similarity": round(sim, 4),
                    })
        
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return {"query_dims": len(emb), "results": scored[:limit]}
    else:
        # In-memory fallback
        entry_ids = find_entry_points(graph, req.query_embedding,
                                      req.threshold, limit=req.limit)
        results = []
        for eid in entry_ids:
            node = graph.get_node(eid)
            if node:
                results.append({
                    "node_id": node.id,
                    "source_text": node.get("source_text", ""),
                    "layer": node.get("layer", "community"),
                    "authority_weight": float(node.get("authority_weight", 0.5)),
                    "similarity": 1.0,
                })
        return {"query_dims": len(req.query_embedding), "results": results}


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
    if HAS_FALKORDB and isinstance(graph, FalkorPropertyGraph):
        # Use efficient Cypher for large graphs
        import redis
        host = os.getenv("FALKORDB_HOST", "localhost")
        port = int(os.getenv("FALKORDB_PORT", "6379"))
        pw = os.getenv("FALKORDB_PASSWORD", "")
        gname = os.getenv("FALKORDB_GRAPH", "graph_zero")
        try:
            r = redis.Redis(host=host, port=port, password=pw, decode_responses=False)
            lr = r.execute_command("GRAPH.QUERY", gname, "CALL db.labels()")
            labels = [row[0].decode() if isinstance(row[0], bytes) else str(row[0]) for row in (lr[1] if lr else [])]
            label_counts = {}
            for label in labels:
                try:
                    cr = r.execute_command("GRAPH.QUERY", gname, f"MATCH (n:`{label}`) RETURN count(n)")
                    label_counts[label] = cr[1][0][0] if cr and cr[1] else 0
                except:
                    label_counts[label] = 0
            return {
                "nodes": graph.node_count,
                "edges": graph.edge_count,
                "state_hash": "live",
                "node_types": {k: v for k, v in label_counts.items() if v > 0},
            }
        except Exception as e:
            pass
    return {
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "state_hash": graph.compute_state_hash() if graph.node_count < 10000 else f"live_{graph.node_count}n_{graph.edge_count}e",
        "node_types": {nt: len(graph.get_nodes_by_type(nt))
                       for nt in set(n.node_type for n in graph._nodes.values())},
    }

@app.get("/dashboard", response_class=None)
async def serve_dashboard():
    """Serve the Graph Zero dashboard."""
    import os
    from fastapi.responses import HTMLResponse
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(html_path):
        html_path = "/app/static/index.html"
    try:
        with open(html_path) as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

@app.post("/falkordb/query")
async def falkordb_raw_query(body: dict):
    """Run a raw Cypher query against any graph. For schema discovery."""
    import redis
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    graph_name = body.get("graph", "FalkorDB")
    query = body.get("query", "RETURN 1")
    
    r = redis.Redis(host=host, port=port, password=pw, decode_responses=False)
    try:
        result = r.execute_command("GRAPH.QUERY", graph_name, query)
        # Parse result - result[0] is header, result[1] is data
        headers = [h.decode() if isinstance(h, bytes) else str(h) for h in (result[0] if result else [])]
        rows = []
        for row in (result[1] if len(result) > 1 else []):
            parsed_row = []
            for cell in row:
                if isinstance(cell, bytes):
                    parsed_row.append(cell.decode())
                elif isinstance(cell, list):
                    # Node or relationship - extract properties
                    parsed_row.append(str(cell)[:500])
                else:
                    parsed_row.append(cell)
            rows.append(parsed_row)
        return {"headers": headers, "rows": rows[:50], "total_rows": len(result[1]) if len(result) > 1 else 0}
    except Exception as e:
        return {"error": str(e)}

@app.get("/falkordb/sample/{graph_name}/{label}")
async def falkordb_sample(graph_name: str, label: str, limit: int = 3):
    """Sample nodes from any graph/label to inspect schema."""
    from falkordb import FalkorDB as FDB
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    try:
        db = FDB(host=host, port=port, password=pw)
        g = db.select_graph(graph_name)
        result = g.query(f"MATCH (n:`{label}`) RETURN n LIMIT {min(limit,5)}")
        samples = []
        for row in result.result_set:
            node = row[0]
            props = {}
            for k, v in node.properties.items():
                if isinstance(v, (list,)) and len(v) > 10:
                    props[k] = f"[vector: {len(v)} dims, first3={v[:3]}]"
                elif isinstance(v, str) and len(v) > 300:
                    props[k] = v[:300] + f"... ({len(v)} total chars)"
                else:
                    props[k] = v
            samples.append({"labels": list(node.labels), "properties": props})
        return {"graph": graph_name, "label": label, "count": len(samples), "samples": samples}
    except Exception as e:
        return {"error": str(e)}

@app.get("/falkordb/schema/{graph_name}/{label}")
async def falkordb_schema(graph_name: str, label: str):
    """Get property keys and types for a label."""
    from falkordb import FalkorDB as FDB
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    try:
        db = FDB(host=host, port=port, password=pw)
        g = db.select_graph(graph_name)
        result = g.query(f"MATCH (n:`{label}`) RETURN n LIMIT 1")
        if not result.result_set:
            return {"graph": graph_name, "label": label, "properties": {}}
        node = result.result_set[0][0]
        schema = {}
        for k, v in node.properties.items():
            if isinstance(v, list):
                schema[k] = f"list[{type(v[0]).__name__ if v else '?'}] len={len(v)}"
            else:
                schema[k] = type(v).__name__
        # Count total
        count_r = g.query(f"MATCH (n:`{label}`) RETURN count(n)")
        total = count_r.result_set[0][0] if count_r.result_set else 0
        return {"graph": graph_name, "label": label, "total": total, "properties": schema}
    except Exception as e:
        return {"error": str(e)}

@app.get("/falkordb/query/{graph_name}")
async def falkordb_query(graph_name: str, q: str, limit: int = 20):
    """Run a read-only Cypher query against any graph."""
    from falkordb import FalkorDB as FDB
    if any(kw in q.upper() for kw in ["DELETE", "SET ", "CREATE", "MERGE", "REMOVE", "DROP"]):
        return {"error": "Read-only queries only"}
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    try:
        db = FDB(host=host, port=port, password=pw)
        g = db.select_graph(graph_name)
        result = g.query(q)
        rows = []
        for row in result.result_set[:limit]:
            r = []
            for val in row:
                if hasattr(val, 'properties'):
                    r.append({"type": "node", "labels": list(val.labels),
                              "id_prop": val.properties.get("id", "?")})
                elif hasattr(val, 'relation'):
                    r.append({"type": "edge", "relation": val.relation})
                else:
                    r.append(val)
            rows.append(r)
        return {"query": q, "rows": len(rows), "results": rows}
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/migrate")
async def migrate_terrain():
    """Migrate TerrainNodes from legacy FalkorDB graph to graph_zero.
    
    1. Wipes bad duplicates from graph_zero
    2. Reads all TerrainNodes from legacy FalkorDB graph
    3. Deduplicates by text content
    4. Maps schema: courage→compassion, text→source_text, individual virtues→array
    5. Writes to graph_zero with proper _id
    """
    from falkordb import FalkorDB as FDB
    import hashlib, time
    
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    
    try:
        db = FDB(host=host, port=port, password=pw)
        legacy = db.select_graph("FalkorDB")
        target = db.select_graph("graph_zero")
        
        log = []
        
        # Step 1: Count existing terrain in graph_zero
        r = target.query("MATCH (n:TerrainNode) RETURN count(n)")
        existing = r.result_set[0][0] if r.result_set else 0
        log.append(f"Existing TerrainNodes in graph_zero: {existing}")
        
        # Step 2: Wipe TerrainNodes that came from botched migration (have terrain_role)
        target.query("MATCH (n:TerrainNode) WHERE n.terrain_role IS NOT NULL DETACH DELETE n")
        r2 = target.query("MATCH (n:TerrainNode) RETURN count(n)")
        remaining = r2.result_set[0][0] if r2.result_set else 0
        log.append(f"After wiping migrated dupes: {remaining} TerrainNodes remain (our originals)")
        
        # Step 3: Read ALL TerrainNodes from legacy in batches
        batch_size = 100
        offset = 0
        all_legacy = []
        while True:
            r = legacy.query(f"MATCH (n:TerrainNode) RETURN n SKIP {offset} LIMIT {batch_size}")
            if not r.result_set:
                break
            for row in r.result_set:
                node = row[0]
                all_legacy.append(node.properties)
            if len(r.result_set) < batch_size:
                break
            offset += batch_size
        
        log.append(f"Read {len(all_legacy)} TerrainNodes from legacy FalkorDB graph")
        
        # Step 4: Deduplicate by text content
        seen_texts = {}
        unique_nodes = []
        dupes = 0
        for props in all_legacy:
            text = props.get("text", "")
            if not text:
                continue
            text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            if text_hash in seen_texts:
                dupes += 1
                continue
            seen_texts[text_hash] = True
            unique_nodes.append(props)
        
        log.append(f"Deduplicated: {len(unique_nodes)} unique, {dupes} duplicates removed")
        
        # Step 5: Migrate with schema mapping
        # Legacy virtues: unity, justice, truthfulness, love, detachment, humility, service, courage, wisdom
        # Graph Zero:     unity, justice, truthfulness, love, detachment, humility, compassion, wisdom, service
        # Mapping: courage→compassion, reorder: service moves to end, wisdom stays at 8
        
        VIRTUE_MAP = [
            "virtue_unity",        # 0 → 0
            "virtue_justice",      # 1 → 1  
            "virtue_truthfulness", # 2 → 2
            "virtue_love",         # 3 → 3
            "virtue_detachment",   # 4 → 4
            "virtue_humility",     # 5 → 5
            "virtue_courage",      # courage → compassion (index 6)
            "virtue_wisdom",       # 7 → wisdom (index 7)
            "virtue_service",      # 6 → service (index 8)
        ]
        
        MOMENTUM_MAP = [
            "momentum_unity",
            "momentum_justice",
            "momentum_truthfulness",
            "momentum_love",
            "momentum_detachment",
            "momentum_humility",
            "momentum_courage",
            "momentum_wisdom",
            "momentum_service",
        ]
        
        migrated = 0
        errors = 0
        
        for props in unique_nodes:
            try:
                text = props.get("text", "")
                source_work = props.get("source_work", "")
                author = props.get("author", "")
                terrain_role = props.get("terrain_role", "community")
                chunk_id = props.get("chunk_id", props.get("id", ""))
                
                # Generate stable _id from text hash
                node_id = f"legacy_{hashlib.sha256(text.encode()).hexdigest()[:12]}"
                
                # Map terrain_role to layer
                layer_map = {"bedrock": "bedrock", "living": "community", "community": "community", "earned": "earned", "personal": "personal"}
                layer = layer_map.get(terrain_role, "community")
                
                # Extract virtue scores in Graph Zero order
                virtue_scores = []
                for vkey in VIRTUE_MAP:
                    virtue_scores.append(float(props.get(vkey, 0.5)))
                
                # Extract momentum
                momentum = []
                for mkey in MOMENTUM_MAP:
                    momentum.append(float(props.get(mkey, 0.0)))
                
                # Embedding
                embedding = props.get("embedding", [])
                
                kala = float(props.get("kala_accumulated", 0.0))
                traversals = int(props.get("traversal_count", 0))
                
                # Provenance
                if terrain_role == "bedrock":
                    prov = "BEDROCK"
                elif author:
                    prov = "CROSS_VERIFIED"
                else:
                    prov = "WITNESS"
                
                # Build SET clause manually for complex data
                # Can't use parameterized queries with lists in FalkorDB easily
                # So we MERGE by _id and set all props
                
                # Escape text for Cypher
                safe_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", "")
                safe_source = source_work.replace("\\", "\\\\").replace("'", "\\'")
                safe_author = author.replace("\\", "\\\\").replace("'", "\\'")
                
                vs_str = "[" + ",".join(str(v) for v in virtue_scores) + "]"
                mom_str = "[" + ",".join(str(m) for m in momentum) + "]"
                
                q = (
                    f"MERGE (n:TerrainNode {{_id: '{node_id}'}}) "
                    f"SET n.source_text = '{safe_text}', "
                    f"n.text = '{safe_text}', "
                    f"n.source_work = '{safe_source}', "
                    f"n.author = '{safe_author}', "
                    f"n.layer = '{layer}', "
                    f"n.terrain_role = '{terrain_role}', "
                    f"n.provenance_type = '{prov}', "
                    f"n.virtue_scores = {vs_str}, "
                    f"n.momentum = {mom_str}, "
                    f"n.kala_accumulated = {kala}, "
                    f"n.traversal_count = {traversals}, "
                    f"n.authority_weight = 0.5, "
                    f"n.chunk_id = '{chunk_id}', "
                    f"n.created_at = {int(time.time() * 1000)}"
                )
                
                target.query(q)
                
                # Embedding stored separately (too large for inline Cypher)
                if embedding and len(embedding) > 0:
                    # Store embedding dimensions as a property
                    target.query(f"MATCH (n:TerrainNode {{_id: '{node_id}'}}) SET n.embedding_dims = {len(embedding)}")
                    # FalkorDB can handle list properties but large ones need special handling
                    # For now store the embedding via the list syntax
                    if len(embedding) <= 2000:
                        emb_str = "[" + ",".join(str(e) for e in embedding) + "]"
                        target.query(f"MATCH (n:TerrainNode {{_id: '{node_id}'}}) SET n.embedding = {emb_str}")
                
                migrated += 1
                if migrated % 100 == 0:
                    log.append(f"  Migrated {migrated}/{len(unique_nodes)}...")
                    
            except Exception as e:
                errors += 1
                if errors <= 5:
                    log.append(f"  Error on node: {str(e)[:100]}")
        
        # Step 6: Final count
        r_final = target.query("MATCH (n:TerrainNode) RETURN count(n)")
        final_count = r_final.result_set[0][0] if r_final.result_set else 0
        
        log.append(f"Migration complete: {migrated} migrated, {errors} errors, {final_count} total TerrainNodes")
        
        return {
            "status": "complete",
            "legacy_total": len(all_legacy),
            "unique": len(unique_nodes),
            "duplicates_removed": dupes,
            "migrated": migrated,
            "errors": errors,
            "final_terrain_count": final_count,
            "log": log
        }
        
    except Exception as e:
        return {"error": str(e), "log": log if 'log' in dir() else []}

@app.post("/migrate/terrain")
async def migrate_terrain(batch_size: int = 100, offset: int = 0, dry_run: bool = False):
    """Migration already complete — 8740 nodes present. Use /migrate/vector-index instead."""
    return {
        "status": "already_migrated",
        "terrain_nodes": 8744,
        "message": "TerrainNodes already present in graph_zero. Use POST /migrate/vector-index to create vector search index."
    }

@app.post("/migrate/vector-index")
async def create_vector_index():
    """Create a vector index on TerrainNode embeddings for fast similarity search."""
    from falkordb import FalkorDB as FDB
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    try:
        db = FDB(host=host, port=port, password=pw)
        gz = db.select_graph("graph_zero")
        
        # Create vector index: 1536 dims (OpenAI ada-002), cosine similarity
        gz.query("""
            CREATE VECTOR INDEX FOR (n:TerrainNode) ON (n.embedding) 
            OPTIONS {dim: 1536, similarityFunction: 'cosine'}
        """)
        
        # Verify
        indexes = gz.query("CALL db.indexes()")
        vector_found = False
        for row in indexes.result_set:
            if 'VECTOR' in str(row):
                vector_found = True
                break
        
        return {
            "status": "created",
            "index": "TerrainNode.embedding",
            "dimensions": 1536,
            "similarity": "cosine",
            "verified": vector_found
        }
    except Exception as e:
        err = str(e)
        if "already" in err.lower() or "exist" in err.lower():
            return {"status": "already_exists", "message": err}
        return {"error": err}

@app.post("/terrain/vector-search")
async def terrain_vector_search(query_embedding: list[float] = None,
                                 query_text: str = None, limit: int = 10):
    """Search terrain using FalkorDB native vector index. Fast."""
    from falkordb import FalkorDB as FDB
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    
    if not query_embedding:
        return {"error": "query_embedding required (1536 floats)"}
    
    try:
        db = FDB(host=host, port=port, password=pw)
        gz = db.select_graph("graph_zero")
        
        # Use FalkorDB vector query
        emb_str = ",".join(str(f) for f in query_embedding)
        result = gz.query(f"""
            CALL db.idx.vector.queryNodes('TerrainNode', 'embedding', {limit}, 
                vecf32([{emb_str}]))
            YIELD node, score
            RETURN node._id AS id, 
                   node.text AS text, 
                   node.source_text AS source_text,
                   node.source_work AS source_work,
                   node.author AS author,
                   node.layer AS layer,
                   node.terrain_role AS terrain_role,
                   score
            ORDER BY score ASC
        """)
        
        results = []
        for row in result.result_set:
            results.append({
                "node_id": row[0],
                "text": row[1] or row[2] or "",
                "source_work": row[3] or "",
                "author": row[4] or "",
                "layer": row[5] or row[6] or "community",
                "score": row[7],
            })
        
        return {"query_dims": len(query_embedding), "results": results}
    except Exception as e:
        return {"error": str(e)}

@app.post("/falkordb/migrate")
async def migrate_legacy_terrain(batch_size: int = 100, dry_run: bool = False):
    """Migrate TerrainNodes from legacy FalkorDB graph into graph_zero.
    
    Schema mapping:
      legacy.id -> graph_zero._id  
      legacy.text -> source_text property
      legacy.source_work -> source_work property
      legacy.author -> author property
      legacy.terrain_role -> layer (bedrock->bedrock, living->community)
      legacy.virtue_* -> stored as virtue position properties
      legacy.momentum_* -> stored as momentum properties
      legacy.kala_accumulated -> kala_accumulated property
      legacy.traversal_count -> traversal_count property
      legacy.embedding (1536d) -> embedding property
      legacy.courage -> mapped to compassion index
    
    Provenance: BEDROCK for bedrock, COMMUNITY_CONSENSUS for living terrain.
    """
    from falkordb import FalkorDB as FDB
    import time
    
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    
    db = FDB(host=host, port=port, password=pw)
    legacy = db.select_graph("FalkorDB")
    target = db.select_graph("graph_zero")
    
    VIRTUE_MAP = {
        "virtue_unity": 0, "virtue_justice": 1, "virtue_truthfulness": 2,
        "virtue_love": 3, "virtue_detachment": 4, "virtue_humility": 5,
        "virtue_courage": 6,  # maps to compassion slot
        "virtue_wisdom": 7, "virtue_service": 8,
    }
    MOMENTUM_MAP = {
        "momentum_unity": 0, "momentum_justice": 1, "momentum_truthfulness": 2,
        "momentum_love": 3, "momentum_detachment": 4, "momentum_humility": 5,
        "momentum_courage": 6, "momentum_wisdom": 7, "momentum_service": 8,
    }
    LAYER_MAP = {"bedrock": "bedrock", "living": "community"}
    PROV_MAP = {"bedrock": "BEDROCK", "living": "COMMUNITY_CONSENSUS"}
    
    # Count total
    count_r = legacy.query("MATCH (n:TerrainNode) RETURN count(n)")
    total = count_r.result_set[0][0]
    
    if dry_run:
        return {"status": "dry_run", "total_legacy_nodes": total, "batch_size": batch_size}
    
    migrated = 0
    skipped = 0
    errors = []
    start = time.time()
    offset = 0
    
    while offset < total:
        # Batch read from legacy
        batch_r = legacy.query(
            f"MATCH (n:TerrainNode) RETURN n ORDER BY n.id SKIP {offset} LIMIT {batch_size}")
        
        if not batch_r.result_set:
            break
        
        for row in batch_r.result_set:
            node = row[0]
            props = node.properties
            node_id = props.get("id", f"legacy_{offset}_{migrated}")
            
            # Check if already migrated
            exists_r = target.query(
                f"MATCH (n:TerrainNode {{_id: '{node_id}'}}) RETURN count(n)")
            if exists_r.result_set and exists_r.result_set[0][0] > 0:
                skipped += 1
                continue
            
            # Build virtue position [9]
            virtues = [0.5] * 9
            for vk, vi in VIRTUE_MAP.items():
                val = props.get(vk)
                if val is not None:
                    virtues[vi] = float(val)
            
            # Build momentum [9]
            momenta = [0.0] * 9
            for mk, mi in MOMENTUM_MAP.items():
                val = props.get(mk)
                if val is not None:
                    momenta[mi] = float(val)
            
            text = props.get("text", "")
            source_work = props.get("source_work", "")
            author = props.get("author", "")
            terrain_role = props.get("terrain_role", "living")
            layer = LAYER_MAP.get(terrain_role, "community")
            prov = PROV_MAP.get(terrain_role, "COMMUNITY_CONSENSUS")
            kala = float(props.get("kala_accumulated", 0))
            traversals = int(props.get("traversal_count", 0))
            chunk_id = props.get("chunk_id", "")
            embedding = props.get("embedding", [])
            
            # Escape text for Cypher
            safe_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
            safe_work = source_work.replace("\\", "\\\\").replace("'", "\\'")
            safe_author = author.replace("\\", "\\\\").replace("'", "\\'")
            safe_chunk = chunk_id.replace("\\", "\\\\").replace("'", "\\'")
            
            # Build the Cypher CREATE
            try:
                # Create node with all properties
                virtue_str = str(virtues)
                momentum_str = str(momenta)
                
                q = (f"CREATE (n:TerrainNode {{"
                     f"_id: '{node_id}', "
                     f"source_text: '{safe_text}', "
                     f"source_work: '{safe_work}', "
                     f"author: '{safe_author}', "
                     f"chunk_id: '{safe_chunk}', "
                     f"layer: '{layer}', "
                     f"provenance: '{prov}', "
                     f"virtue_position: {virtue_str}, "
                     f"momentum: {momentum_str}, "
                     f"kala_accumulated: {kala}, "
                     f"traversal_count: {traversals}, "
                     f"authority_weight: 1.0"
                     f"}})")
                
                target.query(q)
                
                # Store embedding separately (large vector)
                if embedding and len(embedding) > 0:
                    # FalkorDB can store lists as properties
                    emb_str = str(embedding)
                    target.query(
                        f"MATCH (n:TerrainNode {{_id: '{node_id}'}}) "
                        f"SET n.embedding = {emb_str}")
                
                # Connect to community node
                target.query(
                    f"MATCH (n:TerrainNode {{_id: '{node_id}'}}), "
                    f"(c:Community {{_id: 'lower_puna'}}) "
                    f"CREATE (n)-[:MEMBER_OF {{provenance: '{prov}'}}]->(c)")
                
                migrated += 1
                
            except Exception as e:
                errors.append({"node_id": node_id, "error": str(e)[:200]})
                if len(errors) > 50:
                    break
        
        offset += batch_size
        
        # Progress check
        if migrated % 500 == 0 and migrated > 0:
            elapsed = time.time() - start
            rate = migrated / elapsed
            print(f"  Migration: {migrated}/{total} ({rate:.0f}/s), skipped {skipped}, errors {len(errors)}")
    
    elapsed = time.time() - start
    
    # Final count
    final_r = target.query("MATCH (n:TerrainNode) RETURN count(n)")
    final_count = final_r.result_set[0][0] if final_r.result_set else 0
    
    return {
        "status": "complete",
        "total_legacy": total,
        "migrated": migrated,
        "skipped": skipped,
        "errors": len(errors),
        "error_samples": errors[:10],
        "elapsed_seconds": round(elapsed, 1),
        "graph_zero_terrain_nodes": final_count,
        "rate_per_second": round(migrated / max(elapsed, 0.1), 1),
    }

@app.post("/admin/migrate")
async def run_migration(dry_run: bool = False):
    """Migrate legacy FalkorDB graph data into graph_zero."""
    from falkordb import FalkorDB as FDB
    from migrate import migrate_terrain, migrate_agents

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")

    try:
        db = FDB(host=host, port=port, password=pw)
        legacy = db.select_graph("FalkorDB")
        gz = db.select_graph("graph_zero")

        results = {"dry_run": dry_run}

        if dry_run:
            # Just count what would be migrated
            t_count = legacy.query("MATCH (n:TerrainNode) RETURN count(n)").result_set[0][0]
            a_count = legacy.query("MATCH (n:Agent) RETURN count(n)").result_set[0][0]
            gz_t = gz.query("MATCH (n:TerrainNode) RETURN count(n)").result_set[0][0]
            gz_a = gz.query("MATCH (n:VesselAnchor) RETURN count(n)").result_set[0][0]
            results["legacy_terrain"] = t_count
            results["legacy_agents"] = a_count
            results["gz_terrain_existing"] = gz_t
            results["gz_vessels_existing"] = gz_a
            results["terrain_to_migrate"] = t_count  # minus already migrated
            results["agents_to_migrate"] = a_count
            return results

        import time
        t0 = time.time()

        print("Starting terrain migration...")
        terrain_stats = migrate_terrain(legacy, gz, batch_size=50)
        results["terrain"] = terrain_stats

        print("Starting agent migration...")
        agent_stats = migrate_agents(legacy, gz)
        results["agents"] = agent_stats

        results["duration_seconds"] = round(time.time() - t0, 1)

        # Final counts
        gz_nodes = gz.query("MATCH (n) RETURN count(n)").result_set[0][0]
        gz_edges = gz.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
        results["graph_zero_final"] = {"nodes": gz_nodes, "edges": gz_edges}

        return results
    except Exception as e:
        return {"error": str(e)}

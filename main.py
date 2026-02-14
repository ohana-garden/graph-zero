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

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
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
# Text-based Terrain Search (embeds query via Voyage AI)
# ============================================================

class TerrainAsk(BaseModel):
    text: str
    top_k: int = 5

@app.post("/terrain/ask")
async def terrain_ask(req: TerrainAsk):
    """Search terrain by text. Embeds query via Voyage AI, finds nearest nodes via vector index."""
    import httpx as hx
    import math
    
    voyage_key = os.getenv("VOYAGE_API_KEY", "")
    if not voyage_key:
        raise HTTPException(500, "VOYAGE_API_KEY not set")
    
    # Embed the query
    async with hx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {voyage_key}", "Content-Type": "application/json"},
            json={"input": [req.text], "model": "voyage-3.5", "input_type": "query"}
        )
        if resp.status_code != 200:
            raise HTTPException(502, f"Voyage API error: {resp.text[:200]}")
        emb_data = resp.json()
        query_emb = emb_data["data"][0]["embedding"]
    
    # Use FalkorDB vector index search
    from falkordb import FalkorDB as FDB
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    db = FDB(host=host, port=port, password=pw)
    g = db.select_graph(os.getenv("FALKORDB_GRAPH", "graph_zero"))
    
    limit = min(req.top_k, 20)
    
    # FalkorDB vector index query
    emb_str = "[" + ",".join(str(v) for v in query_emb) + "]"
    try:
        result = g.query(
            f"CALL db.idx.vector.queryNodes('TerrainNode', 'embedding', {limit}, "
            f"vecf32({emb_str})) "
            f"YIELD node, score "
            f"RETURN node._id AS id, node.source_text AS text, node.source_work AS work, "
            f"node.layer AS layer, node.virtue_scores AS virtues, score "
            f"ORDER BY score ASC"
        )
    except Exception as vec_err:
        return {"query": req.text, "error": f"Vector search failed: {str(vec_err)[:200]}", "results": []}
    
    results = []
    for row in result.result_set:
        node_id, text, work, layer, virtues, score = row
        results.append({
            "id": node_id,
            "score": round(1.0 - float(score), 4) if score is not None else 0,
            "text": text[:300] if text else "",
            "source_work": work,
            "layer": layer,
            "virtue_scores": virtues
        })
    
    return {"query": req.text, "results": results}


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
# Admin Endpoints
# ============================================================

@app.post("/admin/cleanup")
async def admin_cleanup():
    """Purge legacy duplicate TerrainNodes and reset stalled ingest job."""

@app.delete("/admin/graph/{graph_name}")
async def admin_drop_graph(graph_name: str):
    """Drop an entire graph from FalkorDB. Cannot drop graph_zero."""
    if graph_name == "graph_zero":
        return {"error": "Cannot drop the active graph"}
    import redis
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    r = redis.Redis(host=host, port=port, password=pw)
    try:
        r.execute_command("GRAPH.DELETE", graph_name)
        return {"dropped": graph_name}
    except Exception as e:
        return {"error": str(e)}
    from falkordb import FalkorDB as FDB
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    
    log = []
    try:
        db = FDB(host=host, port=port, password=pw)
        g = db.select_graph("graph_zero")
        
        # Count before
        r = g.query("MATCH (n:TerrainNode) RETURN count(n)")
        before = r.result_set[0][0] if r.result_set else 0
        log.append(f"Before cleanup: {before} TerrainNodes")
        
        # Delete legacy dupes: nodes with terrain_role but no _id starting with 'gz_'
        # These are from the botched migration - source_work = 'from a Tablet...'
        r_legacy = g.query("MATCH (n:TerrainNode) WHERE n.source_work = 'from a Tablet - translated from the Persian' RETURN count(n)")
        legacy_count = r_legacy.result_set[0][0] if r_legacy.result_set else 0
        
        if legacy_count > 100:
            # Delete in batches to avoid timeout
            deleted = 0
            for _ in range(100):
                g.query("MATCH (n:TerrainNode) WHERE n.source_work = 'from a Tablet - translated from the Persian' WITH n LIMIT 500 DETACH DELETE n")
                r_check = g.query("MATCH (n:TerrainNode) WHERE n.source_work = 'from a Tablet - translated from the Persian' RETURN count(n)")
                remaining = r_check.result_set[0][0] if r_check.result_set else 0
                deleted = legacy_count - remaining
                if remaining == 0:
                    break
            log.append(f"Deleted {deleted} legacy duplicates ('from a Tablet...')")
        elif legacy_count > 0:
            g.query("MATCH (n:TerrainNode) WHERE n.source_work = 'from a Tablet - translated from the Persian' DETACH DELETE n")
            log.append(f"Deleted {legacy_count} legacy duplicates")
        else:
            log.append("No legacy duplicates found")
        
        # Also clean other known legacy stragglers (old schema nodes without proper _id)
        old_schemas = [
            "MATCH (n:TerrainNode) WHERE n._id STARTS WITH 'legacy_' DETACH DELETE n",
        ]
        for q in old_schemas:
            try:
                g.query(q)
            except:
                pass
        
        # Count after
        r2 = g.query("MATCH (n:TerrainNode) RETURN count(n)")
        after = r2.result_set[0][0] if r2.result_set else 0
        log.append(f"After cleanup: {after} TerrainNodes")
        log.append(f"Removed: {before - after} total")
        
        # Reset stalled ingest flag
        app._ingest_running = False
        if hasattr(app, '_ingest_result'):
            if app._ingest_result.get("status") == "running":
                app._ingest_result["status"] = "reset"
        log.append("Reset ingest job flag")
        
        return {"before": before, "after": after, "removed": before - after, "log": log}
    except Exception as e:
        return {"error": str(e), "log": log}


@app.get("/admin/ingest/sources")
async def list_ingest_sources(use_fallback: bool = False):
    """List available sources for ingestion."""
    from ingest import SOURCES, FALLBACK_SOURCES
    sources = FALLBACK_SOURCES if use_fallback else SOURCES
    return {
        "count": len(sources),
        "sources": [
            {"index": i, "work": s[0], "author": s[1], "role": s[2]}
            for i, s in enumerate(sources)
        ]
    }


@app.post("/admin/ingest")
async def ingest_terrain_endpoint(
    source_index: int = -1,
    embed_model: str = "voyage-3.5",
    embed_dims: int = 1024,
    score_model: str = "llama-3.1-8b-instant",
    use_fallback: bool = False,
):
    """
    Ingest Bahá'í sacred texts as bedrock terrain.
    Runs in background thread. Check /admin/ingest/status for progress.
    
    - source_index=-1: ingest ALL sources
    - source_index=0..N: ingest a specific source
    - use_fallback=true: use bahai-library.com instead of bahai.org
    """
    from ingest import SOURCES, FALLBACK_SOURCES
    import threading
    
    voyage_key = os.getenv("VOYAGE_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if not voyage_key:
        return {"error": "VOYAGE_API_KEY not set"}
    if not groq_key:
        return {"error": "GROQ_API_KEY not set"}
    
    if getattr(app, '_ingest_running', False):
        return {"status": "already_running", "check": "/admin/ingest/status"}
    
    sources = FALLBACK_SOURCES if use_fallback else SOURCES
    
    if source_index >= 0 and source_index >= len(sources):
        return {"error": f"source_index {source_index} out of range (max {len(sources)-1})"}
    
    app._ingest_running = True
    app._ingest_result = {
        "status": "running",
        "started": time.time(),
        "source_index": source_index,
        "progress": [],
    }
    
    def _run():
        import asyncio
        from ingest import ingest_all, ingest_source
        from falkordb import FalkorDB as FDB
        
        try:
            host = os.getenv("FALKORDB_HOST", "localhost")
            port = int(os.getenv("FALKORDB_PORT", "6379"))
            pw = os.getenv("FALKORDB_PASSWORD", "")
            db = FDB(host=host, port=port, password=pw)
            graph = db.select_graph("graph_zero")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if source_index >= 0:
                src = sources[source_index]
                result = loop.run_until_complete(ingest_source(
                    src[0], src[1], src[2], src[3],
                    graph, voyage_key, groq_key,
                    embed_model, embed_dims, score_model
                ))
                app._ingest_result.update(result)
            else:
                result = loop.run_until_complete(ingest_all(
                    graph, voyage_key, groq_key, sources,
                    embed_model, embed_dims, score_model
                ))
                app._ingest_result.update(result)
            
            loop.close()
            app._ingest_result["status"] = "complete"
            app._ingest_result["duration_sec"] = round(time.time() - app._ingest_result["started"], 1)
            
        except Exception as e:
            import traceback
            app._ingest_result["status"] = "error"
            app._ingest_result["error"] = str(e)
            app._ingest_result["traceback"] = traceback.format_exc()
        finally:
            app._ingest_running = False
    
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    
    target = sources[source_index][0] if source_index >= 0 else f"all {len(sources)} sources"
    return {"status": "started", "target": target, "check": "/admin/ingest/status"}


@app.get("/admin/ingest/status")
async def ingest_status():
    """Current ingest job status + terrain statistics."""
    from falkordb import FalkorDB as FDB
    result = {}
    
    # Background job status
    if hasattr(app, '_ingest_result'):
        r = app._ingest_result.copy()
        r["running"] = getattr(app, '_ingest_running', False)
        if r.get("started"):
            r["elapsed_sec"] = round(time.time() - r["started"], 1)
        r.pop("results", None)
        result["job"] = r
    else:
        result["job"] = {"status": "never_run"}
    
    # Terrain stats
    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    pw = os.getenv("FALKORDB_PASSWORD", "")
    try:
        db = FDB(host=host, port=port, password=pw)
        g = db.select_graph("graph_zero")
        total = g.query("MATCH (n:TerrainNode) RETURN count(n)").result_set[0][0]
        by_source = g.query(
            "MATCH (n:TerrainNode) WHERE n.source_work IS NOT NULL "
            "RETURN n.source_work AS work, count(n) AS c ORDER BY c DESC"
        ).result_set
        with_emb = g.query(
            "MATCH (n:TerrainNode) WHERE n.embedding IS NOT NULL RETURN count(n)"
        ).result_set[0][0]
        result["terrain"] = {
            "total": total,
            "with_embeddings": with_emb,
            "by_source": {r[0]: r[1] for r in by_source},
        }
    except Exception as e:
        result["terrain"] = {"error": str(e)}
    return result


# ============================================================
# Ingest Pipeline (background tasks)
# ============================================================

# In-memory job status tracker
_ingest_jobs: dict = {}

@app.post("/admin/ingest/{work_id}")
async def admin_ingest_work(work_id: str, skip_virtues: bool = False,
                            background_tasks: BackgroundTasks = None):
    """Ingest a single Bahá'í sacred text. Runs in background, poll /admin/ingest-status/{job_id}."""
    from graph_zero.ingest.pipeline import WORKS
    
    if work_id == "list":
        return {"available_works": {k: {"title": v["title"], "author": v["author"]} 
                for k, v in WORKS.items()}}
    
    if work_id not in WORKS:
        return {"error": f"Unknown work: {work_id}", "available": list(WORKS.keys())}
    
    voyage_key = os.getenv("VOYAGE_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if not voyage_key:
        return {"error": "VOYAGE_API_KEY not set"}
    
    import uuid, asyncio
    job_id = str(uuid.uuid4())[:8]
    _ingest_jobs[job_id] = {"status": "started", "work_id": work_id, "started": time.time()}
    
    async def _run():
        from graph_zero.ingest.pipeline import ingest_work
        try:
            result = await ingest_work(work_id, graph, voyage_key, groq_key, skip_virtues)
            _ingest_jobs[job_id] = result
        except Exception as e:
            _ingest_jobs[job_id] = {"status": "error", "error": str(e)}
    
    asyncio.ensure_future(_run())
    return {"job_id": job_id, "status": "started", "work_id": work_id,
            "poll": f"/admin/ingest-status/{job_id}"}

@app.get("/admin/ingest-status/{job_id}")
async def admin_ingest_status(job_id: str):
    """Poll ingest job status."""
    if job_id not in _ingest_jobs:
        return {"error": "Unknown job_id"}
    return _ingest_jobs[job_id]

@app.post("/admin/ingest-all")
async def admin_ingest_all(skip_virtues: bool = False):
    """Ingest all Bahá'í sacred texts in background."""
    from graph_zero.ingest.pipeline import WORKS
    
    voyage_key = os.getenv("VOYAGE_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    
    if not voyage_key:
        return {"error": "VOYAGE_API_KEY not set"}
    
    import uuid, asyncio
    job_id = str(uuid.uuid4())[:8]
    _ingest_jobs[job_id] = {"status": "started", "works": list(WORKS.keys()),
                            "started": time.time(), "progress": {}}
    
    async def _run_all():
        from graph_zero.ingest.pipeline import ingest_work, WORKS
        total_stored = 0
        for wid in WORKS:
            _ingest_jobs[job_id]["current"] = wid
            try:
                result = await ingest_work(wid, graph, voyage_key, groq_key, skip_virtues)
                _ingest_jobs[job_id]["progress"][wid] = {
                    "status": result.get("status", "unknown"),
                    "stored": result.get("stored", 0),
                    "chunks": result.get("chunks", 0),
                }
                total_stored += result.get("stored", 0)
            except Exception as e:
                _ingest_jobs[job_id]["progress"][wid] = {"status": "error", "error": str(e)}
        
        _ingest_jobs[job_id]["status"] = "complete"
        _ingest_jobs[job_id]["total_stored"] = total_stored
        _ingest_jobs[job_id]["elapsed"] = round(time.time() - _ingest_jobs[job_id]["started"], 1)
    
    asyncio.ensure_future(_run_all())
    return {"job_id": job_id, "status": "started", "works": len(WORKS),
            "poll": f"/admin/ingest-status/{job_id}"}


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


"""
Graph Zero MCP Server

Exposes Graph Zero as an MCP server so Agent Zero and other
MCP-compatible clients can interact with it natively.

Tools exposed:
  Agents:    create_agent, get_agent, get_trust, record_interaction
  Terrain:   add_terrain, connect_terrain, query_terrain
  Memory:    ingest_episode, extract_fact, retrieve_memories
  Moral:     score_action, project_position
  Community: dashboard, vitals, graph_stats
  Session:   create_session, process_query, close_session
  Snapshot:  create_snapshot
"""

import json
from typing import Any
from mcp.server import Server
from mcp.types import Tool, TextContent


def create_mcp_server(get_graph, get_engine, get_session_mgr,
                      community_id: str, community_name: str):
    """Create an MCP server backed by Graph Zero.

    get_graph, get_engine, get_session_mgr are callables that return
    the current graph/engine/session_mgr (they may not exist at import time).
    """
    server = Server("graph-zero")

    # --------------------------------------------------------
    # Tool definitions
    # --------------------------------------------------------

    TOOLS = [
        # Agents
        Tool(name="create_agent", description="Create a new agent in the community",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string", "description": "Unique agent ID"},
                 "name": {"type": "string", "description": "Display name"},
                 "agent_type": {"type": "string", "enum": ["human", "ai", "hybrid"], "default": "human"},
             }, "required": ["vessel_id", "name"]}),

        Tool(name="get_agent", description="Get an agent's full constellation (position, trust, tools, memories)",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string"},
             }, "required": ["vessel_id"]}),

        Tool(name="get_trust", description="Get an agent's trust ceiling and interaction diversity",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string"},
             }, "required": ["vessel_id"]}),

        Tool(name="record_interaction", description="Record an interaction between two agents",
             inputSchema={"type": "object", "properties": {
                 "agent_a": {"type": "string"},
                 "agent_b": {"type": "string"},
                 "interaction_type": {"type": "string", "description": "e.g. collaborated, helped, taught, traded"},
                 "context": {"type": "string", "default": ""},
             }, "required": ["agent_a", "agent_b", "interaction_type"]}),

        # Terrain
        Tool(name="add_terrain", description="Add a knowledge node to the community terrain",
             inputSchema={"type": "object", "properties": {
                 "node_id": {"type": "string", "description": "Unique terrain node ID"},
                 "source_text": {"type": "string", "description": "The knowledge content"},
                 "layer": {"type": "string", "enum": ["bedrock", "community", "earned", "personal"], "default": "community"},
                 "provenance_type": {"type": "string", "enum": ["BEDROCK", "WITNESS", "CROSS_VERIFIED", "EMPIRICAL", "COMMUNITY_CONSENSUS"], "default": "WITNESS"},
                 "embedding": {"type": "array", "items": {"type": "number"}, "description": "Optional embedding vector"},
             }, "required": ["node_id", "source_text"]}),

        Tool(name="connect_terrain", description="Connect two terrain nodes",
             inputSchema={"type": "object", "properties": {
                 "source_id": {"type": "string"},
                 "target_id": {"type": "string"},
                 "weight": {"type": "number", "default": 1.0},
                 "provenance_type": {"type": "string", "default": "WITNESS"},
             }, "required": ["source_id", "target_id"]}),

        Tool(name="query_terrain", description="Search terrain by embedding similarity or entry points",
             inputSchema={"type": "object", "properties": {
                 "query_embedding": {"type": "array", "items": {"type": "number"}, "description": "Embedding to search by"},
                 "entry_point_ids": {"type": "array", "items": {"type": "string"}, "description": "Specific terrain node IDs to start from"},
                 "max_depth": {"type": "integer", "default": 5},
                 "limit": {"type": "integer", "default": 10},
             }}),

        # Memory
        Tool(name="ingest_episode", description="Store an episodic memory for an agent",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string"},
                 "participants": {"type": "array", "items": {"type": "string"}},
                 "interaction_type": {"type": "string"},
                 "content": {"type": "string"},
                 "summary": {"type": "string"},
                 "emotional_valence": {"type": "number", "default": 0.0, "description": "-1 to 1"},
                 "emotional_arousal": {"type": "number", "default": 0.0, "description": "0 to 1"},
                 "embedding": {"type": "array", "items": {"type": "number"}},
             }, "required": ["vessel_id", "participants", "interaction_type", "content", "summary"]}),

        Tool(name="extract_fact", description="Extract a semantic fact from episodes",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string"},
                 "subject": {"type": "string"},
                 "predicate": {"type": "string"},
                 "object_value": {"type": "string"},
                 "confidence": {"type": "number", "default": 0.8},
                 "embedding": {"type": "array", "items": {"type": "number"}},
             }, "required": ["vessel_id", "subject", "predicate", "object_value"]}),

        Tool(name="retrieve_memories", description="Retrieve memories relevant to a query",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string"},
                 "query_embedding": {"type": "array", "items": {"type": "number"}},
                 "limit": {"type": "integer", "default": 10},
             }, "required": ["vessel_id", "query_embedding"]}),

        # Moral Geometry
        Tool(name="score_action", description="Score an action's moral valence for an agent",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string"},
                 "action_impacts": {"type": "array", "items": {"type": "number"},
                     "description": "9 floats: impact on [unity, justice, truthfulness, love, detachment, humility, compassion, wisdom, service]"},
             }, "required": ["vessel_id", "action_impacts"]}),

        Tool(name="project_position", description="Project a desired moral position onto the constraint manifold",
             inputSchema={"type": "object", "properties": {
                 "desired": {"type": "array", "items": {"type": "number"}, "description": "9 virtue values [0-1]"},
             }, "required": ["desired"]}),

        # Community
        Tool(name="dashboard", description="Get the community dashboard with vital signs",
             inputSchema={"type": "object", "properties": {}}),

        Tool(name="vitals", description="Get community vital signs",
             inputSchema={"type": "object", "properties": {}}),

        Tool(name="graph_stats", description="Get graph node/edge counts and state hash",
             inputSchema={"type": "object", "properties": {}}),

        # Session
        Tool(name="create_session", description="Start an interactive session for an agent",
             inputSchema={"type": "object", "properties": {
                 "vessel_id": {"type": "string"},
             }, "required": ["vessel_id"]}),

        Tool(name="process_query", description="Process a query through the full context assembly pipeline",
             inputSchema={"type": "object", "properties": {
                 "session_id": {"type": "string"},
                 "query_text": {"type": "string"},
                 "query_embedding": {"type": "array", "items": {"type": "number"}},
             }, "required": ["session_id", "query_text", "query_embedding"]}),

        Tool(name="close_session", description="Close an active session",
             inputSchema={"type": "object", "properties": {
                 "session_id": {"type": "string"},
             }, "required": ["session_id"]}),

        # Snapshot
        Tool(name="create_snapshot", description="Create a verifiable snapshot of the community graph",
             inputSchema={"type": "object", "properties": {
                 "format": {"type": "string", "enum": ["full", "terrain_only", "agents_only"], "default": "full"},
                 "creator_key": {"type": "string", "default": "mcp"},
             }}),
    ]

    @server.list_tools()
    async def list_tools():
        return TOOLS

    # --------------------------------------------------------
    # Tool dispatch
    # --------------------------------------------------------

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        graph = get_graph()
        engine = get_engine()
        session_mgr = get_session_mgr()

        try:
            result = _dispatch(name, arguments, graph, engine, session_mgr,
                               community_id, community_name)
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    return server


def _dispatch(name: str, args: dict, graph, engine, session_mgr,
              community_id: str, community_name: str) -> dict:
    """Route tool calls to Graph Zero functions."""

    # Lazy imports to avoid circular deps
    from graph_zero.graph.schema import (
        create_agent as _create_agent, assemble_constellation,
        add_terrain_node, connect_terrain as _connect_terrain,
        traverse_terrain, find_entry_points, compute_trust_flow,
        compute_vital_signs, record_interaction as _record_interaction,
    )
    from graph_zero.moral.geometry import (
        PhaseState, score_action as _score_action,
        project_position as _project_position, all_constraints_satisfied,
        NUM_VIRTUES,
    )
    from graph_zero.memory.memory import MemoryStore, EpisodeData, SemanticFactData
    from graph_zero.interface.interface import build_dashboard
    from graph_zero.federation.federation import create_snapshot as _create_snapshot, SnapshotFormat

    # -- Agents --
    if name == "create_agent":
        c = _create_agent(graph, community_id, args["vessel_id"], args["name"],
                          args.get("agent_type", "human"))
        if not c:
            return {"error": "Failed to create agent"}
        return {"vessel_id": c.vessel_id, "name": c.name, "type": c.agent_type,
                "kala_balance": c.kala_balance, "moral_position": c.moral_position}

    elif name == "get_agent":
        c = assemble_constellation(graph, args["vessel_id"])
        if not c:
            return {"error": "Agent not found"}
        return {"vessel_id": c.vessel_id, "name": c.name, "type": c.agent_type,
                "trust_ceiling": c.trust_ceiling, "kala_balance": c.kala_balance,
                "moral_position": c.moral_position,
                "tools": [t.get("name") for t in c.tools],
                "skills": [s.get("name") for s in c.skills],
                "memories": len(c.memories)}

    elif name == "get_trust":
        r = compute_trust_flow(graph, args["vessel_id"])
        return {"vessel_id": r.vessel_id, "trust_ceiling": r.trust_ceiling,
                "path_count": r.path_count, "interaction_diversity": r.interaction_diversity}

    elif name == "record_interaction":
        edge = _record_interaction(graph, args["agent_a"], args["agent_b"],
                                   args["interaction_type"], args.get("context", ""))
        if not edge:
            return {"error": "Failed to record interaction"}
        return {"status": "recorded", "edge_id": edge.id}

    # -- Terrain --
    elif name == "add_terrain":
        node = add_terrain_node(graph, args["node_id"], args["source_text"],
                                args.get("layer", "community"),
                                args.get("embedding"), None,
                                args.get("provenance_type", "WITNESS"))
        return {"node_id": node.id, "layer": args.get("layer", "community")}

    elif name == "connect_terrain":
        edge = _connect_terrain(graph, args["source_id"], args["target_id"],
                                args.get("weight", 1.0), args.get("provenance_type", "WITNESS"))
        if not edge:
            return {"error": "Failed to connect terrain"}
        return {"edge_id": edge.id}

    elif name == "query_terrain":
        entry_ids = args.get("entry_point_ids", [])
        emb = args.get("query_embedding")
        if emb and not entry_ids:
            entry_ids = find_entry_points(graph, emb, threshold=0.3, limit=5)
        results = traverse_terrain(graph, entry_ids,
                                   args.get("max_depth", 5), args.get("limit", 10))
        return {"entry_points": entry_ids, "results": [{
            "node_id": r.node.id, "source_text": r.source_text,
            "layer": r.layer, "authority_weight": r.authority_weight,
            "depth": r.depth,
        } for r in results]}

    # -- Memory --
    elif name == "ingest_episode":
        ms = MemoryStore(graph, args["vessel_id"])
        eid = ms.ingest_episode(EpisodeData(
            participants=args["participants"],
            interaction_type=args["interaction_type"],
            content=args["content"], summary=args["summary"],
            embedding=args.get("embedding", [0.5]*8),
            emotional_valence=args.get("emotional_valence", 0.0),
            emotional_arousal=args.get("emotional_arousal", 0.0)))
        return {"episode_id": eid}

    elif name == "extract_fact":
        ms = MemoryStore(graph, args["vessel_id"])
        fid = ms.extract_fact(SemanticFactData(
            subject=args["subject"], predicate=args["predicate"],
            object_value=args["object_value"],
            confidence=args.get("confidence", 0.8),
            embedding=args.get("embedding", [0.5]*8),
            source_episodes=[]))
        return {"fact_id": fid}

    elif name == "retrieve_memories":
        ms = MemoryStore(graph, args["vessel_id"])
        results = ms.retrieve(args["query_embedding"], limit=args.get("limit", 10))
        return {"results": [{"memory_id": r.memory_id, "type": r.memory_type.value,
                             "content": r.content, "score": round(r.score, 4)}
                            for r in results]}

    # -- Moral Geometry --
    elif name == "score_action":
        c = assemble_constellation(graph, args["vessel_id"])
        if not c:
            return {"error": "Agent not found"}
        if len(args["action_impacts"]) != NUM_VIRTUES:
            return {"error": f"Need exactly {NUM_VIRTUES} impacts"}
        state = PhaseState(position=c.moral_position, momentum=[0.0]*NUM_VIRTUES)
        r = _score_action(state, args["action_impacts"])
        return {"total_valence": r.total_valence, "virtue_impacts": r.virtue_impacts,
                "constraint_violations": r.constraint_violations}

    elif name == "project_position":
        projected = _project_position(args["desired"])
        return {"projected": projected,
                "constraints_satisfied": all_constraints_satisfied(projected)}

    # -- Community --
    elif name == "dashboard":
        dash = build_dashboard(graph, community_id, session_mgr)
        return {"community_id": dash.community_id, "total_agents": dash.total_agents,
                "total_terrain": dash.total_terrain, "active_sessions": dash.active_sessions,
                "open_conflicts": dash.open_conflicts}

    elif name == "vitals":
        s = compute_vital_signs(graph, community_id)
        return {"terrain_30d": s.terrain_additions_30d, "active_agents": s.active_agents,
                "open_conflicts": s.open_conflicts, "kala_gini": round(s.kala_concentration, 4)}

    elif name == "graph_stats":
        return {"nodes": graph.node_count, "edges": graph.edge_count,
                "state_hash": graph.compute_state_hash()}

    # -- Session --
    elif name == "create_session":
        session = session_mgr.create_session(args["vessel_id"], community_id)
        if not session:
            return {"error": "Agent not found"}
        return {"session_id": session.session_id, "trust_ceiling": session.trust_ceiling,
                "tools": session.active_tools}

    elif name == "process_query":
        session = session_mgr.get_session(args["session_id"])
        if not session:
            return {"error": "Session not found"}
        return session_mgr.process_query(session, args["query_text"], args["query_embedding"])

    elif name == "close_session":
        ok = session_mgr.close_session(args["session_id"])
        return {"status": "closed" if ok else "not_found"}

    # -- Snapshot --
    elif name == "create_snapshot":
        fmt = getattr(SnapshotFormat, args.get("format", "full").upper(), SnapshotFormat.FULL)
        snap = _create_snapshot(graph, community_id, args.get("creator_key", "mcp"), fmt)
        return {"snapshot_id": snap.manifest.snapshot_id, "nodes": snap.manifest.node_count,
                "edges": snap.manifest.edge_count, "verified": snap.verify()}

    else:
        return {"error": f"Unknown tool: {name}"}

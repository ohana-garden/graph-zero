"""
FalkorDB Property Graph Backend

Drop-in replacement for the in-memory PropertyGraph.
Same interface, backed by FalkorDB Cypher queries.

Node model:
  - Label = node_type
  - Property `_id` = our string ID (FalkorDB uses internal integer IDs)
  - All other properties stored as node properties

Edge model:
  - Relationship type = edge_type
  - Property `_id` = our edge ID
  - Properties stored on the relationship

Thread safety: FalkorDB client handles connection pooling.
"""

import hashlib
import json
import os
import time
from typing import Any, Optional

from graph_zero.graph.backend import Node, Edge

try:
    from falkordb import FalkorDB as FalkorDBClient
    HAS_FALKORDB = True
except ImportError:
    HAS_FALKORDB = False


def _escape(s: str) -> str:
    """Escape a string for Cypher."""
    if s is None:
        return "null"
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _prop_to_cypher(value: Any) -> str:
    """Convert a Python value to a Cypher literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f"'{_escape(value)}'"
    if isinstance(value, list):
        # FalkorDB supports lists of homogeneous types
        if not value:
            return "[]"
        inner = ", ".join(_prop_to_cypher(v) for v in value)
        return f"[{inner}]"
    # Fallback: JSON string
    return f"'{_escape(json.dumps(value))}'"


def _props_to_set(alias: str, props: dict) -> str:
    """Build SET clauses for properties."""
    if not props:
        return ""
    parts = []
    for k, v in props.items():
        if v is not None:
            parts.append(f"{alias}.`{k}` = {_prop_to_cypher(v)}")
    if not parts:
        return ""
    return "SET " + ", ".join(parts)


def _row_to_node(record: dict) -> Node:
    """Convert a FalkorDB node result to our Node dataclass."""
    props = dict(record.properties) if hasattr(record, 'properties') else {}
    node_id = props.pop("_id", str(record.id) if hasattr(record, 'id') else "?")
    # Labels → node_type (take first label)
    labels = record.labels if hasattr(record, 'labels') else []
    node_type = labels[0] if labels else props.pop("_node_type", "Unknown")
    return Node(id=node_id, node_type=node_type, properties=props)


def _row_to_edge(record: dict) -> Edge:
    """Convert a FalkorDB relationship result to our Edge dataclass."""
    props = dict(record.properties) if hasattr(record, 'properties') else {}
    edge_id = props.pop("_id", str(record.id) if hasattr(record, 'id') else "?")
    source_id = props.pop("_source_id", "")
    target_id = props.pop("_target_id", "")
    edge_type = record.relation if hasattr(record, 'relation') else props.pop("_edge_type", "UNKNOWN")
    return Edge(id=edge_id, source_id=source_id, target_id=target_id,
                edge_type=edge_type, properties=props)


class FalkorPropertyGraph:
    """FalkorDB-backed property graph. Same interface as PropertyGraph."""

    def __init__(self, host: str = None, port: int = None,
                 password: str = None, graph_name: str = "graph_zero"):
        if not HAS_FALKORDB:
            raise ImportError("falkordb package not installed")

        self._host = host or os.getenv("FALKORDB_HOST", "localhost")
        self._port = int(port or os.getenv("FALKORDB_PORT", "6379"))
        self._password = password or os.getenv("FALKORDB_PASSWORD", "")
        self._graph_name = graph_name

        kwargs = {"host": self._host, "port": self._port}
        if self._password:
            kwargs["password"] = self._password
        self._db = FalkorDBClient(**kwargs)
        self._graph = self._db.select_graph(self._graph_name)
        self._edge_counter = int(time.time() * 1000) % 1_000_000

        # Create indexes for fast lookup
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes for _id property on all nodes."""
        try:
            self._graph.query("CREATE INDEX FOR (n:_Any) ON (n._id)")
        except Exception:
            pass  # Index may already exist or _Any not supported
        # We'll create per-label indexes as nodes are added

    def _q(self, query: str, params: dict = None):
        """Execute a Cypher query."""
        try:
            return self._graph.query(query, params or {})
        except Exception as e:
            # Log but don't crash
            print(f"FalkorDB query error: {e}\n  Query: {query[:200]}")
            raise

    # --------------------------------------------------------
    # Node operations
    # --------------------------------------------------------

    def add_node(self, node_id: str, node_type: str,
                 properties: Optional[dict] = None) -> Node:
        props = dict(properties or {})
        props["_id"] = node_id

        # Build property map for Cypher
        prop_str = ", ".join(f"`{k}`: {_prop_to_cypher(v)}" for k, v in props.items()
                             if v is not None)

        # MERGE on _id, set all props (overwrite if exists)
        label = node_type.replace(" ", "_").replace("-", "_")
        query = f"MERGE (n:`{label}` {{_id: '{_escape(node_id)}'}}) SET n = {{{prop_str}}} RETURN n"
        self._q(query)

        return Node(id=node_id, node_type=node_type, properties=properties or {})

    def get_node(self, node_id: str) -> Optional[Node]:
        result = self._q(
            f"MATCH (n {{_id: '{_escape(node_id)}'}}) RETURN n LIMIT 1")
        if result.result_set:
            return _row_to_node(result.result_set[0][0])
        return None

    def has_node(self, node_id: str) -> bool:
        result = self._q(
            f"MATCH (n {{_id: '{_escape(node_id)}'}}) RETURN count(n) AS c")
        return result.result_set[0][0] > 0

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        label = node_type.replace(" ", "_").replace("-", "_")
        result = self._q(f"MATCH (n:`{label}`) RETURN n")
        return [_row_to_node(row[0]) for row in result.result_set]

    def find_nodes(self, node_type: str, **props) -> list[Node]:
        label = node_type.replace(" ", "_").replace("-", "_")
        where_parts = [f"n.`{k}` = {_prop_to_cypher(v)}" for k, v in props.items()]
        where = " AND ".join(where_parts) if where_parts else "true"
        result = self._q(f"MATCH (n:`{label}`) WHERE {where} RETURN n")
        return [_row_to_node(row[0]) for row in result.result_set]

    def update_node(self, node_id: str, **props) -> Optional[Node]:
        set_clause = _props_to_set("n", props)
        if not set_clause:
            return self.get_node(node_id)
        result = self._q(
            f"MATCH (n {{_id: '{_escape(node_id)}'}}) {set_clause} RETURN n")
        if result.result_set:
            return _row_to_node(result.result_set[0][0])
        return None

    def remove_node(self, node_id: str) -> bool:
        result = self._q(
            f"MATCH (n {{_id: '{_escape(node_id)}'}}) DETACH DELETE n RETURN count(n) AS c")
        # FalkorDB doesn't return count on DELETE easily
        return True  # if node didn't exist, MATCH found nothing, still fine

    # --------------------------------------------------------
    # Edge operations
    # --------------------------------------------------------

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                 properties: Optional[dict] = None,
                 edge_id: Optional[str] = None) -> Optional[Edge]:
        if edge_id is None:
            self._edge_counter += 1
            edge_id = f"e_{self._edge_counter}"

        props = dict(properties or {})
        props["_id"] = edge_id
        props["_source_id"] = source_id
        props["_target_id"] = target_id

        rel_type = edge_type.replace(" ", "_").replace("-", "_")
        prop_str = ", ".join(f"`{k}`: {_prop_to_cypher(v)}" for k, v in props.items()
                             if v is not None)

        query = (f"MATCH (a {{_id: '{_escape(source_id)}'}}), (b {{_id: '{_escape(target_id)}'}}) "
                 f"CREATE (a)-[r:`{rel_type}` {{{prop_str}}}]->(b) RETURN r")
        try:
            result = self._q(query)
            if result.result_set:
                return Edge(id=edge_id, source_id=source_id, target_id=target_id,
                            edge_type=edge_type, properties=properties or {})
        except Exception:
            pass
        return None

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        result = self._q(
            f"MATCH (a)-[r {{_id: '{_escape(edge_id)}'}}]->(b) "
            f"RETURN r, a._id AS src, b._id AS tgt LIMIT 1")
        if result.result_set:
            row = result.result_set[0]
            rel = row[0]
            props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            props.pop("_id", None)
            props.pop("_source_id", None)
            props.pop("_target_id", None)
            return Edge(id=edge_id, source_id=row[1], target_id=row[2],
                        edge_type=rel.relation if hasattr(rel, 'relation') else "UNKNOWN",
                        properties=props)
        return None

    def get_edges_between(self, source_id: str, target_id: str,
                          edge_type: Optional[str] = None) -> list[Edge]:
        if edge_type:
            rel = edge_type.replace(" ", "_").replace("-", "_")
            query = (f"MATCH (a {{_id: '{_escape(source_id)}'}})"
                     f"-[r:`{rel}`]->"
                     f"(b {{_id: '{_escape(target_id)}'}}) RETURN r")
        else:
            query = (f"MATCH (a {{_id: '{_escape(source_id)}'}})"
                     f"-[r]->"
                     f"(b {{_id: '{_escape(target_id)}'}}) RETURN r")
        result = self._q(query)
        edges = []
        for row in result.result_set:
            rel = row[0]
            props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            eid = props.pop("_id", "?")
            props.pop("_source_id", None)
            props.pop("_target_id", None)
            edges.append(Edge(id=eid, source_id=source_id, target_id=target_id,
                              edge_type=rel.relation if hasattr(rel, 'relation') else (edge_type or "?"),
                              properties=props))
        return edges

    def get_outgoing(self, node_id: str, edge_type: Optional[str] = None) -> list[Edge]:
        if edge_type:
            rel = edge_type.replace(" ", "_").replace("-", "_")
            query = (f"MATCH (a {{_id: '{_escape(node_id)}'}})-[r:`{rel}`]->(b) "
                     f"RETURN r, b._id AS tgt")
        else:
            query = (f"MATCH (a {{_id: '{_escape(node_id)}'}})-[r]->(b) "
                     f"RETURN r, b._id AS tgt")
        result = self._q(query)
        edges = []
        for row in result.result_set:
            rel = row[0]
            props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            eid = props.pop("_id", "?")
            props.pop("_source_id", None)
            props.pop("_target_id", None)
            edges.append(Edge(id=eid, source_id=node_id, target_id=row[1],
                              edge_type=rel.relation if hasattr(rel, 'relation') else "?",
                              properties=props))
        return edges

    def get_incoming(self, node_id: str, edge_type: Optional[str] = None) -> list[Edge]:
        if edge_type:
            rel = edge_type.replace(" ", "_").replace("-", "_")
            query = (f"MATCH (a)-[r:`{rel}`]->(b {{_id: '{_escape(node_id)}'}}) "
                     f"RETURN r, a._id AS src")
        else:
            query = (f"MATCH (a)-[r]->(b {{_id: '{_escape(node_id)}'}}) "
                     f"RETURN r, a._id AS src")
        result = self._q(query)
        edges = []
        for row in result.result_set:
            rel = row[0]
            props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            eid = props.pop("_id", "?")
            props.pop("_source_id", None)
            props.pop("_target_id", None)
            edges.append(Edge(id=eid, source_id=row[1], target_id=node_id,
                              edge_type=rel.relation if hasattr(rel, 'relation') else "?",
                              properties=props))
        return edges

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None,
                      direction: str = "out") -> list[Node]:
        results = []
        if direction in ("out", "both"):
            for edge in self.get_outgoing(node_id, edge_type):
                node = self.get_node(edge.target_id)
                if node:
                    results.append(node)
        if direction in ("in", "both"):
            for edge in self.get_incoming(node_id, edge_type):
                node = self.get_node(edge.source_id)
                if node:
                    results.append(node)
        return results

    def remove_edge(self, edge_id: str) -> bool:
        self._q(f"MATCH ()-[r {{_id: '{_escape(edge_id)}'}}]->() DELETE r")
        return True

    # --------------------------------------------------------
    # Traversal (delegate to in-memory style BFS using edge queries)
    # --------------------------------------------------------

    def traverse(self, start_id: str, edge_type: str,
                 max_depth: int = 5,
                 edge_filter: Optional[callable] = None,
                 direction: str = "out") -> list[tuple[Node, int, list[Edge]]]:
        """BFS traversal. Uses repeated queries (not a single Cypher path query)
        to support the edge_filter callback."""
        visited = {start_id}
        queue = [(start_id, 0, [])]
        results = []

        while queue:
            current_id, depth, path = queue.pop(0)
            if depth > 0:
                node = self.get_node(current_id)
                if node:
                    results.append((node, depth, path))
            if depth >= max_depth:
                continue

            if direction == "out":
                edges = self.get_outgoing(current_id, edge_type)
            elif direction == "in":
                edges = self.get_incoming(current_id, edge_type)
            else:
                edges = self.get_outgoing(current_id, edge_type) + self.get_incoming(current_id, edge_type)

            for edge in edges:
                next_id = edge.target_id if edge.source_id == current_id else edge.source_id
                if next_id not in visited:
                    if edge_filter is None or edge_filter(edge):
                        visited.add(next_id)
                        queue.append((next_id, depth + 1, path + [edge]))

        return results

    def find_paths(self, start_id: str, end_id: str,
                   edge_type: str, max_depth: int = 5,
                   edge_filter: Optional[callable] = None) -> list[list[Edge]]:
        """Find all paths between two nodes."""
        if start_id == end_id:
            return [[]]
        queue = [(start_id, [])]
        all_paths = []
        visited_at_depth = {start_id: 0}

        while queue:
            current_id, path = queue.pop(0)
            depth = len(path)
            if depth >= max_depth:
                continue
            for edge in self.get_outgoing(current_id, edge_type):
                next_id = edge.target_id
                if edge_filter and not edge_filter(edge):
                    continue
                new_path = path + [edge]
                if next_id == end_id:
                    all_paths.append(new_path)
                    continue
                prev_depth = visited_at_depth.get(next_id)
                if prev_depth is None or depth + 1 <= prev_depth:
                    visited_at_depth[next_id] = depth + 1
                    queue.append((next_id, new_path))

        return all_paths

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    @property
    def node_count(self) -> int:
        result = self._q("MATCH (n) RETURN count(n) AS c")
        return result.result_set[0][0] if result.result_set else 0

    @property
    def edge_count(self) -> int:
        result = self._q("MATCH ()-[r]->() RETURN count(r) AS c")
        return result.result_set[0][0] if result.result_set else 0

    def compute_state_hash(self) -> str:
        """Deterministic hash. Queries all nodes and edges sorted by _id."""
        h = hashlib.sha256()
        # Nodes
        result = self._q("MATCH (n) RETURN n ORDER BY n._id")
        for row in result.result_set:
            node = _row_to_node(row[0])
            h.update(node.id.encode())
            h.update(node.node_type.encode())
            for k in sorted(node.properties.keys()):
                h.update(k.encode())
                h.update(str(node.properties[k]).encode())
        # Edges
        result = self._q("MATCH (a)-[r]->(b) RETURN r, a._id, b._id ORDER BY r._id")
        for row in result.result_set:
            rel = row[0]
            props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            eid = props.get("_id", "?")
            h.update(eid.encode())
            h.update(str(row[1]).encode())
            h.update(str(row[2]).encode())
            h.update((rel.relation if hasattr(rel, 'relation') else "?").encode())
            clean = {k: v for k, v in props.items()
                     if not k.startswith("_")}
            for k in sorted(clean.keys()):
                h.update(k.encode())
                h.update(str(clean[k]).encode())
        return h.hexdigest()

    # --------------------------------------------------------
    # FalkorDB-specific: expose _nodes/_edges for federation
    # compatibility (federation serializes via these dicts)
    # --------------------------------------------------------

    @property
    def _nodes(self) -> dict:
        """Compatibility: return all nodes as a dict. Use sparingly."""
        result = self._q("MATCH (n) RETURN n")
        nodes = {}
        for row in result.result_set:
            node = _row_to_node(row[0])
            nodes[node.id] = node
        return nodes

    @property
    def _edges(self) -> dict:
        """Compatibility: return all edges as a dict. Use sparingly."""
        result = self._q("MATCH (a)-[r]->(b) RETURN r, a._id, b._id")
        edges = {}
        for row in result.result_set:
            rel = row[0]
            props = dict(rel.properties) if hasattr(rel, 'properties') else {}
            eid = props.pop("_id", "?")
            src = props.pop("_source_id", row[1])
            tgt = props.pop("_target_id", row[2])
            etype = rel.relation if hasattr(rel, 'relation') else "?"
            edges[eid] = Edge(id=eid, source_id=src, target_id=tgt,
                              edge_type=etype, properties=props)
        return edges

    def clear(self):
        """Drop all data in this graph."""
        self._q("MATCH (n) DETACH DELETE n")

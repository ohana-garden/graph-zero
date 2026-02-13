"""
Graph Zero In-Memory Property Graph Backend

Production uses FalkorDB. This is a dict-based in-memory graph
that implements the same semantics for development and testing.

Nodes have: id, type, properties
Edges have: id, source, target, type, properties
Both are queryable by type and property values.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator
from enum import Enum


@dataclass
class Node:
    """A node in the property graph."""
    id: str
    node_type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None) -> Any:
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id


@dataclass
class Edge:
    """An edge in the property graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default=None) -> Any:
        return self.properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.id == other.id


class PropertyGraph:
    """In-memory property graph with typed nodes and edges.

    Provides the query interface that the rest of Graph Zero uses.
    In production, these queries become Cypher against FalkorDB.
    """

    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, Edge] = {}
        # Indexes for fast lookup
        self._nodes_by_type: dict[str, set[str]] = {}
        self._edges_by_type: dict[str, set[str]] = {}
        self._outgoing: dict[str, set[str]] = {}   # node_id -> set of edge_ids
        self._incoming: dict[str, set[str]] = {}   # node_id -> set of edge_ids
        self._edge_counter = 0

    # --------------------------------------------------------
    # Node operations
    # --------------------------------------------------------

    def add_node(self, node_id: str, node_type: str,
                 properties: Optional[dict] = None) -> Node:
        """Add a node. Overwrites if exists."""
        node = Node(id=node_id, node_type=node_type,
                    properties=properties or {})
        self._nodes[node_id] = node
        self._nodes_by_type.setdefault(node_type, set()).add(node_id)
        if node_id not in self._outgoing:
            self._outgoing[node_id] = set()
        if node_id not in self._incoming:
            self._incoming[node_id] = set()
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_nodes_by_type(self, node_type: str) -> list[Node]:
        ids = self._nodes_by_type.get(node_type, set())
        return [self._nodes[nid] for nid in ids if nid in self._nodes]

    def find_nodes(self, node_type: str, **props) -> list[Node]:
        """Find nodes by type and property values."""
        results = []
        for node in self.get_nodes_by_type(node_type):
            if all(node.get(k) == v for k, v in props.items()):
                results.append(node)
        return results

    def update_node(self, node_id: str, **props) -> Optional[Node]:
        """Update node properties."""
        node = self._nodes.get(node_id)
        if node:
            node.properties.update(props)
        return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges."""
        if node_id not in self._nodes:
            return False
        # Remove edges
        for eid in list(self._outgoing.get(node_id, set())):
            self.remove_edge(eid)
        for eid in list(self._incoming.get(node_id, set())):
            self.remove_edge(eid)
        # Remove from indexes
        node = self._nodes[node_id]
        self._nodes_by_type.get(node.node_type, set()).discard(node_id)
        del self._nodes[node_id]
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        return True

    # --------------------------------------------------------
    # Edge operations
    # --------------------------------------------------------

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                 properties: Optional[dict] = None,
                 edge_id: Optional[str] = None) -> Optional[Edge]:
        """Add an edge between two nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None
        if edge_id is None:
            self._edge_counter += 1
            edge_id = f"e_{self._edge_counter}"
        edge = Edge(id=edge_id, source_id=source_id, target_id=target_id,
                    edge_type=edge_type, properties=properties or {})
        self._edges[edge_id] = edge
        self._edges_by_type.setdefault(edge_type, set()).add(edge_id)
        self._outgoing.setdefault(source_id, set()).add(edge_id)
        self._incoming.setdefault(target_id, set()).add(edge_id)
        return edge

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        return self._edges.get(edge_id)

    def get_edges_between(self, source_id: str, target_id: str,
                          edge_type: Optional[str] = None) -> list[Edge]:
        """Get edges between two nodes, optionally filtered by type."""
        results = []
        for eid in self._outgoing.get(source_id, set()):
            edge = self._edges.get(eid)
            if edge and edge.target_id == target_id:
                if edge_type is None or edge.edge_type == edge_type:
                    results.append(edge)
        return results

    def get_outgoing(self, node_id: str, edge_type: Optional[str] = None) -> list[Edge]:
        """Get all outgoing edges from a node."""
        results = []
        for eid in self._outgoing.get(node_id, set()):
            edge = self._edges.get(eid)
            if edge and (edge_type is None or edge.edge_type == edge_type):
                results.append(edge)
        return results

    def get_incoming(self, node_id: str, edge_type: Optional[str] = None) -> list[Edge]:
        """Get all incoming edges to a node."""
        results = []
        for eid in self._incoming.get(node_id, set()):
            edge = self._edges.get(eid)
            if edge and (edge_type is None or edge.edge_type == edge_type):
                results.append(edge)
        return results

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None,
                      direction: str = "out") -> list[Node]:
        """Get neighbor nodes via edges of given type."""
        results = []
        if direction in ("out", "both"):
            for edge in self.get_outgoing(node_id, edge_type):
                node = self._nodes.get(edge.target_id)
                if node:
                    results.append(node)
        if direction in ("in", "both"):
            for edge in self.get_incoming(node_id, edge_type):
                node = self._nodes.get(edge.source_id)
                if node:
                    results.append(node)
        return results

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge."""
        edge = self._edges.get(edge_id)
        if not edge:
            return False
        self._outgoing.get(edge.source_id, set()).discard(edge_id)
        self._incoming.get(edge.target_id, set()).discard(edge_id)
        self._edges_by_type.get(edge.edge_type, set()).discard(edge_id)
        del self._edges[edge_id]
        return True

    # --------------------------------------------------------
    # Traversal
    # --------------------------------------------------------

    def traverse(self, start_id: str, edge_type: str,
                 max_depth: int = 5,
                 edge_filter: Optional[callable] = None,
                 direction: str = "out") -> list[tuple[Node, int, list[Edge]]]:
        """BFS traversal from a starting node.

        Returns list of (node, depth, path_edges).
        edge_filter: callable(Edge) -> bool to filter edges.
        """
        visited = {start_id}
        # (node_id, depth, path_edges)
        queue = [(start_id, 0, [])]
        results = []

        while queue:
            current_id, depth, path = queue.pop(0)
            if depth > 0:
                node = self._nodes.get(current_id)
                if node:
                    results.append((node, depth, path))

            if depth >= max_depth:
                continue

            edges = (self.get_outgoing(current_id, edge_type) if direction == "out"
                     else self.get_incoming(current_id, edge_type) if direction == "in"
                     else self.get_outgoing(current_id, edge_type) + self.get_incoming(current_id, edge_type))

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
        """Find all paths between two nodes (BFS, bounded depth)."""
        if start_id == end_id:
            return [[]]
        # (current_id, path_edges)
        queue = [(start_id, [])]
        all_paths = []
        visited_at_depth: dict[str, int] = {start_id: 0}

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
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def compute_state_hash(self) -> str:
        """Deterministic hash of entire graph state.
        Corresponds to StateRoot computation in the spec."""
        h = hashlib.sha256()
        # Sort nodes by ID
        for nid in sorted(self._nodes.keys()):
            node = self._nodes[nid]
            h.update(nid.encode())
            h.update(node.node_type.encode())
            for k in sorted(node.properties.keys()):
                h.update(k.encode())
                h.update(str(node.properties[k]).encode())
        # Sort edges by ID
        for eid in sorted(self._edges.keys()):
            edge = self._edges[eid]
            h.update(eid.encode())
            h.update(edge.source_id.encode())
            h.update(edge.target_id.encode())
            h.update(edge.edge_type.encode())
            for k in sorted(edge.properties.keys()):
                h.update(k.encode())
                h.update(str(edge.properties[k]).encode())
        return h.hexdigest()

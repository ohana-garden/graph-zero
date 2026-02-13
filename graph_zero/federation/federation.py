"""
Graph Zero Federation Layer

Communities are sovereign. Federation is voluntary cooperation.

Architecture:
  - Snapshot: portable, signed point-in-time image of a community's graph
  - CloneEnvelope: agent identity for cross-community travel
  - LogReplicator: selective replication between federated communities
  - VisitingAgent: guest access with trust dampening

Key invariants:
  - Clone trust ceiling is DAMPENED (home trust * 0.6)
  - Visiting agents cannot touch protected domains without local endorsement
  - Snapshots are verifiable (hash matches state_root)
  - Replication is selective (only specified node/edge types)
"""

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import NT, ET


# ============================================================
# Snapshot Protocol
# ============================================================

class SnapshotFormat(Enum):
    FULL = "full"
    TERRAIN_ONLY = "terrain_only"
    AGENTS_ONLY = "agents_only"
    SELECTIVE = "selective"


@dataclass
class SnapshotManifest:
    snapshot_id: str
    community_id: str
    created_at: int
    format: SnapshotFormat
    node_count: int
    edge_count: int
    state_root: str
    log_head_hash: str
    included_types: list[str]
    creator_key: str
    signature: bytes = b""


@dataclass
class Snapshot:
    manifest: SnapshotManifest
    nodes: list[dict]
    edges: list[dict]

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.manifest.community_id.encode())
        h.update(str(self.manifest.created_at).encode())
        for node in sorted(self.nodes, key=lambda n: n["id"]):
            h.update(json.dumps(node, sort_keys=True).encode())
        for edge in sorted(self.edges, key=lambda e: e["id"]):
            h.update(json.dumps(edge, sort_keys=True).encode())
        return h.hexdigest()

    def verify(self) -> bool:
        return self.compute_hash() == self.manifest.state_root


def _serialize_node(node: Node) -> dict:
    d = {"id": node.id, "node_type": node.node_type}
    for k, v in node.properties.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            d[k] = v
    return d


def _serialize_edge(edge: Edge) -> dict:
    d = {"id": edge.id, "source_id": edge.source_id,
         "target_id": edge.target_id, "edge_type": edge.edge_type}
    for k, v in edge.properties.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            d[k] = v
    return d


def create_snapshot(graph: PropertyGraph, community_id: str,
                    creator_key: str,
                    format: SnapshotFormat = SnapshotFormat.FULL,
                    included_types: Optional[list[str]] = None,
                    log_head_hash: str = "") -> Snapshot:
    nodes = []
    edges = []

    if format == SnapshotFormat.FULL:
        for nid in sorted(graph._nodes):
            nodes.append(_serialize_node(graph._nodes[nid]))
        for eid in sorted(graph._edges):
            edges.append(_serialize_edge(graph._edges[eid]))
        type_list = list(set(n["node_type"] for n in nodes))

    elif format == SnapshotFormat.TERRAIN_ONLY:
        type_filter = {NT.TERRAIN_NODE, NT.VIRTUE_ANCHOR}
        edge_filter = {ET.CONNECTS_TO, ET.COUPLES_WITH}
        for nid in sorted(graph._nodes):
            node = graph._nodes[nid]
            if node.node_type in type_filter:
                nodes.append(_serialize_node(node))
        for eid in sorted(graph._edges):
            edge = graph._edges[eid]
            if edge.edge_type in edge_filter:
                edges.append(_serialize_edge(edge))
        type_list = list(type_filter)

    elif format == SnapshotFormat.AGENTS_ONLY:
        type_filter = {NT.VESSEL_ANCHOR, NT.VESSEL_POSITION, NT.VESSEL_KALA}
        edge_filter = {ET.PART_OF_VESSEL, ET.MEMBER_OF}
        for nid in sorted(graph._nodes):
            node = graph._nodes[nid]
            if node.node_type in type_filter:
                nodes.append(_serialize_node(node))
        for eid in sorted(graph._edges):
            edge = graph._edges[eid]
            if edge.edge_type in edge_filter:
                edges.append(_serialize_edge(edge))
        type_list = list(type_filter)

    elif format == SnapshotFormat.SELECTIVE:
        type_filter = set(included_types or [])
        for nid in sorted(graph._nodes):
            node = graph._nodes[nid]
            if node.node_type in type_filter:
                nodes.append(_serialize_node(node))
        included_ids = {n["id"] for n in nodes}
        for eid in sorted(graph._edges):
            edge = graph._edges[eid]
            if edge.source_id in included_ids and edge.target_id in included_ids:
                edges.append(_serialize_edge(edge))
        type_list = list(type_filter)
    else:
        type_list = []

    snap = Snapshot(
        manifest=SnapshotManifest(
            snapshot_id=f"snap_{community_id}_{int(time.time()*1000)}",
            community_id=community_id,
            created_at=int(time.time() * 1000),
            format=format,
            node_count=len(nodes),
            edge_count=len(edges),
            state_root="",
            log_head_hash=log_head_hash,
            included_types=type_list,
            creator_key=creator_key,
        ),
        nodes=nodes,
        edges=edges,
    )
    snap.manifest.state_root = snap.compute_hash()
    return snap


def restore_snapshot(snapshot: Snapshot,
                     target_graph: Optional[PropertyGraph] = None) -> PropertyGraph:
    if not snapshot.verify():
        raise ValueError("Snapshot integrity check failed")

    graph = target_graph or PropertyGraph()
    for nd in snapshot.nodes:
        props = {k: v for k, v in nd.items() if k not in ("id", "node_type")}
        graph.add_node(nd["id"], nd["node_type"], props)
    for ed in snapshot.edges:
        props = {k: v for k, v in ed.items()
                 if k not in ("id", "source_id", "target_id", "edge_type")}
        graph.add_edge(ed["source_id"], ed["target_id"],
                       ed["edge_type"], props, edge_id=ed["id"])
    return graph


# ============================================================
# Clone Protocol
# ============================================================

TRUST_DAMPENING_FACTOR = 0.6

@dataclass
class CloneEnvelope:
    clone_id: str
    home_community_id: str
    home_vessel_id: str
    public_key: str
    moral_position: list[float]
    trust_ceiling_home: float
    trust_ceiling_dampened: float
    skills: list[str]
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    expires_at: int = 0
    signature: bytes = b""

    def is_expired(self, now: Optional[int] = None) -> bool:
        if self.expires_at == 0:
            return False
        return (now or int(time.time() * 1000)) > self.expires_at

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.clone_id.encode())
        h.update(self.home_community_id.encode())
        h.update(self.home_vessel_id.encode())
        h.update(self.public_key.encode())
        for v in self.moral_position:
            h.update(float.hex(v).encode())
        h.update(float.hex(self.trust_ceiling_home).encode())
        return h.hexdigest()


def create_clone_envelope(graph: PropertyGraph, vessel_id: str,
                          home_community_id: str,
                          validity_days: int = 90) -> Optional[CloneEnvelope]:
    anchor = graph.get_node(vessel_id)
    if not anchor or anchor.node_type != NT.VESSEL_ANCHOR:
        return None

    position = [0.5] * 9
    for edge in graph.get_incoming(vessel_id, ET.PART_OF_VESSEL):
        node = graph.get_node(edge.source_id)
        if node and node.node_type == NT.VESSEL_POSITION:
            position = [node.get(f"v{i}", 0.5) for i in range(9)]
            break

    trust_home = anchor.get("trust_ceiling", 0.0)
    skills = [s.get("name", "?") for s in
              graph.get_neighbors(vessel_id, ET.HAS_SKILL, direction="out")]

    now = int(time.time() * 1000)
    return CloneEnvelope(
        clone_id=f"clone_{vessel_id}_{now}",
        home_community_id=home_community_id,
        home_vessel_id=vessel_id,
        public_key=anchor.get("public_key", ""),
        moral_position=position,
        trust_ceiling_home=trust_home,
        trust_ceiling_dampened=trust_home * TRUST_DAMPENING_FACTOR,
        skills=skills,
        created_at=now,
        expires_at=now + (validity_days * 86400000) if validity_days > 0 else 0,
    )


def admit_visiting_agent(graph: PropertyGraph, target_community_id: str,
                         envelope: CloneEnvelope) -> Optional[str]:
    if envelope.is_expired():
        return None

    vid = f"visiting_{envelope.home_vessel_id}_{target_community_id}"

    graph.add_node(vid, NT.VESSEL_ANCHOR, {
        "name": f"Visiting:{envelope.home_vessel_id}",
        "type": "visiting",
        "home_community": envelope.home_community_id,
        "home_vessel_id": envelope.home_vessel_id,
        "clone_id": envelope.clone_id,
        "trust_ceiling": envelope.trust_ceiling_dampened,
        "trust_ceiling_home": envelope.trust_ceiling_home,
        "active": True,
        "frozen": False,
        "attestation_depth": 2,
        "skills": envelope.skills,
        "created_at": envelope.created_at,
        "expires_at": envelope.expires_at,
    })

    pos_props = {f"v{i}": envelope.moral_position[i] for i in range(9)}
    pos_props["vessel_id"] = vid
    graph.add_node(f"{vid}_pos", NT.VESSEL_POSITION, pos_props)
    graph.add_edge(f"{vid}_pos", vid, ET.PART_OF_VESSEL)

    graph.add_edge(vid, target_community_id, "VISITING", {
        "since": int(time.time() * 1000),
        "role": "visitor",
    })

    graph.add_node(f"cloneref_{envelope.clone_id}", NT.CLONE_ENVELOPE, {
        "clone_id": envelope.clone_id,
        "home_community_id": envelope.home_community_id,
    })
    graph.add_edge(vid, f"cloneref_{envelope.clone_id}", ET.CLONE_OF)

    return vid


# ============================================================
# Log Replication
# ============================================================

class ReplicationMode(Enum):
    TERRAIN_SYNC = "terrain_sync"
    TRUST_ATTESTATION = "trust_attest"
    CONFLICT_SHARE = "conflict_share"
    FULL_MIRROR = "full_mirror"


@dataclass
class FederationLink:
    link_id: str
    community_a: str
    community_b: str
    mode: ReplicationMode
    established_at: int
    active: bool = True
    last_sync_at: int = 0
    sync_count: int = 0


@dataclass
class ReplicationBatch:
    source_community: str
    target_community: str
    batch_id: str
    nodes: list[dict]
    edges: list[dict]
    timestamp: int
    batch_hash: str = ""

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.source_community.encode())
        h.update(str(self.timestamp).encode())
        for n in sorted(self.nodes, key=lambda x: x.get("id", "")):
            h.update(json.dumps(n, sort_keys=True).encode())
        for e in sorted(self.edges, key=lambda x: x.get("id", "")):
            h.update(json.dumps(e, sort_keys=True).encode())
        return h.hexdigest()


class LogReplicator:
    def __init__(self):
        self._links: dict[str, FederationLink] = {}

    def establish_link(self, community_a: str, community_b: str,
                       mode: ReplicationMode) -> FederationLink:
        link_id = f"fed_{community_a}_{community_b}"
        link = FederationLink(
            link_id=link_id, community_a=community_a, community_b=community_b,
            mode=mode, established_at=int(time.time() * 1000),
        )
        self._links[link_id] = link
        return link

    def dissolve_link(self, link_id: str) -> bool:
        link = self._links.get(link_id)
        if link:
            link.active = False
            return True
        return False

    def get_active_links(self, community_id: str) -> list[FederationLink]:
        return [l for l in self._links.values()
                if l.active and (l.community_a == community_id
                                 or l.community_b == community_id)]

    def prepare_batch(self, source_graph: PropertyGraph,
                      link: FederationLink,
                      since_ms: int = 0) -> ReplicationBatch:
        nodes = []
        edges = []

        if link.mode == ReplicationMode.TERRAIN_SYNC:
            nfilter = {NT.TERRAIN_NODE, NT.VIRTUE_ANCHOR}
            efilter = {ET.CONNECTS_TO, ET.COUPLES_WITH}
        elif link.mode == ReplicationMode.TRUST_ATTESTATION:
            nfilter = {NT.VESSEL_ANCHOR, NT.TRUST_PROFILE}
            efilter = {ET.INTERACTED_WITH, ET.ATTESTED_BY}
        elif link.mode == ReplicationMode.CONFLICT_SHARE:
            nfilter = {NT.CONFLICT_SET, NT.QUARANTINE_FLAG}
            efilter = {ET.CONTENDS_IN, ET.FLAGS}
        elif link.mode == ReplicationMode.FULL_MIRROR:
            nfilter = None
            efilter = None
        else:
            nfilter, efilter = set(), set()

        for nid, node in source_graph._nodes.items():
            if nfilter is not None and node.node_type not in nfilter:
                continue
            if node.get("created_at", 0) >= since_ms:
                nodes.append(_serialize_node(node))

        included_ids = {n["id"] for n in nodes}
        for eid, edge in source_graph._edges.items():
            if efilter is not None and edge.edge_type not in efilter:
                continue
            if edge.source_id in included_ids or edge.target_id in included_ids:
                edges.append(_serialize_edge(edge))

        batch = ReplicationBatch(
            source_community=link.community_a,
            target_community=link.community_b,
            batch_id=f"batch_{link.link_id}_{int(time.time()*1000)}",
            nodes=nodes, edges=edges,
            timestamp=int(time.time() * 1000),
        )
        batch.batch_hash = batch.compute_hash()
        return batch

    def apply_batch(self, batch: ReplicationBatch,
                    target_graph: PropertyGraph) -> dict:
        if batch.compute_hash() != batch.batch_hash:
            return {"error": "Hash mismatch", "applied_nodes": 0, "applied_edges": 0}

        prefix = f"fed:{batch.source_community}:"
        id_map = {}
        applied_n = 0
        applied_e = 0

        for nd in batch.nodes:
            old_id = nd["id"]
            new_id = f"{prefix}{old_id}"
            id_map[old_id] = new_id
            props = {k: v for k, v in nd.items() if k not in ("id", "node_type")}
            props["_federated_from"] = batch.source_community
            props["_original_id"] = old_id
            target_graph.add_node(new_id, nd["node_type"], props)
            applied_n += 1

        for ed in batch.edges:
            src = id_map.get(ed["source_id"], ed["source_id"])
            tgt = id_map.get(ed["target_id"], ed["target_id"])
            if target_graph.has_node(src) and target_graph.has_node(tgt):
                props = {k: v for k, v in ed.items()
                         if k not in ("id", "source_id", "target_id", "edge_type")}
                props["_federated_from"] = batch.source_community
                target_graph.add_edge(src, tgt, ed["edge_type"], props,
                                      edge_id=f"{prefix}{ed['id']}")
                applied_e += 1

        link_id = f"fed_{batch.source_community}_{batch.target_community}"
        link = self._links.get(link_id)
        if link:
            link.last_sync_at = int(time.time() * 1000)
            link.sync_count += 1

        return {"applied_nodes": applied_n, "applied_edges": applied_e,
                "batch_hash": batch.batch_hash}

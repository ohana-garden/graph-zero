"""
Graph Zero Provenance and Conflict Resolution

Provenance: tracks where every piece of knowledge came from.
The structural firewall between "AI thinks" and "community knows."

Conflict Resolution: ConflictSets, Resolve, Challenges, Quarantine.
Disagreement is normal. The system records it, never hides it.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import NT, ET, VERIFIED_PROVENANCE, TAINTED_PROVENANCE


class ProvenanceType(Enum):
    BEDROCK = "BEDROCK"
    WITNESS = "WITNESS"
    CROSS_VERIFIED = "CROSS_VERIFIED"
    EMPIRICAL = "EMPIRICAL"
    COMMUNITY_CONSENSUS = "COMMUNITY_CONSENSUS"
    INFERENCE = "INFERENCE"
    SOURCE_UNVERIFIED = "SOURCE_UNVERIFIED"


# ============================================================
# Taint Analysis
# ============================================================

def is_tainted(graph: PropertyGraph, entity_id: str,
               _visited: Optional[set] = None) -> bool:
    """Check if an entity is tainted by walking the provenance DAG.

    Tainted = ANY path from entity to sources passes through
    INFERENCE or SOURCE_UNVERIFIED without intervening verification.
    """
    if _visited is None:
        _visited = set()
    if entity_id in _visited:
        return False
    _visited.add(entity_id)

    node = graph.get_node(entity_id)
    if not node:
        return True

    prov = node.get("provenance_type", "")
    if prov == "BEDROCK":
        return False
    if prov in TAINTED_PROVENANCE:
        return True
    if prov in VERIFIED_PROVENANCE:
        return False

    # Check PROMOTED_FROM (earned terrain chain)
    for edge in graph.get_incoming(entity_id, ET.PROMOTED_FROM):
        verification = graph.get_node(edge.source_id)
        if verification and verification.node_type in (
            NT.WITNESS_VERIFICATION, NT.CROSS_VERIFICATION,
            NT.EMPIRICAL_VERIFICATION
        ):
            return False

    # Check VERIFIES edges
    for edge in graph.get_incoming(entity_id, ET.VERIFIES):
        verifier = graph.get_node(edge.source_id)
        if verifier:
            vtype = verifier.get("provenance_type", verifier.get("type", ""))
            if vtype in VERIFIED_PROVENANCE:
                return False

    # Walk SOURCED_FROM edges
    for edge in graph.get_outgoing(entity_id, ET.SOURCED_FROM):
        if is_tainted(graph, edge.target_id, _visited):
            return True

    return not bool(prov)


def check_authority_closure(graph: PropertyGraph, entity_id: str) -> bool:
    """Is this entity citable? True = yes, it has clean provenance."""
    return not is_tainted(graph, entity_id)


def get_provenance_chain(graph: PropertyGraph, entity_id: str,
                         max_depth: int = 10,
                         _visited: Optional[set] = None) -> list[dict]:
    """Walk the full provenance chain for an entity."""
    if _visited is None:
        _visited = set()
    if entity_id in _visited or max_depth <= 0:
        return []
    _visited.add(entity_id)

    node = graph.get_node(entity_id)
    if not node:
        return []

    chain = [{
        "entity_id": entity_id,
        "node_type": node.node_type,
        "provenance_type": node.get("provenance_type", "UNKNOWN"),
        "tainted": is_tainted(graph, entity_id),
    }]

    for edge in graph.get_incoming(entity_id, ET.PROMOTED_FROM):
        chain.extend(get_provenance_chain(graph, edge.source_id, max_depth - 1, _visited))
    for edge in graph.get_incoming(entity_id, ET.VERIFIES):
        chain.extend(get_provenance_chain(graph, edge.source_id, max_depth - 1, _visited))
    for edge in graph.get_outgoing(entity_id, ET.SOURCED_FROM):
        chain.extend(get_provenance_chain(graph, edge.target_id, max_depth - 1, _visited))

    return chain


# ============================================================
# Earned Terrain Protocol
# ============================================================

def promote_external_claim(graph: PropertyGraph,
                           claim_id: str, verification_id: str,
                           terrain_node_id: str, source_text: str,
                           embedding: Optional[list[float]] = None,
                           virtue_scores: Optional[list[float]] = None) -> Optional[Node]:
    """Promote external claim to terrain through verification.

    Terrain cites the verification, NOT the original claim.
    """
    claim = graph.get_node(claim_id)
    if not claim or claim.node_type != NT.EXTERNAL_CLAIM:
        return None

    verification = graph.get_node(verification_id)
    if not verification or verification.node_type not in (
        NT.WITNESS_VERIFICATION, NT.CROSS_VERIFICATION,
        NT.EMPIRICAL_VERIFICATION
    ):
        return None

    existing = graph.get_edges_between(verification_id, claim_id, ET.VERIFIES)
    if not existing:
        graph.add_edge(verification_id, claim_id, ET.VERIFIES)

    prov_type = verification.get("provenance_type", "WITNESS")
    terrain = graph.add_node(terrain_node_id, NT.TERRAIN_NODE, {
        "source_text": source_text,
        "embedding": embedding or [],
        "virtue_scores": virtue_scores or [0.5] * 9,
        "layer": "earned",
        "authority_weight": 0.4,
        "provenance_type": prov_type,
        "created_at": int(time.time() * 1000),
    })

    # PROMOTED_FROM points to verification
    graph.add_edge(terrain_node_id, verification_id, ET.PROMOTED_FROM)
    return terrain


# ============================================================
# ConflictSet Management
# ============================================================

class ConflictStatus(Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class Contender:
    mutation_hash: str
    author_key: str
    hlc: int
    base_root: str


def create_conflict_set(graph: PropertyGraph, community_id: str,
                        object_key: str, contenders: list[Contender],
                        contested_fields: list[str]) -> Node:
    """Create a ConflictSet for concurrent incompatible mutations."""
    conflict_id = f"conflict_{object_key}_{int(time.time()*1000)}"
    return graph.add_node(conflict_id, NT.CONFLICT_SET, {
        "community_id": community_id,
        "object_key": object_key,
        "contested_fields": contested_fields,
        "status": ConflictStatus.OPEN.value,
        "created_at_entry": contenders[0].mutation_hash if contenders else "",
        "contender_count": len(contenders),
        "contenders": [vars(c) for c in contenders],
    })


def resolve_conflict(graph: PropertyGraph, conflict_id: str,
                     chosen_mutation_hash: str, resolver_key: str,
                     justification: str = "") -> bool:
    """Resolve a ConflictSet. Losers are superseded, not deleted."""
    conflict = graph.get_node(conflict_id)
    if not conflict or conflict.node_type != NT.CONFLICT_SET:
        return False
    if conflict.get("status") != ConflictStatus.OPEN.value:
        return False

    conflict.set("status", ConflictStatus.RESOLVED.value)
    conflict.set("resolved_by", resolver_key)
    conflict.set("chosen_mutation", chosen_mutation_hash)
    conflict.set("resolved_at", int(time.time() * 1000))
    conflict.set("justification", justification)
    return True


def create_challenge(graph: PropertyGraph, target_mutation_hash: str,
                     challenger_key: str, condition_id: str,
                     proof_blob: str = "") -> Node:
    """Challenge a mutation. Attaches quarantine flag, does NOT retract."""
    flag_id = f"quarantine_{target_mutation_hash[:16]}_{int(time.time()*1000)}"
    return graph.add_node(flag_id, NT.QUARANTINE_FLAG, {
        "target_mutation": target_mutation_hash,
        "condition_id": condition_id,
        "status": "ASSERTED_VALID",
        "attached_by": challenger_key,
        "proof_blob": proof_blob,
        "created_at": int(time.time() * 1000),
    })


def get_open_conflicts(graph: PropertyGraph, community_id: str) -> list[Node]:
    return [n for n in graph.get_nodes_by_type(NT.CONFLICT_SET)
            if n.get("community_id") == community_id
            and n.get("status") == ConflictStatus.OPEN.value]


def get_quarantine_flags(graph: PropertyGraph, target_mutation_hash: str) -> list[Node]:
    return [n for n in graph.get_nodes_by_type(NT.QUARANTINE_FLAG)
            if n.get("target_mutation") == target_mutation_hash]

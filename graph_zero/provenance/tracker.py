"""
Graph Zero Provenance Tracker

Tracks where knowledge came from, how it was verified, and
whether it's been tainted by unverified inference.

The provenance DAG is the immune system of the knowledge base.
If an upstream node gets tainted, everything downstream is suspect.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from graph_zero.graph.backend import PropertyGraph, Node, Edge
from graph_zero.graph.schema import NT, ET, VERIFIED_PROVENANCE, TAINTED_PROVENANCE


# ============================================================
# Provenance Types
# ============================================================

class ProvenanceType(Enum):
    BEDROCK = "BEDROCK"                      # foundational, pre-loaded
    WITNESS = "WITNESS"                      # human attestation
    CROSS_VERIFIED = "CROSS_VERIFIED"        # 2+ AI models from different providers
    EMPIRICAL = "EMPIRICAL"                  # sensor data
    COMMUNITY_CONSENSUS = "COMMUNITY_CONSENSUS"
    INFERENCE = "INFERENCE"                  # AI-derived, NOT citable
    SOURCE_UNVERIFIED = "SOURCE_UNVERIFIED"  # unchecked external


# ============================================================
# Taint Detection
# ============================================================

@dataclass
class TaintResult:
    """Result of checking a node for provenance taint."""
    node_id: str
    is_tainted: bool
    taint_sources: list[str]    # IDs of tainted upstream nodes
    taint_depth: int            # how far back taint was found
    provenance_chain: list[str] # provenance types along the chain


def check_taint(graph: PropertyGraph, node_id: str,
                max_depth: int = 10) -> TaintResult:
    """Walk the provenance DAG upstream to detect taint.

    A node is tainted if ANY upstream ancestor in the provenance
    chain has a tainted provenance type (INFERENCE or SOURCE_UNVERIFIED).
    """
    node = graph.get_node(node_id)
    if not node:
        return TaintResult(node_id=node_id, is_tainted=True,
                           taint_sources=[], taint_depth=0,
                           provenance_chain=["MISSING"])

    # BFS upstream through provenance edges
    visited = {node_id}
    queue = [(node_id, 0)]
    taint_sources = []
    provenance_chain = []
    taint_depth = 0

    while queue:
        current_id, depth = queue.pop(0)
        if depth > max_depth:
            continue

        current = graph.get_node(current_id)
        if not current:
            continue

        prov = current.get("provenance_type", "")
        if prov:
            provenance_chain.append(prov)

        if prov in TAINTED_PROVENANCE:
            taint_sources.append(current_id)
            taint_depth = max(taint_depth, depth)

        # Walk upstream: SOURCED_FROM, PROMOTED_FROM, SUPERSEDES
        for edge_type in [ET.SOURCED_FROM, ET.PROMOTED_FROM, ET.SUPERSEDES]:
            for edge in graph.get_outgoing(current_id, edge_type):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, depth + 1))

        # Also check incoming verification edges (the verifiers)
        for edge in graph.get_incoming(current_id, ET.VERIFIES):
            verifier = graph.get_node(edge.source_id)
            if verifier and verifier.get("provenance_type", "") in TAINTED_PROVENANCE:
                taint_sources.append(edge.source_id)
                taint_depth = max(taint_depth, depth + 1)

    return TaintResult(
        node_id=node_id,
        is_tainted=len(taint_sources) > 0,
        taint_sources=taint_sources,
        taint_depth=taint_depth,
        provenance_chain=provenance_chain,
    )


# ============================================================
# Authority Closure
# ============================================================

@dataclass
class AuthorityClosure:
    """Result of computing the authority closure for a terrain node."""
    node_id: str
    total_authority: float
    verification_count: int
    verification_types: dict[str, int]  # type -> count
    is_promotable: bool                 # meets threshold for promotion


def compute_authority(graph: PropertyGraph, node_id: str,
                      promotion_threshold: float = 0.7) -> AuthorityClosure:
    """Compute the authority closure for a terrain node.

    Authority = base_weight + sum(verification_weights)
    A node can be promoted to a higher terrain layer when its
    accumulated authority exceeds the threshold.
    """
    node = graph.get_node(node_id)
    if not node or node.node_type != NT.TERRAIN_NODE:
        return AuthorityClosure(node_id=node_id, total_authority=0.0,
                                verification_count=0, verification_types={},
                                is_promotable=False)

    base_weight = node.get("authority_weight", 0.0)
    verification_types: dict[str, int] = {}
    verification_weight = 0.0

    # Count verifications
    for edge in graph.get_incoming(node_id, ET.VERIFIES):
        verifier = graph.get_node(edge.source_id)
        if not verifier:
            continue
        v_type = verifier.node_type
        verification_types[v_type] = verification_types.get(v_type, 0) + 1

        # Weight by verification type
        weights = {
            NT.WITNESS_VERIFICATION: 0.3,
            NT.CROSS_VERIFICATION: 0.2,
            NT.EMPIRICAL_VERIFICATION: 0.25,
        }
        verification_weight += weights.get(v_type, 0.1)

    # Also count support edges
    support_count = len(graph.get_incoming(node_id, ET.SUPPORTED_BY))
    verification_weight += support_count * 0.05

    total = base_weight + verification_weight
    total = min(1.0, total)

    count = sum(verification_types.values())

    return AuthorityClosure(
        node_id=node_id,
        total_authority=total,
        verification_count=count,
        verification_types=verification_types,
        is_promotable=total >= promotion_threshold,
    )


# ============================================================
# Promote (layer transition)
# ============================================================

def can_promote(graph: PropertyGraph, node_id: str,
                target_layer: str) -> tuple[bool, str]:
    """Check if a terrain node can be promoted to a higher layer.

    Rules:
    - earned → community: requires authority >= 0.6 and no taint
    - community → climate: requires authority >= 0.8 and no taint
    - climate → bedrock: not possible (bedrock is immutable)
    - Anything tainted: cannot promote
    """
    taint = check_taint(graph, node_id)
    if taint.is_tainted:
        return False, f"Node is tainted by: {', '.join(taint.taint_sources[:3])}"

    node = graph.get_node(node_id)
    if not node:
        return False, "Node not found"

    current_layer = node.get("layer", "")
    authority = compute_authority(graph, node_id)

    layer_thresholds = {
        ("earned", "community"): 0.6,
        ("community", "climate"): 0.8,
    }

    key = (current_layer, target_layer)
    if key not in layer_thresholds:
        return False, f"Cannot promote from {current_layer} to {target_layer}"

    threshold = layer_thresholds[key]
    if authority.total_authority < threshold:
        return False, (f"Authority {authority.total_authority:.2f} "
                       f"below threshold {threshold}")

    return True, "Promotion allowed"


# ============================================================
# Declassification Protocol
# ============================================================

def declassify_inference(graph: PropertyGraph, inference_node_id: str,
                         verification_node_id: str) -> tuple[bool, str]:
    """Attempt to declassify an INFERENCE node by attaching verification.

    An inference can be "promoted" from INFERENCE to a verified type
    if a proper verification node (WITNESS, CROSS_VERIFIED, EMPIRICAL)
    is attached to it.
    """
    inference = graph.get_node(inference_node_id)
    if not inference:
        return False, "Inference node not found"
    if inference.get("provenance_type") != "INFERENCE":
        return False, "Node is not INFERENCE type"

    verifier = graph.get_node(verification_node_id)
    if not verifier:
        return False, "Verification node not found"

    valid_types = {NT.WITNESS_VERIFICATION, NT.CROSS_VERIFICATION,
                   NT.EMPIRICAL_VERIFICATION}
    if verifier.node_type not in valid_types:
        return False, f"Invalid verification type: {verifier.node_type}"

    # Attach verification
    graph.add_edge(verification_node_id, inference_node_id, ET.VERIFIES, {
        "declassification": True,
    })

    # Map verifier type to provenance type
    type_map = {
        NT.WITNESS_VERIFICATION: "WITNESS",
        NT.CROSS_VERIFICATION: "CROSS_VERIFIED",
        NT.EMPIRICAL_VERIFICATION: "EMPIRICAL",
    }
    new_type = type_map.get(verifier.node_type, "INFERENCE")
    inference.set("provenance_type", new_type)
    inference.set("declassified_from", "INFERENCE")

    return True, f"Declassified to {new_type}"

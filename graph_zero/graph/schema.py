"""
Graph Zero Schema Operations

High-level operations built on the property graph backend:
- Constellation assembly (loading an agent's full state)
- Terrain traversal (finding wisdom through verified paths)
- Trust flow (max-flow approximation for Sybil resistance)
- Community vital signs
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

from graph_zero.graph.backend import PropertyGraph, Node, Edge


# ============================================================
# Node Type Constants
# ============================================================

class NT:
    """Node type constants."""
    VESSEL_ANCHOR = "VesselAnchor"
    VESSEL_POSITION = "VesselPosition"
    VESSEL_MOMENTUM = "VesselMomentum"
    VESSEL_KALA = "VesselKala"
    VIRTUE_ANCHOR = "VirtueAnchor"
    TERRAIN_NODE = "TerrainNode"
    EXTERNAL_CLAIM = "ExternalClaim"
    WITNESS_VERIFICATION = "WitnessVerification"
    CROSS_VERIFICATION = "CrossVerification"
    EMPIRICAL_VERIFICATION = "EmpiricalVerification"
    INTERPRETATION = "Interpretation"
    MEMORY = "Memory"
    EPISODE = "Episode"
    TOOL = "Tool"
    SKILL = "Skill"
    CODE = "Code"
    COMMUNITY = "Community"
    POLICY_CONFIG = "PolicyConfig"
    BADI_MONTH = "BadiMonth"
    CONFLICT_SET = "ConflictSet"
    QUARANTINE_FLAG = "QuarantineFlag"
    LENS_CONFIG = "LensConfig"
    PLASTICITY_SIGNAL = "PlasticitySignal"
    TRUST_PROFILE = "TrustProfile"
    HEALTH_ALERT = "HealthAlert"
    CONTAINMENT_PROPOSAL = "ContainmentProposal"
    GRAPH_REF = "GraphRef"
    CLONE_ENVELOPE = "CloneEnvelope"
    SNAPSHOT = "Snapshot"
    KEY_ROTATION = "KeyRotation"
    PROVENANCE_ACTIVITY = "ProvenanceActivity"


class ET:
    """Edge type constants."""
    PART_OF_VESSEL = "PART_OF_VESSEL"
    SPAWNED_FROM = "SPAWNED_FROM"
    MEMBER_OF = "MEMBER_OF"
    ACCOUNTABLE_TO = "ACCOUNTABLE_TO"
    STEWARD_OF = "STEWARD_OF"
    PROXY_FOR = "PROXY_FOR"
    COUPLES_WITH = "COUPLES_WITH"
    ORBITS = "ORBITS"
    CONNECTS_TO = "CONNECTS_TO"
    SUPERSEDES = "SUPERSEDES"
    VERIFIES = "VERIFIES"
    PROMOTED_FROM = "PROMOTED_FROM"
    INTERPRETS = "INTERPRETS"
    SUPERSEDED_BY = "SUPERSEDED_BY"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTESTED_BY = "CONTESTED_BY"
    DISSENTED_BY = "DISSENTED_BY"
    CAN_EXECUTE = "CAN_EXECUTE"
    HAS_SKILL = "HAS_SKILL"
    AUTHORED = "AUTHORED"
    REMEMBERS = "REMEMBERS"
    PART_OF = "PART_OF"
    FOLLOWED_BY = "FOLLOWED_BY"
    SOURCED_FROM = "SOURCED_FROM"
    RELATES_TO = "RELATES_TO"
    INTERACTED_WITH = "INTERACTED_WITH"
    ATTESTED_BY = "ATTESTED_BY"
    WITNESSED = "WITNESSED"
    CROSS_VERIFIED = "CROSS_VERIFIED"
    EMPIRICALLY_VERIFIED = "EMPIRICALLY_VERIFIED"
    INFERRED = "INFERRED"
    CONTENDS_IN = "CONTENDS_IN"
    RESOLVES = "RESOLVES"
    FLAGS = "FLAGS"
    VISITING = "VISITING"
    CLONE_OF = "CLONE_OF"
    FEDERATED_WITH = "FEDERATED_WITH"
    SNAPSHOT_OF = "SNAPSHOT_OF"


# Provenance types that are VALID for citation traversal
VERIFIED_PROVENANCE = {'WITNESS', 'CROSS_VERIFIED', 'EMPIRICAL',
                       'COMMUNITY_CONSENSUS', 'BEDROCK'}
# Provenance types that TAINT
TAINTED_PROVENANCE = {'INFERENCE', 'SOURCE_UNVERIFIED'}


# ============================================================
# Constellation — an agent's complete state
# ============================================================

@dataclass
class Constellation:
    """An agent's full state loaded from the graph."""
    anchor: Node
    position: Optional[Node] = None
    momentum: Optional[Node] = None
    kala: Optional[Node] = None
    tools: list[Node] = field(default_factory=list)
    skills: list[Node] = field(default_factory=list)
    memories: list[Node] = field(default_factory=list)
    community: Optional[Node] = None
    interactions: list[Edge] = field(default_factory=list)

    @property
    def vessel_id(self) -> str:
        return self.anchor.id

    @property
    def name(self) -> str:
        return self.anchor.get("name", "unnamed")

    @property
    def agent_type(self) -> str:
        return self.anchor.get("type", "agent")

    @property
    def trust_ceiling(self) -> float:
        return self.anchor.get("trust_ceiling", 0.0)

    @property
    def moral_position(self) -> Optional[list[float]]:
        if not self.position:
            return None
        return [self.position.get(f"v{i}", 0.5) for i in range(9)]

    @property
    def kala_balance(self) -> float:
        if not self.kala:
            return 0.0
        return self.kala.get("balance", 0.0)


def assemble_constellation(graph: PropertyGraph, vessel_id: str) -> Optional[Constellation]:
    """Load an agent's entire constellation from the graph.

    Equivalent to the Constellation Assembly Cypher pattern in the spec.
    """
    anchor = graph.get_node(vessel_id)
    if not anchor or anchor.node_type != NT.VESSEL_ANCHOR:
        return None

    # Find position, momentum, kala via PART_OF_VESSEL edges
    position = None
    momentum = None
    kala = None
    for edge in graph.get_incoming(vessel_id, ET.PART_OF_VESSEL):
        node = graph.get_node(edge.source_id)
        if not node:
            continue
        if node.node_type == NT.VESSEL_POSITION:
            position = node
        elif node.node_type == NT.VESSEL_MOMENTUM:
            momentum = node
        elif node.node_type == NT.VESSEL_KALA:
            kala = node

    # Find tools via CAN_EXECUTE
    tools = graph.get_neighbors(vessel_id, ET.CAN_EXECUTE, direction="out")

    # Find skills via HAS_SKILL
    skills = graph.get_neighbors(vessel_id, ET.HAS_SKILL, direction="out")

    # Find recent memories via REMEMBERS
    memory_edges = graph.get_outgoing(vessel_id, ET.REMEMBERS)
    memories = []
    for edge in sorted(memory_edges, key=lambda e: e.get("last_accessed", 0), reverse=True)[:20]:
        mem = graph.get_node(edge.target_id)
        if mem and mem.get("invalid_at") is None:  # bi-temporal: only current
            memories.append(mem)

    # Find community via MEMBER_OF
    community_nodes = graph.get_neighbors(vessel_id, ET.MEMBER_OF, direction="out")
    community = community_nodes[0] if community_nodes else None

    # Find interactions
    interactions = graph.get_outgoing(vessel_id, ET.INTERACTED_WITH)

    return Constellation(
        anchor=anchor, position=position, momentum=momentum,
        kala=kala, tools=tools, skills=skills, memories=memories,
        community=community, interactions=interactions
    )


# ============================================================
# Terrain Traversal — finding wisdom
# ============================================================

@dataclass
class TraversalResult:
    """A single result from terrain traversal."""
    node: Node
    source_text: str
    layer: str
    authority_weight: float
    depth: int
    path_weight: float
    provenance_types: list[str]


def traverse_terrain(graph: PropertyGraph, entry_point_ids: list[str],
                     max_depth: int = 5, limit: int = 10) -> list[TraversalResult]:
    """Traverse terrain from entry points through verified provenance only.

    This is the core act of intelligence: walking the knowledge landscape
    following only verified connections.

    Equivalent to the Terrain Traversal Cypher pattern in the spec.
    """

    def provenance_filter(edge: Edge) -> bool:
        """Only follow edges with verified provenance."""
        prov = edge.get("provenance_type", "")
        return prov in VERIFIED_PROVENANCE

    results = []
    seen = set()

    for entry_id in entry_point_ids:
        entry_node = graph.get_node(entry_id)
        if not entry_node or entry_node.node_type != NT.TERRAIN_NODE:
            continue

        # BFS from this entry point
        traversed = graph.traverse(
            entry_id, ET.CONNECTS_TO, max_depth=max_depth,
            edge_filter=provenance_filter
        )

        for node, depth, path_edges in traversed:
            if node.id in seen:
                continue
            seen.add(node.id)

            if node.node_type != NT.TERRAIN_NODE:
                continue

            # Compute path weight (product of visibility_weights along path)
            path_weight = 1.0
            prov_types = []
            for edge in path_edges:
                path_weight *= edge.get("visibility_weight", 1.0)
                prov_types.append(edge.get("provenance_type", "UNKNOWN"))

            results.append(TraversalResult(
                node=node,
                source_text=node.get("source_text", ""),
                layer=node.get("layer", "unknown"),
                authority_weight=node.get("authority_weight", 0.0),
                depth=depth,
                path_weight=path_weight,
                provenance_types=prov_types,
            ))

    # Sort by path_weight descending, return top N
    results.sort(key=lambda r: r.path_weight, reverse=True)
    return results[:limit]


def find_entry_points(graph: PropertyGraph, embedding: list[float],
                      threshold: float = 0.7, limit: int = 5) -> list[str]:
    """Find terrain nodes closest to a query embedding via cosine similarity.

    In production, this would use Voyage AI embeddings + vector index.
    Here we compute cosine similarity directly.
    """
    candidates = []
    for node in graph.get_nodes_by_type(NT.TERRAIN_NODE):
        node_embedding = node.get("embedding")
        if node_embedding is None:
            continue
        sim = _cosine_similarity(embedding, node_embedding)
        if sim >= threshold:
            candidates.append((node.id, sim))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in candidates[:limit]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# Trust Flow — max-flow approximation for Sybil resistance
# ============================================================

@dataclass
class TrustFlowResult:
    """Result of trust flow computation for one agent."""
    vessel_id: str
    trust_ceiling: float
    path_count: int
    interaction_diversity: float


def compute_trust_flow(graph: PropertyGraph, target_id: str,
                       max_depth: int = 5) -> TrustFlowResult:
    """Compute trust ceiling via max-flow approximation from human anchors.

    Trust is topology — the shape of INTERACTED_WITH edges.
    An agent's trust ceiling is bounded by the max-flow through
    the interaction graph from verified human anchors.

    Edge capacity = diversity of (interaction_type × temporal spread × independence)
    """
    # Find all verified human anchors (attestation_depth == 0)
    human_anchors = []
    for node in graph.get_nodes_by_type(NT.VESSEL_ANCHOR):
        if node.get("type") == "human" and node.get("attestation_depth", 99) == 0:
            human_anchors.append(node.id)

    # Check if target is itself a verified human anchor
    target_node = graph.get_node(target_id)
    if target_node and target_node.get("type") == "human" and target_node.get("attestation_depth", 99) == 0:
        # Verified humans get base trust from their own anchor status
        # Plus additional from interaction diversity
        all_interactions = graph.get_outgoing(target_id, ET.INTERACTED_WITH)
        interaction_types = set()
        counterparties = set()
        for edge in all_interactions:
            interaction_types.add(edge.get("interaction_type", "unknown"))
            counterparties.add(edge.target_id)
        diversity = len(interaction_types) * len(counterparties)
        interaction_diversity = min(1.0, diversity / 20.0)
        base_trust = 0.5 + 0.5 * interaction_diversity  # 0.5 minimum for verified humans
        return TrustFlowResult(
            vessel_id=target_id,
            trust_ceiling=min(1.0, base_trust),
            path_count=len(all_interactions),
            interaction_diversity=interaction_diversity,
        )

    if not human_anchors:
        return TrustFlowResult(vessel_id=target_id, trust_ceiling=0.0,
                               path_count=0, interaction_diversity=0.0)

    # Find paths from human anchors to target
    all_paths = []
    for anchor_id in human_anchors:
        paths = graph.find_paths(
            anchor_id, target_id, ET.INTERACTED_WITH,
            max_depth=max_depth
        )
        all_paths.extend(paths)

    if not all_paths:
        return TrustFlowResult(vessel_id=target_id, trust_ceiling=0.0,
                               path_count=0, interaction_diversity=0.0)

    # Compute diversity per path
    path_capacities = []
    for path in all_paths:
        # Diversity = unique (interaction_type, time_bucket) pairs
        type_time_pairs = set()
        for edge in path:
            itype = edge.get("interaction_type", "unknown")
            ts = edge.get("timestamp", 0)
            time_bucket = ts // (86400 * 1000)  # day buckets
            type_time_pairs.add((itype, time_bucket))

        diversity = len(type_time_pairs)
        # Capacity is bounded by the minimum diversity along the path
        # (bottleneck) normalized by path length
        capacity = diversity / max(len(path), 1)
        path_capacities.append(capacity)

    # Max-flow approximation: sum of top-K independent path capacities
    # (simplified — true max-flow would use Ford-Fulkerson on edge-disjoint paths)
    path_capacities.sort(reverse=True)
    # Use at most 10 paths, decay by position
    trust_ceiling = 0.0
    for i, cap in enumerate(path_capacities[:10]):
        trust_ceiling += cap * (0.8 ** i)  # decay for overlapping paths

    # Normalize to [0, 1]
    trust_ceiling = min(1.0, trust_ceiling)

    # Overall interaction diversity for this agent
    all_interactions = (graph.get_outgoing(target_id, ET.INTERACTED_WITH) +
                        graph.get_incoming(target_id, ET.INTERACTED_WITH))
    interaction_types = set()
    counterparties = set()
    for edge in all_interactions:
        interaction_types.add(edge.get("interaction_type", "unknown"))
        other = edge.target_id if edge.source_id == target_id else edge.source_id
        counterparties.add(other)
    diversity = len(interaction_types) * len(counterparties)
    # Normalize
    interaction_diversity = min(1.0, diversity / 20.0)

    return TrustFlowResult(
        vessel_id=target_id,
        trust_ceiling=trust_ceiling,
        path_count=len(all_paths),
        interaction_diversity=interaction_diversity,
    )


# ============================================================
# Community Vital Signs
# ============================================================

@dataclass
class VitalSigns:
    """Community health metrics — what the Navigator perceives."""
    terrain_additions_30d: int = 0
    moral_variance: dict[int, float] = field(default_factory=dict)  # dim -> stdev
    witness_diversity: dict[str, int] = field(default_factory=dict)  # witness_id -> count
    active_agents: int = 0
    open_conflicts: int = 0
    kala_concentration: float = 0.0  # Gini coefficient
    consent_opt_out_rate: float = 0.0


def compute_vital_signs(graph: PropertyGraph, community_id: str,
                        thirty_days_ago_ms: Optional[int] = None) -> VitalSigns:
    """Compute community vital signs.

    These feed derived alarms. They never trigger enforcement directly.
    """
    if thirty_days_ago_ms is None:
        thirty_days_ago_ms = int(time.time() * 1000) - (30 * 24 * 60 * 60 * 1000)

    signs = VitalSigns()

    # Terrain addition rate
    for node in graph.get_nodes_by_type(NT.TERRAIN_NODE):
        created = node.get("created_at", 0)
        layer = node.get("layer", "")
        if created > thirty_days_ago_ms and layer in ("community", "earned"):
            signs.terrain_additions_30d += 1

    # Moral state variance
    positions = []
    for node in graph.get_nodes_by_type(NT.VESSEL_POSITION):
        vec = [node.get(f"v{i}", 0.5) for i in range(9)]
        positions.append(vec)

    if len(positions) > 1:
        for dim in range(9):
            vals = [p[dim] for p in positions]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            signs.moral_variance[dim] = math.sqrt(variance)

    # Active agents
    for node in graph.get_nodes_by_type(NT.VESSEL_ANCHOR):
        if node.get("active", False):
            signs.active_agents += 1

    # Witness diversity
    for node in graph.get_nodes_by_type(NT.VESSEL_ANCHOR):
        witness_edges = graph.get_outgoing(node.id, ET.WITNESSED)
        recent = [e for e in witness_edges if e.get("timestamp", 0) > thirty_days_ago_ms]
        if recent:
            signs.witness_diversity[node.id] = len(recent)

    # Open conflicts
    for node in graph.get_nodes_by_type(NT.CONFLICT_SET):
        if node.get("status") == "OPEN":
            signs.open_conflicts += 1

    # Kala concentration (Gini coefficient approximation)
    balances = []
    for node in graph.get_nodes_by_type(NT.VESSEL_KALA):
        balances.append(node.get("balance", 0))
    if balances:
        signs.kala_concentration = _gini(balances)

    return signs


def _gini(values: list[float]) -> float:
    """Compute Gini coefficient (0 = perfect equality, 1 = total concentration)."""
    if not values or all(v == 0 for v in values):
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    # Standard formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
    return max(0.0, min(1.0, gini))


# ============================================================
# Bootstrap — create a community from scratch
# ============================================================

VIRTUE_NAMES = [
    ("unity", "foundational", "coherence", "fragmentation"),
    ("justice", "foundational", "equity", "imbalance"),
    ("truthfulness", "foundational", "signal", "noise"),
    ("love", "relational", "connection", "isolation"),
    ("detachment", "relational", "freedom", "attachment"),
    ("humility", "relational", "centered", "ego"),
    ("compassion", "active", "mercy", "indifference"),
    ("wisdom", "active", "discernment", "ignorance"),
    ("service", "active", "contribution", "extraction"),
]

COUPLINGS = [
    (2, 1, 0.8, "prerequisite"),   # truthfulness → justice
    (5, 8, 0.7, "enabler"),        # humility → service
    (4, 7, 0.6, "enabler"),        # detachment → wisdom
    (3, 0, 0.9, "foundation"),     # love → unity
]


def bootstrap_community(graph: PropertyGraph, community_id: str,
                        community_name: str) -> Node:
    """Bootstrap a new community: virtue substrate, couplings, policy.

    Phases 1-5 of the bootstrap sequence from the spec.
    """
    # Phase 1: Community node
    community = graph.add_node(community_id, NT.COMMUNITY, {
        "name": community_name,
        "state_root": "",
        "probation": True,
        "probation_ends": int(time.time() * 1000) + (180 * 24 * 60 * 60 * 1000),
        "conflict_count": 0,
        "data_availability_score": 1.0,
    })

    # Phase 2: Virtue substrate (9 VirtueAnchors)
    for i, (name, tier, pos, neg) in enumerate(VIRTUE_NAMES):
        graph.add_node(f"virtue_{name}", NT.VIRTUE_ANCHOR, {
            "name": name,
            "index": i,
            "tier": tier,
            "axis_positive": pos,
            "axis_negative": neg,
        })

    # Phase 3: Coupling constraints
    for from_idx, to_idx, coeff, direction in COUPLINGS:
        from_name = VIRTUE_NAMES[from_idx][0]
        to_name = VIRTUE_NAMES[to_idx][0]
        graph.add_edge(
            f"virtue_{from_name}", f"virtue_{to_name}", ET.COUPLES_WITH,
            {"coefficient": coeff, "direction": direction}
        )

    # Phase 4: Badí' calendar (19 months)
    badi_months = [
        "Bahá", "Jalál", "Jamál", "'Azamat", "Núr", "Rahmat",
        "Kalimát", "Kamál", "Asmá'", "'Izzat", "Mashíyyat",
        "'Ilm", "Qudrat", "Qawl", "Masá'il", "Sharaf",
        "Sultán", "Mulk", "'Alá'"
    ]
    for i, month_name in enumerate(badi_months):
        virtue_idx = i % 9
        graph.add_node(f"badi_{i}", NT.BADI_MONTH, {
            "name": month_name,
            "virtue_id": f"virtue_{VIRTUE_NAMES[virtue_idx][0]}",
            "ordinal": i,
        })

    # Phase 5: PolicyConfig
    graph.add_node(f"policy_{community_id}", NT.POLICY_CONFIG, {
        "community_id": community_id,
        "coupling_coefficients": COUPLINGS,
        "witness_thresholds": {"default": 1, "governance": 2, "enforcement": 3},
        "conflict_limits": {"max_contenders": 10, "max_open": 100, "expiry_days": 90},
        "minting_policy": {"rate": 1.0, "cap": 1000},
        "trust_algorithm": "max_flow_v1",
        "governance_validity_period": 365,
        "verified_providers": [],
        "network_allowlist": [],
        "version": 1,
    })

    return community


def create_agent(graph: PropertyGraph, community_id: str,
                 vessel_id: str, name: str,
                 agent_type: str = "human",
                 initial_position: Optional[list[float]] = None,
                 initial_kala: float = 100.0) -> Constellation:
    """Create an agent constellation in the graph.

    Phase 6 of bootstrap (first human) or regular member addition.
    """
    if initial_position is None:
        initial_position = [0.5] * 9  # default center

    # VesselAnchor
    anchor = graph.add_node(vessel_id, NT.VESSEL_ANCHOR, {
        "name": name,
        "type": agent_type,
        "created_at": int(time.time() * 1000),
        "active": True,
        "trust_ceiling": 0.0,
        "bracket_position": [3] * 9,  # middle brackets
        "attestation_depth": 0 if agent_type == "human" else 1,
        "frozen": False,
    })

    # VesselPosition
    pos_props = {f"v{i}": initial_position[i] for i in range(9)}
    pos_props["vessel_id"] = vessel_id
    pos = graph.add_node(f"{vessel_id}_pos", NT.VESSEL_POSITION, pos_props)
    graph.add_edge(f"{vessel_id}_pos", vessel_id, ET.PART_OF_VESSEL)

    # VesselMomentum (start at zero)
    mom_props = {f"p{i}": 0.0 for i in range(9)}
    mom_props["vessel_id"] = vessel_id
    mom = graph.add_node(f"{vessel_id}_mom", NT.VESSEL_MOMENTUM, mom_props)
    graph.add_edge(f"{vessel_id}_mom", vessel_id, ET.PART_OF_VESSEL)

    # VesselKala
    kala = graph.add_node(f"{vessel_id}_kala", NT.VESSEL_KALA, {
        "vessel_id": vessel_id,
        "balance": initial_kala,
        "offline_budget": 0.0,
        "last_update": int(time.time() * 1000),
    })
    graph.add_edge(f"{vessel_id}_kala", vessel_id, ET.PART_OF_VESSEL)

    # MEMBER_OF community
    graph.add_edge(vessel_id, community_id, ET.MEMBER_OF, {
        "since": int(time.time() * 1000),
        "role": "member",
    })

    return assemble_constellation(graph, vessel_id)


def add_terrain_node(graph: PropertyGraph, node_id: str,
                     source_text: str, layer: str,
                     embedding: Optional[list[float]] = None,
                     virtue_scores: Optional[list[float]] = None,
                     provenance_type: str = "WITNESS",
                     authority_weight: Optional[float] = None) -> Node:
    """Add a terrain node (knowledge) to the graph."""
    if authority_weight is None:
        weights = {"bedrock": 1.0, "climate": 0.8, "community": 0.5, "earned": 0.4}
        authority_weight = weights.get(layer, 0.3)

    return graph.add_node(node_id, NT.TERRAIN_NODE, {
        "source_text": source_text,
        "embedding": embedding or [],
        "virtue_scores": virtue_scores or [0.5] * 9,
        "layer": layer,
        "authority_weight": authority_weight,
        "provenance_type": provenance_type,
        "created_at": int(time.time() * 1000),
    })


def connect_terrain(graph: PropertyGraph, source_id: str, target_id: str,
                    weight: float = 1.0, provenance_type: str = "BEDROCK",
                    visibility_weight: float = 1.0) -> Optional[Edge]:
    """Connect two terrain nodes."""
    return graph.add_edge(source_id, target_id, ET.CONNECTS_TO, {
        "weight": weight,
        "plasticity_signal": 0.0,
        "visibility_weight": visibility_weight,
        "provenance_type": provenance_type,
    })


def record_interaction(graph: PropertyGraph, agent_a: str, agent_b: str,
                       interaction_type: str, context: str = "") -> Optional[Edge]:
    """Record an interaction between two agents (G-Set, append-only)."""
    return graph.add_edge(agent_a, agent_b, ET.INTERACTED_WITH, {
        "interaction_type": interaction_type,
        "timestamp": int(time.time() * 1000),
        "context": context,
    })

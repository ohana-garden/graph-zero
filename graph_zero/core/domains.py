"""
Graph Zero Protected Domains

Four protected zones. If a mutation's write-set touches any of these,
it needs a Class B certificate. No exceptions.

The gate computes the write-set and checks intersection. This is the
structural boundary — what you TOUCH determines what proof you need.
"""

from enum import Enum, auto
from typing import Set


class ProtectedDomain(Enum):
    """The four protected zones."""
    AUTHORITY = "protected_authority"         # citable terrain, witness edges, commitments
    GOVERNANCE = "protected_governance"       # policy, constraints, weights
    VISIBILITY = "protected_visibility"       # ranking, display, lens defaults
    ENFORCEMENT = "protected_enforcement"     # quarantine, throttles, containment


# Fields that belong to each domain.
# The gate uses this to compute which domains a write-set intersects.
DOMAIN_FIELDS: dict[ProtectedDomain, Set[str]] = {
    ProtectedDomain.AUTHORITY: {
        "terrain_node", "witness_verification", "cross_verification",
        "empirical_verification", "interpretation", "witness_edge",
        "cross_verified_edge", "empirically_verified_edge",
        "authority_weight", "provenance_type", "promote_payload",
        "terrain_connects_to", "promoted_from", "bedrock_node",
    },
    ProtectedDomain.GOVERNANCE: {
        "policy_config", "coupling_coefficients", "witness_thresholds",
        "conflict_limits", "minting_policy", "trust_algorithm",
        "governance_validity_period", "verified_providers",
        "network_allowlist", "virtue_anchor", "badi_month",
        "coupling_edge", "community_policy",
    },
    ProtectedDomain.VISIBILITY: {
        "lens_config", "inference_overlay", "inference_prominence",
        "visibility_weight", "branch_selections", "default_lens",
        "display_weight_edge",
    },
    ProtectedDomain.ENFORCEMENT: {
        "quarantine_flag", "containment_proposal_adoption",
        "terrain_freeze", "clone_health_tag", "kala_rate_update",
        "agent_capability_restriction",
    },
}

# Flatten for fast lookup: field -> domain
_FIELD_TO_DOMAIN: dict[str, ProtectedDomain] = {}
for domain, fields in DOMAIN_FIELDS.items():
    for field in fields:
        _FIELD_TO_DOMAIN[field] = domain


def classify_write_set(write_set: Set[str]) -> Set[ProtectedDomain]:
    """Given a mutation's write-set, return which protected domains it touches."""
    touched = set()
    for field in write_set:
        if field in _FIELD_TO_DOMAIN:
            touched.add(_FIELD_TO_DOMAIN[field])
    return touched


def is_protected(write_set: Set[str]) -> bool:
    """Quick check: does this write-set touch ANY protected domain?"""
    return bool(classify_write_set(write_set))

"""
Graph Zero Validity Predicate — "The One Gate"

Every single mutation passes through this code. Every computer in the
community runs the same gate with the same rules. If any computer
produces a different result, the Replay Theorem is violated and the
architecture has failed.

The gate is deterministic. No randomness. No model inference.
No heuristics. Same inputs → same outputs, always.
"""

from dataclasses import dataclass
from typing import Optional, Set
import hashlib

from graph_zero.core.identity import AgentIdentity
from graph_zero.core.mutation import (
    MutationEnvelope, MutationPayload, MutationType, GateResult, GateError,
    ImpactVector, CanonicalMutation, CertA, CertB, CertC, Certificate,
    UA_CLASS_TYPES, VerificationPath, VerifierType, ClaimType
)
from graph_zero.core.domains import (
    ProtectedDomain, classify_write_set, is_protected
)
from graph_zero.core.log import CanonicalLog


# ============================================================
# Community State (minimal in-memory representation)
# ============================================================

@dataclass
class CommunityState:
    """Minimal community state needed for gate evaluation.
    In production, this reads from FalkorDB.
    """
    community_id: str
    policy_version: int = 1
    governance_validity_days: int = 365
    max_contenders_per_object: int = 10
    max_open_conflicts: int = 100
    open_conflict_count: int = 0
    coupling_coefficients: list[tuple[int, int, float, str]] = None  # (from, to, coeff, direction)
    trust_epoch_seconds: int = 86400  # 24 hours
    registered_devices: dict[str, set[bytes]] = None  # agent_key_hash -> set of device_ids
    agent_trust_ceilings: dict[str, float] = None     # agent_key_hash -> trust_ceiling
    tool_trust_requirements: dict[str, float] = None   # tool_id -> trust_requirement
    last_trust_updates: dict[str, int] = None          # agent_key_hash -> last update timestamp
    moral_velocities: dict[str, float] = None          # agent_key_hash -> velocity magnitude
    max_moral_velocity: float = 0.5                    # max position change per epoch
    kala_balances: dict[str, float] = None             # agent_key_hash -> balance

    def __post_init__(self):
        self.coupling_coefficients = self.coupling_coefficients or [
            (2, 1, 0.8, 'prerequisite'),   # truthfulness → justice
            (5, 8, 0.7, 'enabler'),        # humility → service
            (4, 7, 0.6, 'enabler'),        # detachment → wisdom
            (3, 0, 0.9, 'foundation'),     # love → unity
        ]
        self.registered_devices = self.registered_devices or {}
        self.agent_trust_ceilings = self.agent_trust_ceilings or {}
        self.tool_trust_requirements = self.tool_trust_requirements or {}
        self.last_trust_updates = self.last_trust_updates or {}
        self.moral_velocities = self.moral_velocities or {}
        self.kala_balances = self.kala_balances or {}


# ============================================================
# The Gate
# ============================================================

class Gate:
    """The validity predicate. One gate. Every mutation. No exceptions."""

    def __init__(self, state: CommunityState, log: CanonicalLog):
        self.state = state
        self.log = log

    def evaluate(self, m: MutationEnvelope) -> GateResult:
        """Evaluate a mutation. Returns ACCEPT or REJECT with reason.

        Steps follow Section 5.4 of the spec exactly.
        Every step is deterministic. Every rejection has a code.
        """

        # Step 0: Basic integrity
        result = self._step0_integrity(m)
        if not result.accepted:
            return result

        # Step 1: Compute write-set and domains
        write_set = m.payload.write_set
        if not write_set:
            return GateResult(accepted=False, error=GateError.E_EMPTY,
                              error_detail="Empty write-set")

        domains_touched = classify_write_set(write_set)

        # Step 2: Certificate class checks
        result = self._step2_certificate(m, domains_touched)
        if not result.accepted:
            return result

        # Step 3: Provenance + taint constraints
        result = self._step3_provenance(m, domains_touched)
        if not result.accepted:
            return result

        # Step 4: Local invariants
        result = self._step4_invariants(m)
        if not result.accepted:
            return result

        # Step 5: Concurrency (handled at merge level, not here)

        # Step 6: Compute impact vector
        impact = self._step6_impact(m, write_set, domains_touched)

        # Step 7: Accept
        return GateResult(accepted=True, impact_vector=impact)

    # --------------------------------------------------------
    # Step 0: Basic integrity
    # --------------------------------------------------------

    def _step0_integrity(self, m: MutationEnvelope) -> GateResult:
        """Verify signature, seq, schema, identity chain."""

        # 0.1: Verify signature
        if not AgentIdentity.verify(m.author_key, m.mutation_hash, m.signature):
            return GateResult(accepted=False, error=GateError.E_SIG,
                              error_detail="Invalid envelope signature")

        # 0.2: Verify certificate signature
        if isinstance(m.certificate, CertA):
            if not m.certificate.verify(m.author_key, m.mutation_hash):
                return GateResult(accepted=False, error=GateError.E_SIG,
                                  error_detail="Invalid CertA signature")
        elif isinstance(m.certificate, CertB):
            if not m.certificate.verify(m.author_key, m.mutation_hash):
                return GateResult(accepted=False, error=GateError.E_SIG,
                                  error_detail="Invalid CertB signature or insufficient endorsements")
        elif isinstance(m.certificate, CertC):
            if not m.certificate.verify(m.author_key, m.mutation_hash):
                return GateResult(accepted=False, error=GateError.E_SIG,
                                  error_detail="Invalid CertC signature")

        # 0.3: Monotonic seq check
        if not self.log.check_seq(m.author_key, m.device_id, m.seq):
            return GateResult(accepted=False, error=GateError.E_SEQ,
                              error_detail=f"Sequence {m.seq} not monotonically increasing")

        # 0.4: Schema validation (basic)
        if m.payload.mutation_type is None:
            return GateResult(accepted=False, error=GateError.E_SCHEMA,
                              error_detail="Missing mutation_type")

        # 0.5: Device binding check
        key_hash = hashlib.sha256(m.author_key).hexdigest()
        if key_hash in self.state.registered_devices:
            if m.device_id not in self.state.registered_devices[key_hash]:
                # Allow DEVICE_BIND mutations from new devices
                if m.payload.mutation_type != MutationType.DEVICE_BIND:
                    return GateResult(accepted=False, error=GateError.E_DEVICE,
                                      error_detail="Device not registered for this agent")

        return GateResult(accepted=True)

    # --------------------------------------------------------
    # Step 2: Certificate class checks
    # --------------------------------------------------------

    def _step2_certificate(self, m: MutationEnvelope, domains: Set[ProtectedDomain]) -> GateResult:
        """Verify certificate class matches the domains being touched."""

        if domains:
            # Protected domains require Class B (or B+C)
            if isinstance(m.certificate, CertA):
                return GateResult(accepted=False, error=GateError.E_CERT,
                                  error_detail=f"Protected domains {[d.value for d in domains]} require Class B, got Class A")

            if isinstance(m.certificate, CertB):
                cert_b: CertB = m.certificate
                # Verify verification path is appropriate for claim type
                result = self._check_verification_path(m, cert_b)
                if not result.accepted:
                    return result

        else:
            # Unprotected — Class A is fine for Ua-class
            if m.payload.is_ua_class and not isinstance(m.certificate, CertA):
                pass  # Ua with B or C is fine (overqualified but valid)

        # Class C: must be a position update
        if isinstance(m.certificate, CertC):
            if m.payload.mutation_type != MutationType.POSITION_UPDATE:
                return GateResult(accepted=False, error=GateError.E_CERT,
                                  error_detail="Class C certificate only valid for POSITION_UPDATE")

            # Verify coupling constraints
            result = self._check_coupling_constraints(m.certificate)
            if not result.accepted:
                return result

        # Ua-class mutations MUST NOT touch protected domains
        if m.payload.is_ua_class and domains:
            return GateResult(accepted=False, error=GateError.E_CERT,
                              error_detail="Ua-class mutations cannot touch protected domains")

        return GateResult(accepted=True)

    def _check_verification_path(self, m: MutationEnvelope, cert: CertB) -> GateResult:
        """Verify the verification path is valid for the claim type."""
        claim_type = m.payload.claim_type

        if claim_type == ClaimType.WISDOM:
            # Wisdom requires WITNESS only
            if cert.verification_path != VerificationPath.WITNESS:
                return GateResult(accepted=False, error=GateError.E_CERT,
                                  error_detail="Wisdom claims require WITNESS verification path")
            # All endorsers must be HUMAN
            for e in cert.endorsements:
                if e.verifier_type != VerifierType.HUMAN:
                    return GateResult(accepted=False, error=GateError.E_CERT,
                                      error_detail="Wisdom verification requires human endorsers")

        # Governance and visibility mutations also require WITNESS
        gov_vis_types = {
            MutationType.POLICY_UPDATE, MutationType.COUPLING_UPDATE,
            MutationType.LENS_UPDATE, MutationType.CONTAINMENT_ADOPT,
            MutationType.TERRAIN_FREEZE, MutationType.KALA_RATE_UPDATE,
        }
        if m.payload.mutation_type in gov_vis_types:
            if cert.verification_path != VerificationPath.WITNESS:
                return GateResult(accepted=False, error=GateError.E_CERT,
                                  error_detail="Governance/visibility mutations require WITNESS verification")

        # CROSS_VERIFIED requires 2+ AI models from different providers
        if cert.verification_path == VerificationPath.CROSS_VERIFIED:
            ai_providers = set()
            for e in cert.endorsements:
                if e.verifier_type == VerifierType.AI_MODEL:
                    ai_providers.add(e.role)  # role stores provider name
            if len(ai_providers) < 2:
                return GateResult(accepted=False, error=GateError.E_CERT,
                                  error_detail="CROSS_VERIFIED requires 2+ AI models from different providers")

        return GateResult(accepted=True)

    def _check_coupling_constraints(self, cert: CertC) -> GateResult:
        """Verify moral position satisfies coupling constraints.
        Pure math — k inequality evaluations."""
        v = cert.new_vector
        if len(v) != 9:
            return GateResult(accepted=False, error=GateError.E_VARIANCE,
                              error_detail=f"Moral vector must have 9 dimensions, got {len(v)}")

        # Check bounds: all values must be in [0, 1]
        for i, val in enumerate(v):
            if not (0.0 <= val <= 1.0):
                return GateResult(accepted=False, error=GateError.E_VARIANCE,
                                  error_detail=f"Dimension {i} value {val} outside [0, 1]")

        # Check coupling constraints
        for (from_dim, to_dim, coeff, direction) in self.state.coupling_coefficients:
            if direction == 'prerequisite':
                # from is prerequisite for to: v[to] <= v[from] + (1 - coeff) * (1 - v[from])
                # Simplified: if from is low, to can't be high
                max_to = v[from_dim] + (1.0 - coeff) * (1.0 - v[from_dim])
                if v[to_dim] > max_to + 1e-6:  # floating point tolerance
                    return GateResult(accepted=False, error=GateError.E_VARIANCE,
                                      error_detail=f"Coupling violation: dim {to_dim} ({v[to_dim]:.3f}) > max allowed ({max_to:.3f}) given dim {from_dim} ({v[from_dim]:.3f})")
            elif direction == 'enabler':
                # Similar but softer constraint
                max_to = v[from_dim] + (1.0 - coeff * 0.5) * (1.0 - v[from_dim])
                if v[to_dim] > max_to + 1e-6:
                    return GateResult(accepted=False, error=GateError.E_VARIANCE,
                                      error_detail=f"Coupling violation: dim {to_dim} ({v[to_dim]:.3f}) > max allowed ({max_to:.3f}) given dim {from_dim} ({v[from_dim]:.3f})")
            elif direction == 'foundation':
                # Strongest: from must be >= to * coeff
                min_from = v[to_dim] * coeff
                if v[from_dim] < min_from - 1e-6:
                    return GateResult(accepted=False, error=GateError.E_VARIANCE,
                                      error_detail=f"Foundation violation: dim {from_dim} ({v[from_dim]:.3f}) < min required ({min_from:.3f}) given dim {to_dim} ({v[to_dim]:.3f})")

        return GateResult(accepted=True)

    # --------------------------------------------------------
    # Step 3: Provenance
    # --------------------------------------------------------

    def _step3_provenance(self, m: MutationEnvelope, domains: Set[ProtectedDomain]) -> GateResult:
        """Check provenance taint for authority-bearing mutations.
        In full implementation, this walks the provenance DAG.
        Here we check basic structural requirements."""

        if ProtectedDomain.AUTHORITY not in domains:
            return GateResult(accepted=True)

        # For terrain/promote mutations, check claim_type matches verification path
        if m.payload.mutation_type in {MutationType.PROMOTE, MutationType.TERRAIN_ADD}:
            if isinstance(m.certificate, CertB):
                # Already checked in step 2, but double-check taint
                if m.payload.data.get('provenance_type') in ('INFERENCE', 'SOURCE_UNVERIFIED'):
                    return GateResult(accepted=False, error=GateError.E_TAINT,
                                      error_detail="Cannot write INFERENCE or SOURCE_UNVERIFIED to protected_authority")

        return GateResult(accepted=True)

    # --------------------------------------------------------
    # Step 4: Local invariants
    # --------------------------------------------------------

    def _step4_invariants(self, m: MutationEnvelope) -> GateResult:
        """Check type-specific local invariants."""
        key_hash = hashlib.sha256(m.author_key).hexdigest()

        # Kala balance check
        if m.payload.mutation_type == MutationType.KALA_TIP:
            amount = m.payload.data.get('amount', 0)
            balance = self.state.kala_balances.get(key_hash, 0)
            if amount > balance:
                return GateResult(accepted=False, error=GateError.E_KALA,
                                  error_detail=f"Insufficient Kala: {amount} > {balance}")
            # Check balanced transition
            recipient = m.payload.data.get('recipient')
            if recipient is None:
                return GateResult(accepted=False, error=GateError.E_KALA,
                                  error_detail="Kala tip missing recipient")

        # Trust epoch throttling
        if m.payload.mutation_type == MutationType.TRUST_UPDATE:
            target = m.payload.data.get('target_agent')
            if target:
                last_update = self.state.last_trust_updates.get(target, 0)
                current_epoch = m.timestamp_hlc // (self.state.trust_epoch_seconds * 1000)
                last_epoch = last_update // (self.state.trust_epoch_seconds * 1000)
                if current_epoch == last_epoch and last_update > 0:
                    return GateResult(accepted=False, error=GateError.E_EPOCH,
                                      error_detail=f"Trust update for {target} already exists in current epoch")

        # Moral velocity check (for position updates)
        if m.payload.mutation_type == MutationType.POSITION_UPDATE:
            velocity = self.state.moral_velocities.get(key_hash, 0)
            if velocity > self.state.max_moral_velocity:
                return GateResult(accepted=False, error=GateError.E_VELOCITY,
                                  error_detail=f"Moral velocity {velocity:.3f} exceeds max {self.state.max_moral_velocity}")

        # Conflict cap check
        if self.state.open_conflict_count >= self.state.max_open_conflicts:
            if not m.payload.is_ua_class and m.payload.mutation_type != MutationType.RESOLVE:
                return GateResult(accepted=False, error=GateError.E_CONFLICT_CAP,
                                  error_detail=f"Community has {self.state.open_conflict_count} open conflicts (max {self.state.max_open_conflicts})")

        return GateResult(accepted=True)

    # --------------------------------------------------------
    # Step 6: Impact vector
    # --------------------------------------------------------

    def _step6_impact(self, m: MutationEnvelope, write_set: Set[str],
                      domains: Set[ProtectedDomain]) -> ImpactVector:
        """Compute deterministic impact vector from local inputs only."""

        # Write scope
        scope = 0  # self
        if m.payload.data.get('recipient'):
            scope = 1  # dyad
        if m.payload.data.get('community_wide'):
            scope = 3  # community
        if m.payload.data.get('federation'):
            scope = 4  # federation

        # Domains touched as bitfield
        domain_bits = 0
        if ProtectedDomain.AUTHORITY in domains: domain_bits |= 1
        if ProtectedDomain.GOVERNANCE in domains: domain_bits |= 2
        if ProtectedDomain.VISIBILITY in domains: domain_bits |= 4
        if ProtectedDomain.ENFORCEMENT in domains: domain_bits |= 8

        # Kala delta
        kala_delta = 0
        if m.payload.mutation_type == MutationType.KALA_TIP:
            kala_delta = -int(m.payload.data.get('amount', 0))
        elif m.payload.mutation_type == MutationType.KALA_MINT:
            kala_delta = int(m.payload.data.get('amount', 0))

        # Witness count
        witness_count = 0
        if isinstance(m.certificate, CertB):
            witness_count = len(m.certificate.endorsements)

        return ImpactVector(
            write_scope=scope,
            domains_touched=domain_bits,
            kala_delta=kala_delta,
            provenance_depth=m.payload.data.get('provenance_depth', 0),
            witness_count=witness_count,
            is_challenge=(m.payload.mutation_type == MutationType.CHALLENGE),
            conflict_count=0,
            target_agent_count=len(m.payload.data.get('target_agents', []))
        )


# ============================================================
# Convenience: process a mutation end-to-end
# ============================================================

def process_mutation(gate: Gate, log: CanonicalLog,
                     envelope: MutationEnvelope) -> tuple[GateResult, Optional[CanonicalMutation]]:
    """Evaluate a mutation through the gate and append to log if accepted.

    Returns (result, canonical_mutation_or_None).
    """
    result = gate.evaluate(envelope)

    if not result.accepted:
        return result, None

    # Append to log
    canonical = log.append(
        envelope=envelope,
        impact_vector=result.impact_vector,
        write_set=envelope.payload.write_set,
        conflict_ref=result.conflict_ref,
    )

    return result, canonical

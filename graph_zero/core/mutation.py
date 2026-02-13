"""
Graph Zero Mutation System

Every change to the community's shared graph is a mutation.
Every mutation carries a certificate proving its legitimacy.
The gate accepts or rejects based on deterministic rules.

MutationEnvelope: proposed change (not yet accepted)
CanonicalMutation: accepted change (in the permanent log)
Certificate: proof of legitimacy (A, B, or C)
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Set

from graph_zero.core.identity import AgentIdentity


# ============================================================
# Mutation Types
# ============================================================

class MutationType(Enum):
    """All mutation types in the system."""
    # Class A (Ua / everyday)
    MESSAGE = "message"
    CHECK_IN = "check_in"
    SURPLUS_DECLARE = "surplus_declare"
    NEED_DECLARE = "need_declare"
    STATUS_UPDATE = "status_update"
    LOCATION_UPDATE = "location_update"
    CAPABILITY_DECLARE = "capability_declare"
    SENSOR_READING = "sensor_reading"
    KALA_TIP = "kala_tip"
    MEMORY_ADD = "memory_add"
    TRUST_UPDATE = "trust_update"
    INTERACTION_RECORD = "interaction_record"
    HEALTH_ALERT = "health_alert"
    CONTAINMENT_PROPOSE = "containment_propose"
    DEVICE_BIND = "device_bind"

    # Class B (authority-bearing)
    TERRAIN_ADD = "terrain_add"
    TERRAIN_MODIFY = "terrain_modify"
    WITNESS_ATTEST = "witness_attest"
    CROSS_VERIFY = "cross_verify"
    EMPIRICAL_VERIFY = "empirical_verify"
    PROMOTE = "promote"
    INTERPRET = "interpret"
    POLICY_UPDATE = "policy_update"
    COUPLING_UPDATE = "coupling_update"
    LENS_UPDATE = "lens_update"
    RESOLVE = "resolve"
    KEY_ROTATE = "key_rotate"
    CONSTELLATION_CREATE = "constellation_create"
    SNAPSHOT_CREATE = "snapshot_create"
    SKILL_INSTALL = "skill_install"
    CONTAINMENT_ADOPT = "containment_adopt"
    TERRAIN_FREEZE = "terrain_freeze"
    CLONE_HEALTH_TAG = "clone_health_tag"
    KALA_RATE_UPDATE = "kala_rate_update"
    KALA_MINT = "kala_mint"

    # Class C (moral position update)
    POSITION_UPDATE = "position_update"

    # Conflict resolution
    CHALLENGE = "challenge"
    DEVICE_RESOLVE = "device_resolve"


# Which mutation types are Class A (Ua, everyday, CRDT-mergeable)
UA_CLASS_TYPES = {
    MutationType.MESSAGE, MutationType.CHECK_IN,
    MutationType.SURPLUS_DECLARE, MutationType.NEED_DECLARE,
    MutationType.STATUS_UPDATE, MutationType.LOCATION_UPDATE,
    MutationType.CAPABILITY_DECLARE, MutationType.SENSOR_READING,
    MutationType.KALA_TIP, MutationType.MEMORY_ADD,
    MutationType.TRUST_UPDATE, MutationType.INTERACTION_RECORD,
    MutationType.HEALTH_ALERT, MutationType.CONTAINMENT_PROPOSE,
    MutationType.DEVICE_BIND,
}


# ============================================================
# CRDT Semantics
# ============================================================

class CRDTType(Enum):
    """CRDT merge strategies for Ua-class mutations."""
    G_SET = "g_set"            # grow-only set (messages, check-ins, etc.)
    LWW_REGISTER = "lww"       # last-writer-wins (status, location)
    MV_REGISTER = "mv"         # multi-value (surplus, needs)
    PN_COUNTER = "pn_counter"  # positive-negative counter (kala budgets)


# Default CRDT type for each Ua mutation type
MUTATION_CRDT: dict[MutationType, CRDTType] = {
    MutationType.MESSAGE: CRDTType.G_SET,
    MutationType.CHECK_IN: CRDTType.G_SET,
    MutationType.SURPLUS_DECLARE: CRDTType.MV_REGISTER,
    MutationType.NEED_DECLARE: CRDTType.MV_REGISTER,
    MutationType.STATUS_UPDATE: CRDTType.LWW_REGISTER,
    MutationType.LOCATION_UPDATE: CRDTType.LWW_REGISTER,
    MutationType.CAPABILITY_DECLARE: CRDTType.G_SET,
    MutationType.SENSOR_READING: CRDTType.G_SET,
    MutationType.KALA_TIP: CRDTType.PN_COUNTER,
    MutationType.MEMORY_ADD: CRDTType.G_SET,
    MutationType.TRUST_UPDATE: CRDTType.LWW_REGISTER,
    MutationType.INTERACTION_RECORD: CRDTType.G_SET,
    MutationType.HEALTH_ALERT: CRDTType.G_SET,
    MutationType.CONTAINMENT_PROPOSE: CRDTType.G_SET,
    MutationType.DEVICE_BIND: CRDTType.G_SET,
}


# ============================================================
# Verification Paths
# ============================================================

class VerificationPath(Enum):
    """How a Class B mutation was verified."""
    WITNESS = "witness"                      # human attestation
    CROSS_VERIFIED = "cross_verified"        # 2+ AI models from different providers
    EMPIRICAL = "empirical"                  # sensor data
    COMMUNITY_CONSENSUS = "community_consensus"  # t-of-n agents


class VerifierType(Enum):
    """Type of entity that endorsed a mutation."""
    HUMAN = "human"
    AI_MODEL = "ai_model"
    SENSOR = "sensor"
    CONSENSUS = "consensus"


class ClaimType(Enum):
    """What kind of claim is being made."""
    KNOWLEDGE = "knowledge"     # factual, testable
    WISDOM = "wisdom"           # interpretive, values-laden
    OBSERVATION = "observation" # sensor-derived, time-bound


# ============================================================
# Certificates
# ============================================================

@dataclass
class Endorsement:
    """A single endorsement on a Class B certificate."""
    verifier_key: bytes        # public key of verifier
    signature: bytes           # verifier signs the mutation hash
    role: str                  # role of the verifier
    verifier_type: VerifierType


@dataclass
class CertA:
    """Class A certificate — just a signature. For everyday Ua-class mutations."""
    author_sig: bytes

    @staticmethod
    def create(identity: AgentIdentity, mutation_hash: bytes) -> 'CertA':
        return CertA(author_sig=identity.sign(mutation_hash))

    def verify(self, public_key: bytes, mutation_hash: bytes) -> bool:
        return AgentIdentity.verify(public_key, mutation_hash, self.author_sig)


@dataclass
class CertB:
    """Class B certificate — signature + verified endorsements.
    For authority-bearing mutations touching protected domains."""
    author_sig: bytes
    endorsements: list[Endorsement]
    verification_path: VerificationPath
    threshold: int             # t-of-n required

    @staticmethod
    def create(identity: AgentIdentity, mutation_hash: bytes,
               endorsements: list[Endorsement],
               verification_path: VerificationPath,
               threshold: int = 1) -> 'CertB':
        return CertB(
            author_sig=identity.sign(mutation_hash),
            endorsements=endorsements,
            verification_path=verification_path,
            threshold=threshold
        )

    def verify(self, public_key: bytes, mutation_hash: bytes) -> bool:
        # Check author signature
        if not AgentIdentity.verify(public_key, mutation_hash, self.author_sig):
            return False
        # Check endorsement count meets threshold
        valid_endorsements = 0
        for e in self.endorsements:
            if AgentIdentity.verify(e.verifier_key, mutation_hash, e.signature):
                valid_endorsements += 1
        return valid_endorsements >= self.threshold


@dataclass
class CertC:
    """Class C certificate — moral position update.
    The proof IS the vector. Gate verifies coupling constraints."""
    author_sig: bytes
    new_vector: list[float]    # 9 floats (v0-v8)

    @staticmethod
    def create(identity: AgentIdentity, mutation_hash: bytes,
               new_vector: list[float]) -> 'CertC':
        # Sign mutation_hash + hash of vector
        vector_bytes = b''.join(float.hex(v).encode() for v in new_vector)
        combined = mutation_hash + hashlib.sha256(vector_bytes).digest()
        return CertC(
            author_sig=identity.sign(combined),
            new_vector=new_vector
        )

    def verify(self, public_key: bytes, mutation_hash: bytes) -> bool:
        vector_bytes = b''.join(float.hex(v).encode() for v in self.new_vector)
        combined = mutation_hash + hashlib.sha256(vector_bytes).digest()
        return AgentIdentity.verify(public_key, combined, self.author_sig)


# Union type for certificates
Certificate = CertA | CertB | CertC


# ============================================================
# Mutation Payload
# ============================================================

@dataclass
class MutationPayload:
    """The content of a mutation — what's actually being changed."""
    mutation_type: MutationType
    data: dict[str, Any]       # type-specific payload data
    write_set: Set[str]        # fields this mutation modifies
    crdt_type: Optional[CRDTType] = None  # for Ua-class mutations
    claim_type: Optional[ClaimType] = None  # for authority-bearing mutations

    @property
    def is_ua_class(self) -> bool:
        return self.mutation_type in UA_CLASS_TYPES


# ============================================================
# Mutation Envelope (proposed change, not yet accepted)
# ============================================================

@dataclass
class MutationEnvelope:
    """A proposed state transition. Not yet accepted."""
    community_id: str
    author_key: bytes          # Ed25519 public key
    device_id: bytes           # 16 bytes
    seq: int                   # per-device monotonic sequence
    base_root: str             # author's view of StateRoot when created
    timestamp_hlc: int         # hybrid logical clock
    payload: MutationPayload
    certificate: Certificate
    signature: bytes           # Sig(author_key, H(envelope_without_sig))

    @property
    def mutation_hash(self) -> bytes:
        """Deterministic hash of the envelope (excluding signature)."""
        h = hashlib.sha256()
        h.update(self.community_id.encode())
        h.update(self.author_key)
        h.update(self.device_id)
        h.update(self.seq.to_bytes(8, 'big'))
        h.update(self.base_root.encode())
        h.update(self.timestamp_hlc.to_bytes(8, 'big'))
        h.update(self.payload.mutation_type.value.encode())
        # Sort write_set for determinism
        for field in sorted(self.payload.write_set):
            h.update(field.encode())
        return h.digest()

    @staticmethod
    def compute_hash_for(community_id: str, author_key: bytes, device_id: bytes,
                         seq: int, base_root: str, timestamp_hlc: int,
                         payload: MutationPayload) -> bytes:
        """Pre-compute the mutation hash for endorsement signing."""
        h = hashlib.sha256()
        h.update(community_id.encode())
        h.update(author_key)
        h.update(device_id)
        h.update(seq.to_bytes(8, 'big'))
        h.update(base_root.encode())
        h.update(timestamp_hlc.to_bytes(8, 'big'))
        h.update(payload.mutation_type.value.encode())
        for field in sorted(payload.write_set):
            h.update(field.encode())
        return h.digest()

    @staticmethod
    def create(identity: AgentIdentity, community_id: str, device_id: bytes,
               seq: int, base_root: str, payload: MutationPayload,
               endorsements: Optional[list[Endorsement]] = None,
               verification_path: Optional[VerificationPath] = None,
               threshold: int = 1,
               new_vector: Optional[list[float]] = None,
               timestamp_hlc: Optional[int] = None) -> 'MutationEnvelope':
        """Create a signed mutation envelope with the appropriate certificate."""
        if timestamp_hlc is None:
            timestamp_hlc = int(time.time() * 1000)

        # Build a temporary envelope to get the mutation hash
        temp = MutationEnvelope(
            community_id=community_id,
            author_key=identity.public_key,
            device_id=device_id,
            seq=seq,
            base_root=base_root,
            timestamp_hlc=timestamp_hlc,
            payload=payload,
            certificate=CertA(author_sig=b''),  # placeholder
            signature=b''
        )
        m_hash = temp.mutation_hash

        # Create the appropriate certificate
        if payload.mutation_type == MutationType.POSITION_UPDATE and new_vector:
            cert = CertC.create(identity, m_hash, new_vector)
        elif payload.is_ua_class:
            cert = CertA.create(identity, m_hash)
        else:
            # Class B
            cert = CertB.create(
                identity, m_hash,
                endorsements=endorsements or [],
                verification_path=verification_path or VerificationPath.WITNESS,
                threshold=threshold
            )

        # Sign the full envelope
        sig = identity.sign(m_hash)

        return MutationEnvelope(
            community_id=community_id,
            author_key=identity.public_key,
            device_id=device_id,
            seq=seq,
            base_root=base_root,
            timestamp_hlc=timestamp_hlc,
            payload=payload,
            certificate=cert,
            signature=sig
        )


# ============================================================
# Impact Vector (derived by gate — never by submitter)
# ============================================================

@dataclass
class ImpactVector:
    """Fixed-schema struct computed by the gate for each accepted mutation.
    No model inference. No heuristics. Deterministic from local inputs only."""
    write_scope: int           # 0=self, 1=dyad, 2=cluster, 3=community, 4=federation
    domains_touched: int       # bitfield: authority|governance|visibility|enforcement
    kala_delta: int            # net Kala flow
    provenance_depth: int      # max depth of provenance DAG
    witness_count: int         # number of witness endorsements
    is_challenge: bool
    conflict_count: int        # ConflictSets created/modified
    target_agent_count: int    # how many agents affected


# ============================================================
# Canonical Mutation (accepted, in the permanent log)
# ============================================================

@dataclass
class CanonicalMutation:
    """An accepted mutation in the permanent log."""
    envelope: MutationEnvelope
    write_set: Set[str]
    impact_vector: ImpactVector
    conflict_ref: Optional[str] = None
    quarantine_flags: list[str] = field(default_factory=list)
    canonical_prev: str = ""   # prev canonical entry hash
    canonical_hash: str = ""   # this entry's hash

    def compute_hash(self) -> str:
        """Deterministic hash of this canonical entry."""
        h = hashlib.sha256()
        h.update(self.canonical_prev.encode())
        h.update(self.envelope.mutation_hash)
        h.update(str(self.impact_vector.write_scope).encode())
        h.update(str(self.impact_vector.domains_touched).encode())
        for field in sorted(self.write_set):
            h.update(field.encode())
        return h.hexdigest()


# ============================================================
# Gate Error Codes
# ============================================================

class GateError(Enum):
    """Deterministic error codes for gate rejections."""
    E_SIG = "E_SIG"             # signature invalid
    E_SEQ = "E_SEQ"             # replayed sequence number
    E_EMPTY = "E_EMPTY"         # empty write-set
    E_CERT = "E_CERT"           # wrong certificate class for domain
    E_TAINT = "E_TAINT"         # provenance tainted in authority path
    E_EPOCH = "E_EPOCH"         # trust update too soon
    E_VELOCITY = "E_VELOCITY"   # moral position changing too fast
    E_VARIANCE = "E_VARIANCE"   # moral position exceeding curvature
    E_CONFLICT_CAP = "E_CONFLICT_CAP"  # too many conflicts on object
    E_GOV_EXPIRED = "E_GOV_EXPIRED"    # governance decision expired
    E_KALA = "E_KALA"           # unbalanced Kala transfer
    E_TRUST = "E_TRUST"         # insufficient trust for tool
    E_DEVICE = "E_DEVICE"       # dual-device conflict
    E_SCHEMA = "E_SCHEMA"       # malformed payload


@dataclass
class GateResult:
    """Result of gate evaluation."""
    accepted: bool
    error: Optional[GateError] = None
    error_detail: str = ""
    impact_vector: Optional[ImpactVector] = None
    conflict_ref: Optional[str] = None

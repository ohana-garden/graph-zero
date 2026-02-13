"""Tests for Graph Zero: mutations, log, gate, certificates, domains.

Full pipeline: create identity → build mutation → gate evaluates → log appends.
"""

import sys, os, hashlib
sys.path.insert(0, '/home/claude')

from graph_zero.core.identity import generate_identity, AgentIdentity, DeviceBinding
from graph_zero.core.mutation import (
    MutationEnvelope, MutationPayload, MutationType, CRDTType,
    CertA, CertB, CertC, GateError, Endorsement,
    VerificationPath, VerifierType, ClaimType, ImpactVector
)
from graph_zero.core.domains import classify_write_set, ProtectedDomain, is_protected
from graph_zero.core.log import CanonicalLog
from graph_zero.core.gate import Gate, CommunityState, process_mutation


# ============================================================
# Helpers
# ============================================================

def make_community() -> tuple[CommunityState, CanonicalLog, Gate]:
    """Create a fresh community with gate."""
    state = CommunityState(community_id="test_community")
    log = CanonicalLog(community_id="test_community")
    gate = Gate(state, log)
    return state, log, gate

def make_agent(state: CommunityState) -> tuple:
    """Create an agent with registered device."""
    identity = generate_identity()
    device_id = os.urandom(16)
    key_hash = identity.key_hash
    state.registered_devices[key_hash] = {device_id}
    state.kala_balances[key_hash] = 100.0
    return identity, device_id, key_hash

def make_ua_mutation(identity, device_id, seq, mutation_type=MutationType.MESSAGE,
                     data=None, write_set=None) -> MutationEnvelope:
    """Create a simple Class A (Ua) mutation."""
    payload = MutationPayload(
        mutation_type=mutation_type,
        data=data or {"content": "hello"},
        write_set=write_set or {"message"},
        crdt_type=CRDTType.G_SET,
    )
    return MutationEnvelope.create(
        identity=identity,
        community_id="test_community",
        device_id=device_id,
        seq=seq,
        base_root="",
        payload=payload,
    )

def make_endorsement(identity, mutation_hash, role="witness",
                     verifier_type=VerifierType.HUMAN) -> Endorsement:
    """Create an endorsement from an identity."""
    return Endorsement(
        verifier_key=identity.public_key,
        signature=identity.sign(mutation_hash),
        role=role,
        verifier_type=verifier_type,
    )


# ============================================================
# Domain tests
# ============================================================

def test_domain_classification():
    """Write-sets are correctly classified into domains."""
    assert classify_write_set({"message"}) == set()
    assert classify_write_set({"terrain_node"}) == {ProtectedDomain.AUTHORITY}
    assert classify_write_set({"policy_config"}) == {ProtectedDomain.GOVERNANCE}
    assert classify_write_set({"lens_config"}) == {ProtectedDomain.VISIBILITY}
    assert classify_write_set({"quarantine_flag"}) == {ProtectedDomain.ENFORCEMENT}
    # Multiple domains
    assert classify_write_set({"terrain_node", "policy_config"}) == {
        ProtectedDomain.AUTHORITY, ProtectedDomain.GOVERNANCE
    }
    assert not is_protected({"message", "status_update"})
    assert is_protected({"terrain_node"})
    print("  ✓ domain_classification")


# ============================================================
# Class A (Ua) mutation tests
# ============================================================

def test_class_a_accepted():
    """Simple Class A mutation passes the gate."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    m = make_ua_mutation(identity, device_id, seq=1)
    result, canonical = process_mutation(gate, log, m)

    assert result.accepted, f"Should accept: {result.error_detail}"
    assert canonical is not None
    assert log.length == 1
    assert log.verify_chain()
    print("  ✓ class_a_accepted")


def test_bad_signature_rejected():
    """Forged signature is rejected at Step 0."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    m = make_ua_mutation(identity, device_id, seq=1)
    # Tamper with signature
    m = MutationEnvelope(
        community_id=m.community_id, author_key=m.author_key,
        device_id=m.device_id, seq=m.seq, base_root=m.base_root,
        timestamp_hlc=m.timestamp_hlc, payload=m.payload,
        certificate=m.certificate, signature=b'\x00' * 64
    )
    result, _ = process_mutation(gate, log, m)
    assert not result.accepted
    assert result.error == GateError.E_SIG
    print("  ✓ bad_signature_rejected")


def test_seq_replay_rejected():
    """Replayed sequence number is rejected."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    m1 = make_ua_mutation(identity, device_id, seq=1)
    result1, _ = process_mutation(gate, log, m1)
    assert result1.accepted

    # Same seq again
    m2 = make_ua_mutation(identity, device_id, seq=1)
    result2, _ = process_mutation(gate, log, m2)
    assert not result2.accepted
    assert result2.error == GateError.E_SEQ
    print("  ✓ seq_replay_rejected")


def test_seq_monotonic():
    """Sequence numbers must increase."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    for seq in [1, 2, 3, 5, 10]:  # gaps are OK, must be increasing
        m = make_ua_mutation(identity, device_id, seq=seq)
        result, _ = process_mutation(gate, log, m)
        assert result.accepted, f"seq {seq} should be accepted"

    assert log.length == 5
    print("  ✓ seq_monotonic")


def test_empty_write_set_rejected():
    """Empty write-set is rejected."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    payload = MutationPayload(
        mutation_type=MutationType.MESSAGE,
        data={"content": "hello"},
        write_set=set(),  # empty!
        crdt_type=CRDTType.G_SET,
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
    )
    result, _ = process_mutation(gate, log, m)
    assert not result.accepted
    assert result.error == GateError.E_EMPTY
    print("  ✓ empty_write_set_rejected")


# ============================================================
# Class A touching protected domains → rejected
# ============================================================

def test_ua_protected_rejected():
    """Ua-class mutation trying to touch protected domain is rejected."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    # Try to write terrain_node (protected_authority) with Class A
    payload = MutationPayload(
        mutation_type=MutationType.MESSAGE,
        data={"content": "sneaky"},
        write_set={"terrain_node"},  # protected!
        crdt_type=CRDTType.G_SET,
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
    )
    result, _ = process_mutation(gate, log, m)
    assert not result.accepted
    assert result.error == GateError.E_CERT
    print("  ✓ ua_protected_rejected")


# ============================================================
# Class B (authority-bearing) tests
# ============================================================

def test_class_b_with_witness():
    """Class B mutation with valid witness endorsement passes."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)
    witness = generate_identity()

    payload = MutationPayload(
        mutation_type=MutationType.TERRAIN_ADD,
        data={"source_text": "Taro grows best in wetland conditions",
              "provenance_type": "WITNESS"},
        write_set={"terrain_node"},
        claim_type=ClaimType.KNOWLEDGE,
    )

    # Fix timestamp and pre-compute hash
    import time as _time
    ts = int(_time.time() * 1000)
    m_hash = MutationEnvelope.compute_hash_for(
        "test_community", identity.public_key, device_id,
        1, "", ts, payload
    )

    endorsement = make_endorsement(witness, m_hash)
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
        endorsements=[endorsement],
        verification_path=VerificationPath.WITNESS,
        threshold=1,
        timestamp_hlc=ts,
    )

    result, canonical = process_mutation(gate, log, m)
    assert result.accepted, f"Should accept: {result.error} - {result.error_detail}"
    assert canonical is not None
    assert result.impact_vector.domains_touched & 1  # authority bit set
    print("  ✓ class_b_with_witness")


def test_class_b_wisdom_needs_human():
    """Wisdom claims require WITNESS path with human endorsers."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)
    ai_endorser = generate_identity()

    payload = MutationPayload(
        mutation_type=MutationType.INTERPRET,
        data={"resolution_text": "We should prioritize water rights",
              "provenance_type": "WITNESS"},
        write_set={"interpretation"},
        claim_type=ClaimType.WISDOM,
    )

    import time as _time
    ts = int(_time.time() * 1000)
    m_hash = MutationEnvelope.compute_hash_for(
        "test_community", identity.public_key, device_id,
        1, "", ts, payload
    )

    # Try with AI endorser — should fail
    endorsement = Endorsement(
        verifier_key=ai_endorser.public_key,
        signature=ai_endorser.sign(m_hash),
        role="claude",
        verifier_type=VerifierType.AI_MODEL,  # not human!
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
        endorsements=[endorsement],
        verification_path=VerificationPath.WITNESS,
        threshold=1,
        timestamp_hlc=ts,
    )
    result, _ = process_mutation(gate, log, m)
    assert not result.accepted
    assert result.error == GateError.E_CERT
    assert "human" in result.error_detail.lower()
    print("  ✓ class_b_wisdom_needs_human")


# ============================================================
# Class C (moral position) tests
# ============================================================

def test_class_c_valid_position():
    """Valid moral position update passes coupling constraints."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    # Position where all constraints are satisfied
    # High truthfulness (v2=0.8) → high justice (v1=0.7) is OK (prerequisite met)
    # High humility (v5=0.7) → high service (v8=0.6) is OK
    # High detachment (v4=0.6) → moderate wisdom (v7=0.5) is OK
    # High love (v3=0.8) → high unity (v0=0.7) is OK (foundation met)
    vector = [0.7, 0.7, 0.8, 0.8, 0.6, 0.7, 0.5, 0.5, 0.6]

    payload = MutationPayload(
        mutation_type=MutationType.POSITION_UPDATE,
        data={"new_vector": vector},
        write_set={"vessel_position"},
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
        new_vector=vector,
    )
    result, canonical = process_mutation(gate, log, m)
    assert result.accepted, f"Should accept valid position: {result.error_detail}"
    print("  ✓ class_c_valid_position")


def test_class_c_coupling_violation():
    """Position violating coupling constraints is rejected."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    # Low truthfulness (v2=0.1) but high justice (v1=0.9)
    # Violates: truthfulness → justice (prerequisite, coeff 0.8)
    vector = [0.5, 0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    payload = MutationPayload(
        mutation_type=MutationType.POSITION_UPDATE,
        data={"new_vector": vector},
        write_set={"vessel_position"},
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
        new_vector=vector,
    )
    result, _ = process_mutation(gate, log, m)
    assert not result.accepted
    assert result.error == GateError.E_VARIANCE
    print("  ✓ class_c_coupling_violation")


def test_class_c_foundation_violation():
    """Position violating foundation constraint (love → unity) is rejected."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    # High unity (v0=0.9) but low love (v3=0.1)
    # Violates: love → unity (foundation, coeff 0.9)
    # Foundation check: v[from=3] >= v[to=0] * coeff → 0.1 >= 0.9 * 0.9 = 0.81 → FAIL
    vector = [0.9, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5]

    payload = MutationPayload(
        mutation_type=MutationType.POSITION_UPDATE,
        data={"new_vector": vector},
        write_set={"vessel_position"},
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
        new_vector=vector,
    )
    result, _ = process_mutation(gate, log, m)
    assert not result.accepted
    assert result.error == GateError.E_VARIANCE
    assert "foundation" in result.error_detail.lower()
    print("  ✓ class_c_foundation_violation")


# ============================================================
# Kala invariant tests
# ============================================================

def test_kala_insufficient_balance():
    """Kala tip exceeding balance is rejected."""
    state, log, gate = make_community()
    identity, device_id, key_hash = make_agent(state)
    state.kala_balances[key_hash] = 10.0

    payload = MutationPayload(
        mutation_type=MutationType.KALA_TIP,
        data={"amount": 50, "recipient": "someone"},
        write_set={"kala_balance"},
        crdt_type=CRDTType.PN_COUNTER,
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
    )
    result, _ = process_mutation(gate, log, m)
    assert not result.accepted
    assert result.error == GateError.E_KALA
    print("  ✓ kala_insufficient_balance")


def test_kala_valid_tip():
    """Kala tip within balance passes."""
    state, log, gate = make_community()
    identity, device_id, key_hash = make_agent(state)
    state.kala_balances[key_hash] = 100.0

    payload = MutationPayload(
        mutation_type=MutationType.KALA_TIP,
        data={"amount": 5, "recipient": "someone"},
        write_set={"kala_balance"},
        crdt_type=CRDTType.PN_COUNTER,
    )
    m = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
    )
    result, _ = process_mutation(gate, log, m)
    assert result.accepted, f"Should accept: {result.error_detail}"
    print("  ✓ kala_valid_tip")


# ============================================================
# Trust epoch throttling
# ============================================================

def test_trust_epoch_throttle():
    """Only one trust update per agent per epoch."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    # First update — should pass
    payload = MutationPayload(
        mutation_type=MutationType.TRUST_UPDATE,
        data={"target_agent": "agent_abc", "new_ceiling": 0.7},
        write_set={"trust_profile"},
        crdt_type=CRDTType.LWW_REGISTER,
    )
    m1 = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=1, base_root="", payload=payload,
    )
    result1, _ = process_mutation(gate, log, m1)
    assert result1.accepted

    # Record the update time in state
    state.last_trust_updates["agent_abc"] = m1.timestamp_hlc

    # Second update same epoch — should fail
    m2 = MutationEnvelope.create(
        identity=identity, community_id="test_community",
        device_id=device_id, seq=2, base_root="", payload=payload,
    )
    result2, _ = process_mutation(gate, log, m2)
    assert not result2.accepted
    assert result2.error == GateError.E_EPOCH
    print("  ✓ trust_epoch_throttle")


# ============================================================
# Log integrity tests
# ============================================================

def test_log_hash_chain():
    """Log maintains valid hash chain across multiple entries."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    for seq in range(1, 21):
        m = make_ua_mutation(identity, device_id, seq=seq,
                             data={"content": f"message {seq}"})
        result, _ = process_mutation(gate, log, m)
        assert result.accepted

    assert log.length == 20
    assert log.verify_chain()

    # Tamper test: modify an entry hash
    original_hash = log.entries[5].canonical_hash
    log.entries[5].canonical_hash = "tampered"
    assert not log.verify_chain()
    log.entries[5].canonical_hash = original_hash  # restore
    assert log.verify_chain()
    print("  ✓ log_hash_chain (20 entries)")


def test_log_state_root():
    """State root changes with each new entry."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    roots = set()
    for seq in range(1, 6):
        m = make_ua_mutation(identity, device_id, seq=seq)
        process_mutation(gate, log, m)
        roots.add(log.state_root.root_hash)

    assert len(roots) == 5  # each entry produces different state root
    print("  ✓ log_state_root")


def test_replay_theorem():
    """Two logs processing same mutations produce same state root.
    This is the foundational invariant."""
    state1, log1, gate1 = make_community()
    state2, log2, gate2 = make_community()
    identity, device_id, key_hash = make_agent(state1)
    # Register same agent in both states
    state2.registered_devices[key_hash] = {device_id}
    state2.kala_balances[key_hash] = 100.0

    # Create mutations (save them for replay)
    mutations = []
    for seq in range(1, 11):
        m = make_ua_mutation(identity, device_id, seq=seq,
                             data={"content": f"msg {seq}"})
        mutations.append(m)

    # Apply to both logs
    for m in mutations:
        r1, _ = process_mutation(gate1, log1, m)
        r2, _ = process_mutation(gate2, log2, m)
        assert r1.accepted == r2.accepted

    # REPLAY THEOREM: same log → same state root
    assert log1.state_root.root_hash == log2.state_root.root_hash
    assert log1.length == log2.length
    print("  ✓ replay_theorem (10 mutations)")


# ============================================================
# Impact vector tests
# ============================================================

def test_impact_vector():
    """Impact vector is computed correctly."""
    state, log, gate = make_community()
    identity, device_id, _ = make_agent(state)

    m = make_ua_mutation(identity, device_id, seq=1,
                         data={"content": "hello", "recipient": "bob"})
    result, _ = process_mutation(gate, log, m)

    assert result.accepted
    iv = result.impact_vector
    assert iv.write_scope == 1  # dyad (has recipient)
    assert iv.domains_touched == 0  # no protected domains
    assert iv.kala_delta == 0
    assert not iv.is_challenge
    print("  ✓ impact_vector")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("Testing Graph Zero core pipeline...\n")

    print("Domains:")
    test_domain_classification()

    print("\nClass A (Ua) mutations:")
    test_class_a_accepted()
    test_bad_signature_rejected()
    test_seq_replay_rejected()
    test_seq_monotonic()
    test_empty_write_set_rejected()
    test_ua_protected_rejected()

    print("\nClass B (authority-bearing):")
    test_class_b_with_witness()
    test_class_b_wisdom_needs_human()

    print("\nClass C (moral position):")
    test_class_c_valid_position()
    test_class_c_coupling_violation()
    test_class_c_foundation_violation()

    print("\nKala invariants:")
    test_kala_insufficient_balance()
    test_kala_valid_tip()

    print("\nTrust epochs:")
    test_trust_epoch_throttle()

    print("\nLog integrity:")
    test_log_hash_chain()
    test_log_state_root()
    test_replay_theorem()

    print("\nImpact vectors:")
    test_impact_vector()

    print("\n" + "=" * 50)
    print("ALL CORE TESTS PASSED ✓")
    print("=" * 50)

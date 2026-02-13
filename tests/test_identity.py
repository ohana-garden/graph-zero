"""Tests for Graph Zero identity system."""

import sys
sys.path.insert(0, '/home/claude')

from graph_zero.core.identity import (
    generate_identity, AgentIdentity, DeviceBinding, KeyRotation, KeyChain
)


def test_generate_identity():
    """Generate keypair, verify basic properties."""
    identity = generate_identity()
    assert len(identity.public_key) == 32
    assert identity.signing_key is not None
    assert len(identity.key_hash) == 64  # SHA-256 hex
    print("  ✓ generate_identity")


def test_sign_and_verify():
    """Sign a message, verify it succeeds."""
    identity = generate_identity()
    msg = b"test mutation payload"
    sig = identity.sign(msg)
    assert len(sig) == 64
    assert AgentIdentity.verify(identity.public_key, msg, sig)
    print("  ✓ sign_and_verify")


def test_bad_signature_fails():
    """Tampered message fails verification."""
    identity = generate_identity()
    msg = b"original message"
    sig = identity.sign(msg)
    # Tamper with message
    assert not AgentIdentity.verify(identity.public_key, b"tampered message", sig)
    # Wrong key
    other = generate_identity()
    assert not AgentIdentity.verify(other.public_key, msg, sig)
    print("  ✓ bad_signature_fails")


def test_public_only():
    """Public-only identity can verify but not sign."""
    identity = generate_identity()
    pub = identity.public_only()
    assert pub.signing_key is None
    assert pub.public_key == identity.public_key
    assert pub.key_hash == identity.key_hash
    # Verify works
    msg = b"test"
    sig = identity.sign(msg)
    assert AgentIdentity.verify(pub.public_key, msg, sig)
    # Sign fails
    try:
        pub.sign(b"nope")
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  ✓ public_only")


def test_device_binding():
    """Create and verify device binding."""
    identity = generate_identity()
    binding = DeviceBinding.create(identity)
    assert len(binding.device_id) == 16
    assert binding.agent_key_hash == identity.key_hash
    assert binding.verify(identity.public_key)
    # Wrong key fails
    other = generate_identity()
    assert not binding.verify(other.public_key)
    print("  ✓ device_binding")


def test_key_rotation():
    """Rotate key, verify chain."""
    old_identity = generate_identity()
    rotation, new_identity = KeyRotation.rotate(old_identity)
    assert rotation.verify()
    assert rotation.old_public_key == old_identity.public_key
    assert rotation.new_public_key == new_identity.public_key
    # New identity can sign
    msg = b"post-rotation message"
    sig = new_identity.sign(msg)
    assert AgentIdentity.verify(new_identity.public_key, msg, sig)
    print("  ✓ key_rotation")


def test_key_chain():
    """Key chain tracks valid keys through rotations."""
    id1 = generate_identity()
    chain = KeyChain(id1.public_key)
    assert chain.is_current_key(id1.public_key)
    assert chain.is_valid_key(id1.public_key)

    # Rotate
    rot1, id2 = KeyRotation.rotate(id1)
    assert chain.add_rotation(rot1)
    assert chain.is_current_key(id2.public_key)
    assert chain.is_valid_key(id1.public_key)  # old key still valid
    assert chain.is_valid_key(id2.public_key)

    # Rotate again
    rot2, id3 = KeyRotation.rotate(id2)
    assert chain.add_rotation(rot2)
    assert chain.is_current_key(id3.public_key)
    assert chain.is_valid_key(id1.public_key)
    assert chain.is_valid_key(id2.public_key)
    assert chain.is_valid_key(id3.public_key)

    # Bad rotation (wrong old key)
    random_id = generate_identity()
    bad_rot, _ = KeyRotation.rotate(random_id)
    assert not chain.add_rotation(bad_rot)  # old_key doesn't match current
    print("  ✓ key_chain")


def test_unique_identities():
    """Every generated identity is unique."""
    ids = [generate_identity() for _ in range(100)]
    keys = {i.key_hash for i in ids}
    assert len(keys) == 100
    print("  ✓ unique_identities (100)")


if __name__ == "__main__":
    print("Testing identity system...")
    test_generate_identity()
    test_sign_and_verify()
    test_bad_signature_fails()
    test_public_only()
    test_device_binding()
    test_key_rotation()
    test_key_chain()
    test_unique_identities()
    print("\nAll identity tests passed ✓")

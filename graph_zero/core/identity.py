"""
Graph Zero Identity System

Ed25519 keypairs for agent identity. Every agent has a keypair.
Every mutation is signed. Every signature is verified by the gate.

This is Step 0 of the gate — if the signature is bad, nothing else matters.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError


@dataclass(frozen=True)
class AgentIdentity:
    """An agent's cryptographic identity."""
    public_key: bytes          # 32 bytes Ed25519 public key
    signing_key: Optional[bytes] = field(default=None, repr=False)  # 64 bytes (private)

    @property
    def key_hash(self) -> str:
        """Hex-encoded SHA-256 of the public key. Used as agent ID."""
        return hashlib.sha256(self.public_key).hexdigest()

    def sign(self, message: bytes) -> bytes:
        """Sign a message. Returns 64-byte signature."""
        if self.signing_key is None:
            raise ValueError("Cannot sign without private key (read-only identity)")
        sk = SigningKey(self.signing_key[:32])  # nacl wants the seed (first 32 bytes)
        signed = sk.sign(message)
        return signed.signature  # 64 bytes

    @staticmethod
    def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify a signature against a public key. Returns True/False."""
        try:
            vk = VerifyKey(public_key)
            vk.verify(message, signature)
            return True
        except BadSignatureError:
            return False

    def public_only(self) -> 'AgentIdentity':
        """Return a copy without the signing key (for sharing)."""
        return AgentIdentity(public_key=self.public_key)


def generate_identity() -> AgentIdentity:
    """Generate a new Ed25519 keypair."""
    sk = SigningKey.generate()
    return AgentIdentity(
        public_key=bytes(sk.verify_key),
        signing_key=bytes(sk) + bytes(sk.verify_key)  # seed + pubkey = 64 bytes
    )


@dataclass
class DeviceBinding:
    """Binds a device to an agent identity."""
    device_id: bytes           # 16 bytes
    agent_key_hash: str        # SHA-256 of agent's public key
    bound_at: int              # HLC timestamp
    signature: bytes           # agent signs the binding

    @staticmethod
    def create(identity: AgentIdentity, device_id: Optional[bytes] = None) -> 'DeviceBinding':
        """Create a device binding for an identity."""
        if device_id is None:
            device_id = os.urandom(16)
        bound_at = int(time.time() * 1000)
        # Sign the binding data
        binding_data = device_id + identity.key_hash.encode() + bound_at.to_bytes(8, 'big')
        signature = identity.sign(binding_data)
        return DeviceBinding(
            device_id=device_id,
            agent_key_hash=identity.key_hash,
            bound_at=bound_at,
            signature=signature
        )

    def verify(self, public_key: bytes) -> bool:
        """Verify the device binding signature."""
        key_hash = hashlib.sha256(public_key).hexdigest()
        if key_hash != self.agent_key_hash:
            return False
        binding_data = self.device_id + self.agent_key_hash.encode() + self.bound_at.to_bytes(8, 'big')
        return AgentIdentity.verify(public_key, binding_data, self.signature)


@dataclass
class KeyRotation:
    """Records a key rotation: old key signs new key."""
    old_public_key: bytes
    new_public_key: bytes
    old_signs_new: bytes       # signature of new_public_key by old_key
    rotated_at: int

    @staticmethod
    def rotate(old_identity: AgentIdentity) -> tuple['KeyRotation', AgentIdentity]:
        """Rotate to a new keypair. Returns (rotation record, new identity)."""
        new_identity = generate_identity()
        rotated_at = int(time.time() * 1000)

        # Old key signs the new public key
        sig = old_identity.sign(new_identity.public_key)

        rotation = KeyRotation(
            old_public_key=old_identity.public_key,
            new_public_key=new_identity.public_key,
            old_signs_new=sig,
            rotated_at=rotated_at
        )
        return rotation, new_identity

    def verify(self) -> bool:
        """Verify that the old key legitimately signed the new key."""
        return AgentIdentity.verify(
            self.old_public_key,
            self.new_public_key,
            self.old_signs_new
        )


class KeyChain:
    """Tracks valid key chain for an agent (rotation history)."""

    def __init__(self, initial_key: bytes):
        self.current_key: bytes = initial_key
        self.rotations: list[KeyRotation] = []
        self._valid_keys: set[bytes] = {initial_key}

    def add_rotation(self, rotation: KeyRotation) -> bool:
        """Add a rotation. Returns False if invalid."""
        if rotation.old_public_key != self.current_key:
            return False
        if not rotation.verify():
            return False
        self.rotations.append(rotation)
        self.current_key = rotation.new_public_key
        self._valid_keys.add(rotation.new_public_key)
        return True

    def is_valid_key(self, key: bytes) -> bool:
        """Check if a key is in the valid chain."""
        return key in self._valid_keys

    def is_current_key(self, key: bytes) -> bool:
        """Check if a key is the current active key."""
        return key == self.current_key

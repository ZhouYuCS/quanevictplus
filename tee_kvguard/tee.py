"""CPU-simulated Trusted Execution Environment (TEE) enclave.

In a real confidential-computing deployment the TEE would be a hardware
enclave (e.g. Intel SGX / AMD SEV) on the CPU side. Here we simulate it on
plain CPU. The enclave is the *only* place that ever sees the plaintext
``scale`` tensor of the KV decomposition; everything else (centroids, assign,
residual_int) lives in the REE / GPU bank in plaintext.

The enclave provides two primitives:

* ``seal_scale``  -- encrypt + persist a scale tensor, returning an opaque ref
* ``reveal_scale`` -- decrypt + return the scale tensor, only to an authorized,
  attested caller

Sealing uses AES-GCM so the at-rest blob is confidential and tamper-evident;
the capability token models a successful remote-attestation handshake.
"""

from __future__ import annotations

import os
import secrets

import torch

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    _HAS_AESGCM = True
except Exception:  # pragma: no cover - pure-stdlib fallback
    _HAS_AESGCM = False


class EnclaveAttestation:
    """Simulated remote attestation measurement."""

    def __init__(self, measurement: bytes):
        self.measurement = measurement


class SealRef:
    """Opaque handle to a sealed scale blob stored inside the enclave."""

    def __init__(self, ref: str):
        self.ref = ref

    def __repr__(self) -> str:
        return f"SealRef({self.ref[:8]}...)"


def _serialize_tensor(t: torch.Tensor) -> bytes:
    import io
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return buf.getvalue()


def _deserialize_tensor(b: bytes) -> torch.Tensor:
    import io
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)


class _FallbackAEAD:
    """Minimal AEAD built from hashlib/hmac (only used without ``cryptography``)."""

    def __init__(self, key: bytes):
        self.key = key

    def _keystream(self, nonce: bytes, n: int) -> bytes:
        import hashlib
        out = b""
        ctr = 0
        while len(out) < n:
            out += hashlib.sha256(self.key + nonce + ctr.to_bytes(4, "big")).digest()
            ctr += 1
        return out[:n]

    def encrypt(self, nonce: bytes, data: bytes, aad: bytes) -> bytes:
        import hmac
        ks = self._keystream(nonce, len(data))
        ct = bytes(a ^ b for a, b in zip(data, ks))
        tag = hmac.new(self.key, aad + nonce + ct, "sha256").digest()
        return ct + tag

    def decrypt(self, nonce: bytes, data: bytes, aad: bytes) -> bytes:
        import hmac
        ct, tag = data[:-32], data[-32:]
        expect = hmac.new(self.key, aad + nonce + ct, "sha256").digest()
        if not hmac.compare_digest(tag, expect):
            raise ValueError("AEAD tag mismatch")
        ks = self._keystream(nonce, len(ct))
        return bytes(a ^ b for a, b in zip(ct, ks))


class TEEEnclave:
    """A CPU-side enclave that seals / reveals only the quantization scale."""

    _CODE_IDENTITY = "QuanEvict.KVGuard.TEE.v1"

    def __init__(self):
        if _HAS_AESGCM:
            self._key = AESGCM.generate_key(bit_length=256)
            self._aead = AESGCM(self._key)
        else:
            self._key = secrets.token_bytes(32)
            self._aead = _FallbackAEAD(self._key)

        self._blobs: dict[str, tuple[bytes, bytes, bytes]] = {}  # ref -> (nonce, ct, aad)
        self._callers: dict[str, str] = {}  # caller_id -> capability token
        self._resident = 0

    # -- attestation / caller management ---------------------------------- #
    def attest(self) -> EnclaveAttestation:
        import hashlib
        return EnclaveAttestation(hashlib.sha256(self._CODE_IDENTITY.encode()).digest())

    def register_caller(self, caller_id: str) -> str:
        cap = secrets.token_hex(16)
        self._callers[caller_id] = cap
        return cap

    def _check_caller(self, caller_id: str, capability: str) -> None:
        if self._callers.get(caller_id) != capability:
            raise PermissionError("TEE: caller is not attested / authorized")

    # -- seal / reveal ----------------------------------------------------- #
    def seal_scale(self, scale: torch.Tensor, caller_id: str, capability: str,
                   label: str = "") -> SealRef:
        self._check_caller(caller_id, capability)
        payload = _serialize_tensor(scale)
        nonce = os.urandom(12)
        aad = label.encode()
        ct = self._aead.encrypt(nonce, payload, aad)
        ref = secrets.token_hex(16)
        self._blobs[ref] = (nonce, ct, aad)
        self._resident += len(ct)
        return SealRef(ref)

    def reveal_scale(self, ref: SealRef, caller_id: str, capability: str) -> torch.Tensor:
        self._check_caller(caller_id, capability)
        nonce, ct, aad = self._blobs[ref.ref]
        payload = self._aead.decrypt(nonce, ct, aad)
        return _deserialize_tensor(payload)

    # -- introspection ------------------------------------------------------ #
    def resident_bytes(self) -> int:
        return self._resident

    def num_sealed(self) -> int:
        return len(self._blobs)

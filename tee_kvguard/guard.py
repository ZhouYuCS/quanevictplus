"""KV-Guard orchestration: ties the TEE (scale) and REE (center+residual)
together into a single protect / reconstruct flow.
"""

from __future__ import annotations

import torch

from .tee import TEEEnclave, SealRef
from .ree import REEGPUKVBank
from .quantize import decompose_kv, reconstruct_kv, kv_breakdown


class KVGuard:
    """Trusted orchestrator that knows how to split KV and route the parts.

    * protect     -> decompose KV; seal ``scale`` in TEE; write center+residual
                     to REE. Only a breakdown summary (sizes) is returned, never
                     the raw scale.
    * reconstruct -> pull the sealed scale from TEE and rebuild KV in the REE.
    """

    def __init__(self, tee: TEEEnclave = None, ree: REEGPUKVBank = None):
        self.tee = tee if tee is not None else TEEEnclave()
        self.ree = ree if ree is not None else REEGPUKVBank()
        self._engine_cap = self.tee.register_caller("inference-engine")
        self._handles: dict[str, SealRef] = {}
        self._parts: dict[str, dict] = {}

    # -- protect ------------------------------------------------------------ #
    def protect(self, handle: str, kv: torch.Tensor, **decompose_kwargs) -> dict:
        parts = decompose_kv(kv, **decompose_kwargs)
        ref = self.tee.seal_scale(parts['scale'], "inference-engine",
                                  self._engine_cap, label=handle)
        self.ree.write(handle, parts)
        self._handles[handle] = ref
        self._parts[handle] = parts
        return {'handle': handle, 'ref': ref, 'breakdown': kv_breakdown(parts)}

    # -- reconstruct --------------------------------------------------------- #
    def reconstruct(self, handle: str, out_dtype=None) -> torch.Tensor:
        ref = self._handles[handle]
        scale = self.tee.reveal_scale(ref, "inference-engine", self._engine_cap)
        return self.ree.reconstruct_with(handle, scale, out_dtype=out_dtype)

    def breakdown(self, handle: str) -> dict:
        return kv_breakdown(self._parts[handle])

    def attacker_reconstruct(self, handle: str, scale_guess: torch.Tensor,
                             out_dtype=None) -> torch.Tensor:
        """Model an adversary with full REE access but **no** TEE access.

        They must substitute a guessed scale; the returned reconstruction is
        what they would see from the leaked plaintext alone.
        """
        return self.ree.reconstruct_with(handle, scale_guess, out_dtype=out_dtype)

    # -- introspection ------------------------------------------------------- #
    def tee_resident_bytes(self) -> int:
        return self.tee.resident_bytes()

    def ree_resident_bytes(self) -> int:
        return self.ree.resident_bytes()


def verify_additive_nonlinear(bits: int = 2, H: int = 4, D: int = 64, T: int = 256,
                              num_centroids: int = 32, group_size=None, seed: int = 0):
    """Show that the encode/decode round-trip is *not* a linear map.

    KV-Cloak's naive defense ``K' = S K M`` is linear, which is exactly why a
    differential / CPA attack can solve for ``S`` and ``M`` from crafted
    input differences. The additive quantized decoding here is different: the
    encoding ``x -> (center, residual_int)`` depends on ``argmin`` (k-means
    assignment) and ``round`` (quantization), both of which are non-linear.
    """
    gen = torch.Generator().manual_seed(seed)

    def f(x):
        return reconstruct_kv(
            decompose_kv(x, bits=bits, num_centroids=num_centroids,
                         group_size=group_size, seed=seed))

    x1 = torch.randn(T, H, D, generator=gen)
    x2 = torch.randn(T, H, D, generator=gen)
    alpha = 1.7

    fx1, fx2 = f(x1), f(x2)
    f_sum = f(x1 + x2)
    f_hom = f(alpha * x1)

    lin_sum = fx1 + fx2
    lin_hom = alpha * fx1

    def rel(a, b):
        denom = b.norm() + 1e-9
        return ((a - b).norm() / denom).item()

    return {
        'superposition_rel_err': rel(f_sum, lin_sum),
        'homogeneity_rel_err': rel(f_hom, lin_hom),
    }

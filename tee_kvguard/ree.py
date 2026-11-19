"""REE (Rich Execution Environment) / GPU KV bank.

This is the *untrusted* side. It stores the plaintext, non-secret components
of the KV decomposition -- centroids, assignment indices and the quantized
residual content -- exactly as they would live in GPU / host memory or an
external KV-bank pool. It has **no** access to the ``scale`` tensors, which
are sealed inside the TEE.

In a real deployment this class would wrap GPU tensors (``.cuda()``) and
device-side decompression kernels; here everything runs on CPU because the
task only asks for Python and a CPU-simulated TEE.
"""

from __future__ import annotations

import torch

from .quantize import reconstruct_kv, _bytes


class REEGPUKVBank:
    """Plaintext KV bank. Holds center + quantized residual, never the scale."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._blocks: dict[str, dict] = {}

    def write(self, handle: str, parts: dict) -> None:
        """Store the non-secret part of a decomposed KV block.

        ``parts['scale']`` is deliberately *ignored*: the REE must never hold it.
        """
        self._blocks[handle] = {
            'centroids': parts['centroids'],
            'assign': parts['assign'],
            'residual_int': parts['residual_int'],
            'bits': parts['bits'],
            'group_size': parts['group_size'],
            'orig_shape': parts['orig_shape'],
        }

    def has(self, handle: str) -> bool:
        return handle in self._blocks

    def reconstruct_with(self, handle: str, scale: torch.Tensor,
                         out_dtype=None) -> torch.Tensor:
        """Reconstruct KV by combining REE plaintext with a TEE-provided scale."""
        parts = self._blocks[handle]
        parts = dict(parts)
        parts['scale'] = scale
        return reconstruct_kv(parts, out_dtype=out_dtype)

    def resident_bytes(self) -> int:
        return sum(
            _bytes(b['centroids']) + _bytes(b['assign']) + _bytes(b['residual_int'])
            for b in self._blocks.values()
        )

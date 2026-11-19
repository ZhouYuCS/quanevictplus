"""QuanEvict KV-Guard: TEE/REE split for KV-cache privacy.

The KV cache is decomposed (see :mod:`tee_kvguard.quantize`) into:

* center vectors (``centroids`` + ``assign``)      -> plaintext REE
* quantized residual content (``residual_int``)     -> plaintext REE
* quantization scale (``scale``)                    -> sealed in TEE (the only secret)

Only ``scale`` (a small fraction of the KV bank, ~5% under per-head grouping)
is kept inside the TEE. Recovering the original KV from the leaked plaintext
requires both the corresponding center vector and the scale, and the additive
``center + dequant(residual_int, scale)`` decoding breaks the linear-scaling
attack that defeats naive ``K' = S K M`` obfuscation.
"""

from .quantize import decompose_kv, reconstruct_kv, kv_breakdown
from .tee import TEEEnclave, SealRef, EnclaveAttestation
from .ree import REEGPUKVBank
from .guard import KVGuard, verify_additive_nonlinear

__all__ = [
    'decompose_kv',
    'reconstruct_kv',
    'kv_breakdown',
    'TEEEnclave',
    'SealRef',
    'EnclaveAttestation',
    'REEGPUKVBank',
    'KVGuard',
    'verify_additive_nonlinear',
]

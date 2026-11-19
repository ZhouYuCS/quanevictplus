"""KV-cache decomposition / reconstruction for QuanEvict KV-Guard.

The KV-cache tensor ``x`` (shape ``[..., T, H, D]``) is split into two parts,
mirroring QuanEvict's residual quantization (PRQ + symmetric group quant):

1. **中心向量 (center)**  --  ``centroids`` codebook + ``assign`` indices.
   Reconstructed per-token center is ``gather(centroids, assign)``.
2. **校正残差 (correction residual)**  --  ``x - center``, which is further
   split by symmetric per-group quantization into:
   * ``scale``       (量化 scale)      -> the only secret kept inside TEE
   * ``residual_int`` (量化存储内容)   -> plaintext, stays in REE

Reconstruction is *additive*::

    x_hat = center + dequant(residual_int, scale)

so recovering the original KV requires **both** the corresponding center
vector and the scale parameter. This is deliberately *not* a pure linear
scaling ``K' = S K M`` (the naive KV-Cloak obfuscation), which is what
breaks the linear / differential (CPA) attack surface.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  residual k-means (PRQ) helpers
# --------------------------------------------------------------------------- #
def _clamp_k(x: torch.Tensor, k: int) -> int:
    return min(k, x.shape[0])


def _kmeans_init(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    T, H, D = x.shape
    if k >= T:
        centroids = torch.zeros(H, k, D, device=x.device, dtype=x.dtype)
        centroids[:, :T, :] = x.transpose(0, 1)
        return centroids
    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(T, generator=g, device=x.device)[:k]
    return x[idx].transpose(0, 1).contiguous()


def _pairwise_sqdist(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=-1, keepdim=True)          # [T,H,1]
    c_norm = (c * c).sum(dim=-1)                        # [H,K]
    dot = torch.einsum('thd,hkd->thk', x, c)            # [T,H,K]
    return x_norm + c_norm.unsqueeze(0) - 2.0 * dot


def _kmeans(x: torch.Tensor, k: int, iters: int, seed: int):
    k = _clamp_k(x, k)
    T, H, D = x.shape
    centroids = _kmeans_init(x, k, seed)
    for _ in range(iters):
        dist = _pairwise_sqdist(x, centroids)
        assign = dist.argmin(dim=-1)                       # [T,H]
        onehot = F.one_hot(assign, num_classes=k).float()  # [T,H,K]
        counts = onehot.sum(dim=0).clamp_min(1.0)          # [H,K]
        sums = torch.einsum('thk,thd->hkd', onehot, x)     # [H,K,D]
        centroids = sums / counts.unsqueeze(-1)
    assign = _pairwise_sqdist(x, centroids).argmin(dim=-1)
    return centroids, assign


def _gather_centroids(centroids: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    H, K, D = centroids.shape
    T = assign.shape[0]
    h_idx = torch.arange(H, device=centroids.device).view(1, H).expand(T, H).reshape(-1)
    idx = h_idx * K + assign.reshape(-1)
    return centroids.reshape(H * K, D)[idx].reshape(T, H, D)


# --------------------------------------------------------------------------- #
#  symmetric per-group quantization (scale + packed int storage)
# --------------------------------------------------------------------------- #
def _effective_group_size(D: int, group_size) -> int:
    if group_size is None or group_size <= 0 or D % group_size != 0:
        return D
    return group_size


def _pack_signed(q: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed quantized ints (``[-qmax, qmax]``) into uint8 words."""
    qmax = (1 << (bits - 1)) - 1
    u = (q + qmax).to(torch.uint8)                 # shift to unsigned
    per = 8 // bits
    B = u.shape[-1]
    Bp = (B + per - 1) // per
    if B % per != 0:
        pad = per - (B % per)
        z = torch.zeros(*u.shape[:-1], pad, dtype=torch.uint8, device=u.device)
        u = torch.cat([u, z], dim=-1)
    u = u.reshape(*u.shape[:-1], Bp, per)
    packed = torch.zeros(*u.shape[:-1], dtype=torch.uint8, device=u.device)
    for i in range(per):
        packed = packed | (u[..., i] << (bits * i))
    return packed


def _unpack_signed(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    per = 8 // bits
    mask = (1 << bits) - 1
    shifts = torch.tensor([bits * i for i in range(per)], dtype=torch.uint8, device=packed.device)
    u = (packed.unsqueeze(-1) >> shifts) & mask
    u = u.reshape(*packed.shape[:-1], packed.shape[-1] * per)
    u = u[..., :numel]
    qmax = (1 << (bits - 1)) - 1
    return (u.to(torch.int32) - qmax).to(torch.int8)


def _symmetric_quantize(x: torch.Tensor, bits: int, group_size: int):
    T, H, D = x.shape
    group_size = _effective_group_size(D, group_size)
    qmax = float((1 << (bits - 1)) - 1)

    xg = x.reshape(T, H, D // group_size, group_size)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax  # [T,H,G,1]
    q = (xg / scale).round().clamp(-qmax, qmax).to(torch.int8)           # [T,H,G,B]

    packed = _pack_signed(q.reshape(T, H, D), bits)
    return scale.squeeze(-1).to(torch.bfloat16), packed                  # [T,H,G], packed


def _dequantize_with_meta(scale: torch.Tensor, packed: torch.Tensor,
                          bits: int, group_size: int) -> torch.Tensor:
    T, H, G = scale.shape
    D = G * group_size
    q = _unpack_signed(packed, bits, D)
    q = q.reshape(T, H, G, group_size)
    return (q.float() * scale.float().unsqueeze(-1)).reshape(T, H, D)


# --------------------------------------------------------------------------- #
#  public API
# --------------------------------------------------------------------------- #
def decompose_kv(
    x: torch.Tensor,
    bits: int = 2,
    num_centroids: int = 256,
    num_stages: int = 1,
    group_size=None,
    kmeans_iters: int = 4,
    seed: int = 0,
) -> dict:
    """Split ``x`` into center (centroids+assign) and quantized residual.

    Returns a dict with keys ``centroids``, ``assign`` (lists, one per PRQ
    stage), ``scale`` (bf16), ``residual_int`` (packed uint8), plus metadata.
    """
    if bits is None or bits <= 0 or x.numel() == 0:
        return None

    orig_shape = x.shape
    H, D = x.shape[-2], x.shape[-1]
    xf = x.reshape(-1, H, D).float()
    group_size = _effective_group_size(D, group_size)

    residual = xf
    centroids_list, assign_list = [], []
    for t in range(num_stages):
        centroids, assign = _kmeans(residual, num_centroids, kmeans_iters, seed + t)
        residual = residual - _gather_centroids(centroids, assign)
        centroids_list.append(centroids.to(torch.bfloat16))
        assign_list.append(assign.to(torch.uint8))

    scale, packed = _symmetric_quantize(residual, bits, group_size)

    return {
        'centroids': centroids_list,
        'assign': assign_list,
        'scale': scale,
        'residual_int': packed,
        'bits': bits,
        'group_size': group_size,
        'orig_shape': orig_shape,
    }


def reconstruct_kv(parts: dict, out_dtype=None) -> torch.Tensor:
    """Additive reconstruction: ``center + dequant(residual_int, scale)``."""
    if parts is None:
        return None

    scale = parts['scale'].float()
    packed = parts['residual_int']
    bits = parts['bits']
    group_size = parts['group_size']

    x = _dequantize_with_meta(scale, packed, bits, group_size)
    for t in range(len(parts['centroids']) - 1, -1, -1):
        x = x + _gather_centroids(parts['centroids'][t].float(), parts['assign'][t].long())

    x = x.reshape(parts['orig_shape'])
    if out_dtype is not None:
        x = x.to(out_dtype)
    return x


def _bytes(t: torch.Tensor) -> int:
    if t is None:
        return 0
    if isinstance(t, (list, tuple)):
        return sum(_bytes(x) for x in t)
    return t.numel() * t.element_size()


def kv_breakdown(parts: dict) -> dict:
    """Byte-level composition of the compressed KV bank.

    Returns sizes for the three conceptual components and their percentages:
    * center       (centroids + assign)
    * residual_int (quantized storage content)
    * scale        (quant scale -> the TEE-resident secret)
    """
    center_bytes = _bytes(parts['centroids']) + _bytes(parts['assign'])
    residual_bytes = _bytes(parts['residual_int'])
    scale_bytes = _bytes(parts['scale'])

    total = center_bytes + residual_bytes + scale_bytes
    pct = lambda n: 100.0 * n / total if total else 0.0

    return {
        'center_bytes': center_bytes,
        'residual_int_bytes': residual_bytes,
        'scale_bytes': scale_bytes,
        'total_bytes': total,
        'center_pct': pct(center_bytes),
        'residual_int_pct': pct(residual_bytes),
        'scale_pct': pct(scale_bytes),
    }


import time

import torch
import torch.nn.functional as F

from wan.utils import qvg_triton as _qvt

_QVG_PROFILE = {"quant_time": 0.0, "evict_time": 0.0}


def reset_qvg_profile():
    _QVG_PROFILE["quant_time"] = 0.0
    _QVG_PROFILE["evict_time"] = 0.0


_NORM_COLLECT = {"raw": [], "residual": [], "n": 0,
                 "raw_val": [], "res_val": [], "n_val": 0}
_NORM_COLLECT_ENABLED = False
_NORM_COLLECT_MAX = 20_000_000


def enable_norm_collect(enabled=True, max_samples=20_000_000):
    global _NORM_COLLECT_ENABLED, _NORM_COLLECT_MAX
    _NORM_COLLECT_ENABLED = enabled
    _NORM_COLLECT_MAX = max_samples
    _NORM_COLLECT["raw"] = []
    _NORM_COLLECT["residual"] = []
    _NORM_COLLECT["n"] = 0
    _NORM_COLLECT["raw_val"] = []
    _NORM_COLLECT["res_val"] = []
    _NORM_COLLECT["n_val"] = 0


def get_norm_collect():
    raw = torch.cat(_NORM_COLLECT["raw"]) if _NORM_COLLECT["raw"] else torch.empty(0)
    res = torch.cat(_NORM_COLLECT["residual"]) if _NORM_COLLECT["residual"] else torch.empty(0)
    return raw, res


def get_value_collect():
    rv = torch.cat(_NORM_COLLECT["raw_val"]) if _NORM_COLLECT["raw_val"] else torch.empty(0)
    sv = torch.cat(_NORM_COLLECT["res_val"]) if _NORM_COLLECT["res_val"] else torch.empty(0)
    return rv, sv


def _packed_breakdown(packed):
    out = {}
    if packed is None:
        return out
    for key, value in packed.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.numel() * value.element_size()
        elif isinstance(value, list):
            out[key] = sum(t.numel() * t.element_size() for t in value)
    return out


def _clamp_centroids(x: torch.Tensor, k: int) -> int:
    T = x.shape[0]
    return min(k, T)


def _kmeans_init(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    T, H, D = x.shape
    if k >= T:
        centroids = torch.zeros(H, k, D, device=x.device, dtype=x.dtype)
        centroids[:, :T, :] = x.transpose(0, 1)
        return centroids

    g = torch.Generator(device=x.device).manual_seed(seed)
    idx = torch.randperm(T, generator=g, device=x.device)[:k]
    return x[idx, :, :].transpose(0, 1).contiguous()


def _pairwise_sqdist(x: torch.Tensor, c: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    x_norm = (x * x).sum(dim=-1, keepdim=True)
    c_norm = (c * c).sum(dim=-1)
    dot = _qvt._pairwise_dot(x, c) if use_triton else torch.einsum(
        'thd,hkd->thk', x, c)
    return x_norm + c_norm.unsqueeze(0) - 2.0 * dot


def _kmeans_batched(x: torch.Tensor, k: int, iters: int, seed: int,
                    use_triton: bool = True):
    k = _clamp_centroids(x, k)
    T, H, D = x.shape
    centroids = _kmeans_init(x, k, seed)

    can_triton = (
        use_triton and _qvt.has_triton() and x.is_cuda
        and D >= 16 and (D & (D - 1)) == 0
    )

    if can_triton:
        sums = torch.zeros((H, k, D), device=x.device, dtype=torch.float32)
        counts = torch.zeros((H, k), device=x.device, dtype=torch.float32)
        for _ in range(iters):
            assign32 = _qvt.kmeans_assign(x, centroids)
            sums.zero_()
            counts.zero_()
            _qvt.kmeans_scatter_update(x, assign32, sums, counts)
            centroids = sums / counts.clamp_min(1.0).unsqueeze(-1)
        assign32 = _qvt.kmeans_assign(x, centroids)
        return centroids, assign32

    for _ in range(iters):
        dist = _pairwise_sqdist(x, centroids, use_triton=False)
        assign = dist.argmin(dim=-1)
        onehot = F.one_hot(assign, num_classes=k).float()
        counts = onehot.sum(dim=0).clamp_min(1.0)
        sums = torch.einsum('thk,thd->hkd', onehot, x)
        centroids = sums / counts.unsqueeze(-1)

    dist = _pairwise_sqdist(x, centroids, use_triton=False)
    assign = dist.argmin(dim=-1)
    return centroids, assign


def _gather_centroids(centroids: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    H, K, D = centroids.shape
    T = assign.shape[0]
    h_idx = torch.arange(H, device=centroids.device).view(1, H).expand(T, H).reshape(-1)
    idx = h_idx * K + assign.reshape(-1)
    return centroids.reshape(H * K, D)[idx].reshape(T, H, D)


def _symmetric_quantize_dequantize(x: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    T, H, D = x.shape
    if D % group_size != 0:
        group_size = D
    qmax = float((1 << (bits - 1)) - 1)

    xg = x.reshape(T, H, D // group_size, group_size)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    x_int = (xg / scale).round().clamp(-qmax, qmax)
    return (x_int * scale).reshape(T, H, D)


def _pack_int2(x_int: torch.Tensor) -> torch.Tensor:
    B = x_int.shape[-1]
    v = (x_int + 1).to(torch.uint8)
    Bp = (B + 3) // 4
    if B % 4 != 0:
        pad = 4 - (B % 4)
        z = torch.zeros(*v.shape[:-1], pad, dtype=torch.uint8, device=v.device)
        v = torch.cat([v, z], dim=-1)
    v = v.reshape(*v.shape[:-1], Bp, 4)
    return v[..., 0] | (v[..., 1] << 2) | (v[..., 2] << 4) | (v[..., 3] << 6)


def _unpack_int2(packed: torch.Tensor, B: int) -> torch.Tensor:
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    v = (packed.unsqueeze(-1) >> shifts) & 0b11
    v = v.reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    v = v[..., :B]
    return v.to(torch.int8) - 1


def _qvg_roundtrip_batched(
    x: torch.Tensor,
    bits: int,
    num_centroids: int,
    num_stages: int,
    group_size: int,
    kmeans_iters: int,
    seed: int,
) -> torch.Tensor:
    residual = x
    centroids_list = []
    assign_list = []

    for t in range(num_stages):
        centroids, assign = _kmeans_batched(
            residual, num_centroids, kmeans_iters, seed + t)
        residual = residual - _gather_centroids(centroids, assign)
        centroids_list.append(centroids)
        assign_list.append(assign)

    reconstructed = _symmetric_quantize_dequantize(residual, bits, group_size)

    for t in range(num_stages - 1, -1, -1):
        reconstructed = reconstructed + _gather_centroids(
            centroids_list[t], assign_list[t])

    return reconstructed


def qvg_roundtrip(
    x: torch.Tensor,
    bits: int = 2,
    num_centroids: int = 256,
    num_stages: int = 1,
    group_size: int = 64,
    kmeans_iters: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    if bits is None or bits <= 0 or x.numel() == 0:
        return x

    orig_shape = x.shape
    H = x.shape[-2]
    D = x.shape[-1]
    xf = x.reshape(-1, H, D).float()

    out = _qvg_roundtrip_batched(
        xf, bits, num_centroids, num_stages, group_size, kmeans_iters, seed)

    return out.reshape(orig_shape).to(x.dtype)


def _symmetric_quantize(
    x: torch.Tensor, bits: int, group_size: int
) -> tuple:
    T, H, D = x.shape
    if D % group_size != 0:
        group_size = D
    qmax = float((1 << (bits - 1)) - 1)

    xg = x.reshape(T, H, D // group_size, group_size)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    x_int = (xg / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return scale.squeeze(-1).to(torch.bfloat16), _pack_int2(x_int)


def _symmetric_dequantize(scale: torch.Tensor, x_int: torch.Tensor, group_size: int) -> torch.Tensor:
    x_int = _unpack_int2(x_int, group_size)
    T, H, G, B = x_int.shape
    return (x_int.float() * scale.float().unsqueeze(-1)).reshape(T, H, G * B)


def _symmetric_quantize_flat(
    x: torch.Tensor, bits: int, group_size: int
) -> tuple:
    N, D = x.shape
    if D % group_size != 0:
        group_size = D
    qmax = float((1 << (bits - 1)) - 1)

    xg = x.reshape(N, D // group_size, group_size)
    scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    x_int = (xg / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return scale.squeeze(-1).to(torch.bfloat16), _pack_int2(x_int)


def _symmetric_dequantize_flat(scale: torch.Tensor, x_int: torch.Tensor, group_size: int) -> torch.Tensor:
    x_int = _unpack_int2(x_int, group_size)
    N, G, B = x_int.shape
    return (x_int.float() * scale.float().unsqueeze(-1)).reshape(N, G * B)


def _residual_keep_mask(residual: torch.Tensor, ratio, bits: int, group_size: int,
                        random_evict: bool = False) -> torch.Tensor:
    T, H, _ = residual.shape
    ratio = torch.as_tensor(ratio, device=residual.device, dtype=torch.float32)
    if ratio.dim() == 0:
        ratio = ratio.expand(H)
    k = (ratio * T).round().long().clamp(min=1, max=T)
    if random_evict:
        score = torch.rand(T, H, device=residual.device, dtype=torch.float32)
    else:
        q = _symmetric_quantize_dequantize(residual, bits, group_size)
        r2 = (residual * residual).sum(dim=-1)
        diff = residual - q
        score = r2 - (diff * diff).sum(dim=-1)
    sorted_score, _ = torch.sort(score, dim=0, descending=True)
    thr = sorted_score[k - 1, torch.arange(H, device=residual.device)]
    return score >= thr.unsqueeze(0)


def quanevict_ratios(w, ratio_base, gamma=1.0, eps=1e-12, w_bar=None):
    w = torch.as_tensor(w, dtype=torch.float32)
    w = w.clamp_min(eps)
    if w_bar is None:
        w_bar = w.mean()
    else:
        w_bar = torch.as_tensor(w_bar, dtype=torch.float32)
    w_bar = w_bar.clamp_min(eps)
    ratio = ratio_base * w / w_bar
    return ratio.clamp(0.0, 1.0).to(w.dtype)


def qvg_compress(
    x: torch.Tensor,
    bits: int = 2,
    num_centroids: int = 256,
    num_stages: int = 1,
    group_size: int = 64,
    kmeans_iters: int = 4,
    seed: int = 0,
    sparsify: bool = False,
    sparse_ratio: float = 0.5,
    random_evict: bool = False,
    use_triton: bool = True,
    num_frames: int = 1,
    return_residual_mag: bool = False,
) -> dict:
    if bits is None or bits <= 0 or x.numel() == 0:
        return None

    orig_shape = x.shape
    H = x.shape[-2]
    D = x.shape[-1]
    xf = x.reshape(-1, H, D).float()

    residual = xf
    centroids_list = []
    assign_list = []
    t_kmeans = time.perf_counter()
    for t in range(num_stages):
        centroids, assign = _kmeans_batched(
            residual, num_centroids, kmeans_iters, seed + t, use_triton=use_triton)
        residual = residual - _gather_centroids(centroids, assign)
        centroids_list.append(centroids.to(torch.bfloat16))
        assign_list.append(assign.to(torch.uint8))
    kmeans_time = time.perf_counter() - t_kmeans

    residual_mag = residual.norm(dim=-1) if return_residual_mag else None
    extra = {'residual_mag': residual_mag} if return_residual_mag else {}

    if _NORM_COLLECT_ENABLED:
        raw_mag = xf.norm(dim=-1).reshape(-1)
        res_mag = residual.norm(dim=-1).reshape(-1)
        Nm = raw_mag.numel()
        bm = 20000
        if Nm > bm:
            im = torch.linspace(0, Nm - 1, bm, dtype=torch.long, device=raw_mag.device)
            raw_mag = raw_mag[im]
            res_mag = res_mag[im]
        if _NORM_COLLECT["n"] < _NORM_COLLECT_MAX:
            _NORM_COLLECT["raw"].append(raw_mag.float().cpu())
            _NORM_COLLECT["residual"].append(res_mag.float().cpu())
            _NORM_COLLECT["n"] += raw_mag.numel()

        raw_val = xf.reshape(-1)
        res_val = residual.reshape(-1)
        Nv = raw_val.numel()
        bv = 20000
        if Nv > bv:
            iv = torch.linspace(0, Nv - 1, bv, dtype=torch.long, device=raw_val.device)
            raw_val = raw_val[iv]
            res_val = res_val[iv]
        if _NORM_COLLECT["n_val"] < _NORM_COLLECT_MAX:
            _NORM_COLLECT["raw_val"].append(raw_val.float().cpu())
            _NORM_COLLECT["res_val"].append(res_val.float().cpu())
            _NORM_COLLECT["n_val"] += raw_val.numel()

    if sparsify:
        t_evict = time.perf_counter()
        keep_mask = _residual_keep_mask(residual, sparse_ratio, bits, group_size,
                                        random_evict=random_evict)
        keep_idx = keep_mask.reshape(-1).nonzero().squeeze(1).to(torch.int32)
        residual_kept = residual.reshape(-1, D)[keep_idx.long()]
        _QVG_PROFILE["evict_time"] += time.perf_counter() - t_evict
        t_quant = time.perf_counter()
        scale, x_int = _symmetric_quantize_flat(residual_kept, bits, group_size)
        _QVG_PROFILE["quant_time"] += kmeans_time + (time.perf_counter() - t_quant)
        frame_offsets = _frame_offsets_from_mask(keep_mask, num_frames)
        return {
            'centroids': centroids_list,
            'assign': assign_list,
            'scale': scale,
            'residual_int': x_int,
            'keep_idx': keep_idx,
            'frame_offsets': frame_offsets,
            'num_frames': num_frames,
            'group_size': group_size,
            'orig_shape': orig_shape,
            'sparse': True,
            **extra,
        }

    t_quant = time.perf_counter()
    scale, x_int = _symmetric_quantize(residual, bits, group_size)
    _QVG_PROFILE["quant_time"] += kmeans_time + (time.perf_counter() - t_quant)
    return {
        'centroids': centroids_list,
        'assign': assign_list,
        'scale': scale,
        'residual_int': x_int,
        'group_size': group_size,
        'orig_shape': orig_shape,
        'num_frames': num_frames,
        'sparse': False,
        **extra,
    }


def _frame_offsets_from_mask(keep_mask: torch.Tensor, num_frames: int) -> torch.Tensor:
    T, H = keep_mask.shape
    frame_seqlen = T // num_frames
    per_frame = keep_mask.reshape(num_frames, frame_seqlen, H).sum(dim=(1, 2)).to(torch.int32)
    return torch.cat([torch.zeros(1, dtype=torch.int32, device=keep_mask.device),
                      per_frame.cumsum(0)])


def _decompress_sparse_frames(packed: dict, frame_ids, out_dtype=None, use_triton: bool = True, out=None) -> torch.Tensor:
    centroids = packed['centroids']
    assign = packed['assign']
    keep_idx = packed['keep_idx']
    frame_offsets = packed['frame_offsets']
    scale = packed['scale'].float()
    residual_int = packed['residual_int']
    num_frames = packed['num_frames']
    group_size = packed['group_size']

    T, H = assign[-1].shape
    frame_seqlen = T // num_frames
    G = residual_int.shape[1]
    B = group_size
    D = G * B

    frame_ids = [int(f) for f in frame_ids]
    n_frames = len(frame_ids)

    base = None
    for t in range(len(centroids) - 1):
        g = _gather_centroids(centroids[t].float(), assign[t].long())
        base = g if base is None else base + g

    last_c = centroids[-1].float()
    last_a = assign[-1].long()

    if out is None:
        dtype = out_dtype if out_dtype is not None else last_c.dtype
        out = torch.empty([n_frames * frame_seqlen, H, D], device=last_c.device, dtype=dtype)

    use_fused = (
        use_triton
        and _qvt.has_triton()
        and last_c.is_cuda
        and D >= 16
        and (D & (D - 1)) == 0
    )
    if use_fused:
        scratch = torch.empty([frame_seqlen, H, D], device=last_c.device, dtype=torch.float32)
        last_a32 = last_a.to(torch.int32)

    for i, f in enumerate(frame_ids):
        lo = int(frame_offsets[f])
        hi = int(frame_offsets[f + 1])
        seg_idx = keep_idx[lo:hi]
        token_local = (seg_idx // H) - f * frame_seqlen
        head_idx = seg_idx % H
        flat_idx = token_local * H + head_idx

        if use_fused:
            _qvt.decompress_frame_fused(
                last_c,
                last_a32[f * frame_seqlen:(f + 1) * frame_seqlen],
                residual_int[lo:hi],
                scale[lo:hi],
                flat_idx,
                out[i * frame_seqlen:(i + 1) * frame_seqlen],
                scratch,
                base[f * frame_seqlen:(f + 1) * frame_seqlen] if base is not None else None,
                group_size=group_size,
            )
            continue

        c = _gather_centroids(last_c, last_a[f * frame_seqlen:(f + 1) * frame_seqlen])
        c_flat = c.reshape(-1, D)
        if use_triton:
            _qvt.scatter_residual(residual_int[lo:hi], scale[lo:hi], flat_idx, c_flat, group_size)
        else:
            resid = _symmetric_dequantize_flat(scale[lo:hi], residual_int[lo:hi], group_size)
            c_flat.index_add_(0, flat_idx.long(), resid)
        if base is not None:
            c = c + base[f * frame_seqlen:(f + 1) * frame_seqlen]
        out[i * frame_seqlen:(i + 1) * frame_seqlen] = c

    return out


def qvg_decompress(packed: dict, out_dtype=None, use_triton: bool = True) -> torch.Tensor:
    if packed.get('sparse', False):
        num_frames = packed.get('num_frames', 1)
        x = _decompress_sparse_frames(packed, range(num_frames), out_dtype, use_triton)
        return x.reshape(packed['orig_shape'])

    x = _symmetric_dequantize(packed['scale'].float(), packed['residual_int'], packed['group_size'])
    for t in range(len(packed['centroids']) - 1, -1, -1):
        x = x + _gather_centroids(packed['centroids'][t].float(), packed['assign'][t].long())
    x = x.reshape(packed['orig_shape'])
    if out_dtype is not None:
        x = x.to(out_dtype)
    return x


def _decompress_dense_frames(packed: dict, frame_ids, out_dtype=None, use_triton: bool = True, out=None) -> torch.Tensor:
    centroids = packed['centroids']
    assign = packed['assign']
    scale = packed['scale'].float()
    residual_int = packed['residual_int']
    num_frames = packed['num_frames']
    group_size = packed['group_size']

    T, H = assign[-1].shape
    frame_seqlen = T // num_frames
    D = centroids[-1].shape[-1]

    frame_ids = [int(f) for f in frame_ids]
    n_frames = len(frame_ids)
    if out is None:
        dtype = out_dtype if out_dtype is not None else centroids[-1].dtype
        out = torch.empty([n_frames * frame_seqlen, H, D], device=centroids[-1].device, dtype=dtype)

    for i, f in enumerate(frame_ids):
        lo = f * frame_seqlen
        hi = (f + 1) * frame_seqlen
        x = _symmetric_dequantize(scale[lo:hi], residual_int[lo:hi], group_size)
        for t in range(len(centroids) - 1, -1, -1):
            x = x + _gather_centroids(centroids[t].float(), assign[t][lo:hi].long())
        out[i * frame_seqlen:(i + 1) * frame_seqlen] = x
    return out


def qvg_decompress_frames(packed: dict, frame_ids, out_dtype=None, use_triton: bool = True, out=None) -> torch.Tensor:
    if not packed.get('sparse', False):
        return _decompress_dense_frames(packed, frame_ids, out_dtype, use_triton, out=out)
    return _decompress_sparse_frames(packed, frame_ids, out_dtype, use_triton, out=out)


import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


def has_triton() -> bool:
    return _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _pairwise_dot_kernel(
        x_ptr, c_ptr, out_ptr,
        T, H, K, D,
        BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)

        x_ptrs = x_ptr + offs_t[:, None] * (H * D) + pid_h * D + offs_d[None, :]
        x_tile = tl.load(x_ptrs, mask=offs_t[:, None] < T, other=0.0)

        c_ptrs = c_ptr + pid_h * (K * D) + offs_k[:, None] * D + offs_d[None, :]
        c_tile = tl.load(c_ptrs, mask=offs_k[:, None] < K, other=0.0)

        acc = tl.dot(x_tile, tl.trans(c_tile), input_precision="ieee")

        out_ptrs = out_ptr + offs_t[:, None] * (H * K) + pid_h * K + offs_k[None, :]
        tl.store(out_ptrs, acc, mask=(offs_t[:, None] < T) & (offs_k[None, :] < K))

    @triton.jit
    def _scatter_residual_kernel(
        residual_ptr,
        scale_ptr,
        idx_ptr,
        out_ptr,
        N, D,
        G: tl.constexpr, B: tl.constexpr, BP: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offs_g = tl.arange(0, G)
        offs_bp = tl.arange(0, BP)
        packed = tl.load(
            residual_ptr + pid * (G * BP) + offs_g[:, None] * BP + offs_bp[None, :])

        offs_sub = tl.arange(0, 4)
        sub = (packed[:, :, None] >> (offs_sub[None, None, :] * 2)) & 3
        sub = tl.reshape(sub, (G, BP * 4)).to(tl.int32) - 1

        sc = tl.load(scale_ptr + pid * G + offs_g)
        r = sub.to(tl.float32) * sc[:, None]
        r_flat = tl.reshape(r, (BLOCK_D,))

        tgt = tl.load(idx_ptr + pid).to(tl.int32)
        offs_d = tl.arange(0, BLOCK_D)
        tl.atomic_add(out_ptr + tgt * D + offs_d, r_flat)


    @triton.jit
    def _gather_centroids_kernel(
        centroids_ptr,
        assign_ptr,
        scratch_ptr,
        H, K, D,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        a = tl.load(assign_ptr + pid_t * H + pid_h)
        offs_d = tl.arange(0, BLOCK_D)
        c = tl.load(centroids_ptr + pid_h * K * D + a * D + offs_d)
        tl.store(scratch_ptr + pid_t * H * D + pid_h * D + offs_d, c)

    @triton.jit
    def _add_base_cast_kernel(
        scratch_ptr,
        base_ptr,
        out_ptr,
        H, D,
        HAS_BASE: tl.constexpr,
        OUT_BF16: tl.constexpr,
        OUT_FP16: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)
        v = tl.load(scratch_ptr + pid_t * H * D + pid_h * D + offs_d)
        if HAS_BASE:
            v = v + tl.load(base_ptr + pid_t * H * D + pid_h * D + offs_d)
        if OUT_BF16:
            v = v.to(tl.bfloat16)
        elif OUT_FP16:
            v = v.to(tl.float16)
        tl.store(out_ptr + pid_t * H * D + pid_h * D + offs_d, v)


def _pairwise_dot(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    if not (_HAS_TRITON and x.is_cuda):
        return torch.einsum('thd,hkd->thk', x, c)

    T, H, D = x.shape
    K = c.shape[1]
    if D < 16 or (D & (D - 1)) != 0:
        return torch.einsum('thd,hkd->thk', x, c)

    BLOCK_T = 32
    BLOCK_K = 32
    BLOCK_D = D
    out = torch.empty((T, H, K), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_K), H)
    _pairwise_dot_kernel[grid](
        x, c, out, T, H, K, D, BLOCK_T, BLOCK_K, BLOCK_D)
    return out


def _unpack_int2(packed: torch.Tensor, B: int) -> torch.Tensor:
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
    v = (packed.unsqueeze(-1) >> shifts) & 0b11
    v = v.reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    v = v[..., :B]
    return v.to(torch.int8) - 1


def scatter_residual(
    residual_int: torch.Tensor,
    scale: torch.Tensor,
    flat_idx: torch.Tensor,
    c_flat: torch.Tensor,
    group_size: int,
) -> None:
    N, G, BP = residual_int.shape
    B = group_size
    D = G * B

    if not (_HAS_TRITON and residual_int.is_cuda) or D < 16 or (D & (D - 1)) != 0 or (B % 4) != 0:
        resid = _unpack_int2(residual_int, B).float() * scale.float().unsqueeze(-1)
        resid = resid.reshape(N, D)
        c_flat.index_add_(0, flat_idx.long(), resid)
        return

    _scatter_residual_kernel[(N,)](
        residual_int, scale, flat_idx, c_flat, N, D,
        G=G, B=B, BP=BP, BLOCK_D=D)

def decompress_frame_fused(
    centroids: torch.Tensor,
    assign: torch.Tensor,
    residual_int: torch.Tensor,
    scale: torch.Tensor,
    flat_idx: torch.Tensor,
    out: torch.Tensor,
    scratch: torch.Tensor,
    base: torch.Tensor = None,
    group_size: int = None,
) -> None:
    H, K, D = centroids.shape
    frame_seqlen = assign.shape[0]
    assign32 = assign if assign.dtype == torch.int32 else assign.to(torch.int32)
    grid = (frame_seqlen, H)

    _gather_centroids_kernel[grid](
        centroids, assign32, scratch, H, K, D, BLOCK_D=D)

    scatter_residual(residual_int, scale, flat_idx, scratch.reshape(-1, D), group_size)

    base_arg = base if base is not None else scratch
    _add_base_cast_kernel[grid](
        scratch, base_arg, out, H, D,
        HAS_BASE=base is not None,
        OUT_BF16=out.dtype == torch.bfloat16,
        OUT_FP16=out.dtype == torch.float16,
        BLOCK_D=D)



if _HAS_TRITON:

    @triton.jit
    def _kmeans_scatter_update_kernel(
        x_ptr, assign_ptr, sums_ptr, counts_ptr,
        T, H, K, D,
        BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_d = tl.arange(0, BLOCK_D)
        x_t = tl.load(x_ptr + pid_t * (H * D) + pid_h * D + offs_d)
        k = tl.load(assign_ptr + pid_t * H + pid_h)

        tl.atomic_add(sums_ptr + pid_h * (K * D) + k * D + offs_d, x_t)
        tl.atomic_add(counts_ptr + pid_h * K + k, 1.0)


def kmeans_scatter_update(
    x: torch.Tensor,
    assign: torch.Tensor,
    sums: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    T, H, D = x.shape
    K = sums.shape[1]
    _kmeans_scatter_update_kernel[(T, H)](x, assign, sums, counts, T, H, K, D, BLOCK_D=D)


if _HAS_TRITON:

    @triton.jit
    def _kmeans_assign_kernel(
        x_ptr, c_ptr, assign_ptr,
        T, H, K, D,
        BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)

        x_ptrs = x_ptr + offs_t[:, None] * (H * D) + pid_h * D + offs_d[None, :]
        x_tile = tl.load(x_ptrs, mask=offs_t[:, None] < T, other=0.0)
        x_norm = tl.sum(x_tile * x_tile, axis=1)

        INF = 1e30
        best_dist = tl.full((BLOCK_T,), INF, dtype=tl.float32)
        best_idx = tl.zeros((BLOCK_T,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            c_ptrs = c_ptr + pid_h * (K * D) + offs_k[:, None] * D + offs_d[None, :]
            c_tile = tl.load(c_ptrs, mask=offs_k[:, None] < K, other=0.0)
            dot = tl.dot(x_tile, tl.trans(c_tile), input_precision="tf32")
            c_norm = tl.sum(c_tile * c_tile, axis=1)
            dist = x_norm[:, None] - 2.0 * dot + c_norm[None, :]
            dist = tl.where((offs_k < K)[None, :], dist, INF)

            local_idx = tl.argmin(dist, axis=1)
            local_min = tl.min(dist, axis=1)
            global_idx = local_idx + k0
            update = local_min < best_dist
            best_dist = tl.where(update, local_min, best_dist)
            best_idx = tl.where(update, global_idx, best_idx)

        tl.store(assign_ptr + offs_t * H + pid_h, best_idx, mask=offs_t < T)


def kmeans_assign(
    x: torch.Tensor,
    c: torch.Tensor,
    BLOCK_T: int = 32,
    BLOCK_K: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    T, H, D = x.shape
    K = c.shape[1]
    assign = torch.empty((T, H), device=x.device, dtype=torch.int32)
    grid = (triton.cdiv(T, BLOCK_T), H)
    _kmeans_assign_kernel[grid](
        x, c, assign, T, H, K, D,
        BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K, BLOCK_D=D,
        num_warps=num_warps,
    )
    return assign


if _HAS_TRITON:

    @triton.jit
    def _fastar_top1_kernel(
        kf_ptr, bank_ptr, maxsim_ptr, argmax_ptr,
        S, G,
        NHEADS: tl.constexpr, D: tl.constexpr,
        BLOCK_S: tl.constexpr, BLOCK_G: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
        offs_d = tl.arange(0, D)
        s_mask = offs_s < S

        NEG_INF = -1e30
        best = tl.full([BLOCK_S], NEG_INF, tl.float32)
        best_idx = tl.zeros([BLOCK_S], tl.int64)

        for g0 in range(0, G, BLOCK_G):
            offs_g = g0 + tl.arange(0, BLOCK_G)
            g_mask = offs_g < G

            acc = tl.zeros([BLOCK_S, BLOCK_G], tl.float32)
            for h in range(0, NHEADS):
                kf_h = tl.load(
                    kf_ptr + offs_s[:, None] * (NHEADS * D) + h * D + offs_d[None, :],
                    mask=s_mask[:, None], other=0.0)
                bank_h = tl.load(
                    bank_ptr + offs_g[:, None] * (NHEADS * D) + h * D + offs_d[None, :],
                    mask=g_mask[:, None], other=0.0)
                acc += tl.dot(kf_h, tl.trans(bank_h), input_precision="ieee")

            sim = acc / NHEADS
            sim = tl.where(g_mask[None, :], sim, NEG_INF)

            local_max = tl.max(sim, axis=1)
            local_arg = tl.argmax(sim, axis=1)
            update = local_max > best
            best = tl.where(update, local_max, best)
            best_idx = tl.where(update, (g0 + local_arg).to(tl.int64), best_idx)

        tl.store(maxsim_ptr + offs_s, best, mask=s_mask)
        tl.store(argmax_ptr + offs_s, best_idx, mask=s_mask)


def fastar_top1_merge(
    kf_n: torch.Tensor,
    bank_k_n: torch.Tensor,
    BLOCK_S: int = 16,
    BLOCK_G: int = 64,
    num_warps: int = 4,
):
    S, n, d = kf_n.shape
    G = bank_k_n.shape[0]

    if not (_HAS_TRITON and kf_n.is_cuda) or G == 0:
        max_sim = torch.full([S], -float('inf'), device=kf_n.device, dtype=torch.float32)
        argmax = torch.zeros([S], device=kf_n.device, dtype=torch.int64)
        chunk = 1024
        for start in range(0, G, chunk):
            end = min(start + chunk, G)
            blk = bank_k_n[start:end]
            sim_blk = torch.einsum('snd,gnd->sgn', kf_n, blk).mean(dim=-1)
            blk_max, blk_arg = sim_blk.max(dim=1)
            better = blk_max > max_sim
            max_sim[better] = blk_max[better]
            argmax[better] = start + blk_arg[better]
        return max_sim, argmax

    max_sim = torch.empty([S], device=kf_n.device, dtype=torch.float32)
    argmax = torch.empty([S], device=kf_n.device, dtype=torch.int64)
    grid = (triton.cdiv(S, BLOCK_S),)
    _fastar_top1_kernel[grid](
        kf_n, bank_k_n, max_sim, argmax, S, G,
        NHEADS=n, D=d,
        BLOCK_S=BLOCK_S, BLOCK_G=BLOCK_G,
        num_warps=num_warps,
    )
    return max_sim, argmax

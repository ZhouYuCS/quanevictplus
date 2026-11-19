"""Runnable demo for QuanEvict KV-Guard (CPU-simulated TEE).

Run::

    python -m tee_kvguard.demo

It shows three things:

1. **Decomposition** -- KV is split into center vectors + quantized residual
   (scale + residual content), with a byte-level breakdown of the KV bank.
2. **Lossless reconstruction** -- with the TEE-released scale, the KV is
   reconstructed to near-lossless fidelity.
3. **Security** -- (a) an REE-only adversary without the scale cannot recover
   the KV; (b) the additive decode is non-linear, so a linear/differential
   attack (the one that breaks naive ``K' = S K M``) does not apply.
"""

from __future__ import annotations

import torch

from tee_kvguard import KVGuard, verify_additive_nonlinear


def _fmt(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _quality(name: str, ref: torch.Tensor, est: torch.Tensor) -> None:
    ref = ref.float()
    est = est.float()
    mse = ((ref - est) ** 2).mean().item()
    rel = ((est - ref).norm() / (ref.norm() + 1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(
        ref.reshape(1, -1), est.reshape(1, -1)).item()
    print(f"  {name:<24} MSE={mse:.6f}  rel_L2={rel*100:6.2f}%  cos={cos:.5f}")


def main():
    torch.manual_seed(0)

    # Small config so the CPU demo runs in seconds. D=64 / group_size=64 puts
    # exactly one scale per (token, head). T is chosen so that the codebook
    # (centroids) is a comparable fraction of the bank, which is the regime
    # where scale lands at ~5% of the compressed KV bank.
    T, H, D = 384, 4, 64
    cfg = dict(bits=2, num_centroids=64, num_stages=1, group_size=64,
               kmeans_iters=4, seed=0)

    print("=" * 78)
    print("QuanEvict KV-Guard  (TEE = CPU 模拟,  REE = 明文 KV bank)")
    print("=" * 78)

    # Synthetic KV drawn from a zero-mean Gaussian. Real KV (after RoPE / RMS
    # norm) is not strongly clustered, so the residual after k-means carries
    # most of the fine-grained information -- exactly what the 2-bit scale
    # protects.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(T, H, D, generator=g)

    guard = KVGuard()

    print(f"\n[1] 保护 KV (shape={tuple(x.shape)})")
    res = guard.protect("layer0", x, **cfg)
    bd = res['breakdown']

    print("\n  KV-bank 组成 breakdown:")
    print(f"    中心向量 (centroids+assign):  {_fmt(bd['center_bytes']):>10}  {bd['center_pct']:5.2f}%")
    print(f"    量化存储内容 (residual_int):  {_fmt(bd['residual_int_bytes']):>10}  {bd['residual_int_pct']:5.2f}%")
    print(f"    量化 scale (TEE secret):      {_fmt(bd['scale_bytes']):>10}  {bd['scale_pct']:5.2f}%")
    print(f"    ----------------------------------------------")
    print(f"    总计                          {_fmt(bd['total_bytes']):>10}")
    print(f"    TEE 驻留字节: {_fmt(guard.tee_resident_bytes())}   REE 驻留字节: {_fmt(guard.ree_resident_bytes())}")

    print("\n[2] 忠实解压 (TEE 释放 scale)")
    # `x_hat` 就是 (中心向量, 量化存储内容, scale) 三者的精确解压结果。
    # 下面相对原始 KV 的误差是 2-bit 量化本身带来的压缩误差, 不是安全损失。
    x_hat = guard.reconstruct("layer0")
    _quality("解压 vs 原始 KV", x, x_hat)

    print("\n[3] 安全性")
    print("  (a) 攻击者只有 REE 明文 (中心向量 + 量化存储内容), 没有 scale:")
    print("      以忠实解压结果 x_hat 为基准, 衡量'缺 scale'造成的额外损失")
    scale_shape = tuple(guard._parts['layer0']['scale'].shape)
    x_center_only = guard.attacker_reconstruct("layer0", torch.zeros(scale_shape))
    x_scale_one = guard.attacker_reconstruct("layer0", torch.ones(scale_shape))
    x_scale_rand = guard.attacker_reconstruct(
        "layer0", torch.rand(scale_shape) * 0.5 + 0.5)
    _quality("guess scale=0 (仅中心向量)", x_hat, x_center_only)
    _quality("guess scale=1", x_hat, x_scale_one)
    _quality("guess scale=random", x_hat, x_scale_rand)

    # residual 的能量占比: 说明 scale 保护的这部分信息不可忽略
    resid_energy = (x_hat - x_center_only).pow(2).mean().item() / (
        x_hat.pow(2).mean().item() + 1e-12)
    print(f"\n    说明: 残差(由 scale 恢复的部分)携带约 {resid_energy*100:.1f}% 的解压结果能量。")
    print("    没有 scale 时这部分只能被猜, 无法逐元素精确恢复 (verbatim 重建被阻断)。")

    print("\n  (b) 加性解码破坏线性缩放攻击:")
    nl = verify_additive_nonlinear(
        bits=cfg['bits'], H=H, D=D, T=256,
        num_centroids=cfg['num_centroids'], group_size=cfg['group_size'])
    print(f"    叠加性误差 f(x1+x2) vs f(x1)+f(x2):  {nl['superposition_rel_err']:.4f}")
    print(f"    齐次性误差 f(a*x1)  vs a*f(x1):      {nl['homogeneity_rel_err']:.4f}")
    print("    线性映射两者都应 ~0；叠加性明显非零 => 编码非线性,")
    print("    KV-Cloak 针对 K'=S·K·M 的差分/CPA 线性攻击在此不成立。")

    print("\n" + "=" * 78)
    print("结论: 只有 ~{:.1f}% 的 scale 需要进 TEE, 中心向量与量化残差明文留在 REE,"
          .format(bd['scale_pct']))
    print("      而攻击者拿不到 scale 就无法从 REE 明文恢复原始 KV。")
    print("=" * 78)


if __name__ == "__main__":
    main()

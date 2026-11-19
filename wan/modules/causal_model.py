from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from .action_module import ActionModule



def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    f, h, w = grid_sizes.tolist()

    for i in range(len(x)):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).type_as(x)


def _rope_apply_frames(k_pre_rope, frame_ids, grid_sizes, freqs):
    n_frames = len(frame_ids)
    if n_frames == 0 or k_pre_rope.shape[1] == 0:
        return k_pre_rope
    frame_seqlen = k_pre_rope.shape[1] // n_frames
    h = int(grid_sizes[1].item())
    w = int(grid_sizes[2].item())
    parts = []
    for i, fid in enumerate(frame_ids):
        g = torch.tensor([1, h, w], device=grid_sizes.device, dtype=grid_sizes.dtype)
        k_frame = k_pre_rope[:, i * frame_seqlen:(i + 1) * frame_seqlen]
        parts.append(causal_rope_apply(k_frame, g, freqs, start_frame=int(fid)))
    return torch.cat(parts, dim=1)


def _rope_apply_per_token(k_pre_rope, token_frame_ids, grid_sizes, freqs):
    b, s, n, d = k_pre_rope.shape
    c = d // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    h = int(grid_sizes[1].item())
    w = int(grid_sizes[2].item())

    fids = token_frame_ids
    h_idx = torch.arange(s, device=k_pre_rope.device) // w
    w_idx = torch.arange(s, device=k_pre_rope.device) % w

    freqs_i = torch.cat([freqs[0][fids], freqs[1][h_idx], freqs[2][w_idx]], dim=-1)

    x_i = torch.view_as_complex(
        k_pre_rope.to(torch.float64).reshape(b, s, n, c, 2))
    x_i = x_i * freqs_i.view(1, s, 1, c)
    return torch.view_as_real(x_i).flatten(3).type_as(k_pre_rope)


def _rope_apply_per_token_full(k_pre_rope, token_frame_ids, token_sp_pos, grid_sizes, freqs):
    b, s, n, d = k_pre_rope.shape
    c = d // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    h = int(grid_sizes[1].item())
    w = int(grid_sizes[2].item())

    fids = token_frame_ids
    sp = token_sp_pos
    h_idx = sp // w
    w_idx = sp % w

    freqs_i = torch.cat([freqs[0][fids], freqs[1][h_idx], freqs[2][w_idx]], dim=-1)

    x_i = torch.view_as_complex(
        k_pre_rope.to(torch.float64).reshape(b, s, n, c, 2))
    x_i = x_i * freqs_i.view(1, s, 1, c)
    return torch.view_as_real(x_i).flatten(3).type_as(k_pre_rope)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 15 * 1 * 880 if local_attn_size == -1 else local_attn_size * 880
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        qvg_bits=0,
        qvg_num_centroids=256,
        qvg_num_stages=1,
        qvg_group_size=64,
        qvg_kmeans_iters=10,
        qvg_seed=0,
        retrieval_kv=None,
        flowcache=False,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

            padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                dim=1
            )

            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )

            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            assert grid_sizes.ndim == 1
            frame_seqlen = math.prod(grid_sizes[1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            if flowcache:
                kv_cache["flow_q"] = roped_query.detach()
                kv_cache["flow_grid_sizes"] = grid_sizes

            k_pre_rope = k


            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen

            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]

            if (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):

                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] =\
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] =\
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                local_end_index = kv_cache["local_end_index"].item() + current_end -\
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = k_pre_rope
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                kv_cache["k"][:, local_start_index:local_end_index] = k_pre_rope.to(kv_cache["k"].device)
                kv_cache["v"][:, local_start_index:local_end_index] = v.to(kv_cache["v"].device)

            offset = current_end - local_end_index

            def frame_ids(start_token, end_token):
                fids = []
                for p in range(start_token, end_token, frame_seqlen):
                    if p < sink_tokens:
                        fids.append(p // frame_seqlen)
                    else:
                        fids.append((p + offset) // frame_seqlen)
                return fids

            attn_bias = None
            if retrieval_kv is not None:
                sink_k = kv_cache["k"][:, :sink_tokens]
                sink_v = kv_cache["v"][:, :sink_tokens]
                retr_k = retrieval_kv["k"].to(device=roped_query.device, dtype=roped_query.dtype)
                retr_v = retrieval_kv["v"].to(device=roped_query.device, dtype=roped_query.dtype)
                recent_frames = retrieval_kv.get('recent_frames', self.local_attn_size - self.sink_size)
                recent_budget = recent_frames * frame_seqlen
                recent_start = max(sink_tokens, local_end_index - recent_budget)
                recent_k = kv_cache["k"][:, recent_start:local_end_index]
                recent_v = kv_cache["v"][:, recent_start:local_end_index]

                sink_k = _rope_apply_frames(
                    sink_k, frame_ids(0, sink_tokens), grid_sizes, freqs).type_as(v)
                if "token_sp_pos" in retrieval_kv:
                    retr_k = _rope_apply_per_token_full(
                        retr_k, retrieval_kv["token_frame_ids"],
                        retrieval_kv["token_sp_pos"], grid_sizes, freqs).type_as(v)
                elif "token_frame_ids" in retrieval_kv:
                    retr_k = _rope_apply_per_token(
                        retr_k, retrieval_kv["token_frame_ids"], grid_sizes, freqs).type_as(v)
                else:
                    retr_k = _rope_apply_frames(
                        retr_k, retrieval_kv.get("frame_ids", []), grid_sizes, freqs).type_as(v)
                recent_k = _rope_apply_frames(
                    recent_k, frame_ids(recent_start, local_end_index), grid_sizes, freqs).type_as(v)

                k_cache = torch.cat([sink_k, retr_k, recent_k], dim=1)
                v_cache = torch.cat([sink_v, retr_v, recent_v], dim=1)

                log_bias = retrieval_kv.get("log_bias")
                if log_bias is not None:
                    sink_t = sink_k.shape[1]
                    retr_t = retr_k.shape[1]
                    attn_bias = torch.zeros([1, 1, 1, k_cache.shape[1]],
                                            device=k_cache.device, dtype=torch.float32)
                    attn_bias[..., sink_t:sink_t + retr_t] = log_bias.to(
                        attn_bias.device, attn_bias.dtype)
            else:
                recent_start = max(0, local_end_index - self.max_attention_size)
                recent_k = kv_cache["k"][:, recent_start:local_end_index].to(roped_query.device)
                recent_v = kv_cache["v"][:, recent_start:local_end_index].to(roped_query.device)
                recent_k = _rope_apply_frames(
                    recent_k, frame_ids(recent_start, local_end_index), grid_sizes, freqs).type_as(v)
                k_cache = recent_k
                v_cache = recent_v

            x = attention(roped_query, k_cache, v_cache, attn_bias=attn_bias)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 action_config={},
                 block_idx=0,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        if len(action_config) != 0 and block_idx in action_config['blocks']:
            self.action_model = ActionModule(**action_config, local_attn_size=self.local_attn_size)
        else:
            self.action_model = None
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        block_mask,
        block_mask_mouse,
        block_mask_keyboard,
        num_frame_per_block=3,
        use_rope_keyboard=False,
        mouse_cond=None,
        keyboard_cond=None,
        kv_cache=None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        context_lens=None,
        qvg_bits=0,
        qvg_num_centroids=256,
        qvg_num_stages=1,
        qvg_group_size=64,
        qvg_kmeans_iters=10,
        qvg_seed=0,
        retrieval_kv=None,
        flowcache=False,
    ):
        assert e.ndim == 4
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start,
            qvg_bits, qvg_num_centroids, qvg_num_stages, qvg_group_size,
            qvg_kmeans_iters, qvg_seed, retrieval_kv=retrieval_kv,
            flowcache=flowcache)


        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        def cross_attn_ffn(x, context, e, mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard, kv_cache_mouse=None, kv_cache_keyboard=None, crossattn_cache=None, start_frame=0, use_rope_keyboard=False, num_frame_per_block=3):
            x = x + self.cross_attn(self.norm3(x.to(context.dtype)), context, crossattn_cache=crossattn_cache)
            if self.action_model is not None:
                assert mouse_cond is not None or keyboard_cond is not None
                x = self.action_model(x.to(context.dtype), grid_sizes[0], grid_sizes[1], grid_sizes[2], mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard, is_causal=True, kv_cache_mouse=kv_cache_mouse, kv_cache_keyboard=kv_cache_keyboard, start_frame=start_frame, use_rope_keyboard=use_rope_keyboard, num_frame_per_block=num_frame_per_block)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x
        assert grid_sizes.ndim == 1
        x = cross_attn_ffn(x, context, e, mouse_cond, keyboard_cond, block_mask_mouse, block_mask_keyboard, kv_cache_mouse, kv_cache_keyboard, crossattn_cache, start_frame=current_start // math.prod(grid_sizes[1:]).item(), use_rope_keyboard=use_rope_keyboard, num_frame_per_block=num_frame_per_block)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=36,
                 dim=1536,
                 ffn_dim=8960,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=12,
                 num_layers=30,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 action_config={},
                 eps=1e-6):

        super().__init__()

        assert model_type in ['i2v']
        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, action_config=action_config, eps=eps, block_idx=idx)
            for idx in range(num_layers)
        ])

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self.block_mask_keyboard = None
        self.block_mask_mouse = None
        self.use_rope_keyboard = True
        self.num_frame_per_block = 1

        self.qvg_bits = 2
        self.qvg_num_centroids = 256
        self.qvg_num_stages = 1
        self.qvg_group_size = 64
        self.qvg_kmeans_iters = 10
        self.qvg_seed = 0
        self.qvg_sparsify = False
        self.qvg_sparse_ratio = 0.5
        self.qvg_use_triton = True
        self.qvg_sensitivity_wk = None
        self.qvg_sensitivity_wv = None
        self.qvg_gamma = 1.0

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 880, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        total_length = num_frames * frame_seqlen

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp +\
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_keyboard(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 880, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        total_length2 = num_frames * frame_seqlen

        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2,
                           device=device, dtype=torch.long)

        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)

        block_mask2 = create_block_mask(attention_mask2, B=None, H=None, Q_LEN=total_length2 + padded_length2,
                                       KV_LEN=num_frames + padded_length_kv2, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_action(
        device: torch.device | str, num_frames: int = 9,
        frame_seqlen: int = 1, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        total_length2 = num_frames * frame_seqlen

        padded_length2 = math.ceil(total_length2 / 32) * 32 - total_length2
        padded_length_kv2 = math.ceil(num_frames / 32) * 32 - num_frames
        ends2 = torch.zeros(total_length2 + padded_length2,
                           device=device, dtype=torch.long)

        frame_indices2 = torch.arange(
            start=0,
            end=total_length2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        cnt = num_frame_per_block
        for tmp in frame_indices2:
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = cnt
            cnt += num_frame_per_block

        def attention_mask2(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends2[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size))) | (q_idx == kv_idx)

        block_mask2 = create_block_mask(attention_mask2, B=None, H=None, Q_LEN=total_length2 + padded_length2,
                                       KV_LEN=num_frames + padded_length_kv2, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")

        return block_mask2

    def _forward_inference(
        self,
        x,
        t,
        visual_context, cond_concat, mouse_cond=None, keyboard_cond=None,
        kv_cache: dict = None,
        kv_cache_mouse=None,
        kv_cache_keyboard=None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        retrieval_kv=None,
        flowcache=False,
    ):

        if mouse_cond is not None or keyboard_cond is not None:
            assert self.use_action_module == True
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        x = torch.cat([x, cond_concat], dim=1)

        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        assert seq_lens[0] <= 15 * 1 * 880
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        context_lens = None
        context = self.img_emb(visual_context)
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            mouse_cond=mouse_cond,
            context_lens=context_lens,
            keyboard_cond=keyboard_cond,
            block_mask=self.block_mask,
            block_mask_mouse=self.block_mask_mouse,
            block_mask_keyboard=self.block_mask_keyboard,
            use_rope_keyboard=self.use_rope_keyboard,
            num_frame_per_block=self.num_frame_per_block,
            qvg_bits=self.qvg_bits,
            qvg_num_centroids=self.qvg_num_centroids,
            qvg_num_stages=self.qvg_num_stages,
            qvg_group_size=self.qvg_group_size,
            qvg_kmeans_iters=self.qvg_kmeans_iters,
            qvg_seed=self.qvg_seed,
            flowcache=flowcache,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "kv_cache_mouse": kv_cache_mouse[block_index],
                        "kv_cache_keyboard": kv_cache_keyboard[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "retrieval_kv": retrieval_kv[block_index] if retrieval_kv else None,
                        "flowcache": flowcache,
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "kv_cache_mouse": kv_cache_mouse[block_index],
                        "kv_cache_keyboard": kv_cache_keyboard[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "retrieval_kv": retrieval_kv[block_index] if retrieval_kv else None,
                        "flowcache": flowcache,
                    }
                )
                x = block(x, **kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return x

    def _forward_train(
        self,
        x,
        t,
        visual_context, cond_concat, mouse_cond=None, keyboard_cond=None,
    ):
        if mouse_cond is not None or keyboard_cond is not None:
            assert self.use_action_module == True
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        x = torch.cat([x, cond_concat], dim=1)
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames=x.shape[2],
                frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )
        if self.block_mask_keyboard is None:
            if self.use_rope_keyboard==False:
                self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_keyboard(
                    device, num_frames=x.shape[2],
                    frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]) ,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
                )
            else:
                self.block_mask_keyboard = self._prepare_blockwise_causal_attn_mask_action(
                    device, num_frames=x.shape[2],
                    frame_seqlen=1,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size
            )
        if self.block_mask_mouse is None:
            self.block_mask_mouse = self._prepare_blockwise_causal_attn_mask_action(
                device, num_frames=x.shape[2],
                frame_seqlen=1,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=self.local_attn_size
            )
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        context_lens = None
        context = self.img_emb(visual_context)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            mouse_cond=mouse_cond,
            context_lens=context_lens,
            keyboard_cond=keyboard_cond,
            block_mask=self.block_mask,
            block_mask_mouse=self.block_mask_mouse,
            block_mask_keyboard=self.block_mask_keyboard,
            use_rope_keyboard=self.use_rope_keyboard,
            num_frame_per_block=self.num_frame_per_block
            )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)


        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return x

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):

        c = self.out_dim
        bs = x.shape[0]
        x = x.view(bs, *grid_sizes, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return x

    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)
        if self.use_action_module == True:
            for m in self.blocks:
                try:
                    nn.init.zeros_(m.action_model.proj_mouse.weight)
                    if m.action_model.proj_mouse.bias is not None:
                        nn.init.zeros_(m.action_model.proj_mouse.bias)
                    nn.init.zeros_(m.action_model.proj_keyboard.weight)
                    if m.action_model.proj_keyboard.bias is not None:
                        nn.init.zeros_(m.action_model.proj_keyboard.bias)
                except:
                    pass
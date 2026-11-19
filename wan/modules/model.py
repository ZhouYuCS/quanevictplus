import math
import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat, rearrange
from .action_module import ActionModule
from .attention import flash_attention
DISABLE_COMPILE = False
__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    f, h, w = grid_sizes.tolist()
    for i in range(len(x)):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        x = x.flatten(2)
        x = self.o(x)
        return x
















class WanI2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=None)

        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    'i2v_cross_attn': WanI2VCrossAttention,
}
def mul_add(x, y, z):
    return x.float() + y.float() * z.float()


def mul_add_add(x, y, z):
    return x.float() * (1 + y) + z

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 action_config={},
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        if len(action_config) != 0:
            self.action_model = ActionModule(**action_config)
        else:
            self.action_model = None
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
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
        mouse_cond=None,
        keyboard_cond=None
    ):
        if e.dim() == 3:
            modulation = self.modulation
            e = (self.modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)
            e = (modulation + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]

        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        x = x + y * e[2]

        def cross_attn_ffn(x, context, e, mouse_cond, keyboard_cond):
            dtype = context.dtype
            x = x + self.cross_attn(self.norm3(x.to(dtype)), context)
            if self.action_model is not None:
                assert mouse_cond is not None or keyboard_cond is not None
                x = self.action_model(x.to(dtype), grid_sizes[0], grid_sizes[1], grid_sizes[2], mouse_cond, keyboard_cond)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, e, mouse_cond, keyboard_cond)
        return x



class Head(nn.Module):

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
        if e.dim() == 2:
                modulation = self.modulation
                e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            modulation = self.modulation.unsqueeze(2)
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens






class WanModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='i2v',
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
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 inject_sample_info=False,
                 action_config={},
                 eps=1e-6):

        super().__init__()

        assert model_type in ['i2v']
        self.model_type = model_type
        self.use_action_module = len(action_config) > 0
        assert self.use_action_module == True
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
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = -1

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps=eps, action_config=action_config)
            for _ in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)

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

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        *args,
        **kwargs
    ):
        return self._forward(*args, **kwargs)

    def _forward(
        self,
        x,
        t,
        visual_context,
        cond_concat,
        mouse_cond=None, keyboard_cond=None, fps=None
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

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))

        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context_lens = None

        context = self.img_emb(visual_context)

        kwargs = dict(
            e=e0,
            grid_sizes=grid_sizes,
            seq_lens=seq_lens,
            freqs=self.freqs,
            context=context,
            mouse_cond=mouse_cond,
            keyboard_cond=keyboard_cond)
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)


        x = self.head(x, e)

        x = self.unpatchify(x, grid_sizes)

        return x.float()
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
                nn.init.zeros_(m.action_model.proj_mouse.weight)
                if m.action_model.proj_mouse.bias is not None:
                    nn.init.zeros_(m.action_model.proj_mouse.bias)
                nn.init.zeros_(m.action_model.proj_keyboard.weight)
                if m.action_model.proj_keyboard.bias is not None:
                    nn.init.zeros_(m.action_model.proj_keyboard.bias)
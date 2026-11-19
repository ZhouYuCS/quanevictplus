import torch
from typing import Union, Tuple, List


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32, device=torch.cuda.current_device())[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)

    return grid




def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            shape = [1, freqs_cis[0].shape[0], 1, freqs_cis[0].shape[1]]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
    start_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xk_out = None
    assert isinstance(freqs_cis, tuple)
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        xq_out = (xq.float() * cos[:, start_offset:start_offset + xq.shape[1], :, :] + rotate_half(xq.float()) * sin[:, start_offset:start_offset + xq.shape[1], :, :]).type_as(xq)
        xk_out = (xk.float() * cos[:, start_offset:start_offset + xk.shape[1], :, :] + rotate_half(xk.float()) * sin[:, start_offset:start_offset + xk.shape[1], :, :]).type_as(xk)
    else:
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], -1, 2)
        )
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(pos, int):
        pos = torch.arange(pos, device=torch.cuda.current_device()).float()

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=torch.cuda.current_device())[: (dim // 2)].float() / dim)
    )
    freqs = torch.outer(pos * interpolation_factor, freqs)
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )
        return freqs_cis
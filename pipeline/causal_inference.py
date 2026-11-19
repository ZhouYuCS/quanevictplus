from typing import List, Optional
import numpy as np
import torch
import time
import copy
import gc

from einops import rearrange
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from utils.visualize import process_video
import torch.nn.functional as F
from demo_utils.constant import ZERO_VAE_CACHE
from tqdm import tqdm
from wan.utils.qvg import (
    qvg_compress, qvg_decompress_frames, quanevict_ratios,
    _packed_breakdown, reset_qvg_profile, _QVG_PROFILE,
)
from wan.modules.causal_model import _rope_apply_per_token_full


def _move_packed(packed, device):
    if packed is None:
        return None
    out = {}
    for key, value in packed.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, list):
            out[key] = [t.to(device) for t in value]
        else:
            out[key] = value
    return out

def _packed_bytes(packed):
    if packed is None:
        return 0
    total = 0
    for value in packed.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
        elif isinstance(value, list):
            total += sum(t.numel() * t.element_size() for t in value)
    return total


def _flowcache_compress(fc_k, fc_v, fc_fids, fc_sp_pos, query, grid_sizes, freqs,
                        budget, lam, pool_k):
    T = fc_k.shape[1]
    if T <= budget:
        return fc_k, fc_v, fc_fids, fc_sp_pos
    n, d = fc_k.shape[2], fc_k.shape[3]
    K = _rope_apply_per_token_full(fc_k, fc_fids, fc_sp_pos, grid_sizes, freqs).float()
    Q = query.float()
    nq = min(50, Q.shape[1])
    Qc = Q[:, -nq:]
    scale = d ** -0.5
    A = torch.einsum('qnd,tnd->nqt', Qc[0], K[0]) * scale
    A = A.softmax(dim=-1)
    Imp = A.mean(dim=1)
    Imp = F.max_pool1d(Imp.unsqueeze(0), kernel_size=pool_k, stride=1,
                       padding=pool_k // 2)[0]
    K_norm = F.normalize(K, dim=-1)
    mean_k = K_norm.mean(dim=1)
    Red = torch.einsum('nd,tnd->nt', mean_k[0], K_norm[0]).softmax(dim=-1)
    Score = lam * Imp - (1.0 - lam) * Red
    Score_token = Score.mean(dim=0)
    B = min(budget, T)
    top_idx = Score_token.topk(B, largest=True).indices.sort().values
    return fc_k[:, top_idx], fc_v[:, top_idx], fc_fids[top_idx], fc_sp_pos[top_idx]


def _worldkv_compress_chunk(k_chunk, block_size, frame_seqlen, retain_ratio):
    T = frame_seqlen
    anchor = k_chunk[:, :T]
    non = k_chunk[:, T:]
    anchor_n = F.normalize(anchor.float(), dim=-1)
    non_n = F.normalize(non.float(), dim=-1)
    mean_anchor = anchor_n.mean(dim=1)
    s = (non_n * mean_anchor[:, None]).sum(dim=-1).mean(dim=-1)[0]
    n_non = non.shape[1]
    keep_n = max(1, min(n_non, int(round(n_non * retain_ratio))))
    _, rel = s.topk(keep_n, largest=False)
    rel = rel.sort().values
    keep_non_abs = rel + T

    keep_per_frame = []
    sp_per_frame = []
    for j in range(block_size):
        if j == 0:
            keep_j = torch.arange(T, device=k_chunk.device, dtype=torch.long)
        else:
            lo, hi = j * T, (j + 1) * T
            keep_j = keep_non_abs[(keep_non_abs >= lo) & (keep_non_abs < hi)] - lo
        keep_per_frame.append(keep_j)
        sp_per_frame.append(keep_j.clone())
    return keep_per_frame, sp_per_frame


def _lsh_codes(x, R, powers):
    bits = (x @ R > 0).long()
    return (bits * powers).sum(dim=1)


_LSH_PROJECTION_CACHE = {}


def _lsh_projections(lsh_seed, lsh_tables, lsh_bits, d, device):
    key = (lsh_seed, lsh_tables, lsh_bits, d, str(device))
    projs = _LSH_PROJECTION_CACHE.get(key)
    if projs is None:
        gen = torch.Generator(device=device).manual_seed(lsh_seed)
        projs = [
            torch.randn(d, lsh_bits, generator=gen, device=device,
                        dtype=torch.float32)
            for _ in range(lsh_tables)
        ]
        _LSH_PROJECTION_CACHE[key] = projs
    return projs


def _tempcache_merge_full(bank_k, bank_k_n, bank_v_sum, bank_count, bank_fids, bank_sp_pos,
                          kf, vf, global_fid, threshold,
                          lsh_tables=0, lsh_bits=8, lsh_seed=0,
                          lsh_bucket_cap=32):
    if bank_k is None:
        S = kf.shape[1]
        sp = torch.arange(S, device=kf.device, dtype=torch.long)
        kf_n = torch.nn.functional.normalize(kf.float(), dim=-1).to(kf.dtype)
        return (
            kf.clone(),
            kf_n,
            vf.clone(),
            torch.ones(S, device=kf.device, dtype=torch.float32),
            torch.full([S], global_fid, device=kf.device, dtype=torch.long),
            sp,
        )

    kf_n = torch.nn.functional.normalize(kf.float(), dim=-1).to(kf.dtype)
    S = kf_n.shape[1]
    g = bank_k_n.shape[1]
    max_sim = torch.full([S], -float('inf'), device=kf.device, dtype=torch.float32)
    argmax = torch.zeros([S], device=kf.device, dtype=torch.long)
    if lsh_tables > 0:
        d = bank_k_n.shape[-1]
        bank_agg = torch.nn.functional.normalize(
            bank_k_n.mean(dim=2).float(), dim=-1)[0]
        kf_agg = torch.nn.functional.normalize(
            kf_n.mean(dim=2).float(), dim=-1)[0]
        powers = 1 << torch.arange(lsh_bits, device=kf.device)
        chunk_q = 64
        for R in _lsh_projections(lsh_seed, lsh_tables, lsh_bits, d, kf.device):
            bank_codes = _lsh_codes(bank_agg, R, powers)
            q_codes = _lsh_codes(kf_agg, R, powers)
            order = torch.argsort(bank_codes, stable=True)
            sorted_codes = bank_codes[order]
            lo = torch.searchsorted(sorted_codes, q_codes, side='left')
            hi = torch.searchsorted(sorted_codes, q_codes, side='right')
            sizes = hi - lo
            candidate_cap = max(1, int(lsh_bucket_cap))
            for qs in range(0, S, chunk_q):
                qe = min(qs + chunk_q, S)
                kmax = candidate_cap
                kk = torch.arange(kmax, device=kf.device)
                cand = lo[qs:qe, None] + kk[None, :]
                valid = kk[None, :] < sizes[qs:qe, None]
                safe_mask = valid & (cand >= 0) & (cand < order.numel())
                cand_safe = torch.where(safe_mask, cand, torch.zeros_like(cand))
                cand_idx = order[cand_safe]
                blk = bank_k_n[0, cand_idx]
                sim = torch.einsum('snd,sknd->skn',
                                   kf_n[0, qs:qe], blk).mean(dim=-1).float()
                sim = torch.where(safe_mask, sim, float('-inf'))
                blk_max, blk_arg = sim.max(dim=1)
                rows = torch.arange(qs, qe, device=kf.device)
                local_rows = rows - qs
                better = blk_max > max_sim[qs:qe]
                better_rows = rows[better]
                max_sim[better_rows] = blk_max[better]
                valid_idx = cand_idx[local_rows[better], blk_arg[better]]
                argmax[better_rows] = valid_idx
                del cand, valid, safe_mask, cand_safe, cand_idx, blk, sim, blk_max, blk_arg, rows, local_rows, better
            del bank_codes, q_codes, order, sorted_codes, lo, hi, sizes
        del bank_agg, kf_agg, powers
    else:
        chunk = 1024
        for start in range(0, g, chunk):
            end = min(start + chunk, g)
            blk = bank_k_n[0, start:end]
            sim_blk = torch.einsum('snd,gnd->sgn', kf_n[0], blk).mean(dim=-1).float()
            blk_max, blk_arg = sim_blk.max(dim=1)
            better = blk_max > max_sim
            max_sim[better] = blk_max[better]
            argmax[better] = start + blk_arg[better]
            del blk, sim_blk, blk_max, blk_arg, better
    merge = max_sim >= threshold
    del max_sim

    new_pos = torch.nonzero(~merge, as_tuple=False).squeeze(1)
    if new_pos.numel() > 0:
        bank_k = torch.cat([bank_k, kf[:, new_pos]], dim=1)
        bank_k_n = torch.cat([bank_k_n, kf_n[:, new_pos]], dim=1)
        bank_v_sum = torch.cat([bank_v_sum, vf[:, new_pos]], dim=1)
        bank_count = torch.cat([
            bank_count,
            torch.ones(new_pos.numel(), device=kf.device, dtype=torch.float32)])
        bank_fids = torch.cat([
            bank_fids,
            torch.full([new_pos.numel()], global_fid, device=kf.device, dtype=torch.long)])
        bank_sp_pos = torch.cat([bank_sp_pos, new_pos])

    m_pos = torch.nonzero(merge, as_tuple=False).squeeze(1)
    if m_pos.numel() > 0:
        m_target = argmax[m_pos]
        bank_v_sum.index_add_(1, m_target, vf[:, m_pos])
        bank_count.index_add_(
            0, m_target,
            torch.ones(m_pos.numel(), device=kf.device, dtype=torch.float32))
        latest_pos = torch.full(
            [bank_k.shape[1]], -1, device=kf.device, dtype=torch.long)
        latest_pos.scatter_reduce_(0, m_target, m_pos, reduce='amax')
        upd = latest_pos >= 0
        upd_idx = torch.nonzero(upd, as_tuple=False).squeeze(1)
        bank_k[:, upd_idx] = kf[:, latest_pos[upd_idx]]
        bank_k_n[:, upd_idx] = kf_n[:, latest_pos[upd_idx]]
        bank_sp_pos[upd_idx] = latest_pos[upd_idx]
        bank_fids[upd_idx] = global_fid

    return bank_k, bank_k_n, bank_v_sum, bank_count, bank_fids, bank_sp_pos


def get_current_action(mode="universal"):

    CAM_VALUE = 0.1
    if mode == 'universal':
        print()
        print('-'*30)
        print("PRESS [I, K, J, L, U] FOR CAMERA TRANSFORM\n (I: up, K: down, J: left, L: right, U: no move)")
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "i":  [CAM_VALUE, 0],
            "k":  [-CAM_VALUE, 0],
            "j":  [0, -CAM_VALUE],
            "l":  [0, CAM_VALUE],
            "u":  [0, 0]
        }
        KEYBOARD_IDX = {
            "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
            "q": [0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_mouse = input('Please input the mouse action (e.g. `U`):\n').strip().lower()
                idx_keyboard = input('Please input the keyboard action (e.g. `W`):\n').strip().lower()
                if idx_mouse in CAMERA_VALUE_MAP.keys() and idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    elif mode == 'gta_drive':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Q] FOR MOVEMENT\n (W: forward, S: back, A: left, D: right, Q: no move)")
        print('-'*30)
        CAMERA_VALUE_MAP = {
            "a":  [0, -CAM_VALUE],
            "d":  [0, CAM_VALUE],
            "q":  [0, 0]
        }
        KEYBOARD_IDX = {
            "w": [1, 0], "s": [0, 1],
            "q": [0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                indexes = input('Please input the actions (split with ` `):\n(e.g. `W` for forward, `W A` for forward and left)\n').strip().lower().split(' ')
                idx_mouse = []
                idx_keyboard = []
                for i in indexes:
                    if i in CAMERA_VALUE_MAP.keys():
                        idx_mouse += [i]
                    elif i in KEYBOARD_IDX.keys():
                        idx_keyboard += [i]
                if len(idx_mouse) == 0:
                    idx_mouse += ['q']
                if len(idx_keyboard) == 0:
                    idx_keyboard += ['q']
                assert idx_mouse in [['a'], ['d'], ['q']] and idx_keyboard in [['q'], ['w'], ['s']]
                flag = 1
            except:
                pass
        mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse[0]]).cuda()
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard[0]]).cuda()
    elif mode == 'templerun':
        print()
        print('-'*30)
        print("PRESS [W, S, A, D, Z, C, Q] FOR ACTIONS\n (W: jump, S: slide, A: left side, D: right side, Z: turn left, C: turn right, Q: no move)")
        print('-'*30)
        KEYBOARD_IDX = {
            "w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
            "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
            "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
            "q": [1, 0, 0, 0, 0, 0, 0]
        }
        flag = 0
        while flag != 1:
            try:
                idx_keyboard = input('Please input the action: \n(e.g. `W` for forward, `Z` for turning left)\n').strip().lower()
                if idx_keyboard in KEYBOARD_IDX.keys():
                    flag = 1
            except:
                pass
        keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()
    if mode != 'templerun':
        return {
            "mouse": mouse_cond,
            "keyboard": keyboard_cond
        }
    return {
        "keyboard": keyboard_cond
    }

def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode='universal'):
    new_cond = {}
    new_cond["cond_concat"] = conditional_dict["cond_concat"][:, :, current_start_frame: current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace != None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block-1)
        if mode != 'templerun':
            conditional_dict["mouse_cond"][:, -last_frame_num + final_frame: final_frame] = replace['mouse'][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["keyboard_cond"][:, -last_frame_num + final_frame: final_frame] = replace['keyboard'][None, None, :].repeat(1, last_frame_num, 1)
    if mode != 'templerun':
        new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace != None:
        return new_cond, conditional_dict
    else:
        return new_cond

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            generator=None,
            vae_decoder=None,
    ):
        super().__init__()
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.vae_decoder = vae_decoder
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        self.sink_size = getattr(self.generator.model, 'sink_size', 0)
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent = None,
        return_latents = False,
        mode = 'universal',
        profile = False,
    ) -> torch.Tensor:
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None

        self.kv_cache1 = self.kv_cache_keyboard = self.kv_cache_mouse = self.crossattn_cache=None
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents =\
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block


        all_num_frames = [self.num_frame_per_block] * num_blocks
        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
        for current_num_frames in tqdm(all_num_frames):

            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            if profile:
                torch.cuda.synchronize()
                diffusion_start.record()
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, mode=mode),
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += current_num_frames

            denoised_pred = denoised_pred.transpose(1,2)
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video]

            if profile:
                torch.cuda.synchronize()
                diffusion_end.record()
                diffusion_time = diffusion_start.elapsed_time(diffusion_end)
                print(f"diffusion_time: {diffusion_time}", flush=True)
                fps = video.shape[1]*1000/ diffusion_time
                print(f"  - FPS: {fps:.2f}")

        if return_latents:
            return output
        else:
            return videos

    def _initialize_kv_cache(self, batch_size, dtype, device):
        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 15 * 1 * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard
        self.kv_cache_mouse = kv_cache_mouse


    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache


class CausalInferenceStreamingPipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device="cuda",
            vae_decoder=None,
            generator=None,
    ):
        super().__init__()
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.vae_decoder = vae_decoder

        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        self.sink_size = getattr(self.generator.model, 'sink_size', 0)
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        output_folder = None,
        name = None,
        mode = 'universal',
        scripted_actions = None,
        use_retrieval = False,
        retrieval_topk_frames = 4,
        traj_threshold = None,
        kv_bank_on_gpu = False,
        quant_chunk_frames = 12,
        worldkv = False,
        worldkv_compress = False,
        worldkv_retain_ratio = 0.25,
        fastar_full = False,
        fastar_sim_threshold = 0.8,
        fastar_lsh_tables = 1,
        fastar_lsh_bits = 10,
        fastar_lsh_seed = 0,
        fastar_lsh_bucket_cap = 128,
        residual_mag_sink = None,
        flowcache = False,
        flowcache_budget = 10000,
        flowcache_lambda = 0.07,
        flowcache_maxpool_kernel = 3,
        full_kv = False,
        cache_transfer = False,
        profiling = False,
    ) -> torch.Tensor:
        kv_bank_on_gpu = kv_bank_on_gpu and not cache_transfer

        wan_t0 = time.perf_counter()

        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        videos = []
        vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for j in range(len(vae_cache)):
            vae_cache[j] = None
        self.kv_cache1=self.kv_cache_keyboard=self.kv_cache_mouse=self.crossattn_cache=None
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_frames=num_output_frames if full_kv else None,
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        if full_kv:
            self._full_kv_saved_max = [
                blk.self_attn.max_attention_size for blk in self.generator.model.blocks]
            for blk in self.generator.model.blocks:
                blk.self_attn.max_attention_size = num_output_frames * self.frame_seq_length
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents =\
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=True),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        block_size = self.num_frame_per_block
        frame_seqlen = self.frame_seq_length
        assert quant_chunk_frames % block_size == 0,\
            "quant_chunk_frames must be a multiple of num_frame_per_block"
        self.quant_chunk_frames = quant_chunk_frames
        sink_frames = self.sink_size
        sink_blocks = (self.sink_size + block_size - 1) // block_size
        bank_base_frame = sink_blocks * block_size
        recent_frames = max(1, self.local_attn_size - self.sink_size)

        kv_bank_k = [[] for _ in range(self.num_transformer_blocks)]
        kv_bank_v = [[] for _ in range(self.num_transformer_blocks)]
        kv_bank_sp = [[] for _ in range(self.num_transformer_blocks)]
        bank_frame_poses = []

        acc_k = [[] for _ in range(self.num_transformer_blocks)]
        acc_v = [[] for _ in range(self.num_transformer_blocks)]
        acc_frames = 0
        frame_pose_acc = []

        running_state = torch.zeros(4, device=noise.device, dtype=torch.float32)

        tc_full_k = [None] * self.num_transformer_blocks
        tc_full_k_n = [None] * self.num_transformer_blocks
        tc_full_v_sum = [None] * self.num_transformer_blocks
        tc_full_count = [None] * self.num_transformer_blocks
        tc_full_fids = [None] * self.num_transformer_blocks
        tc_full_sp_pos = [None] * self.num_transformer_blocks

        fc_k = [None] * self.num_transformer_blocks
        fc_v = [None] * self.num_transformer_blocks
        fc_fids = [None] * self.num_transformer_blocks
        fc_sp_pos = [None] * self.num_transformer_blocks
        fc_acc_k = [None] * self.num_transformer_blocks
        fc_acc_v = [None] * self.num_transformer_blocks
        fc_acc_fids = [None] * self.num_transformer_blocks
        fc_acc_sp_pos = [None] * self.num_transformer_blocks
        fc_acc_frames = 0

        self._last_bank_bytes = 0
        self._last_vae_latency = 0.0
        if profiling:
            reset_qvg_profile()
            self._prof = {"forward": 0.0, "quant": 0.0, "dequant": 0.0, "evict": 0.0, "vae": 0.0, "retrieval": 0.0}
            self._bank_prof = {}

        m = self.generator.model
        qvg_kwargs = dict(
            bits=m.qvg_bits,
            num_centroids=m.qvg_num_centroids,
            num_stages=m.qvg_num_stages,
            group_size=m.qvg_group_size,
            kmeans_iters=m.qvg_kmeans_iters,
            seed=m.qvg_seed,
            sparsify=m.qvg_sparsify,
            sparse_ratio=m.qvg_sparse_ratio,
            random_evict=getattr(m, 'qvg_random_evict', False),
            use_triton=m.qvg_use_triton,
            num_frames=quant_chunk_frames,
            return_residual_mag=residual_mag_sink is not None,
        )

        ratio_k = ratio_v = None
        if m.qvg_sparsify and m.qvg_sensitivity_wk and m.qvg_sensitivity_wv:
            wk = torch.from_numpy(np.load(m.qvg_sensitivity_wk)).to(noise.device)
            wv = torch.from_numpy(np.load(m.qvg_sensitivity_wv)).to(noise.device)
            L, H = self.num_transformer_blocks, m.num_heads
            assert wk.shape == (L, H), f"wK sensitivity shape {tuple(wk.shape)} does not match ({L},{H})"
            assert wv.shape == (L, H), f"wV sensitivity shape {tuple(wv.shape)} does not match ({L},{H})"
            if getattr(m, 'qvg_shared_wbar', False):
                w_bar = torch.cat([wk.ravel(), wv.ravel()]).mean()
                ratio_k = quanevict_ratios(wk, m.qvg_sparse_ratio, m.qvg_gamma, w_bar=w_bar)
                ratio_v = quanevict_ratios(wv, m.qvg_sparse_ratio, m.qvg_gamma, w_bar=w_bar)
            else:
                ratio_k = quanevict_ratios(wk, m.qvg_sparse_ratio, m.qvg_gamma)
                ratio_v = quanevict_ratios(wv, m.qvg_sparse_ratio, m.qvg_gamma)

        all_num_frames = [self.num_frame_per_block] * num_blocks
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            block_id = current_start_frame // self.num_frame_per_block
            if scripted_actions is not None and block_id < len(scripted_actions):
                current_actions = scripted_actions[block_id]
            else:
                current_actions = get_current_action(mode=mode)
            new_act, conditional_dict = cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=current_actions, mode=mode)

            retrieval_kv = None
            query_state = None
            frame_deltas = None
            if profiling:
                _t_retr = time.perf_counter()
            if (use_retrieval or worldkv or fastar_full) and mode != 'templerun':
                kb = current_actions["keyboard"].float()
                delta = torch.zeros(4, device=noise.device, dtype=torch.float32)
                delta[0] = kb[0] - kb[1]
                delta[1] = kb[3] - kb[2]
                if "mouse" in current_actions:
                    delta[2:] = current_actions["mouse"].float()
                query_state = running_state + block_size * delta
                frame_deltas = torch.stack(
                    [running_state + (j + 1) * delta for j in range(block_size)], dim=0)

                if fastar_full:
                    if tc_full_k[0] is not None:
                        retrieval_kv = []
                        for layer_idx in range(self.num_transformer_blocks):
                            bk = tc_full_k[layer_idx].to(noise.device)
                            bv = tc_full_v_sum[layer_idx].to(noise.device)
                            bc = tc_full_count[layer_idx].to(noise.device)
                            bf = tc_full_fids[layer_idx]
                            bsp = tc_full_sp_pos[layer_idx]
                            v_mean = bv / bc.view(1, -1, 1, 1).clamp_min(1)
                            log_bias = torch.log(bc.clamp_min(1))
                            retrieval_kv.append({
                                'k': bk, 'v': v_mean,
                                'token_frame_ids': bf,
                                'token_sp_pos': bsp,
                                'recent_frames': recent_frames,
                                'log_bias': log_bias,
                            })

                recent_start_frame = max(sink_frames, current_start_frame - recent_frames)
                num_cand = min(recent_start_frame - sink_frames, len(bank_frame_poses))
                if num_cand > 0:
                    cand_poses = torch.stack(bank_frame_poses[:num_cand], dim=0)
                    trans_d = ((cand_poses[:, :2] - query_state[:2]) ** 2).sum(dim=-1)
                    rot_d = ((cand_poses[:, 2:] - query_state[2:]) ** 2).sum(dim=-1)
                    trans_n = trans_d / trans_d.max() if trans_d.max() > 0 else trans_d
                    rot_n = rot_d / rot_d.max() if rot_d.max() > 0 else rot_d
                    dists = 0.5 * trans_n + 0.5 * rot_n
                    if traj_threshold is not None:
                        keep = dists < traj_threshold
                        if keep.any():
                            kept_idx = keep.nonzero(as_tuple=False).squeeze(1)
                            _, rel_order = dists[kept_idx].topk(
                                min(retrieval_topk_frames, kept_idx.numel()), largest=False)
                            sel_frame_ids = [int(kept_idx[i]) for i in rel_order.tolist()]
                        else:
                            sel_frame_ids = []
                    else:
                        k = min(retrieval_topk_frames, num_cand)
                        _, top_idx = dists.topk(k, largest=False)
                        sel_frame_ids = [int(i) for i in top_idx.tolist()]
                    if sel_frame_ids:
                        print(f"[SparseTraj] frame {current_start_frame}: attend frames {sel_frame_ids} "
                              f"(query={[round(x, 2) for x in query_state.tolist()]})", flush=True)
                        retrieval_kv = []
                        for layer_idx in range(self.num_transformer_blocks):
                            token_frame_ids = None
                            token_sp_pos = None
                            if worldkv:
                                if worldkv_compress:
                                    k_parts, v_parts, fid_parts, sp_parts = [], [], [], []
                                    for fid in sel_frame_ids:
                                        k_parts.append(kv_bank_k[layer_idx][fid].to(noise.device))
                                        v_parts.append(kv_bank_v[layer_idx][fid].to(noise.device))
                                        sp = kv_bank_sp[layer_idx][fid].to(noise.device)
                                        sp_parts.append(sp)
                                        fid_parts.append(torch.full_like(sp, bank_base_frame + fid))
                                    retr_k = torch.cat(k_parts, dim=1)
                                    retr_v = torch.cat(v_parts, dim=1)
                                    token_frame_ids = torch.cat(fid_parts)
                                    token_sp_pos = torch.cat(sp_parts)
                                    cat_frame_ids = None
                                else:
                                    k_parts = [kv_bank_k[layer_idx][fid].to(noise.device) for fid in sel_frame_ids]
                                    v_parts = [kv_bank_v[layer_idx][fid].to(noise.device) for fid in sel_frame_ids]
                                    cat_frame_ids = [bank_base_frame + fid for fid in sel_frame_ids]
                                    retr_k = torch.cat(k_parts, dim=1)
                                    retr_v = torch.cat(v_parts, dim=1)
                            else:
                                chunks_frames = {}
                                for fid in sel_frame_ids:
                                    cid = fid // quant_chunk_frames
                                    local_fid = fid % quant_chunk_frames
                                    chunks_frames.setdefault(cid, []).append(local_fid)
                                ref_packed = kv_bank_k[layer_idx][next(iter(chunks_frames))]
                                head_num, head_dim = ref_packed['orig_shape'][-2], ref_packed['orig_shape'][-1]
                                total_tokens = len(sel_frame_ids) * frame_seqlen
                                retr_k = torch.empty([1, total_tokens, head_num, head_dim],
                                                     device=noise.device, dtype=noise.dtype)
                                retr_v = torch.empty([1, total_tokens, head_num, head_dim],
                                                     device=noise.device, dtype=noise.dtype)
                                cat_frame_ids = []
                                tok = 0
                                for cid, local_fids in chunks_frames.items():
                                    n_tok = len(local_fids) * frame_seqlen
                                    if profiling:
                                        _t_dq = time.perf_counter()
                                    qvg_decompress_frames(
                                        _move_packed(kv_bank_k[layer_idx][cid], noise.device),
                                        local_fids, out=retr_k[0, tok:tok + n_tok], use_triton=m.qvg_use_triton)
                                    qvg_decompress_frames(
                                        _move_packed(kv_bank_v[layer_idx][cid], noise.device),
                                        local_fids, out=retr_v[0, tok:tok + n_tok], use_triton=m.qvg_use_triton)
                                    if profiling:
                                        self._prof["dequant"] += time.perf_counter() - _t_dq
                                    cat_frame_ids.extend([bank_base_frame + cid * quant_chunk_frames + lf for lf in local_fids])
                                    tok += n_tok
                            rv = {'k': retr_k, 'v': retr_v, 'recent_frames': recent_frames}
                            if worldkv and worldkv_compress:
                                rv['token_frame_ids'] = token_frame_ids
                                rv['token_sp_pos'] = token_sp_pos
                            else:
                                rv['frame_ids'] = cat_frame_ids
                            retrieval_kv.append(rv)

            if profiling:
                self._prof["retrieval"] += time.perf_counter() - _t_retr

            if flowcache and mode != 'templerun' and fc_k[0] is not None:
                retrieval_kv = []
                for layer_idx in range(self.num_transformer_blocks):
                    retrieval_kv.append({
                        'k': fc_k[layer_idx].to(noise.device),
                        'v': fc_v[layer_idx].to(noise.device),
                        'token_frame_ids': fc_fids[layer_idx].to(noise.device),
                        'token_sp_pos': fc_sp_pos[layer_idx].to(noise.device),
                        'recent_frames': recent_frames,
                    })


            if profiling:
                _t_fwd = time.perf_counter()
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_kv=retrieval_kv,
                        flowcache=flowcache,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        retrieval_kv=retrieval_kv,
                        flowcache=flowcache,
                    )

            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                retrieval_kv=retrieval_kv,
                flowcache=flowcache,
            )
            if profiling:
                self._prof["forward"] += time.perf_counter() - _t_fwd

            if (use_retrieval or worldkv or fastar_full) and mode != 'templerun' and query_state is not None:
                block_tokens = block_size * frame_seqlen
                if block_id >= sink_blocks:
                    if worldkv:
                        bank_device = noise.device if kv_bank_on_gpu else torch.device('cpu')
                        for layer_idx, lc in enumerate(self.kv_cache1):
                            local_end = lc['local_end_index'].item()
                            k_slice = lc['k'][:, local_end - block_tokens:local_end].detach().clone()
                            v_slice = lc['v'][:, local_end - block_tokens:local_end].detach().clone()
                            kf_list = torch.chunk(k_slice, block_size, dim=1)
                            vf_list = torch.chunk(v_slice, block_size, dim=1)
                            if worldkv_compress:
                                if profiling:
                                    _t_ev = time.perf_counter()
                                keep_per_frame, sp_per_frame = _worldkv_compress_chunk(
                                    k_slice, block_size, frame_seqlen, worldkv_retain_ratio)
                                if profiling:
                                    self._prof["evict"] += time.perf_counter() - _t_ev
                                for j in range(block_size):
                                    keep = keep_per_frame[j]
                                    kf = kf_list[j][:, keep].to(bank_device)
                                    vf = vf_list[j][:, keep].to(bank_device)
                                    sp = sp_per_frame[j].to(bank_device)
                                    kv_bank_k[layer_idx].append(kf)
                                    kv_bank_v[layer_idx].append(vf)
                                    kv_bank_sp[layer_idx].append(sp)
                                    self._last_bank_bytes += (
                                        kf.numel() * kf.element_size()
                                        + vf.numel() * vf.element_size()
                                        + sp.numel() * sp.element_size())
                                    if profiling:
                                        self._bank_prof["K"] = self._bank_prof.get("K", 0) + kf.numel() * kf.element_size()
                                        self._bank_prof["V"] = self._bank_prof.get("V", 0) + vf.numel() * vf.element_size()
                                        self._bank_prof["sp"] = self._bank_prof.get("sp", 0) + sp.numel() * sp.element_size()
                            else:
                                for kf, vf in zip(kf_list, vf_list):
                                    kv_bank_k[layer_idx].append(kf.to(bank_device))
                                    kv_bank_v[layer_idx].append(vf.to(bank_device))
                                    self._last_bank_bytes += kf.numel() * kf.element_size() + vf.numel() * vf.element_size()
                                    if profiling:
                                        self._bank_prof["K"] = self._bank_prof.get("K", 0) + kf.numel() * kf.element_size()
                                        self._bank_prof["V"] = self._bank_prof.get("V", 0) + vf.numel() * vf.element_size()
                        bank_frame_poses.extend([p.detach().clone() for p in frame_deltas])
                    elif fastar_full:
                        for layer_idx, lc in enumerate(self.kv_cache1):
                            local_end = lc['local_end_index'].item()
                            k_slice = lc['k'][:, local_end - block_tokens:local_end].detach().clone()
                            v_slice = lc['v'][:, local_end - block_tokens:local_end].detach().clone()
                            for j, (kf, vf) in enumerate(zip(torch.chunk(k_slice, block_size, dim=1),
                                                              torch.chunk(v_slice, block_size, dim=1))):
                                global_fid = current_start_frame + j
                                if profiling:
                                    _t_ev = time.perf_counter()
                                (tc_full_k[layer_idx], tc_full_k_n[layer_idx],
                                 tc_full_v_sum[layer_idx],
                                 tc_full_count[layer_idx], tc_full_fids[layer_idx],
                                 tc_full_sp_pos[layer_idx]) = _tempcache_merge_full(
                                    tc_full_k[layer_idx], tc_full_k_n[layer_idx],
                                    tc_full_v_sum[layer_idx],
                                    tc_full_count[layer_idx], tc_full_fids[layer_idx],
                                    tc_full_sp_pos[layer_idx],
                                    kf, vf, global_fid, fastar_sim_threshold,
                                    lsh_tables=fastar_lsh_tables,
                                    lsh_bits=fastar_lsh_bits,
                                    lsh_seed=fastar_lsh_seed,
                                    lsh_bucket_cap=fastar_lsh_bucket_cap)
                                if profiling:
                                    self._prof["evict"] += time.perf_counter() - _t_ev
                        self._last_bank_bytes = 0
                        if profiling:
                            self._bank_prof = {"K": 0, "V_sum": 0, "count": 0, "fids": 0, "sp_pos": 0}
                        for layer_idx in range(self.num_transformer_blocks):
                            if tc_full_k[layer_idx] is not None:
                                self._last_bank_bytes += (
                                    tc_full_k[layer_idx].numel() * tc_full_k[layer_idx].element_size()
                                    + tc_full_v_sum[layer_idx].numel() * tc_full_v_sum[layer_idx].element_size()
                                    + tc_full_count[layer_idx].numel() * tc_full_count[layer_idx].element_size()
                                    + tc_full_fids[layer_idx].numel() * tc_full_fids[layer_idx].element_size()
                                    + tc_full_sp_pos[layer_idx].numel() * tc_full_sp_pos[layer_idx].element_size())
                                if profiling:
                                    self._bank_prof["K"] += tc_full_k[layer_idx].numel() * tc_full_k[layer_idx].element_size()
                                    self._bank_prof["V_sum"] += tc_full_v_sum[layer_idx].numel() * tc_full_v_sum[layer_idx].element_size()
                                    self._bank_prof["count"] += tc_full_count[layer_idx].numel() * tc_full_count[layer_idx].element_size()
                                    self._bank_prof["fids"] += tc_full_fids[layer_idx].numel() * tc_full_fids[layer_idx].element_size()
                                    self._bank_prof["sp_pos"] += tc_full_sp_pos[layer_idx].numel() * tc_full_sp_pos[layer_idx].element_size()
                    else:
                        for layer_idx, lc in enumerate(self.kv_cache1):
                            local_end = lc['local_end_index'].item()
                            k_slice = lc['k'][:, local_end - block_tokens:local_end].detach().clone()
                            v_slice = lc['v'][:, local_end - block_tokens:local_end].detach().clone()
                            acc_k[layer_idx].append(k_slice)
                            acc_v[layer_idx].append(v_slice)
                        acc_frames += block_size
                        frame_pose_acc.extend([p.detach().clone() for p in frame_deltas])

                        if acc_frames >= quant_chunk_frames:
                            bank_device = noise.device if kv_bank_on_gpu else torch.device('cpu')
                            for layer_idx in range(self.num_transformer_blocks):
                                k_cat = torch.cat(acc_k[layer_idx], dim=1)
                                v_cat = torch.cat(acc_v[layer_idx], dim=1)
                                k_kwargs = dict(qvg_kwargs)
                                v_kwargs = dict(qvg_kwargs)
                                if ratio_k is not None:
                                    k_kwargs['sparse_ratio'] = ratio_k[layer_idx]
                                    v_kwargs['sparse_ratio'] = ratio_v[layer_idx]
                                k_packed = qvg_compress(k_cat, **k_kwargs)
                                v_packed = qvg_compress(v_cat, **v_kwargs)
                                if residual_mag_sink is not None:
                                    bank_chunk_id = len(kv_bank_k[layer_idx])
                                    residual_mag_sink.append(
                                        (layer_idx, 'k', bank_chunk_id, k_packed['residual_mag'].detach().cpu()))
                                    residual_mag_sink.append(
                                        (layer_idx, 'v', bank_chunk_id, v_packed['residual_mag'].detach().cpu()))
                                    k_packed.pop('residual_mag', None)
                                    v_packed.pop('residual_mag', None)
                                kv_bank_k[layer_idx].append(_move_packed(k_packed, bank_device))
                                kv_bank_v[layer_idx].append(_move_packed(v_packed, bank_device))
                                self._last_bank_bytes += _packed_bytes(k_packed) + _packed_bytes(v_packed)
                                if profiling:
                                    for p in (k_packed, v_packed):
                                        for fld, nb in _packed_breakdown(p).items():
                                            self._bank_prof[fld] = self._bank_prof.get(fld, 0) + nb
                                acc_k[layer_idx] = []
                                acc_v[layer_idx] = []
                            bank_frame_poses.extend(frame_pose_acc)
                            acc_frames = 0
                            frame_pose_acc = []
                running_state = query_state.detach().clone()

            if flowcache and mode != 'templerun':
                block_tokens = block_size * frame_seqlen
                if block_id >= sink_blocks:
                    for layer_idx, lc in enumerate(self.kv_cache1):
                        local_end = lc['local_end_index'].item()
                        k_slice = lc['k'][:, local_end - block_tokens:local_end].detach()
                        v_slice = lc['v'][:, local_end - block_tokens:local_end].detach()
                        fids = torch.cat([
                            torch.full([frame_seqlen], current_start_frame + j,
                                       device=noise.device, dtype=torch.long)
                            for j in range(block_size)])
                        sp_pos = torch.arange(frame_seqlen, device=noise.device,
                                              dtype=torch.long).repeat(block_size)
                        if fc_acc_k[layer_idx] is None:
                            fc_acc_k[layer_idx] = k_slice
                            fc_acc_v[layer_idx] = v_slice
                            fc_acc_fids[layer_idx] = fids
                            fc_acc_sp_pos[layer_idx] = sp_pos
                        else:
                            fc_acc_k[layer_idx] = torch.cat([fc_acc_k[layer_idx], k_slice], dim=1)
                            fc_acc_v[layer_idx] = torch.cat([fc_acc_v[layer_idx], v_slice], dim=1)
                            fc_acc_fids[layer_idx] = torch.cat([fc_acc_fids[layer_idx], fids])
                            fc_acc_sp_pos[layer_idx] = torch.cat([fc_acc_sp_pos[layer_idx], sp_pos])
                    fc_acc_frames += block_size

                    if fc_acc_frames >= quant_chunk_frames:
                        for layer_idx in range(self.num_transformer_blocks):
                            lc = self.kv_cache1[layer_idx]
                            if fc_k[layer_idx] is None:
                                fc_k[layer_idx] = fc_acc_k[layer_idx]
                                fc_v[layer_idx] = fc_acc_v[layer_idx]
                                fc_fids[layer_idx] = fc_acc_fids[layer_idx]
                                fc_sp_pos[layer_idx] = fc_acc_sp_pos[layer_idx]
                            else:
                                fc_k[layer_idx] = torch.cat([fc_k[layer_idx], fc_acc_k[layer_idx]], dim=1)
                                fc_v[layer_idx] = torch.cat([fc_v[layer_idx], fc_acc_v[layer_idx]], dim=1)
                                fc_fids[layer_idx] = torch.cat([fc_fids[layer_idx], fc_acc_fids[layer_idx]])
                                fc_sp_pos[layer_idx] = torch.cat([fc_sp_pos[layer_idx], fc_acc_sp_pos[layer_idx]])
                            if fc_k[layer_idx].shape[1] > flowcache_budget:
                                if profiling:
                                    _t_ev = time.perf_counter()
                                ck, cv, cf, cs = _flowcache_compress(
                                    fc_k[layer_idx], fc_v[layer_idx],
                                    fc_fids[layer_idx], fc_sp_pos[layer_idx],
                                    lc['flow_q'], lc['flow_grid_sizes'],
                                    m.freqs, flowcache_budget, flowcache_lambda,
                                    flowcache_maxpool_kernel)
                                if profiling:
                                    self._prof["evict"] += time.perf_counter() - _t_ev
                                fc_k[layer_idx] = ck
                                fc_v[layer_idx] = cv
                                fc_fids[layer_idx] = cf
                                fc_sp_pos[layer_idx] = cs
                            fc_acc_k[layer_idx] = None
                            fc_acc_v[layer_idx] = None
                            fc_acc_fids[layer_idx] = None
                            fc_acc_sp_pos[layer_idx] = None
                        fc_acc_frames = 0
                        self._last_bank_bytes = 0
                        if profiling:
                            self._bank_prof = {"K": 0, "V": 0, "fids": 0, "sp_pos": 0}
                        for layer_idx in range(self.num_transformer_blocks):
                            if fc_k[layer_idx] is not None:
                                self._last_bank_bytes += (
                                    fc_k[layer_idx].numel() * fc_k[layer_idx].element_size()
                                    + fc_v[layer_idx].numel() * fc_v[layer_idx].element_size()
                                    + fc_fids[layer_idx].numel() * fc_fids[layer_idx].element_size()
                                    + fc_sp_pos[layer_idx].numel() * fc_sp_pos[layer_idx].element_size())
                                if profiling:
                                    self._bank_prof["K"] += fc_k[layer_idx].numel() * fc_k[layer_idx].element_size()
                                    self._bank_prof["V"] += fc_v[layer_idx].numel() * fc_v[layer_idx].element_size()
                                    self._bank_prof["fids"] += fc_fids[layer_idx].numel() * fc_fids[layer_idx].element_size()
                                    self._bank_prof["sp_pos"] += fc_sp_pos[layer_idx].numel() * fc_sp_pos[layer_idx].element_size()

            denoised_pred = denoised_pred.transpose(1,2)
            if fastar_full:
                del retrieval_kv
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            vae_t0 = time.perf_counter()
            vae_dev = next(self.vae_decoder.parameters()).device
            if vae_dev.type == "cpu":
                cpu_cache = [None if c is None else c.to("cpu") for c in vae_cache]
                video, vae_cache = self.vae_decoder(denoised_pred.half().to("cpu"), *cpu_cache)
                video = video.to(noise.device)
                vae_cache = [None if c is None else c.to(noise.device) for c in vae_cache]
            else:
                video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)
            videos += [video.cpu()]
            self._last_vae_latency += time.perf_counter() - vae_t0
            video = rearrange(video, "B T C H W -> B T H W C")
            video = ((video.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
            video = np.ascontiguousarray(video)
            mouse_icon = 'assets/images/mouse.png'
            if mode != 'templerun':
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                    conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                )
            else:
                config = (
                    conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
                )
            process_video(video.astype(np.uint8), output_folder+f'/{name}_current.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)
            current_start_frame += current_num_frames

            if scripted_actions is None and input("Continue? (Press `n` to break)").strip() == "n":
                break
        videos_tensor = torch.cat(videos, dim=1)
        videos = rearrange(videos_tensor, "B T C H W -> B T H W C")
        videos = ((videos.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
        video = np.ascontiguousarray(videos)
        mouse_icon = 'assets/images/mouse.png'
        if mode != 'templerun':
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
                conditional_dict["mouse_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy(),
            )
        else:
            config = (
                conditional_dict["keyboard_cond"][0, : 1 + 4 * (current_start_frame + self.num_frame_per_block-1)].float().cpu().numpy()
            )
        process_video(video.astype(np.uint8), output_folder+f'/{name}_icon.mp4', config, mouse_icon, mouse_scale=0.1, mode=mode)
        process_video(video.astype(np.uint8), output_folder+f'/{name}.mp4', config, mouse_icon, mouse_scale=0.1, process_icon=False, mode=mode)

        if full_kv:
            for blk, orig in zip(self.generator.model.blocks, self._full_kv_saved_max):
                blk.self_attn.max_attention_size = orig

        self._last_wan_latency = time.perf_counter() - wan_t0 - self._last_vae_latency

        if profiling:
            self._prof["quant"] = _QVG_PROFILE["quant_time"]
            self._prof["evict"] += _QVG_PROFILE["evict_time"]
            self._prof["vae"] = self._last_vae_latency

        if return_latents:
            return output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, num_frames=None):
        kv_cache1 = []
        if num_frames is not None:
            kv_cache_size = num_frames * self.frame_seq_length
        elif self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 15 * 1 * self.frame_seq_length

        kv_cache_device = device

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=kv_cache_device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=kv_cache_device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard
        self.kv_cache_mouse = kv_cache_mouse


    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

import os
import argparse
import torch
import numpy as np
import copy

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image

from pipeline import CausalInferenceStreamingPipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.visualize import process_video
from utils.misc import set_seed
from utils.conditions import *
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["quanevict"],
                        required=True, help="Method: quanevict = quantization + residual sparse eviction + sensitivity-guided retention")
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml",
                        help="Path to the inference config file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Distilled checkpoint path (safetensors)")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0",
                        help="Pretrained model (VAE) directory")
    parser.add_argument("--img_path", type=str, default="demo_images/universal/0000.png",
                        help="Input image path (non-interactive, single image)")
    parser.add_argument("--output_folder", type=str, default="outputs/", help="Output directory")
    parser.add_argument("--max_num_output_frames", type=int, default=360,
                        help="Maximum number of output latent frames")
    parser.add_argument("--mode", type=str, choices=["universal", "gta_drive", "templerun"], default=None,
                        help="Interaction mode; defaults to the 'mode' field in the config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Parallel batch size; >1 repeats the current input to that batch size")
    parser.add_argument("--kv_bank_on_gpu", action=argparse.BooleanOptionalAction, default=True,
                        help="Store the KV bank on GPU")
    parser.add_argument("--quant_chunk_frames", type=int, default=12,
                        help="Latent frames per compression chunk (must be a multiple of num_frame_per_block)")

    parser.add_argument("--qvg_bits", type=int, default=2,
                        help="QVG symmetric quantization bit width (0 = no quantization, default 2-bit)")
    parser.add_argument("--qvg_num_centroids", type=int, default=256,
                        help="QVG number of centroids per PRQ stage K")
    parser.add_argument("--qvg_num_stages", type=int, default=1,
                        help="QVG number of PRQ residual stages S")
    parser.add_argument("--qvg_group_size", type=int, default=64,
                        help="QVG groupwise symmetric quantization group size B")
    parser.add_argument("--qvg_kmeans_iters", type=int, default=4,
                        help="QVG k-means iterations")
    parser.add_argument("--qvg_seed", type=int, default=0,
                        help="QVG k-means random seed")
    parser.add_argument("--qvg_sparsify", action=argparse.BooleanOptionalAction, default=True,
                        help="QVG residual sparsification (low-magnitude residuals are not stored, approximated by centroids at read time)")
    parser.add_argument("--qvg_sparse_ratio", type=float, default=0.5,
                        help="Per-head retention ratio (0~1; base ratio ratio_base)")
    parser.add_argument("--qvg_use_triton", action=argparse.BooleanOptionalAction, default=True,
                        help="Use Triton to accelerate k-means distance and sparse decompression")
    parser.add_argument("--qvg_sensitivity_wk", type=str, default=None,
                        help="QuanEvict: visual K sensitivity npy path ([L,H])")
    parser.add_argument("--qvg_sensitivity_wv", type=str, default=None,
                        help="QuanEvict: visual V sensitivity npy path ([L,H])")
    parser.add_argument("--qvg_gamma", type=float, default=1.0,
                        help="QuanEvict: sensitivity scaling strength (kept for compatibility; currently unused)")
    args = parser.parse_args()
    return args


def build_revisit_trajectory(mode, num_blocks, device):
    assert mode == 'universal', 'revisit trajectory currently only supports universal mode'

    KB = {
        'forward': [1, 0, 0, 0],
        'back':    [0, 1, 0, 0],
        'left':    [0, 0, 1, 0],
        'right':   [0, 0, 0, 1],
        'stop':    [0, 0, 0, 0],
    }
    CAM = 0.1
    MOUSE = {
        'stop':  [0.0, 0.0],
        'left':  [0.0, -CAM],
        'right': [0.0, CAM],
        'up':    [CAM, 0.0],
        'down':  [-CAM, 0.0],
    }

    phases = [
        ('forward', 'stop',  0.25),
        ('forward', 'right', 0.10),
        ('back',    'stop',  0.25),
        ('back',    'left',  0.10),
        ('forward', 'stop',  0.15),
        ('forward', 'right', 0.15),
    ]
    actions = []
    for kb_name, mouse_name, frac in phases:
        for _ in range(max(1, round(num_blocks * frac))):
            actions.append({
                'keyboard': torch.tensor(KB[kb_name], device=device, dtype=torch.int64),
                'mouse': torch.tensor(MOUSE[mouse_name], device=device, dtype=torch.float32),
            })
    while len(actions) < num_blocks:
        actions.append({
            'keyboard': torch.tensor(KB['stop'], device=device, dtype=torch.int64),
            'mouse': torch.tensor(MOUSE['stop'], device=device, dtype=torch.float32),
        })
    return actions[:num_blocks]


class InteractiveGameInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        pipeline = CausalInferenceStreamingPipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.args.checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        m = self.pipeline.generator.model
        m.qvg_bits = self.args.qvg_bits
        m.qvg_num_centroids = self.args.qvg_num_centroids
        m.qvg_num_stages = self.args.qvg_num_stages
        m.qvg_group_size = self.args.qvg_group_size
        m.qvg_kmeans_iters = self.args.qvg_kmeans_iters
        m.qvg_seed = self.args.qvg_seed
        m.qvg_sparsify = self.args.qvg_sparsify
        m.qvg_sparse_ratio = self.args.qvg_sparse_ratio
        m.qvg_use_triton = self.args.qvg_use_triton
        m.qvg_gamma = self.args.qvg_gamma
        m.qvg_sensitivity_wk = self.args.qvg_sensitivity_wk
        m.qvg_sensitivity_wv = self.args.qvg_sensitivity_wv

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image

    def generate_videos(self, mode='universal'):
        assert mode in ['universal', 'gta_drive', 'templerun']

        img_path = self.args.img_path
        image = load_image(img_path)

        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        batch_size = max(1, self.args.batch_size)
        if batch_size > 1:
            image = image.repeat(batch_size, 1, 1, 1, 1)
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.args.max_num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = self.vae.clip.encode_video(image)
        del self.vae
        torch.cuda.empty_cache()
        sampled_noise = torch.randn(
            [batch_size, 16, self.args.max_num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )
        num_frames = (self.args.max_num_output_frames - 1) * 4 + 1

        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }

        if mode == 'universal':
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        elif mode == 'gta_drive':
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)
        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).expand(batch_size, -1, -1).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition

        num_blocks = self.args.max_num_output_frames // self.pipeline.num_frame_per_block
        scripted_actions = build_revisit_trajectory(mode, num_blocks, self.device) if mode == 'universal' else None

        with torch.no_grad():
            videos = self.pipeline.inference(
                noise=sampled_noise,
                conditional_dict=conditional_dict,
                return_latents=False,
                output_folder=self.args.output_folder,
                name=os.path.basename(img_path),
                mode=mode,
                scripted_actions=scripted_actions,
                use_retrieval=False,
                retrieval_topk_frames=4,
                traj_threshold=None,
                kv_bank_on_gpu=self.args.kv_bank_on_gpu,
                quant_chunk_frames=self.args.quant_chunk_frames,
                worldkv=False,
                worldkv_compress=False,
                worldkv_retain_ratio=0.25,
                fastar_full=False,
                fastar_sim_threshold=0.8,
                fastar_lsh_tables=1,
                fastar_lsh_bits=10,
                fastar_lsh_seed=0,
                fastar_lsh_bucket_cap=128,
                residual_mag_sink=None,
                flowcache=False,
                flowcache_budget=10000,
                flowcache_lambda=0.07,
                flowcache_maxpool_kernel=3,
                full_kv=False,
                cache_transfer=False,
                profiling=False,
            )


def main():
    args = parse_args()

    assert args.qvg_sensitivity_wk and args.qvg_sensitivity_wv,\
        "quanevict requires --qvg_sensitivity_wk and --qvg_sensitivity_wv npy paths"

    set_seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    pipeline = InteractiveGameInference(args)
    mode = args.mode if args.mode is not None else pipeline.config.pop('mode')
    pipeline.generate_videos(mode)


if __name__ == "__main__":
    main()

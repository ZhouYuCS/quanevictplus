# app.pdf
app.pdf is the supplementary material for the main paper, which  is independent of other part (code implementaiton of QuanEvict) of this zip file

# QuanEvict

QuanEvict is a KV-cache compression method for Wan 2.1 14B interactive video
generation (Matrix-Game / Wan interactive game video). It consists of three parts:

1. **2-bit KV quantization**: each layer's K/V is symmetrically quantized per group
   (PRQ, residual quantization), compressing the KV cache to roughly 2 bits.
2. **Residual sparse eviction**: low-magnitude parts of the quantized residual are
   not stored; they are approximated by centroids at read time, further reducing
   storage.
3. **Sensitivity-guided retention ratio**: per-(layer, head) sensitivity allocates
   the KV bank budget, keeping more residuals for sensitive heads and evicting more
   for insensitive heads while keeping the total budget unchanged.

This repository is a minimal runnable integration of the QuanEvict-related code from
several repositories (Matrix-Game / lingbot / inspatio). It is based on the mola
repository, with unrelated methods such as WorldKV / FAST-AR / trajectory retrieval /
flowcache removed, keeping only the model and inference code required by QuanEvict.

## Directory Structure

```
QuanEvict_plus/
├── run_quanevict.py          # Unified entry point for QuanEvict inference
├── pipeline/                 # Causal inference pipeline (CausalInferenceStreamingPipeline)
├── configs/                  # Inference YAML and distilled-model configs
├── assets/                   # Interactive UI assets
├── wan/                      # Wan 2.1 14B model code (QVG quantization / QuanEvict ratios)
├── utils/                    # Condition construction / visualization / scheduler / Wan wrapper, etc.
├── demo_utils/               # VAE decoder and constants (only constant.py and vae_block3.py)
├── requirements.txt
└── README.md
```

## Quick Start

### Environment

```bash
pip install -r requirements.txt
```

> `flash-attn` must match your CUDA / torch version; see the official install
> instructions. When `triton` is missing, `wan/utils/qvg_triton.py` automatically
> falls back to a pure PyTorch implementation (same results, slightly slower).

### QuanEvict

```bash
python run_quanevict.py \
  --method quanevict \
  --config_path configs/inference_yaml/inference_universal.yaml \
  --checkpoint_path /path/to/model/base_distill.safetensors \
  --pretrained_model_path /path/to/model \
  --img_path /path/to/images/0000.png \
  --output_folder outputs/ \
  --max_num_output_frames 12 \
  --mode universal \
  --qvg_bits 2 \
  --qvg_sparsify --qvg_sparse_ratio 0.5 \
  --qvg_sensitivity_wk /path/to/sensitivity_wK.npy \
  --qvg_sensitivity_wv /path/to/sensitivity_wV.npy
```

## Arguments

### Method

| Argument | Description |
| --- | --- |
| `--method` | `quanevict`: quantization + residual sparse eviction + sensitivity-guided retention |

### Model / Data / Output

| Argument | Description |
| --- | --- |
| `--config_path` | Path to the inference config file |
| `--checkpoint_path` | Distilled checkpoint (safetensors) path |
| `--pretrained_model_path` | Pretrained model directory (containing `Wan2.1_VAE.pth`) |
| `--img_path` | Input image path (single image) |
| `--output_folder` | Output directory |
| `--max_num_output_frames` | Maximum number of output latent frames |
| `--mode` | `universal` / `gta_drive` / `templerun` (defaults to the `mode` in the config) |
| `--seed` | Random seed |
| `--batch_size` | Parallel batch size |
| `--kv_bank_on_gpu` | Whether to store the KV bank on GPU |
| `--quant_chunk_frames` | Number of latent frames per compression chunk |

### QVG / QuanEvict hyperparameters

| Argument | Description |
| --- | --- |
| `--qvg_bits` | Symmetric quantization bit width (0 = no quantization, default 2) |
| `--qvg_num_centroids` | Number of centroids per PRQ stage K |
| `--qvg_num_stages` | Number of PRQ residual stages S |
| `--qvg_group_size` | Per-group quantization group size B |
| `--qvg_kmeans_iters` | Number of k-means iterations |
| `--qvg_seed` | k-means random seed |
| `--qvg_sparsify` | Whether to enable residual sparsification |
| `--qvg_sparse_ratio` | Base retention ratio per head (`ratio_base`) |
| `--qvg_use_triton` | Whether to use Triton to accelerate k-means / sparse decompression |
| `--qvg_sensitivity_wk` | K sensitivity npy path (shape `[layers, heads]`) |
| `--qvg_sensitivity_wv` | V sensitivity npy path (shape `[layers, heads]`) |
| `--qvg_gamma` | Sensitivity scaling strength (kept for compatibility; currently unused) |

## Placeholder paths

This project is intended for GitHub upload, so **model weights / datasets / sensitivity
npy paths are placeholders** and must be filled in by the user:

- `--checkpoint_path`, `--pretrained_model_path`: point to your own Wan 2.1 14B /
  Matrix-Game distilled model.
- `--img_path`: point to your own input image.
- `--qvg_sensitivity_wk/wv`: point to precomputed sensitivity npy files.

## Sensitivity computation

`quanevict` requires precomputed `sensitivity_wK.npy` / `sensitivity_wV.npy`
(shape `[layers, heads]`). **This repository does not yet include the sensitivity
computation script**; it can be ported from lingbot's `sensitivity.py` later.

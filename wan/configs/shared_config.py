import torch
from easydict import EasyDict

wan_shared_cfg = EasyDict()

wan_shared_cfg.t5_model = 'umt5_xxl'
wan_shared_cfg.t5_dtype = torch.bfloat16
wan_shared_cfg.text_len = 512

wan_shared_cfg.param_dtype = torch.bfloat16

wan_shared_cfg.num_train_timesteps = 1000
wan_shared_cfg.sample_fps = 16
wan_shared_cfg.sample_neg_prompt = 'vivid colors, overexposure, static, blurry details, subtitles, style, artwork, painting, static frame, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, cluttered background, three legs, many people in background, walking backwards'

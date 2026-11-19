import torch


def build_action_state(keyboard_cond, mouse_cond=None, num_latent_frames=None,
                       vae_time_compression_ratio=4):
    B = keyboard_cond.shape[0]
    if num_latent_frames is None:
        num_latent_frames = (keyboard_cond.shape[1] - 1) // vae_time_compression_ratio + 1

    device = keyboard_cond.device
    cum = torch.zeros(B, 4, device=device, dtype=torch.float32)
    states = []
    for i in range(num_latent_frames):
        rgb_idx = min(i * vae_time_compression_ratio, keyboard_cond.shape[1] - 1)
        kb = keyboard_cond[:, rgb_idx].float()
        cum[:, 0] += (kb[:, 0] - kb[:, 1])
        cum[:, 1] += (kb[:, 3] - kb[:, 2])
        if mouse_cond is not None:
            m = mouse_cond[:, rgb_idx].float()
            cum[:, 2] += m[:, 0]
            cum[:, 3] += m[:, 1]
        states.append(cum.clone())
    return torch.stack(states, dim=1)

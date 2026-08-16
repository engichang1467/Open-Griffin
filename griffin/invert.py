"""DDIM inversion of a source image, caching its self-attention K/V.

Run this on a plain SD pipeline, before any IP-Adapter is loaded -- inversion
takes text conditioning only, and the adapter would change the UNet signature.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDIMInverseScheduler
from skimage.filters import threshold_otsu

from .attn import GriffinState, SourceCache, install


def find_token(prompt_ids: list[int], subject_ids: list[int]) -> int:
    """Index of the subject's first token inside the tokenized prompt."""
    if not subject_ids:
        raise ValueError("empty subject")
    for i in range(len(prompt_ids) - len(subject_ids) + 1):
        if prompt_ids[i : i + len(subject_ids)] == subject_ids:
            return i
    raise ValueError(f"subject tokens {subject_ids} not in prompt")


def subject_mask(cache: SourceCache, size: int, res: int = 16) -> torch.Tensor:
    """Cross-attention maps of the subject token -> a binary source mask.

    Only the `res` x `res` layers are used; the coarser ones are semantic mush
    and the finer ones are dominated by texture.
    """
    maps = [m for m in cache.cross.values() if m.numel() == res * res]
    if not maps:
        raise ValueError(f"no {res}x{res} cross-attention maps were captured")
    avg = torch.stack(maps).mean(0).reshape(res, res).float()
    avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
    up = F.interpolate(avg[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
    return up > threshold_otsu(up.numpy())


@torch.no_grad()
def invert(pipe, image, prompt: str, subject: str, num_steps: int = 50) -> SourceCache:
    """Invert one source image and return its cached K/V plus the subject mask."""
    device, dtype = pipe.unet.device, pipe.unet.dtype

    pixels = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    latents = pipe.vae.encode(pixels).latent_dist.mean * pipe.vae.config.scaling_factor

    embeds, _ = pipe.encode_prompt(prompt, device, 1, do_classifier_free_guidance=False)

    prompt_ids = pipe.tokenizer(prompt).input_ids
    subject_ids = pipe.tokenizer(subject, add_special_tokens=False).input_ids

    state = GriffinState(mode="capture", capture=SourceCache())
    state.subject_token = find_token(prompt_ids, subject_ids)

    scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_steps, device=device)

    restore = install(pipe.unet, state, subject_maps=True)
    try:
        for t in scheduler.timesteps:
            state.t = int(t)
            noise = pipe.unet(latents, t, encoder_hidden_states=embeds).sample
            latents = scheduler.step(noise, t, latents).prev_sample
    finally:
        restore()

    cache = state.capture
    cache.mask = subject_mask(cache, size=latents.shape[-1])
    return cache



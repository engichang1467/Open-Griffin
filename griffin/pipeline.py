"""The Griffin denoising loop.

Three overlapping stages on one 50-step DDIM schedule:
  steps  0-9   structure init -- masked IP-Adapter alone, scale 1.8
  steps 10-49  attention sharing on top, IP-Adapter scale dropping 0.8 -> 0.4
  steps 15,20,25,30  layout refinement via DIFT/DINO correspondence + SAM
"""

from __future__ import annotations

from pathlib import Path

import torch

from .attn import GriffinState, SourceCache, alpha_schedule, install
from .layout import save_mask, update_layouts

T_LBA = 10
LAYOUT_STEPS = (15, 20, 25, 30)
IP_SCALES = ((0, 1.8), (10, 0.8), (30, 0.4))  # (first step, scale)


def ip_scale(step: int) -> float:
    return [s for start, s in IP_SCALES if step >= start][-1]


def load_adapters(pipe, n: int, repo: str = "h94/IP-Adapter", subfolder: str = "models",
                  weight_name: str = "ip-adapter_sd15.bin"):
    """One IP-Adapter instance per subject -- that is how diffusers does regional conditioning."""
    pipe.load_ip_adapter(repo, subfolder=subfolder, weight_name=[weight_name] * n)
    return pipe


def _ip_masks(layouts: list[torch.Tensor], device, dtype) -> list[torch.Tensor]:
    """One (1, 1, H, W) mask per adapter, which is the shape diffusers checks for.

    The list length has to equal the number of loaded adapters, and each entry
    carries one image, so both leading dims are 1.
    """
    return [m.to(device=device, dtype=dtype)[None, None] for m in layouts]


@torch.no_grad()
def compose(
    pipe,
    prompt: str,
    references: list,
    caches: list[SourceCache],
    layouts: list[torch.Tensor],
    *,
    negative_prompt: str | None = None,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    generator: torch.Generator | None = None,
    extractor=None,
    segmenter=None,
    restrict_target: bool = True,
    debug_dir=None,
):
    """Compose `references` into one image under `layouts`.

    references: one PIL image per subject, the same ones `invert()` consumed.
    caches:     `SourceCache` per subject, in the same order.
    layouts:    (height, width) bool tensors marking where each subject goes.
    extractor / segmenter: pass both to enable the dynamic layout update; omit
    them to run with the layouts exactly as given.
    debug_dir:  where to dump the predicted clean image and the refined masks at
    every layout step, so a bad composition can be traced to a bad mask.
    """
    if not (len(references) == len(caches) == len(layouts)):
        raise ValueError("references, caches and layouts must line up one-to-one")

    device, dtype = pipe.unet.device, pipe.unet.dtype
    do_cfg = guidance_scale > 1.0
    n_subjects = len(references)

    cond, uncond = pipe.encode_prompt(prompt, device, 1, do_cfg, negative_prompt)
    embeds = torch.cat([uncond, cond]) if do_cfg else cond

    image_embeds = pipe.prepare_ip_adapter_image_embeds([[r] for r in references], None, device, 1, do_cfg)
    added_cond_kwargs = {"image_embeds": image_embeds}

    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height // 8, width // 8),
        generator=generator, device=device, dtype=dtype,
    ) * pipe.scheduler.init_noise_sigma

    state = GriffinState(sources=caches, layouts=layouts, restrict_target=restrict_target)
    ip_masks = _ip_masks(layouts, device, dtype)

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for index, layout in enumerate(layouts):
            save_mask(layout, debug_dir / f"layout_init_subject{index}.png")

    dynamic = extractor is not None and segmenter is not None
    if dynamic:
        # DIFT passes are single-latent and unguided, so they need their own embeds
        extractor.added_cond_kwargs = {
            "image_embeds": pipe.prepare_ip_adapter_image_embeds([[r] for r in references], None, device, 1, False)
        }
        source_latents = [
            pipe.vae.encode(pipe.image_processor.preprocess(r).to(device, dtype)).latent_dist.mean
            * pipe.vae.config.scaling_factor
            for r in references
        ]
        source_masks = [c.mask for c in caches]

    restore = install(pipe.unet, state)
    try:
        for i, t in enumerate(timesteps):
            state.t = int(t)
            state.alpha = alpha_schedule(i, T_LBA, num_steps)
            state.mode = "share" if i >= T_LBA else "off"
            pipe.set_ip_adapter_scale([ip_scale(i)] * n_subjects)

            model_input = torch.cat([latents] * 2) if do_cfg else latents
            model_input = pipe.scheduler.scale_model_input(model_input, t)
            noise = pipe.unet(
                model_input,
                t,
                encoder_hidden_states=embeds,
                added_cond_kwargs=added_cond_kwargs,
                cross_attention_kwargs={"ip_adapter_masks": ip_masks},
            ).sample
            if do_cfg:
                noise_uncond, noise_cond = noise.chunk(2)
                noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            step = pipe.scheduler.step(noise, t, latents)
            latents = step.prev_sample

            if dynamic and i in LAYOUT_STEPS:
                state.mode = "off"  # the extra UNet passes must not share attention
                pipe.set_ip_adapter_scale([0.0] * n_subjects)
                clean = step.pred_original_sample
                preview = pipe.image_processor.postprocess(
                    pipe.vae.decode(clean / pipe.vae.config.scaling_factor).sample
                )[0]
                layouts = update_layouts(
                    extractor, segmenter, clean, preview,
                    source_latents, source_masks, layouts, cond,
                )
                state.layouts = layouts
                ip_masks = _ip_masks(layouts, device, dtype)

                if debug_dir is not None:
                    preview.save(debug_dir / f"preview_step{i:02d}.png")
                    for index, layout in enumerate(layouts):
                        save_mask(layout, debug_dir / f"layout_step{i:02d}_subject{index}.png", preview)
    finally:
        restore()

    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    return pipe.image_processor.postprocess(image)[0]

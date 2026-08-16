"""Self-attention capture and layout-controlled sharing.

During DDIM inversion of a source image we stash every self-attention K/V.
During target generation we splice those back in, one layout region at a time,
so region n only ever sees itself and its own source (Griffin eq. for K̂/V̂).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

SELF_ATTN = "attn1"
CROSS_ATTN = "attn2"


@dataclass
class SourceCache:
    """Everything one inverted source image contributes to the target."""

    kv: dict = field(default_factory=dict)  # (layer, timestep) -> (k, v) on CPU
    cross: dict = field(default_factory=dict)  # layer -> summed subject cross-attn map
    mask: torch.Tensor | None = None  # (H, W) bool, the subject's region in the source

    # ponytail: KV lives on CPU, ~2GB per source at 50 steps. If the host->device
    # copies dominate runtime, cache up-block layers only -- they carry appearance.
    def put(self, layer: str, t, k: torch.Tensor, v: torch.Tensor) -> None:
        self.kv[(layer, int(t))] = (k.cpu(), v.cpu())

    def get(self, layer: str, t, device, dtype):
        k, v = self.kv[(layer, int(t))]
        return k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype)


@dataclass
class GriffinState:
    """Shared, mutated by the denoising loop between steps."""

    mode: str = "off"  # "capture" during inversion, "share" during generation
    t: int = 0  # current timestep, the cache key
    alpha: float = 1.0  # appearance transfer strength, 1.2 / (1 + 2exp(-10t))
    sources: list[SourceCache] = field(default_factory=list)
    layouts: list[torch.Tensor] = field(default_factory=list)  # (H, W) bool, one per source
    capture: SourceCache | None = None
    subject_token: int = -1  # cross-attn column to accumulate while capturing
    restrict_target: bool = True


def alpha_schedule(step: int, t_lba: int = 10, total: int = 50) -> float:
    """1.2 / (1 + 2exp(-10t)), t running 1 -> 0 across [T_LBA, end]."""
    if step < t_lba:
        return 0.0
    t = 1.0 - (step - t_lba) / max(total - t_lba, 1)
    return 1.2 / (1.0 + 2.0 * math.exp(-10.0 * t))


def flatten_mask(mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """(H, W) bool -> flat indices at the resolution of an attention layer."""
    side = int(round(math.sqrt(seq_len)))
    if side * side != seq_len:
        raise ValueError(f"non-square attention map: {seq_len}")  # ponytail: square latents only
    small = F.interpolate(mask[None, None].float(), size=(side, side), mode="nearest")
    return small.flatten().nonzero(as_tuple=True)[0]


def shared_attention(query, key, value, regions, bg, alpha: float, restrict_target: bool = True):
    """Attend, overwriting each layout region with its source-augmented attention.

    query/key/value: (B, heads, L, D) for the target.
    regions: list of (target_idx, k_src, v_src); k_src/v_src are (1, heads, Ls, D)
             already restricted to the source subject mask.
    bg: flat indices of the pixels in no region at all.

    Per Griffin eq. (6) and (7), K^n_T and V^n_T are the target keys and values
    "restricted to the pixels of layout component n and the background pixels",
    so a region gathers over `idx ++ bg` while still only writing back to `idx`.
    Excluding the *other* regions is what stops identity leaking between
    subjects; the background was never meant to be cut off with them.

    Pixels in `bg` keep plain target self-attention -- that is eq. (8), the
    background driven by text rather than by any source.
    """
    out = F.scaled_dot_product_attention(query, key, value)
    batch = query.shape[0]
    for idx, k_src, v_src in regions:
        if idx.numel() == 0 or k_src.shape[2] == 0:
            continue
        ctx = torch.cat([idx, bg])
        k_tgt = key[:, :, ctx] if restrict_target else key
        v_tgt = value[:, :, ctx] if restrict_target else value
        # alpha scales the source keys only, never the target or background ones
        k_hat = torch.cat([alpha * k_src.expand(batch, -1, -1, -1), k_tgt], dim=2)
        v_hat = torch.cat([v_src.expand(batch, -1, -1, -1), v_tgt], dim=2)
        out[:, :, idx] = F.scaled_dot_product_attention(query[:, :, idx], k_hat, v_hat)
    return out


def _heads(x: torch.Tensor, attn) -> torch.Tensor:
    b, l, _ = x.shape
    return x.view(b, l, attn.heads, -1).transpose(1, 2)


class GriffinAttnProcessor:
    """Replaces the processor on every `attn1` module."""

    def __init__(self, state: GriffinState, name: str):
        self.state = state
        self.name = name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        state = self.state
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = _heads(attn.to_q(hidden_states), attn)
        key = _heads(attn.to_k(context), attn)
        value = _heads(attn.to_v(context), attn)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if state.mode == "capture" and state.capture is not None:
            state.capture.put(self.name, state.t, key, value)
            hidden_states = F.scaled_dot_product_attention(query, key, value)
        elif state.mode == "share" and state.sources:
            regions, bg = self._regions(key.shape[2], query.device, query.dtype)
            hidden_states = shared_attention(query, key, value, regions, bg, state.alpha, state.restrict_target)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(query.shape[0], -1, attn.heads * query.shape[-1])
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.to(query.dtype)))

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

    def _regions(self, seq_len, device, dtype):
        """The per-source (idx, k_src, v_src) triples, plus the background index.

        Occupancy is accumulated at attention resolution, after downsampling,
        rather than by running `~(union of layouts)` back through
        `flatten_mask`. Both give the same answer while the downsampling is
        nearest -- sampling one pixel per cell commutes with union and
        complement -- but only this one stays exactly disjoint from `idx` if
        that ever becomes an averaging resize. Overlapping keys would otherwise
        appear twice in one softmax.
        """
        state = self.state
        regions = []
        occupied = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for src, layout in zip(state.sources, state.layouts):
            idx = flatten_mask(layout.to(device), seq_len).to(device)
            occupied[idx] = True
            k_src, v_src = src.get(self.name, state.t, device, dtype)
            if src.mask is not None:
                src_idx = flatten_mask(src.mask.to(device), k_src.shape[2]).to(device)
                k_src, v_src = k_src[:, :, src_idx], v_src[:, :, src_idx]
            regions.append((idx, k_src, v_src))
        return regions, (~occupied).nonzero(as_tuple=True)[0]


class SubjectMapProcessor:
    """Replaces `attn2` during inversion to record where the subject token looks.

    Needs the explicit softmax, so it is only installed while capturing.
    """

    def __init__(self, state: GriffinState, name: str):
        self.state = state
        self.name = name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        state = self.state
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        if encoder_hidden_states is not None and attn.norm_cross:
            context = attn.norm_encoder_hidden_states(context)

        query = _heads(attn.to_q(hidden_states), attn)
        key = _heads(attn.to_k(context), attn)
        value = _heads(attn.to_v(context), attn)

        probs = (query.float() @ key.float().transpose(-1, -2) * (query.shape[-1] ** -0.5)).softmax(-1)
        if state.capture is not None and state.subject_token >= 0:
            token_map = probs[..., state.subject_token].mean(dim=(0, 1))  # over batch and heads
            prev = state.capture.cross.get(self.name)
            state.capture.cross[self.name] = token_map.cpu() if prev is None else prev + token_map.cpu()

        hidden_states = (probs.to(value.dtype) @ value).transpose(1, 2)
        hidden_states = hidden_states.reshape(query.shape[0], -1, attn.heads * query.shape[-1])
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install(unet, state: GriffinState, subject_maps: bool = False):
    """Swap in Griffin's processors. Returns a callable that puts the old ones back."""
    original = unet.attn_processors
    procs = {}
    for name, proc in original.items():
        if SELF_ATTN in name:
            procs[name] = GriffinAttnProcessor(state, name)
        elif subject_maps and CROSS_ATTN in name:
            procs[name] = SubjectMapProcessor(state, name)
        else:
            procs[name] = proc
    unet.set_attn_processor(procs)
    return lambda: unet.set_attn_processor(original)



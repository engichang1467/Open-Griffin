"""Dynamic layout update: coarse user masks -> tight masks, mid-generation.

Correspond the partly-denoised target against each source using mixed DIFT and
DINO features, keep the pixels that match well, sample a few spread-out
keypoints from each subject, and let SAM turn those into a clean mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu

from .attn import flatten_mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def box_mask(box, height: int, width: int) -> torch.Tensor:
    """Normalized xyxy box -> (H, W) bool mask."""
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"box must be x0,y0,x1,y1 with x0<x1 and y0<y1, got {box}")
    mask = torch.zeros(height, width, dtype=torch.bool)
    rows = slice(max(0, int(y0 * height)), min(height, max(1, round(y1 * height))))
    cols = slice(max(0, int(x0 * width)), min(width, max(1, round(x1 * width))))
    mask[rows, cols] = True
    return mask


def load_mask(path, height: int, width: int) -> torch.Tensor:
    """Read a mask image (anything non-black is inside) at the target size."""
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("L").resize((width, height), Image.NEAREST)
    return torch.from_numpy(np.array(image)) > 127


def save_mask(mask: torch.Tensor, path, image=None) -> None:
    """Write a mask to `path`, tinted red over `image` when one is given.

    The mask is upsampled to the image, not the other way round, so a 64x64
    source mask stays readable against its 512x512 reference.
    """
    import numpy as np
    from PIL import Image

    flat = (mask.detach().cpu().float().numpy() * 255).astype("uint8")
    if image is None:
        Image.fromarray(flat).save(path)
        return

    base = np.asarray(image.convert("RGB")).astype(float)
    height, width = base.shape[:2]
    grown = np.array(Image.fromarray(flat).resize((width, height), Image.NEAREST)) > 127
    base[grown] = 0.5 * base[grown] + 0.5 * np.array([255.0, 0.0, 0.0])
    Image.fromarray(base.astype("uint8")).save(path)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def mix_features(dift: torch.Tensor, dino: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """F = beta * norm(F_DIFT) (+) (1 - beta) * norm(F_DINO), both (L, C)."""
    return l2norm(torch.cat([beta * l2norm(dift), (1.0 - beta) * l2norm(dino)], dim=-1))


def correspond(target: torch.Tensor, sources: list[torch.Tensor]):
    """Best-matching source for every target pixel.

    target: (Lt, C). sources: list of (Ls_n, C), already restricted to each
    subject's region. All L2-normalized, so a dot product is a cosine.
    Returns (score, source_id), both length Lt.
    """
    best = torch.stack([(target @ s.T).max(dim=1).values for s in sources])  # (N, Lt)
    score, source_id = best.max(dim=0)
    return score, source_id


def farthest_points(coords: torch.Tensor, k: int, seed: int = 0) -> torch.Tensor:
    """Greedy farthest-point sampling over (M, 2) coordinates."""
    if coords.shape[0] <= k:
        return coords
    picked = [seed]
    dist = (coords - coords[seed]).float().norm(dim=1)
    for _ in range(k - 1):
        nxt = int(dist.argmax())
        picked.append(nxt)
        dist = torch.minimum(dist, (coords - coords[nxt]).float().norm(dim=1))
    return coords[picked]


def select_keypoints(score, source_id, n_sources: int, ratio: float = 0.5, k: int = 5):
    """Otsu-filter the matches, then pull k spread-out keypoints per source.

    Returns a list of (<=k, 2) row/col grids, one per source; empty when a
    subject has no confident match this step.
    """
    side = int(round(math.sqrt(score.numel())))
    keep = score > threshold_otsu(score.detach().float().cpu().numpy())
    out = []
    for n in range(n_sources):
        idx = (keep & (source_id == n)).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            out.append(torch.empty(0, 2, dtype=torch.long))
            continue
        order = score[idx].argsort(descending=True)
        idx = idx[order[: max(1, int(idx.numel() * ratio))]]
        coords = torch.stack([idx // side, idx % side], dim=1)  # highest score first
        out.append(farthest_points(coords, k))
    return out


class FeatureExtractor:
    """DIFT from the diffusion UNet, DINOv2 from the decoded image."""

    def __init__(self, pipe, dino_id: str = "facebook/dinov2-base", block: int = 1, timestep: int = 261):
        from transformers import Dinov2Model

        self.pipe = pipe
        self.block = block
        self.timestep = timestep
        self.dino = Dinov2Model.from_pretrained(dino_id).to(pipe.unet.device, pipe.unet.dtype).eval()
        # Set by the pipeline once an IP-Adapter is loaded: its attention
        # processors demand image embeds on every UNet call, DIFT passes included.
        self.added_cond_kwargs = None

    @torch.no_grad()
    def dift(self, latents: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        """(B, C, h, w) activations of one up block at a fixed diffusion timestep."""
        grabbed = {}
        handle = self.pipe.unet.up_blocks[self.block].register_forward_hook(
            lambda module, args, output: grabbed.__setitem__("f", output)
        )
        try:
            t = torch.tensor(self.timestep, device=latents.device)
            # ponytail: single noise draw; DIFT's paper averages ~8. Ensemble if
            # correspondences come out noisy.
            noisy = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents), t)
            self.pipe.unet(
                noisy,
                t,
                encoder_hidden_states=embeds.expand(latents.shape[0], -1, -1),
                added_cond_kwargs=self.added_cond_kwargs,
            )
        finally:
            handle.remove()
        return grabbed["f"]

    @torch.no_grad()
    def dino_features(self, latents: torch.Tensor, size: int) -> torch.Tensor:
        """Decode the latents and read DINOv2 patch tokens, resampled to size x size."""
        images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        x = F.interpolate((images.float() + 1) / 2, size=(518, 518), mode="bilinear", align_corners=False)
        mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        tokens = self.dino(((x - mean) / std).to(self.dino.dtype)).last_hidden_state[:, 1:]  # drop CLS

        side = int(round(math.sqrt(tokens.shape[1])))
        grid = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)
        return F.interpolate(grid.float(), size=(size, size), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def __call__(self, latents: torch.Tensor, embeds: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        """Mixed features as (B, L, C) on the DIFT grid."""
        dift = self.dift(latents, embeds).float()
        dino = self.dino_features(latents, size=dift.shape[-1])
        flat = lambda g: g.flatten(2).transpose(1, 2)  # (B, C, h, w) -> (B, L, C)
        return mix_features(flat(dift), flat(dino), beta)


class Segmenter:
    """SAM, prompted with point keypoints."""

    def __init__(self, pipe, model_id: str = "facebook/sam-vit-base"):
        from transformers import SamModel, SamProcessor

        self.device = pipe.unet.device
        self.model = SamModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = SamProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def __call__(self, image, points) -> torch.Tensor:
        """image: PIL. points: list of (x, y) in pixels. -> (H, W) bool mask."""
        inputs = self.processor(image, input_points=[[points]], return_tensors="pt").to(self.device)
        out = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        return masks[0][0, 0].bool()


@torch.no_grad()
def update_layouts(
    extractor: FeatureExtractor,
    segmenter: Segmenter,
    target_latent: torch.Tensor,
    target_image,
    source_latents: list[torch.Tensor],
    source_masks: list[torch.Tensor],
    layouts: list[torch.Tensor],
    embeds: torch.Tensor,
    ratio: float = 0.5,
    k: int = 5,
    beta: float = 0.5,
) -> list[torch.Tensor]:
    """Return refined layouts, falling back to the current one where SAM has nothing to go on.

    Caller must put the attention state in "off" mode first -- this runs extra
    UNet passes that would otherwise trigger attention sharing.
    """
    target = extractor(target_latent, embeds, beta)[0]
    sources = []
    for latent, mask in zip(source_latents, source_masks):
        feats = extractor(latent, embeds, beta)[0]
        sources.append(feats[flatten_mask(mask.to(feats.device), feats.shape[0])])

    score, source_id = correspond(target, sources)
    groups = select_keypoints(score, source_id, len(sources), ratio, k)

    width, height = target_image.size
    side = int(round(math.sqrt(target.shape[0])))
    updated = []
    for coords, previous in zip(groups, layouts):
        if coords.numel() == 0:
            updated.append(previous)
            continue
        points = [[int((c + 0.5) * width / side), int((r + 0.5) * height / side)] for r, c in coords.tolist()]
        mask = segmenter(target_image, points)
        # a mask that swallowed the frame is a SAM failure, not a layout
        updated.append(previous if mask.float().mean() > 0.9 else mask.to(previous.device))
    return updated

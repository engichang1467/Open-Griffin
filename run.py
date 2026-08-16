"""Compose an image with Griffin.

    python run.py examples/dog_eagle.json --out out.png

The spec file names the target prompt and one entry per subject. Inversion is
the expensive part (50 UNet steps per subject), so its result is cached on disk
and keyed by the inputs that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from griffin.invert import invert
from griffin.layout import FeatureExtractor, Segmenter, box_mask, load_mask, save_mask
from griffin.pipeline import compose, load_adapters


def cache_path(directory: Path, subject: dict, steps: int) -> Path:
    key = json.dumps([subject["image"], subject["prompt"], subject["subject"], steps], sort_keys=True)
    return directory / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.pt"


def build_layout(subject: dict, height: int, width: int) -> torch.Tensor:
    if ("box" in subject) == ("mask" in subject):
        raise ValueError(f"subject {subject.get('subject')!r} needs exactly one of 'box' or 'mask'")
    return box_mask(subject["box"], height, width) if "box" in subject else load_mask(subject["mask"], height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path, help="JSON file: target prompt plus one entry per subject")
    parser.add_argument("--out", type=Path, default=Path("output.png"))
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cache-dir", type=Path, default=Path(".griffin_cache"))
    parser.add_argument("--no-dynamic", action="store_true", help="skip the SAM layout refinement")
    parser.add_argument("--debug-masks", type=Path, metavar="DIR",
                        help="dump source masks, layout masks and clean-image previews here")
    parser.add_argument("--no-restrict-target", action="store_true",
                        help="let each region attend to the whole target, not just itself")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text())
    subjects = spec["subjects"]
    root = args.spec.parent
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    pipe.set_progress_bar_config(disable=True)

    # Inversion has to happen before the IP-Adapter is loaded: it changes the
    # UNet's call signature, and inversion is text-conditioned only.
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    references, caches = [], []
    for subject in subjects:
        image = Image.open(root / subject["image"]).convert("RGB").resize((args.size, args.size))
        references.append(image)

        cached = cache_path(args.cache_dir, subject, args.steps)
        if cached.exists():
            print(f"inversion cache hit: {subject['subject']}")
            caches.append(torch.load(cached, weights_only=False))
        else:
            print(f"inverting {subject['subject']}...")
            cache = invert(pipe, image, subject["prompt"], subject["subject"], args.steps)
            torch.save(cache, cached)
            caches.append(cache)

    layouts = [build_layout(s, args.size, args.size) for s in subjects]

    if args.debug_masks:
        args.debug_masks.mkdir(parents=True, exist_ok=True)
        for subject, reference, cache in zip(subjects, references, caches):
            # what the inversion decided the subject was; if this clips the
            # subject, no amount of scale tuning downstream will recover it
            save_mask(cache.mask, args.debug_masks / f"source_{subject['subject']}.png", reference)

    load_adapters(pipe, len(subjects))

    extractor = segmenter = None
    if not args.no_dynamic:
        extractor, segmenter = FeatureExtractor(pipe), Segmenter(pipe)

    generator = torch.Generator(args.device).manual_seed(args.seed) if args.seed is not None else None
    print(f"composing {len(subjects)} subjects...")
    image = compose(
        pipe,
        spec["prompt"],
        references,
        caches,
        layouts,
        negative_prompt=spec.get("negative_prompt"),
        num_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.size,
        width=args.size,
        generator=generator,
        extractor=extractor,
        segmenter=segmenter,
        restrict_target=not args.no_restrict_target,
        debug_dir=args.debug_masks,
    )
    image.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

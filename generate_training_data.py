#!/usr/bin/env python3
"""
Synthetic Training Data Generation

Generates portrait photos using SDXL itself, creating two balanced groups
(light-skin and dark-skin) that serve as training data for direction
extraction. Using the same model for both groups ensures consistent image
quality and eliminates dataset licensing concerns.

Usage:
    python3 generate_training_data.py [--n N] [--force]

Args:
    --n      Number of portraits per group (default: 8, minimum recommended: 6)
    --force  Regenerate even if images already exist
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.models.stable_diffusion import StableDiffusionWrapper
from src.utils.reproducibility import collect_provenance, seed_for_index

# ---------------------------------------------------------------------------
# Prompt templates — skin tone is the only variable; everything else is fixed
# so all latent differences encode skin tone, not pose/lighting/gender/etc.
# ---------------------------------------------------------------------------

NEGATIVE = (
    "multiple people, accessories, sunglasses, jewelry, hat, cap, hood, "
    "blurry, low quality, cartoon, illustration, painting, watermark, text, "
    "extreme lighting, heavy shadows, overexposed, underexposed, cropped face, "
    "deformed, ugly, disfigured"
)

# Each tuple: (skin_tone_descriptor, seed_offset)
# Varying the descriptor slightly ("light" vs "fair" vs "pale") adds intra-group
# diversity while keeping skin tone the consistent signal.
LIGHT_DESCRIPTORS = [
    "light skin tone",
    "fair skin tone",
    "light complexion",
    "pale skin tone",
    "fair complexion",
    "light skin",
    "fair skin",
    "light-skinned",
]

DARK_DESCRIPTORS = [
    "dark skin tone",
    "deep skin tone",
    "dark complexion",
    "rich dark skin tone",
    "deep brown skin tone",
    "dark skin",
    "deep complexion",
    "dark-skinned",
]

BASE_PROMPT = (
    "professional headshot portrait of a person with {skin_tone}, "
    "neutral expression, clean white studio background, soft diffused studio "
    "lighting, sharp focus on face, centered composition, no glasses no jewelry "
    "no hat, facing camera directly, high quality photography, 85mm lens"
)

# Seeds chosen to produce varied gender/age/facial structure while keeping
# skin tone as the controlled variable. Using the same seed across both groups
# at each index would produce the same face structure — intentional.
SEEDS = [42, 137, 256, 512, 777, 1536, 2048, 3141]


def generate_group(
    model: StableDiffusionWrapper,
    descriptors: list,
    seeds: list,
    out_dir: Path,
    label: str,
    n: int,
    force: bool,
) -> list:
    """Generate n portraits for one skin-tone group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []

    for i in range(n):
        path = out_dir / f"portrait_{i:02d}.png"

        if path.exists() and not force:
            print(f"  [{label}] {path.name} — already exists, skipping")
            images.append(Image.open(path).convert("RGB"))
            continue

        descriptor = descriptors[i % len(descriptors)]
        seed = seed_for_index(i, seeds)
        prompt = BASE_PROMPT.format(skin_tone=descriptor)

        print(f"  [{label}] {i+1}/{n}  seed={seed}  '{descriptor}'...", end=" ", flush=True)

        img, _ = model.generate_from_prompt(
            prompt,
            negative_prompt=NEGATIVE,
            seed=seed,
            num_inference_steps=25,
            guidance_scale=7.5,
            return_latent=False,
        )

        img.save(path)
        images.append(img)
        print("done")

    return images


def check_contrast(light_images: list, dark_images: list) -> float:
    """Return brightness gap between group means (centre-crop)."""
    def group_brightness(imgs):
        bs = []
        for img in imgs:
            arr = np.array(img)
            h, w = arr.shape[:2]
            crop = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            bs.append(crop.mean())
        return float(np.mean(bs))

    lb = group_brightness(light_images)
    db = group_brightness(dark_images)
    return lb, db


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8,
                   help="Portraits per group (default 8)")
    p.add_argument("--force", action="store_true",
                   help="Regenerate existing images")
    args = p.parse_args()

    n = max(args.n, 4)  # enforce minimum for a stable vector
    if n < 6:
        print("WARNING: fewer than 6 images per group may produce an unstable vector.")

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    print("=" * 60)
    print("SYNTHETIC TRAINING DATA GENERATION")
    print("=" * 60)
    print(f"Device:      {device}")
    print(f"Per group:   {n} portraits")
    print(f"Total:       {n * 2} images")
    print()

    print("Loading SDXL...")
    model = StableDiffusionWrapper(
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        enable_xformers=True,
        enable_cpu_offload=(device == "cpu"),
    )
    print()

    light_dir = Path("data/photos/light_skin")
    dark_dir  = Path("data/photos/dark_skin")

    print("Generating light-skin portraits...")
    light_imgs = generate_group(
        model, LIGHT_DESCRIPTORS, SEEDS, light_dir, "light", n, args.force
    )

    print("\nGenerating dark-skin portraits...")
    dark_imgs = generate_group(
        model, DARK_DESCRIPTORS, SEEDS, dark_dir, "dark", n, args.force
    )

    # Verify contrast
    print()
    lb, db = check_contrast(light_imgs, dark_imgs)
    diff = abs(lb - db)

    print("=" * 60)
    print("CONTRAST CHECK")
    print("=" * 60)
    print(f"Light-skin avg brightness: {lb:.1f}/255")
    print(f"Dark-skin  avg brightness: {db:.1f}/255")
    print(f"Difference:                {diff:.1f}")

    if diff < 20:
        print("\nWARNING: Low contrast — try --n 10 or --force to regenerate.")
    elif diff < 40:
        print("\nOK: Moderate contrast. Results will be visible but subtle.")
    else:
        print(f"\nGood contrast ({diff:.1f}). Ready to estimate the direction.")

    print()
    manifest = {
        "schema_version": "1.0",
        "status": "synthetic_paired_training_data",
        "model_id": model.model_id,
        "pairing": "same seed and composition prompt; skin-tone descriptor differs",
        "n_pairs": n,
        "pairs": [
            {
                "pair_id": i,
                "seed": seed_for_index(i, SEEDS),
                "light": {
                    "descriptor": LIGHT_DESCRIPTORS[i % len(LIGHT_DESCRIPTORS)],
                    "path": str(light_dir / f"portrait_{i:02d}.png"),
                },
                "dark": {
                    "descriptor": DARK_DESCRIPTORS[i % len(DARK_DESCRIPTORS)],
                    "path": str(dark_dir / f"portrait_{i:02d}.png"),
                },
            }
            for i in range(n)
        ],
        "provenance": collect_provenance(Path(__file__).parent),
    }
    manifest_path = Path("data/generated/training_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")
    print()
    print("Next step:")
    print("  python3 run_race_vector_extraction.py")
    print()


if __name__ == "__main__":
    main()

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
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.stable_diffusion import StableDiffusionWrapper
from src.study.config import load_study_config
from src.utils.reproducibility import (
    collect_provenance,
    seed_for_index,
    stable_fingerprint,
)

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

# Seeds chosen to produce varied apparent age, gender presentation, and facial
# structure. Reusing a seed couples the initial diffusion noise and reduces one
# source of variation, but changed text conditioning can still change apparent
# identity or composition; these are not identity-matched photographs.
SEEDS = [42, 137, 256, 512, 777, 1024, 2048, 3141]


def configured_seed_schedule(config) -> list[int]:
    """Return the frozen paired-data seed schedule, with a legacy default."""

    values = config.data.get("seed_schedule", SEEDS)
    seeds = [int(value) for value in values]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("data.seed_schedule must be non-empty and unique")
    return seeds


def generate_group(
    model: StableDiffusionWrapper,
    descriptors: list,
    seeds: list,
    out_dir: Path,
    label: str,
    n: int,
    force: bool,
    *,
    inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    prompt_template: str,
    negative_prompt: str,
    generation_signature: str,
    ledger_path: Path,
) -> list:
    """Generate n portraits for one skin-tone group."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []

    for i in range(n):
        path = out_dir / f"portrait_{i:02d}.png"
        descriptor = descriptors[i % len(descriptors)]
        seed = seed_for_index(i, seeds)

        expected = {
            "group": label,
            "index": i,
            "path": str(path),
            "descriptor": descriptor,
            "seed": seed,
            "generation_signature": generation_signature,
        }
        attested = valid_generation_attestation(path, expected, ledger_path)
        if path.exists() and not force and attested:
            print(f"  [{label}] {path.name} — already exists, skipping")
            images.append(Image.open(path).convert("RGB"))
            continue

        if path.exists() and not force:
            print(
                f"  [{label}] {path.name} — missing a valid campaign attestation; "
                "regenerating"
            )

        prompt = prompt_template.format(skin_tone=descriptor)

        print(f"  [{label}] {i+1}/{n}  seed={seed}  '{descriptor}'...", end=" ", flush=True)

        img, _ = model.generate_from_prompt(
            prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            return_latent=False,
        )

        img.save(path)
        append_generation_attestation(path, expected, ledger_path)
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


def sha256_file(path: Path) -> str:
    """Hash an image without loading its pixels into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generation_signature(config) -> str:
    """Fingerprint only inputs that can affect paired portrait generation."""

    return stable_fingerprint(
        {
            "schema_version": "1.0",
            "model_id": config.model["id"],
            "model_revision": config.model["revision"],
            "inference_steps": int(config.model["inference_steps"]),
            "guidance_scale": float(config.model["guidance_scale"]),
            "height": int(config.model["height"]),
            "width": int(config.model["width"]),
            "prompt_template": str(config.prompts["attribute_template"]),
            "negative_prompt": str(config.prompts["negative"]),
            "light_descriptors": LIGHT_DESCRIPTORS,
            "dark_descriptors": DARK_DESCRIPTORS,
            "seed_schedule": configured_seed_schedule(config),
            "seed_extension": "base_seed_plus_cycle_times_10000",
        },
        length=64,
    )


def load_generation_attestations(ledger_path: Path) -> dict[str, dict]:
    """Load the latest strict-JSON attestation for each generated image path."""

    records = {}
    if not ledger_path.exists():
        return records
    with ledger_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                records[str(row["path"])] = row
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Malformed generation ledger row {line_number}: {exc}"
                ) from exc
    return records


def valid_generation_attestation(
    image_path: Path,
    expected: dict,
    ledger_path: Path,
) -> bool:
    """Return whether an image is hash-attested under the active campaign."""

    if not image_path.is_file():
        return False
    record = load_generation_attestations(ledger_path).get(str(image_path))
    if record is None:
        return False
    if any(record.get(key) != value for key, value in expected.items()):
        return False
    recorded_hash = record.get("sha256")
    return isinstance(recorded_hash, str) and sha256_file(image_path) == recorded_hash


def append_generation_attestation(
    image_path: Path,
    expected: dict,
    ledger_path: Path,
) -> None:
    """Append an image hash only after generation and durable image saving."""

    row = {
        "schema_version": "1.0",
        **expected,
        "sha256": sha256_file(image_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, allow_nan=False)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()


def write_manifest(
    config,
    n: int,
    light_dir: Path,
    dark_dir: Path,
    *,
    ledger_path: Path,
    signature: str,
) -> Path:
    """Write a content-addressed paired-data manifest."""

    seeds = configured_seed_schedule(config)
    pairs = []
    for i in range(n):
        light_path = light_dir / f"portrait_{i:02d}.png"
        dark_path = dark_dir / f"portrait_{i:02d}.png"
        if not light_path.is_file() or not dark_path.is_file():
            raise FileNotFoundError(
                f"Missing paired images for pair {i}: {light_path}, {dark_path}"
            )
        pairs.append(
            {
                "pair_id": i,
                "seed": seed_for_index(i, seeds),
                "light": {
                    "descriptor": LIGHT_DESCRIPTORS[i % len(LIGHT_DESCRIPTORS)],
                    "path": str(light_path),
                    "sha256": sha256_file(light_path),
                },
                "dark": {
                    "descriptor": DARK_DESCRIPTORS[i % len(DARK_DESCRIPTORS)],
                    "path": str(dark_path),
                    "sha256": sha256_file(dark_path),
                },
            }
        )
    attestations = load_generation_attestations(ledger_path)
    expected_attestations = []
    for pair in pairs:
        for group in ("light", "dark"):
            image = pair[group]
            expected = {
                "group": group,
                "index": int(pair["pair_id"]),
                "path": str(image["path"]),
                "descriptor": image["descriptor"],
                "seed": int(pair["seed"]),
                "generation_signature": signature,
                "sha256": image["sha256"],
            }
            observed = attestations.get(str(image["path"]), {})
            expected_attestations.append(
                all(observed.get(key) == value for key, value in expected.items())
            )
    campaign_complete = len(expected_attestations) == n * 2 and all(
        expected_attestations
    )
    manifest = {
        "schema_version": "2.0",
        "status": "synthetic_paired_training_data",
        "study_config_fingerprint": config.fingerprint,
        "model_id": config.model["id"],
        "model_revision": config.model["revision"],
        "inference_steps": int(config.model["inference_steps"]),
        "guidance_scale": float(config.model["guidance_scale"]),
        "height": int(config.model["height"]),
        "width": int(config.model["width"]),
        "prompt_template": str(config.prompts["attribute_template"]),
        "negative_prompt": str(config.prompts["negative"]),
        "light_descriptors": LIGHT_DESCRIPTORS,
        "dark_descriptors": DARK_DESCRIPTORS,
        "seed_schedule": seeds,
        # The legacy field remains for runner compatibility. Its strengthened
        # meaning is that every image was observed in this resumable campaign.
        "generation_observed_in_this_run": campaign_complete,
        "generation_observed_in_campaign": campaign_complete,
        "generation_signature": signature,
        "generation_ledger": str(ledger_path),
        "generation_ledger_sha256": (
            sha256_file(ledger_path) if ledger_path.is_file() else None
        ),
        "pairing": (
            "noise-coupled prompt pair: same seed and prompt template; "
            "skin-tone descriptor differs; apparent identity is not guaranteed"
        ),
        "n_pairs": n,
        "pairs": pairs,
        "provenance": collect_provenance(Path(__file__).parent),
    }
    manifest_path = Path(config.data["training_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8,
                   help="Portraits per group (default 8)")
    p.add_argument("--force", action="store_true",
                   help="Regenerate existing images")
    p.add_argument("--config", type=Path, default=Path("configs/full_study.yaml"))
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="Hash existing paired images without loading SDXL",
    )
    args = p.parse_args()
    config = load_study_config(args.config)
    seeds = configured_seed_schedule(config)
    signature = generation_signature(config)
    manifest_path = Path(config.data["training_manifest"])
    ledger_path = manifest_path.with_name(
        f"{manifest_path.stem}.generation.jsonl"
    )

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

    light_dir = Path(config.data["light_dir"])
    dark_dir = Path(config.data["dark_dir"])

    if args.manifest_only:
        manifest_path = write_manifest(
            config,
            n,
            light_dir,
            dark_dir,
            ledger_path=ledger_path,
            signature=signature,
        )
        print(f"Manifest: {manifest_path}")
        return

    print("Loading SDXL...")
    model = StableDiffusionWrapper(
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        model_id=str(config.model["id"]),
        model_revision=str(config.model["revision"]),
        local_files_only=bool(config.model.get("local_files_only", False)),
        enable_xformers=device == "cuda",
        enable_cpu_offload=(device == "cpu"),
    )
    print()

    print("Generating light-skin portraits...")
    light_imgs = generate_group(
        model,
        LIGHT_DESCRIPTORS,
        seeds,
        light_dir,
        "light",
        n,
        args.force,
        inference_steps=int(config.model["inference_steps"]),
        guidance_scale=float(config.model["guidance_scale"]),
        height=int(config.model["height"]),
        width=int(config.model["width"]),
        prompt_template=str(config.prompts["attribute_template"]),
        negative_prompt=str(config.prompts["negative"]),
        generation_signature=signature,
        ledger_path=ledger_path,
    )

    print("\nGenerating dark-skin portraits...")
    dark_imgs = generate_group(
        model,
        DARK_DESCRIPTORS,
        seeds,
        dark_dir,
        "dark",
        n,
        args.force,
        inference_steps=int(config.model["inference_steps"]),
        guidance_scale=float(config.model["guidance_scale"]),
        height=int(config.model["height"]),
        width=int(config.model["width"]),
        prompt_template=str(config.prompts["attribute_template"]),
        negative_prompt=str(config.prompts["negative"]),
        generation_signature=signature,
        ledger_path=ledger_path,
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
    manifest_path = write_manifest(
        config,
        n,
        light_dir,
        dark_dir,
        ledger_path=ledger_path,
        signature=signature,
    )
    print(f"Manifest: {manifest_path}")
    print()
    print("Next step:")
    print("  python3 run_race_vector_extraction.py")
    print()


if __name__ == "__main__":
    main()

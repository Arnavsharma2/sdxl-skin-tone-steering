#!/usr/bin/env python3
"""
Skin-Tone Direction Extraction and Counterfactual Generation

Extracts a "skin tone direction" from portrait photos in the SDXL latent
space, then uses steered denoising to generate counterfactual images that
vary along that axis while preserving identity and pose.

Usage:
    python3 run_race_vector_extraction.py [--steps N] [--alphas A B C ...]

Requirements:
    - Photos in data/photos/light_skin/ and data/photos/dark_skin/
    - At least 3 photos in each directory
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.latent.vector_discovery import SkinToneDirectionExtractor, VectorAnalyzer
from src.metrics.evaluator import CounterfactualEvaluator
from src.models.stable_diffusion import StableDiffusionWrapper
from src.utils.reproducibility import (
    collect_provenance,
    seed_everything,
    stable_fingerprint,
)
from src.visualization.grid_generator import CounterfactualGridGenerator

# ---------------------------------------------------------------------------
# Prompts — crafted for consistent, evaluable portraits
# ---------------------------------------------------------------------------
PORTRAIT_PROMPT = (
    "professional headshot portrait, neutral expression, clean white studio background, "
    "soft diffused studio lighting, sharp focus on face, centered composition, "
    "natural skin tones, no glasses no jewelry no hat, facing camera directly, "
    "high quality photography, 85mm lens"
)

NEGATIVE_PROMPT = (
    "multiple people, accessories, sunglasses, jewelry, hat, cap, hood, "
    "blurry, low quality, cartoon, illustration, painting, watermark, text, "
    "extreme lighting, heavy shadows, overexposed, underexposed, cropped face"
)


def _clear_cache(device: str):
    """Free GPU/MPS memory."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


class SkinToneSteeringPipeline:
    """End-to-end pipeline for skin-tone direction estimation and evaluation."""

    def __init__(self, device=None, output_dir="experiments/results",
                 num_inference_steps=25):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cf_dir = self.output_dir / "counterfactuals"
        self.cf_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = Path("data/generated")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.num_inference_steps = num_inference_steps

        # Populated later
        self.model = None
        self.extractor = None
        self.race_vector = None
        self.base_image = None
        self.counterfactual_images = []
        self.alphas = []
        self.results = []
        self.evaluator = None
        self.generation_prompt = PORTRAIT_PROMPT
        self.generation_seed = 999
        self.generation_negative_prompt = NEGATIVE_PROMPT

        print("=" * 70)
        print("SKIN-TONE DIRECTION EXTRACTION PIPELINE")
        print("=" * 70)
        print(f"Device:              {self.device}")
        print(f"Inference steps:     {self.num_inference_steps} (DPM++ 2M Karras)")
        print(f"Output directory:    {self.output_dir}")
        print()

    # -----------------------------------------------------------------------
    # Step 1 — Model
    # -----------------------------------------------------------------------
    def load_model(self):
        print("STEP 1: Loading Stable Diffusion XL...")
        print("-" * 70)
        self.model = StableDiffusionWrapper(
            device=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            enable_xformers=True,
            enable_cpu_offload=(self.device == "cpu"),
        )
        print("Model loaded.\n")

    # -----------------------------------------------------------------------
    # Step 2 — Photos
    # -----------------------------------------------------------------------
    def load_photos(self, light_dir="data/photos/light_skin",
                    dark_dir="data/photos/dark_skin", max_photos=10):
        print("STEP 2: Loading and encoding photos...")
        print("-" * 70)

        light_path = Path(light_dir)
        dark_path = Path(dark_dir)

        def find_images(p):
            return sorted(
                list(p.glob("*.jpg")) +
                list(p.glob("*.jpeg")) +
                list(p.glob("*.png")) +
                list(p.glob("*.webp"))
            )

        light_files = find_images(light_path)
        dark_files = find_images(dark_path)

        print(f"Found {len(light_files)} light-skin photos")
        print(f"Found {len(dark_files)} dark-skin photos")

        if not light_files or not dark_files:
            print("\nERROR: No photos found.")
            print(f"  Expected: {light_path.absolute()}")
            print(f"  Expected: {dark_path.absolute()}")
            sys.exit(1)

        if len(light_files) < 3 or len(dark_files) < 3:
            print("\nWARNING: Recommend at least 3 photos per group for a stable vector.")

        def encode_group(files, label):
            images, latents = [], []
            for p in files[:max_photos]:
                print(f"  Encoding [{label}] {p.name}...", end=" ", flush=True)
                img = Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
                lat = self.model.encode_image(img)
                images.append(img)
                latents.append(lat)
                print("done")
            return images, latents

        self.light_images, self.light_latents = encode_group(light_files, "light")
        self.dark_images, self.dark_latents = encode_group(dark_files, "dark")

        print(f"\nEncoded {len(self.light_images)} light + {len(self.dark_images)} dark photos\n")

    # -----------------------------------------------------------------------
    # Step 3 — Quality check
    # -----------------------------------------------------------------------
    def check_photo_quality(self):
        print("STEP 3: Checking photo contrast...")
        print("-" * 70)

        def avg_brightness(images):
            bs = []
            for img in images:
                arr = np.array(img)
                h, w = arr.shape[:2]
                center = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
                bs.append(center.mean())
            return float(np.mean(bs))

        lb = avg_brightness(self.light_images)
        db = avg_brightness(self.dark_images)
        diff = abs(lb - db)

        print(f"Light-skin avg brightness: {lb:.1f}/255")
        print(f"Dark-skin  avg brightness: {db:.1f}/255")
        print(f"Difference:                {diff:.1f}")

        if diff < 20:
            print("\nWARNING: Very small brightness gap — direction may be weak.")
            print("  For best results aim for a gap > 40.\n")
        elif diff < 40:
            print("\nWARNING: Moderate gap; results may be subtle.\n")
        else:
            print(f"\nGood contrast ({diff:.1f} points). Proceeding.\n")

    # -----------------------------------------------------------------------
    # Step 4 — Race vector
    # -----------------------------------------------------------------------
    def extract_race_vector(self, radius=1.0, edge_weight=0.3, optimize=False):
        print("STEP 4: Extracting skin-tone direction...")
        print("-" * 70)

        self.extractor = SkinToneDirectionExtractor(device=self.device)

        lat_shape = self.light_latents[0].shape
        h, w = lat_shape[-2], lat_shape[-1]

        spatial_mask = self.extractor.create_center_mask(
            height=h, width=w,
            center_weight=1.0,
            edge_weight=edge_weight,
            falloff="gaussian",
            radius=radius,
        )
        print(f"Spatial mask: center={spatial_mask[h//2, w//2].item():.3f}, "
              f"corner={spatial_mask[0, 0].item():.3f}")

        self.race_vector = self.extractor.extract_from_pairs(
            self.light_latents,
            self.dark_latents,
            normalize=False,
            spatial_mask=spatial_mask,
        )
        print(f"Raw vector  shape: {self.race_vector.shape}")
        print(f"Raw vector  norm:  {self.race_vector.norm().item():.4f}")

        self._visualize_vector(spatial_mask, tag="raw")

        # ------------------------------------------------------------------
        # Optional: optimise vector for identity preservation
        # ------------------------------------------------------------------
        if not optimize:
            print("\nSTEP 4b: Optimisation disabled (pilot protocol default).\n")
            return

        print("\nSTEP 4b: Running experimental vector optimisation...")
        print("-" * 70)
        try:
            import torch.nn.functional as F
            from facenet_pytorch import InceptionResnetV1

            facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            for p in facenet.parameters():
                p.requires_grad = False

            def identity_loss(latents_orig, latents_mod):
                total = 0.0
                for orig, mod in zip(latents_orig, latents_mod):
                    if orig.dim() == 3:
                        orig = orig.unsqueeze(0)
                    if mod.dim() == 3:
                        mod = mod.unsqueeze(0)
                    o_img = self.model.vae.decode(
                        orig / self.model.vae.config.scaling_factor
                    ).sample
                    m_img = self.model.vae.decode(
                        mod / self.model.vae.config.scaling_factor
                    ).sample
                    o_face = F.interpolate(o_img, size=(160, 160), mode="bilinear",
                                           align_corners=False)
                    m_face = F.interpolate(m_img, size=(160, 160), mode="bilinear",
                                           align_corners=False)
                    e_o = facenet(o_face)
                    e_m = facenet(m_face)
                    total += 1.0 - F.cosine_similarity(e_o, e_m).mean()
                return total / len(latents_orig)

            def attribute_loss(latents_orig, latents_mod):
                diff = torch.stack(latents_mod) - torch.stack(latents_orig)
                return diff.norm(p=2)

            # Use a small subset to avoid OOM during gradient-tracked VAE decoding
            train_latents = self.light_latents[:2] + self.dark_latents[:2]

            self.race_vector = self.extractor.optimize_vector(
                initial_vector=self.race_vector,
                latents=train_latents,
                identity_loss_fn=identity_loss,
                attribute_change_fn=attribute_loss,
                num_iterations=50,
                lr=0.01,
                lambda_identity=0.7,
                lambda_attribute=0.3,
            )
            _clear_cache(self.device)
            print(f"Optimised vector norm: {self.race_vector.norm().item():.4f}")
            self._visualize_vector(spatial_mask, tag="optimised")

        except ImportError:
            print("  facenet-pytorch not installed — skipping optimisation.")
            print("  Install: pip install facenet-pytorch")
        except Exception as e:
            print(f"  Optimisation failed ({e}) — using raw vector.")

        print()

    def _visualize_vector(self, spatial_mask, tag=""):
        analyzer = VectorAnalyzer(device=self.device)
        analysis = analyzer.analyze_spatial_pattern(self.race_vector)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        im0 = axes[0].imshow(spatial_mask.cpu().numpy(), cmap="hot")
        axes[0].set_title("Spatial Mask\n(high = face region)")
        plt.colorbar(im0, ax=axes[0], label="Weight")

        im1 = axes[1].imshow(analysis["spatial_heatmap"].cpu().numpy(), cmap="hot")
        axes[1].set_title("Race Vector Activation\n(after masking)")
        plt.colorbar(im1, ax=axes[1], label="Magnitude²")

        plt.tight_layout()
        fname = f"spatial_mask_and_vector{'_' + tag if tag else ''}.png"
        plt.savefig(self.output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {self.output_dir / fname}")

    # -----------------------------------------------------------------------
    # Step 5 — Base image
    # -----------------------------------------------------------------------
    def generate_base_image(self, prompt=None, seed=999):
        print("STEP 5: Generating base image...")
        print("-" * 70)

        if prompt is not None:
            self.generation_prompt = prompt
        self.generation_seed = seed

        cache_key = stable_fingerprint(
            {
                "model": self.model.model_id,
                "prompt": self.generation_prompt,
                "negative_prompt": self.generation_negative_prompt,
                "seed": self.generation_seed,
                "steps": self.num_inference_steps,
                "guidance_scale": 7.5,
            }
        )
        cache_path = self.cache_dir / f"base_image_{cache_key}.png"
        out_path = self.output_dir / "base_image.png"

        if cache_path.exists():
            print(f"Loading cached base image from {cache_path}")
            self.base_image = Image.open(cache_path).convert("RGB").resize(
                (512, 512), Image.LANCZOS
            )
            self.base_image.save(out_path)
            print(f"Copied to {out_path}\n")
            return

        print(f"Prompt:  {self.generation_prompt[:80]}...")
        print(f"Seed:    {self.generation_seed}")
        print(f"Steps:   {self.num_inference_steps}")

        self.base_image, _ = self.model.generate_from_prompt(
            self.generation_prompt,
            negative_prompt=self.generation_negative_prompt,
            seed=self.generation_seed,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=7.5,
        )

        self.base_image.save(cache_path)
        self.base_image.save(out_path)
        print(f"Saved: {out_path}\n")

    # -----------------------------------------------------------------------
    # Step 6 — Counterfactuals
    # -----------------------------------------------------------------------
    def generate_counterfactuals(self, alphas=None):
        print("STEP 6: Generating counterfactuals (steered denoising)...")
        print("-" * 70)

        if alphas is None:
            alphas = [-1.5, -0.75, 0, 0.75, 1.5]

        self.alphas = alphas
        self.counterfactual_images = []

        print(f"Alpha values: {alphas}")
        print("Negative α = lighter skin, Positive α = darker skin\n")

        for alpha in alphas:
            print(f"  α = {alpha:+.1f} ...", end=" ", flush=True)

            if abs(alpha) < 0.01:
                self.counterfactual_images.append(self.base_image)
                # Save α=0 as the unmodified base
                self.base_image.save(self.cf_dir / "alpha_+0.0.png")
                print("(base image)")
                continue

            img, _ = self.model.generate_steered(
                prompt=self.generation_prompt,
                race_vector=self.race_vector,
                alpha=alpha,
                seed=self.generation_seed,
                negative_prompt=self.generation_negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=7.5,
            )
            self.counterfactual_images.append(img)

            # Save each counterfactual individually
            fname = f"alpha_{alpha:+.1f}.png"
            img.save(self.cf_dir / fname)
            print(f"done → {self.cf_dir / fname}")

        print(f"\nGenerated {len(self.counterfactual_images)} images total\n")
        self._visualize_counterfactuals()

    def _visualize_counterfactuals(self):
        n = len(self.alphas)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]

        for ax, img, alpha in zip(axes, self.counterfactual_images, self.alphas):
            ax.imshow(img)
            ax.set_title(f"α = {alpha:+.1f}", fontsize=13)
            ax.axis("off")

        plt.suptitle("Skin Tone Steering via Race Vector in Latent Space",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = self.output_dir / "counterfactuals_strip.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved strip: {path}")

    # -----------------------------------------------------------------------
    # Step 7 — Evaluation
    # -----------------------------------------------------------------------
    def evaluate_counterfactuals(self):
        print("STEP 7: Evaluating identity & structural preservation...")
        print("-" * 70)

        self.evaluator = CounterfactualEvaluator(device=self.device)
        self.results = []

        for cf_img, alpha in zip(self.counterfactual_images, self.alphas):
            if abs(alpha) < 0.01:
                continue
            print(f"\n  α = {alpha:+.1f}")
            result = self.evaluator.evaluate_pair(
                self.base_image, cf_img, alpha=alpha, verbose=True
            )
            self.results.append((alpha, result))

        print("\nEvaluation complete.\n")

    # -----------------------------------------------------------------------
    # Step 8 — Final grid
    # -----------------------------------------------------------------------
    def create_final_grid(self):
        print("STEP 8: Creating publication-quality grid...")
        print("-" * 70)

        generator = CounterfactualGridGenerator(font_size=18)

        cf_imgs = [img for img, a in zip(self.counterfactual_images, self.alphas)
                   if abs(a) >= 0.01]
        labels = [f"α = {a:+.1f}" for a in self.alphas if abs(a) >= 0.01]
        metrics_list = [r.to_dict() for _, r in self.results]

        grid = generator.generate_grid(
            self.base_image,
            cf_imgs,
            labels=labels,
            metrics=metrics_list if metrics_list else None,
            title="Race Vector Counterfactual Generation (SDXL Latent Space)",
        )

        path = self.output_dir / "final_grid.png"
        grid.save(path)
        print(f"Final grid saved: {path}\n")

    # -----------------------------------------------------------------------
    # Step 9 — Save metadata
    # -----------------------------------------------------------------------
    def save_metadata(self):
        metadata = {
            "schema_version": "1.0",
            "status": "pilot",
            "terminology": {
                "estimated_attribute": "visual skin-tone direction",
                "explicitly_not_estimated": "race or ethnicity",
            },
            "prompt": self.generation_prompt,
            "negative_prompt": self.generation_negative_prompt,
            "seed": self.generation_seed,
            "num_inference_steps": self.num_inference_steps,
            "alphas": self.alphas,
            "device": self.device,
            "n_light_photos": len(self.light_images),
            "n_dark_photos": len(self.dark_images),
            "direction_norm": float(self.race_vector.norm().item()),
            "model_id": self.model.model_id if self.model is not None else None,
            "scheduler": "DPMSolverMultistepScheduler (DPM-Solver++)",
            "thresholds": (
                vars(self.evaluator.thresholds) if self.evaluator is not None else None
            ),
            "metric_provenance": (
                self.evaluator.metric_provenance() if self.evaluator is not None else None
            ),
            "provenance": collect_provenance(Path(__file__).parent),
            "results": [
                {"alpha": a, **r.to_dict()}
                for a, r in self.results
            ],
        }
        path = self.output_dir / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved: {path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    def print_summary(self):
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"\nPhotos:  {len(self.light_images)} light  +  {len(self.dark_images)} dark")
        print(f"Direction norm: {self.race_vector.norm().item():.4f}")
        print(f"Counterfactuals:  {len(self.counterfactual_images)}")

        if self.results:
            print("\n  α     FaceSim  BgSSIM  Pose°  ΔrelL*  Score  Valid")
            print("  " + "-" * 55)
            for alpha, r in self.results:
                fs  = f"{r.face_similarity:.3f}" if r.face_similarity is not None else "  N/A "
                bg  = f"{r.background_ssim:.3f}" if r.background_ssim is not None else "  N/A "
                pd_val = r.total_pose_diff
                pd  = f"{pd_val:.1f}°" if (pd_val is not None and not np.isinf(pd_val)) else " N/A "
                tone = (
                    f"{r.skin_tone_change:+.2f}"
                    if r.skin_tone_change is not None
                    else " N/A "
                )
                sc = f"{r.overall_score:.3f}" if r.overall_score is not None else " N/A "
                valid = "YES" if r.counterfactual_success else " NO"
                print(
                    f"  {alpha:+5.1f}   {fs}    {bg}   {pd:>5}  "
                    f"{tone:>6}  {sc}  {valid}"
                )

        print(f"\nOutputs in: {self.output_dir.absolute()}")
        print("  base_image.png           — unsteered base portrait")
        print("  counterfactuals_strip.png — all alpha values in one strip")
        print("  final_grid.png           — grid with metrics overlay")
        print("  counterfactuals/         — individual alpha images")
        print("  metadata.json            — full run metadata")
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Race vector extraction pipeline")
    p.add_argument("--steps", type=int, default=25,
                   help="Denoising steps (default 25 with DPM++ 2M)")
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[-1.5, -0.75, 0, 0.75, 1.5],
        help="Alpha values (default: -1.5 -0.75 0 0.75 1.5)",
    )
    p.add_argument("--seed", type=int, default=999,
                   help="Generation seed for reproducibility")
    p.add_argument("--output", type=str, default="experiments/results",
                   help="Output directory")
    p.add_argument("--light-dir", type=str, default="data/photos/light_skin")
    p.add_argument("--dark-dir", type=str, default="data/photos/dark_skin")
    p.add_argument("--max-photos", type=int, default=10)
    p.add_argument("--eval-only", action="store_true",
                   help="Skip generation; re-evaluate existing images in output dir")
    p.add_argument(
        "--optimize-vector",
        action="store_true",
        help=(
            "Enable experimental vector optimisation. Disabled by default because "
            "it is not part of the preregistered pilot protocol."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    pipeline = SkinToneSteeringPipeline(
        output_dir=args.output,
        num_inference_steps=args.steps,
    )

    try:
        if args.eval_only:
            _eval_only(pipeline, args)
        else:
            _full_run(pipeline, args)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


def _full_run(pipeline, args):
    seed_everything(args.seed)
    pipeline.load_model()
    pipeline.load_photos(
        light_dir=args.light_dir,
        dark_dir=args.dark_dir,
        max_photos=args.max_photos,
    )
    pipeline.check_photo_quality()
    pipeline.extract_race_vector(
        radius=1.0,
        edge_weight=0.3,
        optimize=args.optimize_vector,
    )
    pipeline.generate_base_image(seed=args.seed)
    pipeline.generate_counterfactuals(alphas=args.alphas)
    pipeline.evaluate_counterfactuals()
    pipeline.create_final_grid()
    pipeline.save_metadata()
    pipeline.print_summary()


def _eval_only(pipeline, args):
    """Re-evaluate existing images without regenerating anything."""
    out = Path(args.output)
    base_path = out / "base_image.png"
    cf_dir = out / "counterfactuals"

    if not base_path.exists():
        print(f"ERROR: {base_path} not found. Run without --eval-only first.")
        sys.exit(1)

    print("EVAL-ONLY MODE — loading existing images")
    print("-" * 70)

    pipeline.base_image = Image.open(base_path).convert("RGB")
    print(f"Loaded base image: {base_path}")

    # Load counterfactuals sorted by alpha value
    cf_files = sorted(cf_dir.glob("alpha_*.png")) if cf_dir.exists() else []
    if not cf_files:
        print(f"ERROR: No counterfactual images found in {cf_dir}")
        sys.exit(1)

    pipeline.counterfactual_images = []
    pipeline.alphas = []
    for f in cf_files:
        # Parse alpha from filename e.g. alpha_+1.5.png → 1.5
        try:
            alpha = float(f.stem.replace("alpha_", ""))
        except ValueError:
            continue
        pipeline.alphas.append(alpha)
        pipeline.counterfactual_images.append(Image.open(f).convert("RGB"))
        print(f"  Loaded {f.name}")

    print(f"\nEvaluating {len(pipeline.alphas)} counterfactuals...\n")
    pipeline.evaluate_counterfactuals()
    pipeline.create_final_grid()

    # Stub missing fields for summary
    pipeline.light_images = []
    pipeline.dark_images = []
    pipeline.race_vector = type("V", (), {"norm": lambda self: type("N", (), {"item": lambda self: 0.0})()})()

    pipeline.print_summary()


if __name__ == "__main__":
    main()

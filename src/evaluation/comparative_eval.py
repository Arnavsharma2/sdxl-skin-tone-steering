"""Legacy exploratory comparison script.

This pre-protocol script compares post-hoc latent manipulation with a prompt
baseline. It is retained for provenance and must not be used as the confirmatory
runner described in ``docs/RESEARCH_PROTOCOL.md``.

Usage:
    python3 src/evaluation/comparative_eval.py --num_samples 10 --batch_size 2
"""

import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stable_diffusion import StableDiffusionWrapper
from src.latent.vector_discovery import SkinToneDirectionExtractor
from src.latent.manipulator import LatentManipulator
from src.metrics.evaluator import CounterfactualEvaluator

class ComparativeEvaluator:
    """
    Evaluates and compares different manipulation methods.
    """

    def __init__(self, device=None, output_dir="experiments/comparative_results"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Model
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        print(f"Loading Stable Diffusion on {self.device} with {dtype}...")
        self.model = StableDiffusionWrapper(
            device=self.device,
            dtype=dtype,
            enable_xformers=True
        )
        
        # Initialize Metrics
        self.evaluator = CounterfactualEvaluator(device=self.device)
        self.manipulator = LatentManipulator(device=self.device)
        
        # Load direction (assumes it is already extracted, or estimates a fresh one).
        # For this script, we'll try to load a cached one or extract on the fly if needed
        # But for robustness, let's just extract a fresh one quickly from the data folder
        # to ensure we're self-contained.
        self.vector = self._load_or_extract_vector()
        
        # Latent stats for clipping (to prevent fog)
        self.latent_stats = self._compute_latent_stats()

    def _load_or_extract_vector(self):
        """Extracts vector from default data paths."""
        print("Estimating a fresh skin-tone direction for evaluation...")
        extractor = SkinToneDirectionExtractor(device=self.device)
        
        # Hardcoded paths as per project structure
        light_dir = PROJECT_ROOT / "data/photos/light_skin"
        dark_dir = PROJECT_ROOT / "data/photos/dark_skin"
        
        light_latents = self._encode_folder(light_dir)
        dark_latents = self._encode_folder(dark_dir)
        
        if not light_latents or not dark_latents:
            raise ValueError("Could not find photos to extract vector from!")
            
        # Create mask
        h, w = light_latents[0].shape[-2:]
        mask = extractor.create_center_mask(h, w, radius=0.8)
        
        vector = extractor.extract_from_pairs(
            light_latents, dark_latents, spatial_mask=mask
        )
        return vector

    def _encode_folder(self, folder_path, limit=10):
        latents = []
        files = sorted(list(folder_path.glob("*")))
        for f in files[:limit]:
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = Image.open(f).convert("RGB").resize((512, 512))
                latents.append(self.model.encode_image(img))
        return latents

    def _compute_latent_stats(self):
        """Computes stats from data folders for range clipping."""
        # We can reuse the same latents from vector extraction ideally
        # But let's just do a quick re-scan or use a subset
        light_dir = PROJECT_ROOT / "data/photos/light_skin"
        latents = self._encode_folder(light_dir, limit=5)
        
        stats = self.manipulator.compute_latent_statistics(latents)
        return stats

    def evaluate_vector_method(self, num_samples=10, alphas=None):
        """
        Evaluates the Vector Manipulation method.
        """
        if alphas is None:
            alphas = [-0.8, -0.4, 0.4, 0.8]
            
        print(f"\n--- Evaluating Vector Method (N={num_samples}) ---")
        
        results = []
        
        for i in tqdm(range(num_samples)):
            seed = 1000 + i
            prompt = "portrait photo of a person, professional headshot, neutral background"
            
            # 1. Generate Base
            base_img, base_latent = self.model.generate_from_prompt(
                prompt, seed=seed, num_inference_steps=40
            )
            
            for alpha in alphas:
                # 2. Apply Vector
                mod_latent = self.manipulator.apply_vector(base_latent, self.vector, alpha)
                mod_latent = self.manipulator.clip_to_valid_range(mod_latent, self.latent_stats)
                
                mod_img = self.model.decode_latent(mod_latent)
                
                # 3. Evaluate
                eval_res = self.evaluator.evaluate_pair(base_img, mod_img)
                row = eval_res.to_dict()
                row.update({
                    "method": "Vector",
                    "sample_id": i,
                    "alpha": alpha,
                    "target_race": "Dark" if alpha > 0 else "Light"
                })
                results.append(row)
                
                # Save sample images for first few
                if i < 3:
                    save_path = self.output_dir / f"vector_sample_{i}_alpha_{alpha}.png"
                    mod_img.save(save_path)

        return pd.DataFrame(results)

    def evaluate_prompt_baseline(self, num_samples=10):
        """
        Evaluates the Prompt Engineering Baseline.
        Generates pairs: (Base Prompt, Modified Prompt) with SAME SEED.
        """
        print(f"\n--- Evaluating Prompt Baseline (N={num_samples}) ---")
        
        results = []
        
        base_prompt_template = "portrait photo of a {} person, professional headshot, neutral background"
        
        # Pairs of (Original Race, Target Race) to simulate the vector directions
        # Vector -0.8 -> Light, +0.8 -> Dark
        # So baseline comparison is:
        #  - Base: "person" -> Target: "light skinned person"
        #  - Base: "person" -> Target: "dark skinned person"
        
        for i in tqdm(range(num_samples)):
            seed = 1000 + i
            
            # Base generation (Neutral "person")
            # We use this as the reference "original" just like in the vector method
            # Note: This is a bit tricky because "person" -> "light person" is a shift.
            # To be strictly comparable to vector method which takes a generated "person" and shifts it:
            prompt_neutral = "portrait photo of a person, professional headshot, neutral background"
            base_img, _ = self.model.generate_from_prompt(prompt_neutral, seed=seed, num_inference_steps=40)
            
            targets = [
                ("light skinned", -0.8), # Assigning equivalent alpha for grouping
                ("dark skinned", 0.8)
            ]
            
            for race_adj, equiv_alpha in targets:
                prompt_target = f"portrait photo of a {race_adj} person, professional headshot, neutral background"
                
                # Generate with SAME SEED
                target_img, _ = self.model.generate_from_prompt(prompt_target, seed=seed, num_inference_steps=40)
                
                # Evaluate pair
                eval_res = self.evaluator.evaluate_pair(base_img, target_img)
                row = eval_res.to_dict()
                row.update({
                    "method": "Prompt",
                    "sample_id": i,
                    "alpha": equiv_alpha,
                    "target_race": "Dark" if equiv_alpha > 0 else "Light"
                })
                results.append(row)
                
                if i < 3:
                   save_path = self.output_dir / f"prompt_sample_{i}_{race_adj.replace(' ', '_')}.png"
                   target_img.save(save_path)
                   
        return pd.DataFrame(results)
        
    def run_comparison(self, num_samples=10):
        df_vector = self.evaluate_vector_method(num_samples=num_samples)
        df_prompt = self.evaluate_prompt_baseline(num_samples=num_samples)
        
        full_df = pd.concat([df_vector, df_prompt], ignore_index=True)
        
        # Save results
        csv_path = self.output_dir / "comparative_metrics.csv"
        full_df.to_csv(csv_path, index=False)
        print(f"\nSaved full results to {csv_path}")
        
        # aggregated summary
        summary = full_df.groupby(["method", "target_race"])[
            ["face_similarity", "background_ssim", "is_disentangled", "overall_score"]
        ].mean()
        
        print("\n=== Comparative Summary ===")
        print(summary)
        
        summary_path = self.output_dir / "summary_table.csv"
        summary.to_csv(summary_path)
        
        return full_df

def main():
    parser = argparse.ArgumentParser(description="Run comparative evaluation")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per method")
    parser.add_argument("--output_dir", type=str, default="experiments/comparative_results")
    args = parser.parse_args()
    
    evaluator = ComparativeEvaluator(output_dir=args.output_dir)
    evaluator.run_comparison(num_samples=args.num_samples)

if __name__ == "__main__":
    main()

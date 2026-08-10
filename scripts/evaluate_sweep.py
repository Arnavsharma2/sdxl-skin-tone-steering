#!/usr/bin/env python3
"""Evaluate a saved alpha sweep and write an auditable metric bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from src.metrics.evaluator import CounterfactualEvaluator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_alpha(path: Path) -> float:
    try:
        return float(path.stem.removeprefix("alpha_"))
    except ValueError as exc:
        raise ValueError(f"Cannot parse alpha from {path.name}") from exc


def finite_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def evaluate(
    input_dir: Path,
    output_dir: Path,
    device: str,
    operating_max_alpha: float | None = None,
) -> dict:
    base_path = input_dir / "base_image.png"
    counterfactual_dir = input_dir / "counterfactuals"
    if not base_path.is_file():
        raise FileNotFoundError(f"Missing {base_path}")
    image_paths = sorted(counterfactual_dir.glob("alpha_*.png"), key=parse_alpha)
    if not image_paths:
        raise FileNotFoundError(f"No alpha_*.png files under {counterfactual_dir}")

    base = Image.open(base_path).convert("RGB")
    evaluator = CounterfactualEvaluator(device=device)
    rows = []
    input_hashes = {str(base_path): sha256(base_path)}
    sweep_measurements = []

    base_tone = evaluator.skin_tone_metrics.measure(base)
    if base_tone is not None:
        sweep_measurements.append((0.0, base_tone.relative_lstar))

    for path in image_paths:
        alpha = parse_alpha(path)
        image = Image.open(path).convert("RGB")
        input_hashes[str(path)] = sha256(path)
        tone = evaluator.skin_tone_metrics.measure(image)
        if tone is not None:
            sweep_measurements.append((alpha, tone.relative_lstar))
        if alpha == 0:
            continue
        row = evaluator.evaluate_pair(base, image, alpha=alpha).to_dict()
        row["image"] = str(path)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("alpha")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "pair_metrics.csv", index=False)

    unique_sweep = {}
    for alpha, relative_lstar in sweep_measurements:
        unique_sweep[alpha] = relative_lstar
    ordered = sorted(unique_sweep.items())
    alphas = np.asarray([item[0] for item in ordered], dtype=float)
    relative_lstar = np.asarray([item[1] for item in ordered], dtype=float)
    rho = None
    monotonic_fraction = None
    if len(ordered) >= 3:
        statistic = spearmanr(alphas, relative_lstar)
        rho = float(statistic.statistic)
        monotonic_fraction = float(np.mean(np.diff(relative_lstar) < 0))

    complete = frame["evaluation_complete"].astype(bool)
    target_available = frame["target_direction_correct"].notna()
    summary = {
        "schema_version": "2.0",
        "status": "pilot" if len(frame) < 30 else "requires_protocol_audit",
        "construct": "rendered visual skin tone; not race or ethnicity",
        "pair_count": int(len(frame)),
        "complete_count": int(complete.sum()),
        "evaluation_completion_rate": float(complete.mean()) if len(frame) else 0.0,
        "counterfactual_success_rate": float(frame["counterfactual_success"].mean()),
        "target_direction_accuracy": (
            float(frame.loc[target_available, "target_direction_correct"].mean())
            if target_available.any()
            else None
        ),
        "skin_tone_monotonic_spearman_rho": rho,
        "adjacent_monotonic_fraction": monotonic_fraction,
        "mean_face_similarity": finite_mean(frame, "face_similarity"),
        "mean_lpips": finite_mean(frame, "lpips"),
        "mean_background_ssim": finite_mean(frame, "background_ssim"),
        "mean_pose_difference_degrees": finite_mean(frame, "total_pose_diff"),
        "mean_quality_rubric": finite_mean(frame, "overall_score"),
        "metric_provenance": evaluator.metric_provenance(),
        "input_sha256": input_hashes,
        "limitations": [
            "A high engineering-gate score is not evidence of bias mitigation.",
            "Confirmatory claims require the prespecified held-out seeds and baselines.",
            "CIELAB skin colour remains camera, rendering, and illumination sensitive.",
        ],
    }
    if operating_max_alpha is not None:
        operating = frame[frame["alpha"].abs() <= operating_max_alpha]
        summary["exploratory_operating_band"] = {
            "max_abs_alpha": operating_max_alpha,
            "pair_count": int(len(operating)),
            "counterfactual_success_rate": float(
                operating["counterfactual_success"].mean()
            ),
            "mean_face_similarity": finite_mean(operating, "face_similarity"),
            "mean_lpips": finite_mean(operating, "lpips"),
            "mean_background_ssim": finite_mean(operating, "background_ssim"),
            "mean_pose_difference_degrees": finite_mean(
                operating, "total_pose_diff"
            ),
            "mean_quality_rubric": finite_mean(operating, "overall_score"),
            "claim_status": "post-hoc pilot operating band; requires held-out confirmation",
        }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Run directory with base_image.png and counterfactuals/")
    parser.add_argument("--output", type=Path, default=Path("experiments/evaluation"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--operating-max-alpha", type=float)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any required metric is missing",
    )
    args = parser.parse_args()
    summary = evaluate(
        args.input,
        args.output,
        args.device,
        operating_max_alpha=args.operating_max_alpha,
    )
    print(json.dumps(summary, indent=2))
    if args.strict and summary["evaluation_completion_rate"] < 1.0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

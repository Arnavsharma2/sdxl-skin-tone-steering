#!/usr/bin/env python3
"""Exploratory split-half stability of the paired latent direction estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.latent.vector_discovery import SkinToneDirectionExtractor
from src.study.config import load_study_config

if __package__:
    from scripts.run_study import StudyRunner
else:
    from run_study import StudyRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/analysis/direction_stability"),
    )
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    parser.add_argument("--pair-counts", type=int, nargs="+", default=(8, 16, 32))
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--reference-direction", type=Path)
    return parser.parse_args()


def summarize_records(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for pair_count, group in frame.groupby("pair_count"):
        for metric in ("raw_cosine", "masked_cosine", "norm_agreement"):
            values = group[metric].to_numpy(dtype=float)
            records.append(
                {
                    "pair_count": int(pair_count),
                    "metric": metric,
                    "n_resamples": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q025": float(np.quantile(values, 0.025)),
                    "q975": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame.from_records(records)


def _cosine(left, right) -> float:
    import torch.nn.functional as functional

    return float(
        functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()
    )


def _plot(summary: pd.DataFrame, output: Path) -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    styles = {
        "raw_cosine": ("Raw direction", "#4477AA", "o"),
        "masked_cosine": ("Masked direction", "#AA3377", "s"),
    }
    for metric, (label, color, marker) in styles.items():
        group = summary[summary["metric"] == metric].sort_values("pair_count")
        medians = group["median"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                medians - group["q025"].to_numpy(dtype=float),
                group["q975"].to_numpy(dtype=float) - medians,
            ]
        )
        ax.errorbar(
            group["pair_count"],
            medians,
            yerr=errors,
            label=label,
            color=color,
            marker=marker,
            capsize=3,
            linewidth=1.5,
        )
    ax.set_xticks(sorted(summary["pair_count"].unique()))
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("Pairs per disjoint half")
    ax.set_ylabel("Split-half cosine similarity")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"direction_stability.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.resamples < 20:
        raise SystemExit("--resamples must be at least 20")
    counts = sorted(set(int(value) for value in args.pair_counts))
    config = load_study_config(args.config)
    config.assert_confirmatory_ready()
    train_pairs = int(config.direction["train_pairs"])
    if not counts or any(value < 2 or 2 * value > train_pairs for value in counts):
        raise SystemExit(
            f"Pair counts must be >=2 and at most half of train_pairs={train_pairs}"
        )

    import torch

    runner = StudyRunner(
        config,
        args.output / "_runner_unused",
        device=args.device,
        allow_calibration=False,
    )
    runner.validate_execution()
    pairs = runner._load_pair_manifest()
    runner._load_model()
    differences = []
    image_size = int(config.data["image_size"])
    for index, pair in enumerate(pairs, start=1):
        print(f"Encoding stability pair {index}/{len(pairs)}: {pair.pair_id}")
        light = Image.open(pair.light_path).convert("RGB")
        dark = Image.open(pair.dark_path).convert("RGB")
        light_latent = runner.model.encode_image(light, size=(image_size, image_size))
        dark_latent = runner.model.encode_image(dark, size=(image_size, image_size))
        differences.append((dark_latent - light_latent).float())
    pair_differences = torch.stack(differences)
    full_direction = pair_differences.mean(dim=0)
    height, width = full_direction.shape[-2:]
    mask_spec = config.direction["spatial_mask"]
    extractor = SkinToneDirectionExtractor(device=runner._select_device())
    mask = extractor.create_center_mask(
        height,
        width,
        center_weight=float(mask_spec["center_weight"]),
        edge_weight=float(mask_spec["edge_weight"]),
        radius=float(mask_spec["radius"]),
    )
    masked_full = full_direction * mask.view(
        *((1,) * (full_direction.ndim - 2)), height, width
    )

    rng = np.random.default_rng(args.seed)
    records = []
    for pair_count in counts:
        for resample in range(args.resamples):
            permutation = rng.permutation(train_pairs)
            left_indices = torch.as_tensor(
                permutation[:pair_count], device=pair_differences.device
            )
            right_indices = torch.as_tensor(
                permutation[pair_count : 2 * pair_count],
                device=pair_differences.device,
            )
            left = pair_differences.index_select(0, left_indices).mean(dim=0)
            right = pair_differences.index_select(0, right_indices).mean(dim=0)
            masked_left = left * mask.view(
                *((1,) * (left.ndim - 2)), height, width
            )
            masked_right = right * mask.view(
                *((1,) * (right.ndim - 2)), height, width
            )
            left_norm = float(torch.linalg.vector_norm(left).item())
            right_norm = float(torch.linalg.vector_norm(right).item())
            records.append(
                {
                    "pair_count": pair_count,
                    "resample": resample,
                    "raw_cosine": _cosine(left, right),
                    "masked_cosine": _cosine(masked_left, masked_right),
                    "norm_agreement": min(left_norm, right_norm)
                    / max(left_norm, right_norm),
                    "left_to_full_cosine": _cosine(left, full_direction),
                    "right_to_full_cosine": _cosine(right, full_direction),
                }
            )
    frame = pd.DataFrame.from_records(records)
    summary = summarize_records(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "direction_stability_resamples.csv", index=False)
    summary.to_csv(args.output / "direction_stability_summary.csv", index=False)
    torch.save(full_direction.detach().cpu(), args.output / "full_direction.pt")
    reference = {}
    if args.reference_direction is not None:
        recorded = torch.load(
            args.reference_direction,
            map_location=full_direction.device,
            weights_only=True,
        ).float()
        reference = {
            "path": str(args.reference_direction),
            "cosine_to_recorded": _cosine(full_direction, recorded),
            "max_abs_difference": float((full_direction - recorded).abs().max().item()),
        }
    audit = {
        "schema_version": "1.0",
        "analysis_role": "post_confirmatory_exploratory_robustness",
        "config_fingerprint": config.fingerprint,
        "train_pairs": train_pairs,
        "pair_counts": counts,
        "resamples": args.resamples,
        "sampling": "disjoint_random_halves_without_replacement",
        "full_direction_norm": float(torch.linalg.vector_norm(full_direction).item()),
        "masked_full_direction_norm": float(
            torch.linalg.vector_norm(masked_full).item()
        ),
        "reference_direction_check": reference,
        "note": "Representation stability does not establish causal disentanglement.",
    }
    with (args.output / "direction_stability_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    _plot(summary, args.output)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate the frozen independent-seed LPIPS replication family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.study.analysis import holm_adjust
from src.study.config import load_study_config

DIRECTION_LABELS = {-1: "lighter", 1: "darker"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replication_family(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Select and multiplicity-adjust the two locked LPIPS contrasts."""

    selected = contrasts[
        (contrasts["reference_method"] == "stepwise_masked")
        & (contrasts["comparator"] == "stepwise_unmasked")
        & (contrasts["outcome"] == "lpips")
        & (contrasts["target_change"] == 5.0)
        & (contrasts["direction"].isin(DIRECTION_LABELS))
    ].copy()
    selected = selected.sort_values("direction").reset_index(drop=True)
    if len(selected) != 2 or set(selected["direction"].astype(int)) != {-1, 1}:
        raise ValueError(
            "Replication analysis requires exactly the lighter and darker "
            "masked-versus-unmasked LPIPS rows"
        )
    if selected["p_value"].isna().any():
        raise ValueError("Replication-family p-values are missing")
    selected["direction_label"] = selected["direction"].map(DIRECTION_LABELS)
    selected["p_holm_replication_family"] = holm_adjust(
        selected["p_value"].astype(float).tolist()
    )
    return selected


def assess_replication(family: pd.DataFrame, minimum_pairs: int = 12) -> str:
    """Apply the prospective replication decision rule without reinterpretation."""

    if (family["n_pairs"].astype(int) < minimum_pairs).any():
        return "inconclusive_coverage"
    estimates = family["mean_reference_advantage"].astype(float)
    if (estimates <= 0).any():
        return "failure_to_replicate"
    strict = (
        (family["ci_low"].astype(float) > 0).all()
        and (family["p_holm_replication_family"].astype(float) < 0.05).all()
    )
    return "strict_replication" if strict else "directional_replication"


def direction_agreement(
    parent_path: Path,
    replication_path: Path,
    config_path: Path,
) -> dict[str, float]:
    """Compare raw and frozen-mask direction tensors across campaigns."""

    import torch
    import torch.nn.functional as functional

    from src.latent.vector_discovery import SkinToneDirectionExtractor

    parent = torch.load(parent_path, map_location="cpu", weights_only=True).float()
    replication = torch.load(
        replication_path, map_location="cpu", weights_only=True
    ).float()
    if parent.shape != replication.shape:
        raise ValueError(
            f"Direction shapes differ: parent={tuple(parent.shape)}, "
            f"replication={tuple(replication.shape)}"
        )
    config = load_study_config(config_path)
    mask_spec = config.direction["spatial_mask"]
    extractor = SkinToneDirectionExtractor(device="cpu")
    mask = extractor.create_center_mask(
        parent.shape[-2],
        parent.shape[-1],
        center_weight=float(mask_spec["center_weight"]),
        edge_weight=float(mask_spec["edge_weight"]),
        radius=float(mask_spec["radius"]),
    )
    while mask.ndim < parent.ndim:
        mask = mask.unsqueeze(0)

    def cosine(left, right) -> float:
        return float(
            functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()
        )

    return {
        "raw_cosine": cosine(parent, replication),
        "masked_cosine": cosine(parent * mask, replication * mask),
        "parent_norm": float(torch.linalg.vector_norm(parent).item()),
        "replication_norm": float(torch.linalg.vector_norm(replication).item()),
        "norm_ratio_replication_to_parent": float(
            torch.linalg.vector_norm(replication).item()
            / torch.linalg.vector_norm(parent).item()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replication_config", type=Path)
    parser.add_argument("replication_analysis", type=Path)
    parser.add_argument("--parent-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=12)
    parser.add_argument("--parent-direction", type=Path)
    parser.add_argument("--replication-direction", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.replication_config)
    config.assert_confirmatory_ready()
    replication_path = args.replication_analysis / "paired_contrasts.csv"
    parent_path = args.parent_analysis / "paired_contrasts.csv"
    replication = replication_family(pd.read_csv(replication_path))
    parent = replication_family(pd.read_csv(parent_path))
    parent_columns = [
        "direction",
        "n_pairs",
        "mean_reference_advantage",
        "ci_low",
        "ci_high",
    ]
    parent = parent[parent_columns].rename(
        columns={column: f"parent_{column}" for column in parent_columns if column != "direction"}
    )
    result = replication.merge(parent, on="direction", validate="one_to_one")
    result["same_sign_as_parent"] = (
        result["mean_reference_advantage"]
        * result["parent_mean_reference_advantage"]
        > 0
    )
    status = assess_replication(result, minimum_pairs=args.minimum_pairs)

    agreement = None
    if (args.parent_direction is None) != (args.replication_direction is None):
        raise SystemExit(
            "--parent-direction and --replication-direction must be provided together"
        )
    if args.parent_direction is not None:
        agreement = direction_agreement(
            args.parent_direction,
            args.replication_direction,
            args.replication_config,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "replication_family.csv", index=False)
    audit = {
        "schema_version": "1.0",
        "analysis_role": "prospective_independent_seed_replication",
        "study_id": config.study_id,
        "config_fingerprint": config.fingerprint,
        "decision_family": "stepwise_masked_vs_stepwise_unmasked_lpips_by_direction",
        "minimum_shared_matched_pairs_per_direction": args.minimum_pairs,
        "decision": status,
        "strict_rule": (
            "both estimates positive, both 95% bootstrap CIs exclude zero, "
            "both two-test Holm p-values below 0.05"
        ),
        "replication_paired_contrasts_sha256": sha256_file(replication_path),
        "parent_paired_contrasts_sha256": sha256_file(parent_path),
        "direction_agreement": agreement,
        "note": (
            "Independent replication rows are reported separately; no pooled "
            "60-seed preregistration claim is made."
        ),
    }
    with (args.output / "replication_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    if any(not math.isfinite(float(value)) for value in result["p_value"]):
        raise SystemExit("Replication output contains a non-finite p-value")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

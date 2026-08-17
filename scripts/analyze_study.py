#!/usr/bin/env python3
"""Create auditable descriptive and matched-change study tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.study.analysis import (
    descriptive_summary,
    load_jsonl_results,
    match_at_target_change,
    matched_coverage_summary,
    missingness_summary,
    monotonicity_summary,
    paired_method_contrasts,
)
from src.study.config import load_study_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="One or more result files/directories from the same study fingerprint",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/analysis"))
    parser.add_argument(
        "--targets",
        type=float,
        nargs="*",
        help="Override matched-change targets for calibration analysis only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    frames = [load_jsonl_results(path) for path in args.input]
    nonempty = [item for item in frames if not item.empty]
    frame = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    if frame.empty:
        raise SystemExit(f"No results.jsonl rows found under {args.input}")
    fingerprints = set(frame.get("config_fingerprint", ()))
    if fingerprints != {config.fingerprint}:
        raise SystemExit(
            "Input result fingerprints do not exactly match the analysis config: "
            f"observed={sorted(fingerprints)}, expected={config.fingerprint}"
        )
    condition_columns = ["seed", "method", "alpha"]
    expected_condition_count = len(config.seeds) * sum(
        len(config.alphas_for(method)) for method in config.methods
    )
    duplicates = frame.duplicated(condition_columns, keep=False)
    if duplicates.any():
        preview = frame.loc[duplicates, condition_columns].head(5).to_dict("records")
        raise SystemExit(f"Duplicate condition rows found across inputs: {preview}")
    if config.status == "preregistered":
        expected_keys = {
            (seed, method, alpha)
            for seed in config.seeds
            for method in config.methods
            for alpha in config.alphas_for(method)
        }
        observed_keys = {
            (int(row.seed), str(row.method), float(row.alpha))
            for row in frame[condition_columns].itertuples(index=False)
        }
        missing_keys = sorted(expected_keys - observed_keys)
        extra_keys = sorted(observed_keys - expected_keys)
        if missing_keys or extra_keys:
            raise SystemExit(
                "Preregistered analysis requires the exact frozen condition set: "
                f"missing={len(missing_keys)} {missing_keys[:5]}, "
                f"extra={len(extra_keys)} {extra_keys[:5]}"
            )
    bootstrap = config.evaluation["bootstrap"]
    resamples = int(bootstrap["resamples"])
    confidence = float(bootstrap["confidence_level"])
    targets = (
        args.targets
        if args.targets is not None
        else config.analysis.get("matched_change_targets", [])
    )

    args.output.mkdir(parents=True, exist_ok=True)
    descriptive_summary(
        frame, resamples=resamples, confidence=confidence
    ).to_csv(args.output / "descriptive.csv", index=False)
    missingness_summary(frame).to_csv(args.output / "missingness.csv", index=False)
    monotonicity_summary(frame).to_csv(args.output / "monotonicity.csv", index=False)

    matched_rows = 0
    contrast_rows = 0
    if targets:
        matched_input = frame[
            frame["method"].isin(config.matched_change_methods)
        ].copy()
        matched = match_at_target_change(
            matched_input,
            targets,
            tolerance=float(config.analysis["match_tolerance_ita"]),
        )
        contrasts = paired_method_contrasts(
            matched,
            reference_method=str(config.analysis["reference_method"]),
            resamples=resamples,
            confidence=confidence,
        )
        matched.to_csv(args.output / "matched_conditions.csv", index=False)
        matched_coverage_summary(matched).to_csv(
            args.output / "matched_summary.csv", index=False
        )
        contrasts.to_csv(args.output / "paired_contrasts.csv", index=False)
        matched_rows = len(matched)
        contrast_rows = len(contrasts)

    audit = {
        "study_id": config.study_id,
        "config_fingerprint": config.fingerprint,
        "input_rows": len(frame),
        "unique_seeds": int(frame["seed"].nunique()),
        "expected_conditions": expected_condition_count,
        "matched_change_targets": list(targets),
        "matched_change_methods": list(config.matched_change_methods),
        "feasibility_only_methods": list(
            config.analysis.get("feasibility_only_methods", [])
        ),
        "matched_rows": matched_rows,
        "contrast_rows": contrast_rows,
    }
    with (args.output / "analysis_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

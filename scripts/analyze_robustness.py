#!/usr/bin/env python3
"""Post-confirmatory sensitivity of matched results to the ITA tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.study.analysis import (
    load_jsonl_results,
    match_at_target_change,
    matched_coverage_summary,
    paired_method_contrasts,
)
from src.study.config import load_study_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 3.0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/analysis/robustness"),
    )
    return parser.parse_args()


def validate_exact_conditions(frame: pd.DataFrame, config) -> None:
    condition_columns = ["seed", "method", "alpha"]
    if frame.duplicated(condition_columns).any():
        raise ValueError("Result table contains duplicate condition keys")
    expected = {
        (seed, method, alpha)
        for seed in config.seeds
        for method in config.methods
        for alpha in config.alphas_for(method)
    }
    observed = {
        (int(row.seed), str(row.method), float(row.alpha))
        for row in frame[condition_columns].itertuples(index=False)
    }
    if expected != observed:
        raise ValueError(
            f"Condition set mismatch: missing={len(expected - observed)}, "
            f"extra={len(observed - expected)}"
        )
    fingerprints = set(frame["config_fingerprint"])
    if fingerprints != {config.fingerprint}:
        raise ValueError(
            f"Fingerprint mismatch: observed={sorted(fingerprints)}, "
            f"expected={config.fingerprint}"
        )


def analyze_tolerances(
    frame: pd.DataFrame,
    config,
    tolerances: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap = config.evaluation["bootstrap"]
    matched_input = frame[frame["method"].isin(config.matched_change_methods)].copy()
    coverage_frames = []
    contrast_frames = []
    for tolerance in tolerances:
        matched = match_at_target_change(
            matched_input,
            config.analysis["matched_change_targets"],
            tolerance=tolerance,
        )
        coverage = matched_coverage_summary(matched)
        coverage.insert(0, "tolerance_ita", tolerance)
        coverage_frames.append(coverage)
        contrasts = paired_method_contrasts(
            matched,
            reference_method=str(config.analysis["reference_method"]),
            resamples=int(bootstrap["resamples"]),
            confidence=float(bootstrap["confidence_level"]),
        )
        contrasts.insert(0, "tolerance_ita", tolerance)
        contrast_frames.append(contrasts)
    return (
        pd.concat(coverage_frames, ignore_index=True),
        pd.concat(contrast_frames, ignore_index=True),
    )


def main() -> None:
    args = parse_args()
    tolerances = sorted(set(float(value) for value in args.tolerances))
    if not tolerances or any(value <= 0 for value in tolerances):
        raise SystemExit("Every tolerance must be positive")
    config = load_study_config(args.config)
    config.assert_confirmatory_ready()
    frame = load_jsonl_results(args.run)
    if frame.empty:
        raise SystemExit(f"No results found below {args.run}")
    validate_exact_conditions(frame, config)
    coverage, contrasts = analyze_tolerances(frame, config, tolerances)
    args.output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output / "tolerance_coverage.csv", index=False)
    contrasts.to_csv(args.output / "tolerance_contrasts.csv", index=False)
    audit = {
        "schema_version": "1.0",
        "analysis_role": "post_confirmatory_exploratory_robustness",
        "config_fingerprint": config.fingerprint,
        "input_rows": len(frame),
        "tolerances_ita": tolerances,
        "coverage_rows": len(coverage),
        "contrast_rows": len(contrasts),
        "note": "This analysis does not replace the preregistered 3-ITA tolerance.",
    }
    with (args.output / "robustness_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

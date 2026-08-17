#!/usr/bin/env python3
"""Synthesize adaptive calibration runs into a provenance-preserving audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.study.analysis import match_at_target_change, monotonicity_summary
from src.study.calibration import deduplicate_calibration_rows, matched_coverage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--target", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=3.0)
    return parser.parse_args()


def load_inputs(paths: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    sources = []
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Calibration input does not exist: {path}")
        frame = pd.read_json(path, lines=True)
        frame["source_path"] = str(path)
        frames.append(frame)
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(frame),
                "config_fingerprints": sorted(
                    str(value) for value in frame["config_fingerprint"].dropna().unique()
                ),
            }
        )
    return pd.concat(frames, ignore_index=True), sources


def main() -> None:
    args = parse_args()
    expected_seeds = tuple(int(seed) for seed in args.expected_seeds)
    if len(expected_seeds) != len(set(expected_seeds)):
        raise SystemExit("--expected-seeds must be unique")
    if args.target <= 0 or args.tolerance <= 0:
        raise SystemExit("--target and --tolerance must be positive")

    raw, sources = load_inputs(args.input)
    observed_seeds = {int(seed) for seed in raw["seed"].unique()}
    if observed_seeds != set(expected_seeds):
        raise SystemExit(
            "Observed calibration seeds do not match --expected-seeds: "
            f"observed={sorted(observed_seeds)}, expected={sorted(expected_seeds)}"
        )
    rows, repeats = deduplicate_calibration_rows(raw)
    complete_mask = (
        rows["generation_complete"].fillna(False).astype(bool)
        & rows["evaluation_complete"].fillna(False).astype(bool)
    )
    incomplete = rows.loc[~complete_mask].copy()
    eligible = rows.loc[complete_mask].copy()

    # Failed exploratory endpoints are evidence about unsafe or unusable grid
    # regions. Preserve them in the audit, but never allow a row missing a
    # required metric to satisfy the target-matching gate.
    matched = match_at_target_change(eligible, [args.target], tolerance=args.tolerance)
    coverage = matched_coverage(matched, expected_seeds=expected_seeds)
    expected_strata = set(rows["method"].unique())
    expected_strata = {
        (method, direction) for method in expected_strata for direction in (-1, 1)
    }
    observed_strata = set(zip(coverage["method"], coverage["direction"]))
    full_coverage = (
        expected_strata == observed_strata
        and not coverage.empty
        and bool((coverage["matched_seeds"] == len(expected_seeds)).all())
    )

    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "deduplicated_conditions.csv", index=False)
    repeats.to_csv(args.output / "repeated_conditions.csv", index=False)
    incomplete.to_csv(args.output / "incomplete_conditions.csv", index=False)
    matched.to_csv(args.output / "matched_conditions.csv", index=False)
    coverage.to_csv(args.output / "coverage.csv", index=False)
    monotonicity_summary(eligible).to_csv(args.output / "monotonicity.csv", index=False)
    audit = {
        "purpose": "adaptive_calibration_only_not_confirmatory_evidence",
        "source_files": sources,
        "input_rows": len(raw),
        "deduplicated_rows": len(rows),
        "eligible_complete_rows": len(eligible),
        "incomplete_condition_rows": len(incomplete),
        "repeated_condition_keys": len(repeats),
        "expected_seeds": list(expected_seeds),
        "target_change": float(args.target),
        "match_tolerance_ita": float(args.tolerance),
        "expected_method_direction_strata": len(expected_strata),
        "full_target_coverage": full_coverage,
    }
    with (args.output / "calibration_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(audit, indent=2))
    if not full_coverage:
        raise SystemExit("Calibration target coverage is incomplete; do not freeze the grid")


if __name__ == "__main__":
    main()

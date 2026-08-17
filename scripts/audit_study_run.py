#!/usr/bin/env python3
"""Audit a completed frozen study run before statistical analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.study.config import load_study_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_run(config_path: Path, run_dir: Path) -> dict:
    """Return a strict, machine-readable integrity audit for one run directory."""

    config = load_study_config(config_path)
    config.assert_confirmatory_ready()
    results_path = run_dir / "results.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    archived_config_path = run_dir / "study_config.yaml"
    if not all(path.is_file() for path in (results_path, manifest_path, archived_config_path)):
        raise ValueError(
            "Run must contain results.jsonl, run_manifest.json, and study_config.yaml"
        )

    rows = []
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed result row {line_number}: {exc}") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archived_config = load_study_config(archived_config_path)

    expected_keys = {
        (int(seed), str(method), float(alpha))
        for seed in config.seeds
        for method in config.methods
        for alpha in config.alphas_for(method)
    }
    observed_keys = [
        (int(row["seed"]), str(row["method"]), float(row["alpha"])) for row in rows
    ]
    counts = Counter(observed_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    observed_set = set(observed_keys)
    missing = sorted(expected_keys - observed_set)
    extra = sorted(observed_set - expected_keys)

    fingerprints = sorted({str(row.get("config_fingerprint")) for row in rows})
    study_ids = sorted({str(row.get("study_id")) for row in rows})
    generation_failures = [
        key
        for key, row in zip(observed_keys, rows, strict=True)
        if row.get("generation_complete") is not True
    ]
    evaluation_incomplete = [
        {
            "key": key,
            "skin_tone_failure": row.get("skin_tone_failure"),
            "missing_required_metrics": row.get("missing_required_metrics", []),
        }
        for key, row in zip(observed_keys, rows, strict=True)
        if row.get("evaluation_complete") is not True
    ]
    missing_images = sorted(
        {
            str(row.get("image_path"))
            for row in rows
            if not isinstance(row.get("image_path"), str)
            or not (run_dir / str(row["image_path"])).is_file()
        }
    )
    direction_paths = [run_dir / "direction" / name for name in ("raw.pt", "masked.pt")]
    direction_hashes = {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in direction_paths
        if path.is_file()
    }

    checks = {
        "exact_condition_keys": not missing and not extra and not duplicates,
        "row_count": len(rows) == len(expected_keys),
        "result_fingerprint": fingerprints == [config.fingerprint],
        "result_study_id": study_ids == [config.study_id],
        "manifest_fingerprint": manifest.get("config_fingerprint") == config.fingerprint,
        "manifest_study_id": manifest.get("study_id") == config.study_id,
        "manifest_study_status": manifest.get("study_status") == config.status,
        "manifest_execution_mode": manifest.get("execution_mode") == "confirmatory",
        "manifest_condition_count": manifest.get("condition_count") == len(expected_keys),
        "manifest_seed_set": set(manifest.get("execution_seeds", []))
        == set(config.seeds),
        "all_generations_complete": not generation_failures,
        "all_referenced_images_present": not missing_images,
        "direction_tensors_present": len(direction_hashes) == 2,
        "archived_config_fingerprint": archived_config.fingerprint
        == config.fingerprint,
    }
    return {
        "schema_version": "1.0",
        "study_id": config.study_id,
        "config_fingerprint": config.fingerprint,
        "run_dir": str(run_dir),
        "result_ledger_sha256": sha256_file(results_path),
        "archived_config_sha256": sha256_file(archived_config_path),
        "expected_conditions": len(expected_keys),
        "observed_rows": len(rows),
        "unique_condition_keys": len(observed_set),
        "unique_seeds": len({key[0] for key in observed_set}),
        "generation_failure_count": len(generation_failures),
        "evaluation_incomplete_count": len(evaluation_incomplete),
        "evaluation_incomplete": evaluation_incomplete,
        "missing_keys": missing,
        "extra_keys": extra,
        "duplicate_keys": duplicates,
        "missing_images": missing_images,
        "direction_sha256": direction_hashes,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_run(args.config, args.run_dir)
    payload = json.dumps(audit, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

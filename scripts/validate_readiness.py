#!/usr/bin/env python3
"""Build the Step 6 pre-collection readiness report without generating images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.validation.readiness import build_readiness_report


def _artifact_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise argparse.ArgumentTypeError("--artifact must use NAME=PATH")
        if name in overrides:
            raise argparse.ArgumentTypeError(f"duplicate --artifact name: {name}")
        overrides[name] = Path(path)
    return overrides


def _artifact_manifest(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"cannot read --artifact-manifest: {exc}") from exc
    if document.get("registry_id") != "tmlr_metric_artifact_sources_v1":
        raise argparse.ArgumentTypeError("unexpected artifact-manifest registry_id")
    values = document.get("artifact_overrides")
    if not isinstance(values, dict) or not values:
        raise argparse.ArgumentTypeError("artifact manifest has no artifact_overrides")
    if not all(isinstance(name, str) and isinstance(value, str) for name, value in values.items()):
        raise argparse.ArgumentTypeError("artifact manifest paths must be strings")
    return {name: Path(value) for name, value in values.items()}


def write_json(path: Path, payload: dict) -> None:
    """Atomically write a sorted machine-readable report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/full_study.yaml"))
    parser.add_argument(
        "--evaluation-protocol",
        type=Path,
        default=Path("configs/evaluation_protocol.yaml"),
    )
    parser.add_argument(
        "--readiness-protocol",
        type=Path,
        default=Path("configs/step6_readiness.yaml"),
    )
    parser.add_argument(
        "--storage-policy",
        type=Path,
        default=Path("configs/collection_policy.yaml"),
    )
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=Path("configs/validation_sources.yaml"),
    )
    parser.add_argument(
        "--metric-artifact-registry",
        type=Path,
        default=Path("configs/metric_artifacts.yaml"),
    )
    parser.add_argument("--external-validation", type=Path)
    parser.add_argument("--generation-provenance", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override one frozen metric-artifact location; repeat as needed",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        help="Use checksum-reverified paths written by download_metric_models.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/readiness/step6_readiness.json"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 after writing the report when collection is not ready",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        overrides = _artifact_overrides(args.artifact)
        manifest_overrides = _artifact_manifest(args.artifact_manifest)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    duplicates = sorted(set(overrides) & set(manifest_overrides))
    if duplicates:
        raise SystemExit(f"artifact override duplicated by manifest: {', '.join(duplicates)}")
    overrides = {**manifest_overrides, **overrides}
    report = build_readiness_report(
        project_root=args.project_root,
        study_config_path=args.config,
        evaluation_protocol_path=args.evaluation_protocol,
        readiness_protocol_path=args.readiness_protocol,
        storage_policy_path=args.storage_policy,
        source_registry_path=args.source_registry,
        metric_artifact_registry_path=args.metric_artifact_registry,
        external_validation_path=args.external_validation,
        generation_provenance_path=args.generation_provenance,
        approval_path=args.approval,
        artifact_overrides=overrides,
    )
    write_json(args.output, report)
    decision = report["decision"]
    print(
        f"Step 6 readiness: collection_ready={decision['collection_ready']} "
        f"passed={decision['passed_criteria_count']}/"
        f"{decision['required_criteria_count']} report={args.output}"
    )
    print("No SDXL model was loaded and no confirmatory image was generated.")
    if args.strict and not decision["collection_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

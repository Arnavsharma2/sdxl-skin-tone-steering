#!/usr/bin/env python3
"""Prepare a checksum-bound pending approval record without granting approval."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml

from src.metrics.artifacts import inspect_artifact, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a mapping in {path}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-validation", type=Path, required=True)
    parser.add_argument("--generation-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/full_study.yaml")
    parser.add_argument(
        "--evaluation-protocol", type=Path, default=ROOT / "configs/evaluation_protocol.yaml"
    )
    parser.add_argument(
        "--readiness-protocol", type=Path, default=ROOT / "configs/step6_readiness.yaml"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    external_path = args.external_validation.resolve()
    provenance_path = args.generation_provenance.resolve()
    external = _load(external_path)
    provenance = _load(provenance_path)
    validation_id = external.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise SystemExit("External-validation record has no validation_id")
    model = provenance.get("model", {})
    revision = model.get("resolved_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("Generation provenance has no immutable resolved model revision")
    if model.get("requested_revision") != revision:
        raise SystemExit("Requested and resolved model revisions differ")
    direction = provenance.get("direction", {}).get("artifact", {})
    direction_path = (provenance_path.parent / str(direction.get("path", ""))).resolve()
    direction_result = inspect_artifact(
        direction_path, str(direction.get("sha256", "")), name="direction_artifact"
    )
    if not direction_result.verified:
        raise SystemExit(
            f"Direction artifact is not checksum-verifiable: {direction_result.status}"
        )
    payload = {
        "schema_version": "1.0",
        "approval_id": None,
        "approved_by": None,
        "approved_at_utc": None,
        "study_id": "skin_tone_steering_confirmatory_v1",
        "scope": "confirmatory_collection",
        "decision": "pending_authorized_approver",
        "study_config_sha256": sha256_file(args.config.resolve())[0],
        "evaluation_protocol_sha256": sha256_file(args.evaluation_protocol.resolve())[0],
        "readiness_protocol_sha256": sha256_file(args.readiness_protocol.resolve())[0],
        "external_validation_id": validation_id,
        "model_revision": revision,
        "direction_sha256": direction_result.actual_sha256,
        "source_records": {
            "external_validation_sha256": sha256_file(external_path)[0],
            "generation_provenance_sha256": sha256_file(provenance_path)[0],
        },
        "notes": {
            "template_is_approval": False,
            "prepared_by_repository_tool_is_approval": False,
            "authorized_approver_must_supply_decision_and_identity": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Prepared pending checksum-bound approval record: {args.output}")
    print("No collection approval was issued; an authorized approver must complete the record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

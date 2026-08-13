#!/usr/bin/env python3
"""Download every frozen metric artifact from its registered upstream source.

Downloads are for local evaluation only, are checksum-verified before atomic
installation, and are written beneath an ignored artifact root. This command
does not grant redistribution rights for any weight.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import yaml

from src.metrics.artifacts import inspect_artifact
from src.metrics.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "metric_artifacts.yaml"


def _registry(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("registry_id") != (
        "tmlr_metric_artifact_sources_v1"
    ):
        raise SystemExit("Unexpected metric-artifact registry")
    frozen = load_protocol(ROOT / "configs" / "evaluation_protocol.yaml")["required_artifacts"]
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict) or tuple(artifacts) != tuple(frozen):
        raise SystemExit("Metric-artifact registry set or order differs from frozen protocol")
    for name, item in artifacts.items():
        if (
            item.get("sha256") != frozen[name]["sha256"]
            or item.get("protocol_location") != frozen[name]["location"]
        ):
            raise SystemExit(f"Metric-artifact registry drift for {name}")
        if not str(item.get("weight_redistribution_status", "")).startswith("unresolved_"):
            raise SystemExit(f"Metric-artifact redistribution gap was not retained for {name}")
    return document


def _download(url: str, destination: Path, expected_sha256: str, name: str) -> dict:
    existing = inspect_artifact(destination, expected_sha256, name=name)
    if existing.verified:
        return existing.to_dict()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(url) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    result = inspect_artifact(temporary_path, expected_sha256, name=name)
    if not result.verified:
        temporary_path.unlink(missing_ok=True)
        raise SystemExit(
            f"{name} download failed verification: {result.status}; "
            f"expected {expected_sha256}, actual {result.actual_sha256}"
        )
    os.replace(temporary_path, destination)
    return inspect_artifact(destination, expected_sha256, name=name).to_dict()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=Path(".artifacts/metrics"))
    parser.add_argument("--manifest", type=Path, default=Path(".artifacts/metric_artifacts.json"))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry_path = args.registry.resolve()
    registry = _registry(registry_path)
    output_root = args.output_root.resolve()
    verifications = []
    overrides = {}
    for name, item in registry["artifacts"].items():
        destination = output_root / name / item["filename"]
        if args.verify_only:
            verification = inspect_artifact(destination, item["sha256"], name=name).to_dict()
        else:
            verification = _download(item["source_url"], destination, item["sha256"], name)
        verification.update(
            {
                "source_url": item["source_url"],
                "source_revision": item["source_revision"],
                "weight_redistribution_status": item["weight_redistribution_status"],
            }
        )
        verifications.append(verification)
        overrides[name] = str(destination)
    failed = [item for item in verifications if not item["verified"]]
    payload = {
        "schema_version": "1.0",
        "registry_id": registry["registry_id"],
        "registry_path": str(registry_path),
        "artifact_overrides": overrides,
        "verifications": verifications,
        "redistribution_authorized": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.manifest)
    print(
        f"Metric artifacts: verified={len(verifications) - len(failed)}/"
        f"{len(verifications)} manifest={args.manifest}"
    )
    print("Downloaded weights remain local; redistribution is not authorized by this command.")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

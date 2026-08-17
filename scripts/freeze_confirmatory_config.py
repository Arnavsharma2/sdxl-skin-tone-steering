#!/usr/bin/env python3
"""Freeze content hashes into a new confirmatory study configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from src.study.config import StudyConfigError, load_study_config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)
    if config.status != "planned":
        raise StudyConfigError(
            f"Freeze source must have status: planned, observed {config.status!r}"
        )
    if args.output.exists():
        raise StudyConfigError(f"Refusing to overwrite frozen config: {args.output}")

    project_root = config.path.parent.parent
    manifest_path = args.manifest.resolve()
    report_path = args.validation_report.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    required_pairs = int(config.direction["train_pairs"]) + int(
        config.direction["held_out_pairs"]
    )
    if manifest.get("n_pairs") != required_pairs:
        raise StudyConfigError(
            f"Manifest has {manifest.get('n_pairs')} pairs; expected {required_pairs}"
        )
    if (
        manifest.get("generation_observed_in_campaign") is not True
        or manifest.get("generation_observed_in_this_run") is not True
    ):
        raise StudyConfigError("Manifest does not attest a complete generation campaign")
    for key, expected in {
        "model_id": str(config.model["id"]),
        "model_revision": str(config.model["revision"]),
        "inference_steps": int(config.model["inference_steps"]),
        "guidance_scale": float(config.model["guidance_scale"]),
        "height": int(config.model["height"]),
        "width": int(config.model["width"]),
    }.items():
        if manifest.get(key) != expected:
            raise StudyConfigError(
                f"Manifest {key}={manifest.get(key)!r} does not match {expected!r}"
            )
    ledger_path = resolve(project_root, manifest.get("generation_ledger", ""))
    if not ledger_path.is_file():
        raise StudyConfigError(f"Generation ledger is missing: {ledger_path}")
    if sha256_file(ledger_path) != manifest.get("generation_ledger_sha256"):
        raise StudyConfigError("Generation ledger hash does not match the manifest")

    if report.get("passed") is not True:
        raise StudyConfigError("Measurement validation report did not pass")
    if report.get("study_id") != config.study_id:
        raise StudyConfigError("Measurement report study_id does not match the config")
    expected_metric = config.raw["measurement_validation"]["metric"]
    if report.get("metric") != expected_metric:
        raise StudyConfigError("Measurement report metric does not match the config")
    if report.get("split") != "held_out_direction_pairs":
        raise StudyConfigError("Measurement report uses the wrong validation split")

    raw = dict(config.raw)
    raw["status"] = "preregistered"
    raw["data"] = dict(raw["data"])
    raw["measurement_validation"] = dict(raw["measurement_validation"])
    raw["data"]["training_manifest_sha256"] = sha256_file(manifest_path)
    raw["measurement_validation"]["report_sha256"] = sha256_file(report_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    frozen = load_study_config(args.output)
    frozen.assert_confirmatory_ready()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "study_id": frozen.study_id,
                "config_fingerprint": frozen.fingerprint,
                "training_manifest_sha256": raw["data"][
                    "training_manifest_sha256"
                ],
                "measurement_validation_report_sha256": raw[
                    "measurement_validation"
                ]["report_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

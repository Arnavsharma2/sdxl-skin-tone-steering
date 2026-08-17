#!/usr/bin/env python3
"""Audit automated artifacts and report the remaining human submission gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import yaml

try:
    from .build_anonymous_supplement import (
        MAX_SUPPLEMENT_BYTES,
        REQUIRED_SUBMISSION_MANIFESTS,
        ZIP_TIMESTAMP,
    )
except ImportError:  # Direct execution: python scripts/check_submission_readiness.py
    from build_anonymous_supplement import (
        MAX_SUPPLEMENT_BYTES,
        REQUIRED_SUBMISSION_MANIFESTS,
        ZIP_TIMESTAMP,
    )

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "output/pdf/wacv2027_submission_draft.pdf"
SUPPLEMENT_PATH = ROOT / "output/supplement/wacv2027_anonymous_supplement.zip"
CLAIM_AUDIT_PATH = ROOT / "experiments/analysis/manuscript_claim_audit.json"
AUTHOR_INPUTS_PATH = ROOT / "submission/author_inputs.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_gate_failures(data: dict | None, root: Path = ROOT) -> list[str]:
    if not data:
        return ["submission/author_inputs.yaml has not been populated"]

    failures: list[str] = []
    authors = data.get("authors")
    if not isinstance(authors, list) or not authors:
        failures.append("complete author list is missing")
        authors = []
    for index, author in enumerate(authors, start=1):
        if not isinstance(author, dict) or not author.get("name"):
            failures.append(f"author {index} name is missing")
        if not isinstance(author, dict) or not author.get("openreview_id"):
            failures.append(f"author {index} OpenReview ID is missing")
        if not isinstance(author, dict) or author.get("reviewer_obligation_accepted") is not True:
            failures.append(f"author {index} reviewer obligation is not accepted")

    if not data.get("corresponding_author"):
        failures.append("corresponding author is missing")
    if not data.get("paper_id"):
        failures.append("WACV paper ID is missing")
    if data.get("enrollment_completed") is not True:
        failures.append("WACV enrollment is not confirmed")
    if data.get("no_concurrent_review_confirmed") is not True:
        failures.append("no-concurrent-review declaration is not confirmed")
    if not data.get("repository_license") or not (root / "LICENSE").is_file():
        failures.append("repository license is not approved and installed")

    review = data.get("independent_review") or {}
    for field in (
        "completed",
        "technical_claims_checked",
        "statistics_checked",
        "figure_legibility_checked",
        "anonymity_checked",
    ):
        if review.get(field) is not True:
            failures.append(f"independent review field {field} is not confirmed")
    if not review.get("reviewer") or not review.get("date"):
        failures.append("independent reviewer and review date are missing")

    artifact = data.get("large_artifact_review") or {}
    choice = artifact.get("choice")
    if choice not in {"anonymous_url", "unavailable_during_review"}:
        failures.append("large-artifact review option is not selected")
    elif choice == "anonymous_url" and not artifact.get("url"):
        failures.append("anonymous large-artifact URL is missing")
    return failures


def automated_checks(root: Path = ROOT) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    artifacts: dict[str, object] = {}
    for label, path in (("paper", PDF_PATH), ("supplement", SUPPLEMENT_PATH)):
        if not path.is_file():
            failures.append(f"{label} artifact is missing: {path.relative_to(root)}")
            continue
        artifacts[label] = {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    if SUPPLEMENT_PATH.is_file():
        if SUPPLEMENT_PATH.stat().st_size > MAX_SUPPLEMENT_BYTES:
            failures.append("supplement exceeds the 200 MiB limit")
        with zipfile.ZipFile(SUPPLEMENT_PATH) as archive:
            bad_timestamps = [
                info.filename for info in archive.infolist() if info.date_time != ZIP_TIMESTAMP
            ]
            if bad_timestamps:
                failures.append("supplement contains nondeterministic member timestamps")
            names = set(archive.namelist())
            for manifest in REQUIRED_SUBMISSION_MANIFESTS:
                if manifest not in names:
                    failures.append(f"supplement is missing {manifest}")

    if not CLAIM_AUDIT_PATH.is_file():
        failures.append("manuscript claim audit is missing")
    else:
        claim_audit = json.loads(CLAIM_AUDIT_PATH.read_text(encoding="utf-8"))
        if claim_audit.get("passed") is not True or claim_audit.get("claim_count") != 16:
            failures.append("manuscript claim audit is not a passing 16-claim audit")
        for relative, expected in claim_audit.get("papers", {}).items():
            path = root / relative
            if not path.is_file() or sha256_file(path) != expected:
                failures.append(f"manuscript claim audit is stale for {relative}")

    for relative in REQUIRED_SUBMISSION_MANIFESTS:
        if not (root / relative).is_file():
            failures.append(f"checkout is missing {relative}")
    return artifacts, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-human", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts, automated_failures = automated_checks()
    inputs = None
    if AUTHOR_INPUTS_PATH.is_file():
        inputs = yaml.safe_load(AUTHOR_INPUTS_PATH.read_text(encoding="utf-8"))
    human_failures = human_gate_failures(inputs)
    result = {
        "schema_version": "1.0",
        "automated_checks_passed": not automated_failures,
        "submission_ready": not automated_failures and not human_failures,
        "artifacts": artifacts,
        "automated_failures": automated_failures,
        "human_gates_remaining": human_failures,
    }
    print(json.dumps(result, indent=2))
    if automated_failures or (args.require_human and human_failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

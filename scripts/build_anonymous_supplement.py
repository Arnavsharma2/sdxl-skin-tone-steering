#!/usr/bin/env python3
"""Build and identity-scan an anonymized WACV supplementary ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
BANNED_IDENTITY_PATTERNS = (
    b"/users/",
    b"arnav",
    b"arnavgamin",
    b"github.com/arnavsharma2",
    b"@gmail.com",
    b"colab.research.google.com/drive/",
)
MAX_SUPPLEMENT_BYTES = 200 * 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
REQUIRED_SUBMISSION_MANIFESTS = (
    "data/generated/training_manifest_smoke.json",
    "data/generated/training_manifest_calibration_quality.json",
    "data/generated/training_manifest_study_v1.generation.jsonl",
    "data/generated/training_manifest_study_v1.json",
    "data/generated/training_manifest_replication_v1.generation.jsonl",
    "data/generated/training_manifest_replication_v1.json",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def identity_leaks(path: Path, payload: bytes) -> list[str]:
    """Return forbidden author/environment markers found in a text artifact."""

    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    lowered = payload.lower()
    return [
        pattern.decode("utf-8", errors="replace")
        for pattern in BANNED_IDENTITY_PATTERNS
        if pattern in lowered
    ]


def write_deterministic_member(
    archive: zipfile.ZipFile, name: str, payload: bytes
) -> None:
    """Write one stable ZIP member independent of clock, host, and umask."""

    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def collect_files(root: Path) -> list[Path]:
    """Collect only artifacts needed to reproduce claims, never model weights/images."""

    missing_manifests = [
        relative for relative in REQUIRED_SUBMISSION_MANIFESTS if not (root / relative).is_file()
    ]
    if missing_manifests:
        raise FileNotFoundError(
            "Required frozen submission manifests are missing: "
            + ", ".join(missing_manifests)
        )

    fixed = [
        "Makefile",
        "pyproject.toml",
        "requirements-lock.txt",
        "generate_training_data.py",
        "run_race_vector_extraction.py",
        "paper/main.tex",
        "paper/references.bib",
        "paper/wacv/main.tex",
        "paper/wacv/wacv.sty",
        "paper/wacv/ieeenat_fullname.bst",
        *REQUIRED_SUBMISSION_MANIFESTS,
    ]
    patterns = [
        "configs/*.yaml",
        "docs/*.md",
        "scripts/*.py",
        "src/**/*.py",
        "tests/*.py",
        "paper/figures/*.pdf",
        "paper/figures/*.png",
        "experiments/analysis/confirmatory_cuda/*.csv",
        "experiments/analysis/confirmatory_cuda/*.json",
        "experiments/analysis/robustness/*.csv",
        "experiments/analysis/robustness/*.json",
        "experiments/analysis/direction_stability_cuda/*.csv",
        "experiments/analysis/direction_stability_cuda/*.json",
        "experiments/analysis/replication_cuda/*.csv",
        "experiments/analysis/replication_cuda/*.json",
        "experiments/analysis/replication_assessment/*.csv",
        "experiments/analysis/replication_assessment/*.json",
        "experiments/analysis/manuscript_claim_audit.json",
        "experiments/measurement_validation_cuda/*.csv",
        "experiments/measurement_validation_cuda/*.json",
        "experiments/measurement_validation_replication_cuda/*.csv",
        "experiments/measurement_validation_replication_cuda/*.json",
        "experiments/runs/confirmatory_cuda/results.jsonl",
        "experiments/runs/confirmatory_cuda/run_manifest.json",
        "experiments/runs/confirmatory_cuda/study_config.yaml",
        "experiments/runs/replication_cuda/results.jsonl",
        "experiments/runs/replication_cuda/run_manifest.json",
        "experiments/runs/replication_cuda/study_config.yaml",
    ]
    selected = {root / value for value in fixed if (root / value).is_file()}
    for pattern in patterns:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    selected.discard(root / "docs/WACV2027_SUBMISSION_CHECKLIST.md")
    selected.discard(root / "docs/VENUE_PLAN.md")
    selected.discard(root / "docs/RESUME_AND_PROJECT_SUMMARY.md")
    selected.discard(root / "scripts/build_anonymous_supplement.py")
    selected.discard(root / "scripts/check_submission_readiness.py")
    selected.discard(root / "tests/test_anonymous_supplement.py")
    selected.discard(root / "tests/test_submission_readiness.py")
    return sorted(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/supplement/wacv2027_anonymous_supplement.zip"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    files = collect_files(root)
    if not files:
        raise SystemExit("No supplementary files selected")

    records = []
    leaks = []
    payloads = {}
    for path in files:
        relative = path.relative_to(root)
        payload = path.read_bytes()
        payloads[relative] = payload
        found = identity_leaks(relative, payload)
        if found:
            leaks.append({"path": str(relative), "patterns": found})
        records.append(
            {
                "path": str(relative),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    if leaks:
        raise SystemExit("Identity scan failed:\n" + json.dumps(leaks, indent=2))

    readme = (
        "# Anonymous WACV 2027 supplementary artifact\n\n"
        "This package contains source, locked configurations, aggregate result "
        "tables, manifests, tests, and paper figures. It deliberately excludes "
        "all third-party model weights, direction tensors, face embeddings, and "
        "full generated-image campaigns.\n\n"
        "Install the exact dependencies from `requirements-lock.txt`, then run "
        "`python -m pip install -e . --no-deps` and `make check`. See "
        "`docs/REPRODUCIBILITY.md` for the analysis and claim-audit commands.\n"
    ).encode()
    manifest = {
        "schema_version": "1.0",
        "anonymous_review_artifact": True,
        "third_party_weights_included": False,
        "direction_tensors_included": False,
        "face_embeddings_included": False,
        "full_generated_image_runs_included": False,
        "generated_members": [
            "README.md",
            "README_SUPPLEMENT.md",
            "ARTIFACT_MANIFEST.json",
        ],
        "files": records,
    }
    manifest_payload = (json.dumps(manifest, indent=2) + "\n").encode()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, suffix=".zip", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            write_deterministic_member(archive, "README.md", readme)
            write_deterministic_member(archive, "README_SUPPLEMENT.md", readme)
            write_deterministic_member(
                archive, "ARTIFACT_MANIFEST.json", manifest_payload
            )
            for relative, payload in payloads.items():
                write_deterministic_member(archive, str(relative), payload)
        size = temporary_path.stat().st_size
        if size > MAX_SUPPLEMENT_BYTES:
            raise SystemExit(
                f"Supplement is {size / 1024 / 1024:.1f} MiB; WACV limit is 200 MiB"
            )
        temporary_path.replace(args.output)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_files": len(records),
                "archive_members": len(records) + 3,
                "bytes": args.output.stat().st_size,
                "sha256": sha256_bytes(args.output.read_bytes()),
                "identity_scan": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

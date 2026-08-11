"""Fail-closed checksum verification for evaluation model artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


class ArtifactVerificationError(RuntimeError):
    """Raised when a required metric artifact cannot be verified."""


@dataclass(frozen=True)
class ArtifactVerification:
    """Machine-readable result of verifying one required file."""

    name: str
    path: str
    expected_sha256: str
    actual_sha256: str | None
    size_bytes: int | None
    status: str
    verified: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Union[str, Path]) -> tuple[str, int]:
    """Hash a regular file without loading it completely into memory."""
    source = Path(path)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def inspect_artifact(
    path: Union[str, Path],
    expected_sha256: str,
    *,
    name: str,
) -> ArtifactVerification:
    """Return a checksum result without treating any failure as a match."""
    source = Path(path)
    if not source.exists():
        return ArtifactVerification(
            name=name,
            path=str(source),
            expected_sha256=expected_sha256,
            actual_sha256=None,
            size_bytes=None,
            status="missing",
            verified=False,
            error="required artifact does not exist",
        )
    if not source.is_file():
        return ArtifactVerification(
            name=name,
            path=str(source),
            expected_sha256=expected_sha256,
            actual_sha256=None,
            size_bytes=None,
            status="unreadable",
            verified=False,
            error="artifact path is not a regular file",
        )
    try:
        actual_sha256, size_bytes = sha256_file(source)
    except OSError as exc:
        return ArtifactVerification(
            name=name,
            path=str(source),
            expected_sha256=expected_sha256,
            actual_sha256=None,
            size_bytes=None,
            status="unreadable",
            verified=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    matches = actual_sha256 == expected_sha256
    return ArtifactVerification(
        name=name,
        path=str(source),
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        size_bytes=size_bytes,
        status="verified" if matches else "checksum_mismatch",
        verified=matches,
        error=None if matches else "actual SHA-256 does not match the frozen checksum",
    )


def verify_artifact(
    path: Union[str, Path],
    expected_sha256: str,
    *,
    name: str,
) -> ArtifactVerification:
    """Return a verified result or raise a diagnostic that names the failure."""
    result = inspect_artifact(path, expected_sha256, name=name)
    return require_verified(result)


def require_verified(result: ArtifactVerification) -> ArtifactVerification:
    """Raise for an inspected result while allowing callers to retain its status."""
    if result.verified:
        return result
    detail = result.error or result.status
    actual = f"; actual {result.actual_sha256}" if result.actual_sha256 else ""
    readable_status = result.status.replace("_", " ")
    raise ArtifactVerificationError(
        f"{result.name} verification failed ({readable_status}) at {result.path}: "
        f"{detail}; expected {result.expected_sha256}{actual}"
    )

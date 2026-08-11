import hashlib
from pathlib import Path

import pytest

from src.metrics.artifacts import (
    ArtifactVerificationError,
    inspect_artifact,
    verify_artifact,
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_artifact_checksum_match_reports_actual_digest_and_size(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"verified fixture")

    result = verify_artifact(artifact, digest(b"verified fixture"), name="fixture")

    assert result.verified
    assert result.status == "verified"
    assert result.actual_sha256 == result.expected_sha256
    assert result.size_bytes == len(b"verified fixture")


def test_artifact_checksum_mismatch_is_rejected(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"corrupt fixture")

    with pytest.raises(ArtifactVerificationError, match="checksum mismatch"):
        verify_artifact(artifact, digest(b"expected fixture"), name="fixture")

    result = inspect_artifact(artifact, digest(b"expected fixture"), name="fixture")
    assert not result.verified
    assert result.status == "checksum_mismatch"
    assert result.actual_sha256 == digest(b"corrupt fixture")


def test_missing_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "missing.bin"

    with pytest.raises(ArtifactVerificationError, match="missing"):
        verify_artifact(artifact, digest(b"expected"), name="fixture")


def test_non_file_artifact_is_unreadable(tmp_path):
    with pytest.raises(ArtifactVerificationError, match="unreadable"):
        verify_artifact(tmp_path, digest(b"expected"), name="fixture")


def test_read_error_is_reported_as_unreadable(tmp_path, monkeypatch):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"fixture")
    original_open = Path.open

    def fail_open(self, *args, **kwargs):
        if self == artifact:
            raise PermissionError("synthetic unreadable fixture")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)

    result = inspect_artifact(artifact, digest(b"fixture"), name="fixture")
    assert not result.verified
    assert result.status == "unreadable"
    assert "PermissionError" in result.error

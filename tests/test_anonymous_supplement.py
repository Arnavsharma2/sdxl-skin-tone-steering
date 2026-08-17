import io
import zipfile
from pathlib import Path

from scripts.build_anonymous_supplement import (
    ZIP_TIMESTAMP,
    identity_leaks,
    write_deterministic_member,
)


def test_identity_scan_rejects_user_and_repository_markers():
    payload = b"path=/Users/example and github.com/Arnavsharma2/project"
    found = identity_leaks(Path("metadata.json"), payload)
    assert "/users/" in found
    assert "arnav" in found
    assert "github.com/arnavsharma2" in found


def test_identity_scan_ignores_binary_payloads():
    assert identity_leaks(Path("figure.png"), b"/Users/example Arnav") == []


def _archive_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        write_deterministic_member(archive, "result.txt", b"fixed payload\n")
    return output.getvalue()


def test_zip_members_are_byte_reproducible_and_timestamp_free():
    first = _archive_bytes()
    second = _archive_bytes()
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.getinfo("result.txt").date_time == ZIP_TIMESTAMP

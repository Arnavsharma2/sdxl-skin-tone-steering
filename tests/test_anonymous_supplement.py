import io
import zipfile
from pathlib import Path

from scripts.build_anonymous_supplement import (
    REQUIRED_SUBMISSION_MANIFESTS,
    ZIP_TIMESTAMP,
    collect_files,
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


def test_clean_checkout_contains_every_frozen_submission_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    selected = {path.relative_to(root).as_posix() for path in collect_files(root)}

    assert set(REQUIRED_SUBMISSION_MANIFESTS) <= selected


def test_supplement_build_fails_closed_when_frozen_manifests_are_missing(tmp_path) -> None:
    try:
        collect_files(tmp_path)
    except FileNotFoundError as error:
        assert "Required frozen submission manifests are missing" in str(error)
    else:
        raise AssertionError("collect_files accepted a checkout without frozen manifests")

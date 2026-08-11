import pytest

from src.metrics.face_landmarks import FaceLandmarkBackend


def test_missing_model_fails_with_setup_instruction(tmp_path):
    backend = FaceLandmarkBackend(tmp_path / "missing.task")
    with pytest.raises(RuntimeError, match="make metric-models"):
        backend._load()


def test_unverified_model_is_rejected(tmp_path):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"not the pinned model")
    backend = FaceLandmarkBackend(model)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        backend._load()


def test_frozen_runtime_accepts_linux_and_rejects_macos():
    FaceLandmarkBackend.validate_runtime("Linux", (3, 12))

    with pytest.raises(RuntimeError, match="supports Linux only"):
        FaceLandmarkBackend.validate_runtime("Darwin", (3, 12))

    with pytest.raises(RuntimeError, match=r">=3.10,<3.13"):
        FaceLandmarkBackend.validate_runtime("Linux", (3, 13))

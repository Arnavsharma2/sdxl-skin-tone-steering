import numpy as np
import pytest

from src.metrics.identity_metrics import FaceDetectionError, IdentityPreservationMetrics


class EmptyFaceApp:
    def get(self, _image):
        return []


def test_arcface_detection_failure_is_not_encoded_as_zero_similarity():
    metrics = IdentityPreservationMetrics.__new__(IdentityPreservationMetrics)
    metrics.face_app = EmptyFaceApp()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(FaceDetectionError):
        metrics._face_similarity_arcface(image, image)


def test_compute_all_metrics_records_detection_failure_as_missing(monkeypatch):
    metrics = IdentityPreservationMetrics.__new__(IdentityPreservationMetrics)
    monkeypatch.setattr(
        metrics,
        "face_similarity",
        lambda *_args: (_ for _ in ()).throw(FaceDetectionError("missing")),
    )
    monkeypatch.setattr(metrics, "landmark_rmse", lambda *_args: 0.0)
    monkeypatch.setattr(metrics, "perceptual_similarity", lambda *_args: 0.0)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    result = metrics.compute_all_metrics(image, image)
    assert result["face_similarity"] is None

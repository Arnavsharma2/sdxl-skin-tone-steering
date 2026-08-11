
import numpy as np
import pytest
import torch
from PIL import Image

from src.metrics.identity_metrics import IdentityPreservationMetrics


class SequenceDetector:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []

    def __call__(self, image):
        self.inputs.append(image)
        return next(self.outputs)


class SequenceEmbeddingModel:
    def __init__(self, embeddings):
        self.embeddings = iter(embeddings)
        self.inputs = []

    def __call__(self, tensor):
        self.inputs.append(tensor)
        return next(self.embeddings)


def facenet_metric(detector_outputs, embeddings):
    metric = IdentityPreservationMetrics.__new__(IdentityPreservationMetrics)
    metric.device = "cpu"
    metric.facenet_device = "cpu"
    metric.model_type = "facenet"
    metric.mtcnn = SequenceDetector(detector_outputs)
    metric.face_model = SequenceEmbeddingModel(embeddings)
    return metric


def test_facenet_uses_rgb_pil_crops_and_cosine_similarity():
    crop = torch.zeros((3, 160, 160))
    embeddings = [torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]])]
    metric = facenet_metric([crop, crop], embeddings)
    image = np.zeros((32, 24, 3), dtype=np.uint8)

    similarity = metric._face_similarity_facenet(image, image)

    assert similarity == pytest.approx(1.0)
    assert all(isinstance(value, Image.Image) and value.mode == "RGB" for value in metric.mtcnn.inputs)
    assert all(value.shape == (1, 3, 160, 160) for value in metric.face_model.inputs)


def test_facenet_expected_behavior_for_orthogonal_embeddings():
    crop = torch.zeros((3, 160, 160))
    embeddings = [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    metric = facenet_metric([crop, crop], embeddings)

    assert metric._face_similarity_facenet(
        Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))
    ) == pytest.approx(0.0)


def test_facenet_detection_and_invalid_embedding_fail_closed():
    crop = torch.zeros((3, 160, 160))
    metric = facenet_metric([None, crop], [torch.ones((1, 2)), torch.ones((1, 2))])
    with pytest.raises(RuntimeError, match="could not detect"):
        metric._face_similarity_facenet(
            Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))
        )

    metric = facenet_metric([crop, crop], [torch.zeros((1, 2)), torch.ones((1, 2))])
    with pytest.raises(RuntimeError, match="zero-norm"):
        metric._face_similarity_facenet(
            Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))
        )


def test_identity_preprocessing_rejects_non_uint8_arrays():
    with pytest.raises(ValueError, match="uint8 RGB"):
        IdentityPreservationMetrics._as_pil_rgb(np.zeros((8, 8, 3), dtype=np.float32))


def test_lpips_preprocessing_has_exact_shape_and_range():
    black = IdentityPreservationMetrics._prepare_lpips_tensor(
        np.zeros((8, 9, 3), dtype=np.uint8)
    )
    white = IdentityPreservationMetrics._prepare_lpips_tensor(
        np.full((8, 9, 3), 255, dtype=np.uint8)
    )

    assert black.shape == (1, 3, 8, 9)
    assert torch.all(black == -1)
    assert torch.all(white == 1)


def test_lpips_preprocessing_rejects_invalid_tensor_range():
    with pytest.raises(ValueError, match="normalized"):
        IdentityPreservationMetrics._prepare_lpips_tensor(torch.full((3, 8, 8), 2.0))


def test_identity_metric_failures_are_missing_not_passing(monkeypatch):
    metric = IdentityPreservationMetrics.__new__(IdentityPreservationMetrics)
    metric.use_landmarks = False
    monkeypatch.setattr(metric, "face_similarity", lambda *_: (_ for _ in ()).throw(RuntimeError("face")))
    monkeypatch.setattr(
        metric,
        "perceptual_similarity",
        lambda *_: (_ for _ in ()).throw(RuntimeError("lpips")),
    )

    result = metric.compute_all_metrics(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
    )

    assert result == {"face_similarity": None, "landmark_rmse": None, "lpips": None}

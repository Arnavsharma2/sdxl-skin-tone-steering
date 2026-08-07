import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cv2")
pytest.importorskip("skimage")

from src.metrics.evaluator import (  # noqa: E402
    CounterfactualEvaluator,
    EvaluationResult,
    EvaluationThresholds,
)
from src.metrics.structural_metrics import StructuralPreservationMetrics  # noqa: E402


def evaluator_without_models() -> CounterfactualEvaluator:
    evaluator = CounterfactualEvaluator.__new__(CounterfactualEvaluator)
    evaluator.thresholds = EvaluationThresholds()
    return evaluator


def test_missing_required_identity_metric_cannot_pass():
    result = EvaluationResult(
        face_similarity=None,
        lpips=0.1,
        background_ssim=0.95,
    )
    passed, pass_count, total_count, missing = evaluator_without_models()._evaluate_disentanglement(
        result
    )
    assert not passed
    assert pass_count == total_count == 2
    assert missing == ("face_similarity",)


def test_complete_core_metrics_can_pass():
    result = EvaluationResult(
        face_similarity=0.95,
        lpips=0.1,
        background_ssim=0.95,
    )
    passed, _, _, missing = evaluator_without_models()._evaluate_disentanglement(result)
    assert passed
    assert missing == ()


def test_background_ssim_is_missing_without_a_face_mask(monkeypatch):
    metrics = StructuralPreservationMetrics(device="cpu")
    monkeypatch.setattr(
        metrics,
        "create_face_mask",
        lambda image: np.zeros((8, 8), dtype=np.uint8),
    )
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    assert metrics.background_ssim(image, image) is None

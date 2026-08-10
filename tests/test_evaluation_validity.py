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
        total_pose_diff=1.0,
        skin_tone_change=-5.0,
        target_direction_correct=True,
        target_response_pass=True,
    )
    passed, pass_count, total_count, missing = evaluator_without_models()._evaluate_disentanglement(
        result
    )
    assert not passed
    assert pass_count == 4
    assert total_count == 5
    assert missing == ("face_similarity",)
    assert evaluator_without_models()._compute_overall_score(result) is None


def test_complete_core_metrics_can_pass():
    result = EvaluationResult(
        face_similarity=0.95,
        lpips=0.1,
        background_ssim=0.95,
        total_pose_diff=1.0,
        skin_tone_change=-5.0,
        target_direction_correct=True,
        target_response_pass=True,
    )
    passed, _, _, missing = evaluator_without_models()._evaluate_disentanglement(result)
    assert passed
    assert missing == ()


def test_target_metric_and_alpha_are_required():
    result = EvaluationResult(
        face_similarity=0.95,
        lpips=0.1,
        background_ssim=0.95,
        total_pose_diff=1.0,
    )
    passed, _, _, missing = evaluator_without_models()._evaluate_counterfactual(result)
    assert not passed
    assert missing == ("skin_tone_change", "target_direction_correct")


def test_background_ssim_is_missing_without_a_face_mask(monkeypatch):
    metrics = StructuralPreservationMetrics(device="cpu")
    monkeypatch.setattr(
        metrics,
        "create_face_mask",
        lambda image: np.zeros((8, 8), dtype=np.uint8),
    )
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    assert metrics.background_ssim(image, image) is None


def test_background_ssim_averages_only_real_background():
    metrics = StructuralPreservationMetrics(device="cpu")
    original = np.full((64, 64, 3), 120, dtype=np.uint8)
    changed_face = original.copy()
    changed_face[20:44, 20:44] = 240
    face_mask = np.zeros((64, 64), dtype=np.uint8)
    face_mask[16:48, 16:48] = 1
    assert metrics.background_ssim(original, changed_face, mask=face_mask) == pytest.approx(1.0)

    changed_background = original.copy()
    changed_background[:, :12] = 30
    assert metrics.background_ssim(original, changed_background, mask=face_mask) < 0.95

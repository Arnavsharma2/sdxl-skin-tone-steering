
import numpy as np
import pytest

from src.metrics.face_landmarks import FaceLandmarkResult
from src.metrics.structural_metrics import StructuralPreservationMetrics


class SequenceBackend:
    def __init__(self, results):
        self.results = iter(results)

    def detect(self, _image):
        return next(self.results)


def transform_for_roll(degrees):
    radians = np.radians(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=float
    )
    return matrix


def test_face_detection_failure_produces_no_mask():
    metrics = StructuralPreservationMetrics(
        device="cpu", landmark_backend=SequenceBackend([None])
    )

    assert metrics.detect_face_bbox(np.zeros((32, 32, 3), dtype=np.uint8)) is None

    metrics = StructuralPreservationMetrics(
        device="cpu", landmark_backend=SequenceBackend([None])
    )
    assert metrics.create_face_mask(np.zeros((32, 32, 3), dtype=np.uint8)) is None


def test_pose_uses_valid_transformation_matrix_and_degrees():
    result = FaceLandmarkResult(landmarks=(), transformation_matrix=transform_for_roll(30))
    metrics = StructuralPreservationMetrics(
        device="cpu", landmark_backend=SequenceBackend([result])
    )

    pose = metrics.estimate_3d_pose(np.zeros((8, 8, 3), dtype=np.uint8))

    assert pose["yaw"] == pytest.approx(0.0, abs=1e-7)
    assert pose["pitch"] == pytest.approx(0.0, abs=1e-7)
    assert pose["roll"] == pytest.approx(30.0)


@pytest.mark.parametrize(
    "matrix",
    [np.eye(3), np.full((4, 4), np.nan), np.diag([2.0, 1.0, 1.0, 1.0])],
)
def test_pose_rejects_malformed_or_nonrotation_matrices(matrix):
    result = FaceLandmarkResult(landmarks=(), transformation_matrix=matrix)
    metrics = StructuralPreservationMetrics(
        device="cpu", landmark_backend=SequenceBackend([result])
    )

    assert metrics.estimate_3d_pose(np.zeros((8, 8, 3), dtype=np.uint8)) is None


def test_pose_difference_handles_angle_wraparound(monkeypatch):
    metrics = StructuralPreservationMetrics(
        device="cpu", landmark_backend=SequenceBackend([])
    )
    poses = iter(
        [
            {"yaw": 179.0, "pitch": 0.0, "roll": 0.0},
            {"yaw": -179.0, "pitch": 0.0, "roll": 0.0},
        ]
    )
    monkeypatch.setattr(metrics, "estimate_3d_pose", lambda _: next(poses))

    result = metrics.pose_difference(None, None)

    assert result["yaw_diff"] == pytest.approx(2.0)
    assert result["total_diff"] == pytest.approx(2.0)


def test_pose_failure_returns_missing_values(monkeypatch):
    metrics = StructuralPreservationMetrics(
        device="cpu", landmark_backend=SequenceBackend([])
    )
    monkeypatch.setattr(metrics, "estimate_3d_pose", lambda _: None)

    assert metrics.pose_difference(None, None) == {
        "yaw_diff": None,
        "pitch_diff": None,
        "roll_diff": None,
        "total_diff": None,
    }


def test_background_ssim_uses_union_of_both_face_masks():
    metrics = StructuralPreservationMetrics(device="cpu")
    original = np.full((64, 64, 3), 120, dtype=np.uint8)
    changed = original.copy()
    changed[20:44, 20:44] = 240
    first_mask = np.zeros((64, 64), dtype=np.uint8)
    first_mask[20:32, 20:44] = 1
    second_mask = np.zeros((64, 64), dtype=np.uint8)
    second_mask[32:44, 20:44] = 1

    score = metrics.background_ssim(
        original,
        changed,
        mask=first_mask,
        counterfactual_mask=second_mask,
    )

    assert score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("shape", "mask_factory"),
    [
        ((64, 63, 3), lambda: np.ones((64, 64), dtype=np.uint8)),
        ((64, 64, 3), lambda: np.ones((63, 64), dtype=np.uint8)),
        ((64, 64, 3), lambda: np.zeros((64, 64), dtype=np.uint8)),
        ((64, 64, 3), lambda: np.ones((64, 64), dtype=np.uint8)),
        ((64, 64, 3), lambda: np.full((64, 64), 2, dtype=np.uint8)),
    ],
)
def test_background_ssim_mask_and_shape_failures_return_none(shape, mask_factory):
    metrics = StructuralPreservationMetrics(device="cpu")
    original = np.zeros((64, 64, 3), dtype=np.uint8)
    counterfactual = np.zeros(shape, dtype=np.uint8)

    assert metrics.background_ssim(original, counterfactual, mask=mask_factory()) is None


def test_background_ssim_rejects_too_little_eroded_background():
    metrics = StructuralPreservationMetrics(device="cpu")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.ones((16, 16), dtype=np.uint8)
    mask[:6, :6] = 0

    assert metrics.background_ssim(image, image, mask=mask) is None


def test_background_ssim_rejects_images_smaller_than_ssim_window():
    metrics = StructuralPreservationMetrics(device="cpu")
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[:, 3:5] = 1

    assert metrics.background_ssim(image, image, mask=mask) is None

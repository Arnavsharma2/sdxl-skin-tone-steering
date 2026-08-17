import numpy as np

from src.metrics.structural_metrics import StructuralPreservationMetrics


def test_symmetric_five_point_landmarks_have_zero_pose_proxy():
    points = np.array(
        [[10.0, 10.0], [30.0, 10.0], [20.0, 20.0], [14.0, 30.0], [26.0, 30.0]]
    )
    pose = StructuralPreservationMetrics._pose_from_landmarks(points)
    assert np.allclose([pose["yaw"], pose["pitch"], pose["roll"]], 0.0)


def test_nose_shift_changes_yaw_proxy():
    points = np.array(
        [[10.0, 10.0], [30.0, 10.0], [24.0, 20.0], [14.0, 30.0], [26.0, 30.0]]
    )
    pose = StructuralPreservationMetrics._pose_from_landmarks(points)
    assert pose["yaw"] > 10.0

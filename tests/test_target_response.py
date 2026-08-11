import math

import pytest

from src.metrics.target_response import calculate_monotonicity

ALPHAS = (-1.5, -0.75, 0.0, 0.75, 1.5)


def test_strict_decreasing_target_response_is_monotonic():
    result = calculate_monotonicity(
        zip(ALPHAS, (12.0, 8.0, 4.0, 1.0, -3.0)),
        expected_alphas=ALPHAS,
    )

    assert result.valid
    assert result.spearman_rho == pytest.approx(-1.0)
    assert result.adjacent_monotonic_fraction == 1.0
    assert result.strictly_monotonic is True


def test_partial_monotonicity_is_calculated_without_becoming_a_pass():
    result = calculate_monotonicity(
        zip(ALPHAS, (12.0, 8.0, 9.0, 1.0, -3.0)),
        expected_alphas=ALPHAS,
    )

    assert result.valid
    assert result.adjacent_monotonic_fraction == 0.75
    assert result.strictly_monotonic is False


@pytest.mark.parametrize(
    ("measurements", "reason"),
    [
        ([(-1.5, 3.0), (-0.75, 2.0), (0.0, 1.0)], "missing_or_unexpected_alpha"),
        ([(-1.5, 3.0), (-0.75, 2.0), (0.0, 1.0), (0.75, 0.0), (0.75, -1.0)], "duplicate_alpha"),
        (list(zip(ALPHAS, (3.0, 2.0, None, 0.0, -1.0))), "missing_target_measurement"),
        (list(zip(ALPHAS, (3.0, 2.0, math.nan, 0.0, -1.0))), "nonfinite_target_measurement"),
        (list(zip(ALPHAS, (1.0, 1.0, 1.0, 1.0, 1.0))), "constant_target_response"),
    ],
)
def test_invalid_sweeps_never_use_complete_case_subsets(measurements, reason):
    result = calculate_monotonicity(measurements, expected_alphas=ALPHAS)

    assert not result.valid
    assert result.reason == reason
    assert result.spearman_rho is None
    assert result.adjacent_monotonic_fraction is None

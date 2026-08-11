"""Deterministic sweep-level rendered-skin-tone response calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

FROZEN_ALPHA_GRID = (-1.5, -0.75, 0.0, 0.75, 1.5)


@dataclass(frozen=True)
class MonotonicityResult:
    """Validity and values for one prespecified alpha sweep."""

    valid: bool
    reason: str | None
    expected_alphas: tuple[float, ...]
    observed_alphas: tuple[float, ...]
    point_count: int
    spearman_rho: float | None
    adjacent_monotonic_fraction: float | None
    strictly_monotonic: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_monotonicity(
    measurements: Iterable[tuple[float, float | None]],
    *,
    expected_alphas: Sequence[float],
) -> MonotonicityResult:
    """Calculate a decreasing response, invalidating incomplete or ambiguous sweeps."""
    expected = tuple(float(value) for value in expected_alphas)
    if len(expected) < 3 or len(set(expected)) != len(expected):
        raise ValueError("expected_alphas must contain at least three unique values")
    if not np.isfinite(np.asarray(expected, dtype=float)).all():
        raise ValueError("expected_alphas must be finite")
    expected = tuple(sorted(expected))

    observed_pairs = [(float(alpha), value) for alpha, value in measurements]
    observed_alphas = tuple(sorted(alpha for alpha, _ in observed_pairs))
    if not np.isfinite(np.asarray(observed_alphas, dtype=float)).all():
        return _invalid("nonfinite_alpha", expected, observed_alphas)
    if len(observed_alphas) != len(set(observed_alphas)):
        return _invalid("duplicate_alpha", expected, observed_alphas)
    if set(observed_alphas) != set(expected):
        return _invalid("missing_or_unexpected_alpha", expected, observed_alphas)

    by_alpha = dict(observed_pairs)
    responses = []
    for alpha in expected:
        value = by_alpha[alpha]
        if value is None:
            return _invalid("missing_target_measurement", expected, observed_alphas)
        numeric = float(value)
        if not np.isfinite(numeric):
            return _invalid("nonfinite_target_measurement", expected, observed_alphas)
        responses.append(numeric)

    if len(set(responses)) < 2:
        return _invalid("constant_target_response", expected, observed_alphas)

    statistic = spearmanr(np.asarray(expected), np.asarray(responses))
    rho = float(statistic.statistic)
    if not np.isfinite(rho):
        return _invalid("undefined_spearman_rho", expected, observed_alphas)
    differences = np.diff(np.asarray(responses, dtype=float))
    adjacent_fraction = float(np.mean(differences < 0))
    return MonotonicityResult(
        valid=True,
        reason=None,
        expected_alphas=expected,
        observed_alphas=observed_alphas,
        point_count=len(expected),
        spearman_rho=rho,
        adjacent_monotonic_fraction=adjacent_fraction,
        strictly_monotonic=bool(np.all(differences < 0)),
    )


def _invalid(
    reason: str,
    expected: tuple[float, ...],
    observed: tuple[float, ...],
) -> MonotonicityResult:
    return MonotonicityResult(
        valid=False,
        reason=reason,
        expected_alphas=expected,
        observed_alphas=observed,
        point_count=len(observed),
        spearman_rho=None,
        adjacent_monotonic_fraction=None,
        strictly_monotonic=None,
    )

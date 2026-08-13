"""Prespecified matched-change estimators for the confirmatory study.

The generation seed is the only sampling, pairing, and bootstrap unit. Image
rows are inputs to a deterministic within-seed curve construction and are
never treated as independent observations.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.utils.config import AnalysisComparisonConfig, ExperimentConfig

TIE_ATOL = 1e-12


@dataclass(frozen=True)
class MatchResult:
    """One seed-level matched-change contrast or its failure reason."""

    seed: int
    comparison_id: str
    computable: bool
    reason: str | None
    support_low: float | None = None
    support_high: float | None = None
    grid_points: int = 0
    method_a_mean: float | None = None
    method_b_mean: float | None = None
    effect_a_minus_b: float | None = None
    favorable_effect: float | None = None


def _derived_seed(base_seed: int, label: str) -> int:
    """Derive an order-independent NumPy seed from the recorded analysis seed."""
    digest = hashlib.sha256(f"{base_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _invalid_match(seed: int, comparison_id: str, reason: str) -> MatchResult:
    return MatchResult(seed, comparison_id, False, reason)


def _finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _collapse_ties(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average preservation at target changes equal within the frozen tolerance."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    sorted_y = y[order]
    grouped_x: list[float] = []
    grouped_y: list[float] = []
    start = 0
    while start < len(sorted_x):
        stop = start + 1
        while stop < len(sorted_x) and np.isclose(
            sorted_x[stop], sorted_x[start], rtol=0.0, atol=TIE_ATOL
        ):
            stop += 1
        grouped_x.append(float(np.mean(sorted_x[start:stop])))
        grouped_y.append(float(np.mean(sorted_y[start:stop])))
        start = stop
    return np.asarray(grouped_x), np.asarray(grouped_y)


def _method_curve(
    seed_frame: pd.DataFrame,
    *,
    method: str,
    metric: str,
    expected_alphas: Sequence[float],
    minimum_unique_points: int,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, str | None]:
    method_frame = seed_frame.loc[seed_frame["method"].eq(method)].copy()
    observed = pd.to_numeric(method_frame["alpha"], errors="coerce")
    expected = np.asarray(expected_alphas, dtype=float)
    if observed.isna().any():
        return None, f"{method}:nonfinite_alpha"
    observed_values = observed.to_numpy(dtype=float)
    unexpected = [
        value
        for value in observed_values
        if not np.any(np.isclose(value, expected, rtol=0, atol=0))
    ]
    if unexpected:
        return None, f"{method}:unexpected_alpha"
    for alpha in expected:
        count = int(np.count_nonzero(observed_values == alpha))
        if count == 0:
            return None, f"{method}:missing_alpha:{alpha:g}"
        if count > 1:
            return None, f"{method}:duplicate_alpha:{alpha:g}"

    method_frame["alpha"] = observed_values
    method_frame = method_frame.sort_values("alpha", kind="mergesort")
    target = pd.to_numeric(method_frame["skin_tone_change"], errors="coerce")
    if not np.isfinite(target.to_numpy(dtype=float)).all():
        return None, f"{method}:nonfinite_target_change"
    target_values = target.to_numpy(dtype=float)
    if not np.all(np.diff(target_values) < 0):
        return None, f"{method}:nonmonotonic_target_sweep"

    nonzero = method_frame.loc[method_frame["alpha"].ne(0)].copy()
    if "analysis_row_valid" in nonzero and not nonzero["analysis_row_valid"].astype(bool).all():
        return None, f"{method}:invalid_evaluation_row"
    preservation = pd.to_numeric(nonzero[metric], errors="coerce")
    if not np.isfinite(preservation.to_numpy(dtype=float)).all():
        return None, f"{method}:nonfinite_metric:{metric}"
    x = pd.to_numeric(nonzero["skin_tone_change"], errors="coerce").abs().to_numpy()
    y = preservation.to_numpy(dtype=float)
    x, y = _collapse_ties(x, y)
    if len(x) < minimum_unique_points:
        return None, f"{method}:insufficient_unique_target_changes"
    if not np.all(np.diff(x) > 0):
        return None, f"{method}:invalid_tie_collapse"
    return (x, y), None


def matched_seed_contrast(
    seed_frame: pd.DataFrame,
    *,
    seed: int,
    comparison: AnalysisComparisonConfig,
    expected_alphas: Sequence[float],
    minimum_abs_target_change: float,
    grid_points: int,
    minimum_unique_points: int,
) -> MatchResult:
    """Estimate one method contrast on the seed-specific common target support."""
    required = {"method", "alpha", "skin_tone_change", comparison.metric}
    missing = sorted(required - set(seed_frame.columns))
    if missing:
        return _invalid_match(seed, comparison.id, f"missing_columns:{','.join(missing)}")
    curve_a, reason = _method_curve(
        seed_frame,
        method=comparison.method_a,
        metric=comparison.metric,
        expected_alphas=expected_alphas,
        minimum_unique_points=minimum_unique_points,
    )
    if reason:
        return _invalid_match(seed, comparison.id, reason)
    curve_b, reason = _method_curve(
        seed_frame,
        method=comparison.method_b,
        metric=comparison.metric,
        expected_alphas=expected_alphas,
        minimum_unique_points=minimum_unique_points,
    )
    if reason:
        return _invalid_match(seed, comparison.id, reason)
    assert curve_a is not None and curve_b is not None
    x_a, y_a = curve_a
    x_b, y_b = curve_b
    support_low = max(float(x_a[0]), float(x_b[0]), minimum_abs_target_change)
    support_high = min(float(x_a[-1]), float(x_b[-1]))
    if support_high - support_low <= TIE_ATOL:
        return _invalid_match(seed, comparison.id, "no_positive_width_common_support")
    if grid_points < 2:
        return _invalid_match(seed, comparison.id, "invalid_grid_points")

    grid = np.linspace(support_low, support_high, grid_points)
    # np.interp is used only after the explicit inclusive support check above;
    # therefore its endpoint behavior cannot silently extrapolate.
    if grid[0] < x_a[0] - TIE_ATOL or grid[-1] > x_a[-1] + TIE_ATOL:
        return _invalid_match(seed, comparison.id, f"{comparison.method_a}:extrapolation")
    if grid[0] < x_b[0] - TIE_ATOL or grid[-1] > x_b[-1] + TIE_ATOL:
        return _invalid_match(seed, comparison.id, f"{comparison.method_b}:extrapolation")
    matched_a = np.interp(grid, x_a, y_a)
    matched_b = np.interp(grid, x_b, y_b)
    mean_a = float(np.mean(matched_a))
    mean_b = float(np.mean(matched_b))
    effect = mean_a - mean_b
    favorable = effect if comparison.favorable_direction == "higher" else -effect
    return MatchResult(
        seed=seed,
        comparison_id=comparison.id,
        computable=True,
        reason=None,
        support_low=support_low,
        support_high=support_high,
        grid_points=grid_points,
        method_a_mean=mean_a,
        method_b_mean=mean_b,
        effect_a_minus_b=effect,
        favorable_effect=favorable,
    )


def seed_cluster_bootstrap_ci(
    seed_effects: pd.DataFrame | Iterable[tuple[int, float]],
    *,
    resamples: int,
    confidence: float,
    rng_seed: int,
    label: str,
) -> tuple[float, float]:
    """Bootstrap an unweighted seed mean by resampling complete seed clusters."""
    if isinstance(seed_effects, pd.DataFrame):
        pairs = list(seed_effects[["seed", "effect_a_minus_b"]].itertuples(index=False, name=None))
    else:
        pairs = list(seed_effects)
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap resamples or confidence")
    seeds = [int(seed) for seed, _ in pairs]
    values = np.asarray([value for _, value in pairs], dtype=float)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed-cluster bootstrap requires one effect per unique seed")
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    order = np.argsort(seeds, kind="mergesort")
    values = values[order]
    rng = np.random.default_rng(_derived_seed(rng_seed, f"bootstrap:{label}"))
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(low), float(high)


def paired_sign_flip_pvalue(
    values: Sequence[float], *, resamples: int, rng_seed: int, label: str
) -> float:
    """Two-sided Monte Carlo paired sign-flip p-value with plus-one correction."""
    effects = np.asarray(values, dtype=float)
    if len(effects) < 2 or not np.isfinite(effects).all() or resamples < 1:
        return float("nan")
    observed = abs(float(np.mean(effects)))
    rng = np.random.default_rng(_derived_seed(rng_seed, f"sign_flip:{label}"))
    signs = rng.integers(0, 2, size=(resamples, len(effects)), dtype=np.int8) * 2 - 1
    permuted = np.abs((signs * effects).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed - 1e-15) + 1) / (resamples + 1))


def holm_adjust(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Holm-adjust prespecified tests with stable ties and conservative missingness.

    Invalid or missing p-values remain unadjusted and non-rejecting. They still
    count in the frozen family size, so unavailable tests cannot make the
    remaining comparisons less conservative.
    """
    output = [dict(record) for record in records]
    family_size = len(output)
    valid: list[tuple[float, int]] = []
    for index, record in enumerate(output):
        value = record.get("p_value")
        if _finite_number(value) and 0 <= float(value) <= 1:
            valid.append((float(value), index))
            output[index]["holm_reason"] = None
        else:
            output[index]["holm_adjusted_p"] = None
            output[index]["holm_reject_0_05"] = False
            output[index]["holm_reason"] = "missing_or_invalid_p_value"
    valid.sort(key=lambda item: (item[0], item[1]))
    running = 0.0
    for rank, (p_value, index) in enumerate(valid):
        adjusted = min(1.0, (family_size - rank) * p_value)
        running = max(running, adjusted)
        output[index]["holm_adjusted_p"] = running
        output[index]["holm_reject_0_05"] = bool(running <= 0.05)
    return output


def analyze_comparisons(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return seed-level and aggregate prespecified matched-change contrasts."""
    matched = config.analysis.matched_change
    seed_records: list[dict[str, Any]] = []
    aggregate_records: list[dict[str, Any]] = []
    for comparison in config.analysis.comparisons:
        results = [
            matched_seed_contrast(
                frame.loc[frame["seed"].eq(seed)],
                seed=seed,
                comparison=comparison,
                expected_alphas=config.evaluation.alphas,
                minimum_abs_target_change=matched.minimum_abs_target_change,
                grid_points=matched.grid_points,
                minimum_unique_points=matched.minimum_unique_points_per_curve,
            )
            for seed in config.evaluation.seeds
        ]
        seed_records.extend(
            {
                **asdict(result),
                "role": comparison.role,
                "hypothesis": comparison.hypothesis,
                "method_a": comparison.method_a,
                "method_b": comparison.method_b,
                "metric": comparison.metric,
                "favorable_direction": comparison.favorable_direction,
            }
            for result in results
        )
        valid = [result for result in results if result.computable]
        complete = len(valid) == len(config.evaluation.seeds)
        effects = [float(result.effect_a_minus_b) for result in valid]
        record: dict[str, Any] = {
            "comparison_id": comparison.id,
            "role": comparison.role,
            "hypothesis": comparison.hypothesis,
            "method_a": comparison.method_a,
            "method_b": comparison.method_b,
            "metric": comparison.metric,
            "favorable_direction": comparison.favorable_direction,
            "expected_seed_count": len(config.evaluation.seeds),
            "included_seed_count": len(valid),
            "invalid_seed_count": len(config.evaluation.seeds) - len(valid),
            "confirmatory_computable": complete,
            "not_computable_reason": None if complete else "incomplete_prespecified_seed_set",
            "estimate_a_minus_b": None,
            "favorable_effect": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_value": None,
            "exploratory_valid_seed_estimate": float(np.mean(effects)) if effects else None,
        }
        if complete:
            estimate = float(np.mean(effects))
            low, high = seed_cluster_bootstrap_ci(
                [(result.seed, float(result.effect_a_minus_b)) for result in valid],
                resamples=config.evaluation.bootstrap.resamples,
                confidence=config.evaluation.bootstrap.confidence_level,
                rng_seed=config.analysis.rng_seed,
                label=comparison.id,
            )
            record.update(
                {
                    "estimate_a_minus_b": estimate,
                    "favorable_effect": (
                        estimate if comparison.favorable_direction == "higher" else -estimate
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": paired_sign_flip_pvalue(
                        effects,
                        resamples=config.analysis.randomization_resamples,
                        rng_seed=config.analysis.rng_seed,
                        label=comparison.id,
                    ),
                }
            )
        aggregate_records.append(record)

    secondaries = [record for record in aggregate_records if record["role"] == "secondary"]
    adjusted = {record["comparison_id"]: record for record in holm_adjust(secondaries)}
    for record in aggregate_records:
        if record["role"] == "primary":
            record.update(
                {
                    "holm_adjusted_p": None,
                    "holm_reject_0_05": False,
                    "holm_reason": "primary_not_in_secondary_family",
                }
            )
        else:
            record.update(
                {
                    key: adjusted[record["comparison_id"]][key]
                    for key in ("holm_adjusted_p", "holm_reject_0_05", "holm_reason")
                }
            )
    return pd.DataFrame(seed_records), pd.DataFrame(aggregate_records)

import math

from scripts.summarize_results import bootstrap_mean_ci


def test_bootstrap_interval_contains_sample_mean():
    values = [0.7, 0.8, 0.9, 1.0]
    low, high = bootstrap_mean_ci(values, resamples=2_000, seed=7)
    assert low < sum(values) / len(values) < high


def test_bootstrap_requires_two_finite_values():
    low, high = bootstrap_mean_ci([0.5, math.nan])
    assert math.isnan(low)
    assert math.isnan(high)

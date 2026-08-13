"""Deterministic, fail-closed confirmatory statistical analysis."""

from .statistics import (
    MatchResult,
    analyze_comparisons,
    holm_adjust,
    matched_seed_contrast,
    seed_cluster_bootstrap_ci,
)

__all__ = [
    "MatchResult",
    "analyze_comparisons",
    "holm_adjust",
    "matched_seed_contrast",
    "seed_cluster_bootstrap_ci",
]

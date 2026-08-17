"""Seed-level descriptive and matched-change analyses for study results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

OUTCOME_ORIENTATION = {
    "face_similarity": 1.0,
    "lpips": -1.0,
    "background_ssim": 1.0,
    "total_pose_diff": -1.0,
}

DEFAULT_METRICS = (
    "skin_tone_change",
    "lightness_change",
    "face_similarity",
    "landmark_rmse",
    "lpips",
    "background_ssim",
    "overall_ssim",
    "total_pose_diff",
)


def load_jsonl_results(root: str | Path) -> pd.DataFrame:
    """Load every results.jsonl below a file or directory."""

    path = Path(root)
    paths = [path] if path.is_file() else sorted(path.rglob("results.jsonl"))
    frames = []
    for result_path in paths:
        frame = pd.read_json(result_path, lines=True)
        frame["run"] = str(result_path.parent)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval over independent seed values."""

    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def descriptive_summary(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize each method/alpha with missingness and seed bootstrap CIs."""

    records = []
    if frame.empty:
        return pd.DataFrame()
    for (method, alpha), group in frame.groupby(["method", "alpha"], dropna=False):
        total = int(group["seed"].nunique())
        for metric in metrics:
            series = (
                pd.to_numeric(group[metric], errors="coerce")
                if metric in group
                else pd.Series(dtype=float)
            )
            values = series[np.isfinite(series)]
            low, high = bootstrap_mean_ci(
                values, resamples=resamples, confidence=confidence
            )
            records.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "metric": metric,
                    "n_seeds": int(values.size),
                    "n_missing": total - int(values.size),
                    "mean": float(values.mean()) if values.size else math.nan,
                    "std": float(values.std(ddof=1)) if values.size > 1 else math.nan,
                    "median": float(values.median()) if values.size else math.nan,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame.from_records(records)


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report attempted, generated, and complete conditions without dropping failures."""

    records = []
    if frame.empty:
        return pd.DataFrame()
    for (method, alpha), group in frame.groupby(["method", "alpha"], dropna=False):
        attempted = len(group)
        generated = int(group.get("generation_complete", False).fillna(False).sum())
        complete = int(group.get("evaluation_complete", False).fillna(False).sum())
        records.append(
            {
                "method": method,
                "alpha": alpha,
                "attempted": attempted,
                "generated": generated,
                "evaluation_complete": complete,
                "generation_failure_rate": (attempted - generated) / attempted,
                "metric_failure_rate": (attempted - complete) / attempted,
            }
        )
    return pd.DataFrame.from_records(records)


def monotonicity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute per-seed Spearman response then summarize by method."""

    per_seed = []
    required = {"seed", "method", "alpha", "skin_tone_change"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    valid = frame.copy()
    valid["skin_tone_change"] = pd.to_numeric(valid["skin_tone_change"], errors="coerce")
    valid = valid[np.isfinite(valid["skin_tone_change"])]
    for (seed, method), group in valid.groupby(["seed", "method"]):
        group = group.drop_duplicates("alpha")
        if len(group) < 3:
            continue
        alpha_rank = group["alpha"].rank(method="average").to_numpy()
        change_rank = group["skin_tone_change"].rank(method="average").to_numpy()
        correlation = float(np.corrcoef(alpha_rank, change_rank)[0, 1])
        per_seed.append({"seed": seed, "method": method, "spearman": correlation})
    seed_frame = pd.DataFrame(per_seed)
    if seed_frame.empty:
        return seed_frame
    records = []
    for method, group in seed_frame.groupby("method"):
        low, high = bootstrap_mean_ci(group["spearman"])
        records.append(
            {
                "method": method,
                "n_seeds": len(group),
                "mean_spearman": float(group["spearman"].mean()),
                "median_spearman": float(group["spearman"].median()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame.from_records(records)


def match_at_target_change(
    frame: pd.DataFrame,
    targets: Sequence[float],
    *,
    tolerance: float,
) -> pd.DataFrame:
    """Select each seed/method's closest observed aligned change per direction.

    Targets are positive absolute ITA-degree changes.  Negative and positive
    alpha directions are matched separately.  A selected observation outside
    ``tolerance`` is retained with ``match_complete=False`` so it remains part
    of the missingness audit, but is excluded from paired effect estimates.
    """

    if any(float(target) <= 0 for target in targets):
        raise ValueError("matched-change targets must be positive")
    records = []
    valid = frame.copy()
    valid["skin_tone_change"] = pd.to_numeric(valid["skin_tone_change"], errors="coerce")
    valid = valid[(valid["alpha"] != 0) & np.isfinite(valid["skin_tone_change"])]
    valid["direction"] = np.sign(valid["alpha"]).astype(int)
    valid["aligned_change"] = valid["direction"] * valid["skin_tone_change"]
    for (seed, method, direction), group in valid.groupby(
        ["seed", "method", "direction"]
    ):
        for target in targets:
            distances = (group["aligned_change"] - float(target)).abs()
            selected = group.loc[distances.idxmin()].to_dict()
            distance = float(distances.min())
            selected.update(
                {
                    "direction": int(direction),
                    "target_change": float(target),
                    "match_distance": distance,
                    "match_complete": distance <= tolerance,
                }
            )
            records.append(selected)
    return pd.DataFrame.from_records(records)


def matched_coverage_summary(matched: pd.DataFrame) -> pd.DataFrame:
    """Summarize prespecified target coverage and achieved changes by stratum."""

    if matched.empty:
        return pd.DataFrame()
    records = []
    for (method, direction, target), group in matched.groupby(
        ["method", "direction", "target_change"], dropna=False
    ):
        complete = group[group["match_complete"].fillna(False)].copy()
        achieved = pd.to_numeric(complete.get("aligned_change"), errors="coerce")
        distances = pd.to_numeric(complete.get("match_distance"), errors="coerce")
        attempted = int(group["seed"].nunique())
        n_complete = int(complete["seed"].nunique())
        records.append(
            {
                "method": method,
                "direction": int(direction),
                "target_change": float(target),
                "attempted_seeds": attempted,
                "complete_seeds": n_complete,
                "coverage_rate": n_complete / attempted if attempted else math.nan,
                "mean_achieved_change": float(achieved.mean()) if n_complete else math.nan,
                "std_achieved_change": (
                    float(achieved.std(ddof=1)) if n_complete > 1 else math.nan
                ),
                "mean_match_distance": float(distances.mean()) if n_complete else math.nan,
                "median_match_distance": float(distances.median()) if n_complete else math.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def randomization_pvalue(
    paired_advantages: Iterable[float],
    *,
    resamples: int = 100_000,
    seed: int = 2026,
) -> float:
    """Two-sided paired sign-flip randomization p-value."""

    values = np.asarray(list(paired_advantages), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = int(resamples)
    chunk_size = 10_000
    while remaining:
        chunk = min(chunk_size, remaining)
        signs = rng.choice((-1.0, 1.0), size=(chunk, values.size))
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
        remaining -= chunk
    return (extreme + 1.0) / (resamples + 1.0)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values, preserving input order."""

    array = np.asarray(p_values, dtype=float)
    adjusted = np.full(array.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(array))
    if not finite_indices.size:
        return adjusted.tolist()
    ordered = finite_indices[np.argsort(array[finite_indices])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(array[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_method_contrasts(
    matched: pd.DataFrame,
    *,
    reference_method: str,
    outcomes: Mapping[str, float] = OUTCOME_ORIENTATION,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compare each method with the reference using within-seed matched rows."""

    usable = matched[matched["match_complete"].fillna(False)].copy()
    methods = sorted(set(usable.get("method", ())) - {reference_method})
    records = []
    for comparator in methods:
        for direction in sorted(usable["direction"].dropna().unique()):
            for target in sorted(usable["target_change"].dropna().unique()):
                stratum = usable[
                    (usable["direction"] == direction)
                    & (usable["target_change"] == target)
                ]
                reference = stratum[stratum["method"] == reference_method]
                other = stratum[stratum["method"] == comparator]
                for outcome, orientation in outcomes.items():
                    if outcome not in stratum:
                        continue
                    paired = reference[["seed", outcome]].merge(
                        other[["seed", outcome]],
                        on="seed",
                        suffixes=("_reference", "_comparator"),
                    )
                    left = pd.to_numeric(
                        paired[f"{outcome}_reference"], errors="coerce"
                    )
                    right = pd.to_numeric(
                        paired[f"{outcome}_comparator"], errors="coerce"
                    )
                    advantage = orientation * (left - right)
                    advantage = advantage[np.isfinite(advantage)]
                    low, high = bootstrap_mean_ci(
                        advantage,
                        resamples=resamples,
                        confidence=confidence,
                    )
                    records.append(
                        {
                            "reference_method": reference_method,
                            "comparator": comparator,
                            "direction": int(direction),
                            "target_change": float(target),
                            "outcome": outcome,
                            "n_pairs": int(advantage.size),
                            "mean_reference_advantage": (
                                float(advantage.mean()) if advantage.size else math.nan
                            ),
                            "median_reference_advantage": (
                                float(advantage.median()) if advantage.size else math.nan
                            ),
                            "ci_low": low,
                            "ci_high": high,
                            "p_value": randomization_pvalue(advantage),
                        }
                    )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result["p_holm"] = holm_adjust(result["p_value"].tolist())
    return result

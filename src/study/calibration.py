"""Auditable helpers for synthesizing deliberately adaptive calibration runs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

SCIENTIFIC_RESULT_COLUMNS = (
    "generation_complete",
    "evaluation_complete",
    "skin_tone_change",
    "lightness_change",
    "face_similarity",
    "landmark_rmse",
    "lpips",
    "background_ssim",
    "overall_ssim",
    "total_pose_diff",
)


def _equivalent(left: object, right: object, *, atol: float) -> bool:
    """Return whether two recorded scientific values agree."""

    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return bool(left) == bool(right)
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return left == right
    return math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=atol)


def deduplicate_calibration_rows(
    frame: pd.DataFrame,
    *,
    atol: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate repeated calibration conditions only when results agree.

    Adaptive calibration may repeat a condition in a later endpoint probe. The
    repetition is safe to collapse only if every common scientific result is
    numerically equivalent. Conflicting repeats are hard errors.
    """

    keys = ["seed", "method", "alpha"]
    if frame.empty:
        return frame.copy(), pd.DataFrame(columns=keys + ["n_repeats"])
    missing = [column for column in keys if column not in frame]
    if missing:
        raise ValueError(f"Calibration rows are missing key columns: {missing}")

    kept = []
    repeated = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        first = group.iloc[0]
        if len(group) > 1:
            common = [column for column in SCIENTIFIC_RESULT_COLUMNS if column in group]
            conflicts = []
            for row_index in range(1, len(group)):
                candidate = group.iloc[row_index]
                conflicts.extend(
                    column
                    for column in common
                    if not _equivalent(first[column], candidate[column], atol=atol)
                )
            if conflicts:
                seed, method, alpha = key
                raise ValueError(
                    "Conflicting repeated calibration condition "
                    f"seed={seed}, method={method}, alpha={alpha}: "
                    f"{sorted(set(conflicts))}"
                )
            repeated.append(
                {
                    "seed": int(key[0]),
                    "method": str(key[1]),
                    "alpha": float(key[2]),
                    "n_repeats": len(group),
                }
            )
        kept.append(first)
    return pd.DataFrame(kept).reset_index(drop=True), pd.DataFrame(repeated)


def matched_coverage(
    matched: pd.DataFrame,
    *,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """Summarize target-match completeness for every method and direction."""

    expected = {int(seed) for seed in expected_seeds}
    records = []
    if matched.empty:
        return pd.DataFrame()
    for (method, direction, target), group in matched.groupby(
        ["method", "direction", "target_change"], sort=True
    ):
        complete = group[group["match_complete"].fillna(False)]
        observed = {int(seed) for seed in group["seed"]}
        distances = pd.to_numeric(group["match_distance"], errors="coerce")
        records.append(
            {
                "method": method,
                "direction": int(direction),
                "target_change": float(target),
                "expected_seeds": len(expected),
                "observed_seeds": len(observed & expected),
                "matched_seeds": int(complete["seed"].nunique()),
                "coverage_rate": (
                    float(complete["seed"].nunique()) / len(expected)
                    if expected
                    else math.nan
                ),
                "median_match_distance": float(distances.median()),
                "max_match_distance": float(distances.max()),
            }
        )
    return pd.DataFrame.from_records(records)

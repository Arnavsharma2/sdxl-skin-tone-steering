"""Locked decision functions for the prospective independent-seed replication."""

from __future__ import annotations

import pandas as pd

from src.study.analysis import holm_adjust

DIRECTION_LABELS = {-1: "lighter", 1: "darker"}


def replication_family(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Select and multiplicity-adjust the two locked LPIPS contrasts."""

    selected = contrasts[
        (contrasts["reference_method"] == "stepwise_masked")
        & (contrasts["comparator"] == "stepwise_unmasked")
        & (contrasts["outcome"] == "lpips")
        & (contrasts["target_change"] == 5.0)
        & (contrasts["direction"].isin(DIRECTION_LABELS))
    ].copy()
    selected = selected.sort_values("direction").reset_index(drop=True)
    if len(selected) != 2 or set(selected["direction"].astype(int)) != {-1, 1}:
        raise ValueError(
            "Replication analysis requires exactly the lighter and darker "
            "masked-versus-unmasked LPIPS rows"
        )
    if selected["p_value"].isna().any():
        raise ValueError("Replication-family p-values are missing")
    selected["direction_label"] = selected["direction"].map(DIRECTION_LABELS)
    selected["p_holm_replication_family"] = holm_adjust(
        selected["p_value"].astype(float).tolist()
    )
    return selected


def assess_replication(family: pd.DataFrame, minimum_pairs: int = 12) -> str:
    """Apply the prospective replication decision rule without reinterpretation."""

    if (family["n_pairs"].astype(int) < minimum_pairs).any():
        return "inconclusive_coverage"
    estimates = family["mean_reference_advantage"].astype(float)
    if (estimates <= 0).any():
        return "failure_to_replicate"
    strict = (
        (family["ci_low"].astype(float) > 0).all()
        and (family["p_holm_replication_family"].astype(float) < 0.05).all()
    )
    return "strict_replication" if strict else "directional_replication"

"""Pure helpers for validating the colourimetric target outcome."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from PIL import Image


def apply_rgb_gain(image: Image.Image, gains: list[float]) -> Image.Image:
    """Apply a deterministic per-channel gain perturbation."""

    if len(gains) != 3 or any(float(gain) <= 0 for gain in gains):
        raise ValueError("rgb_gain must contain three positive values")
    array = np.asarray(image.convert("RGB"), dtype=np.float64)
    changed = np.clip(array * np.asarray(gains, dtype=float), 0, 255).astype(np.uint8)
    return Image.fromarray(changed)


def validation_summary(
    pair_rows: pd.DataFrame,
    perturbation_rows: pd.DataFrame,
    gates: Mapping[str, float],
) -> dict:
    """Summarize detection, ordering, separation, and perturbation sensitivity."""

    pair_count = len(pair_rows)
    image_count = 2 * pair_count
    detected = int(pair_rows[["light_detected", "dark_detected"]].sum().sum())
    complete_pairs = pair_rows[pair_rows["pair_complete"].fillna(False)]
    shifts = pd.to_numeric(perturbation_rows.get("abs_ita_shift"), errors="coerce")
    shifts = shifts[np.isfinite(shifts)]
    detection_rate = detected / image_count if image_count else 0.0
    pair_order_accuracy = (
        float(complete_pairs["ordered_light_above_dark"].mean())
        if len(complete_pairs)
        else 0.0
    )
    median_gap = (
        float(complete_pairs["ita_gap"].median()) if len(complete_pairs) else float("nan")
    )
    median_shift = float(shifts.median()) if len(shifts) else float("nan")
    p95_shift = float(shifts.quantile(0.95)) if len(shifts) else float("nan")
    checks = {
        "detection_rate": detection_rate >= float(gates["min_detection_rate"]),
        "pair_order_accuracy": pair_order_accuracy
        >= float(gates["min_pair_order_accuracy"]),
        "median_pair_gap_ita": median_gap
        >= float(gates["min_median_pair_gap_ita"]),
        "median_abs_ita_shift": median_shift
        <= float(gates["max_median_abs_ita_shift"]),
        "p95_abs_ita_shift": p95_shift <= float(gates["max_p95_abs_ita_shift"]),
    }
    return {
        "pair_count": pair_count,
        "complete_pair_count": len(complete_pairs),
        "detection_rate": detection_rate,
        "pair_order_accuracy": pair_order_accuracy,
        "median_pair_gap_ita": median_gap,
        "median_abs_ita_shift": median_shift,
        "p95_abs_ita_shift": p95_shift,
        "gates": dict(gates),
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
    }

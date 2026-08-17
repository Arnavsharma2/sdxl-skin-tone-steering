import numpy as np
import pandas as pd
from PIL import Image

from src.study.measurement_validation import apply_rgb_gain, validation_summary

GATES = {
    "min_detection_rate": 0.95,
    "min_pair_order_accuracy": 0.90,
    "min_median_pair_gap_ita": 8.0,
    "max_median_abs_ita_shift": 2.0,
    "max_p95_abs_ita_shift": 5.0,
}


def test_rgb_gain_is_deterministic_and_clipped():
    image = Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8))
    changed = np.asarray(apply_rgb_gain(image, [2.0, 1.0, 0.5]))
    assert changed[0, 0].tolist() == [255, 200, 100]


def test_validation_summary_passes_only_when_every_gate_passes():
    pairs = pd.DataFrame(
        [
            {
                "light_detected": True,
                "dark_detected": True,
                "pair_complete": True,
                "ordered_light_above_dark": True,
                "ita_gap": 12.0,
            }
            for _ in range(10)
        ]
    )
    perturbations = pd.DataFrame({"abs_ita_shift": [0.2, 0.5, 1.0, 2.0]})
    report = validation_summary(pairs, perturbations, GATES)
    assert report["passed"]
    pairs.loc[0:1, "ordered_light_above_dark"] = False
    failed = validation_summary(pairs, perturbations, GATES)
    assert not failed["passed"]
    assert not failed["checks"]["pair_order_accuracy"]

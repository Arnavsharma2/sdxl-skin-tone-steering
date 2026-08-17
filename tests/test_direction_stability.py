from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_direction_stability import summarize_records


def test_summarize_direction_stability_quantiles() -> None:
    frame = pd.DataFrame(
        {
            "pair_count": [8, 8, 8, 8],
            "raw_cosine": [0.1, 0.2, 0.3, 0.4],
            "masked_cosine": [0.2, 0.3, 0.4, 0.5],
            "norm_agreement": [0.7, 0.8, 0.9, 1.0],
        }
    )

    summary = summarize_records(frame)

    assert set(summary["metric"]) == {
        "raw_cosine",
        "masked_cosine",
        "norm_agreement",
    }
    raw = summary[summary["metric"] == "raw_cosine"].iloc[0]
    assert raw["n_resamples"] == 4
    assert np.isclose(raw["median"], 0.25)
    assert raw["q025"] < raw["median"] < raw["q975"]

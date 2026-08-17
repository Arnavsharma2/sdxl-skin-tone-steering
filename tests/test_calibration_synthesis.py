import pandas as pd
import pytest

from src.study.calibration import deduplicate_calibration_rows, matched_coverage


def test_deduplicate_accepts_equivalent_repeated_measurements():
    frame = pd.DataFrame(
        [
            {
                "seed": 1,
                "method": "stepwise_masked",
                "alpha": -0.75,
                "generation_complete": True,
                "evaluation_complete": True,
                "skin_tone_change": -4.0,
            },
            {
                "seed": 1,
                "method": "stepwise_masked",
                "alpha": -0.75,
                "generation_complete": True,
                "evaluation_complete": True,
                "skin_tone_change": -4.0 + 1e-10,
            },
        ]
    )

    rows, repeats = deduplicate_calibration_rows(frame)

    assert len(rows) == 1
    assert repeats.to_dict("records") == [
        {
            "seed": 1,
            "method": "stepwise_masked",
            "alpha": -0.75,
            "n_repeats": 2,
        }
    ]


def test_deduplicate_rejects_conflicting_repeated_measurements():
    frame = pd.DataFrame(
        [
            {"seed": 1, "method": "prompt_only", "alpha": 0.1, "lpips": 0.1},
            {"seed": 1, "method": "prompt_only", "alpha": 0.1, "lpips": 0.2},
        ]
    )

    with pytest.raises(ValueError, match="Conflicting repeated calibration"):
        deduplicate_calibration_rows(frame)


def test_matched_coverage_counts_expected_seed_matches():
    matched = pd.DataFrame(
        [
            {
                "seed": 1,
                "method": "stepwise_masked",
                "direction": -1,
                "target_change": 5.0,
                "match_complete": True,
                "match_distance": 1.0,
            },
            {
                "seed": 2,
                "method": "stepwise_masked",
                "direction": -1,
                "target_change": 5.0,
                "match_complete": False,
                "match_distance": 4.0,
            },
        ]
    )

    coverage = matched_coverage(matched, expected_seeds=[1, 2])

    assert coverage.loc[0, "observed_seeds"] == 2
    assert coverage.loc[0, "matched_seeds"] == 1
    assert coverage.loc[0, "coverage_rate"] == 0.5

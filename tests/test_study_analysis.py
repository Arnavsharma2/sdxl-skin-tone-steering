import numpy as np
import pandas as pd

from src.study.analysis import (
    holm_adjust,
    match_at_target_change,
    matched_coverage_summary,
    paired_method_contrasts,
)


def synthetic_results():
    rows = []
    for seed in (1, 2, 3):
        for method, offset in (("stepwise_masked", 0.05), ("prompt_only", 0.0)):
            for alpha, change in ((-1.0, -5.0), (-2.0, -10.0), (1.0, 5.0), (2.0, 10.0)):
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "alpha": alpha,
                        "skin_tone_change": change + np.sign(alpha) * (seed - 2) * 0.1,
                        "face_similarity": 0.8 + offset,
                        "lpips": 0.2 - offset,
                        "background_ssim": 0.8 + offset,
                        "total_pose_diff": 2.0 - offset,
                    }
                )
    return pd.DataFrame(rows)


def test_match_at_target_change_keeps_direction_and_nearest_observation():
    matched = match_at_target_change(synthetic_results(), [9.5], tolerance=1.0)
    assert matched["match_complete"].all()
    assert set(matched["direction"]) == {-1, 1}
    assert set(matched["alpha"].abs()) == {2.0}


def test_paired_contrasts_report_reference_advantage():
    matched = match_at_target_change(synthetic_results(), [10.0], tolerance=1.0)
    contrasts = paired_method_contrasts(
        matched,
        reference_method="stepwise_masked",
        resamples=2_000,
    )
    face = contrasts[contrasts["outcome"] == "face_similarity"]
    lpips = contrasts[contrasts["outcome"] == "lpips"]
    assert (face["n_pairs"] == 3).all()
    assert (face["mean_reference_advantage"] > 0).all()
    assert (lpips["mean_reference_advantage"] > 0).all()


def test_matched_coverage_summary_reports_achieved_target():
    matched = match_at_target_change(synthetic_results(), [10.0], tolerance=1.0)
    summary = matched_coverage_summary(matched)

    assert len(summary) == 4
    assert (summary["attempted_seeds"] == 3).all()
    assert (summary["complete_seeds"] == 3).all()
    assert (summary["coverage_rate"] == 1.0).all()
    assert np.allclose(summary["mean_achieved_change"], 10.0)


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adjusted[:3], [0.03, 0.06, 0.06])
    assert np.isnan(adjusted[3])

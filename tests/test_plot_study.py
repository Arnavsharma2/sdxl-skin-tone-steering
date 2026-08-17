from __future__ import annotations

import pandas as pd

from scripts.plot_study import plot_completion, plot_dose_response, plot_matched_contrasts


def test_publication_plots_are_written(tmp_path) -> None:
    descriptive = pd.DataFrame(
        [
            {
                "method": method,
                "alpha": alpha,
                "metric": "skin_tone_change",
                "mean": 5.0 * alpha,
                "ci_low": 5.0 * alpha - 0.5,
                "ci_high": 5.0 * alpha + 0.5,
            }
            for method in (
                "prompt_only",
                "posthoc_latent",
                "stepwise_unmasked",
                "stepwise_masked",
            )
            for alpha in (-1.0, 0.0, 1.0)
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "comparator": comparator,
                "direction": direction,
                "outcome": outcome,
                "mean_reference_advantage": 0.1,
                "ci_low": 0.05,
                "ci_high": 0.15,
            }
            for comparator in ("prompt_only", "stepwise_unmasked")
            for direction in (-1, 1)
            for outcome in (
                "face_similarity",
                "lpips",
                "background_ssim",
                "total_pose_diff",
            )
        ]
    )
    missingness = pd.DataFrame(
        [
            {
                "method": method,
                "attempted": 30,
                "generated": 30,
                "evaluation_complete": 29,
            }
            for method in (
                "prompt_only",
                "posthoc_latent",
                "stepwise_unmasked",
                "stepwise_masked",
            )
        ]
    )

    plot_dose_response(descriptive, tmp_path)
    plot_matched_contrasts(contrasts, tmp_path)
    plot_completion(missingness, tmp_path)

    for stem in (
        "confirmatory_dose_response",
        "confirmatory_matched_contrasts",
        "confirmatory_completion",
    ):
        assert (tmp_path / f"{stem}.pdf").stat().st_size > 0
        assert (tmp_path / f"{stem}.png").stat().st_size > 0

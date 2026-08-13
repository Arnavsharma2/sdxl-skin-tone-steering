import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.summarize_results import (
    _normalise_planned_result,
    run_analysis,
    sha256_file,
)
from src.analysis.statistics import (
    analyze_comparisons,
    holm_adjust,
    matched_seed_contrast,
    paired_sign_flip_pvalue,
    seed_cluster_bootstrap_ci,
)
from src.metrics.protocol import protocol_record
from src.utils.config import (
    AnalysisComparisonConfig,
    ExperimentConfig,
)


def comparison(metric="face_similarity", favorable_direction="higher"):
    return AnalysisComparisonConfig(
        id="synthetic",
        hypothesis="H-test",
        role="primary",
        method_a="stepwise_masked",
        method_b="prompt_only",
        metric=metric,
        favorable_direction=favorable_direction,
    )


def method_rows(
    seed,
    method,
    *,
    target_scale=2.0,
    effect=0.0,
    metric="face_similarity",
    tie_offsets=None,
):
    tie_offsets = tie_offsets or {}
    rows = []
    for alpha in (-1.5, -0.75, 0.0, 0.75, 1.5):
        target = -target_scale * alpha
        value = 10.0 + 0.5 * abs(target) + effect + tie_offsets.get(alpha, 0.0)
        rows.append(
            {
                "seed": seed,
                "method": method,
                "alpha": alpha,
                "skin_tone_change": target,
                metric: value,
                "analysis_row_valid": True,
            }
        )
    return rows


def matched(frame, *, comp=None):
    return matched_seed_contrast(
        frame,
        seed=int(frame["seed"].iloc[0]),
        comparison=comp or comparison(),
        expected_alphas=(-1.5, -0.75, 0.0, 0.75, 1.5),
        minimum_abs_target_change=2.0,
        grid_points=101,
        minimum_unique_points=2,
    )


def test_matched_change_recovers_exact_known_effect_with_different_alpha_response():
    frame = pd.DataFrame(
        method_rows(1, "stepwise_masked", target_scale=2.0, effect=0.25)
        + method_rows(1, "prompt_only", target_scale=8 / 3, effect=0.0)
    )

    result = matched(frame)

    assert result.computable
    assert result.support_low == pytest.approx(2.0)
    assert result.support_high == pytest.approx(3.0)
    assert result.effect_a_minus_b == pytest.approx(0.25)
    assert result.favorable_effect == pytest.approx(0.25)


def test_ties_are_averaged_within_seed_method_before_interpolation():
    frame = pd.DataFrame(
        method_rows(
            1,
            "stepwise_masked",
            effect=0.5,
            tie_offsets={-0.75: 0.2, 0.75: -0.2, -1.5: 0.4, 1.5: -0.4},
        )
        + method_rows(1, "prompt_only")
    )

    result = matched(frame)

    assert result.computable
    assert result.effect_a_minus_b == pytest.approx(0.5)


def test_touching_support_boundary_is_not_a_positive_width_estimand():
    frame = pd.DataFrame(
        method_rows(1, "stepwise_masked", target_scale=4 / 3)
        + method_rows(1, "prompt_only", target_scale=8 / 3)
    )

    result = matched(frame)

    assert not result.computable
    assert result.reason == "no_positive_width_common_support"


@pytest.mark.parametrize(
    ("mutation", "reason_fragment"),
    [
        (lambda df: df.drop(df.index[0]), "missing_alpha"),
        (lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True), "duplicate_alpha"),
        (
            lambda df: df.assign(
                skin_tone_change=np.where(df["alpha"].eq(0.75), np.nan, df["skin_tone_change"])
            ),
            "nonfinite_target_change",
        ),
        (
            lambda df: df.assign(
                face_similarity=np.where(df["alpha"].eq(0.75), np.inf, df["face_similarity"])
            ),
            "nonfinite_metric",
        ),
        (
            lambda df: df.assign(
                skin_tone_change=np.where(df["alpha"].eq(0.75), 2.0, df["skin_tone_change"])
            ),
            "nonmonotonic_target_sweep",
        ),
    ],
)
def test_invalid_sweep_inputs_fail_closed(mutation, reason_fragment):
    method_a = pd.DataFrame(method_rows(1, "stepwise_masked"))
    frame = pd.concat(
        [mutation(method_a), pd.DataFrame(method_rows(1, "prompt_only"))],
        ignore_index=True,
    )

    result = matched(frame)

    assert not result.computable
    assert reason_fragment in result.reason


def test_missing_method_fails_closed_instead_of_unpaired_comparison():
    frame = pd.DataFrame(method_rows(1, "stepwise_masked"))

    result = matched(frame)

    assert not result.computable
    assert result.reason.startswith("prompt_only:missing_alpha")


def test_invalid_nonzero_row_prevents_matched_estimate():
    frame = pd.DataFrame(method_rows(1, "stepwise_masked") + method_rows(1, "prompt_only"))
    frame.loc[
        frame["method"].eq("stepwise_masked") & frame["alpha"].eq(0.75),
        "analysis_row_valid",
    ] = False

    result = matched(frame)

    assert not result.computable
    assert result.reason == "stepwise_masked:invalid_evaluation_row"


def test_seed_cluster_bootstrap_is_repeatable_and_rejects_duplicate_clusters():
    effects = [(11, -1.0), (12, 1.0), (13, 3.0), (14, 5.0)]

    first = seed_cluster_bootstrap_ci(
        effects, resamples=1000, confidence=0.95, rng_seed=7, label="x"
    )
    second = seed_cluster_bootstrap_ci(
        list(reversed(effects)), resamples=1000, confidence=0.95, rng_seed=7, label="x"
    )

    assert first == second
    with pytest.raises(ValueError, match="unique seed"):
        seed_cluster_bootstrap_ci(
            [(11, 1.0), (11, 2.0)],
            resamples=10,
            confidence=0.95,
            rng_seed=7,
            label="x",
        )


def test_paired_sign_flip_is_repeatable_and_uses_seed_effects():
    first = paired_sign_flip_pvalue(
        [0.1, 0.2, 0.3, 0.4], resamples=1000, rng_seed=99, label="paired"
    )
    second = paired_sign_flip_pvalue(
        [0.1, 0.2, 0.3, 0.4], resamples=1000, rng_seed=99, label="paired"
    )

    assert first == second
    assert 0 < first <= 1


def test_holm_is_stable_for_ties_and_keeps_invalid_tests_in_family_size():
    adjusted = holm_adjust(
        [
            {"comparison_id": "first", "p_value": 0.01},
            {"comparison_id": "second", "p_value": 0.01},
            {"comparison_id": "missing", "p_value": None},
            {"comparison_id": "invalid", "p_value": 1.2},
        ]
    )

    assert adjusted[0]["holm_adjusted_p"] == pytest.approx(0.04)
    assert adjusted[1]["holm_adjusted_p"] == pytest.approx(0.04)
    assert adjusted[2]["holm_adjusted_p"] is None
    assert adjusted[3]["holm_adjusted_p"] is None
    assert adjusted[2]["holm_reason"] == "missing_or_invalid_p_value"


def test_analyze_comparisons_pairs_within_seed_and_requires_every_seed():
    config = ExperimentConfig.from_yaml("configs/full_study.yaml")
    config.evaluation.seeds = [1, 2]
    config.evaluation.bootstrap.resamples = 200
    config.analysis.randomization_resamples = 200
    config.analysis.comparisons = [comparison()]
    frame = pd.DataFrame(
        method_rows(1, "stepwise_masked", effect=1.0)
        + method_rows(1, "prompt_only")
        + method_rows(2, "stepwise_masked", effect=3.0)
        + method_rows(2, "prompt_only")
    )

    seed_results, aggregate = analyze_comparisons(frame, config)

    assert seed_results["effect_a_minus_b"].tolist() == pytest.approx([1.0, 3.0])
    assert aggregate.loc[0, "estimate_a_minus_b"] == pytest.approx(2.0)
    assert aggregate.loc[0, "included_seed_count"] == 2

    incomplete = frame.loc[~((frame["seed"] == 2) & (frame["method"] == "prompt_only"))]
    _, failed = analyze_comparisons(incomplete, config)
    assert not bool(failed.loc[0, "confirmatory_computable"])
    assert pd.isna(failed.loc[0, "estimate_a_minus_b"])
    assert failed.loc[0, "exploratory_valid_seed_estimate"] == pytest.approx(1.0)


def test_failure_normalisation_reports_detector_generation_integrity_and_missingness():
    config = ExperimentConfig.from_yaml("configs/full_study.yaml")
    planned = {
        "row_id": "row",
        "seed": 1000,
        "method": "stepwise_masked",
        "alpha": 0.75,
        "status": "planned",
    }
    failed = _normalise_planned_result(
        planned,
        {
            **planned,
            "status": "failed",
            "face_detection_failed": True,
            "failure": {"stage": "row_generation", "exception_type": "RuntimeError"},
        },
        config,
    )
    reasons = json.loads(failed["failure_reasons"])

    assert failed["face_detection_failure"]
    assert failed["generation_failure"]
    assert not failed["analysis_row_valid"]
    assert "face_detection_failure" in reasons
    assert "metric_missing:face_similarity" in reasons

    complete_without_hashes = _normalise_planned_result(
        planned,
        {
            **planned,
            "status": "complete",
            "evaluation_complete": True,
            **{
                metric: (True if metric == "target_direction_correct" else 1.0)
                for metric in config.evaluation.required_metrics
            },
        },
        config,
    )
    assert complete_without_hashes["file_integrity_failure"]
    assert not complete_without_hashes["analysis_row_valid"]


def test_lower_is_better_contrast_records_favorable_sign():
    frame = pd.DataFrame(
        method_rows(1, "stepwise_masked", effect=-0.2, metric="lpips")
        + method_rows(1, "prompt_only", metric="lpips")
    )

    result = matched(frame, comp=comparison("lpips", "lower"))

    assert result.effect_a_minus_b == pytest.approx(-0.2)
    assert result.favorable_effect == pytest.approx(0.2)


def test_analysis_bundle_preserves_matrix_and_machine_readable_failures(tmp_path):
    document = yaml.safe_load(open("configs/full_study.yaml", encoding="utf-8"))
    document["evaluation"]["seeds"] = [1000, 1001]
    document["evaluation"]["matrix"].update(
        {
            "expected_seeds": 2,
            "expected_rows": 40,
            "expected_nonzero_alpha_rows": 32,
        }
    )
    config_path = tmp_path / "full_study.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config_hash = sha256_file(config_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = []
    by_seed = {1000: [], 1001: []}
    for seed in (1000, 1001):
        seed_dir = run_dir / "seeds" / str(seed)
        seed_dir.mkdir(parents=True)
        base_path = seed_dir / "base.png"
        base_path.write_bytes(f"base:{seed}".encode())
        base_hash = sha256_file(base_path)
        for method_index, method in enumerate(document["evaluation"]["methods"]):
            for alpha in document["evaluation"]["alphas"]:
                row_id = f"{seed}:{method}:{alpha}"
                relative_path = f"seeds/{seed}/images/{method}/{alpha}.png"
                image_path = run_dir / relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(f"{seed}:{method}:{alpha}".encode())
                planned = {
                    "row_id": row_id,
                    "seed": seed,
                    "method": method,
                    "alpha": alpha,
                    "status": "complete",
                    "output_path": relative_path,
                    "image_sha256": sha256_file(image_path),
                    "base_image_sha256": base_hash,
                }
                result = {
                    **planned,
                    "evaluation_complete": True,
                    "skin_tone_change": -2.0 * alpha,
                    "target_direction_correct": alpha != 0,
                    "face_similarity": 0.9 + method_index * 0.001,
                    "lpips": 0.1 - method_index * 0.001,
                    "background_ssim": 0.8 + method_index * 0.001,
                    "total_pose_diff": 1.0,
                }
                if seed == 1001 and method == "stepwise_masked" and alpha == 0.75:
                    result.update(
                        {
                            "evaluation_complete": False,
                            "face_similarity": None,
                            "face_detection_failed": True,
                            "failure_reasons": ["face_detection_failure"],
                        }
                    )
                rows.append(planned)
                by_seed[seed].append(result)
    manifest = {
        "study_id": document["study_id"],
        "run_id": "synthetic-run",
        "config": {"sha256": config_hash},
        "metric_protocol": {"sha256": protocol_record()["sha256"]},
        "provenance": {"git": {"commit": "synthetic-source-commit"}},
        "rows": rows,
    }
    (run_dir / "study_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for seed, results in by_seed.items():
        seed_dir = run_dir / "seeds" / str(seed)
        (seed_dir / "metadata.json").write_text(
            json.dumps({"seed": seed, "config_sha256": config_hash, "results": results}),
            encoding="utf-8",
        )
    (run_dir / "seeds/1000/images/prompt_only/-0.75.png").write_bytes(b"tampered")

    output = tmp_path / "summary"
    audit = run_analysis(run_dir, output, config_path)

    assert audit["observed_long_rows"] == 40
    assert audit["face_detection_failure_rows"] == 1
    assert audit["file_integrity_failure_rows"] == 1
    assert not audit["confirmatory_analysis_computable"]
    assert set(audit["outputs"].values()) == {
        "results_long.csv",
        "aggregate_metrics.csv",
        "seed_matched_contrasts.csv",
        "confirmatory_contrasts.csv",
        "failure_counts.csv",
    }
    for filename in (*audit["outputs"].values(), "audit.json"):
        assert (output / filename).is_file()
    failures = pd.read_csv(output / "failure_counts.csv")
    detector = failures.loc[failures["failure_reason"].eq("face_detection_failure")]
    assert detector[["method", "alpha", "count"]].to_dict(orient="records") == [
        {"method": "stepwise_masked", "alpha": 0.75, "count": 1}
    ]
    assert failures["failure_reason"].eq("file_integrity_failure:image_mismatch").sum() == 1
    long_df = pd.read_csv(output / "results_long.csv")
    assert long_df["config_sha256"].nunique() == 1
    assert long_df["protocol_sha256"].nunique() == 1
    assert long_df["source_commit"].unique().tolist() == ["synthetic-source-commit"]

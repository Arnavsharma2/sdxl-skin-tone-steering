import json
import math

from scripts.summarize_results import (
    bootstrap_mean_ci,
    build_audit,
    load_runs,
    run_legacy_summary,
)


def test_bootstrap_interval_contains_sample_mean():
    values = [0.7, 0.8, 0.9, 1.0]
    low, high = bootstrap_mean_ci(values, resamples=2_000, seed=7)
    assert low < sum(values) / len(values) < high


def test_bootstrap_requires_two_finite_values():
    low, high = bootstrap_mean_ci([0.5, math.nan])
    assert math.isnan(low)
    assert math.isnan(high)


def test_load_runs_maps_legacy_success_field(tmp_path):
    metadata = {
        "seed": 17,
        "results": [{"alpha": 0.5, "is_disentangled": True}],
    }
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    frame = load_runs(tmp_path)

    assert bool(frame.loc[0, "counterfactual_success"])
    assert frame.loc[0, "counterfactual_success_source"] == "legacy_is_disentangled"


def test_audit_separates_unreported_success_from_unsuccessful_rows(tmp_path):
    metadata = {
        "seed": 23,
        "results": [
            {"alpha": 0.0, "counterfactual_success": False},
            {"alpha": 0.5},
        ],
    }
    run_dir = tmp_path / "mixed"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    audit = build_audit(load_runs(tmp_path), resamples=100)

    assert audit["successful_rows"] == 0
    assert audit["unsuccessful_rows"] == 1
    assert audit["unreported_success_rows"] == 1


def test_legacy_summary_is_explicitly_exploratory(tmp_path):
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "seed": 999,
                "results": [
                    {
                        "method": "stepwise_masked",
                        "alpha": 0.75,
                        "evaluation_complete": True,
                        "face_similarity": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = run_legacy_summary(run_dir, tmp_path / "summary", resamples=100)

    assert audit["summary_role"] == "pilot_exploratory_descriptive"
    assert audit["confirmatory_analysis"] is False
    assert (tmp_path / "summary" / "summary.csv").is_file()

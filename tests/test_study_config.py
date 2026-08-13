from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from src.metrics.evaluator import EvaluationThresholds
from src.metrics.protocol import PROTOCOL_ID, load_protocol, protocol_record
from src.metrics.skin_tone_metrics import SkinToneMetrics
from src.metrics.structural_metrics import StructuralPreservationMetrics
from src.utils.config import ExperimentConfig

REPOSITORY_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("filename", "expected_rows", "expected_nonzero_rows"),
    [
        ("pilot.yaml", 5, 4),
        ("full_study.yaml", 600, 480),
    ],
)
def test_study_configs_load_and_declare_complete_matrix(
    filename, expected_rows, expected_nonzero_rows
):
    config = ExperimentConfig.from_yaml(REPOSITORY_ROOT / "configs" / filename)

    assert config.model.name == "stabilityai/stable-diffusion-xl-base-1.0"
    assert config.evaluation.matrix.expected_rows == expected_rows
    assert config.evaluation.matrix.expected_nonzero_alpha_rows == expected_nonzero_rows
    assert "total_pose_diff" in config.evaluation.required_metrics
    assert "pose_difference" not in config.evaluation.required_metrics
    assert config.evaluation.protocol_id == PROTOCOL_ID


def test_confirmatory_training_and_evaluation_seeds_are_disjoint():
    config = ExperimentConfig.from_yaml(REPOSITORY_ROOT / "configs" / "full_study.yaml")

    assert len(config.direction.training_seeds()) == 64
    assert len(config.direction.held_out_seeds()) == 32
    assert len(set(config.direction.training_seeds())) == 64
    assert len(set(config.direction.held_out_seeds())) == 32
    assert set(config.direction.training_seeds()).isdisjoint(
        config.direction.held_out_seeds()
    )
    assert set(config.direction.training_seeds()).isdisjoint(config.evaluation.seeds)
    assert set(config.direction.held_out_seeds()).isdisjoint(config.evaluation.seeds)


def test_confirmatory_analysis_settings_are_frozen_before_collection():
    config = ExperimentConfig.from_yaml(REPOSITORY_ROOT / "configs" / "full_study.yaml")

    assert config.analysis.version == "tmlr_statistical_analysis_v1"
    assert config.analysis.status == "frozen"
    assert config.analysis.rng_seed == 20260813
    assert config.evaluation.bootstrap.resamples == 10_000
    assert config.evaluation.bootstrap.cluster_unit == "seed"
    assert config.analysis.randomization_resamples == 10_000
    assert config.analysis.matched_change.grid_points == 101
    assert config.analysis.matched_change.extrapolation == "prohibited"
    assert [comparison.id for comparison in config.analysis.comparisons] == [
        "h2_identity_masked_vs_prompt",
        "h3_background_masked_vs_unmasked",
        "h4_identity_masked_vs_posthoc",
        "h4_lpips_masked_vs_posthoc",
    ]
    assert [comparison.role for comparison in config.analysis.comparisons].count(
        "primary"
    ) == 1


def test_confirmatory_thresholds_match_evaluator_defaults():
    config = ExperimentConfig.from_yaml(REPOSITORY_ROOT / "configs" / "full_study.yaml")

    assert asdict(config.thresholds) == asdict(EvaluationThresholds())


def test_frozen_protocol_matches_configuration_and_records_actual_checksum():
    config = ExperimentConfig.from_yaml(REPOSITORY_ROOT / "configs" / "full_study.yaml")
    protocol = load_protocol()
    record = protocol_record()

    assert protocol["status"] == "frozen"
    assert protocol["protocol_id"] == config.evaluation.protocol_id
    assert protocol["thresholds"] == asdict(config.thresholds)
    assert set(protocol["required_pair_metrics"]) == set(
        config.evaluation.required_metrics
    )
    assert protocol["metrics"]["monotonicity"]["expected_alphas"] == (
        config.evaluation.alphas
    )
    assert len(record["sha256"]) == 64
    assert record["size_bytes"] > 0


def test_frozen_mask_parameters_match_metric_code():
    protocol = load_protocol()["metrics"]
    skin = SkinToneMetrics(landmark_backend=object())

    assert skin.min_skin_pixels == protocol["target_response"][
        "skin_minimum_pixels_before_and_after_trim"
    ]
    assert skin.min_reference_pixels == protocol["target_response"][
        "reference_minimum_pixels"
    ]
    assert list(skin.trim_quantiles) == protocol["target_response"][
        "lstar_trim_quantiles"
    ]
    assert StructuralPreservationMetrics.background_erosion_kernel == 7
    assert StructuralPreservationMetrics.minimum_background_pixels == protocol[
        "background_ssim"
    ]["minimum_background_pixels"]


def test_confirmatory_config_round_trip(tmp_path):
    config = ExperimentConfig.from_yaml(REPOSITORY_ROOT / "configs" / "full_study.yaml")
    round_trip_path = tmp_path / "round_trip.yaml"

    config.to_yaml(round_trip_path)
    reloaded = ExperimentConfig.from_yaml(round_trip_path)

    assert reloaded.study_id == config.study_id
    assert reloaded.evaluation.matrix.expected_rows == 600


def test_loading_rejects_incorrect_matrix_count(tmp_path):
    source = REPOSITORY_ROOT / "configs" / "full_study.yaml"
    document = yaml.safe_load(source.read_text())
    document["evaluation"]["matrix"]["expected_rows"] = 599
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="expected_rows is 599; expected 600"):
        ExperimentConfig.from_yaml(invalid_path)


def test_loading_rejects_training_evaluation_seed_overlap(tmp_path):
    source = REPOSITORY_ROOT / "configs" / "full_study.yaml"
    document = deepcopy(yaml.safe_load(source.read_text()))
    document["evaluation"]["seeds"][0] = 42
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="training and evaluation seeds overlap: 42"):
        ExperimentConfig.from_yaml(invalid_path)


def test_matching_protocol_id_cannot_hide_threshold_drift(tmp_path):
    source = REPOSITORY_ROOT / "configs" / "full_study.yaml"
    document = yaml.safe_load(source.read_text())
    document["thresholds"]["face_similarity"] = 0.80
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="thresholds do not match"):
        ExperimentConfig.from_yaml(invalid_path)


def test_matching_protocol_id_cannot_hide_alpha_grid_drift(tmp_path):
    source = REPOSITORY_ROOT / "configs" / "full_study.yaml"
    document = yaml.safe_load(source.read_text())
    document["evaluation"]["alphas"][-1] = 1.25
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="evaluation.alphas do not match"):
        ExperimentConfig.from_yaml(invalid_path)


def test_analysis_definition_rejects_bootstrap_and_support_drift(tmp_path):
    source = REPOSITORY_ROOT / "configs" / "full_study.yaml"
    document = yaml.safe_load(source.read_text())
    document["analysis"]["randomization_resamples"] = 9_999
    invalid_path = tmp_path / "invalid_resamples.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="randomization_resamples"):
        ExperimentConfig.from_yaml(invalid_path)

    document = yaml.safe_load(source.read_text())
    document["analysis"]["matched_change"]["extrapolation"] = "linear"
    invalid_path = tmp_path / "invalid_extrapolation.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="matched_change"):
        ExperimentConfig.from_yaml(invalid_path)

    document = yaml.safe_load(source.read_text())
    document["analysis"]["rng_seed"] = 7
    invalid_path = tmp_path / "invalid_rng.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="frozen seed"):
        ExperimentConfig.from_yaml(invalid_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("empty_id", "ids"),
        ("duplicate_id", "ids"),
        ("no_primary", "one primary"),
        ("unknown_method", "unknown method_a"),
        ("identical_methods", "different methods"),
        ("unsupported_metric", "unsupported metric"),
        ("invalid_direction", "invalid favorable_direction"),
    ],
)
def test_analysis_definition_rejects_comparison_family_drift(tmp_path, mutation, message):
    source = REPOSITORY_ROOT / "configs" / "full_study.yaml"
    document = yaml.safe_load(source.read_text())
    comparisons = document["analysis"]["comparisons"]
    if mutation == "empty_id":
        comparisons[0]["id"] = ""
    elif mutation == "duplicate_id":
        comparisons[1]["id"] = comparisons[0]["id"]
    elif mutation == "no_primary":
        comparisons[0]["role"] = "secondary"
    elif mutation == "unknown_method":
        comparisons[0]["method_a"] = "unknown"
    elif mutation == "identical_methods":
        comparisons[0]["method_b"] = comparisons[0]["method_a"]
    elif mutation == "unsupported_metric":
        comparisons[0]["metric"] = "overall_score"
    elif mutation == "invalid_direction":
        comparisons[0]["favorable_direction"] = "sideways"
    invalid_path = tmp_path / f"{mutation}.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_yaml(invalid_path)


def test_nonpilot_cannot_omit_confirmatory_comparisons(tmp_path):
    source = REPOSITORY_ROOT / "configs" / "full_study.yaml"
    document = yaml.safe_load(source.read_text())
    document["analysis"]["comparisons"] = []
    invalid_path = tmp_path / "missing_comparisons.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="comparisons must not be empty"):
        ExperimentConfig.from_yaml(invalid_path)


def test_protocol_loader_rejects_matching_id_with_field_drift(tmp_path):
    document = load_protocol()
    document["runtime"]["face_landmarker_delegate"] = "GPU"
    invalid_path = tmp_path / "evaluation_protocol.yaml"
    invalid_path.write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="does not match metric code"):
        load_protocol(invalid_path)

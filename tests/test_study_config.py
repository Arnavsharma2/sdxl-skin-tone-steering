from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from src.metrics.evaluator import EvaluationThresholds
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


def test_confirmatory_thresholds_match_evaluator_defaults():
    config = ExperimentConfig.from_yaml(REPOSITORY_ROOT / "configs" / "full_study.yaml")

    assert asdict(config.thresholds) == asdict(EvaluationThresholds())


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

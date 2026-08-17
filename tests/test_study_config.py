from pathlib import Path

import pytest
import yaml

from src.study.config import StudyConfigError, load_study_config

ROOT = Path(__file__).resolve().parents[1]


def test_full_study_config_is_valid_and_still_planned():
    config = load_study_config(ROOT / "configs/full_study.yaml")
    assert len(config.seeds) == 30
    assert set(config.methods) == {
        "prompt_only",
        "posthoc_latent",
        "stepwise_unmasked",
        "stepwise_masked",
    }
    assert config.prompt_for("prompt_only", -2.0) != config.prompt_for(
        "stepwise_masked", -2.0
    )
    with pytest.raises(StudyConfigError, match="preregistered"):
        config.assert_confirmatory_ready()


def test_config_fingerprint_is_stable():
    first = load_study_config(ROOT / "configs/full_study.yaml")
    second = load_study_config(ROOT / "configs/full_study.yaml")
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 16


def test_replication_config_freezes_disjoint_data_and_evaluation_seeds():
    config = load_study_config(ROOT / "configs/replication_study.yaml")
    assert config.status == "planned"
    assert config.fingerprint == "5057ea79d01525ba"
    assert set(config.data["seed_schedule"]).isdisjoint(config.seeds)


def test_config_rejects_expanded_direction_seed_overlap(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/replication_study.yaml").read_text())
    raw["data"]["seed_schedule"] = [600000]
    path = tmp_path / "overlap.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(StudyConfigError, match="must be disjoint"):
        load_study_config(path)


def test_method_specific_alpha_grids_are_supported(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/full_study.yaml").read_text())
    raw["evaluation"].pop("alphas", None)
    raw["evaluation"]["method_alphas"] = {
        "prompt_only": [-1.5, 0.0, 0.25],
        "posthoc_latent": [-1.25, 0.0, 0.5],
        "stepwise_unmasked": [-0.5, 0.0, 0.25],
        "stepwise_masked": [-0.5, 0.0, 0.25],
    }
    raw["prompts"]["prompt_only_levels"] = {
        "-1.5": "very light skin tone",
        "0.25": "medium-dark skin tone",
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_study_config(path)
    assert config.alphas_for("prompt_only") == (-1.5, 0.0, 0.25)
    assert config.alphas_for("stepwise_masked") == (-0.5, 0.0, 0.25)
    assert set(config.alphas) == {-1.5, -1.25, -0.5, 0.0, 0.25, 0.5}


def test_analysis_can_prespecify_a_feasibility_only_method(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/full_study.yaml").read_text())
    raw["analysis"]["matched_change_methods"] = [
        "prompt_only",
        "stepwise_unmasked",
        "stepwise_masked",
    ]
    raw["analysis"]["feasibility_only_methods"] = ["posthoc_latent"]
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_study_config(path)

    assert config.matched_change_methods == (
        "prompt_only",
        "stepwise_unmasked",
        "stepwise_masked",
    )


def test_analysis_method_roles_must_cover_evaluated_methods(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/full_study.yaml").read_text())
    raw["analysis"]["matched_change_methods"] = ["stepwise_masked"]
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(StudyConfigError, match="cover every evaluated method"):
        load_study_config(path)


def test_calibration_can_probe_one_direction_only(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/full_study.yaml").read_text())
    raw["status"] = "calibration"
    raw["evaluation"].pop("alphas", None)
    raw["evaluation"]["method_alphas"] = {
        "prompt_only": [0.0],
        "posthoc_latent": [0.0],
        "stepwise_unmasked": [-1.0, -0.75, 0.0],
        "stepwise_masked": [-1.0, -0.75, 0.0],
    }
    raw["prompts"]["prompt_only_levels"] = {}
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_study_config(path)
    assert config.alphas_for("prompt_only") == (0.0,)
    assert config.alphas_for("stepwise_masked") == (-1.0, -0.75, 0.0)

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from scripts.run_confirmatory import (
    BASE_PROMPT,
    NEGATIVE_PROMPT,
    SUPPORTED_METHODS,
    ConfirmatoryRunner,
    _masked_direction,
    _resize_direction,
    build_plan,
    prompt_for_alpha,
)
from src.metrics.protocol import protocol_record
from src.utils.config import ExperimentConfig
from src.validation.readiness import REQUIRED_CRITERIA

REPOSITORY_ROOT = Path(__file__).parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "full_study.yaml"
MODEL_REVISION = "a" * 40


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_passing_readiness(path, config_path, direction_path, model_revision=MODEL_REVISION):
    report = {
        "schema_version": "1.0",
        "readiness_id": "tmlr_collection_readiness_v1",
        "study_id": "skin_tone_steering_confirmatory_v1",
        "decision": {"collection_ready": True, "blockers": []},
        "authorities": {
            "study_config": {"sha256": file_digest(config_path)},
            "evaluation_protocol": {"sha256": protocol_record()["sha256"]},
        },
        "inputs": {
            "generation_provenance": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "model_revision": model_revision,
                "direction_sha256": file_digest(direction_path),
            }
        },
        "criteria": [
            {"criterion_id": criterion_id, "required": True, "status": "passed"}
            for criterion_id in REQUIRED_CRITERIA
        ],
    }
    path.write_text(json.dumps(report))
    return path


def test_confirmatory_plan_contains_exact_declared_matrix():
    config = ExperimentConfig.from_yaml(CONFIG_PATH)

    rows = build_plan(config)

    assert len(rows) == 600
    assert sum(row.alpha != 0 for row in rows) == 480
    assert {row.method for row in rows} == set(SUPPORTED_METHODS)
    assert len({row.row_id for row in rows}) == 600


def test_prompt_baseline_changes_only_directional_descriptor():
    config = ExperimentConfig.from_yaml(CONFIG_PATH)

    assert prompt_for_alpha(0, config.evaluation.alphas) == BASE_PROMPT
    assert prompt_for_alpha(-0.75, config.evaluation.alphas).startswith(BASE_PROMPT)
    assert "lighter visible skin tone" in prompt_for_alpha(
        -0.75, config.evaluation.alphas
    )
    assert "darker visible skin tone" in prompt_for_alpha(1.5, config.evaluation.alphas)


def test_plan_only_writes_complete_metadata_without_loading_model(tmp_path):
    def fail_if_loaded(*_args):
        raise AssertionError("model factory must not run in plan-only mode")

    runner = ConfirmatoryRunner(
        CONFIG_PATH,
        tmp_path,
        model_factory=fail_if_loaded,
    )

    manifest_path = runner.write_plan()
    manifest = json.loads(manifest_path.read_text())
    first_seed = json.loads((tmp_path / "seeds" / "1000" / "metadata.json").read_text())

    assert manifest["status"] == "planned"
    assert manifest["schema_version"] == "2.0"
    assert manifest["summary"] == {
        "completed_rows": 0,
        "failed_rows": 0,
        "planned_rows": 600,
        "unattempted_rows": 600,
    }
    assert len(manifest["rows"]) == 600
    assert manifest["config"]["sha256"]
    assert manifest["config"]["parsed"]["analysis"]["version"] == (
        "tmlr_statistical_analysis_v1"
    )
    assert manifest["config"]["parsed"]["analysis"]["rng_seed"] == 20260813
    assert manifest["provenance"]["git"]["commit"]
    assert manifest["thresholds"] == vars(runner.config.thresholds)
    assert manifest["metric_protocol"]["document"]["status"] == "frozen"
    assert len(manifest["metric_protocol"]["sha256"]) == 64
    assert first_seed["generation"]["negative_prompt"] == NEGATIVE_PROMPT
    assert len(first_seed["results"]) == 20
    assert all(result["status"] == "planned" for result in first_seed["results"])


def test_execute_requires_revision_and_direction_artifact(tmp_path):
    runner = ConfirmatoryRunner(CONFIG_PATH, tmp_path)

    with pytest.raises(ValueError, match="--model-revision"):
        runner._validate_execution_inputs()

    runner.model_revision = MODEL_REVISION
    with pytest.raises(ValueError, match="--direction"):
        runner._validate_execution_inputs()

    direction_path = tmp_path / "direction.pt"
    torch.save(torch.ones((1, 4, 2, 2)), direction_path)
    runner.direction_path = direction_path
    runner.direction_metadata = runner._direction_metadata()
    with pytest.raises(ValueError, match="--readiness-report"):
        runner._validate_execution_inputs()


def test_setup_failure_is_checkpointed_without_marking_rows_attempted(tmp_path, monkeypatch):
    direction_path = tmp_path / "direction.pt"
    torch.save(torch.ones((1, 4, 2, 2)), direction_path)

    def fail_model_setup(*_args):
        raise RuntimeError("synthetic setup failure")

    runner = ConfirmatoryRunner(
        CONFIG_PATH,
        tmp_path / "run",
        direction_path=direction_path,
        model_revision=MODEL_REVISION,
        readiness_report_path=write_passing_readiness(
            tmp_path / "readiness.json", CONFIG_PATH, direction_path
        ),
        device="cpu",
        model_factory=fail_model_setup,
    )
    runner.provenance["git"]["dirty"] = False
    monkeypatch.setattr(
        "scripts.run_confirmatory.validate_collection_readiness_report",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(RuntimeError, match="synthetic setup failure"):
        runner.execute()

    manifest = json.loads((tmp_path / "run" / "study_manifest.json").read_text())
    assert manifest["status"] == "setup_failed"
    assert manifest["summary"]["failed_rows"] == 0
    assert manifest["summary"]["unattempted_rows"] == 600
    assert manifest["failures"][0]["stage"] == "setup"


def test_direction_resize_and_mask_preserve_latent_shape():
    config = ExperimentConfig.from_yaml(CONFIG_PATH)
    direction = torch.ones((1, 4, 8, 8))
    latent = torch.zeros((1, 4, 16, 16))

    resized = _resize_direction(direction, latent)
    masked = _masked_direction(direction, config)

    assert resized.shape == latent.shape
    assert masked.shape == direction.shape
    assert masked[..., 4, 4].mean() > masked[..., 0, 0].mean()


class FakeModel:
    def __init__(self, resolved_revision=MODEL_REVISION):
        self.calls = []
        self.resolved_revision = resolved_revision

    def generate_from_prompt(self, **kwargs):
        self.calls.append(("prompt_only", kwargs))
        return Image.new("RGB", (8, 8)), torch.zeros((1, 4, 2, 2))

    def decode_latent(self, latent):
        self.calls.append(("posthoc_latent", latent.clone()))
        return Image.new("RGB", (8, 8))

    def generate_steered(self, **kwargs):
        self.calls.append(("stepwise", kwargs))
        return Image.new("RGB", (8, 8)), torch.zeros((1, 4, 2, 2))


def test_generation_adapters_support_all_four_methods(tmp_path):
    runner = ConfirmatoryRunner(CONFIG_PATH, tmp_path)
    model = FakeModel()
    base_image = Image.new("RGB", (8, 8))
    base_latent = torch.zeros((1, 4, 2, 2))
    direction = torch.ones_like(base_latent)
    masked_direction = direction * 0.5
    rows = {
        row.method: row
        for row in runner.rows
        if row.seed == 1000 and row.alpha == 0.75
    }

    outputs = {
        method: runner._generate_row(
            model,
            rows[method],
            base_image,
            base_latent,
            direction,
            masked_direction,
        )
        for method in SUPPORTED_METHODS
    }

    assert set(outputs) == set(SUPPORTED_METHODS)
    assert all(isinstance(image, Image.Image) for image in outputs.values())
    prompt_call = next(call for call in model.calls if call[0] == "prompt_only")
    assert prompt_call[1]["seed"] == 1000
    assert prompt_call[1]["num_inference_steps"] == 25
    stepwise_calls = [call for call in model.calls if call[0] == "stepwise"]
    assert len(stepwise_calls) == 2
    assert torch.equal(stepwise_calls[0][1]["race_vector"], direction)
    assert torch.equal(stepwise_calls[1][1]["race_vector"], masked_direction)


def test_synthetic_execution_checkpoints_hashes_for_every_method(tmp_path, monkeypatch):
    document = yaml.safe_load(CONFIG_PATH.read_text())
    document["evaluation"]["seeds"] = [1000]
    document["evaluation"]["matrix"] = {
        "pairing_unit": "seed",
        "expected_seeds": 1,
        "expected_methods": 4,
        "expected_alphas": 5,
        "expected_rows": 20,
        "expected_nonzero_alpha_rows": 16,
    }
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text(yaml.safe_dump(document))
    direction_path = tmp_path / "direction.pt"
    torch.save(torch.ones((1, 4, 2, 2)), direction_path)
    fake_model = FakeModel()
    readiness_path = write_passing_readiness(
        tmp_path / "readiness.json", config_path, direction_path
    )
    runner = ConfirmatoryRunner(
        config_path,
        tmp_path / "run",
        direction_path=direction_path,
        model_revision=MODEL_REVISION,
        readiness_report_path=readiness_path,
        device="cpu",
        model_factory=lambda *_args: fake_model,
    )
    runner.provenance["git"]["dirty"] = False
    monkeypatch.setattr(
        "scripts.run_confirmatory.validate_collection_readiness_report",
        lambda *_args, **_kwargs: {},
    )

    runner.execute()

    manifest = json.loads((tmp_path / "run" / "study_manifest.json").read_text())
    metadata = json.loads((tmp_path / "run" / "seeds" / "1000" / "metadata.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["model"]["resolved_revision"] == MODEL_REVISION
    assert manifest["summary"]["completed_rows"] == 20
    assert manifest["summary"]["failed_rows"] == 0
    assert manifest["summary"]["unattempted_rows"] == 0
    assert {result["method"] for result in metadata["results"]} == set(
        SUPPORTED_METHODS
    )
    assert all(result["image_sha256"] for result in metadata["results"])
    assert all(result["base_image_sha256"] for result in metadata["results"])


def test_failure_records_machine_readable_reason():
    error = RuntimeError("synthetic generation failure")

    failure = ConfirmatoryRunner._failure("row_generation", error, "row-1")

    assert failure["row_id"] == "row-1"
    assert failure["stage"] == "row_generation"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "synthetic generation failure"

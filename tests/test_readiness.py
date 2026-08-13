import hashlib
import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("cv2")

import src.validation.readiness as readiness_module  # noqa: E402
from scripts.validate_readiness import main as readiness_main  # noqa: E402
from src.validation.readiness import (  # noqa: E402
    ReadinessError,
    _runtime_observation,
    build_readiness_report,
    run_synthetic_sensitivity_checks,
    validate_collection_readiness_report,
)

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "full_study.yaml"
EVALUATION_PROTOCOL = ROOT / "configs" / "evaluation_protocol.yaml"
READINESS_PROTOCOL = ROOT / "configs" / "step6_readiness.yaml"
STORAGE_POLICY = ROOT / "configs" / "collection_policy.yaml"
SOURCE_REGISTRY = ROOT / "configs" / "validation_sources.yaml"
MODEL_REVISION = "b" * 40


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def criterion(report, criterion_id):
    return next(item for item in report["criteria"] if item["criterion_id"] == criterion_id)


def report_kwargs(**overrides):
    values = {
        "project_root": ROOT,
        "study_config_path": CONFIG,
        "evaluation_protocol_path": EVALUATION_PROTOCOL,
        "readiness_protocol_path": READINESS_PROTOCOL,
        "storage_policy_path": STORAGE_POLICY,
        "source_registry_path": SOURCE_REGISTRY,
    }
    values.update(overrides)
    return values


def write_external_validation(tmp_path, **agreement_overrides):
    artifact = tmp_path / "external_results.csv"
    artifact.write_text("metric,value\nmedian_absolute_delta_error,1.0\n")
    agreement = {
        "minimum_reference_change": 2.0,
        "median_absolute_delta_error": 1.0,
        "percentile_95_absolute_delta_error": 2.0,
        "direction_agreement": 0.95,
        "confidence_intervals_reported": True,
        "confidence_level": 0.95,
    }
    agreement.update(agreement_overrides)
    document = {
        "schema_version": "1.0",
        "validation_id": "held_out_colorimeter_validation_v1",
        "status": "approved",
        "metric_id": "relative_cheek_CIELAB_Lstar_v1",
        "held_out_from_metric_development": True,
        "reference_method": {
            "type": "calibrated_colorimeter_or_spectrophotometer",
            "calibration_recorded": True,
        },
        "domain": {
            "controlled_frontal_portraits": True,
            "neutral_studio_background": True,
            "independent_people": 30,
            "reference_lstar_span": 30.0,
            "lighting_and_capture_conditions_reported": True,
        },
        "agreement": agreement,
        "failure_reporting": {
            "full_denominator_reported": True,
            "detector_and_mask_failures_are_outcomes": True,
        },
        "governance": {
            "license_and_consent_documented": True,
            "limitations_and_error_profile_reported": True,
            "independent_review_status": "approved",
        },
        "evidence_artifacts": [{"path": artifact.name, "sha256": digest(artifact)}],
    }
    path = tmp_path / "external_validation.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def write_generation_provenance(tmp_path):
    direction = tmp_path / "direction.pt"
    direction.write_bytes(b"synthetic direction fixture; not a release artifact")
    source_manifest = tmp_path / "direction_source.json"
    source_manifest.write_text('{"fixture": true}\n')
    document = {
        "schema_version": "1.0",
        "provenance_id": "direction_provenance_fixture_v1",
        "model": {
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "requested_revision": MODEL_REVISION,
            "resolved_revision": MODEL_REVISION,
            "license_accepted": True,
            "license_identifier": "OpenRAIL++-M",
        },
        "direction": {
            "artifact": {"path": direction.name, "sha256": digest(direction)},
            "source_manifest": {
                "path": source_manifest.name,
                "sha256": digest(source_manifest),
            },
            "study_config_sha256": digest(CONFIG),
            "evaluation_protocol_sha256": digest(EVALUATION_PROTOCOL),
            "estimator": "paired_mean_difference",
            "train_pairs": 64,
            "held_out_pairs": 32,
            "optimization": False,
            "deterministic_vae_encoding": True,
            "code_commit": "c" * 40,
        },
    }
    path = tmp_path / "generation_provenance.yaml"
    path.write_text(yaml.safe_dump(document))
    return path, direction


def test_synthetic_sensitivity_checks_cover_declared_cases_and_pass():
    readiness = yaml.safe_load(READINESS_PROTOCOL.read_text())

    evidence = run_synthetic_sensitivity_checks(readiness)

    assert [item["check_id"] for item in evidence] == readiness["criteria"][
        "illumination_and_color_sensitivity"
    ]["required_checks"]
    assert all(item["status"] == "passed" for item in evidence)
    assert all(item["non_confirmatory"] for item in evidence)
    local = next(
        item for item in evidence if item["check_id"] == "local_skin_illumination_susceptibility"
    )
    assert "known confound" in local["limitation"]


def test_runtime_accepts_only_the_frozen_linux_stack():
    assert _runtime_observation(
        system="Linux", python_version=(3, 12), mediapipe_version="0.10.21"
    ) == (True, [])

    passed, failures = _runtime_observation(
        system="Darwin", python_version=(3, 13), mediapipe_version="0.10.20"
    )

    assert not passed
    assert failures == [
        "unsupported_operating_system",
        "unsupported_python_version",
        "mediapipe_version_drift",
    ]


def test_baseline_report_is_deterministic_and_blocks_missing_external_inputs():
    first = build_readiness_report(**report_kwargs())
    second = build_readiness_report(**report_kwargs())

    assert first == second
    assert not first["decision"]["collection_ready"]
    assert first["validation_scope"] == {
        "confirmatory_images_generated": False,
        "confirmatory_matrix_executed": False,
        "expensive_sdxl_model_loaded": False,
        "real_person_images_used_by_synthetic_checks": False,
        "synthetic_fixture_evidence_is_external_validation": False,
    }
    assert criterion(first, "illumination_and_color_sensitivity")["status"] == "passed"
    assert criterion(first, "storage_and_privacy")["status"] == "passed"
    assert criterion(first, "external_target_metric_validity")["blockers"][0]["code"] == (
        "external_validation_missing"
    )
    assert criterion(first, "immutable_generation_provenance")["blockers"][0]["code"] == (
        "generation_provenance_missing"
    )
    assert criterion(first, "explicit_collection_approval")["blockers"][0]["code"] == (
        "collection_approval_missing"
    )


def test_strict_cli_writes_blocked_report_without_generation(tmp_path):
    output = tmp_path / "readiness.json"

    exit_code = readiness_main(
        [
            "--project-root",
            str(ROOT),
            "--config",
            str(CONFIG),
            "--evaluation-protocol",
            str(EVALUATION_PROTOCOL),
            "--readiness-protocol",
            str(READINESS_PROTOCOL),
            "--storage-policy",
            str(STORAGE_POLICY),
            "--source-registry",
            str(SOURCE_REGISTRY),
            "--output",
            str(output),
            "--strict",
        ]
    )

    report = json.loads(output.read_text())
    assert exit_code == 2
    assert not report["decision"]["collection_ready"]
    assert not report["validation_scope"]["confirmatory_images_generated"]


def test_external_validation_acceptance_boundaries_are_inclusive(tmp_path):
    evidence_path = write_external_validation(tmp_path)
    report = build_readiness_report(**report_kwargs(external_validation_path=evidence_path))

    assert criterion(report, "external_target_metric_validity")["status"] == "passed"

    failed_path = write_external_validation(tmp_path, direction_agreement=0.949999)
    failed = build_readiness_report(**report_kwargs(external_validation_path=failed_path))
    external = criterion(failed, "external_target_metric_validity")
    assert external["status"] == "blocked"
    assert any(item["code"] == "direction_agreement_failed" for item in external["blockers"])

    nonfinite_path = write_external_validation(tmp_path, median_absolute_delta_error=float("nan"))
    nonfinite = build_readiness_report(**report_kwargs(external_validation_path=nonfinite_path))
    nonfinite_external = criterion(nonfinite, "external_target_metric_validity")
    assert nonfinite_external["status"] == "blocked"
    assert any(
        item["code"] == "median_absolute_delta_error_failed"
        for item in nonfinite_external["blockers"]
    )


def test_generation_provenance_checks_exact_revision_and_artifact_lineage(tmp_path):
    provenance_path, direction = write_generation_provenance(tmp_path)
    report = build_readiness_report(**report_kwargs(generation_provenance_path=provenance_path))

    provenance = criterion(report, "immutable_generation_provenance")
    assert provenance["status"] == "passed"
    assert report["inputs"]["generation_provenance"] == {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_revision": MODEL_REVISION,
        "direction_sha256": digest(direction),
        "direction_size_bytes": direction.stat().st_size,
        "provenance_record_sha256": digest(provenance_path),
    }

    document = yaml.safe_load(provenance_path.read_text())
    document["model"]["resolved_revision"] = "mutable-tag"
    provenance_path.write_text(yaml.safe_dump(document))
    drifted = build_readiness_report(**report_kwargs(generation_provenance_path=provenance_path))
    assert criterion(drifted, "immutable_generation_provenance")["status"] == "blocked"


def test_approval_must_bind_external_evidence_and_generation_provenance(tmp_path):
    external = write_external_validation(tmp_path)
    provenance, direction = write_generation_provenance(tmp_path)
    approval_document = {
        "schema_version": "1.0",
        "approval_id": "collection_approval_fixture_v1",
        "approved_by": "fixture-reviewer",
        "approved_at_utc": "2026-08-13T00:00:00Z",
        "study_id": "skin_tone_steering_confirmatory_v1",
        "scope": "confirmatory_collection",
        "decision": "approved",
        "study_config_sha256": digest(CONFIG),
        "evaluation_protocol_sha256": digest(EVALUATION_PROTOCOL),
        "readiness_protocol_sha256": digest(READINESS_PROTOCOL),
        "external_validation_id": "held_out_colorimeter_validation_v1",
        "model_revision": MODEL_REVISION,
        "direction_sha256": digest(direction),
    }
    approval = tmp_path / "approval.yaml"
    approval.write_text(yaml.safe_dump(approval_document))
    report = build_readiness_report(
        **report_kwargs(
            external_validation_path=external,
            generation_provenance_path=provenance,
            approval_path=approval,
        )
    )
    assert criterion(report, "explicit_collection_approval")["status"] == "passed"

    approval_document["direction_sha256"] = "0" * 64
    approval.write_text(yaml.safe_dump(approval_document))
    drifted = build_readiness_report(
        **report_kwargs(
            external_validation_path=external,
            generation_provenance_path=provenance,
            approval_path=approval,
        )
    )
    assert criterion(drifted, "explicit_collection_approval")["status"] == "blocked"


def test_execution_gate_rebuilds_bound_inputs_and_rejects_blockers(tmp_path, monkeypatch):
    external = write_external_validation(tmp_path)
    provenance, direction = write_generation_provenance(tmp_path)
    approval_document = {
        "schema_version": "1.0",
        "approval_id": "collection_approval_fixture_v1",
        "approved_by": "fixture-reviewer",
        "approved_at_utc": "2026-08-13T00:00:00Z",
        "study_id": "skin_tone_steering_confirmatory_v1",
        "scope": "confirmatory_collection",
        "decision": "approved",
        "study_config_sha256": digest(CONFIG),
        "evaluation_protocol_sha256": digest(EVALUATION_PROTOCOL),
        "readiness_protocol_sha256": digest(READINESS_PROTOCOL),
        "external_validation_id": "held_out_colorimeter_validation_v1",
        "model_revision": MODEL_REVISION,
        "direction_sha256": digest(direction),
    }
    approval = tmp_path / "approval.yaml"
    approval.write_text(yaml.safe_dump(approval_document))
    artifact_overrides = {}
    frozen_artifacts = {}
    for index, name in enumerate(
        yaml.safe_load(EVALUATION_PROTOCOL.read_text())["required_artifacts"]
    ):
        artifact = tmp_path / f"metric_{index}.bin"
        artifact.write_bytes(f"metric fixture {name}".encode())
        artifact_overrides[name] = artifact
        frozen_artifacts[name] = {"location": artifact.name, "sha256": digest(artifact)}
    monkeypatch.setattr(
        readiness_module,
        "load_protocol",
        lambda *_args, **_kwargs: {"required_artifacts": frozen_artifacts},
    )
    monkeypatch.setattr(
        readiness_module,
        "_runtime_criterion",
        lambda: readiness_module._criterion(
            "supported_metric_runtime",
            passed=True,
            evidence=[{"check_id": "actual_metric_runtime", "status": "passed"}],
        ),
    )
    report = build_readiness_report(
        **report_kwargs(
            external_validation_path=external,
            generation_provenance_path=provenance,
            approval_path=approval,
            artifact_overrides=artifact_overrides,
        )
    )
    assert report["decision"]["collection_ready"]
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(report))

    accepted = validate_collection_readiness_report(
        path,
        study_id="skin_tone_steering_confirmatory_v1",
        study_config_sha256=digest(CONFIG),
        evaluation_protocol_sha256=digest(EVALUATION_PROTOCOL),
        readiness_protocol_sha256=digest(READINESS_PROTOCOL),
        storage_policy_sha256=digest(STORAGE_POLICY),
        source_registry_sha256=digest(SOURCE_REGISTRY),
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        model_revision=MODEL_REVISION,
        direction_sha256=digest(direction),
    )
    assert accepted["decision"]["collection_ready"]

    report["decision"]["collection_ready"] = False
    report["decision"]["blockers"] = [{"code": "approval_missing"}]
    path.write_text(json.dumps(report))
    with pytest.raises(ReadinessError, match="blocks confirmatory collection"):
        validate_collection_readiness_report(
            path,
            study_id="skin_tone_steering_confirmatory_v1",
            study_config_sha256=digest(CONFIG),
            evaluation_protocol_sha256=digest(EVALUATION_PROTOCOL),
            readiness_protocol_sha256=digest(READINESS_PROTOCOL),
            storage_policy_sha256=digest(STORAGE_POLICY),
            source_registry_sha256=digest(SOURCE_REGISTRY),
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            model_revision=MODEL_REVISION,
            direction_sha256=digest(direction),
        )


def test_report_schema_declares_fail_closed_scope():
    schema = json.loads((ROOT / "schemas" / "step6_readiness_report.schema.json").read_text())

    assert schema["properties"]["readiness_id"]["const"] == ("tmlr_collection_readiness_v1")
    scope = schema["properties"]["validation_scope"]["properties"]
    assert all(definition["const"] is False for definition in scope.values())

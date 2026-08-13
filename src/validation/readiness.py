"""Deterministic, fail-closed readiness checks for confirmatory collection.

The checks in this module do not generate portraits and do not turn synthetic
fixtures into external validation evidence. They separate implementation and
sensitivity checks from evidence that must be supplied and reviewed outside
the confirmatory matrix.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import re
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.metrics.artifacts import inspect_artifact, sha256_file
from src.metrics.protocol import load_protocol
from src.metrics.skin_tone_metrics import SkinToneMetrics
from src.utils.config import ExperimentConfig

READINESS_SCHEMA_VERSION = "1.0"
READINESS_ID = "tmlr_collection_readiness_v1"
REQUIRED_CRITERIA = (
    "external_target_metric_validity",
    "illumination_and_color_sensitivity",
    "supported_metric_runtime",
    "checksum_verified_metric_artifacts",
    "immutable_generation_provenance",
    "storage_and_privacy",
    "explicit_collection_approval",
)
IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READINESS_PROTOCOL_PATH = PROJECT_ROOT / "configs" / "step6_readiness.yaml"
DEFAULT_STORAGE_POLICY_PATH = PROJECT_ROOT / "configs" / "collection_policy.yaml"
DEFAULT_SOURCE_REGISTRY_PATH = PROJECT_ROOT / "configs" / "validation_sources.yaml"


class ReadinessError(ValueError):
    """Raised when a readiness input or report cannot be trusted."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ReadinessError(f"Cannot read YAML input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReadinessError(f"Expected a mapping in {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    digest, size = sha256_file(path)
    return {"path": str(path), "sha256": digest, "size_bytes": size}


def _block(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _evidence(
    check_id: str,
    passed: bool,
    *,
    observed: Any,
    expected: Any,
    limitation: str | None = None,
) -> dict[str, Any]:
    item = {
        "check_id": check_id,
        "status": "passed" if passed else "failed",
        "observed": observed,
        "expected": expected,
        "non_confirmatory": True,
    }
    if limitation:
        item["limitation"] = limitation
    return item


def _criterion(
    criterion_id: str,
    *,
    passed: bool,
    evidence: list[dict[str, Any]] | None = None,
    blockers: list[dict[str, str]] | None = None,
    validated_properties: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "criterion_id": criterion_id,
        "required": True,
        "status": "passed" if passed else "blocked",
        "validated_properties": validated_properties or [],
        "evidence": evidence or [],
        "blockers": blockers or [],
        "limitations": limitations or [],
    }


def _validate_readiness_protocol(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ReadinessError("Unsupported readiness protocol schema_version")
    if document.get("readiness_id") != READINESS_ID:
        raise ReadinessError("Unexpected readiness_id")
    if document.get("status") != "frozen_before_confirmatory_collection":
        raise ReadinessError("Readiness criteria must be frozen before collection")
    if document.get("decision_rule") != "all_required_criteria_must_pass":
        raise ReadinessError("Readiness decision must require every criterion")
    criteria = document.get("criteria")
    if not isinstance(criteria, dict) or tuple(criteria) != REQUIRED_CRITERIA:
        raise ReadinessError("Readiness criterion set or order has drifted")
    if not all(value.get("required") is True for value in criteria.values()):
        raise ReadinessError("Every Step 6 readiness criterion must remain required")


def _portrait(
    skin_rgb: tuple[int, int, int],
    background_rgb: tuple[int, int, int] = (200, 200, 200),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((128, 128, 3), background_rgb, dtype=np.uint8)
    skin_mask = np.zeros((128, 128), dtype=bool)
    skin_mask[28:100, 32:96] = True
    image[skin_mask] = skin_rgb
    reference_mask = np.zeros((128, 128), dtype=bool)
    reference_mask[:48, :20] = True
    reference_mask[:48, -20:] = True
    return image, skin_mask, reference_mask


def _relative_change(
    metrics: SkinToneMetrics,
    first: np.ndarray,
    second: np.ndarray,
    skin_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> tuple[float | None, float | None]:
    before = metrics.measure(
        first,
        skin_mask=skin_mask,
        reference_mask=reference_mask,
    )
    after = metrics.measure(
        second,
        skin_mask=skin_mask,
        reference_mask=reference_mask,
    )
    if before is None or after is None:
        return None, None
    reference_shift = after.reference_lstar - before.reference_lstar
    if not math.isfinite(reference_shift) or abs(reference_shift) > metrics.max_reference_shift:
        return None, reference_shift
    return after.relative_lstar - before.relative_lstar, reference_shift


def _reference_shift_stable(value: float, boundary: float) -> bool:
    return math.isfinite(value) and abs(value) <= boundary


def _reference_chroma_acceptable(value: float, boundary: float) -> bool:
    return math.isfinite(value) and value <= boundary


def _runtime_observation(
    *,
    system: str,
    python_version: tuple[int, int],
    mediapipe_version: str | None,
) -> tuple[bool, list[str]]:
    failures = []
    if system != "Linux":
        failures.append("unsupported_operating_system")
    if not ((3, 10) <= python_version < (3, 13)):
        failures.append("unsupported_python_version")
    if mediapipe_version != "0.10.21":
        failures.append("mediapipe_version_drift")
    return not failures, failures


class _NoFaceBackend:
    def detect(self, _image: np.ndarray) -> None:
        return None


def run_synthetic_sensitivity_checks(
    readiness_protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Exercise deterministic non-confirmatory checks with no model loading."""
    settings = readiness_protocol["criteria"]["illumination_and_color_sensitivity"]
    minimum = float(settings["known_color_minimum_absolute_change"])
    exposure_maximum = float(settings["global_exposure_maximum_absolute_residual"])
    local_minimum = float(settings["local_illumination_minimum_apparent_change"])
    shift_boundary = float(settings["reference_shift_boundary"])
    chroma_boundary = float(settings["reference_chroma_boundary"])
    metrics = SkinToneMetrics(landmark_backend=_NoFaceBackend())
    baseline, skin_mask, reference_mask = _portrait((160, 110, 80))
    darker, _, _ = _portrait((125, 76, 52))
    lighter, _, _ = _portrait((195, 145, 112))
    dark_delta, _ = _relative_change(metrics, baseline, darker, skin_mask, reference_mask)
    light_delta, _ = _relative_change(metrics, baseline, lighter, skin_mask, reference_mask)

    exposed = np.clip(baseline.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    exposure_delta, exposure_reference = _relative_change(
        metrics, baseline, exposed, skin_mask, reference_mask
    )
    locally_lit = baseline.copy()
    locally_lit[skin_mask] = np.clip(locally_lit[skin_mask].astype(np.int16) + 20, 0, 255).astype(
        np.uint8
    )
    local_delta, local_reference = _relative_change(
        metrics, baseline, locally_lit, skin_mask, reference_mask
    )

    colored, colored_skin, colored_reference = _portrait((160, 110, 80), (255, 0, 0))
    colored_measurement = metrics.measure(
        colored,
        skin_mask=colored_skin,
        reference_mask=colored_reference,
    )
    no_face = metrics.measure(baseline)
    wrong_shape = np.ones((127, 128), dtype=bool)
    nonbinary = skin_mask.astype(np.uint8) * 2
    invalid_masks_rejected = (
        metrics.measure(
            baseline,
            skin_mask=wrong_shape,
            reference_mask=reference_mask,
        )
        is None
        and metrics.measure(
            baseline,
            skin_mask=nonbinary,
            reference_mask=reference_mask,
        )
        is None
    )
    try:
        metrics.measure(np.full((8, 8, 3), np.nan, dtype=np.float32))
    except ValueError:
        nonfinite_rejected = True
    else:
        nonfinite_rejected = False

    with tempfile.TemporaryDirectory(prefix="step6_checksum_") as temporary:
        artifact = Path(temporary) / "fixture.bin"
        artifact.write_bytes(b"step6 deterministic checksum fixture")
        expected, _ = sha256_file(artifact)
        verified = inspect_artifact(artifact, expected, name="synthetic fixture")
        artifact.write_bytes(b"step6 deterministic checksum drift")
        drift = inspect_artifact(artifact, expected, name="synthetic fixture")
        checksum_ok = verified.verified and drift.status == "checksum_mismatch"

    supported_runtime, _ = _runtime_observation(
        system="Linux",
        python_version=(3, 12),
        mediapipe_version="0.10.21",
    )
    drift_runtime, drift_failures = _runtime_observation(
        system="Darwin",
        python_version=(3, 13),
        mediapipe_version="0.10.20",
    )

    checks = [
        _evidence(
            "known_skin_darkening",
            dark_delta is not None and dark_delta <= -minimum,
            observed={"relative_lstar_change": dark_delta},
            expected={"maximum": -minimum},
        ),
        _evidence(
            "known_skin_lightening",
            light_delta is not None and light_delta >= minimum,
            observed={"relative_lstar_change": light_delta},
            expected={"minimum": minimum},
        ),
        _evidence(
            "moderate_global_exposure_first_order_correction",
            exposure_delta is not None and abs(exposure_delta) <= exposure_maximum,
            observed={
                "relative_lstar_change": exposure_delta,
                "reference_lstar_shift": exposure_reference,
            },
            expected={"maximum_absolute_residual": exposure_maximum},
            limitation="This fixture tests only a uniform additive RGB exposure change.",
        ),
        _evidence(
            "local_skin_illumination_susceptibility",
            local_delta is not None
            and abs(local_delta) >= local_minimum
            and local_reference == 0.0,
            observed={
                "apparent_relative_lstar_change": local_delta,
                "reference_lstar_shift": local_reference,
            },
            expected={"minimum_apparent_change": local_minimum},
            limitation="Passing characterizes a known confound; it does not remove it.",
        ),
        _evidence(
            "reference_shift_boundary_inclusive",
            _reference_shift_stable(shift_boundary, shift_boundary),
            observed={"shift": shift_boundary, "accepted": True},
            expected={"inclusive_maximum": shift_boundary},
        ),
        _evidence(
            "reference_shift_above_boundary_rejected",
            not _reference_shift_stable(math.nextafter(shift_boundary, math.inf), shift_boundary),
            observed={"shift": math.nextafter(shift_boundary, math.inf), "accepted": False},
            expected={"reject_above": shift_boundary},
        ),
        _evidence(
            "reference_chroma_boundary_inclusive",
            _reference_chroma_acceptable(chroma_boundary, chroma_boundary),
            observed={"chroma": chroma_boundary, "accepted": True},
            expected={"inclusive_maximum": chroma_boundary},
        ),
        _evidence(
            "reference_chroma_above_boundary_rejected",
            not _reference_chroma_acceptable(
                math.nextafter(chroma_boundary, math.inf), chroma_boundary
            ),
            observed={
                "chroma": math.nextafter(chroma_boundary, math.inf),
                "accepted": False,
            },
            expected={"reject_above": chroma_boundary},
        ),
        _evidence(
            "colored_reference_rejected",
            colored_measurement is None,
            observed={"measurement": None if colored_measurement is None else "present"},
            expected={"measurement": None},
        ),
        _evidence(
            "face_detection_failure_invalid",
            no_face is None,
            observed={"measurement": None if no_face is None else "present"},
            expected={"measurement": None, "meaning": "outcome_not_exclusion"},
        ),
        _evidence(
            "invalid_masks_rejected",
            invalid_masks_rejected,
            observed={"wrong_shape_and_nonbinary_measurements": None},
            expected={"measurement": None},
        ),
        _evidence(
            "nonfinite_input_rejected",
            nonfinite_rejected,
            observed={"rejected": nonfinite_rejected},
            expected={"rejected": True},
        ),
        _evidence(
            "checksum_match_and_drift",
            checksum_ok,
            observed={
                "matching_status": verified.status,
                "drift_status": drift.status,
            },
            expected={"matching": "verified", "drift": "checksum_mismatch"},
        ),
        _evidence(
            "runtime_drift_rejected",
            supported_runtime and not drift_runtime,
            observed={"drift_failure_codes": drift_failures},
            expected={
                "failure_codes": [
                    "unsupported_operating_system",
                    "unsupported_python_version",
                    "mediapipe_version_drift",
                ]
            },
        ),
    ]
    expected_ids = list(settings["required_checks"])
    if [item["check_id"] for item in checks] != expected_ids:
        raise ReadinessError("Synthetic sensitivity check set has drifted")
    return checks


def _external_validation_criterion(
    path: Path | None,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    criterion_id = "external_target_metric_validity"
    requirements = protocol["criteria"][criterion_id]
    if path is None:
        blocker = _block(
            "external_validation_missing",
            "No held-out paired portrait and calibrated-instrument validation package was supplied.",
        )
        return _criterion(
            criterion_id,
            passed=False,
            blockers=[blocker],
            limitations=[
                "Synthetic fixtures do not establish external validity on human skin or portrait images."
            ],
        ), None
    document = _load_yaml(path)
    blockers: list[dict[str, str]] = []
    validation_id = document.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        blockers.append(_block("external_validation_id_missing", "validation_id is required"))
    if document.get("status") != requirements["evidence_status_required"]:
        blockers.append(_block("external_validation_not_approved", "status must be approved"))
    if document.get("metric_id") != requirements["metric_id"]:
        blockers.append(_block("external_metric_mismatch", "metric_id does not match"))
    if document.get("held_out_from_metric_development") is not True:
        blockers.append(_block("external_validation_not_held_out", "held-out evidence is required"))
    reference = document.get("reference_method", {})
    if reference.get("type") != requirements["reference_method"]["type"]:
        blockers.append(
            _block("invalid_reference_method", "calibrated instrument reference required")
        )
    if reference.get("calibration_recorded") is not True:
        blockers.append(
            _block("calibration_record_missing", "instrument calibration record required")
        )
    domain = document.get("domain", {})
    for field in ("controlled_frontal_portraits", "neutral_studio_background"):
        if domain.get(field) is not True:
            blockers.append(_block(f"domain_{field}_missing", f"{field} must be true"))
    try:
        independent_people = float(domain["independent_people"])
    except (KeyError, TypeError, ValueError):
        independent_people = math.nan
    if (
        not math.isfinite(independent_people)
        or not independent_people.is_integer()
        or independent_people < int(requirements["domain"]["minimum_independent_people"])
    ):
        blockers.append(_block("insufficient_independent_people", "external sample is too small"))
    try:
        lstar_span = float(domain["reference_lstar_span"])
    except (KeyError, TypeError, ValueError):
        lstar_span = math.nan
    if not math.isfinite(lstar_span) or lstar_span < float(
        requirements["domain"]["minimum_reference_lstar_span"]
    ):
        blockers.append(_block("insufficient_lstar_span", "reference L* span is too narrow"))
    if domain.get("lighting_and_capture_conditions_reported") is not True:
        blockers.append(_block("capture_conditions_missing", "capture conditions must be reported"))
    agreement = document.get("agreement", {})
    limits = requirements["agreement"]
    try:
        minimum_reference_change = float(agreement["minimum_reference_change"])
    except (KeyError, TypeError, ValueError):
        minimum_reference_change = math.nan
    if not math.isfinite(minimum_reference_change) or minimum_reference_change != float(
        limits["minimum_reference_change"]
    ):
        blockers.append(
            _block(
                "minimum_reference_change_mismatch",
                "agreement must use the frozen minimum reference change",
            )
        )
    comparisons = (
        ("median_absolute_delta_error", "maximum_median_absolute_delta_error", lambda a, b: a <= b),
        (
            "percentile_95_absolute_delta_error",
            "maximum_95th_percentile_absolute_delta_error",
            lambda a, b: a <= b,
        ),
        ("direction_agreement", "minimum_direction_agreement", lambda a, b: a >= b),
    )
    for observed_key, limit_key, comparator in comparisons:
        try:
            observed = float(agreement[observed_key])
        except (KeyError, TypeError, ValueError):
            blockers.append(_block(f"{observed_key}_missing", f"{observed_key} must be finite"))
            continue
        if not math.isfinite(observed) or not comparator(observed, float(limits[limit_key])):
            blockers.append(_block(f"{observed_key}_failed", f"{observed_key} misses acceptance"))
        if observed_key.endswith("error") and observed < 0:
            blockers.append(_block(f"{observed_key}_invalid", f"{observed_key} is negative"))
        if observed_key == "direction_agreement" and not 0.0 <= observed <= 1.0:
            blockers.append(
                _block("direction_agreement_invalid", "direction agreement is outside [0, 1]")
            )
    if agreement.get("confidence_intervals_reported") is not True:
        blockers.append(_block("confidence_intervals_missing", "confidence intervals are required"))
    try:
        confidence_level = float(agreement["confidence_level"])
    except (KeyError, TypeError, ValueError):
        confidence_level = math.nan
    if not math.isfinite(confidence_level) or confidence_level != float(limits["confidence_level"]):
        blockers.append(_block("confidence_level_mismatch", "confidence level must be 0.95"))
    failures = document.get("failure_reporting", {})
    if failures.get("full_denominator_reported") is not True:
        blockers.append(_block("failure_denominator_missing", "full denominator is required"))
    if failures.get("detector_and_mask_failures_are_outcomes") is not True:
        blockers.append(
            _block("failure_semantics_drift", "detector and mask failures must be outcomes")
        )
    governance = document.get("governance", {})
    for field in ("license_and_consent_documented", "limitations_and_error_profile_reported"):
        if governance.get(field) is not True:
            blockers.append(_block(field, f"{field} must be true"))
    if governance.get("independent_review_status") != "approved":
        blockers.append(_block("independent_review_missing", "independent review must be approved"))
    artifacts = document.get("evidence_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        blockers.append(
            _block("external_artifacts_missing", "checksum-bound evidence artifacts required")
        )
        artifacts = []
    artifact_evidence = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            blockers.append(
                _block("external_artifact_invalid", f"artifact {index} is not a mapping")
            )
            continue
        artifact_path = (path.parent / str(item.get("path", ""))).resolve()
        expected = str(item.get("sha256", ""))
        result = inspect_artifact(artifact_path, expected, name=f"external evidence {index}")
        artifact_evidence.append(result.to_dict())
        if not result.verified:
            blockers.append(
                _block("external_artifact_unverified", f"artifact {index}: {result.status}")
            )
    evidence = [
        {
            "check_id": "external_validation_record",
            "status": "passed" if not blockers else "failed",
            "record": _file_record(path),
            "artifact_verifications": artifact_evidence,
            "non_confirmatory": True,
        }
    ]
    return _criterion(
        criterion_id,
        passed=not blockers,
        evidence=evidence,
        blockers=blockers,
        validated_properties=[
            "Input structure, acceptance values, review status, and attached checksums were verified."
        ]
        if not blockers
        else [],
        limitations=[
            "The harness verifies the supplied record and artifact integrity; it does not authenticate the reviewer or reproduce the external study."
        ],
    ), validation_id if isinstance(validation_id, str) else None


def _runtime_criterion() -> dict[str, Any]:
    try:
        mediapipe_version = metadata.version("mediapipe")
    except metadata.PackageNotFoundError:
        mediapipe_version = None
    observed = {
        "operating_system": platform.system(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "mediapipe": mediapipe_version,
        "face_landmarker_delegate": "CPU",
    }
    passed, failures = _runtime_observation(
        system=platform.system(),
        python_version=sys.version_info[:2],
        mediapipe_version=mediapipe_version,
    )
    blockers = [_block(code, f"Observed runtime failed: {code}") for code in failures]
    return _criterion(
        "supported_metric_runtime",
        passed=passed,
        evidence=[
            {
                "check_id": "actual_metric_runtime",
                "status": "passed" if passed else "failed",
                "observed": observed,
                "expected": {
                    "operating_system": "Linux",
                    "python": ">=3.10,<3.13",
                    "mediapipe": "0.10.21",
                    "face_landmarker_delegate": "CPU",
                },
                "non_confirmatory": True,
            }
        ],
        blockers=blockers,
        validated_properties=["The current process matches the frozen metric runtime."]
        if passed
        else [],
    )


def _resolve_artifact_path(
    location: str,
    *,
    project_root: Path,
    overrides: Mapping[str, Path],
    name: str,
) -> Path:
    if name in overrides:
        return overrides[name].resolve()
    if location.startswith("${TORCH_HOME}"):
        torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        return torch_home / location.removeprefix("${TORCH_HOME}/")
    first, _, remainder = location.partition("/")
    specification = importlib.util.find_spec(first)
    if specification and specification.submodule_search_locations and remainder:
        return Path(next(iter(specification.submodule_search_locations))) / remainder
    return project_root / location


def _artifact_criterion(
    metric_protocol: Mapping[str, Any],
    *,
    project_root: Path,
    overrides: Mapping[str, Path],
) -> dict[str, Any]:
    evidence = []
    blockers = []
    for name, specification in metric_protocol["required_artifacts"].items():
        path = _resolve_artifact_path(
            specification["location"],
            project_root=project_root,
            overrides=overrides,
            name=name,
        )
        result = inspect_artifact(path, specification["sha256"], name=name)
        evidence.append(result.to_dict())
        if not result.verified:
            blockers.append(_block(f"metric_artifact_{result.status}", f"{name}: {result.status}"))
    return _criterion(
        "checksum_verified_metric_artifacts",
        passed=not blockers,
        evidence=evidence,
        blockers=blockers,
        validated_properties=["Every frozen metric artifact checksum matched."]
        if not blockers
        else [],
    )


def _verify_bound_artifact(
    document_path: Path,
    item: Mapping[str, Any],
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    path = (document_path.parent / str(item.get("path", ""))).resolve()
    result = inspect_artifact(path, str(item.get("sha256", "")), name=name)
    blocker = None if result.verified else _block(f"{name}_unverified", result.status)
    return result.to_dict(), blocker


def _provenance_criterion(
    path: Path | None,
    *,
    config: ExperimentConfig,
    config_sha256: str,
    metric_protocol_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    criterion_id = "immutable_generation_provenance"
    if path is None:
        return _criterion(
            criterion_id,
            passed=False,
            blockers=[
                _block(
                    "generation_provenance_missing", "No model/direction provenance record supplied"
                )
            ],
        ), None
    document = _load_yaml(path)
    blockers = []
    model = document.get("model", {})
    revision = model.get("resolved_revision")
    if model.get("id") != config.model.name:
        blockers.append(_block("model_id_mismatch", "SDXL model id does not match the study"))
    if not isinstance(revision, str) or not IMMUTABLE_REVISION.fullmatch(revision):
        blockers.append(
            _block("model_revision_not_immutable", "resolved revision must be lowercase 40-hex")
        )
    if model.get("requested_revision") != revision:
        blockers.append(
            _block("model_revision_resolution_mismatch", "requested and resolved revisions differ")
        )
    if model.get("license_accepted") is not True or not model.get("license_identifier"):
        blockers.append(
            _block("model_license_unrecorded", "model license acceptance must be recorded")
        )
    direction = document.get("direction", {})
    direction_evidence, direction_blocker = _verify_bound_artifact(
        path, direction.get("artifact", {}), name="direction_artifact"
    )
    source_evidence, source_blocker = _verify_bound_artifact(
        path, direction.get("source_manifest", {}), name="direction_source_manifest"
    )
    blockers.extend(item for item in (direction_blocker, source_blocker) if item)
    if direction.get("study_config_sha256") != config_sha256:
        blockers.append(
            _block("direction_config_mismatch", "direction does not bind the frozen study config")
        )
    if direction.get("evaluation_protocol_sha256") != metric_protocol_sha256:
        blockers.append(
            _block("direction_protocol_mismatch", "direction does not bind the metric protocol")
        )
    if direction.get("estimator") != config.direction.estimator:
        blockers.append(_block("direction_estimator_mismatch", "direction estimator drifted"))
    if direction.get("train_pairs") != config.direction.train_pairs:
        blockers.append(
            _block("direction_train_pairs_mismatch", "direction train-pair count drifted")
        )
    if direction.get("held_out_pairs") != config.direction.held_out_pairs:
        blockers.append(
            _block("direction_held_out_pairs_mismatch", "direction held-out count drifted")
        )
    if direction.get("optimization") is not config.direction.optimization:
        blockers.append(
            _block("direction_optimization_mismatch", "direction optimization setting drifted")
        )
    if (
        direction.get("deterministic_vae_encoding")
        is not config.direction.deterministic_vae_encoding
    ):
        blockers.append(_block("direction_vae_mismatch", "direction VAE encoding setting drifted"))
    code_commit = direction.get("code_commit")
    if not isinstance(code_commit, str) or not re.fullmatch(r"[0-9a-f]{40,64}", code_commit):
        blockers.append(
            _block("direction_code_commit_missing", "direction code commit is not immutable")
        )
    evidence = [
        {
            "check_id": "generation_provenance_record",
            "status": "passed" if not blockers else "failed",
            "record": _file_record(path),
            "model": model,
            "direction_artifact": direction_evidence,
            "direction_source_manifest": source_evidence,
            "non_confirmatory": True,
        }
    ]
    normalized = {
        "model_id": model.get("id"),
        "model_revision": revision,
        "direction_sha256": direction_evidence.get("actual_sha256"),
        "direction_size_bytes": direction_evidence.get("size_bytes"),
        "provenance_record_sha256": _file_record(path)["sha256"],
    }
    return _criterion(
        criterion_id,
        passed=not blockers,
        evidence=evidence,
        blockers=blockers,
        validated_properties=["Immutable model revision and direction lineage are checksum-bound."]
        if not blockers
        else [],
        limitations=[
            "A revision identifier and checksum establish provenance, not model behavior."
        ],
    ), normalized


def _policy_criterion(path: Path, *, project_root: Path) -> dict[str, Any]:
    document = _load_yaml(path)
    blockers = []
    if document.get("policy_id") != "tmlr_confirmatory_storage_privacy_v1":
        blockers.append(_block("storage_policy_id_mismatch", "Unexpected storage policy"))
    if document.get("scope") != "skin_tone_steering_confirmatory_v1":
        blockers.append(_block("storage_policy_scope_mismatch", "Policy scope is wrong"))
    if document.get("inputs", {}).get("synthetic_sdxl_portraits_only") is not True:
        blockers.append(
            _block("synthetic_only_not_enforced", "Confirmatory inputs must be synthetic")
        )
    if document.get("inputs", {}).get("real_person_images") != "prohibited":
        blockers.append(
            _block("real_person_inputs_not_prohibited", "Real-person inputs must be prohibited")
        )
    outputs = document.get("outputs", {})
    for field in (
        "commit_generated_portraits",
        "commit_face_embeddings",
        "persist_face_embeddings",
    ):
        if outputs.get(field) != "prohibited":
            blockers.append(_block(f"policy_{field}", f"{field} must be prohibited"))
    run_root = str(outputs.get("generated_run_root", ""))
    ignore_lines = {
        line.strip()
        for line in (project_root / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if outputs.get("must_be_gitignored") is not True or run_root not in ignore_lines:
        blockers.append(
            _block("generated_root_not_gitignored", f"{run_root!r} is not explicitly ignored")
        )
    if (
        document.get("publication", {}).get("separate_license_privacy_and_misuse_review_required")
        is not True
    ):
        blockers.append(
            _block("publication_review_missing", "Separate publication review is required")
        )
    return _criterion(
        "storage_and_privacy",
        passed=not blockers,
        blockers=blockers,
        evidence=[
            {
                "check_id": "storage_privacy_policy",
                "status": "passed" if not blockers else "failed",
                "record": _file_record(path),
                "generated_run_root": run_root,
                "gitignored": run_root in ignore_lines,
                "non_confirmatory": True,
            }
        ],
        validated_properties=["Versioned synthetic-only storage and privacy controls are present."]
        if not blockers
        else [],
        limitations=[
            "Filesystem access controls and later human compliance are not proven by a policy file."
        ],
    )


def _approval_criterion(
    path: Path | None,
    *,
    study_id: str,
    config_sha256: str,
    protocol_sha256: str,
    readiness_sha256: str,
    validation_id: str | None,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    criterion_id = "explicit_collection_approval"
    if path is None:
        return _criterion(
            criterion_id,
            passed=False,
            blockers=[
                _block(
                    "collection_approval_missing", "No explicit scoped collection approval supplied"
                )
            ],
            limitations=[
                "Approval records are intentionally external to synthetic validation evidence."
            ],
        )
    document = _load_yaml(path)
    blockers = []
    required_values = {
        "study_id": study_id,
        "scope": "confirmatory_collection",
        "decision": "approved",
        "study_config_sha256": config_sha256,
        "evaluation_protocol_sha256": protocol_sha256,
        "readiness_protocol_sha256": readiness_sha256,
        "external_validation_id": validation_id,
        "model_revision": provenance.get("model_revision") if provenance else None,
        "direction_sha256": provenance.get("direction_sha256") if provenance else None,
    }
    for field, expected in required_values.items():
        if expected is None or document.get(field) != expected:
            blockers.append(_block(f"approval_{field}_mismatch", f"Approval does not bind {field}"))
    for field in ("approval_id", "approved_by", "approved_at_utc"):
        if not isinstance(document.get(field), str) or not document[field].strip():
            blockers.append(_block(f"approval_{field}_missing", f"{field} is required"))
    return _criterion(
        criterion_id,
        passed=not blockers,
        blockers=blockers,
        evidence=[
            {
                "check_id": "collection_approval_record",
                "status": "passed" if not blockers else "failed",
                "record": _file_record(path),
                "approval_id": document.get("approval_id"),
                "decision": document.get("decision"),
                "scope": document.get("scope"),
                "non_confirmatory": True,
            }
        ],
        validated_properties=["An explicit approval record binds every required study artifact."]
        if not blockers
        else [],
        limitations=[
            "The harness verifies record consistency, not cryptographic approver identity."
        ],
    )


def build_readiness_report(
    *,
    project_root: Path,
    study_config_path: Path,
    evaluation_protocol_path: Path,
    readiness_protocol_path: Path,
    storage_policy_path: Path,
    source_registry_path: Path,
    external_validation_path: Path | None = None,
    generation_provenance_path: Path | None = None,
    approval_path: Path | None = None,
    artifact_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Build a deterministic readiness decision without confirmatory collection."""
    project_root = project_root.resolve()
    study_config_path = study_config_path.resolve()
    evaluation_protocol_path = evaluation_protocol_path.resolve()
    readiness_protocol_path = readiness_protocol_path.resolve()
    storage_policy_path = storage_policy_path.resolve()
    source_registry_path = source_registry_path.resolve()
    readiness = _load_yaml(readiness_protocol_path)
    _validate_readiness_protocol(readiness)
    config = ExperimentConfig.from_yaml(study_config_path)
    metric_protocol = load_protocol(evaluation_protocol_path)
    if readiness["study_id"] != config.study_id:
        raise ReadinessError("Readiness study_id does not match the frozen study")
    if readiness["evaluation_protocol_id"] != config.evaluation.protocol_id:
        raise ReadinessError("Readiness evaluation protocol does not match the study")
    if readiness["statistical_analysis_version"] != config.analysis.version:
        raise ReadinessError("Readiness statistical analysis version does not match")
    sources = _load_yaml(source_registry_path)
    if sources.get("registry_id") != "tmlr_target_metric_sources_v1":
        raise ReadinessError("Unexpected validation source registry")
    if not sources.get("evidence_gaps"):
        raise ReadinessError("Validation source registry must retain evidence gaps")
    config_record = _file_record(study_config_path)
    protocol_record = _file_record(evaluation_protocol_path)
    readiness_record = _file_record(readiness_protocol_path)

    external, validation_id = _external_validation_criterion(
        external_validation_path.resolve() if external_validation_path else None,
        readiness,
    )
    sensitivity_evidence = run_synthetic_sensitivity_checks(readiness)
    sensitivity_failures = [item for item in sensitivity_evidence if item["status"] != "passed"]
    sensitivity = _criterion(
        "illumination_and_color_sensitivity",
        passed=not sensitivity_failures,
        evidence=sensitivity_evidence,
        blockers=[
            _block("synthetic_check_failed", item["check_id"]) for item in sensitivity_failures
        ],
        validated_properties=[
            "Known synthetic RGB color changes produce the expected relative-L* direction.",
            "Uniform exposure is first-order corrected in the declared fixture.",
            "Local illumination remains visibly confounded and is not corrected away.",
            "Boundary, mask, detector, nonfinite, checksum, and runtime drift checks fail closed.",
        ]
        if not sensitivity_failures
        else [],
        limitations=[
            "Synthetic patch behavior is implementation evidence, not validation on human skin.",
            "Local illumination remains indistinguishable from rendered skin-color change.",
        ],
    )
    runtime = _runtime_criterion()
    artifacts = _artifact_criterion(
        metric_protocol,
        project_root=project_root,
        overrides=artifact_overrides or {},
    )
    provenance, normalized_provenance = _provenance_criterion(
        generation_provenance_path.resolve() if generation_provenance_path else None,
        config=config,
        config_sha256=config_record["sha256"],
        metric_protocol_sha256=protocol_record["sha256"],
    )
    policy = _policy_criterion(storage_policy_path, project_root=project_root)
    approval = _approval_criterion(
        approval_path.resolve() if approval_path else None,
        study_id=config.study_id,
        config_sha256=config_record["sha256"],
        protocol_sha256=protocol_record["sha256"],
        readiness_sha256=readiness_record["sha256"],
        validation_id=validation_id,
        provenance=normalized_provenance,
    )
    by_id = {
        item["criterion_id"]: item
        for item in (external, sensitivity, runtime, artifacts, provenance, policy, approval)
    }
    criteria = [by_id[criterion_id] for criterion_id in REQUIRED_CRITERIA]
    blockers = [
        {"criterion_id": item["criterion_id"], **blocker}
        for item in criteria
        for blocker in item["blockers"]
    ]
    passed_count = sum(item["status"] == "passed" for item in criteria)
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "readiness_id": READINESS_ID,
        "study_id": config.study_id,
        "decision": {
            "collection_ready": not blockers and passed_count == len(REQUIRED_CRITERIA),
            "decision_rule": "all_required_criteria_must_pass",
            "required_criteria_count": len(REQUIRED_CRITERIA),
            "passed_criteria_count": passed_count,
            "blocked_criteria_count": len(REQUIRED_CRITERIA) - passed_count,
            "blockers": blockers,
        },
        "authorities": {
            "study_config": config_record,
            "evaluation_protocol": protocol_record,
            "readiness_protocol": readiness_record,
            "storage_policy": _file_record(storage_policy_path),
            "source_registry": _file_record(source_registry_path),
        },
        "inputs": {
            "external_validation_id": validation_id,
            "generation_provenance": normalized_provenance,
        },
        "validation_scope": {
            "confirmatory_images_generated": False,
            "confirmatory_matrix_executed": False,
            "expensive_sdxl_model_loaded": False,
            "real_person_images_used_by_synthetic_checks": False,
            "synthetic_fixture_evidence_is_external_validation": False,
        },
        "criteria": criteria,
    }


def validate_collection_readiness_report(
    path: Path,
    *,
    study_id: str,
    study_config_sha256: str,
    evaluation_protocol_sha256: str,
    readiness_protocol_sha256: str,
    storage_policy_sha256: str,
    source_registry_sha256: str,
    model_id: str,
    model_revision: str,
    direction_sha256: str,
) -> dict[str, Any]:
    """Reject an execution report unless every bound criterion passed."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"Cannot read readiness report {path}: {exc}") from exc
    if report.get("schema_version") != READINESS_SCHEMA_VERSION:
        raise ReadinessError("Readiness report schema_version mismatch")
    if report.get("readiness_id") != READINESS_ID or report.get("study_id") != study_id:
        raise ReadinessError("Readiness report identity mismatch")
    decision = report.get("decision", {})
    if decision.get("collection_ready") is not True or decision.get("blockers") != []:
        raise ReadinessError("Readiness report blocks confirmatory collection")
    if (
        decision.get("decision_rule") != "all_required_criteria_must_pass"
        or decision.get("required_criteria_count") != len(REQUIRED_CRITERIA)
        or decision.get("passed_criteria_count") != len(REQUIRED_CRITERIA)
        or decision.get("blocked_criteria_count") != 0
    ):
        raise ReadinessError("Readiness report decision counts or rule mismatch")
    criteria = report.get("criteria")
    if not isinstance(criteria, list) or [item.get("criterion_id") for item in criteria] != list(
        REQUIRED_CRITERIA
    ):
        raise ReadinessError("Readiness report criterion set or order mismatch")
    if any(
        item.get("required") is not True
        or item.get("status") != "passed"
        or item.get("blockers") != []
        or not isinstance(item.get("evidence"), list)
        for item in criteria
    ):
        raise ReadinessError("Every readiness criterion must be required and passed")
    authorities = report.get("authorities", {})
    expected_authorities = {
        "study_config": study_config_sha256,
        "evaluation_protocol": evaluation_protocol_sha256,
        "readiness_protocol": readiness_protocol_sha256,
        "storage_policy": storage_policy_sha256,
        "source_registry": source_registry_sha256,
    }
    for name, expected_sha256 in expected_authorities.items():
        record = authorities.get(name, {})
        if record.get("sha256") != expected_sha256:
            raise ReadinessError(f"Readiness report {name} checksum mismatch")
        _reverify_report_record(record, label=name)
    expected_scope = {
        "confirmatory_images_generated": False,
        "confirmatory_matrix_executed": False,
        "expensive_sdxl_model_loaded": False,
        "real_person_images_used_by_synthetic_checks": False,
        "synthetic_fixture_evidence_is_external_validation": False,
    }
    if report.get("validation_scope") != expected_scope:
        raise ReadinessError("Readiness report validation scope mismatch")
    current_runtime = _runtime_criterion()
    if current_runtime["status"] != "passed":
        raise ReadinessError("Current process is not on the frozen supported metric runtime")
    provenance = report.get("inputs", {}).get("generation_provenance") or {}
    expected = {
        "model_id": model_id,
        "model_revision": model_revision,
        "direction_sha256": direction_sha256,
    }
    if any(provenance.get(field) != value for field, value in expected.items()):
        raise ReadinessError("Readiness report generation provenance mismatch")
    by_id = {item["criterion_id"]: item for item in criteria}
    sensitivity_evidence = by_id["illumination_and_color_sensitivity"]["evidence"]
    expected_sensitivity = _load_yaml(DEFAULT_READINESS_PROTOCOL_PATH)["criteria"][
        "illumination_and_color_sensitivity"
    ]["required_checks"]
    if [item.get("check_id") for item in sensitivity_evidence] != expected_sensitivity or any(
        item.get("status") != "passed" for item in sensitivity_evidence
    ):
        raise ReadinessError("Synthetic sensitivity evidence set is incomplete")
    external_evidence = by_id["external_target_metric_validity"]["evidence"]
    if len(external_evidence) != 1 or external_evidence[0].get("status") != "passed":
        raise ReadinessError("External-validation evidence record is missing")
    _reverify_report_record(
        external_evidence[0].get("record", {}), label="external_validation_record"
    )
    external_artifacts = external_evidence[0].get("artifact_verifications", [])
    if not external_artifacts:
        raise ReadinessError("External-validation artifacts are missing")
    for item in external_artifacts:
        _reverify_artifact_verification(item, label="external_validation_artifact")
    metric_evidence = by_id["checksum_verified_metric_artifacts"]["evidence"]
    frozen_artifacts = load_protocol()["required_artifacts"]
    if {item.get("name") for item in metric_evidence} != set(frozen_artifacts):
        raise ReadinessError("Metric-artifact evidence set is incomplete")
    for item in metric_evidence:
        if item.get("expected_sha256") != frozen_artifacts[item["name"]]["sha256"]:
            raise ReadinessError("Metric-artifact expected checksum drifted")
        _reverify_artifact_verification(item, label="metric_artifact")
    provenance_evidence = by_id["immutable_generation_provenance"]["evidence"]
    if len(provenance_evidence) != 1 or provenance_evidence[0].get("status") != "passed":
        raise ReadinessError("Generation-provenance evidence record is missing")
    _reverify_report_record(
        provenance_evidence[0].get("record", {}), label="generation_provenance_record"
    )
    _reverify_artifact_verification(
        provenance_evidence[0].get("direction_artifact", {}), label="direction_artifact"
    )
    _reverify_artifact_verification(
        provenance_evidence[0].get("direction_source_manifest", {}),
        label="direction_source_manifest",
    )
    policy_evidence = by_id["storage_and_privacy"]["evidence"]
    if len(policy_evidence) != 1 or policy_evidence[0].get("status") != "passed":
        raise ReadinessError("Storage/privacy evidence record is missing")
    approval_evidence = by_id["explicit_collection_approval"]["evidence"]
    if (
        len(approval_evidence) != 1
        or approval_evidence[0].get("status") != "passed"
        or approval_evidence[0].get("decision") != "approved"
        or approval_evidence[0].get("scope") != "confirmatory_collection"
    ):
        raise ReadinessError("Explicit collection approval evidence is missing")
    _reverify_report_record(
        approval_evidence[0].get("record", {}), label="collection_approval_record"
    )
    rebuilt = build_readiness_report(
        project_root=PROJECT_ROOT,
        study_config_path=Path(authorities["study_config"]["path"]),
        evaluation_protocol_path=Path(authorities["evaluation_protocol"]["path"]),
        readiness_protocol_path=Path(authorities["readiness_protocol"]["path"]),
        storage_policy_path=Path(authorities["storage_policy"]["path"]),
        source_registry_path=Path(authorities["source_registry"]["path"]),
        external_validation_path=Path(external_evidence[0]["record"]["path"]),
        generation_provenance_path=Path(provenance_evidence[0]["record"]["path"]),
        approval_path=Path(approval_evidence[0]["record"]["path"]),
        artifact_overrides={item["name"]: Path(item["path"]) for item in metric_evidence},
    )
    if rebuilt != report:
        raise ReadinessError("Readiness report does not reproduce from its bound inputs")
    return report


def _reverify_report_record(record: Mapping[str, Any], *, label: str) -> None:
    path = record.get("path")
    expected = record.get("sha256")
    if not isinstance(path, str) or not isinstance(expected, str):
        raise ReadinessError(f"{label} does not contain a checksum-bound path")
    result = inspect_artifact(path, expected, name=label)
    if not result.verified or result.size_bytes != record.get("size_bytes"):
        raise ReadinessError(f"{label} no longer matches the readiness report")


def _reverify_artifact_verification(record: Mapping[str, Any], *, label: str) -> None:
    path = record.get("path")
    expected = record.get("expected_sha256")
    if not isinstance(path, str) or not isinstance(expected, str):
        raise ReadinessError(f"{label} verification record is incomplete")
    result = inspect_artifact(path, expected, name=label)
    if (
        not result.verified
        or record.get("verified") is not True
        or record.get("status") != "verified"
        or result.actual_sha256 != record.get("actual_sha256")
        or result.size_bytes != record.get("size_bytes")
    ):
        raise ReadinessError(f"{label} no longer matches the readiness report")

"""Deterministic analysis for held-out calibrated-instrument validation data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.metrics.artifacts import sha256_file

REPORT_SCHEMA_VERSION = "1.0"
REPORT_TYPE = "external_target_metric_analysis_v1"
REQUIRED_COLUMNS = (
    "person_id",
    "pair_id",
    "reference_lstar_before",
    "reference_lstar_after",
    "metric_relative_lstar_before",
    "metric_relative_lstar_after",
    "detector_status",
    "mask_status",
)


class ExternalValidationError(ValueError):
    """Raised when validation inputs cannot support a trustworthy analysis."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExternalValidationError(f"Cannot read validation manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalValidationError("Validation manifest must be a mapping")
    return value


def _finite(value: str, *, field: str, row_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ExternalValidationError(f"row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ExternalValidationError(f"row {row_number}: {field} must be finite")
    return number


def _optional_finite(value: str, *, field: str, row_number: int) -> float | None:
    if value.strip() == "":
        return None
    return _finite(value, field=field, row_number=row_number)


def _percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="linear"))


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = np.asarray([row["absolute_delta_error"] for row in rows], dtype=float)
    directions = np.asarray([row["direction_correct"] for row in rows], dtype=float)
    return {
        "median_absolute_delta_error": float(np.median(errors)),
        "percentile_95_absolute_delta_error": _percentile(errors, 0.95),
        "direction_agreement": float(np.mean(directions)),
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]], *, resamples: int, confidence: float, rng_seed: int
) -> dict[str, dict[str, float]]:
    by_person: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_person.setdefault(row["person_id"], []).append(row)
    people = sorted(by_person)
    if not people:
        raise ExternalValidationError("No successful eligible comparisons are available")
    rng = np.random.default_rng(rng_seed)
    samples = {name: np.empty(resamples, dtype=float) for name in _metrics(rows)}
    for index in range(resamples):
        selected = rng.integers(0, len(people), size=len(people))
        sampled_rows = [row for position in selected for row in by_person[people[position]]]
        values = _metrics(sampled_rows)
        for name, value in values.items():
            samples[name][index] = value
    tail = (1.0 - confidence) / 2.0
    return {
        name: {
            "lower": _percentile(values, tail),
            "upper": _percentile(values, 1.0 - tail),
        }
        for name, values in samples.items()
    }


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ExternalValidationError(f"Cannot read observations CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ExternalValidationError(
                "Observation columns or order drifted; expected " + ",".join(REQUIRED_COLUMNS)
            )
        rows = []
        seen_pairs = set()
        for row_number, raw in enumerate(reader, start=2):
            person_id = raw["person_id"].strip()
            pair_id = raw["pair_id"].strip()
            if not person_id or not pair_id:
                raise ExternalValidationError(f"row {row_number}: person_id and pair_id required")
            if pair_id in seen_pairs:
                raise ExternalValidationError(f"row {row_number}: duplicate pair_id {pair_id}")
            seen_pairs.add(pair_id)
            reference_before = _finite(
                raw["reference_lstar_before"],
                field="reference_lstar_before",
                row_number=row_number,
            )
            reference_after = _finite(
                raw["reference_lstar_after"],
                field="reference_lstar_after",
                row_number=row_number,
            )
            if not (0.0 <= reference_before <= 100.0 and 0.0 <= reference_after <= 100.0):
                raise ExternalValidationError(f"row {row_number}: reference L* outside [0, 100]")
            detector_status = raw["detector_status"].strip()
            mask_status = raw["mask_status"].strip()
            if detector_status not in {"passed", "failed"} or mask_status not in {
                "passed",
                "failed",
            }:
                raise ExternalValidationError(
                    f"row {row_number}: detector_status and mask_status must be passed/failed"
                )
            metric_before = _optional_finite(
                raw["metric_relative_lstar_before"],
                field="metric_relative_lstar_before",
                row_number=row_number,
            )
            metric_after = _optional_finite(
                raw["metric_relative_lstar_after"],
                field="metric_relative_lstar_after",
                row_number=row_number,
            )
            measurement_succeeded = detector_status == "passed" and mask_status == "passed"
            if measurement_succeeded != (metric_before is not None and metric_after is not None):
                raise ExternalValidationError(
                    f"row {row_number}: metric values must be present exactly when detector and mask pass"
                )
            reference_delta = reference_after - reference_before
            metric_delta = metric_after - metric_before if measurement_succeeded else None
            rows.append(
                {
                    "person_id": person_id,
                    "pair_id": pair_id,
                    "reference_lstar_before": reference_before,
                    "reference_lstar_after": reference_after,
                    "reference_delta": reference_delta,
                    "metric_delta": metric_delta,
                    "detector_status": detector_status,
                    "mask_status": mask_status,
                    "measurement_succeeded": measurement_succeeded,
                }
            )
    if not rows:
        raise ExternalValidationError("Observation CSV is empty")
    return rows


def analyze_external_validation(
    manifest_path: Path,
    *,
    readiness_protocol_path: Path,
) -> dict[str, Any]:
    """Analyze a full-denominator portrait/instrument comparison manifest."""
    manifest_path = manifest_path.resolve()
    readiness_protocol_path = readiness_protocol_path.resolve()
    manifest = _load_yaml(manifest_path)
    readiness = _load_yaml(readiness_protocol_path)
    requirements = readiness["criteria"]["external_target_metric_validity"]
    if manifest.get("schema_version") != "1.0":
        raise ExternalValidationError("Unsupported validation manifest schema_version")
    validation_id = manifest.get("validation_id")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ExternalValidationError("validation_id is required")
    if manifest.get("metric_id") != requirements["metric_id"]:
        raise ExternalValidationError("Validation manifest metric_id drifted")
    analysis = manifest.get("analysis", {})
    minimum_change = float(requirements["agreement"]["minimum_reference_change"])
    confidence = float(requirements["agreement"]["confidence_level"])
    if float(analysis.get("minimum_reference_change", math.nan)) != minimum_change:
        raise ExternalValidationError("minimum_reference_change must match the frozen threshold")
    if float(analysis.get("confidence_level", math.nan)) != confidence:
        raise ExternalValidationError("confidence_level must match the frozen threshold")
    try:
        resamples = int(analysis["bootstrap_resamples"])
        rng_seed = int(analysis["rng_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExternalValidationError("bootstrap_resamples and rng_seed must be integers") from exc
    if resamples < 1000 or rng_seed < 0:
        raise ExternalValidationError("bootstrap_resamples must be >=1000 and rng_seed nonnegative")
    observations = manifest.get("observations", {})
    relative_path = observations.get("path")
    expected_sha256 = observations.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        raise ExternalValidationError("observations path and sha256 are required")
    observation_path = (manifest_path.parent / relative_path).resolve()
    actual_sha256, observation_size = sha256_file(observation_path)
    if actual_sha256 != expected_sha256:
        raise ExternalValidationError("observations CSV checksum mismatch")
    rows = _parse_rows(observation_path)
    eligible = [row for row in rows if abs(row["reference_delta"]) >= minimum_change]
    if not eligible:
        raise ExternalValidationError("No comparison meets the frozen minimum reference change")
    successful = []
    for row in eligible:
        if row["measurement_succeeded"]:
            metric_delta = float(row["metric_delta"])
            successful.append(
                {
                    **row,
                    "absolute_delta_error": abs(metric_delta - row["reference_delta"]),
                    "direction_correct": bool(metric_delta * row["reference_delta"] > 0),
                }
            )
    independent_people = len({row["person_id"] for row in eligible})
    reference_values = [
        value
        for row in eligible
        for value in (row["reference_lstar_before"], row["reference_lstar_after"])
    ]
    reference_span = float(max(reference_values) - min(reference_values))
    detector_failures = sum(row["detector_status"] == "failed" for row in eligible)
    mask_failures = sum(row["mask_status"] == "failed" for row in eligible)
    failure_rows = len(eligible) - len(successful)
    full_denominator_complete = len(successful) + failure_rows == len(eligible)
    observed = _metrics(successful) if successful else None
    intervals = (
        _cluster_bootstrap(
            successful, resamples=resamples, confidence=confidence, rng_seed=rng_seed
        )
        if successful
        else None
    )
    thresholds_evaluable = failure_rows == 0 and full_denominator_complete
    limits = requirements["agreement"]
    threshold_checks = {
        "minimum_independent_people": independent_people
        >= int(requirements["domain"]["minimum_independent_people"]),
        "minimum_reference_lstar_span": reference_span
        >= float(requirements["domain"]["minimum_reference_lstar_span"]),
        "median_absolute_delta_error": bool(
            thresholds_evaluable
            and observed
            and observed["median_absolute_delta_error"]
            <= float(limits["maximum_median_absolute_delta_error"])
        ),
        "percentile_95_absolute_delta_error": bool(
            thresholds_evaluable
            and observed
            and observed["percentile_95_absolute_delta_error"]
            <= float(limits["maximum_95th_percentile_absolute_delta_error"])
        ),
        "direction_agreement": bool(
            thresholds_evaluable
            and observed
            and observed["direction_agreement"] >= float(limits["minimum_direction_agreement"])
        ),
        "full_denominator_no_undefined_metric_outcomes": thresholds_evaluable,
    }
    manifest_sha256, manifest_size = sha256_file(manifest_path)
    readiness_sha256, readiness_size = sha256_file(readiness_protocol_path)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "validation_id": validation_id,
        "metric_id": requirements["metric_id"],
        "authorities": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "size_bytes": manifest_size,
            },
            "readiness_protocol": {
                "path": str(readiness_protocol_path),
                "sha256": readiness_sha256,
                "size_bytes": readiness_size,
            },
            "observations": {
                "path": str(observation_path),
                "sha256": actual_sha256,
                "size_bytes": observation_size,
            },
        },
        "analysis": {
            "minimum_reference_change": minimum_change,
            "confidence_level": confidence,
            "bootstrap_resamples": resamples,
            "rng_seed": rng_seed,
            "percentile_method": "linear",
            "bootstrap_cluster_unit": "independent_person",
        },
        "domain": {
            "independent_people_in_eligible_denominator": independent_people,
            "reference_lstar_span": reference_span,
        },
        "denominators": {
            "planned_comparisons": len(rows),
            "eligible_reference_change_comparisons": len(eligible),
            "metric_successes": len(successful),
            "detector_failure_rows": detector_failures,
            "mask_failure_rows": mask_failures,
            "any_detector_or_mask_failure_rows": failure_rows,
            "full_denominator_reported": full_denominator_complete,
            "detector_and_mask_failures_are_outcomes": True,
        },
        "agreement": {
            "successful_measurements_only": observed,
            "confidence_intervals_successful_measurements_only": intervals,
            "acceptance_values_available": observed if thresholds_evaluable else None,
        },
        "decision": {
            "technical_thresholds_evaluable": thresholds_evaluable,
            "technical_thresholds_met": all(threshold_checks.values()),
            "threshold_checks": threshold_checks,
            "external_validity_approved": False,
            "independent_review_supplied": False,
            "collection_approval_implied": False,
        },
        "limitations": [
            "This analysis does not authenticate consent, licenses, calibration, or reviewers.",
            "Synthetic fixtures can test this code but cannot supply external-validity evidence.",
            "Any eligible detector or mask failure leaves acceptance values unavailable rather than complete-case passing.",
        ],
    }


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a stable JSON analysis report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_input_id(path: Path) -> str:
    """Return a short diagnostic fingerprint without exposing observation contents."""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return digest[:16]

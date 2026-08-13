import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.analyze_external_validation import main as analyze_main
from scripts.prepare_collection_approval import main as prepare_approval_main
from src.validation.external_validation import (
    ExternalValidationError,
    analyze_external_validation,
)

ROOT = Path(__file__).parents[1]
READINESS = ROOT / "configs" / "step6_readiness.yaml"
COLUMNS = (
    "person_id",
    "pair_id",
    "reference_lstar_before",
    "reference_lstar_after",
    "metric_relative_lstar_before",
    "metric_relative_lstar_after",
    "detector_status",
    "mask_status",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(tmp_path, *, failed_index=None):
    observations = tmp_path / "observations.csv"
    with observations.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index in range(30):
            failed = index == failed_index
            before = 20.0 + index * (30.0 / 29.0)
            after = before + 2.5
            writer.writerow(
                {
                    "person_id": f"person-{index:02d}",
                    "pair_id": f"pair-{index:02d}",
                    "reference_lstar_before": before,
                    "reference_lstar_after": after,
                    "metric_relative_lstar_before": "" if failed else -10.0,
                    "metric_relative_lstar_after": "" if failed else -7.7,
                    "detector_status": "failed" if failed else "passed",
                    "mask_status": "passed",
                }
            )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "validation_id": "synthetic_analysis_fixture_v1",
                "metric_id": "relative_cheek_CIELAB_Lstar_v1",
                "observations": {"path": observations.name, "sha256": digest(observations)},
                "analysis": {
                    "minimum_reference_change": 2.0,
                    "confidence_level": 0.95,
                    "bootstrap_resamples": 1000,
                    "rng_seed": 20260813,
                },
            },
            sort_keys=False,
        )
    )
    return manifest, observations


def test_external_analysis_is_deterministic_and_never_grants_approval(tmp_path):
    manifest, _ = write_fixture(tmp_path)

    first = analyze_external_validation(manifest, readiness_protocol_path=READINESS)
    second = analyze_external_validation(manifest, readiness_protocol_path=READINESS)

    assert first == second
    assert first["decision"]["technical_thresholds_evaluable"]
    assert first["decision"]["technical_thresholds_met"]
    assert not first["decision"]["external_validity_approved"]
    assert not first["decision"]["collection_approval_implied"]
    assert first["denominators"] == {
        "planned_comparisons": 30,
        "eligible_reference_change_comparisons": 30,
        "metric_successes": 30,
        "detector_failure_rows": 0,
        "mask_failure_rows": 0,
        "any_detector_or_mask_failure_rows": 0,
        "full_denominator_reported": True,
        "detector_and_mask_failures_are_outcomes": True,
    }


def test_detector_failure_stays_in_denominator_and_blocks_acceptance_values(tmp_path):
    manifest, _ = write_fixture(tmp_path, failed_index=7)

    report = analyze_external_validation(manifest, readiness_protocol_path=READINESS)

    assert report["denominators"]["planned_comparisons"] == 30
    assert report["denominators"]["metric_successes"] == 29
    assert report["denominators"]["detector_failure_rows"] == 1
    assert report["denominators"]["any_detector_or_mask_failure_rows"] == 1
    assert report["agreement"]["acceptance_values_available"] is None
    assert not report["decision"]["technical_thresholds_evaluable"]
    assert not report["decision"]["technical_thresholds_met"]


def test_external_analysis_cli_writes_report_before_nonpassing_exit(tmp_path):
    manifest, _ = write_fixture(tmp_path, failed_index=3)
    output = tmp_path / "analysis.json"

    exit_code = analyze_main([str(manifest), "--output", str(output)])

    assert exit_code == 2
    assert json.loads(output.read_text())["denominators"]["detector_failure_rows"] == 1


def test_external_analysis_rejects_checksum_drift(tmp_path):
    manifest, observations = write_fixture(tmp_path)
    observations.write_text(observations.read_text() + "tampered")

    with pytest.raises(ExternalValidationError, match="checksum mismatch"):
        analyze_external_validation(manifest, readiness_protocol_path=READINESS)


def test_prepared_approval_is_bound_but_remains_pending(tmp_path):
    external = tmp_path / "external.yaml"
    external.write_text("validation_id: real_validation_id_required\n")
    direction = tmp_path / "direction.pt"
    direction.write_bytes(b"synthetic test-only direction")
    provenance = tmp_path / "provenance.yaml"
    provenance.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "requested_revision": "a" * 40,
                    "resolved_revision": "a" * 40,
                },
                "direction": {"artifact": {"path": direction.name, "sha256": digest(direction)}},
            }
        )
    )
    output = tmp_path / "pending_approval.yaml"

    assert (
        prepare_approval_main(
            [
                "--external-validation",
                str(external),
                "--generation-provenance",
                str(provenance),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    document = yaml.safe_load(output.read_text())
    assert document["decision"] == "pending_authorized_approver"
    assert document["approved_by"] is None
    assert document["direction_sha256"] == digest(direction)
    assert not document["notes"]["template_is_approval"]

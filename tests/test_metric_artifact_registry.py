import hashlib
from pathlib import Path

import pytest
import yaml

from scripts.download_metric_models import _registry
from scripts.download_metric_models import main as download_main
from src.metrics.protocol import load_protocol
from src.validation.readiness import ReadinessError, _validate_metric_artifact_registry

ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "configs" / "metric_artifacts.yaml"
PROTOCOL = ROOT / "configs" / "evaluation_protocol.yaml"
ENVIRONMENT = ROOT / "containers" / "evaluation" / "environment.yaml"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_linux_environment_authority_binds_lock_and_platform():
    environment = yaml.safe_load(ENVIRONMENT.read_text())
    lock = ROOT / environment["requirements_lock"]["path"]

    assert environment["platform"] == "linux/amd64"
    assert environment["base_image"]["manifest_sha256"] == (
        "c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49"
    )
    assert environment["requirements_lock"]["sha256"] == digest(lock)
    assert environment["scope"] == {
        "evaluation_and_repository_verification_only": True,
        "sdxl_generation_dependencies_included": False,
        "model_weights_included": False,
    }


def test_metric_artifact_registry_matches_every_frozen_hash_and_retains_license_gaps():
    registry = _registry(REGISTRY)
    protocol = load_protocol(PROTOCOL)

    assert tuple(registry["artifacts"]) == tuple(protocol["required_artifacts"])
    for name, item in registry["artifacts"].items():
        assert item["sha256"] == protocol["required_artifacts"][name]["sha256"]
        assert item["source_url"].startswith("https://")
        assert item["weight_redistribution_status"].startswith("unresolved_")


def test_metric_artifact_registry_drift_fails_closed():
    registry = yaml.safe_load(REGISTRY.read_text())
    registry["artifacts"]["mediapipe_face_landmarker"]["sha256"] = "0" * 64

    with pytest.raises(ReadinessError, match="checksum drifted"):
        _validate_metric_artifact_registry(registry, load_protocol(PROTOCOL))


def test_verify_only_writes_blocked_local_manifest_without_downloading(tmp_path):
    manifest = tmp_path / "manifest.json"

    exit_code = download_main(
        [
            "--verify-only",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--manifest",
            str(manifest),
        ]
    )

    assert exit_code == 2
    assert manifest.is_file()

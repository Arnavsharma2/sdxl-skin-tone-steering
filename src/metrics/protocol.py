"""Loader and consistency checks for the frozen evaluation protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .artifacts import sha256_file
from .face_landmarks import MEDIAPIPE_VERSION, MODEL_SHA256, SUPPORTED_SYSTEMS
from .identity_metrics import (
    ALEXNET_SHA256,
    FACENET_VGGFACE2_SHA256,
    LPIPS_ALEX_V01_SHA256,
    MTCNN_SHA256,
)
from .target_response import FROZEN_ALPHA_GRID

PROTOCOL_ID = "tmlr_evaluation_protocol_v1"
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "configs" / "evaluation_protocol.yaml"
FROZEN_THRESHOLDS = {
    "face_similarity": 0.85,
    "landmark_rmse": 5.0,
    "lpips": 0.3,
    "background_ssim": 0.75,
    "pose_angle_diff": 5.0,
    "min_abs_skin_tone_change": 2.0,
}


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Load the protocol and reject drift from code-level artifact/runtime pins."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Invalid frozen evaluation protocol: {path}")
    expected_artifacts = {
        "mediapipe_face_landmarker": MODEL_SHA256,
        "facenet_vggface2": FACENET_VGGFACE2_SHA256,
        "mtcnn_pnet": MTCNN_SHA256["pnet.pt"],
        "mtcnn_rnet": MTCNN_SHA256["rnet.pt"],
        "mtcnn_onet": MTCNN_SHA256["onet.pt"],
        "alexnet_backbone": ALEXNET_SHA256,
        "lpips_alex_v0.1": LPIPS_ALEX_V01_SHA256,
    }
    artifacts = document.get("required_artifacts", {})
    actual_artifacts = {
        name: entry.get("sha256") if isinstance(entry, dict) else None
        for name, entry in artifacts.items()
    }
    if actual_artifacts != expected_artifacts:
        raise ValueError("Frozen artifact checksums do not match metric code")
    runtime = document.get("runtime", {})
    if runtime.get("supported_operating_systems") != list(SUPPORTED_SYSTEMS):
        raise ValueError("Frozen supported operating systems do not match metric code")
    if runtime.get("mediapipe") != MEDIAPIPE_VERSION:
        raise ValueError("Frozen MediaPipe version does not match metric code")
    if document.get("thresholds") != FROZEN_THRESHOLDS:
        raise ValueError("Frozen thresholds do not match metric code")
    monotonicity = document.get("metrics", {}).get("monotonicity", {})
    if monotonicity.get("expected_alphas") != list(FROZEN_ALPHA_GRID):
        raise ValueError("Frozen alpha grid does not match metric code")
    return document


def protocol_record(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Return the exact document and its actual file checksum."""
    document = load_protocol(path)
    digest, size_bytes = sha256_file(path)
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": size_bytes,
        "document": document,
    }

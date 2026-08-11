"""Loader and exact consistency checks for the frozen evaluation protocol."""

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
PYTHON_CONSTRAINT = ">=3.10,<3.13"
FACE_LANDMARKER_DELEGATE = "CPU"
UNSUPPORTED_RUNTIME_BEHAVIOR = "fail before MediaPipe graph construction"
FROZEN_THRESHOLDS = {
    "face_similarity": 0.85,
    "landmark_rmse": 5.0,
    "lpips": 0.3,
    "background_ssim": 0.75,
    "pose_angle_diff": 5.0,
    "min_abs_skin_tone_change": 2.0,
}
FROZEN_REQUIRED_PAIR_METRICS = [
    "skin_tone_change",
    "target_direction_correct",
    "face_similarity",
    "lpips",
    "background_ssim",
    "total_pose_diff",
]
FROZEN_RUNTIME = {
    "supported_operating_systems": list(SUPPORTED_SYSTEMS),
    "python": PYTHON_CONSTRAINT,
    "mediapipe": MEDIAPIPE_VERSION,
    "face_landmarker_delegate": FACE_LANDMARKER_DELEGATE,
    "unsupported_runtime_behavior": UNSUPPORTED_RUNTIME_BEHAVIOR,
}
FROZEN_ARTIFACTS = {
    "mediapipe_face_landmarker": {
        "location": "models/face_landmarker.task",
        "sha256": MODEL_SHA256,
    },
    "facenet_vggface2": {
        "location": "${TORCH_HOME}/checkpoints/20180402-114759-vggface2.pt",
        "sha256": FACENET_VGGFACE2_SHA256,
    },
    "mtcnn_pnet": {
        "location": "facenet_pytorch/data/pnet.pt",
        "sha256": MTCNN_SHA256["pnet.pt"],
    },
    "mtcnn_rnet": {
        "location": "facenet_pytorch/data/rnet.pt",
        "sha256": MTCNN_SHA256["rnet.pt"],
    },
    "mtcnn_onet": {
        "location": "facenet_pytorch/data/onet.pt",
        "sha256": MTCNN_SHA256["onet.pt"],
    },
    "alexnet_backbone": {
        "location": "${TORCH_HOME}/hub/checkpoints/alexnet-owt-7be5be79.pth",
        "sha256": ALEXNET_SHA256,
    },
    "lpips_alex_v0.1": {
        "location": "lpips/weights/v0.1/alex.pth",
        "sha256": LPIPS_ALEX_V01_SHA256,
    },
}
FROZEN_METRICS = {
    "target_response": {
        "id": "relative_cheek_CIELAB_Lstar_v1",
        "input": "uint8 RGB",
        "face_regions_id": "geometric_cheek_mask",
        "face_regions": "bilateral geometric cheek mask inside MediaPipe face oval",
        "excluded_regions": ["eyes", "eyebrows", "lips"],
        "skin_minimum_pixels_before_and_after_trim": 256,
        "lstar_trim_quantiles": [0.10, 0.90],
        "statistic": "median CIELAB L*, a*, b*",
        "illumination_reference_id": "neutral_portrait_border",
        "illumination_reference": "median CIELAB over upper 42% of outer 16% borders",
        "reference_minimum_pixels": 512,
        "reference_max_chroma": 14.0,
        "pair_max_absolute_reference_lstar_shift": 8.0,
        "outcome": "delta of (skin L* - reference L*)",
        "direction_gate": "alpha * target change < 0",
    },
    "identity_similarity": {
        "detector": "MTCNN image_size=160 margin=0 post_process=true",
        "embedding": "InceptionResnetV1 pretrained=vggface2",
        "outcome": "cosine similarity in [-1, 1]",
    },
    "perceptual_distance": {
        "id": "LPIPS AlexNet v0.1",
        "preprocessing": "uint8 RGB mapped to one NCHW batch in [-1, 1]",
    },
    "pose": {
        "source": "MediaPipe 4x4 facial transformation matrix",
        "validation": "finite proper orthonormal 3x3 rotation block",
        "angles": "yaw, pitch, roll in degrees with circular absolute differences",
        "outcome": "Euclidean norm of yaw, pitch, and roll differences",
    },
    "background_ssim": {
        "input": "same-size uint8 images converted to grayscale",
        "face_region": "union of 1.5x expanded face boxes from both images",
        "background": "inverse union mask eroded with a 7x7 kernel",
        "minimum_background_pixels": 49,
        "outcome": "mean full SSIM map over valid background pixels only",
    },
    "monotonicity": {
        "expected_alphas": list(FROZEN_ALPHA_GRID),
        "minimum_points": 3,
        "outcome": "Spearman rho and fraction of strictly decreasing adjacent responses",
        "invalid_when": "duplicate, missing, unexpected, nonfinite, or constant response",
    },
}
FROZEN_FAILURE_SEMANTICS = {
    "face_detection": "invalid metric and incomplete row; count in failure rate",
    "skin_or_reference_mask": "invalid target metric and incomplete row",
    "unstable_illumination": "invalid target change and incomplete row",
    "pose": "invalid metric and incomplete row",
    "background_mask": "invalid metric and incomplete row",
    "identity_preprocessing_or_detection": "invalid metric and incomplete row",
    "artifact_verification": "missing, unreadable, or checksum mismatch fails closed",
    "monotonicity": "invalid sweep; never calculate on a complete-case subset",
    "fallbacks": "prohibited",
}
FROZEN_PROTOCOL = {
    "schema_version": "1.0",
    "protocol_id": PROTOCOL_ID,
    "status": "frozen",
    "construct": "rendered visual skin tone; not race or ethnicity",
    "runtime": FROZEN_RUNTIME,
    "required_artifacts": FROZEN_ARTIFACTS,
    "metrics": FROZEN_METRICS,
    "required_pair_metrics": FROZEN_REQUIRED_PAIR_METRICS,
    "thresholds": FROZEN_THRESHOLDS,
    "failure_semantics": FROZEN_FAILURE_SEMANTICS,
}


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Load the protocol and reject any drift from code-level pins."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document != FROZEN_PROTOCOL:
        raise ValueError(f"Frozen evaluation protocol does not match metric code: {path}")
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

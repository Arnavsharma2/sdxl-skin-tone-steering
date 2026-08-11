"""CPU-only, checksum-verified MediaPipe Tasks face landmarks."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from .artifacts import ArtifactVerification, inspect_artifact, require_verified

MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_FILENAME = "face_landmarker.task"
MEDIAPIPE_VERSION = "0.10.21"
SUPPORTED_SYSTEMS = ("Linux",)


@dataclass(frozen=True)
class FaceLandmarkResult:
    landmarks: tuple
    transformation_matrix: Optional[np.ndarray]


class FaceLandmarkBackend:
    """Run the MediaPipe Face Landmarker task with the CPU delegate."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        configured = model_path or os.environ.get("SKIN_TONE_FACE_LANDMARKER_MODEL")
        self.model_path = (
            Path(configured)
            if configured
            else Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME
        )
        self.landmarker = None
        self._mp = None
        self.artifact_verification: ArtifactVerification | None = None

    @staticmethod
    def validate_runtime(
        system: str | None = None,
        python_version: tuple[int, int] | None = None,
    ) -> None:
        """Reject platforms on which the frozen task cannot run headlessly."""
        detected = system or platform.system()
        if detected not in SUPPORTED_SYSTEMS:
            raise RuntimeError(
                "Unsupported MediaPipe evaluation runtime: "
                f"{detected or 'unknown'}. The frozen protocol supports Linux only. "
                "MediaPipe 0.10.21 Face Landmarker requires an OpenGL pixel format "
                "on macOS even with the CPU delegate, so headless macOS is rejected "
                "instead of returning missing or fallback metrics."
            )
        version = python_version or sys.version_info[:2]
        if not (version >= (3, 10) and version < (3, 13)):
            raise RuntimeError(
                "Unsupported metric Python runtime: expected >=3.10,<3.13; "
                f"found {version[0]}.{version[1]}"
            )

    def _load(self) -> None:
        if self.landmarker is not None:
            return
        self.artifact_verification = inspect_artifact(
            self.model_path,
            MODEL_SHA256,
            name="MediaPipe Face Landmarker",
        )
        if self.artifact_verification.status == "missing":
            raise RuntimeError(
                f"Missing {self.model_path}; run `make metric-models` before evaluation"
            )
        require_verified(self.artifact_verification)
        self.validate_runtime()
        try:
            installed_version = metadata.version("mediapipe")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"MediaPipe {MEDIAPIPE_VERSION} is required for audited evaluation"
            ) from exc
        if installed_version != MEDIAPIPE_VERSION:
            raise RuntimeError(
                "Unsupported MediaPipe version: "
                f"expected {MEDIAPIPE_VERSION}, found {installed_version}"
            )
        try:
            import mediapipe as mp

            base_options = mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            )
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_facial_transformation_matrixes=True,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
        except Exception as exc:
            raise RuntimeError(f"MediaPipe Tasks Face Landmarker unavailable: {exc}") from exc

    @staticmethod
    def _as_rgb(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("Expected a uint8 RGB image with shape (height, width, 3)")
        return array

    def detect(
        self, image: Union[Image.Image, np.ndarray]
    ) -> Optional[FaceLandmarkResult]:
        self._load()
        rgb = np.ascontiguousarray(self._as_rgb(image))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=float)
        return FaceLandmarkResult(tuple(result.face_landmarks[0]), matrix)

"""CPU-only, checksum-pinned MediaPipe Tasks face landmarks."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_FILENAME = "face_landmarker.task"


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

    def _load(self) -> None:
        if self.landmarker is not None:
            return
        if not self.model_path.is_file():
            raise RuntimeError(
                f"Missing {self.model_path}; run `make metric-models` before evaluation"
            )
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise RuntimeError(
                f"Face Landmarker checksum mismatch: expected {MODEL_SHA256}, got {digest}"
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
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3)")
        return array.astype(np.uint8, copy=False)

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

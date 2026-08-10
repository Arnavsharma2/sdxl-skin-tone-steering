"""Illumination-audited skin-tone measurements for portrait sweeps.

The metric intentionally measures rendered colour, not race or ethnicity.  It
uses geometrically selected cheek pixels from MediaPipe Face Mesh and reports
CIELAB values relative to neutral border pixels.  The border reference removes
first-order global exposure shifts; a large reference shift still invalidates
the target metric instead of being silently corrected away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from .face_landmarks import FaceLandmarkBackend

FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)
LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
LEFT_EYEBROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
RIGHT_EYEBROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415)


@dataclass(frozen=True)
class SkinToneMeasurement:
    """One image's audited colour measurement."""

    skin_lstar: float
    skin_astar: float
    skin_bstar: float
    ita_degrees: float
    reference_lstar: float
    reference_chroma: float
    relative_lstar: float
    skin_pixel_count: int
    reference_pixel_count: int


class SkinToneMetrics:
    """Measure a continuous rendered skin-tone response in controlled portraits."""

    metric_name = "relative_cheek_CIELAB_Lstar_v1"

    def __init__(
        self,
        *,
        min_skin_pixels: int = 256,
        min_reference_pixels: int = 512,
        max_reference_chroma: float = 14.0,
        max_reference_shift: float = 8.0,
        landmark_backend: Optional[FaceLandmarkBackend] = None,
    ) -> None:
        self.min_skin_pixels = min_skin_pixels
        self.min_reference_pixels = min_reference_pixels
        self.max_reference_chroma = max_reference_chroma
        self.max_reference_shift = max_reference_shift
        self.landmark_backend = landmark_backend or FaceLandmarkBackend()

    @staticmethod
    def _as_rgb(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3)")
        return array.astype(np.uint8, copy=False)

    @staticmethod
    def _fill_feature(mask: np.ndarray, landmarks, indices, width: int, height: int) -> None:
        points = np.array(
            [[landmarks[index].x * width, landmarks[index].y * height] for index in indices],
            dtype=np.int32,
        )
        if len(points) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)

    def create_skin_mask(self, image: Union[Image.Image, np.ndarray]) -> Optional[np.ndarray]:
        """Return a geometric cheek mask, or ``None`` when no face is detected."""
        rgb = self._as_rgb(image)
        height, width = rgb.shape[:2]
        result = self.landmark_backend.detect(rgb)
        if result is None:
            return None

        landmarks = result.landmarks
        oval = np.array(
            [[landmarks[index].x * width, landmarks[index].y * height] for index in FACE_OVAL],
            dtype=np.int32,
        )
        hull = cv2.convexHull(oval)
        face = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(face, hull, 1)

        x, y, face_width, face_height = cv2.boundingRect(hull)
        cheek_band = np.zeros_like(face)
        y1 = max(0, y + int(0.34 * face_height))
        y2 = min(height, y + int(0.76 * face_height))
        left_x1 = max(0, x + int(0.08 * face_width))
        left_x2 = min(width, x + int(0.44 * face_width))
        right_x1 = max(0, x + int(0.56 * face_width))
        right_x2 = min(width, x + int(0.92 * face_width))
        cheek_band[y1:y2, left_x1:left_x2] = 1
        cheek_band[y1:y2, right_x1:right_x2] = 1

        features = np.zeros_like(face)
        for connections in (
            LEFT_EYE,
            RIGHT_EYE,
            LEFT_EYEBROW,
            RIGHT_EYEBROW,
            LIPS,
        ):
            self._fill_feature(features, landmarks, connections, width, height)
        dilation = max(3, int(round(face_width * 0.035)))
        if dilation % 2 == 0:
            dilation += 1
        features = cv2.dilate(features, np.ones((dilation, dilation), np.uint8))

        skin = face & cheek_band & (1 - features)
        erosion = max(3, int(round(face_width * 0.012)))
        if erosion % 2 == 0:
            erosion += 1
        skin = cv2.erode(skin, np.ones((erosion, erosion), np.uint8))
        return skin.astype(bool)

    @staticmethod
    def create_reference_mask(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Select portrait-border pixels expected to contain the neutral backdrop."""
        rgb = SkinToneMetrics._as_rgb(image)
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        side = max(1, int(round(width * 0.16)))
        top = max(1, int(round(height * 0.42)))
        mask[:top, :side] = True
        mask[:top, width - side :] = True
        return mask

    def measure(
        self,
        image: Union[Image.Image, np.ndarray],
        *,
        skin_mask: Optional[np.ndarray] = None,
        reference_mask: Optional[np.ndarray] = None,
    ) -> Optional[SkinToneMeasurement]:
        """Measure median CIELAB cheek colour with a neutral-background reference."""
        rgb = self._as_rgb(image)
        skin_mask = self.create_skin_mask(rgb) if skin_mask is None else skin_mask.astype(bool)
        if skin_mask is None or int(skin_mask.sum()) < self.min_skin_pixels:
            return None
        reference_mask = (
            self.create_reference_mask(rgb)
            if reference_mask is None
            else reference_mask.astype(bool)
        )
        if int(reference_mask.sum()) < self.min_reference_pixels:
            return None

        lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        skin_values = lab[skin_mask]
        reference_values = lab[reference_mask]

        # Reject extreme highlights and shadows before taking robust medians.
        low, high = np.quantile(skin_values[:, 0], [0.10, 0.90])
        skin_values = skin_values[(skin_values[:, 0] >= low) & (skin_values[:, 0] <= high)]
        if len(skin_values) < self.min_skin_pixels:
            return None

        skin_lab = np.median(skin_values, axis=0)
        reference_lab = np.median(reference_values, axis=0)
        reference_chroma = float(np.hypot(reference_lab[1], reference_lab[2]))
        if reference_chroma > self.max_reference_chroma:
            return None

        ita = float(np.degrees(np.arctan2(skin_lab[0] - 50.0, skin_lab[2])))
        return SkinToneMeasurement(
            skin_lstar=float(skin_lab[0]),
            skin_astar=float(skin_lab[1]),
            skin_bstar=float(skin_lab[2]),
            ita_degrees=ita,
            reference_lstar=float(reference_lab[0]),
            reference_chroma=reference_chroma,
            relative_lstar=float(skin_lab[0] - reference_lab[0]),
            skin_pixel_count=int(len(skin_values)),
            reference_pixel_count=int(len(reference_values)),
        )

    def compare(
        self,
        original: Union[Image.Image, np.ndarray],
        counterfactual: Union[Image.Image, np.ndarray],
    ) -> dict:
        """Return target-attribute metrics and explicit illumination QC."""
        first = self.measure(original)
        second = self.measure(counterfactual)
        if first is None or second is None:
            return {
                "skin_tone_metric": self.metric_name,
                "skin_tone_change": None,
                "skin_delta_ita": None,
                "skin_delta_e": None,
                "reference_lstar_shift": None,
                "illumination_stable": False,
            }

        reference_shift = second.reference_lstar - first.reference_lstar
        illumination_stable = abs(reference_shift) <= self.max_reference_shift
        delta_lab = np.array(
            [
                second.skin_lstar - first.skin_lstar,
                second.skin_astar - first.skin_astar,
                second.skin_bstar - first.skin_bstar,
            ]
        )
        return {
            "skin_tone_metric": self.metric_name,
            "skin_lstar_original": first.skin_lstar,
            "skin_lstar_counterfactual": second.skin_lstar,
            "skin_relative_lstar_original": first.relative_lstar,
            "skin_relative_lstar_counterfactual": second.relative_lstar,
            "skin_tone_change": (
                second.relative_lstar - first.relative_lstar
                if illumination_stable
                else None
            ),
            "skin_ita_original": first.ita_degrees,
            "skin_ita_counterfactual": second.ita_degrees,
            "skin_delta_ita": second.ita_degrees - first.ita_degrees,
            "skin_delta_e": float(np.linalg.norm(delta_lab)),
            "reference_lstar_shift": float(reference_shift),
            "illumination_stable": illumination_stable,
            "skin_pixel_count_original": first.skin_pixel_count,
            "skin_pixel_count_counterfactual": second.skin_pixel_count,
        }

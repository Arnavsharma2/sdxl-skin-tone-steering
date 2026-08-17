"""Transparent, illumination-sensitive skin-tone outcome measurements.

This module intentionally measures image colour rather than demographic
identity.  It uses conservative cheek patches inside a detected face and
reports CIE L*a*b* values plus the Individual Typology Angle (ITA).  ITA is a
useful auditable proxy for visible skin tone, but it is not invariant to
lighting, white balance, makeup, or rendering.  Study code must therefore
report it as a colourimetric outcome and validate its sensitivity before using
it for confirmatory claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SkinToneMeasurement:
    """Robust colour summary for conservative facial skin patches."""

    ita_degrees: float
    lab_l: float
    lab_a: float
    lab_b: float
    pixel_count: int
    face_bbox: tuple[int, int, int, int]
    normalization_applied: bool
    white_reference_rgb: Optional[tuple[float, float, float]]
    method: str = "white_reference_bilateral_cheek_median_cielab_ita_v2"

    def to_dict(self) -> dict:
        return asdict(self)


def srgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in ``[0, 255]`` or ``[0, 1]`` to CIE L*a*b* (D65)."""

    values = np.asarray(rgb, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("rgb must have a final dimension of length 3")
    if values.size and np.nanmax(values) > 1.0:
        values = values / 255.0
    values = np.clip(values, 0.0, 1.0)

    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])

    delta = 6.0 / 29.0
    f_xyz = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    lab = np.empty_like(f_xyz)
    lab[..., 0] = 116.0 * f_xyz[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f_xyz[..., 0] - f_xyz[..., 1])
    lab[..., 2] = 200.0 * (f_xyz[..., 1] - f_xyz[..., 2])
    return lab


def individual_typology_angle(lab_l: float, lab_b: float) -> float:
    """Return ITA in degrees from CIE L* and b* values."""

    return float(np.degrees(np.arctan2(float(lab_l) - 50.0, float(lab_b))))


class SkinToneMetrics:
    """Measure facial colour from bilateral cheek patches.

    A caller may supply a face bounding box for deterministic validation.  In
    production, the largest OpenCV Haar-detected frontal face is used.  No
    centre-image fallback is permitted: detection failure is a study outcome.
    """

    def __init__(self, min_pixels: int = 256):
        self.min_pixels = int(min_pixels)
        if self.min_pixels < 1:
            raise ValueError("min_pixels must be positive")

    @staticmethod
    def _as_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"))
        else:
            array = np.asarray(image)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            if array.ndim != 3 or array.shape[-1] not in (3, 4):
                raise ValueError("image must be an RGB/RGBA array or PIL image")
            array = array[..., :3]
        return np.asarray(array, dtype=np.uint8)

    @staticmethod
    def _validate_bbox(
        bbox: Sequence[int], width: int, height: int
    ) -> tuple[int, int, int, int]:
        if len(bbox) != 4:
            raise ValueError("face_bbox must contain x, y, width, height")
        x, y, w, h = (int(value) for value in bbox)
        if w <= 0 or h <= 0:
            raise ValueError("face_bbox width and height must be positive")
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("face_bbox does not overlap the image")
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def _detect_largest_face(rgb: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        try:
            import cv2
        except ImportError:
            return None

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _cheek_mask(
        shape: tuple[int, int], bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Return two conservative ellipses below the eyes and above the mouth."""

        height, width = shape
        x, y, w, h = bbox
        yy, xx = np.ogrid[:height, :width]
        mask = np.zeros((height, width), dtype=bool)
        for cx_fraction in (0.31, 0.69):
            cx = x + cx_fraction * w
            cy = y + 0.60 * h
            rx = max(1.0, 0.115 * w)
            ry = max(1.0, 0.090 * h)
            mask |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        return mask

    @staticmethod
    def _white_reference_normalize(
        rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, Optional[tuple[float, float, float]]]:
        """Normalize exposure/white balance from bright neutral background pixels.

        The study prompt prespecifies a white studio background.  We use only
        bright, nearly neutral pixels outside the face box as an in-image white
        reference.  When that reference is unavailable, the image is returned
        unchanged and the missing normalization is recorded.
        """

        height, width = rgb.shape[:2]
        x, y, w, h = bbox
        background = np.ones((height, width), dtype=bool)
        pad_x, pad_y = int(0.15 * w), int(0.15 * h)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        background[y0:y1, x0:x1] = False
        pixels = rgb[background].astype(np.float64)
        if pixels.size == 0:
            return rgb, None
        neutral = (
            (pixels.min(axis=1) >= 160.0)
            & ((pixels.max(axis=1) - pixels.min(axis=1)) <= 35.0)
        )
        candidates = pixels[neutral]
        if candidates.shape[0] < 256:
            return rgb, None
        reference = np.median(candidates, axis=0)
        gains = np.clip(235.0 / np.maximum(reference, 1.0), 0.5, 2.0)
        normalized = np.clip(rgb.astype(np.float64) * gains, 0.0, 255.0).astype(np.uint8)
        return normalized, tuple(float(value) for value in reference)

    def measure(
        self,
        image: Image.Image | np.ndarray,
        *,
        face_bbox: Optional[Sequence[int]] = None,
    ) -> Optional[SkinToneMeasurement]:
        """Measure one image, returning ``None`` when no valid region exists."""

        rgb = self._as_rgb(image)
        height, width = rgb.shape[:2]
        detected = (
            self._detect_largest_face(rgb) if face_bbox is None else tuple(face_bbox)
        )
        if detected is None:
            return None
        bbox = self._validate_bbox(detected, width, height)
        rgb, white_reference = self._white_reference_normalize(rgb, bbox)
        mask = self._cheek_mask((height, width), bbox)
        pixels = rgb[mask]
        if pixels.size == 0:
            return None

        lab = srgb_to_cielab(pixels)
        # Remove only extreme clipping/highlights. Broad chromaticity filtering
        # can itself introduce demographic bias, so robust medians do the rest.
        valid = np.isfinite(lab).all(axis=1) & (lab[:, 0] > 2.0) & (lab[:, 0] < 98.0)
        lab = lab[valid]
        if lab.shape[0] < self.min_pixels:
            return None

        lab_l, lab_a, lab_b = np.median(lab, axis=0)
        return SkinToneMeasurement(
            ita_degrees=individual_typology_angle(lab_l, lab_b),
            lab_l=float(lab_l),
            lab_a=float(lab_a),
            lab_b=float(lab_b),
            pixel_count=int(lab.shape[0]),
            face_bbox=bbox,
            normalization_applied=white_reference is not None,
            white_reference_rgb=white_reference,
        )

    def compare(
        self,
        original: Image.Image | np.ndarray,
        edited: Image.Image | np.ndarray,
        *,
        original_bbox: Optional[Sequence[int]] = None,
        edited_bbox: Optional[Sequence[int]] = None,
    ) -> dict:
        """Compare two images using signed colour changes.

        ``skin_tone_change`` is positive when ITA decreases (a darker measured
        appearance) and negative when ITA increases.  It is a colourimetric
        proxy, not a demographic label.
        """

        before = self.measure(original, face_bbox=original_bbox)
        after = self.measure(edited, face_bbox=edited_bbox)
        complete = before is not None and after is not None
        return {
            "skin_tone_metric": "white_reference_bilateral_cheek_median_cielab_ita_v2",
            "skin_tone_complete": complete,
            "skin_tone_failure": (
                None
                if complete
                else "original_face_or_skin_region_missing"
                if before is None
                else "edited_face_or_skin_region_missing"
            ),
            "original_ita": before.ita_degrees if before else None,
            "edited_ita": after.ita_degrees if after else None,
            "skin_tone_change": (
                before.ita_degrees - after.ita_degrees if complete else None
            ),
            "lightness_change": after.lab_l - before.lab_l if complete else None,
            "original_skin_pixels": before.pixel_count if before else 0,
            "edited_skin_pixels": after.pixel_count if after else 0,
            "original_white_normalized": (
                before.normalization_applied if before else False
            ),
            "edited_white_normalized": after.normalization_applied if after else False,
        }

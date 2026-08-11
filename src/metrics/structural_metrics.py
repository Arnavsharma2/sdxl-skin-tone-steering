"""
Structural preservation metrics.

This module implements metrics for measuring preservation of:
- Background (SSIM on non-face regions)
- Pose (3D head pose angles)
- Overall structure
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from .face_landmarks import FaceLandmarkBackend


class StructuralPreservationMetrics:
    """
    Measures preservation of pose, background, and overall structure.

    Metrics:
    - Background SSIM (on masked background)
    - 3D head pose angles (yaw, pitch, roll)
    - Overall structural similarity

    Example:
        >>> metrics = StructuralPreservationMetrics()
        >>> pose_diff = metrics.pose_difference(img1, img2)
        >>> print(f"Yaw difference: {pose_diff['yaw_diff']:.2f}°")
    """

    face_mask_expand_ratio = 1.5
    background_erosion_kernel = 7
    minimum_background_pixels = 49

    def __init__(
        self,
        device: str = "cuda",
        landmark_backend: Optional[FaceLandmarkBackend] = None,
    ):
        """
        Initialize structural metrics.

        Args:
            device: Device to run on
        """
        self.device = device
        self.landmark_backend = landmark_backend or FaceLandmarkBackend()

    def detect_face_bbox(
        self,
        img: Union[Image.Image, np.ndarray],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face bounding box.

        Args:
            img: Input image

        Returns:
            (x, y, width, height) or None if no face detected
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        result = self.landmark_backend.detect(img)
        if result is None:
            return None
        h, w = img.shape[:2]
        xs = np.asarray([landmark.x for landmark in result.landmarks]) * w
        ys = np.asarray([landmark.y for landmark in result.landmarks]) * h
        x = int(np.floor(xs.min()))
        y = int(np.floor(ys.min()))
        width = int(np.ceil(xs.max()) - x)
        height = int(np.ceil(ys.max()) - y)

        return (x, y, width, height)

    def create_face_mask(
        self,
        img: Union[Image.Image, np.ndarray],
        expand_ratio: float = face_mask_expand_ratio,
    ) -> Optional[np.ndarray]:
        """
        Create binary mask of face region.

        Args:
            img: Input image
            expand_ratio: Expand face bbox by this ratio

        Returns:
            Binary mask (1 = face, 0 = background), or ``None`` on detection failure.
        """
        # Convert to numpy
        if isinstance(img, Image.Image):
            img = np.array(img)

        bbox = self.detect_face_bbox(img)

        if bbox is None:
            return None

        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Expand bbox
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        new_w = int(w * expand_ratio)
        new_h = int(h * expand_ratio)
        x = max(0, center_x - new_w // 2)
        y = max(0, center_y - new_h // 2)
        x2 = min(img.shape[1], x + new_w)
        y2 = min(img.shape[0], y + new_h)

        # Fill mask
        mask[y:y2, x:x2] = 1

        return mask

    def background_ssim(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
        mask: Optional[np.ndarray] = None,
        counterfactual_mask: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """
        Compute SSIM on background region (non-face).

        Args:
            img1: First image
            img2: Second image
            mask: Original face mask (1 = face, 0 = background).
            counterfactual_mask: Counterfactual face mask. Auto-detected when
                ``mask`` is not supplied; both masks are unioned.

        Returns:
            SSIM value in [0, 1]. >0.90 is good preservation.
        """
        def to_gray(img):
            if isinstance(img, Image.Image):
                img = np.array(img.convert("L"))
            else:
                img = np.asarray(img)
                if img.dtype != np.uint8:
                    raise ValueError("SSIM inputs must be uint8 images")
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.ndim != 2:
                    raise ValueError("SSIM inputs must be grayscale or RGB images")
            return np.asarray(img, dtype=np.uint8)

        def valid_mask(candidate, shape):
            if candidate is None:
                return None
            candidate = np.asarray(candidate)
            if candidate.shape != shape:
                return None
            if not np.all((candidate == 0) | (candidate == 1)):
                return None
            return candidate.astype(np.uint8)

        img1_gray = to_gray(img1)
        img2_gray = to_gray(img2)

        if img1_gray.shape != img2_gray.shape:
            return None

        # Create mask if not provided
        if mask is None:
            mask = self.create_face_mask(img1)
            counterfactual_mask = self.create_face_mask(img2)
            mask = valid_mask(mask, img1_gray.shape)
            counterfactual_mask = valid_mask(counterfactual_mask, img1_gray.shape)
            if (
                mask is None
                or counterfactual_mask is None
                or not np.any(mask)
                or not np.any(counterfactual_mask)
            ):
                # Without a detected face there is no defensible separation of
                # foreground and background. Returning whole-image SSIM here
                # would silently mislabel the metric.
                return None
            mask = np.maximum(mask, counterfactual_mask)
        else:
            mask = valid_mask(mask, img1_gray.shape)
            if mask is None:
                return None
            if counterfactual_mask is not None:
                counterfactual_mask = valid_mask(counterfactual_mask, img1_gray.shape)
                if counterfactual_mask is None:
                    return None
                mask = np.maximum(mask, counterfactual_mask)

        if not np.any(mask):
            return None

        # Invert mask (we want background)
        bg_mask = 1 - mask

        # Compute SSIM on background only
        if bg_mask.sum() == 0:
            return None

        # Compute a full SSIM map on the unmodified images, then average only
        # genuinely background pixels. Multiplying the face by zero before a
        # global SSIM call inflates the score with identical artificial pixels.
        _, score_map = ssim(img1_gray, img2_gray, data_range=255, full=True)
        valid_background = cv2.erode(
            bg_mask.astype(np.uint8),
            np.ones(
                (self.background_erosion_kernel, self.background_erosion_kernel),
                dtype=np.uint8,
            ),
        ).astype(bool)
        if valid_background.sum() < self.minimum_background_pixels:
            return None
        return float(np.mean(score_map[valid_background]))

    def overall_ssim(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> float:
        """
        Compute SSIM on entire image.

        Args:
            img1: First image
            img2: Second image

        Returns:
            SSIM value in [0, 1]
        """
        # Convert to numpy grayscale
        def to_gray(img):
            if isinstance(img, Image.Image):
                img = np.array(img.convert("L"))
            elif len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return img

        img1_gray = to_gray(img1)
        img2_gray = to_gray(img2)

        if img1_gray.shape != img2_gray.shape:
            raise ValueError("SSIM inputs must have identical dimensions")

        # Compute SSIM
        score = ssim(img1_gray, img2_gray, data_range=255)

        return float(score)

    def estimate_3d_pose(
        self,
        img: Union[Image.Image, np.ndarray],
    ) -> Optional[Dict[str, float]]:
        """
        Estimate 3D head pose (yaw, pitch, roll).

        Args:
            img: Input image

        Returns:
            Dict with 'yaw', 'pitch', 'roll' in degrees, or None if failed
        """
        result = self.landmark_backend.detect(img)
        if result is None or result.transformation_matrix is None:
            return None
        transform = np.asarray(result.transformation_matrix, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            return None
        rotation_mat = transform[:3, :3]
        if not np.allclose(rotation_mat.T @ rotation_mat, np.eye(3), atol=1e-3):
            return None
        if not np.isclose(np.linalg.det(rotation_mat), 1.0, atol=1e-3):
            return None

        # Calculate Euler angles
        sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

        if sy > 1e-6:
            pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
        else:
            pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
            yaw = np.arctan2(-rotation_mat[2, 0], sy)
            roll = 0

        # Convert to degrees
        return {
            "yaw": np.degrees(yaw),
            "pitch": np.degrees(pitch),
            "roll": np.degrees(roll),
        }

    def pose_difference(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> Dict[str, Optional[float]]:
        """
        Compute difference in 3D head pose.

        Args:
            img1: First image
            img2: Second image

        Returns:
            Dict with yaw_diff, pitch_diff, roll_diff, total_diff in degrees
        """
        pose1 = self.estimate_3d_pose(img1)
        pose2 = self.estimate_3d_pose(img2)

        if pose1 is None or pose2 is None:
            return {
                "yaw_diff": None,
                "pitch_diff": None,
                "roll_diff": None,
                "total_diff": None,
            }

        def angular_difference(first: float, second: float) -> float:
            return abs((first - second + 180.0) % 360.0 - 180.0)

        yaw_diff = angular_difference(pose1["yaw"], pose2["yaw"])
        pitch_diff = angular_difference(pose1["pitch"], pose2["pitch"])
        roll_diff = angular_difference(pose1["roll"], pose2["roll"])
        total_diff = np.sqrt(yaw_diff**2 + pitch_diff**2 + roll_diff**2)

        return {
            "yaw_diff": float(yaw_diff),
            "pitch_diff": float(pitch_diff),
            "roll_diff": float(roll_diff),
            "total_diff": float(total_diff),
        }

    def compute_all_metrics(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute all structural metrics.

        Args:
            img1: Original image
            img2: Modified image

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Background SSIM
        try:
            metrics["background_ssim"] = self.background_ssim(img1, img2)
        except Exception as e:
            print(f"WARNING: Could not compute background SSIM: {e}")
            metrics["background_ssim"] = None

        # Overall SSIM
        try:
            metrics["overall_ssim"] = self.overall_ssim(img1, img2)
        except Exception as e:
            print(f"WARNING: Could not compute overall SSIM: {e}")
            metrics["overall_ssim"] = None

        # Pose difference
        try:
            pose_diff = self.pose_difference(img1, img2)
            metrics.update(pose_diff)
        except Exception as e:
            print(f"WARNING: Could not compute pose difference: {e}")
            metrics.update(
                {
                    "yaw_diff": None,
                    "pitch_diff": None,
                    "roll_diff": None,
                    "total_diff": None,
                }
            )

        return metrics

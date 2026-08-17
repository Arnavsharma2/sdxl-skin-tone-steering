"""
Structural preservation metrics.

This module implements metrics for measuring preservation of:
- Background (SSIM on non-face regions)
- Pose (MTCNN five-point landmark geometry)
- Overall structure
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


class StructuralPreservationMetrics:
    """
    Measures preservation of pose, background, and overall structure.

    Metrics:
    - Background SSIM (on masked background)
    - Landmark-based pose proxies (yaw, pitch, roll)
    - Overall structural similarity

    Example:
        >>> metrics = StructuralPreservationMetrics()
        >>> pose_diff = metrics.pose_difference(img1, img2)
        >>> print(f"Yaw difference: {pose_diff['yaw_diff']:.2f}°")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize structural metrics.

        Args:
            device: Device to run on
        """
        self.device = device
        self.face_detector = None

    def _load_face_detector(self):
        """Load face detection model."""
        if self.face_detector is not None:
            return

        try:
            from facenet_pytorch import MTCNN

            detector_device = "cpu" if self.device == "mps" else self.device
            self.face_detector = MTCNN(
                keep_all=True,
                device=detector_device,
                post_process=False,
            )
            print("Loaded MTCNN face detector")
        except Exception as e:
            print(f"WARNING: Could not load MTCNN: {e}")
            print("  Install with: pip install facenet-pytorch")
            self.face_detector = None

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
        self._load_face_detector()

        if self.face_detector is None:
            return None

        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
        else:
            img = img.convert("RGB")
        boxes, probabilities = self.face_detector.detect(img, landmarks=False)
        if boxes is None or probabilities is None:
            return None
        candidates = [
            (box, float(probability))
            for box, probability in zip(boxes, probabilities)
            if probability is not None and float(probability) >= 0.90
        ]
        if not candidates:
            return None
        box, _ = max(
            candidates,
            key=lambda item: max(0.0, item[0][2] - item[0][0])
            * max(0.0, item[0][3] - item[0][1]),
        )
        x0, y0, x1, y1 = (int(round(value)) for value in box)
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    def create_face_mask(
        self,
        img: Union[Image.Image, np.ndarray],
        expand_ratio: float = 1.5,
    ) -> np.ndarray:
        """
        Create binary mask of face region.

        Args:
            img: Input image
            expand_ratio: Expand face bbox by this ratio

        Returns:
            Binary mask (1 = face, 0 = background)
        """
        # Convert to numpy
        if isinstance(img, Image.Image):
            img = np.array(img)

        bbox = self.detect_face_bbox(img)

        if bbox is None:
            # No face detected, return all zeros
            return np.zeros(img.shape[:2], dtype=np.uint8)

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
    ) -> Optional[float]:
        """
        Compute SSIM on background region (non-face).

        Args:
            img1: First image
            img2: Second image
            mask: Face mask (1 = face, 0 = background). Auto-detected if None.

        Returns:
            SSIM value in [0, 1]. >0.90 is good preservation.
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

        # Ensure same size
        if img1_gray.shape != img2_gray.shape:
            img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

        # Create mask if not provided
        if mask is None:
            mask = self.create_face_mask(img1)
            if not np.any(mask):
                # Without a detected face there is no defensible separation of
                # foreground and background. Returning whole-image SSIM here
                # would silently mislabel the metric.
                return None

        # Invert mask (we want background)
        bg_mask = 1 - mask

        # Compute SSIM on background only
        if bg_mask.sum() == 0:
            return 0.0  # No background

        # Apply mask
        img1_bg = img1_gray * bg_mask
        img2_bg = img2_gray * bg_mask

        # Compute SSIM
        score = ssim(img1_bg, img2_bg, data_range=255)

        return float(score)

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

        # Ensure same size
        if img1_gray.shape != img2_gray.shape:
            img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

        # Compute SSIM
        score = ssim(img1_gray, img2_gray, data_range=255)

        return float(score)

    def estimate_3d_pose(
        self,
        img: Union[Image.Image, np.ndarray],
    ) -> Optional[Dict[str, float]]:
        """
        Estimate documented 2D landmark pose proxies (yaw, pitch, roll).

        Args:
            img: Input image

        Returns:
            Dict with proxy 'yaw', 'pitch', 'roll' in degrees, or None if failed.
            These are preservation outcomes, not calibrated camera pose angles.
        """
        self._load_face_detector()

        if self.face_detector is None:
            return None
        if isinstance(img, np.ndarray):
            img = Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB")
        else:
            img = img.convert("RGB")
        boxes, probabilities, landmarks = self.face_detector.detect(img, landmarks=True)
        if boxes is None or probabilities is None or landmarks is None:
            return None
        candidates = [
            (index, box, float(probability))
            for index, (box, probability) in enumerate(zip(boxes, probabilities))
            if probability is not None and float(probability) >= 0.90
        ]
        if not candidates:
            return None
        index, _, _ = max(
            candidates,
            key=lambda item: max(0.0, item[1][2] - item[1][0])
            * max(0.0, item[1][3] - item[1][1]),
        )
        return self._pose_from_landmarks(np.asarray(landmarks[index], dtype=float))

    @staticmethod
    def _pose_from_landmarks(points: np.ndarray) -> Optional[Dict[str, float]]:
        """Convert MTCNN's five landmarks to reproducible pose proxies."""

        if points.shape != (5, 2) or not np.isfinite(points).all():
            return None
        left_eye, right_eye, nose, left_mouth, right_mouth = points
        eye_vector = right_eye - left_eye
        eye_distance = float(np.linalg.norm(eye_vector))
        if eye_distance < 1e-6:
            return None
        eye_mid = (left_eye + right_eye) / 2.0
        mouth_mid = (left_mouth + right_mouth) / 2.0
        eye_mouth_distance = float(mouth_mid[1] - eye_mid[1])
        if abs(eye_mouth_distance) < 1e-6:
            return None
        roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
        yaw = np.degrees(np.arctan2(nose[0] - eye_mid[0], eye_distance))
        vertical_ratio = (nose[1] - eye_mid[1]) / eye_mouth_distance
        pitch = np.degrees(np.arctan(vertical_ratio - 0.5))
        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
        }

    def pose_difference(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute difference in MTCNN landmark pose proxies.

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
                "yaw_diff": float("inf"),
                "pitch_diff": float("inf"),
                "roll_diff": float("inf"),
                "total_diff": float("inf"),
            }

        yaw_diff = abs(pose1["yaw"] - pose2["yaw"])
        pitch_diff = abs(pose1["pitch"] - pose2["pitch"])
        roll_diff = abs(pose1["roll"] - pose2["roll"])
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
            metrics["pose_diff"] = None

        return metrics

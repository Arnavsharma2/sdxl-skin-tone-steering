"""
Face detection and alignment utilities.

This module provides helper functions for:
- Face detection
- Face alignment
- Face mask creation
"""

import warnings
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


def detect_faces(
    image: np.ndarray,
    min_confidence: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect all faces in image.

    Args:
        image: Input image (numpy array)
        min_confidence: Minimum detection confidence

    Returns:
        List of bounding boxes [(x, y, width, height), ...]
    """
    try:
        import mediapipe as mp

        mp_face_detection = mp.solutions.face_detection

        with mp_face_detection.FaceDetection(
            min_detection_confidence=min_confidence
        ) as face_detection:
            # Convert to RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces
            results = face_detection.process(image_rgb)

            if not results.detections:
                return []

            # Extract bounding boxes
            h, w = image.shape[:2]
            bboxes = []

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                bboxes.append((x, y, width, height))

            return bboxes

    except ImportError:
        print("WARNING: MediaPipe not installed. Install with: pip install mediapipe")
        return []


def align_face(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    output_size: Tuple[int, int] = (512, 512),
) -> Optional[np.ndarray]:
    """
    Align face to canonical pose.

    Args:
        image: Input image
        bbox: Face bounding box (x, y, w, h). Auto-detected if None.
        output_size: Size of output aligned face

    Returns:
        Aligned face image or None if no face detected
    """
    # Auto-detect if bbox not provided
    if bbox is None:
        bboxes = detect_faces(image)
        if len(bboxes) == 0:
            return None
        bbox = bboxes[0]

    x, y, w, h = bbox

    # Expand bbox slightly
    expand_ratio = 1.3
    center_x = x + w // 2
    center_y = y + h // 2
    new_w = int(w * expand_ratio)
    new_h = int(h * expand_ratio)

    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(image.shape[1], x1 + new_w)
    y2 = min(image.shape[0], y1 + new_h)

    # Crop face
    face = image[y1:y2, x1:x2]

    # Resize to output size
    face_resized = cv2.resize(face, output_size, interpolation=cv2.INTER_LANCZOS4)

    return face_resized


def extract_face_mask(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    expand_ratio: float = 1.5,
) -> np.ndarray:
    """
    Extract binary mask of face region.

    Args:
        image: Input image
        bbox: Face bounding box. Auto-detected if None.
        expand_ratio: Expand bbox by this ratio

    Returns:
        Binary mask (1 = face, 0 = background)
    """
    # Auto-detect if bbox not provided
    if bbox is None:
        bboxes = detect_faces(image)
        if len(bboxes) == 0:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        bbox = bboxes[0]

    x, y, w, h = bbox

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Expand bbox
    center_x = x + w // 2
    center_y = y + h // 2
    new_w = int(w * expand_ratio)
    new_h = int(h * expand_ratio)

    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(image.shape[1], x1 + new_w)
    y2 = min(image.shape[0], y1 + new_h)

    # Fill ellipse mask (more natural than rectangle)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)

    return mask


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(image)


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    return Image.fromarray(image)


def resize_with_padding(
    image: np.ndarray,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio.

    Adds padding if necessary.

    Args:
        image: Input image
        target_size: Target (width, height)
        fill_color: Padding color

    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Calculate scale
    scale = min(target_w / w, target_h / h)

    # Resize
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create canvas with padding
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)

    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def get_face_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Get 68-point facial landmarks.

    Args:
        image: Input image

    Returns:
        Landmarks as (68, 2) array or None if no face detected
    """
    try:
        import os

        import dlib

        # Try multiple common locations for the landmark predictor
        # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        predictor_paths = [
            "models/shape_predictor_68_face_landmarks.dat",
            os.path.expanduser("~/.cache/dlib/shape_predictor_68_face_landmarks.dat"),
            "/usr/share/dlib/shape_predictor_68_face_landmarks.dat",
        ]

        predictor_path = None
        for path in predictor_paths:
            if os.path.exists(path):
                predictor_path = path
                break

        if predictor_path is None:
            print("WARNING: Landmark predictor not found. Download from:")
            print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("  and place in: models/shape_predictor_68_face_landmarks.dat")
            return None

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect faces
        dets = detector(gray, 1)

        if len(dets) == 0:
            return None

        # Get landmarks for first face
        shape = predictor(gray, dets[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        return landmarks

    except ImportError:
        print("WARNING: dlib not installed. Install with: pip install dlib")
        return None
    except Exception as e:
        print(f"WARNING: Could not load landmark predictor: {e}")
        return None

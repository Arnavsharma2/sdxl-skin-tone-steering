"""
Identity preservation metrics.

This module implements metrics to measure whether counterfactual images
preserve the identity of the original person.
"""

import os
import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from .artifacts import (
    ArtifactVerification,
    inspect_artifact,
    require_verified,
)

# Suppress warnings from face detection libraries
warnings.filterwarnings("ignore")

FACENET_VGGFACE2_SHA256 = (
    "281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b"
)
ALEXNET_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
LPIPS_ALEX_V01_SHA256 = (
    "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
)
MTCNN_SHA256 = {
    "pnet.pt": "a2a71925e0b9996a42f63e47efc1ca19043e69558b5c523b978d611dfae49c8f",
    "rnet.pt": "bbb937de72efc9ef83b186c49f5f558467a1d7e3453a8ece0d71a886633f6a86",
    "onet.pt": "165bfbe42940416ccfb977545cf0e976d5bf321f67083ae2aaaa5c764280118d",
}


class IdentityPreservationMetrics:
    """
    Measures how well identity is preserved across transformations.

    Metrics:
    - Face similarity (ArcFace/FaceNet embeddings)
    - Facial landmark preservation (RMSE)
    - Perceptual similarity (LPIPS)

    Example:
        >>> metrics = IdentityPreservationMetrics()
        >>> similarity = metrics.face_similarity(img1, img2)
        >>> if similarity > 0.85:
        ...     print("Same person!")
    """

    def __init__(
        self,
        device: str = "cuda",
        use_arcface: bool = True,
        use_facenet: bool = False,
        use_landmarks: bool = False,
    ):
        """
        Initialize identity metrics.

        Args:
            device: Device to run on
            use_arcface: Use ArcFace for face recognition
            use_facenet: Use FaceNet for face recognition
        """
        self.device = device
        self.face_model = None
        self.landmark_detector = None
        self.lpips_model = None
        self.model_type = None
        self.use_landmarks = use_landmarks
        self.artifact_verifications: dict[str, ArtifactVerification] = {}

        # Load face recognition model
        if use_arcface:
            self._load_arcface()
        elif use_facenet:
            self._load_facenet()

    def _load_arcface(self):
        """Load ArcFace model for face recognition."""
        try:
            from insightface.app import FaceAnalysis

            self.face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.face_app.prepare(ctx_id=0 if self.device == "cuda" else -1)
            self.model_type = "arcface"
            print("Loaded ArcFace model")
        except Exception as e:
            print(f"WARNING: Could not load ArcFace: {e}")
            print("  Install with: pip install insightface onnxruntime-gpu")
            self.face_app = None

    def _load_facenet(self):
        """Load FaceNet model for face recognition."""
        try:
            import facenet_pytorch
            from facenet_pytorch import MTCNN, InceptionResnetV1

            # FaceNet uses adaptive pooling which is unsupported on MPS — use CPU
            facenet_device = "cpu" if self.device == "mps" else self.device

            torch_home = self._torch_home()
            weights = torch_home / "checkpoints" / "20180402-114759-vggface2.pt"
            initial = inspect_artifact(
                weights,
                FACENET_VGGFACE2_SHA256,
                name="FaceNet VGGFace2 weights",
            )
            self.artifact_verifications["facenet_vggface2"] = initial
            if initial.status != "missing":
                require_verified(initial)
            self.face_model = InceptionResnetV1(pretrained="vggface2").eval()
            self._record_artifact(
                "facenet_vggface2",
                weights,
                FACENET_VGGFACE2_SHA256,
                name="FaceNet VGGFace2 weights",
            )
            package_root = Path(facenet_pytorch.__file__).resolve().parent
            for filename, expected in MTCNN_SHA256.items():
                key = f"mtcnn_{filename.removesuffix('.pt')}"
                self._record_artifact(
                    key,
                    package_root / "data" / filename,
                    expected,
                    name=f"MTCNN {filename} weights",
                )
            self.face_model.to(facenet_device)
            self.facenet_device = facenet_device
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                device=facenet_device,
                # InceptionResnetV1's pretrained weights require MTCNN's
                # fixed-image standardisation. Raw [0, 255] crops produce
                # invalid embedding similarities.
                post_process=True,
            )
            self.model_type = "facenet"
            print("Loaded FaceNet model")
        except Exception as e:
            print(f"WARNING: Could not load FaceNet: {e}")
            print("  Install with: pip install facenet-pytorch")
            self.face_model = None
            self.mtcnn = None

    @staticmethod
    def _torch_home() -> Path:
        configured = os.environ.get("TORCH_HOME")
        if configured:
            return Path(configured).expanduser()
        cache_home = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        return cache_home / "torch"

    def _record_artifact(
        self,
        key: str,
        path: Path,
        expected_sha256: str,
        *,
        name: str,
    ) -> ArtifactVerification:
        result = inspect_artifact(path, expected_sha256, name=name)
        self.artifact_verifications[key] = result
        return require_verified(result)

    def _load_lpips(self):
        """Load LPIPS model for perceptual similarity."""
        if self.lpips_model is not None:
            return

        try:
            import lpips

            torch_home = self._torch_home()
            backbone = torch_home / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
            package_root = Path(lpips.__file__).resolve().parent
            self._record_artifact(
                "lpips_alex_v0.1",
                package_root / "weights" / "v0.1" / "alex.pth",
                LPIPS_ALEX_V01_SHA256,
                name="LPIPS AlexNet v0.1 linear weights",
            )
            initial = inspect_artifact(
                backbone,
                ALEXNET_SHA256,
                name="LPIPS AlexNet backbone weights",
            )
            self.artifact_verifications["alexnet_backbone"] = initial
            if initial.status != "missing":
                require_verified(initial)
            self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
            self._record_artifact(
                "alexnet_backbone",
                backbone,
                ALEXNET_SHA256,
                name="LPIPS AlexNet backbone weights",
            )
            print("Loaded LPIPS model")
        except Exception as e:
            print(f"WARNING: Could not load LPIPS: {e}")
            print("  Install with: pip install lpips")
            self.lpips_model = None

    @staticmethod
    def _as_pil_rgb(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Convert supported face inputs to an explicit uint8 RGB image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("FaceNet inputs must be uint8 RGB images with shape (H, W, 3)")
        return Image.fromarray(array, mode="RGB")

    @staticmethod
    def _prepare_lpips_tensor(
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Return one finite RGB batch normalized exactly to [-1, 1]."""
        if isinstance(image, torch.Tensor):
            tensor = image.detach().float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
                raise ValueError(
                    "LPIPS tensor inputs must have shape (3, H, W) or (1, 3, H, W)"
                )
            if not torch.isfinite(tensor).all() or tensor.min() < -1 or tensor.max() > 1:
                raise ValueError(
                    "LPIPS tensor inputs must be finite and normalized to [-1, 1]"
                )
            return tensor
        rgb = IdentityPreservationMetrics._as_pil_rgb(image)
        array = np.asarray(rgb).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        return tensor.mul(2.0).sub(1.0).unsqueeze(0)

    def _load_landmarks(self):
        """Load facial landmark detector."""
        if self.landmark_detector is not None:
            return

        try:
            import dlib

            # Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"

            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_detector = dlib.shape_predictor(predictor_path)
            print("Loaded dlib landmark detector")
        except Exception as e:
            print(f"WARNING: Could not load dlib: {e}")
            print("  Download predictor: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.landmark_detector = None

    def face_similarity(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> float:
        """
        Compute face similarity using face recognition embeddings.

        Args:
            img1: First image
            img2: Second image

        Returns:
            Cosine similarity in [-1, 1]. The frozen engineering gate is 0.85.
        """
        if self.model_type == "arcface":
            return self._face_similarity_arcface(img1, img2)
        elif self.model_type == "facenet":
            return self._face_similarity_facenet(img1, img2)
        else:
            raise RuntimeError("No face recognition model loaded")

    def _face_similarity_arcface(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> float:
        """Compute similarity using ArcFace."""
        if self.face_app is None:
            raise RuntimeError("ArcFace not loaded")

        # Convert to numpy
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)

        # Detect faces and get embeddings
        faces1 = self.face_app.get(img1)
        faces2 = self.face_app.get(img2)

        if len(faces1) == 0 or len(faces2) == 0:
            raise RuntimeError("ArcFace could not detect a face in both images")

        # Get embeddings (use first face)
        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )

        return float(similarity)

    def _face_similarity_facenet(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> float:
        """Compute similarity using FaceNet."""
        if self.face_model is None:
            raise RuntimeError("FaceNet not loaded")

        img1 = self._as_pil_rgb(img1)
        img2 = self._as_pil_rgb(img2)

        # Detect and crop faces
        face1 = self.mtcnn(img1)
        face2 = self.mtcnn(img2)

        if face1 is None or face2 is None:
            raise RuntimeError("MTCNN could not detect a face in both images")

        # Get embeddings (use facenet_device — may differ from self.device on MPS)
        fd = getattr(self, "facenet_device", self.device)
        with torch.no_grad():
            emb1 = self.face_model(face1.unsqueeze(0).to(fd))
            emb2 = self.face_model(face2.unsqueeze(0).to(fd))

        if not torch.isfinite(emb1).all() or not torch.isfinite(emb2).all():
            raise RuntimeError("FaceNet produced a non-finite embedding")
        if torch.linalg.vector_norm(emb1) == 0 or torch.linalg.vector_norm(emb2) == 0:
            raise RuntimeError("FaceNet produced a zero-norm embedding")

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)

        return float(similarity.cpu().item())

    def landmark_rmse(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> float:
        """
        Compute RMSE of 68 facial landmarks.

        Args:
            img1: First image
            img2: Second image

        Returns:
            RMSE in pixels. <5 pixels is good preservation.
        """
        self._load_landmarks()

        if self.landmark_detector is None:
            raise RuntimeError("Landmark detector not loaded")

        # Convert to numpy grayscale
        if isinstance(img1, Image.Image):
            img1 = np.array(img1.convert("L"))
        if isinstance(img2, Image.Image):
            img2 = np.array(img2.convert("L"))

        # Detect faces
        dets1 = self.face_detector(img1, 1)
        dets2 = self.face_detector(img2, 1)

        if len(dets1) == 0 or len(dets2) == 0:
            return float("inf")  # No face detected

        # Get landmarks
        shape1 = self.landmark_detector(img1, dets1[0])
        shape2 = self.landmark_detector(img2, dets2[0])

        # Convert to numpy arrays
        landmarks1 = np.array([[p.x, p.y] for p in shape1.parts()])
        landmarks2 = np.array([[p.x, p.y] for p in shape2.parts()])

        # Compute RMSE
        rmse = np.sqrt(np.mean((landmarks1 - landmarks2) ** 2))

        return float(rmse)

    def perceptual_similarity(
        self,
        img1: Union[Image.Image, np.ndarray, torch.Tensor],
        img2: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> float:
        """
        Compute LPIPS perceptual similarity.

        Args:
            img1: First image
            img2: Second image

        Returns:
            LPIPS distance. Lower is more similar.
        """
        self._load_lpips()

        if self.lpips_model is None:
            raise RuntimeError("LPIPS model not loaded")

        tensor1 = self._prepare_lpips_tensor(img1).to(self.device)
        tensor2 = self._prepare_lpips_tensor(img2).to(self.device)
        if tensor1.shape != tensor2.shape:
            raise ValueError("LPIPS inputs must have identical dimensions")

        # Compute LPIPS
        with torch.no_grad():
            distance = self.lpips_model(tensor1, tensor2)

        return float(distance.cpu().item())

    def compute_all_metrics(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
    ) -> dict:
        """
        Compute all identity metrics.

        Args:
            img1: Original image
            img2: Modified image

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Face similarity
        try:
            metrics["face_similarity"] = self.face_similarity(img1, img2)
        except Exception as e:
            print(f"WARNING: Could not compute face similarity: {e}")
            metrics["face_similarity"] = None

        # The dlib predictor is optional and is not part of the fixed gate set.
        metrics["landmark_rmse"] = None
        if self.use_landmarks:
            try:
                metrics["landmark_rmse"] = self.landmark_rmse(img1, img2)
            except Exception as e:
                print(f"WARNING: Could not compute landmark RMSE: {e}")

        # Perceptual similarity
        try:
            metrics["lpips"] = self.perceptual_similarity(img1, img2)
        except Exception as e:
            print(f"WARNING: Could not compute LPIPS: {e}")
            metrics["lpips"] = None

        return metrics

    def is_same_person(
        self,
        img1: Union[Image.Image, np.ndarray],
        img2: Union[Image.Image, np.ndarray],
        similarity_threshold: float = 0.85,
    ) -> bool:
        """
        Determine if two images show the same person.

        Args:
            img1: First image
            img2: Second image
            similarity_threshold: Threshold for face similarity

        Returns:
            True if same person, False otherwise
        """
        similarity = self.face_similarity(img1, img2)
        return similarity >= similarity_threshold

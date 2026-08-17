"""
Identity preservation metrics.

This module implements metrics to measure whether counterfactual images
preserve the identity of the original person.
"""

import warnings
from typing import Union

import numpy as np
import torch
from PIL import Image

# Suppress warnings from face detection libraries
warnings.filterwarnings("ignore")


class FaceDetectionError(RuntimeError):
    """Raised when an embedding metric cannot detect both required faces."""


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
        enable_landmarks: bool = True,
    ):
        """
        Initialize identity metrics.

        Args:
            device: Device to run on
            use_arcface: Use ArcFace for face recognition
            use_facenet: Use FaceNet for face recognition
        """
        self.device = device
        self.enable_landmarks = enable_landmarks
        self.face_model = None
        self.landmark_detector = None
        self._landmark_load_attempted = False
        self.lpips_model = None
        self.model_type = None

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
            from facenet_pytorch import MTCNN, InceptionResnetV1

            # FaceNet uses adaptive pooling which is unsupported on MPS — use CPU
            facenet_device = "cpu" if self.device == "mps" else self.device

            self.face_model = InceptionResnetV1(pretrained="vggface2").eval()
            self.face_model.to(facenet_device)
            self.facenet_device = facenet_device
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                device=facenet_device,
                post_process=False,
            )
            self.model_type = "facenet"
            print("Loaded FaceNet model")
        except Exception as e:
            print(f"WARNING: Could not load FaceNet: {e}")
            print("  Install with: pip install facenet-pytorch")
            self.face_model = None

    def _load_lpips(self):
        """Load LPIPS model for perceptual similarity."""
        if self.lpips_model is not None:
            return

        try:
            import lpips

            self.lpips_model = lpips.LPIPS(net="alex").to(self.device)
            print("Loaded LPIPS model")
        except Exception as e:
            print(f"WARNING: Could not load LPIPS: {e}")
            print("  Install with: pip install lpips")
            self.lpips_model = None

    def _load_landmarks(self):
        """Load facial landmark detector."""
        if self.landmark_detector is not None or self._landmark_load_attempted:
            return
        self._landmark_load_attempted = True

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
            Cosine similarity in [0, 1]. >0.85 typically means same person.
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
            raise FaceDetectionError("ArcFace did not detect both faces")

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

        # Convert to PIL
        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1)
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2)

        # Detect and crop faces
        face1 = self.mtcnn(img1)
        face2 = self.mtcnn(img2)

        if face1 is None or face2 is None:
            raise FaceDetectionError("MTCNN did not detect both faces")

        # Get embeddings (use facenet_device — may differ from self.device on MPS)
        fd = getattr(self, "facenet_device", self.device)
        with torch.no_grad():
            emb1 = self.face_model(face1.unsqueeze(0).to(fd))
            emb2 = self.face_model(face2.unsqueeze(0).to(fd))

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

        # Convert to tensor
        def to_tensor(img):
            if isinstance(img, torch.Tensor):
                return img
            if isinstance(img, Image.Image):
                img = np.array(img)
            # Normalize to [-1, 1]
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = (img * 2.0) - 1.0
            return img.unsqueeze(0)

        tensor1 = to_tensor(img1).to(self.device)
        tensor2 = to_tensor(img2).to(self.device)

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

        # Landmark RMSE is a legacy optional metric and is disabled in the
        # frozen study, which does not ship the external dlib predictor file.
        if getattr(self, "enable_landmarks", True):
            try:
                metrics["landmark_rmse"] = self.landmark_rmse(img1, img2)
            except Exception as e:
                print(f"WARNING: Could not compute landmark RMSE: {e}")
                metrics["landmark_rmse"] = None
        else:
            metrics["landmark_rmse"] = None

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

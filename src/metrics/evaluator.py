"""Fail-closed evaluation of portrait counterfactuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .face_landmarks import FaceLandmarkBackend
from .identity_metrics import FACENET_VGGFACE2_SHA256, IdentityPreservationMetrics
from .skin_tone_metrics import SkinToneMetrics
from .structural_metrics import StructuralPreservationMetrics


@dataclass
class EvaluationThresholds:
    """Prespecified engineering gates; these are not inferential evidence."""

    face_similarity: float = 0.85
    landmark_rmse: float = 5.0
    lpips: float = 0.3
    background_ssim: float = 0.75
    pose_angle_diff: float = 5.0
    min_abs_skin_tone_change: float = 2.0


@dataclass
class EvaluationResult:
    """Metrics and validity state for one base/counterfactual pair."""

    alpha: Optional[float] = None

    # Identity and perceptual preservation
    face_similarity: Optional[float] = None
    landmark_rmse: Optional[float] = None
    lpips: Optional[float] = None

    # Structural preservation
    background_ssim: Optional[float] = None
    overall_ssim: Optional[float] = None
    yaw_diff: Optional[float] = None
    pitch_diff: Optional[float] = None
    roll_diff: Optional[float] = None
    total_pose_diff: Optional[float] = None

    # Independently measured target response. Positive relative-L* change is
    # lighter; negative is darker. The method expects alpha to have the
    # opposite sign.
    skin_tone_metric: Optional[str] = None
    skin_tone_change: Optional[float] = None
    skin_delta_ita: Optional[float] = None
    skin_delta_e: Optional[float] = None
    skin_lstar_original: Optional[float] = None
    skin_lstar_counterfactual: Optional[float] = None
    skin_relative_lstar_original: Optional[float] = None
    skin_relative_lstar_counterfactual: Optional[float] = None
    skin_ita_original: Optional[float] = None
    skin_ita_counterfactual: Optional[float] = None
    reference_lstar_shift: Optional[float] = None
    illumination_stable: bool = False
    target_direction_correct: Optional[bool] = None
    target_response_pass: bool = False

    # Assessment. ``is_disentangled`` remains only as a compatibility alias;
    # the repository cannot establish formal representation disentanglement.
    preservation_pass: bool = False
    counterfactual_success: bool = False
    is_disentangled: bool = False
    overall_score: Optional[float] = None
    pass_count: int = 0
    total_count: int = 5
    evaluation_complete: bool = False
    missing_required_metrics: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class CounterfactualEvaluator:
    """Compute target response and preservation metrics without fallbacks."""

    required_metrics = (
        "face_similarity",
        "lpips",
        "background_ssim",
        "total_pose_diff",
        "skin_tone_change",
        "target_direction_correct",
    )

    def __init__(
        self,
        device: str = "cuda",
        thresholds: Optional[EvaluationThresholds] = None,
    ) -> None:
        self.device = device
        self.thresholds = thresholds or EvaluationThresholds()
        self.identity_metrics = IdentityPreservationMetrics(
            device=device, use_arcface=False, use_facenet=True
        )
        landmark_backend = FaceLandmarkBackend()
        self.structural_metrics = StructuralPreservationMetrics(
            device=device, landmark_backend=landmark_backend
        )
        self.skin_tone_metrics = SkinToneMetrics(landmark_backend=landmark_backend)

    @staticmethod
    def metric_provenance() -> dict:
        packages = {}
        for package in (
            "facenet-pytorch",
            "lpips",
            "mediapipe",
            "opencv-contrib-python",
            "scikit-image",
            "torch",
            "torchvision",
        ):
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                packages[package] = None
        return {
            "face_embedding": "InceptionResnetV1 pretrained=vggface2; MTCNN standardised crop",
            "face_embedding_weight_sha256": FACENET_VGGFACE2_SHA256,
            "perceptual": "LPIPS AlexNet",
            "alexnet_weight_sha256": (
                "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
            ),
            "face_mask_and_pose": "MediaPipe Tasks Face Landmarker, CPU, checksum-pinned asset",
            "face_landmarker_model_sha256": (
                "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
            ),
            "target_attribute": SkinToneMetrics.metric_name,
            "background": "mean SSIM map over eroded union-background mask",
            "packages": packages,
        }

    def evaluate_pair(
        self,
        original: Union[Image.Image, np.ndarray],
        counterfactual: Union[Image.Image, np.ndarray],
        *,
        alpha: Optional[float] = None,
        verbose: bool = False,
    ) -> EvaluationResult:
        """Evaluate one pair. ``alpha`` is required for a valid target-direction gate."""
        result = EvaluationResult(alpha=alpha)

        identity = self.identity_metrics.compute_all_metrics(original, counterfactual)
        result.face_similarity = identity.get("face_similarity")
        result.landmark_rmse = identity.get("landmark_rmse")
        result.lpips = identity.get("lpips")

        structural = self.structural_metrics.compute_all_metrics(original, counterfactual)
        result.background_ssim = structural.get("background_ssim")
        result.overall_ssim = structural.get("overall_ssim")
        result.yaw_diff = structural.get("yaw_diff")
        result.pitch_diff = structural.get("pitch_diff")
        result.roll_diff = structural.get("roll_diff")
        pose = structural.get("total_diff")
        result.total_pose_diff = None if pose is None or np.isinf(pose) else pose

        target = self.skin_tone_metrics.compare(original, counterfactual)
        for name in (
            "skin_tone_metric",
            "skin_tone_change",
            "skin_delta_ita",
            "skin_delta_e",
            "skin_lstar_original",
            "skin_lstar_counterfactual",
            "skin_relative_lstar_original",
            "skin_relative_lstar_counterfactual",
            "skin_ita_original",
            "skin_ita_counterfactual",
            "reference_lstar_shift",
            "illumination_stable",
        ):
            if name in target:
                setattr(result, name, target[name])

        if alpha is not None and alpha != 0 and result.skin_tone_change is not None:
            result.target_direction_correct = alpha * result.skin_tone_change < 0
            result.target_response_pass = bool(
                result.target_direction_correct
                and abs(result.skin_tone_change)
                >= self.thresholds.min_abs_skin_tone_change
            )

        (
            result.counterfactual_success,
            result.pass_count,
            result.total_count,
            result.missing_required_metrics,
        ) = self._evaluate_counterfactual(result)
        result.evaluation_complete = not result.missing_required_metrics
        result.preservation_pass = self._preservation_pass(result)
        result.is_disentangled = result.counterfactual_success
        result.overall_score = self._compute_overall_score(result)

        if verbose:
            self._print_results(result)
        return result

    def _missing_required(self, result: EvaluationResult) -> Tuple[str, ...]:
        return tuple(
            name for name in self.required_metrics if getattr(result, name) is None
        )

    def _checks(self, result: EvaluationResult) -> list[bool]:
        checks: list[bool] = []
        if result.face_similarity is not None:
            checks.append(result.face_similarity >= self.thresholds.face_similarity)
        if result.lpips is not None:
            checks.append(result.lpips <= self.thresholds.lpips)
        if result.background_ssim is not None:
            checks.append(result.background_ssim >= self.thresholds.background_ssim)
        if result.total_pose_diff is not None:
            checks.append(result.total_pose_diff <= self.thresholds.pose_angle_diff)
        if result.target_direction_correct is not None:
            checks.append(result.target_response_pass)
        return checks

    def _preservation_pass(self, result: EvaluationResult) -> bool:
        values = (
            result.face_similarity,
            result.lpips,
            result.background_ssim,
            result.total_pose_diff,
        )
        if not all(value is not None for value in values):
            return False
        return all(
            (
                result.face_similarity >= self.thresholds.face_similarity,
                result.lpips <= self.thresholds.lpips,
                result.background_ssim >= self.thresholds.background_ssim,
                result.total_pose_diff <= self.thresholds.pose_angle_diff,
            )
        )

    def _evaluate_counterfactual(
        self, result: EvaluationResult
    ) -> Tuple[bool, int, int, Tuple[str, ...]]:
        missing = self._missing_required(result)
        checks = self._checks(result)
        # Every prespecified gate must pass. There is no 80%-pass shortcut.
        success = not missing and len(checks) == 5 and all(checks)
        return success, int(sum(checks)), 5, missing

    # Kept for callers/tests written against the old private method.
    def _evaluate_disentanglement(
        self, result: EvaluationResult
    ) -> Tuple[bool, int, int, Tuple[str, ...]]:
        return self._evaluate_counterfactual(result)

    def _compute_overall_score(self, result: EvaluationResult) -> Optional[float]:
        """Fixed-weight engineering rubric; unavailable for incomplete rows."""
        if self._missing_required(result):
            return None
        assert result.face_similarity is not None
        assert result.lpips is not None
        assert result.background_ssim is not None
        assert result.total_pose_diff is not None
        assert result.skin_tone_change is not None

        target = min(abs(result.skin_tone_change) / 10.0, 1.0)
        if not result.target_direction_correct:
            target = 0.0
        components = (
            0.30 * np.clip(result.face_similarity, 0.0, 1.0),
            0.20 * np.clip(1.0 - result.lpips, 0.0, 1.0),
            0.20 * np.clip(result.background_ssim, 0.0, 1.0),
            0.10 * np.clip(1.0 - result.total_pose_diff / 30.0, 0.0, 1.0),
            0.20 * target,
        )
        return float(sum(components))

    def _print_results(self, result: EvaluationResult) -> None:
        print("\n" + "=" * 60)
        print("AUDITED EVALUATION")
        print("=" * 60)
        print(f"  Alpha: {result.alpha}")
        print(f"  Face similarity: {result.face_similarity}")
        print(f"  LPIPS: {result.lpips}")
        print(f"  Background-only SSIM: {result.background_ssim}")
        print(f"  Pose difference: {result.total_pose_diff}")
        print(f"  Relative skin L* change: {result.skin_tone_change}")
        print(f"  Target direction correct: {result.target_direction_correct}")
        print(f"  Illumination QC passed: {result.illumination_stable}")
        print(f"  Passed: {result.pass_count}/{result.total_count} gates")
        if result.missing_required_metrics:
            print("  Missing: " + ", ".join(result.missing_required_metrics))
        print(f"  Quality rubric: {result.overall_score}")
        print(f"  Valid counterfactual: {'YES' if result.counterfactual_success else 'NO'}")
        print("=" * 60 + "\n")

    def evaluate_batch(
        self,
        pairs: List[Tuple[Image.Image, Image.Image]],
        *,
        alphas: Optional[List[float]] = None,
        subject_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        if subject_ids is None:
            subject_ids = [f"subject_{i:03d}" for i in range(len(pairs))]
        if alphas is None:
            alphas = [None] * len(pairs)
        if len(subject_ids) != len(pairs) or len(alphas) != len(pairs):
            raise ValueError("pairs, subject_ids, and alphas must have equal lengths")

        rows = []
        for index, ((original, counterfactual), alpha) in enumerate(zip(pairs, alphas)):
            if verbose:
                print(f"Evaluating pair {index + 1}/{len(pairs)}: {subject_ids[index]}")
            row = self.evaluate_pair(original, counterfactual, alpha=alpha).to_dict()
            row["subject_id"] = subject_ids[index]
            rows.append(row)
        return pd.DataFrame(rows)

    def summarize_results(self, frame: pd.DataFrame) -> Dict[str, float]:
        summary: Dict[str, float] = {
            "evaluation_completion_rate": float(frame["evaluation_complete"].mean()),
            "counterfactual_success_rate": float(frame["counterfactual_success"].mean()),
        }
        for column in (
            "face_similarity",
            "lpips",
            "background_ssim",
            "total_pose_diff",
            "skin_tone_change",
            "overall_score",
        ):
            if column in frame:
                summary[f"mean_{column}"] = float(frame[column].mean())
                summary[f"std_{column}"] = float(frame[column].std())
                summary[f"median_{column}"] = float(frame[column].median())
        return summary

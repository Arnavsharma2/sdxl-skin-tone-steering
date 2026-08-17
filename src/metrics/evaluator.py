"""
Composite evaluator for counterfactual pairs.

This module combines all metrics into a single evaluation pipeline.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .identity_metrics import IdentityPreservationMetrics
from .structural_metrics import StructuralPreservationMetrics


@dataclass
class EvaluationThresholds:
    """Thresholds for determining if manipulation is disentangled."""

    face_similarity: float = 0.85
    landmark_rmse: float = 5.0
    lpips: float = 0.3
    background_ssim: float = 0.75  # steered denoising naturally shifts background slightly
    pose_angle_diff: float = 5.0


@dataclass
class EvaluationResult:
    """Result of evaluating a counterfactual pair."""

    # Identity metrics
    face_similarity: Optional[float] = None
    landmark_rmse: Optional[float] = None
    lpips: Optional[float] = None

    # Structural metrics
    background_ssim: Optional[float] = None
    overall_ssim: Optional[float] = None
    yaw_diff: Optional[float] = None
    pitch_diff: Optional[float] = None
    roll_diff: Optional[float] = None
    total_pose_diff: Optional[float] = None

    # Overall assessment
    is_disentangled: bool = False
    overall_score: float = 0.0
    pass_count: int = 0
    total_count: int = 0
    evaluation_complete: bool = False
    missing_required_metrics: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class CounterfactualEvaluator:
    """
    Evaluates how good the counterfactuals are.

    It checks:
    - Did we keep the person's identity? (Face ID)
    - Did we mess up the background? (SSIM)
    - Did we change the pose? (Head pose estimation)
    - Overall, is it a clean edit?

    Example:
        >>> evaluator = CounterfactualEvaluator()
        >>> result = evaluator.evaluate_pair(original, counterfactual)
        >>> if result.is_disentangled:
        ...     print(f"Success! Score: {result.overall_score:.2f}")
    """

    def __init__(
        self,
        device: str = "cuda",
        thresholds: Optional[EvaluationThresholds] = None,
    ):
        """
        Initialize evaluator.

        Args:
            device: Device to run on
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.device = device
        self.thresholds = thresholds or EvaluationThresholds()

        # Use FaceNet (facenet-pytorch) — works without insightface
        self.identity_metrics = IdentityPreservationMetrics(
            device=device,
            use_arcface=False,
            use_facenet=True,
            enable_landmarks=False,
        )
        self.structural_metrics = StructuralPreservationMetrics(device=device)

    def evaluate_pair(
        self,
        original: Union[Image.Image, np.ndarray],
        counterfactual: Union[Image.Image, np.ndarray],
        verbose: bool = False,
    ) -> EvaluationResult:
        """
        Comprehensive evaluation of a counterfactual pair.

        Args:
            original: Original image
            counterfactual: Counterfactual image
            verbose: Print detailed results

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult()

        # Compute identity metrics
        identity = self.identity_metrics.compute_all_metrics(original, counterfactual)
        result.face_similarity = identity.get("face_similarity")
        result.landmark_rmse = identity.get("landmark_rmse")
        result.lpips = identity.get("lpips")

        # Compute structural metrics
        structural = self.structural_metrics.compute_all_metrics(
            original, counterfactual
        )
        result.background_ssim = structural.get("background_ssim")
        result.overall_ssim = structural.get("overall_ssim")
        result.yaw_diff = structural.get("yaw_diff")
        result.pitch_diff = structural.get("pitch_diff")
        result.roll_diff = structural.get("roll_diff")
        result.total_pose_diff = structural.get("total_diff")

        # Evaluate disentanglement
        (
            result.is_disentangled,
            result.pass_count,
            result.total_count,
            result.missing_required_metrics,
        ) = (
            self._evaluate_disentanglement(result)
        )
        result.evaluation_complete = not result.missing_required_metrics

        # Compute overall score
        result.overall_score = self._compute_overall_score(result)

        if verbose:
            self._print_results(result)

        return result

    def _evaluate_disentanglement(
        self, result: EvaluationResult
    ) -> Tuple[bool, int, int, Tuple[str, ...]]:
        """
        Checks whether required preservation metrics pass their thresholds.

        Returns:
            ``(is_disentangled, num_passed, num_total, missing_required)``.
        """
        checks = []
        required = ("face_similarity", "lpips", "background_ssim")
        missing_required = tuple(
            name for name in required if getattr(result, name) is None
        )

        # Check face similarity
        if result.face_similarity is not None:
            checks.append(result.face_similarity >= self.thresholds.face_similarity)

        # Check landmark RMSE
        if result.landmark_rmse is not None:
            checks.append(result.landmark_rmse <= self.thresholds.landmark_rmse)

        # Check LPIPS
        if result.lpips is not None:
            checks.append(result.lpips <= self.thresholds.lpips)

        # Check background SSIM
        if result.background_ssim is not None:
            checks.append(result.background_ssim >= self.thresholds.background_ssim)

        # Check pose difference (skip if mediapipe unavailable — returns inf)
        if result.total_pose_diff is not None and not np.isinf(result.total_pose_diff):
            checks.append(result.total_pose_diff <= self.thresholds.pose_angle_diff)

        if len(checks) == 0:
            return False, 0, 0, missing_required

        num_passed = sum(checks)
        num_total = len(checks)

        # Require at least 80% of checks to pass
        # A successful score requires the core metrics to be present. Missing
        # face or mask models must never turn a partial evaluation into a pass.
        is_disentangled = not missing_required and (num_passed / num_total) >= 0.8

        return is_disentangled, num_passed, num_total, missing_required

    def _compute_overall_score(self, result: EvaluationResult) -> float:
        """
        Compute overall score [0, 1].

        Weighted average of normalized metrics.
        """
        scores = []
        weights = []

        # Face similarity (higher is better)
        if result.face_similarity is not None:
            scores.append(result.face_similarity)
            weights.append(0.3)

        # Landmark RMSE (lower is better, normalize by threshold)
        if result.landmark_rmse is not None:
            normalized = max(0, 1 - result.landmark_rmse / self.thresholds.landmark_rmse)
            scores.append(normalized)
            weights.append(0.2)

        # LPIPS (lower is better, normalize by threshold)
        if result.lpips is not None:
            normalized = max(0, 1 - result.lpips / self.thresholds.lpips)
            scores.append(normalized)
            weights.append(0.2)

        # Background SSIM (higher is better)
        if result.background_ssim is not None:
            scores.append(result.background_ssim)
            weights.append(0.2)

        # Pose difference (lower is better, normalize by threshold)
        if result.total_pose_diff is not None and not np.isinf(result.total_pose_diff):
            normalized = max(
                0, 1 - result.total_pose_diff / self.thresholds.pose_angle_diff
            )
            scores.append(normalized)
            weights.append(0.1)

        if len(scores) == 0:
            return 0.0

        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        overall = np.average(scores, weights=weights)

        return float(overall)

    def _print_results(self, result: EvaluationResult):
        """Print evaluation results."""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        print("\nIdentity Metrics:")
        if result.face_similarity is not None:
            # status = "PASS" if result.face_similarity >= self.thresholds.face_similarity else "FAIL"
            print(f"  Face Similarity: {result.face_similarity:.3f} (threshold: {self.thresholds.face_similarity})")

        if result.landmark_rmse is not None:
            # status = "PASS" if result.landmark_rmse <= self.thresholds.landmark_rmse else "FAIL"
            print(f"  Landmark RMSE: {result.landmark_rmse:.2f}px (threshold: {self.thresholds.landmark_rmse}px)")

        if result.lpips is not None:
            # status = "PASS" if result.lpips <= self.thresholds.lpips else "FAIL"
            print(f"  LPIPS: {result.lpips:.3f} (threshold: {self.thresholds.lpips})")

        print("\nStructural Metrics:")
        if result.background_ssim is not None:
            # status = "PASS" if result.background_ssim >= self.thresholds.background_ssim else "FAIL"
            print(f"  Background SSIM: {result.background_ssim:.3f} (threshold: {self.thresholds.background_ssim})")

        if result.total_pose_diff is not None and not np.isinf(result.total_pose_diff):
            print(f"  Pose Difference: {result.total_pose_diff:.2f}° (threshold: {self.thresholds.pose_angle_diff}°)")
            if result.yaw_diff is not None:
                print(f"     - Yaw: {result.yaw_diff:.2f}°")
                print(f"     - Pitch: {result.pitch_diff:.2f}°")
                print(f"     - Roll: {result.roll_diff:.2f}°")

        print("\nOverall Assessment:")
        print(f"  Passed: {result.pass_count}/{result.total_count} checks")
        if result.missing_required_metrics:
            print(
                "  Incomplete: missing "
                + ", ".join(result.missing_required_metrics)
            )
        print(f"  Overall Score: {result.overall_score:.3f}")
        print(f"  Disentangled: {'YES' if result.is_disentangled else 'NO'}")
        print("=" * 60 + "\n")

    def evaluate_batch(
        self,
        pairs: List[Tuple[Image.Image, Image.Image]],
        subject_ids: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate multiple pairs and return DataFrame.

        Args:
            pairs: List of (original, counterfactual) tuples
            subject_ids: Optional list of subject IDs
            verbose: Print progress

        Returns:
            DataFrame with results for all pairs
        """
        results = []

        if subject_ids is None:
            subject_ids = [f"subject_{i:03d}" for i in range(len(pairs))]

        for i, (original, counterfactual) in enumerate(pairs):
            if verbose:
                print(f"Evaluating pair {i+1}/{len(pairs)}: {subject_ids[i]}")

            result = self.evaluate_pair(original, counterfactual)
            result_dict = result.to_dict()
            result_dict["subject_id"] = subject_ids[i]
            results.append(result_dict)

        df = pd.DataFrame(results)

        # Reorder columns
        cols = ["subject_id"] + [c for c in df.columns if c != "subject_id"]
        df = df[cols]

        return df

    def summarize_results(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Summarize evaluation results.

        Args:
            df: DataFrame from evaluate_batch

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # Percentage disentangled
        summary["pct_disentangled"] = df["is_disentangled"].mean() * 100

        # Average scores
        numeric_cols = [
            "face_similarity",
            "landmark_rmse",
            "lpips",
            "background_ssim",
            "total_pose_diff",
            "overall_score",
        ]

        for col in numeric_cols:
            if col in df.columns:
                summary[f"mean_{col}"] = df[col].mean()
                summary[f"std_{col}"] = df[col].std()
                summary[f"median_{col}"] = df[col].median()

        return summary

    def print_summary(self, summary: Dict[str, float]):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        print(f"\nDisentangled: {summary['pct_disentangled']:.1f}%")
        print("\nAverage Metrics:")
        print(f"  Face Similarity: {summary.get('mean_face_similarity', 0):.3f} ± {summary.get('std_face_similarity', 0):.3f}")
        print(f"  Landmark RMSE: {summary.get('mean_landmark_rmse', 0):.2f} ± {summary.get('std_landmark_rmse', 0):.2f}px")
        print(f"  LPIPS: {summary.get('mean_lpips', 0):.3f} ± {summary.get('std_lpips', 0):.3f}")
        print(f"  Background SSIM: {summary.get('mean_background_ssim', 0):.3f} ± {summary.get('std_background_ssim', 0):.3f}")
        print(f"  Pose Difference: {summary.get('mean_total_pose_diff', 0):.2f} ± {summary.get('std_total_pose_diff', 0):.2f}°")
        print(f"  Overall Score: {summary.get('mean_overall_score', 0):.3f} ± {summary.get('std_overall_score', 0):.3f}")
        print("=" * 60 + "\n")

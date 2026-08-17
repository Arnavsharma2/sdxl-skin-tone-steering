"""
Latent space manipulation tools.

This contains tools for applying transformations in latent space
while preserving desired properties.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch


@dataclass
class ManipulationConfig:
    """Config for latent manipulation."""

    alpha: float = 1.0  # Magnitude of vector application
    preserve_identity: bool = True
    preserve_background: bool = True
    preserve_pose: bool = True
    max_iterations: int = 100
    tolerance: float = 0.01


class LatentManipulator:
    """
    Tools for tweaking vectors in the latent space.

    Example:
        >>> manipulator = LatentManipulator()
        >>> modified = manipulator.apply_vector(latent, race_vector, alpha=1.0)
        >>> counterfactuals = manipulator.generate_counterfactuals(
        ...     latent, race_vector, alphas=[-2, -1, 0, 1, 2]
        ... )
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize manipulator.

        Args:
            device: Device to run on
        """
        self.device = device

    def scale_vector_magnitude(
        self,
        vector: torch.Tensor,
        target_magnitude: float,
    ) -> torch.Tensor:
        """
        Scale vector to a specific magnitude.

        Useful when working with normalized vectors that need consistent scaling.

        Args:
            vector: Input vector
            target_magnitude: Desired magnitude

        Returns:
            Scaled vector
        """
        current_magnitude = vector.norm()
        if current_magnitude < 1e-8:
            return vector
        return vector * (target_magnitude / current_magnitude)

    def apply_vector(
        self,
        latent: torch.Tensor,
        vector: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Adds a vector to the latent code.

        Basically: z' = z + α * v

        Args:
            latent: The starting latent code
            vector: The direction we want to move in
            alpha: How far to move (magnitude)

        Returns:
            The tweaked latent code
        """
        # Handle shape mismatch - resize vector to match latent spatial dimensions
        if latent.shape != vector.shape:
            import torch.nn.functional as F

            # Get spatial dimensions
            if latent.dim() == 4 and vector.dim() == 4:
                # Both have batch dimension
                if latent.shape[2:] != vector.shape[2:]:
                    # Resize vector to match latent
                    vector = F.interpolate(
                        vector,
                        size=latent.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
            elif latent.dim() == 3 and vector.dim() == 3:
                # No batch dimension
                if latent.shape[1:] != vector.shape[1:]:
                    # Add batch dim, resize, remove batch dim
                    vector = F.interpolate(
                        vector.unsqueeze(0),
                        size=latent.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
            elif latent.dim() == 4 and vector.dim() == 3:
                # Latent has batch, vector doesn't
                if latent.shape[2:] != vector.shape[1:]:
                    vector = F.interpolate(
                        vector.unsqueeze(0),
                        size=latent.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
            elif latent.dim() == 3 and vector.dim() == 4:
                # Vector has batch, latent doesn't
                vector = vector.squeeze(0)
                if latent.shape[1:] != vector.shape[1:]:
                    vector = F.interpolate(
                        vector.unsqueeze(0),
                        size=latent.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

        return latent + alpha * vector

    def apply_multiple_vectors(
        self,
        latent: torch.Tensor,
        vectors: Dict[str, torch.Tensor],
        alphas: Dict[str, float],
    ) -> torch.Tensor:
        """
        Apply multiple vectors simultaneously.

        Useful for independent control of multiple attributes.

        Args:
            latent: Original latent code
            vectors: Dict of named vectors
            alphas: Dict of magnitudes for each vector

        Returns:
            Modified latent code
        """
        modified = latent.clone()

        for name, vector in vectors.items():
            alpha = alphas.get(name, 1.0)
            modified = modified + alpha * vector

        return modified

    def generate_counterfactuals(
        self,
        latent: torch.Tensor,
        vector: torch.Tensor,
        alphas: List[float],
    ) -> List[torch.Tensor]:
        """
        Generate multiple counterfactuals at different magnitudes.

        Args:
            latent: Original latent code
            vector: Direction vector
            alphas: List of magnitudes to apply

        Returns:
            List of modified latent codes
        """
        counterfactuals = []

        for alpha in alphas:
            modified = self.apply_vector(latent, vector, alpha)
            counterfactuals.append(modified)

        return counterfactuals

    def constrained_manipulation(
        self,
        latent: torch.Tensor,
        vector: torch.Tensor,
        alpha: float,
        constraint_fn: Optional[Callable] = None,
        max_iterations: int = 50,
    ) -> torch.Tensor:
        """
        Apply vector with constraints.

        Iteratively adjusts the manipulation to satisfy constraints
        (e.g., identity preservation).

        Args:
            latent: Original latent code
            vector: Direction vector
            alpha: Target magnitude
            constraint_fn: Function that returns True if constraints satisfied
            max_iterations: Maximum refinement iterations

        Returns:
            Constrained modified latent code
        """
        if constraint_fn is None:
            # No constraints, just apply directly
            return self.apply_vector(latent, vector, alpha)

        # Start with full application
        modified = self.apply_vector(latent, vector, alpha)

        # Iteratively refine
        current_alpha = alpha
        for i in range(max_iterations):
            # Check constraints
            if constraint_fn(latent, modified):
                break

            # Reduce magnitude
            current_alpha *= 0.9
            modified = self.apply_vector(latent, vector, current_alpha)

        return modified

    def project_to_subspace(
        self,
        latent: torch.Tensor,
        subspace_vectors: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Project latent to a subspace.

        Useful for restricting manipulation to specific dimensions.

        Args:
            latent: Latent code to project
            subspace_vectors: Basis vectors defining subspace

        Returns:
            Projected latent code
        """
        # Flatten
        latent_flat = latent.flatten()

        # Stack basis vectors
        basis = torch.stack([v.flatten() for v in subspace_vectors])

        # Orthonormalize basis (Gram-Schmidt)
        basis_ortho = torch.zeros_like(basis)
        for i in range(len(basis)):
            v = basis[i]
            for j in range(i):
                v = v - (v @ basis_ortho[j]) * basis_ortho[j]
            basis_ortho[i] = v / (v.norm() + 1e-8)

        # Project
        projection = torch.zeros_like(latent_flat)
        for b in basis_ortho:
            projection = projection + (latent_flat @ b) * b

        # Reshape
        return projection.reshape(latent.shape)

    def interpolate(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        num_steps: int = 5,
        interpolation_type: str = "linear",
    ) -> List[torch.Tensor]:
        """
        Interpolate between two latent codes.

        Args:
            latent1: Start latent
            latent2: End latent
            num_steps: Number of interpolation steps
            interpolation_type: 'linear' or 'spherical'

        Returns:
            List of interpolated latent codes
        """
        if interpolation_type == "linear":
            return self._linear_interpolation(latent1, latent2, num_steps)
        elif interpolation_type == "spherical":
            return self._spherical_interpolation(latent1, latent2, num_steps)
        else:
            raise ValueError(f"Unknown interpolation type: {interpolation_type}")

    def _linear_interpolation(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        num_steps: int,
    ) -> List[torch.Tensor]:
        """Linear interpolation between latents."""
        alphas = torch.linspace(0, 1, num_steps)
        interpolated = []

        for alpha in alphas:
            interp = (1 - alpha) * latent1 + alpha * latent2
            interpolated.append(interp)

        return interpolated

    def _spherical_interpolation(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        num_steps: int,
    ) -> List[torch.Tensor]:
        """
        Spherical linear interpolation (SLERP).

        Better for high-dimensional spaces.
        """
        # Flatten
        v1 = latent1.flatten()
        v2 = latent2.flatten()

        # Normalize
        v1_norm = v1 / (v1.norm() + 1e-8)
        v2_norm = v2 / (v2.norm() + 1e-8)

        # Compute angle
        dot = (v1_norm @ v2_norm).clamp(-1, 1)
        theta = torch.acos(dot)

        # Interpolate
        sin_theta = torch.sin(theta)
        alphas = torch.linspace(0, 1, num_steps)
        interpolated = []

        for alpha in alphas:
            if sin_theta < 1e-6:
                # Vectors are parallel, use linear interpolation
                interp = (1 - alpha) * v1 + alpha * v2
            else:
                # SLERP formula
                a = torch.sin((1 - alpha) * theta) / sin_theta
                b = torch.sin(alpha * theta) / sin_theta
                interp = a * v1 + b * v2

            interpolated.append(interp.reshape(latent1.shape))

        return interpolated

    def find_optimal_alpha(
        self,
        latent: torch.Tensor,
        vector: torch.Tensor,
        target_fn: Callable,
        alpha_range: tuple = (-3.0, 3.0),
        num_samples: int = 20,
    ) -> float:
        """
        Find optimal alpha for a given objective.

        Args:
            latent: Original latent code
            vector: Direction vector
            target_fn: Function to optimize (takes modified latent, returns score)
            alpha_range: Range to search
            num_samples: Number of alpha values to try

        Returns:
            Optimal alpha value
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_samples)
        best_alpha = 0.0
        best_score = float("-inf")

        for alpha in alphas:
            modified = self.apply_vector(latent, vector, alpha)
            score = target_fn(modified)

            if score > best_score:
                best_score = score
                best_alpha = alpha

        return best_alpha

    def compute_latent_statistics(
        self,
        latents: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute statistics over a set of latents.

        Useful for understanding latent space distribution.

        Args:
            latents: List of latent codes

        Returns:
            Dict with mean, std, min, max
        """
        stacked = torch.stack(latents)

        return {
            "mean": stacked.mean(dim=0),
            "std": stacked.std(dim=0),
            "min": stacked.min(dim=0)[0],
            "max": stacked.max(dim=0)[0],
        }

    def clip_to_valid_range(
        self,
        latent: torch.Tensor,
        latent_stats: Dict[str, torch.Tensor],
        num_std: float = 3.0,
    ) -> torch.Tensor:
        """
        Clip latent to valid range.

        Prevents out-of-distribution latents that may produce artifacts.

        Args:
            latent: Latent code to clip
            latent_stats: Statistics from compute_latent_statistics
            num_std: Number of standard deviations to allow

        Returns:
            Clipped latent code
        """
        mean = latent_stats["mean"]
        std = latent_stats["std"]

        lower_bound = mean - num_std * std
        upper_bound = mean + num_std * std

        clipped = torch.clamp(latent, lower_bound, upper_bound)

        return clipped

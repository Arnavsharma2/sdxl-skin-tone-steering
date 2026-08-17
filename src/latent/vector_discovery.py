"""Skin-tone direction extraction and analysis.

The implementation estimates a visual skin-tone direction; it does not infer,
represent, or change a person's race.  ``RaceVectorExtractor`` remains as a
compatibility alias for older notebooks and scripts.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA


class SkinToneDirectionExtractor:
    """
    Estimates a linear skin-tone direction in VAE latent space.

    Supported estimators include paired differences, a difference of group
    means, and a label-correlated PCA component.

    Example:
        >>> extractor = SkinToneDirectionExtractor(method="paired_difference")
        >>> direction = extractor.extract_from_pairs(light_latents, dark_latents)
    """

    def __init__(
        self,
        method: str = "paired_difference",
        device: str = "cuda",
    ):
        """
        Initialize vector extractor.

        Args:
            method: ``paired_difference``, ``difference_of_means``, or ``pca``
            device: Device to run on
        """
        self.method = method
        self.device = device

    def create_center_mask(
        self,
        height: int,
        width: int,
        center_weight: float = 1.0,
        edge_weight: float = 0.1,
        falloff: str = "gaussian",
        radius: float = 0.6,
    ) -> torch.Tensor:
        """
        Create a spatial mask that focuses on the center (where faces typically are).

        This helps attenuate the direction in background regions.

        Args:
            height: Mask height (typically 128 for SDXL latents)
            width: Mask width (typically 128 for SDXL latents)
            center_weight: Weight at the center (default: 1.0)
            edge_weight: Weight at the edges (default: 0.1 to reduce background effects)
            falloff: 'gaussian', 'linear', or 'hard' (hard circular cutoff)
            radius: Radius for falloff (0.0-1.0, default: 0.6)
                   Only affects gaussian and hard modes

        Returns:
            Spatial mask tensor of shape (H, W)
        """
        # Create coordinate grids
        y = torch.linspace(-1, 1, height)
        x = torch.linspace(-1, 1, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Distance from center (normalized to 0-1.414)
        dist = torch.sqrt(xx**2 + yy**2)

        if falloff == "gaussian":
            # Gaussian falloff with configurable radius
            # Stronger falloff to reach edge_weight at edges
            sigma = radius
            mask = torch.exp(-3 * (dist / sigma) ** 2)
        elif falloff == "hard":
            # Hard circular mask - everything outside radius is edge_weight
            mask = (dist <= radius).float()
        else:  # linear
            # Linear falloff
            mask = 1 - (dist / radius).clamp(0, 1)

        # Scale to desired range
        mask = edge_weight + (center_weight - edge_weight) * mask

        return mask.to(self.device)

    def extract_from_pairs(
        self,
        latents_a: List[torch.Tensor],
        latents_b: List[torch.Tensor],
        normalize: bool = False,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the vector by comparing pairs of images.

        It just takes the average difference between the two groups.

        Args:
            latents_a: Latents for group A (e.g. light skin)
            latents_b: Latents for group B (e.g. dark skin)
            normalize: Whether to make the vector unit length (default: False to preserve magnitude)
            spatial_mask: Optional spatial mask to focus on specific regions (e.g., face)
                         Shape: (H, W) with values 0-1, where 1 = full weight

        Returns:
            The extracted skin-tone direction.
        """
        if not latents_a or not latents_b:
            raise ValueError("Both latent groups must contain at least one sample")
        if len(latents_a) != len(latents_b):
            raise ValueError("Must have same number of latents in each group")

        expected_shape = latents_a[0].shape
        if any(lat.shape != expected_shape for lat in [*latents_a, *latents_b]):
            raise ValueError("All latents must have the same shape")

        # Compute pairwise differences
        differences = []
        for lat_a, lat_b in zip(latents_a, latents_b):
            diff = lat_b - lat_a
            differences.append(diff)

        # Average all differences
        race_vector = torch.stack(differences).mean(dim=0)
        if not bool(torch.isfinite(race_vector).all()):
            raise ValueError(
                "Paired direction contains non-finite values; check VAE encoding precision"
            )

        # Apply spatial mask if provided (to focus on face regions)
        if spatial_mask is not None:
            # Ensure mask is on same device
            spatial_mask = spatial_mask.to(race_vector.device)
            # Expand mask to all channels: (C, H, W)
            if spatial_mask.dim() == 2:
                spatial_mask = spatial_mask.unsqueeze(0).expand_as(race_vector)
            race_vector = race_vector * spatial_mask

        # Normalize (typically we want to preserve magnitude for better control)
        if normalize:
            race_vector = race_vector / (race_vector.norm() + 1e-8)

        return race_vector

    def extract_from_groups(
        self,
        group_a_latents: List[torch.Tensor],
        group_b_latents: List[torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract a skin-tone direction from two groups (unpaired).

        Approach: difference of means.

        Args:
            group_a_latents: Latent codes for group A
            group_b_latents: Latent codes for group B
            normalize: Normalize the vector

        Returns:
            Race vector
        """
        if not group_a_latents or not group_b_latents:
            raise ValueError("Both latent groups must contain at least one sample")
        expected_shape = group_a_latents[0].shape
        if any(lat.shape != expected_shape for lat in [*group_a_latents, *group_b_latents]):
            raise ValueError("All latents must have the same shape")

        # Compute means
        mean_a = torch.stack(group_a_latents).mean(dim=0)
        mean_b = torch.stack(group_b_latents).mean(dim=0)

        # Compute difference
        race_vector = mean_b - mean_a

        # Normalize
        if normalize:
            race_vector = race_vector / (race_vector.norm() + 1e-8)

        return race_vector

    def pca_based_extraction(
        self,
        latents: List[torch.Tensor],
        labels: np.ndarray,
        n_components: int = 10,
    ) -> torch.Tensor:
        """
        Extract a label-correlated direction using PCA.

        Find principal component that correlates most with race labels.

        Args:
            latents: All latent codes
            labels: Binary labels (0 or 1) for race attribute
            n_components: Number of PCA components

        Returns:
            Race vector (principal component)
        """
        # Stack latents
        latents_stacked = torch.stack(latents).cpu().numpy()

        # Flatten to 2D
        orig_shape = latents_stacked.shape
        latents_flat = latents_stacked.reshape(len(latents), -1)

        # Apply PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(latents_flat)

        # Find component that best separates labels
        best_corr = 0
        best_idx = 0
        for i in range(n_components):
            corr = np.abs(np.corrcoef(components[:, i], labels)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_idx = i

        # Get best component
        race_vector_flat = pca.components_[best_idx]

        # Reshape back to latent shape
        race_vector = race_vector_flat.reshape(orig_shape[1:])
        race_vector = torch.from_numpy(race_vector).to(self.device)

        # Normalize
        race_vector = race_vector / (race_vector.norm() + 1e-8)

        return race_vector

    def optimize_vector(
        self,
        initial_vector: torch.Tensor,
        latents: List[torch.Tensor],
        identity_loss_fn,
        attribute_change_fn,
        num_iterations: int = 100,
        lr: float = 0.01,
        lambda_identity: float = 0.7,
        lambda_attribute: float = 0.3,
    ) -> torch.Tensor:
        """
        Refine vector to maximize disentanglement.

        Objective:
            L = λ_identity * L_identity + λ_attribute * (-L_attribute)

        Args:
            initial_vector: Starting vector
            latents: Training latent codes
            identity_loss_fn: Function to compute identity loss
            attribute_change_fn: Function to compute attribute change
            num_iterations: Optimization steps
            lr: Learning rate
            lambda_identity: Weight for identity preservation
            lambda_attribute: Weight for attribute change

        Returns:
            Optimized direction.
        """
        vector = initial_vector.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([vector], lr=lr)

        for i in range(num_iterations):
            optimizer.zero_grad()

            # Sample batch of latents
            batch_size = min(8, len(latents))
            batch_indices = np.random.choice(len(latents), batch_size, replace=False)
            batch_latents = [latents[i] for i in batch_indices]

            # Apply vector
            modified_latents = [lat + vector for lat in batch_latents]

            # Compute losses
            identity_loss = identity_loss_fn(batch_latents, modified_latents)
            attribute_change = attribute_change_fn(batch_latents, modified_latents)

            # Combined loss (minimize identity loss, maximize attribute change)
            loss = lambda_identity * identity_loss - lambda_attribute * attribute_change

            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print(
                    f"Iter {i+1}/{num_iterations}: "
                    f"Identity Loss: {identity_loss.item():.4f}, "
                    f"Attribute Change: {attribute_change.item():.4f}"
                )

        return vector.detach()

    def decompose_into_subvectors(
        self,
        race_vector: torch.Tensor,
        n_components: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose a direction into orthogonal subcomponents.

        Useful for finer-grained control (e.g., skin tone, hair, features).

        Args:
            race_vector: Full direction (legacy parameter name).
            n_components: Number of subcomponents

        Returns:
            Dictionary of subvectors
        """
        # Drop batch dim if present so we can treat the vector as (C, H, W)
        v = race_vector
        had_batch = v.dim() == 4
        if had_batch:
            v = v.squeeze(0)

        C, H, W = v.shape
        # SVD of (C, H*W) gives orthogonal rank-1 components ordered by contribution
        v_2d = v.reshape(C, -1).cpu().float().numpy()
        U, S, Vt = np.linalg.svd(v_2d, full_matrices=False)

        labels = ["primary", "secondary", "tertiary"]
        subvectors = {}
        for i in range(min(n_components, len(S))):
            comp = (S[i] * np.outer(U[:, i], Vt[i, :])).reshape(C, H, W)
            if had_batch:
                comp = comp[np.newaxis]
            subvectors[labels[i]] = torch.from_numpy(comp).to(race_vector.dtype).to(self.device)

        return subvectors

    def save_vector(self, vector: torch.Tensor, path: Path):
        """Save a direction tensor to disk."""
        torch.save(vector.cpu(), path)
        print(f"Saved direction to {path}")

    def load_vector(self, path: Path) -> torch.Tensor:
        """Load a direction tensor from disk."""
        vector = torch.load(path, map_location=self.device, weights_only=True)
        print(f"Loaded direction from {path}")
        return vector


class VectorAnalyzer:
    """
    Analyze properties of discovered latent directions.

    Tools for understanding:
    - Vector magnitude
    - Activation patterns
    - Correlation with known attributes
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def compute_magnitude(self, vector: torch.Tensor) -> float:
        """Compute L2 norm of vector."""
        return vector.norm().item()

    def analyze_spatial_pattern(
        self,
        vector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze spatial activation pattern.

        Returns:
            Dict with statistics per channel and spatial location
        """
        # Remove batch dimension if present (shape: [1, C, H, W] -> [C, H, W])
        if vector.dim() == 4:
            vector = vector.squeeze(0)

        # Per-channel magnitude
        per_channel = vector.pow(2).sum(dim=(1, 2))

        # Spatial heatmap (average across channels)
        spatial_heatmap = vector.pow(2).mean(dim=0)

        return {
            "per_channel_magnitude": per_channel,
            "spatial_heatmap": spatial_heatmap,
            "total_magnitude": self.compute_magnitude(vector),
        }

    def compute_orthogonality(
        self,
        vector1: torch.Tensor,
        vector2: torch.Tensor,
    ) -> float:
        """
        Compute orthogonality between two vectors.

        Returns:
            Cosine similarity (0 = orthogonal, 1 = parallel)
        """
        v1_flat = vector1.flatten()
        v2_flat = vector2.flatten()

        cosine_sim = torch.nn.functional.cosine_similarity(
            v1_flat.unsqueeze(0),
            v2_flat.unsqueeze(0),
        )

        return cosine_sim.item()


# Backwards compatibility for the original public API. New code should use the
# scientifically narrower name above: skin tone is observable appearance and
# must not be used as a proxy for race or identity.
RaceVectorExtractor = SkinToneDirectionExtractor

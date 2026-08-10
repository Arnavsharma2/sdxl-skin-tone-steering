"""
Disentanglement metrics.

This module implements metrics for measuring disentanglement quality:
- SAP (Separated Attribute Predictability)
- MIG (Mutual Information Gap)
- DCI (Disentanglement, Completeness, Informativeness)
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class DisentanglementMetrics:
    """
    Measures degree of disentanglement in learned representations.

    These metrics require labeled data with known attributes.

    Example:
        >>> metrics = DisentanglementMetrics()
        >>> sap = metrics.sap_score(latents, race_labels)
        >>> print(f"SAP Score: {sap:.3f}")
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize disentanglement metrics.

        Args:
            device: Device to run on
        """
        self.device = device

    def sap_score(
        self,
        latents: List[torch.Tensor],
        attributes: np.ndarray,
        n_components: Optional[int] = None,
    ) -> float:
        """
        Compute Separated Attribute Predictability (SAP) score.

        Measures how well a single latent dimension can predict an attribute.

        Args:
            latents: List of latent codes
            attributes: Attribute labels (N,) array
            n_components: Number of latent components to consider (all if None)

        Returns:
            SAP score in [0, 1]. Higher = more disentangled.
        """
        # Stack and flatten latents
        latents_stacked = torch.stack(latents).cpu().numpy()
        latents_flat = latents_stacked.reshape(len(latents), -1)

        if n_components is None:
            n_components = latents_flat.shape[1]

        # For each latent dimension, train classifier to predict attribute
        accuracies = []

        for i in range(min(n_components, latents_flat.shape[1])):
            X = latents_flat[:, i].reshape(-1, 1)
            y = attributes

            # Train classifier
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X, y)

            # Evaluate
            y_pred = clf.predict(X)
            acc = accuracy_score(y, y_pred)
            accuracies.append(acc)

        # SAP = difference between top two scores
        accuracies = sorted(accuracies, reverse=True)
        if len(accuracies) < 2:
            return 0.0

        sap = accuracies[0] - accuracies[1]

        return float(sap)

    def mig_score(
        self,
        latents: List[torch.Tensor],
        attributes: np.ndarray,
    ) -> float:
        """
        Compute Mutual Information Gap (MIG) score.

        Measures mutual information between latent dimensions and attributes.

        Args:
            latents: List of latent codes
            attributes: Attribute labels

        Returns:
            MIG score. Higher = more disentangled.
        """
        # Stack and flatten latents
        latents_stacked = torch.stack(latents).cpu().numpy()
        latents_flat = latents_stacked.reshape(len(latents), -1)

        # Compute mutual information for each latent dimension
        mi_scores = []

        for i in range(latents_flat.shape[1]):
            # Discretize latent dimension
            latent_dim = latents_flat[:, i]
            latent_discretized = np.digitize(latent_dim, bins=np.linspace(latent_dim.min(), latent_dim.max(), 20))

            # Compute mutual information with attribute
            mi = self._mutual_information(latent_discretized, attributes)
            mi_scores.append(mi)

        # MIG = difference between top two MI scores
        mi_scores = sorted(mi_scores, reverse=True)
        if len(mi_scores) < 2:
            return 0.0

        # Normalize by entropy of attribute
        H_attr = stats.entropy(np.bincount(attributes) / len(attributes))
        if H_attr == 0:
            return 0.0

        mig = (mi_scores[0] - mi_scores[1]) / H_attr

        return float(mig)

    def _mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information between two discrete variables.

        Args:
            x: First variable
            y: Second variable

        Returns:
            Mutual information
        """
        # Joint probability
        contingency = np.histogram2d(x, y, bins=(len(np.unique(x)), len(np.unique(y))))[0]
        p_xy = contingency / contingency.sum()

        # Marginal probabilities
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        # Compute MI
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

        return mi

    def dci_score(
        self,
        latents: List[torch.Tensor],
        attributes: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute DCI (Disentanglement, Completeness, Informativeness) scores.

        Args:
            latents: List of latent codes
            attributes: Attribute labels

        Returns:
            Dict with 'disentanglement', 'completeness', 'informativeness'
        """
        # Stack and flatten latents
        latents_stacked = torch.stack(latents).cpu().numpy()
        latents_flat = latents_stacked.reshape(len(latents), -1)

        # Train gradient boosting to predict attribute from latents
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(latents_flat, attributes)

        # Get feature importances
        importances = clf.feature_importances_

        # Disentanglement: how concentrated is importance (entropy)
        importance_sum = importances.sum()
        if importance_sum < 1e-8:
            importance_dist = np.ones_like(importances) / len(importances)
        else:
            importance_dist = importances / importance_sum

        disentanglement = 1.0 - stats.entropy(importance_dist) / np.log(len(importances) + 1e-8)

        # Informativeness: prediction accuracy
        y_pred = clf.predict(latents_flat)
        informativeness = accuracy_score(attributes, y_pred)

        # Completeness: how many dimensions are needed
        # (simplified version - ideally should measure per attribute)
        sorted_importances = np.sort(importances)[::-1]
        cumsum = np.cumsum(sorted_importances)
        n_needed = np.argmax(cumsum >= 0.99 * cumsum[-1]) + 1
        completeness = 1.0 - (n_needed / len(importances))

        return {
            "disentanglement": float(disentanglement),
            "completeness": float(completeness),
            "informativeness": float(informativeness),
        }

    def compute_all_metrics(
        self,
        latents: List[torch.Tensor],
        attributes: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all disentanglement metrics.

        Args:
            latents: List of latent codes
            attributes: Attribute labels

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        try:
            metrics["sap"] = self.sap_score(latents, attributes)
        except Exception as e:
            print(f"WARNING: Could not compute SAP: {e}")
            metrics["sap"] = None

        try:
            metrics["mig"] = self.mig_score(latents, attributes)
        except Exception as e:
            print(f"WARNING: Could not compute MIG: {e}")
            metrics["mig"] = None

        try:
            dci = self.dci_score(latents, attributes)
            metrics.update(dci)
        except Exception as e:
            print(f"WARNING: Could not compute DCI: {e}")
            metrics["disentanglement"] = None
            metrics["completeness"] = None
            metrics["informativeness"] = None

        return metrics

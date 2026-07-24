"""Mahalanobis++: Mahalanobis distance on L2-normalized features.

Mueller & Hein, "Mahalanobis++: Improving Out-of-Distribution Detection via
Feature Normalization" (ICML 2025): projecting penultimate features onto the
unit sphere before fitting class means and the shared covariance removes the
feature-norm pathology that makes vanilla Mahalanobis unreliable. Everything
else (shared covariance, nearest-centroid negative distance, save/load)
reuses `MahalanobisDistance` unchanged.
"""

import torch
from fd_shifts import logger

from src.csfs.mahalanobis import MahalanobisDistance

ArrayType = torch.Tensor
_EPS = 1e-10


def _l2_normalize_rows(activations: ArrayType) -> ArrayType:
    """Project features onto the unit sphere (row-wise L2 normalization)."""
    return activations / (activations.norm(dim=1, keepdim=True) + _EPS)


class MahalanobisPP(MahalanobisDistance):
    """Mahalanobis distance fitted and evaluated on L2-normalized features."""

    def compute_MahaDist_params(self, activations_train: ArrayType,
                                labels_train: ArrayType) -> None:
        logger.info("MahalanobisPP: Fitting on L2-normalized features...")
        super().compute_MahaDist_params(
            _l2_normalize_rows(activations_train), labels_train)

    def get_scores(self, activations_eval: ArrayType,
                   batch_size: int = 1000) -> ArrayType:
        return super().get_scores(_l2_normalize_rows(activations_eval),
                                  batch_size=batch_size)

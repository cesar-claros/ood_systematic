"""Rank-aware ID geometry and explicitly paired representation measurements.

These functions leave the historical X8 descriptor fields unchanged. Class-mean
spans use the numerical rank of empirical means, not a population-rank estimate.
Paired linear CKA is a shape similarity, not an OOD-performance certificate.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _features(values: ArrayLike) -> FloatArray:
    """Convert a finite, nonempty sample-by-feature matrix to float64."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.isfinite(matrix).all():
        raise ValueError("Features must be a finite, nonempty two-dimensional matrix.")
    return matrix


def _class_mean_basis(
    features: FloatArray,
    labels: ArrayLike,
    n_classes: int,
    rank_rtol: float | None,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Return the numerical mean-span basis, class means and validated labels."""
    targets = np.asarray(labels)
    if (
        n_classes < 1
        or targets.ndim != 1
        or len(targets) != len(features)
        or targets.dtype.kind not in "iu"
        or np.any(targets < 0)
        or np.any(targets >= n_classes)
    ):
        raise ValueError("Labels must be integers in [0, n_classes) for every sample.")
    indices = targets.astype(np.int64, copy=False)
    counts = np.bincount(indices, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("Every declared class must have at least one sample.")
    first_indices = np.full(n_classes, len(features), dtype=np.int64)
    np.minimum.at(first_indices, indices, np.arange(len(features)))
    anchors = features[first_indices]
    offsets = features - anchors[indices]
    means = np.zeros((n_classes, features.shape[1]), dtype=np.float64)
    np.add.at(means, indices, offsets)
    means = anchors + means / counts[:, None]
    shifted_means = means - means[0]
    centered = shifted_means - shifted_means.mean(axis=0)
    if rank_rtol is None:
        rank_rtol = max(centered.shape) * np.finfo(np.float64).eps
    if not np.isfinite(rank_rtol) or not 0 <= rank_rtol < 1:
        raise ValueError("rank_rtol must be finite and in [0, 1).")
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    keep = singular_values > rank_rtol * singular_values[0]
    return right_vectors[keep].T, means, indices


def mean_span_projector(
    features: ArrayLike,
    labels: ArrayLike,
    n_classes: int,
    *,
    rank_rtol: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return the centered-class-mean projector and uncentered class means.

    Args:
        features: Finite sample-by-feature matrix.
        labels: Integer labels, one per row, covering every declared class.
        n_classes: Number of classes, indexed from zero.
        rank_rtol: Relative singular-value cutoff; defaults to matrix-size times
            float64 epsilon. This is a numerical, not statistical, rank threshold.

    Returns:
        A feature-by-feature orthogonal projector and class-by-feature means.

    Raises:
        ValueError: Inputs are invalid or a declared class is missing.
    """
    basis, means, _ = _class_mean_basis(
        _features(features), labels, n_classes, rank_rtol
    )
    return basis @ basis.T, means


def residue_energy_projector(
    features: ArrayLike,
    labels: ArrayLike,
    n_classes: int,
    *,
    rank_rtol: float | None = None,
) -> float:
    """Return within-class energy outside the empirical centered-mean span.

    Args:
        features: Finite sample-by-feature matrix; rows are equally weighted.
        labels: Integer labels covering every class from zero to n_classes - 1.
        n_classes: Number of classes. Centering of class means is unweighted.
        rank_rtol: Numerical relative rank cutoff, as in mean_span_projector.

    Returns:
        Residue energy in [0, 1], evaluated without a dense covariance matrix.

    Raises:
        ValueError: Inputs are invalid, a class is missing, or within-class
            variation is exactly zero, making the fraction undefined.
    """
    matrix = _features(features)
    basis, means, indices = _class_mean_basis(matrix, labels, n_classes, rank_rtol)
    within = matrix - means[indices]
    scale = float(np.max(np.abs(within)))
    if scale == 0:
        raise ValueError("Residue energy is undefined with zero within-class variance.")
    within /= scale
    residual = within - (within @ basis) @ basis.T
    fraction = float(np.sum(residual**2) / np.sum(within**2))
    return float(np.clip(fraction, 0.0, 1.0))


def paired_linear_cka(
    reference: ArrayLike,
    adapted: ArrayLike,
    reference_ids: ArrayLike,
    adapted_ids: ArrayLike,
) -> float:
    """Measure centered linear CKA on explicitly corresponding ID examples.

    Args:
        reference: Reference sample-by-feature matrix with at least three rows.
        adapted: Adapted features of the same examples in the same order.
        reference_ids: Unique sample identifiers, one per reference row.
        adapted_ids: Identical identifiers in identical order; preprocessing and
            example identity must also be correct in the extraction pipeline.

    Returns:
        Biased empirical linear CKA in [0, 1]. It is invariant to orthogonal
        feature rotations, translation and nonzero uniform scaling. It measures
        paired shape similarity, not norm/head drift or preserved OOD detection.

    Raises:
        ValueError: Matrices/IDs are invalid, IDs are duplicated or misaligned,
            there are fewer than three pairs, or either centered matrix is zero.
    """
    first, second = _features(reference), _features(adapted)
    first_ids, second_ids = np.asarray(reference_ids), np.asarray(adapted_ids)
    if (
        len(first) < 3
        or len(first) != len(second)
        or first_ids.ndim != 1
        or second_ids.ndim != 1
        or len(first_ids) != len(first)
        or not np.array_equal(first_ids, second_ids)
        or len(np.unique(first_ids)) != len(first_ids)
    ):
        raise ValueError("CKA requires at least three aligned, unique sample IDs.")
    centered = []
    for matrix in (first, second):
        matrix = matrix - matrix[0]
        matrix = matrix - matrix.mean(axis=0)
        scale = float(np.max(np.abs(matrix)))
        if scale == 0:
            raise ValueError("CKA is undefined for a constant representation.")
        centered.append(matrix / scale)
    first, second = centered
    if len(first) < max(first.shape[1], second.shape[1]):
        first_gram, second_gram = first @ first.T, second @ second.T
        numerator = np.sum(first_gram * second_gram)
        denominator = np.linalg.norm(first_gram) * np.linalg.norm(second_gram)
    else:
        numerator = np.sum((first.T @ second) ** 2)
        denominator = np.linalg.norm(first.T @ first) * np.linalg.norm(
            second.T @ second
        )
    return float(np.clip(numerator / denominator, 0.0, 1.0))

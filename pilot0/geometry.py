"""Measured feature-model geometry for the intervention program.

Numpy mirror of the quantities `src/neural_collapse.py` computes plus the
G-vector extensions the flagship plan requires (section 11): within-class
covariance anisotropy and effective rank, head residual outside the
class-mean span, logit scale, and class-mean radius. All quantities are
computed in globally centered feature space (features minus the train
global mean), matching the NC dictionary convention of X1 section 1.3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureModel:
    """Centered feature-space model measured from one checkpoint's train split.

    Attributes:
        global_mean: (D,) train global mean.
        class_means: (C, D) centered class means.
        radii: (C,) centered class-mean norms.
        radius: mean of `radii` (the model R).
        sigma_w: (D, D) within-class covariance.
        sigma_iso: isotropic noise scale, sqrt(Tr(sigma_w) / D).
        span_basis: (D, k) orthonormal basis of the centered class-mean span.
        class_freq: (C,) empirical train class frequencies.
    """

    global_mean: np.ndarray
    class_means: np.ndarray
    radii: np.ndarray
    radius: float
    sigma_w: np.ndarray
    sigma_iso: float
    span_basis: np.ndarray
    class_freq: np.ndarray

    def center(self, features: np.ndarray) -> np.ndarray:
        """Center features by the train global mean."""
        return features - self.global_mean


def fit_feature_model(h_train: np.ndarray, y_train: np.ndarray,
                      n_classes: int) -> FeatureModel:
    """Fit the centered feature model from train activations.

    Args:
        h_train: (N, D) penultimate activations.
        y_train: (N,) integer labels.
        n_classes: number of classes C.

    Returns:
        FeatureModel with all measured quantities.
    """
    h64 = h_train.astype(np.float64)
    global_mean = h64.mean(0)
    centered = h64 - global_mean
    class_means = np.stack([centered[y_train == c].mean(0)
                            for c in range(n_classes)])
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    within = centered - class_means[y_train]
    sigma_w = within.T @ within / len(within)
    radii = np.linalg.norm(class_means, axis=1)
    span_basis = np.linalg.qr(class_means.T, mode="reduced")[0]
    return FeatureModel(
        global_mean=global_mean, class_means=class_means, radii=radii,
        radius=float(radii.mean()), sigma_w=sigma_w,
        sigma_iso=float(np.sqrt(np.trace(sigma_w) / sigma_w.shape[0])),
        span_basis=span_basis, class_freq=counts / counts.sum())


def anisotropy_stats(sigma_w: np.ndarray) -> dict[str, float]:
    """Within-class covariance anisotropy and effective rank.

    Returns:
        Dict with `eig_max_over_mean` (top eigenvalue over the mean
        eigenvalue; 1.0 for isotropic noise), `effective_rank`
        (exp of the eigenvalue-spectrum entropy), and
        `participation_ratio` ((sum lambda)^2 / sum lambda^2).
    """
    eigvals = np.linalg.eigvalsh(sigma_w)
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    p = eigvals[eigvals > 0] / total
    return {
        "eig_max_over_mean": float(eigvals[-1] / eigvals.mean()),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "participation_ratio": float(total**2 / (eigvals**2).sum()),
    }


def head_residual_fraction(w: np.ndarray, span_basis: np.ndarray) -> float:
    """Fraction of classifier-row energy outside the class-mean span."""
    w64 = w.astype(np.float64)
    in_span = (w64 @ span_basis) @ span_basis.T
    total = float((w64**2).sum())
    return float(((w64 - in_span) ** 2).sum() / total) if total else 0.0


def logit_scale(w: np.ndarray, b: np.ndarray, model: FeatureModel) -> float:
    """Mean target logit at the centered class means, the measured alpha*R^2.

    Defined as mean_c(w_c . mu_c + b_c), the model's analogue of the X1
    logit-scale phase variable at temperature 1.
    """
    diag = np.einsum("cd,cd->c", w.astype(np.float64), model.class_means)
    return float((diag + b.astype(np.float64)).mean())


def self_duality_angles(w: np.ndarray, model: FeatureModel) -> np.ndarray:
    """Per-class angle (radians) between classifier rows and class means."""
    w_hat = w / np.linalg.norm(w, axis=1, keepdims=True)
    mu_hat = model.class_means / model.radii[:, None]
    cos = np.clip(np.einsum("cd,cd->c", w_hat, mu_hat), -1.0, 1.0)
    return np.arccos(cos)


def papyan_self_duality(w: np.ndarray, model: FeatureModel) -> float:
    """The paper's self-duality metric ||W/||W||_F - M/||M||_F||_F^2."""
    w64 = w.astype(np.float64)
    m = model.class_means
    return float(np.sum((w64 / np.linalg.norm(w64)
                         - m / np.linalg.norm(m)) ** 2))


def _pairwise_cos_offdiag(rows: np.ndarray) -> np.ndarray:
    normed = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    cos = normed @ normed.T
    return cos[~np.eye(len(rows), dtype=bool)]


def papyan_metrics(w: np.ndarray, model: FeatureModel) -> dict[str, float]:
    """Numpy mirror of the eight Papyan coordinates in src/neural_collapse.py.

    `_uc` metrics live on the centered class means, `_wc` on the classifier
    rows; var_collapse is Tr(Sigma_W Sigma_B^+)/C; self_duality is the
    Frobenius metric already provided by `papyan_self_duality`.
    """
    m = model.class_means
    w64 = w.astype(np.float64)
    n_classes = len(m)
    beta = -1.0 / (n_classes - 1)
    sigma_b = m.T @ m / n_classes
    sigma_b = (sigma_b + sigma_b.T) / 2.0
    # hermitian=True routes pinv through eigh; the default gesdd SVD can
    # fail to converge on this rank-(C-1) PSD matrix (seen on an early
    # cadence checkpoint of the Pilot 1 sweep).
    var_collapse = float(np.trace(
        model.sigma_w @ np.linalg.pinv(sigma_b, rcond=1e-6, hermitian=True))
        / n_classes)
    cos_m = _pairwise_cos_offdiag(m)
    cos_w = _pairwise_cos_offdiag(w64)
    norms_w = np.linalg.norm(w64, axis=1)
    return {
        "var_collapse": var_collapse,
        "equinorm_uc": float(model.radii.std() / model.radii.mean()),
        "equinorm_wc": float(norms_w.std() / norms_w.mean()),
        "equiangular_uc": float(cos_m.std()),
        "equiangular_wc": float(cos_w.std()),
        "max_equiangular_uc": float(np.abs(cos_m - beta).mean()),
        "max_equiangular_wc": float(np.abs(cos_w - beta).mean()),
        "self_duality": papyan_self_duality(w64, model),
    }


def geometry_record(w: np.ndarray, b: np.ndarray,
                    model: FeatureModel) -> dict[str, float]:
    """Assemble the G-vector extension measurements for one head state."""
    record = {
        "head_residual_fraction": head_residual_fraction(w, model.span_basis),
        "logit_scale": logit_scale(w, b, model),
        "self_duality_papyan": papyan_self_duality(w, model),
        "self_duality_angle_mean_deg": float(
            np.degrees(self_duality_angles(w, model).mean())),
        "class_mean_radius": model.radius,
        "class_mean_radius_cv": float(model.radii.std() / model.radii.mean()),
        "sigma_iso": model.sigma_iso,
        "snr": float(model.radius / model.sigma_iso),
    }
    record.update(anisotropy_stats(model.sigma_w))
    return record

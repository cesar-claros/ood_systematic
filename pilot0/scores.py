"""Numpy score functions mirroring the pipeline CSF definitions.

Conventions match `src/csfs/` exactly: higher score = more in-distribution;
Energy is negated log-sum-exp at temperature 1; Mahalanobis uses one shared
within-class covariance and returns the max over classes of the negative
squared distance; CTM variants take the max cosine over classifier rows
(head) or class means (mean). Head-dependent scores: msr, mls, energy,
ctm_head. Feature-only scores: ctm_mean, maha (never touch W or b).
"""
from __future__ import annotations

import numpy as np

from pilot0.geometry import FeatureModel


def logits(h: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear head logits."""
    return h.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)


def head_scores(g: np.ndarray) -> dict[str, np.ndarray]:
    """MSR, MLS and Energy from a logit matrix."""
    shift = g.max(1, keepdims=True)
    lse = np.log(np.exp(g - shift).sum(1)) + shift[:, 0]
    return {"MSR": np.exp(g.max(1) - lse), "MLS": g.max(1), "Energy": lse}


def ctm(h: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Max cosine similarity to prototype rows (uncentered features)."""
    h_n = h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-12, None)
    p_n = prototypes / np.clip(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12, None)
    return (h_n @ p_n.T).max(1)


class MahalanobisScorer:
    """Shared-covariance Mahalanobis mirror of `src/csfs/mahalanobis.py`.

    Fit on UNcentered train activations (class means and pooled
    class-centered covariance); score = max_c of -d_c^2.
    """

    def __init__(self, h_train: np.ndarray, y_train: np.ndarray,
                 n_classes: int):
        h64 = h_train.astype(np.float64)
        self.means = np.stack([h64[y_train == c].mean(0)
                               for c in range(n_classes)])
        centered = h64 - self.means[y_train]
        cov = centered.T @ centered / len(centered)
        self.precision = np.linalg.pinv(cov, hermitian=True, rcond=1e-6)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        h64 = h.astype(np.float64)
        cross = h64 @ self.precision @ self.means.T
        h_quad = np.einsum("nd,dk,nk->n", h64, self.precision, h64)
        m_quad = np.einsum("cd,dk,ck->c", self.means, self.precision,
                           self.means)
        d2 = h_quad[:, None] - 2.0 * cross + m_quad[None, :]
        return -d2.min(1)


def all_scores(h: np.ndarray, w: np.ndarray, b: np.ndarray,
               model: FeatureModel, maha: MahalanobisScorer,
               mean_prototypes: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the six pilot scores for one eval set under one head state.

    Args:
        h: (N, D) uncentered activations.
        w: (C, D) head rows in effect.
        b: (C,) bias.
        model: feature model (unused for scoring; kept for signature parity).
        maha: fitted Mahalanobis scorer (baseline-head fit, head-free eval).
        mean_prototypes: (C, D) uncentered class means for mean-CTM.

    Returns:
        Dict score name -> (N,) scores, higher = more ID.
    """
    g = logits(h, w, b)
    out = head_scores(g)
    out["CTM_head"] = ctm(h, w)
    out["CTM_mean"] = ctm(h, mean_prototypes)
    out["Maha"] = maha(h)
    return out


HEAD_SCORES = ("MSR", "MLS", "Energy", "CTM_head")
FEATURE_SCORES = ("CTM_mean", "Maha")


def auroc(s_id: np.ndarray, s_ood: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUROC of ID against OOD scores."""
    from scipy.stats import rankdata
    x = np.concatenate([s_ood, s_id])
    r = rankdata(x)
    n0, n1 = len(s_ood), len(s_id)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))

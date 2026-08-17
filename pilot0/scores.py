"""Numpy score functions mirroring the pipeline CSF definitions.

Conventions match `src/csfs/` exactly: higher score = more in-distribution;
Energy is negated log-sum-exp at temperature 1; Mahalanobis uses one shared
within-class covariance and returns the max over classes of the negative
squared distance; CTM variants take the max cosine over classifier rows
(head) or class means (mean). Head-dependent scores: MSR, MLS, Energy,
CTM_head. Feature-only scores: CTM_mean, Maha (never touch W or b).

Heavy pieces are pure BLAS: callers should hoist per-eval-set quantities
(float64 cast, row normalization, Mahalanobis and mean-CTM scores) out of
any loop over head states, since only the head-dependent scores change.
"""
from __future__ import annotations

import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Unit-normalize rows (float64)."""
    x64 = x.astype(np.float64)
    return x64 / np.clip(np.linalg.norm(x64, axis=1, keepdims=True),
                         1e-12, None)


def logits(h: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear head logits."""
    return h.astype(np.float64) @ w.astype(np.float64).T + b.astype(np.float64)


def head_scores(g: np.ndarray) -> dict[str, np.ndarray]:
    """MSR, MLS and Energy from a logit matrix."""
    shift = g.max(1, keepdims=True)
    lse = np.log(np.exp(g - shift).sum(1)) + shift[:, 0]
    return {"MSR": np.exp(g.max(1) - lse), "MLS": g.max(1), "Energy": lse}


def max_cosine(h_n: np.ndarray, p_n: np.ndarray) -> np.ndarray:
    """Max cosine similarity from pre-normalized rows."""
    return (h_n @ p_n.T).max(1)


def ctm(h: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Max cosine similarity to prototype rows (uncentered features)."""
    return max_cosine(normalize_rows(h), normalize_rows(prototypes))


class MahalanobisScorer:
    """Shared-covariance Mahalanobis mirror of `src/csfs/mahalanobis.py`.

    Fit on UNcentered train activations (class means and pooled
    class-centered covariance); score = max_c of -d_c^2. The quadratic
    form is evaluated as one GEMM plus a row-wise product, never through
    a naive einsum contraction.
    """

    def __init__(self, h_train: np.ndarray, y_train: np.ndarray,
                 n_classes: int):
        h64 = h_train.astype(np.float64)
        self.means = np.stack([h64[y_train == c].mean(0)
                               for c in range(n_classes)])
        centered = h64 - self.means[y_train]
        cov = centered.T @ centered / len(centered)
        self.precision = np.linalg.pinv(cov, hermitian=True, rcond=1e-6)
        self._m_quad = np.einsum(
            "cd,dk,ck->c", self.means, self.precision, self.means,
            optimize=True)

    def __call__(self, h: np.ndarray) -> np.ndarray:
        h64 = h.astype(np.float64)
        h_prec = h64 @ self.precision
        cross = h_prec @ self.means.T
        h_quad = (h_prec * h64).sum(1)
        d2 = h_quad[:, None] - 2.0 * cross + self._m_quad[None, :]
        return -d2.min(1)


HEAD_SCORES = ("MSR", "MLS", "Energy", "CTM_head")
FEATURE_SCORES = ("CTM_mean", "Maha")


def compute_feature_scores(h: np.ndarray, maha: MahalanobisScorer,
                           mean_prototypes_n: np.ndarray
                           ) -> dict[str, np.ndarray]:
    """Feature-only scores for one eval set (head-state independent)."""
    return {"CTM_mean": max_cosine(normalize_rows(h), mean_prototypes_n),
            "Maha": maha(h)}


def compute_head_scores(h64: np.ndarray, h_n: np.ndarray, w: np.ndarray,
                        b: np.ndarray) -> dict[str, np.ndarray]:
    """Head-dependent scores for one eval set under one head state.

    Args:
        h64: (N, D) float64 activations.
        h_n: (N, D) unit-normalized activations (hoisted per eval set).
        w: (C, D) head rows in effect.
        b: (C,) bias.
    """
    out = head_scores(h64 @ w.T + b)
    out["CTM_head"] = max_cosine(h_n, normalize_rows(w))
    return out


def auroc(s_id: np.ndarray, s_ood: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUROC of ID against OOD scores."""
    from scipy.stats import rankdata
    x = np.concatenate([s_ood, s_id])
    r = rankdata(x)
    n0, n1 = len(s_ood), len(s_id)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))

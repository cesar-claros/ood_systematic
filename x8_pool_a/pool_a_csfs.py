"""Pool A CSF implementations on cached frozen features (numpy, CPU).

Faithful to the paper's Appendix C definitions (see
`paper/NeurIPS_2026_original/sections/appendix/methods_projections.tex`),
adapted to a linear probe head: logits = Z @ W.T + b on standardized features
Z = (h - mu) / sd. Head-coupled scores (CTM, fDBD, GradNorm, pNML, ViM's
energy term) operate in the standardized space; feature-manifold scores
(Maha, PCA RecError, Residual, NeCo, NNGuide bank) operate on raw features,
matching the pipeline's use of raw penultimate activations.

Every function returns CONFIDENCE (higher = more ID-like); uncertainty-style
definitions are negated here so downstream AUGRC code is uniform.
`Confidence` (the trained auxiliary head) and KPCA RecError are not available
for frozen probes and are documented as excluded from the pilot.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray) -> float:
    """Temperature minimizing validation NLL over a log-spaced grid."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 41))
    nlls = []
    for t in grid:
        p = softmax(logits_val / t)
        nlls.append(-np.log(p[np.arange(len(y_val)), y_val] + EPS).mean())
    return float(grid[int(np.argmin(nlls))])


# ---- head-side ----

def conf_msr(p: np.ndarray) -> np.ndarray:
    return p.max(axis=1)


def conf_mls(logits: np.ndarray) -> np.ndarray:
    return logits.max(axis=1)


def conf_energy(logits: np.ndarray, temp: float) -> np.ndarray:
    z = logits / temp
    m = z.max(axis=1)
    return temp * (m + np.log(np.exp(z - m[:, None]).sum(axis=1)))


def conf_pe(p: np.ndarray) -> np.ndarray:
    return (p * np.log(p + EPS)).sum(axis=1)


def conf_gen(p: np.ndarray, gamma: float, m_top: int) -> np.ndarray:
    ps = np.sort(p, axis=1)[:, ::-1][:, :m_top]
    return -np.power(ps, gamma).__mul__(np.power(1.0 - ps, gamma)).sum(axis=1)


def conf_ren(p: np.ndarray, alpha: float, m_top: int) -> np.ndarray:
    ps = np.sort(p, axis=1)[:, ::-1][:, :m_top]
    return -(1.0 / (1.0 - alpha)) * np.log(np.power(ps, alpha).sum(axis=1) + EPS)


def conf_ge(p: np.ndarray) -> np.ndarray:
    ps = np.sort(p, axis=1)[:, ::-1]
    ranks = np.arange(1, ps.shape[1] + 1)
    return -(ps * ranks).sum(axis=1)


def conf_pce(p: np.ndarray) -> np.ndarray:
    return np.log((p ** 2).sum(axis=1) + EPS)


def conf_gradnorm(p: np.ndarray, z_std: np.ndarray) -> np.ndarray:
    """L1 norm of d KL(u || p) / dW for a linear head: ||p - u||_1 * ||Z||_1."""
    u = 1.0 / p.shape[1]
    return np.abs(p - u).sum(axis=1) * np.abs(z_std).sum(axis=1)


class PNML:
    """pNML regret via the kernel-range projection of the paper's Appendix C."""

    def __init__(self, z_train: np.ndarray) -> None:
        zn = z_train / (np.linalg.norm(z_train, axis=1, keepdims=True) + EPS)
        u, s, vt = np.linalg.svd(zn, full_matrices=False)
        keep = s > 1e-8 * s[0]
        self.v = vt[keep].T
        self.inv_s2 = 1.0 / (s[keep] ** 2)

    def conf(self, z: np.ndarray, p: np.ndarray) -> np.ndarray:
        proj = z @ self.v
        in_span = proj @ self.v.T
        h_perp = z - in_span
        perp_sq = (h_perp ** 2).sum(axis=1)
        quad = ((proj ** 2) * self.inv_s2).sum(axis=1)
        hg = np.where(perp_sq > 1e-8, 1.0, quad / (1.0 + quad))
        pk_hg = np.power(p + EPS, hg[:, None])
        regret = np.log((p / (p + pk_hg * (1.0 - p) + EPS)).sum(axis=1) + EPS)
        return -regret


# ---- feature-side ----

def conf_ctm(z_std: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Max cosine similarity to the classifier weight rows (paper's CTM)."""
    zn = z_std / (np.linalg.norm(z_std, axis=1, keepdims=True) + EPS)
    wn = w / (np.linalg.norm(w, axis=1, keepdims=True) + EPS)
    return (zn @ wn.T).max(axis=1)


def l2n(h: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (Mahalanobis++ preprocessing)."""
    return h / (np.linalg.norm(h, axis=1, keepdims=True) + EPS)


def fit_nci_alpha(h_val: np.ndarray, logits_val: np.ndarray,
                  resid_val: np.ndarray, w_eff: np.ndarray,
                  train_mean: np.ndarray, rc_metric_fn,
                  alphas=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)) -> float:
    """NCI alpha minimizing validation failure-AUGRC (pipeline convention)."""
    best_alpha, best = alphas[0], np.inf
    for alpha in alphas:
        augrc, _ = rc_metric_fn(
            conf_nci(h_val, logits_val, w_eff, train_mean, alpha), resid_val)
        if augrc < best:
            best_alpha, best = alpha, augrc
    return best_alpha


def conf_nci(h: np.ndarray, logits: np.ndarray, w_eff: np.ndarray,
             train_mean: np.ndarray, alpha: float) -> np.ndarray:
    """NCI (Liu & Qin, CVPR 2025): <w_pred, h-u>/||h-u|| + alpha*||h||_1.

    Raw-feature-space form: w_eff are the probe weights mapped back to the
    unstandardized feature space (W / sd), matching the pipeline's use of
    raw penultimate activations with head weights.
    """
    centered = h - train_mean
    pred = logits.argmax(axis=1)
    align = (w_eff[pred] * centered).sum(axis=1)
    align = align / (np.linalg.norm(centered, axis=1) + EPS)
    return align + alpha * np.abs(h).sum(axis=1)


class Mahalanobis:
    """Shared-covariance Mahalanobis distance to the nearest class centroid."""

    def __init__(self, h_fit: np.ndarray, y_fit: np.ndarray, n_cls: int,
                 ridge: float = 1e-3) -> None:
        self.means = np.stack([h_fit[y_fit == c].mean(axis=0)
                               for c in range(n_cls)])
        centered = h_fit - self.means[y_fit]
        cov = centered.T @ centered / len(h_fit)
        cov += ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.prec = np.linalg.inv(cov)

    def conf(self, h: np.ndarray) -> np.ndarray:
        d = np.stack([np.einsum("nd,dk,nk->n", h - m, self.prec, h - m)
                      for m in self.means])
        return -d.min(axis=0)


class NNGuide:
    """Energy score modulated by confidence-scaled cosine to an ID bank."""

    def __init__(self, h_bank: np.ndarray, s_bank: np.ndarray, k: int) -> None:
        self.bank = h_bank / (np.linalg.norm(h_bank, axis=1, keepdims=True) + EPS)
        self.s_bank = s_bank
        self.k = min(k, len(s_bank))

    def conf(self, h: np.ndarray, s_base: np.ndarray) -> np.ndarray:
        hn = h / (np.linalg.norm(h, axis=1, keepdims=True) + EPS)
        sims = (hn @ self.bank.T) * self.s_bank[None, :]
        part = np.partition(sims, -self.k, axis=1)[:, -self.k:]
        return s_base * part.mean(axis=1)


def conf_fdbd(z_std: np.ndarray, logits: np.ndarray, w: np.ndarray,
              mu_train_std: np.ndarray) -> np.ndarray:
    """Mean boundary distance regularized by deviation from the ID mean."""
    n, n_cls = logits.shape
    pred = logits.argmax(axis=1)
    w_diff_norm = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=2) + EPS
    logit_diff = np.abs(logits[:, :, None] - logits[:, None, :])
    dists = logit_diff[np.arange(n), pred, :] / w_diff_norm[pred, :]
    dists[np.arange(n), pred] = 0.0
    dev = np.linalg.norm(z_std - mu_train_std, axis=1) + EPS
    return dists.sum(axis=1) / (n_cls - 1) / dev


class Subspace:
    """Shared PCA machinery for PCA RecError, Residual, ViM, and NeCo."""

    def __init__(self, h_fit: np.ndarray) -> None:
        self.mu = h_fit.mean(axis=0)
        _, s, vt = np.linalg.svd(h_fit - self.mu, full_matrices=False)
        self.vt = vt
        self.s = s

    def conf_pca_recerror(self, h: np.ndarray, dim: int) -> np.ndarray:
        c = h - self.mu
        recon = (c @ self.vt[:dim].T) @ self.vt[:dim] + self.mu
        return -np.linalg.norm(h - recon, axis=1) / (
            np.linalg.norm(h, axis=1) + EPS)

    def residual_norm(self, h: np.ndarray, dim: int) -> np.ndarray:
        c = h - self.mu
        in_span = (c @ self.vt[:dim].T) @ self.vt[:dim]
        return np.linalg.norm(c - in_span, axis=1)

    def conf_residual(self, h: np.ndarray, dim: int) -> np.ndarray:
        return -self.residual_norm(h, dim)

    def conf_vim(self, h: np.ndarray, logits: np.ndarray, dim: int,
                 alpha: float, temp: float) -> np.ndarray:
        return conf_energy(logits, temp) - alpha * self.residual_norm(h, dim)

    def vim_alpha(self, h_fit: np.ndarray, logits_fit: np.ndarray,
                  dim: int) -> float:
        res = self.residual_norm(h_fit, dim)
        return float(logits_fit.max(axis=1).mean() / (res.mean() + EPS))

    def conf_neco(self, h: np.ndarray, dim: int,
                  mls: np.ndarray | None = None) -> np.ndarray:
        num = np.linalg.norm(h @ self.vt[:dim].T, axis=1)
        ratio = num / (np.linalg.norm(h, axis=1) + EPS)
        return ratio * mls if mls is not None else ratio

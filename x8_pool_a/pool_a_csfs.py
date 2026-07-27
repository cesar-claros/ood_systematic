"""Pool A CSF implementations on cached frozen features (torch, device-agnostic).

Faithful to the paper's Appendix C definitions (see
`paper/NeurIPS_2026_original/sections/appendix/methods_projections.tex`),
adapted to a linear probe head: logits = Z @ W.T + b on standardized features
Z = (h - mu) / sd. Head-coupled scores (CTM, fDBD, GradNorm, pNML, ViM's
energy term) operate in the standardized space; feature-manifold scores
(Maha, MahaPP, NCI, PCA RecError, Residual, NeCo, NNGuide bank) operate on
raw features, matching the pipeline's use of raw penultimate activations.

All functions take and return torch tensors on the caller's device (CUDA on
the cluster's V100, CPU otherwise); Mahalanobis uses the quadratic expansion
instead of a per-class loop. Every function returns CONFIDENCE (higher =
more ID-like). `Confidence` (trained auxiliary head) and KPCA RecError are
excluded from the pilot.
"""
from __future__ import annotations

import torch

EPS = 1e-12


def train_probe(h_fit: torch.Tensor, y_fit: torch.Tensor, n_cls: int,
                seed: int = 0, steps: int = 300, lr: float = 0.5,
                weight_decay: float = 1e-4, momentum: float = 0.9) -> dict:
    """Multinomial logistic probe via momentum GD on standardized features.

    Torch port of `probes_and_descriptors.train_probe` (same hyperparameters
    and algorithm; RNG stream differs from the numpy version).
    """
    device = h_fit.device
    mu, sd = h_fit.mean(0), h_fit.std(0) + 1e-8
    z = (h_fit - mu) / sd
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(n_cls, z.shape[1], generator=gen) * 0.01).to(device)
    b = torch.zeros(n_cls, device=device)
    vel_w = torch.zeros_like(w)
    vel_b = torch.zeros_like(b)
    onehot = torch.nn.functional.one_hot(y_fit, n_cls).float()
    n = float(len(y_fit))
    for _ in range(steps):
        logits = z @ w.T + b
        p = torch.softmax(logits, dim=1)
        g = (p - onehot) / n
        gw = g.T @ z + weight_decay * w
        vel_w = momentum * vel_w - lr * gw
        vel_b = momentum * vel_b - lr * g.sum(0)
        w = w + vel_w
        b = b + vel_b
    acc = ((z @ w.T + b).argmax(1) == y_fit).float().mean().item()
    return {"W": w, "b": b, "mu": mu, "sd": sd, "acc": acc}


def fit_temperature(logits_val: torch.Tensor, y_val: torch.Tensor) -> float:
    """Temperature minimizing validation NLL over a log-spaced grid."""
    grid = torch.exp(torch.linspace(-1.3863, 2.0794, 41))
    idx = torch.arange(len(y_val), device=logits_val.device)
    best_t, best_nll = 1.0, float("inf")
    for t in grid.tolist():
        p = torch.softmax(logits_val / t, dim=1)
        nll = -torch.log(p[idx, y_val] + EPS).mean().item()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


def l2n(h: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalization (Mahalanobis++ preprocessing)."""
    return h / (h.norm(dim=1, keepdim=True) + EPS)


# ---- head-side ----

def conf_msr(p: torch.Tensor) -> torch.Tensor:
    return p.max(dim=1).values


def conf_mls(logits: torch.Tensor) -> torch.Tensor:
    return logits.max(dim=1).values


def conf_energy(logits: torch.Tensor, temp: float) -> torch.Tensor:
    return temp * torch.logsumexp(logits / temp, dim=1)


def conf_pe(p: torch.Tensor) -> torch.Tensor:
    return (p * torch.log(p + EPS)).sum(dim=1)


def conf_gen(p: torch.Tensor, gamma: float, m_top: int) -> torch.Tensor:
    ps = torch.sort(p, dim=1, descending=True).values[:, :m_top]
    return -(ps.pow(gamma) * (1.0 - ps).pow(gamma)).sum(dim=1)


def conf_ren(p: torch.Tensor, alpha: float, m_top: int) -> torch.Tensor:
    ps = torch.sort(p, dim=1, descending=True).values[:, :m_top]
    return -(1.0 / (1.0 - alpha)) * torch.log(ps.pow(alpha).sum(dim=1) + EPS)


def conf_ge(p: torch.Tensor) -> torch.Tensor:
    ps = torch.sort(p, dim=1, descending=True).values
    ranks = torch.arange(1, ps.shape[1] + 1, device=p.device, dtype=p.dtype)
    return -(ps * ranks).sum(dim=1)


def conf_pce(p: torch.Tensor) -> torch.Tensor:
    return torch.log(p.pow(2).sum(dim=1) + EPS)


def conf_gradnorm(p: torch.Tensor, z_std: torch.Tensor) -> torch.Tensor:
    """L1 norm of d KL(u || p) / dW for a linear head: ||p - u||_1 * ||Z||_1."""
    u = 1.0 / p.shape[1]
    return (p - u).abs().sum(dim=1) * z_std.abs().sum(dim=1)


class PNML:
    """pNML regret via the kernel-range projection of the paper's Appendix C."""

    def __init__(self, z_train: torch.Tensor) -> None:
        zn = z_train / (z_train.norm(dim=1, keepdim=True) + EPS)
        _, s, vt = torch.linalg.svd(zn, full_matrices=False)
        keep = s > 1e-8 * s[0]
        self.v = vt[keep].T
        self.inv_s2 = 1.0 / s[keep].pow(2)

    def conf(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        proj = z @ self.v
        h_perp = z - proj @ self.v.T
        perp_sq = h_perp.pow(2).sum(dim=1)
        quad = (proj.pow(2) * self.inv_s2).sum(dim=1)
        hg = torch.where(perp_sq > 1e-8,
                         torch.ones_like(quad), quad / (1.0 + quad))
        pk_hg = (p + EPS).pow(hg.unsqueeze(1))
        regret = torch.log((p / (p + pk_hg * (1.0 - p) + EPS)).sum(dim=1) + EPS)
        return -regret


# ---- feature-side ----

def conf_ctm(z_std: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Max cosine similarity to the classifier weight rows (paper's CTM)."""
    return (l2n(z_std) @ l2n(w).T).max(dim=1).values


class Mahalanobis:
    """Shared-covariance Mahalanobis, quadratic expansion (no class loop)."""

    def __init__(self, h_fit: torch.Tensor, y_fit: torch.Tensor, n_cls: int,
                 ridge: float = 1e-3) -> None:
        self.means = torch.stack([h_fit[y_fit == c].mean(dim=0)
                                  for c in range(n_cls)])
        centered = h_fit - self.means[y_fit]
        cov = centered.T @ centered / len(h_fit)
        cov = cov + ridge * torch.trace(cov) / cov.shape[0] * torch.eye(
            cov.shape[0], device=h_fit.device)
        self.prec = torch.linalg.inv(cov)
        self.prec_means = self.prec @ self.means.T
        self.const = (self.means * self.prec_means.T).sum(dim=1)

    def conf(self, h: torch.Tensor) -> torch.Tensor:
        rowquad = ((h @ self.prec) * h).sum(dim=1)
        d = rowquad.unsqueeze(1) - 2.0 * (h @ self.prec_means) + self.const
        return -d.min(dim=1).values


def fit_nci_alpha(h_val: torch.Tensor, logits_val: torch.Tensor,
                  resid_val, w_eff: torch.Tensor, train_mean: torch.Tensor,
                  rc_metric_fn,
                  alphas=(0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)) -> float:
    """NCI alpha minimizing validation failure-AUGRC (pipeline convention)."""
    best_alpha, best = alphas[0], float("inf")
    for alpha in alphas:
        conf = conf_nci(h_val, logits_val, w_eff, train_mean, alpha)
        augrc, _ = rc_metric_fn(conf.cpu().numpy(), resid_val)
        if augrc < best:
            best_alpha, best = alpha, augrc
    return best_alpha


def conf_nci(h: torch.Tensor, logits: torch.Tensor, w_eff: torch.Tensor,
             train_mean: torch.Tensor, alpha: float) -> torch.Tensor:
    """NCI (Liu & Qin, CVPR 2025): <w_pred, h-u>/||h-u|| + alpha*||h||_1."""
    centered = h - train_mean
    pred = logits.argmax(dim=1)
    align = (w_eff[pred] * centered).sum(dim=1) / (centered.norm(dim=1) + EPS)
    return align + alpha * h.abs().sum(dim=1)


class NNGuide:
    """Energy score modulated by confidence-scaled cosine to an ID bank."""

    def __init__(self, h_bank: torch.Tensor, s_bank: torch.Tensor,
                 k: int) -> None:
        self.bank = l2n(h_bank)
        self.s_bank = s_bank
        self.k = min(k, len(s_bank))

    def conf(self, h: torch.Tensor, s_base: torch.Tensor) -> torch.Tensor:
        sims = (l2n(h) @ self.bank.T) * self.s_bank.unsqueeze(0)
        guide = torch.topk(sims, self.k, dim=1).values.mean(dim=1)
        return s_base * guide


def conf_fdbd(z_std: torch.Tensor, logits: torch.Tensor, w: torch.Tensor,
              mu_train_std: torch.Tensor) -> torch.Tensor:
    """Mean boundary distance regularized by deviation from the ID mean."""
    n, n_cls = logits.shape
    pred = logits.argmax(dim=1)
    w_diff_norm = torch.cdist(w, w) + EPS
    logit_diff = (logits.gather(1, pred.unsqueeze(1)) - logits).abs()
    dists = logit_diff / w_diff_norm[pred]
    dists.scatter_(1, pred.unsqueeze(1), 0.0)
    dev = (z_std - mu_train_std).norm(dim=1) + EPS
    return dists.sum(dim=1) / (n_cls - 1) / dev


class Subspace:
    """Shared PCA machinery for PCA RecError, Residual, ViM, and NeCo."""

    def __init__(self, h_fit: torch.Tensor) -> None:
        self.mu = h_fit.mean(dim=0)
        _, s, vt = torch.linalg.svd(h_fit - self.mu, full_matrices=False)
        self.vt = vt
        self.s = s

    def conf_pca_recerror(self, h: torch.Tensor, dim: int) -> torch.Tensor:
        c = h - self.mu
        recon = (c @ self.vt[:dim].T) @ self.vt[:dim] + self.mu
        return -(h - recon).norm(dim=1) / (h.norm(dim=1) + EPS)

    def residual_norm(self, h: torch.Tensor, dim: int) -> torch.Tensor:
        c = h - self.mu
        return (c - (c @ self.vt[:dim].T) @ self.vt[:dim]).norm(dim=1)

    def conf_residual(self, h: torch.Tensor, dim: int) -> torch.Tensor:
        return -self.residual_norm(h, dim)

    def conf_vim(self, h: torch.Tensor, logits: torch.Tensor, dim: int,
                 alpha: float, temp: float) -> torch.Tensor:
        return conf_energy(logits, temp) - alpha * self.residual_norm(h, dim)

    def vim_alpha(self, h_fit: torch.Tensor, logits_fit: torch.Tensor,
                  dim: int) -> float:
        res = self.residual_norm(h_fit, dim)
        return float(logits_fit.max(dim=1).values.mean()
                     / (res.mean() + EPS))

    def conf_neco(self, h: torch.Tensor, dim: int,
                  mls: torch.Tensor | None = None) -> torch.Tensor:
        ratio = (h @ self.vt[:dim].T).norm(dim=1) / (h.norm(dim=1) + EPS)
        return ratio * mls if mls is not None else ratio

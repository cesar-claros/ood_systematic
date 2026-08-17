"""Exact-mean X1 predictors for head-side scores under a given head state.

Implements the general (any theta_w, any alignment profile) forms of
X1 Lemma 1 + Lemma 2/4 and Proposition 4: population score statistics are
computed by the delta method from the MEASURED population feature means
(per-class train means for ID, the OOD set's mean vector for OOD), the
head in effect (W, b), and isotropic within-class noise, then combined
into a binormal AUROC with the ID side treated as a class mixture. This
is the predictor family verified in
documentation/phase_diagram_scripts/verify2.py, ported to measured inputs.
Mean-field cos(theta_w) laws are deliberately not used (inapplicable
under leakage, X1 Theorem 2.3).

Head-level quantities (the Gram matrix W W^T, row norms, normalized rows)
are hoisted into ``HeadContext`` so the per-population work is light; build
one context per head state, never per (class, OOD set) cell.

Two noise arms share one code path via ``NoiseModel``:
- isotropic: sigma^2 I, the X1-idealized plug-in (its level error on real
  anisotropic features is the T4 misspecification diagnostic);
- empirical: the measured feature covariance (within-class Sigma_W for ID,
  per-OOD-set residual covariance for OOD). Logit variances become
  w' Sigma w exactly; the CTM variance uses directional variance for the
  alignment fluctuation and the average eigenvalue for the norm
  fluctuation (declared approximation, reduces to Prop 4 when isotropic).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class HeadContext:
    """Precomputed head-state quantities shared by all populations."""

    w: np.ndarray
    b: np.ndarray
    gram: np.ndarray
    row_norms2: np.ndarray
    w_hat: np.ndarray

    @classmethod
    def from_head(cls, w: np.ndarray, b: np.ndarray) -> HeadContext:
        w64 = w.astype(np.float64)
        return cls(
            w=w64, b=b.astype(np.float64), gram=w64 @ w64.T,
            row_norms2=(w64**2).sum(1),
            w_hat=w64 / np.linalg.norm(w64, axis=1, keepdims=True))


@dataclass
class NoiseModel:
    """Gaussian feature-noise model projected onto one head state.

    Attributes:
        logit_cov: (C, C) covariance of logits, W Sigma W^T.
        dir_var: (C,) variance along each unit head row, w_hat' Sigma w_hat.
        trace: total feature-space variance, Tr(Sigma).
        dim: feature dimension D.
    """

    logit_cov: np.ndarray
    dir_var: np.ndarray
    trace: float
    dim: int

    @classmethod
    def isotropic(cls, sigma: float, ctx: HeadContext,
                  dim: int) -> NoiseModel:
        return cls(logit_cov=sigma**2 * ctx.gram,
                   dir_var=sigma**2 * np.ones(len(ctx.w)),
                   trace=sigma**2 * dim, dim=dim)

    @classmethod
    def empirical(cls, cov: np.ndarray, ctx: HeadContext) -> NoiseModel:
        w_cov = ctx.w @ cov
        wh_cov = ctx.w_hat @ cov
        return cls(logit_cov=w_cov @ ctx.w.T,
                   dir_var=(wh_cov * ctx.w_hat).sum(1),
                   trace=float(np.trace(cov)), dim=cov.shape[0])


def population_stats(m: np.ndarray, noise: NoiseModel,
                     ctx: HeadContext) -> dict[str, tuple[float, float]]:
    """Score (mean, variance) for features h ~ N(m, Sigma).

    Returns:
        Dict score name -> (mean, variance) for MLS, Energy (as -E = LSE),
        MSR (as log MSR, AUROC-equivalent), and CTM_head.
    """
    m_vec = ctx.w @ m + ctx.b
    c_star = int(np.argmax(m_vec))
    mls = (float(m_vec[c_star]), float(noise.logit_cov[c_star, c_star]))

    p = np.exp(m_vec - m_vec[c_star])
    p /= p.sum()
    lse_mean = float(np.log(np.exp(m_vec - m_vec[c_star]).sum())
                     + m_vec[c_star])
    energy = (lse_mean, float(p @ (noise.logit_cov @ p)))

    diff = -p
    diff[c_star] += 1.0
    msr = (float(m_vec[c_star] - lse_mean),
           float(diff @ (noise.logit_cov @ diff)))

    p0_all = ctx.w_hat @ m
    m2 = float(m @ m)
    omega = m2 + noise.trace
    mean_cos = p0_all / np.sqrt(omega)
    k_star = int(np.argmax(mean_cos))
    p0 = float(p0_all[k_star])
    sigma_dir2 = float(noise.dir_var[k_star])
    sigma_avg2 = noise.trace / noise.dim
    q0 = m2 - p0**2 + noise.trace
    om = p0**2 + q0
    v_cos = (q0**2 * sigma_dir2 / om**3
             + p0**2 * (4.0 * (m2 - p0**2) * sigma_avg2
                        + 2.0 * sigma_avg2**2 * noise.dim) / (4.0 * om**3))
    ctm_head = (float(mean_cos[k_star]), float(v_cos))
    return {"MLS": mls, "Energy": energy, "MSR": msr, "CTM_head": ctm_head}


def predicted_aurocs(class_means: np.ndarray, class_freq: np.ndarray,
                     noise_id: NoiseModel, m_ood: np.ndarray,
                     noise_ood: NoiseModel,
                     ctx: HeadContext) -> dict[str, float]:
    """Binormal mixture AUROC predictions for the head-side scores.

    ID is the class mixture sum_y freq_y N(mu_y, Sigma_id); OOD is
    N(m_ood, Sigma_ood). AUROC = sum_y freq_y *
    Phi((mean_y - mean_ood) / sqrt(var_y + var_ood)) per score.
    """
    ood = population_stats(m_ood, noise_ood, ctx)
    totals = dict.fromkeys(ood, 0.0)
    for mu_y, freq in zip(class_means, class_freq):
        stats_y = population_stats(mu_y, noise_id, ctx)
        for name, (m_o, v_o) in ood.items():
            m_y, v_y = stats_y[name]
            totals[name] += freq * float(
                norm.cdf((m_y - m_o) / np.sqrt(v_y + v_o)))
    return totals


def hanley_mcneil_se(auc: float, n_id: int, n_ood: int) -> float:
    """Hanley-McNeil standard error of an AUROC estimate."""
    a = min(max(auc, 0.5), 1.0 - 1e-12)
    q1 = a / (2.0 - a)
    q2 = 2.0 * a**2 / (1.0 + a)
    var = (a * (1.0 - a) + (n_id - 1) * (q1 - a**2)
           + (n_ood - 1) * (q2 - a**2)) / (n_id * n_ood)
    return float(np.sqrt(max(var, 0.0)))

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

    ctm_head = ctm_stats(m, noise.trace, noise.dim, ctx.w_hat, noise.dir_var)
    return {"MLS": mls, "Energy": energy, "MSR": msr, "CTM_head": ctm_head}


def ctm_stats(m: np.ndarray, trace: float, dim: int,
              prototypes_hat: np.ndarray,
              dir_var: np.ndarray) -> tuple[float, float]:
    """Delta-method (mean, var) of the max-cosine score to unit prototypes.

    Proposition 4 generalized: `dir_var[k]` is the noise variance along
    prototype k, `trace`/`dim` set the norm-fluctuation scale. Used for
    CTM_head (prototypes = head rows) and CTM_mean (prototypes = class
    means, with dir_var measured along the mean directions).
    """
    p0_all = prototypes_hat @ m
    m2 = float(m @ m)
    omega = m2 + trace
    mean_cos = p0_all / np.sqrt(omega)
    k_star = int(np.argmax(mean_cos))
    p0 = float(p0_all[k_star])
    sigma_dir2 = float(dir_var[k_star])
    sigma_avg2 = trace / dim
    q0 = m2 - p0**2 + trace
    om = p0**2 + q0
    v_cos = (q0**2 * sigma_dir2 / om**3
             + p0**2 * (4.0 * (m2 - p0**2) * sigma_avg2
                        + 2.0 * sigma_avg2**2 * dim) / (4.0 * om**3))
    return (float(mean_cos[k_star]), float(v_cos))


def predicted_ctm_mean_auroc(class_means: np.ndarray, class_freq: np.ndarray,
                             cov_id: np.ndarray, m_ood: np.ndarray,
                             cov_ood: np.ndarray) -> float:
    """Binormal mixture AUROC prediction for mean-CTM (feature-side).

    Prototypes are the UNcentered class-mean directions (pipeline
    convention); directional variances are measured along them from the
    supplied covariances (pass sigma^2 * I for the isotropic arm).
    """
    mu_hat = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)
    dim = class_means.shape[1]
    dir_id = ((mu_hat @ cov_id) * mu_hat).sum(1)
    dir_ood = ((mu_hat @ cov_ood) * mu_hat).sum(1)
    tr_id, tr_ood = float(np.trace(cov_id)), float(np.trace(cov_ood))
    m_o, v_o = ctm_stats(m_ood, tr_ood, dim, mu_hat, dir_ood)
    total = 0.0
    for mu_y, freq in zip(class_means, class_freq):
        m_y, v_y = ctm_stats(mu_y, tr_id, dim, mu_hat, dir_id)
        total += freq * float(norm.cdf((m_y - m_o) / np.sqrt(v_y + v_o)))
    return total


def predicted_maha_auroc(class_means: np.ndarray, precision: np.ndarray,
                         cov_id: np.ndarray, m_ood: np.ndarray,
                         cov_ood: np.ndarray) -> float:
    """Binormal AUROC prediction for shared-covariance Mahalanobis.

    Gaussian quadratic-form moments with nearest-prototype (argmin)
    stability: for h ~ N(mu, Sigma), (h-mu)' P (h-mu) has mean tr(P Sigma)
    and variance 2 tr(P Sigma P Sigma); the OOD population adds the
    displacement d^2 to the mean and 4 delta' P Sigma P delta to the
    variance at its nearest prototype. Declared approximation: the min
    over non-nearest prototypes is ignored (X1 Prop 5 min-statistic
    caveat). Pass sigma^2 * I covariances for the isotropic arm.
    """
    p_cov_id = precision @ cov_id
    m_id = float(np.trace(p_cov_id))
    v_id = 2.0 * float(np.trace(p_cov_id @ p_cov_id))

    diffs = m_ood - class_means
    d2 = ((diffs @ precision) * diffs).sum(1)
    c_star = int(np.argmin(d2))
    delta = diffs[c_star]
    p_cov_ood = precision @ cov_ood
    m_o = float(d2[c_star]) + float(np.trace(p_cov_ood))
    v_o = (2.0 * float(np.trace(p_cov_ood @ p_cov_ood))
           + 4.0 * float(delta @ precision @ cov_ood @ precision @ delta))
    # Score is the NEGATED distance (higher = more ID).
    return float(norm.cdf((m_o - m_id) / np.sqrt(v_id + v_o)))


def _psd_sqrt(cov: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigh (negative eigenvalues clipped)."""
    sym = (cov + cov.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def predicted_maha_auroc_min(class_means: np.ndarray,
                             class_freq: np.ndarray,
                             precision: np.ndarray, cov_id: np.ndarray,
                             m_ood: np.ndarray, cov_ood: np.ndarray,
                             n_samples: int = 4000, seed: int = 0,
                             diagnostics: bool = False):
    """Min-statistic repair of `predicted_maha_auroc` (B-axis protocol 7).

    Identical Gaussian population model (ID mixture sum_y freq_y
    N(mu_y, Sigma_id), OOD N(m_ood, Sigma_ood)) but the score is the TRUE
    negated min over all class prototypes, evaluated by seeded Monte
    Carlo, removing the declared argmin approximation (X1 Prop 5 caveat).
    That approximation ignores the min-benefit on both populations and is
    the prime suspect for the amplitude bias in high-var_collapse regimes
    where nearest-prototype switching is rampant (Pilot 1 forensics:
    A2 over-predicted 2.6x). No fitted parameters; the original closed
    form stays untouched as the frozen registered operator.

    Returns the AUROC, or ``(auroc, diag)`` with ``diagnostics=True``
    where diag holds prototype-switching rates and score moments (the
    audit's section 5.4 measurables).
    """
    rng = np.random.default_rng(seed)
    n_classes, dim = class_means.shape
    labels = rng.choice(n_classes, size=n_samples,
                        p=class_freq / class_freq.sum())
    h_id = (class_means[labels]
            + rng.standard_normal((n_samples, dim)) @ _psd_sqrt(cov_id).T)
    h_ood = (m_ood[None, :]
             + rng.standard_normal((n_samples, dim)) @ _psd_sqrt(cov_ood).T)

    m_quad = np.einsum("cd,dk,ck->c", class_means, precision, class_means,
                       optimize=True)

    def d2_matrix(h: np.ndarray) -> np.ndarray:
        h_prec = h @ precision
        return ((h_prec * h).sum(1)[:, None]
                - 2.0 * h_prec @ class_means.T + m_quad[None, :])

    d2_id = d2_matrix(h_id)
    d2_ood = d2_matrix(h_ood)
    s_id = -d2_id.min(1)
    s_ood = -d2_ood.min(1)

    from scipy.stats import rankdata
    ranks = rankdata(np.concatenate([s_id, s_ood]))
    auc = float((ranks[:n_samples].sum()
                 - n_samples * (n_samples + 1) / 2.0) / (n_samples ** 2))
    if not diagnostics:
        return auc
    diffs = m_ood - class_means
    c_star = int(np.argmin(((diffs @ precision) * diffs).sum(1)))
    diag = {
        "id_switch_rate": float((d2_id.argmin(1) != labels).mean()),
        "ood_nearest_share": float((d2_ood.argmin(1) == c_star).mean()),
        "id_score_mean": float(s_id.mean()),
        "id_score_var": float(s_id.var(ddof=1)),
        "ood_score_mean": float(s_ood.mean()),
        "ood_score_var": float(s_ood.var(ddof=1)),
    }
    return auc, diag


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

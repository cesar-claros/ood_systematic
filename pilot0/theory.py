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
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _lse_stats(m_vec: np.ndarray, w: np.ndarray,
               sigma: float) -> tuple[float, float, np.ndarray]:
    """Delta-method mean and variance of logsumexp(g), g ~ N(m_vec, s^2 WW^T)."""
    p = np.exp(m_vec - m_vec.max())
    p /= p.sum()
    mean = float(np.log(np.exp(m_vec - m_vec.max()).sum()) + m_vec.max())
    cov_g = sigma**2 * (w @ w.T)
    return mean, float(p @ cov_g @ p), p


def population_stats(m: np.ndarray, sigma: float, w: np.ndarray,
                     b: np.ndarray) -> dict[str, tuple[float, float]]:
    """Score (mean, variance) for features h ~ N(m, sigma^2 I).

    Args:
        m: (D,) population feature mean (uncentered).
        sigma: isotropic noise scale.
        w: (C, D) head rows in effect.
        b: (C,) bias.

    Returns:
        Dict score name -> (mean, variance) for MLS, Energy (as -E = LSE),
        MSR (as log MSR, AUROC-equivalent), and CTM_head.
    """
    m_vec = w @ m + b
    c_star = int(np.argmax(m_vec))
    row_norms2 = (w**2).sum(1)
    mls = (float(m_vec[c_star]), float(sigma**2 * row_norms2[c_star]))

    lse_mean, lse_var, p = _lse_stats(m_vec, w, sigma)
    energy = (lse_mean, lse_var)

    e_star = np.zeros(len(m_vec))
    e_star[c_star] = 1.0
    diff = e_star - p
    cov_g = sigma**2 * (w @ w.T)
    msr = (float(m_vec[c_star] - lse_mean), float(diff @ cov_g @ diff))

    d = len(m)
    w_hat = w / np.linalg.norm(w, axis=1, keepdims=True)
    p0_all = w_hat @ m
    omega = float(m @ m) + sigma**2 * d
    mean_cos = p0_all / np.sqrt(omega)
    k_star = int(np.argmax(mean_cos))
    p0 = float(p0_all[k_star])
    m2 = float(m @ m)
    q0 = m2 - p0**2 + sigma**2 * d
    om = p0**2 + q0
    v_cos = (q0**2 * sigma**2 / om**3
             + p0**2 * (4.0 * (m2 - p0**2) * sigma**2
                        + 2.0 * sigma**4 * d) / (4.0 * om**3))
    ctm_head = (float(mean_cos[k_star]), float(v_cos))
    return {"MLS": mls, "Energy": energy, "MSR": msr, "CTM_head": ctm_head}


def predicted_aurocs(class_means: np.ndarray, class_freq: np.ndarray,
                     sigma: float, m_ood: np.ndarray, sigma_ood: float,
                     w: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Binormal mixture AUROC predictions for the head-side scores.

    ID is the class mixture sum_y freq_y N(mu_y, sigma^2 I); OOD is
    N(m_ood, sigma_ood^2 I). AUROC = sum_y freq_y *
    Phi((mean_y - mean_ood) / sqrt(var_y + var_ood)) per score.
    """
    ood = population_stats(m_ood, sigma_ood, w, b)
    totals = dict.fromkeys(ood, 0.0)
    for mu_y, freq in zip(class_means, class_freq):
        stats_y = population_stats(mu_y, sigma, w, b)
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

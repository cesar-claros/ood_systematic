"""H-estimator definitions for the X1 OOD configuration coordinates.

These definitions FREEZE after Pilot 0 validates them against synthetic
injections with known coordinates (plan section 8, amendment R7d). The
primary convention is population-mean-direction: in the X1 model
h' = gamma*R*u + xi' with zero-mean noise, the mean of centered OOD
features identifies gamma*R*u exactly, so

    m_o    = mean of centered OOD features
    gamma  = ||m_o|| / R
    a_c    = (m_o / ||m_o||) . mu_hat_c,   a = max_c a_c
    rho    = sigma_o / sigma,  sigma_o^2 = mean ||h' - m_o||^2 / D
    w_perp = ||P_perp m_o||^2 / ||m_o||^2 (energy outside the mean span)
    top2_gap = a_(1) - a_(2) over the alignment profile

Real OOD sets are mixtures over directions; the mean-direction convention
collapses that heterogeneity by design. Per-sample summaries are recorded
as secondary diagnostics (`per_sample_*` keys) but do not enter the
frozen coordinates.
"""
from __future__ import annotations

import numpy as np

from pilot0.geometry import FeatureModel


def estimate_ood_coords(h_ood: np.ndarray,
                        model: FeatureModel) -> dict[str, float]:
    """Estimate (gamma, a, rho, w_perp, top2_gap) for one OOD set.

    Args:
        h_ood: (N, D) uncentered OOD activations.
        model: fitted train feature model.

    Returns:
        Dict of frozen coordinates plus per-sample secondary diagnostics.
    """
    centered = model.center(h_ood.astype(np.float64))
    m_o = centered.mean(0)
    m_norm = float(np.linalg.norm(m_o))
    u_hat = m_o / m_norm if m_norm > 0 else m_o
    mu_hat = model.class_means / model.radii[:, None]
    align = mu_hat @ u_hat
    order = np.sort(align)[::-1]
    resid = centered - m_o
    sigma_o = float(np.sqrt((resid**2).sum(1).mean() / centered.shape[1]))
    in_span = model.span_basis @ (model.span_basis.T @ m_o)
    w_perp = float(((m_o - in_span) ** 2).sum() / m_norm**2) if m_norm else 0.0

    sample_norms = np.linalg.norm(centered, axis=1)
    unit = centered / np.clip(sample_norms, 1e-12, None)[:, None]
    per_sample_a = (unit @ mu_hat.T).max(1)
    id_typical_norm = float(np.sqrt(
        model.radius**2 + model.sigma_iso**2 * centered.shape[1]))
    return {
        "gamma": m_norm / model.radius,
        "a": float(order[0]),
        "aligned_class": int(np.argmax(align)),
        "top2_gap": float(order[0] - order[1]),
        "rho": sigma_o / model.sigma_iso,
        "w_perp": w_perp,
        "n_eff_aligned": float((align.clip(0) ** 2).sum()
                               / max(order[0] ** 2, 1e-12)),
        "per_sample_a_mean": float(per_sample_a.mean()),
        "per_sample_norm_ratio_mean": float(
            (sample_norms / id_typical_norm).mean()),
    }


def inject_synthetic_ood(model: FeatureModel, gamma: float, a: float,
                         rho: float, w_perp: float, n: int,
                         rng: np.random.Generator,
                         aligned_class: int = 0) -> np.ndarray:
    """Draw an OOD population with known coordinates in the measured space.

    Constructs u = a * mu_hat_k + b_span * v_span + b_perp * v_perp with
    the in-span/out-of-span split set by `w_perp`, then returns
    UNcentered features global_mean + gamma*R*u + noise so the estimator
    sees exactly what extraction produces.
    """
    d = model.class_means.shape[1]
    mu_hat = model.class_means[aligned_class] / model.radii[aligned_class]
    v = rng.standard_normal(d)
    v -= model.span_basis @ (model.span_basis.T @ v)
    v /= np.linalg.norm(v)
    v_span = model.span_basis @ rng.standard_normal(model.span_basis.shape[1])
    v_span -= (v_span @ mu_hat) * mu_hat
    v_span /= np.linalg.norm(v_span)
    residual = max(1.0 - a**2, 1e-12)
    b_perp = np.sqrt(residual * w_perp)
    b_span = np.sqrt(residual * (1.0 - w_perp))
    u = a * mu_hat + b_span * v_span + b_perp * v
    u /= np.linalg.norm(u)
    noise = rng.standard_normal((n, d)) * (rho * model.sigma_iso)
    return model.global_mean + gamma * model.radius * u + noise


def validate_estimators(model: FeatureModel, rng: np.random.Generator,
                        n: int = 4000) -> list[dict[str, float]]:
    """Recovery test: inject known coordinates, report estimates alongside.

    Returns one record per injected configuration with `true_*` and
    estimated values for gamma, a, rho; recovery error drives the
    estimator-freeze decision in the report.
    """
    configs = [(0.8, 0.9, 1.0, 0.0), (1.0, 0.7, 1.0, 0.2),
               (1.25, 0.5, 1.2, 0.5), (1.5, 0.3, 0.8, 0.8)]
    records = []
    for gamma, a, rho, w_perp in configs:
        # w_perp here is the fraction of the *non-aligned* energy placed
        # out of span; the implied coordinate is that fraction of 1-a^2
        # renormalized by the total direction energy.
        h = inject_synthetic_ood(model, gamma, a, rho, w_perp, n, rng)
        est = estimate_ood_coords(h, model)
        implied_w_perp = (1.0 - a**2) * w_perp
        records.append({
            "true_gamma": gamma, "true_a": a, "true_rho": rho,
            "true_w_perp": implied_w_perp,
            "est_gamma": est["gamma"], "est_a": est["a"],
            "est_rho": est["rho"], "est_w_perp": est["w_perp"],
        })
    return records

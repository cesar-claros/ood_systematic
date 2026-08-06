"""NC metrics for x9: the 8 Papyan metrics, verbatim from
`x8_pool_a/pool_a_analysis.py::papyan_nc` (numpy; formulas mirrored from
src/neural_collapse.py), kept as a standalone copy so x9 never imports the
pool_a module chain. The X8 weak-collapse descriptors come from
`x8_pool_a/probes_and_descriptors.py` (numpy-only), imported directly.
"""
from __future__ import annotations

import numpy as np


def papyan_nc(h_fit: np.ndarray, y_fit: np.ndarray, w_eff: np.ndarray,
              n_cls: int) -> dict[str, float]:
    gmean = h_fit.mean(axis=0)
    means = np.stack([h_fit[y_fit == c].mean(axis=0) for c in range(n_cls)])
    m_cent = means - gmean
    sigma_b = m_cent.T @ m_cent / n_cls
    centered = h_fit - means[y_fit]
    sigma_w = centered.T @ centered / (len(h_fit) * n_cls)
    var_collapse = float(
        np.trace(sigma_w @ np.linalg.pinv(sigma_b, rcond=1e-6)) / n_cls)

    def cosines(a: np.ndarray) -> np.ndarray:
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        c = an @ an.T
        return c[~np.eye(n_cls, dtype=bool)]

    def eq_stats(a: np.ndarray) -> tuple[float, float, float]:
        off = cosines(a)
        norms = np.linalg.norm(a, axis=1)
        return (float(off.std(ddof=1)),
                float(np.abs(off + 1.0 / (n_cls - 1)).mean()),
                float(norms.std(ddof=1) / norms.mean()))

    eq_uc, max_uc, eqn_uc = eq_stats(m_cent)
    eq_wc, max_wc, eqn_wc = eq_stats(w_eff)
    m_t = m_cent / np.linalg.norm(m_cent)
    w_t = w_eff / np.linalg.norm(w_eff)
    return {"var_collapse": var_collapse,
            "equiangular_uc": eq_uc, "equiangular_wc": eq_wc,
            "equinorm_uc": eqn_uc, "equinorm_wc": eqn_wc,
            "max_equiangular_uc": max_uc, "max_equiangular_wc": max_wc,
            "self_duality": float(((w_t - m_t) ** 2).sum())}

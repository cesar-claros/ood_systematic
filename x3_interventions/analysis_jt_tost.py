"""Pre-registered X3 analysis: manipulation check, JT trend, TOST flatness.

Input: long-format results with columns (axis, lam, seed, nc_target_measured,
gap) where gap = AUGRC_A - AUGRC_B per the design's score pairs. Pure
numpy/scipy; self-test with synthetic dial data at bottom.
"""
import numpy as np
from scipy.stats import norm, spearmanr


def manipulation_check(lam, nc_measured, rho_min=0.8):
    rho = spearmanr(lam, nc_measured).statistic
    return {"spearman": float(rho), "gate_passed": abs(rho) >= rho_min}


def jt_pvalue(groups, direction="decreasing"):
    """Jonckheere-Terpstra ordered-trend test, normal approximation."""
    t = sum(float((gj[:, None] > gi[None, :]).sum())
            + 0.5 * float((gj[:, None] == gi[None, :]).sum())
            for i, gi in enumerate(groups) for gj in groups[i + 1:])
    ns = np.array([len(g) for g in groups])
    n = ns.sum()
    mu = (n * n - (ns * ns).sum()) / 4
    var = (n * n * (2 * n + 3) - (ns * ns * (2 * ns + 3)).sum()) / 72
    z = (t - mu) / np.sqrt(var)
    return float(norm.cdf(z) if direction == "decreasing" else norm.sf(z))


def tost_flat(groups, bound_per_step):
    """TOST equivalence for predicted-flat cells: |slope| < bound per dial step."""
    means = np.array([g.mean() for g in groups])
    ses = np.array([g.std(ddof=1) / np.sqrt(len(g)) for g in groups])
    steps = np.diff(means)
    se_steps = np.sqrt(ses[:-1] ** 2 + ses[1:] ** 2)
    p_lo = norm.sf((steps + bound_per_step) / se_steps)
    p_hi = norm.cdf((steps - bound_per_step) / se_steps)
    return float(np.max(np.maximum(p_lo, p_hi)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dials = np.arange(5)
    trend = [rng.normal(2.0 - 0.5 * d, 0.4, 3) for d in dials]
    flat = [rng.normal(1.0, 0.05, 3) for _ in dials]
    lam = np.repeat(dials, 3)
    ncm = -0.1 * lam + 0.01 * rng.standard_normal(15)
    print("manipulation:", manipulation_check(lam, ncm))
    print(f"JT on trending gap: p={jt_pvalue(trend):.4f} (expect small)")
    print(f"JT on flat gap:     p={jt_pvalue(flat):.4f} (expect large)")
    print(f"TOST flat cell:     p={tost_flat(flat, bound_per_step=0.2):.4f} (expect small)")
    print(f"TOST trending cell: p={tost_flat(trend, bound_per_step=0.2):.4f} (expect large)")

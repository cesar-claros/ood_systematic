"""Score-aligned measured-Gaussian diagnostics (plan v3, carried v2
sections 6.4 and 6.5): G0/G1-MC and A0/A1-TAYLOR.

Given actual head (W, b), actual raw CTM prototypes P (uncentered class
means, unit-normalized), an ID model {class means (uncentered), Sigma_W,
class probabilities}, and an OOD model {one Gaussian (G0) or a component
mixture with a shared residual covariance (G1)}:
- MC: 20 independent batches of n ID and n OOD draws per cell from
  streams derived from master 2201, cell id, batch, and role; Energy and
  raw CTM scored on the draws; batch AUROCs averaged with MC standard
  errors; paired Energy-minus-CTM differences from paired batches.
- TAYLOR: per component, quadratic-surrogate moments with the exact
  Energy and fixed-branch CTM gradients and Hessians; binormal mixture
  AUROC with the zero-variance half-credit rule; CTM undefined at an
  exact branch tie or a vanishing norm (AnalyticUndefined).
Covariances are symmetrized; negative eigenvalues within
1e-10 max(lambda_max, 1e-300) are clipped, larger violations raise.

Usage (from code/): python rn18_handoff_replication/theory/gaussian_diagnostics.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_ROOT))

from pilot0.scores import auroc

MASTER = 2201


class AnalyticUndefined(ValueError):
    pass


def psd_sqrt(cov: np.ndarray) -> np.ndarray:
    cov = (cov + cov.T) / 2
    lam, V = np.linalg.eigh(cov)
    floor = -1e-10 * max(float(lam.max()), 1e-300)
    if lam.min() < floor:
        raise ValueError(f"covariance not PSD beyond tolerance: min eig {lam.min()}")
    lam = np.clip(lam, 0.0, None)
    return (V * np.sqrt(lam)) @ V.T


def energy_grad_hess(m, W, b):
    z = W @ m + b
    z = z - z.max()
    p = np.exp(z); p /= p.sum()
    return W.T @ p, W.T @ (np.diag(p) - np.outer(p, p)) @ W


def energy_score(h, W, b):
    z = h @ W.T + b
    zm = z.max(axis=-1, keepdims=True)
    return (zm + np.log(np.exp(z - zm).sum(axis=-1, keepdims=True)))[..., 0]


def ctm_score(h, P):
    hn = h / np.clip(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12, None)
    return (hn @ P.T).max(axis=-1)


def ctm_grad_hess(m, P, floor_scale: float):
    r = float(np.linalg.norm(m))
    if r <= 1e-12 * max(floor_scale, 1e-300):
        raise AnalyticUndefined("vanishing mean norm")
    al = P @ m / r
    k = int(np.argmax(al))
    srt = np.sort(al)
    if srt[-1] - srt[-2] <= 1e-12:
        raise AnalyticUndefined("exact branch tie")
    q = P[k]
    a_m = float(q @ m)
    grad = q / r - a_m * m / r ** 3
    H = -(np.outer(q, m) + np.outer(m, q) + a_m * np.eye(len(m))) / r ** 3 \
        + 3 * a_m * np.outer(m, m) / r ** 5
    return grad, H, k


def taylor_moments(f_m: float, grad, H, S):
    """Exact moments of the quadratic surrogate under N(m, S S)."""
    Sg = S @ grad
    SHS = S @ H @ S
    return float(f_m + 0.5 * np.trace(SHS)), float(Sg @ Sg + 0.5 * (SHS * SHS).sum())


def binormal_mixture_auroc(id_moments, id_w, ood_moments, ood_w) -> float:
    out = 0.0
    for (mc, vc), pc in zip(id_moments, id_w):
        for (mk, vk), wk in zip(ood_moments, ood_w):
            v = vc + vk
            if v <= 0:
                phi = 1.0 if mc > mk else 0.5 if mc == mk else 0.0
            else:
                phi = float(norm.cdf((mc - mk) / np.sqrt(v)))
            out += pc * wk * phi
    return float(out)


def taylor_aurocs(id_means, S_id, id_w, ood_comps, S_ood, W, b, P, floor_scale) -> dict:
    """A0/A1-TAYLOR: ood_comps = list of (mean, weight). Returns per-score
    AUROC or AnalyticUndefined for CTM at a tie / vanishing norm."""
    out = {}
    # Energy
    idm = []
    for m in id_means:
        g, H = energy_grad_hess(m, W, b)
        idm.append(taylor_moments(energy_score(m[None], W, b)[0], g, H, S_id))
    oom = []
    for m, _ in ood_comps:
        g, H = energy_grad_hess(m, W, b)
        oom.append(taylor_moments(energy_score(m[None], W, b)[0], g, H, S_ood))
    out["Energy"] = binormal_mixture_auroc(idm, id_w, oom, [w for _, w in ood_comps])
    try:
        idm, oom = [], []
        for m in id_means:
            g, H, _ = ctm_grad_hess(m, P, floor_scale)
            idm.append(taylor_moments(ctm_score(m[None], P)[0], g, H, S_id))
        for m, _ in ood_comps:
            g, H, _ = ctm_grad_hess(m, P, floor_scale)
            oom.append(taylor_moments(ctm_score(m[None], P)[0], g, H, S_ood))
        out["CTM"] = binormal_mixture_auroc(idm, id_w, oom, [w for _, w in ood_comps])
    except AnalyticUndefined as e:
        out["CTM"] = None
        out["CTM_undefined"] = str(e)
    return out


def mc_aurocs(id_means, S_id, id_w, ood_comps, S_ood, W, b, P, cell_id: int,
              n_batches: int = 20, n: int = 4096) -> dict:
    """G0/G1-MC with paired Energy/CTM draws and paired batch differences."""
    C, D = len(id_means), len(id_means[0])
    ood_w = np.array([w for _, w in ood_comps])
    res = {"Energy": [], "CTM": []}
    for bt in range(n_batches):
        ss = np.random.SeedSequence([MASTER, cell_id, bt])
        r_cls, r_id, r_ood, r_comp = [np.random.default_rng(s) for s in ss.spawn(4)]
        yc = r_cls.choice(C, size=n, p=np.asarray(id_w))
        h_id = np.asarray(id_means)[yc] + r_id.standard_normal((n, D)) @ S_id
        kc = r_comp.choice(len(ood_comps), size=n, p=ood_w / ood_w.sum())
        h_ood = np.asarray([m for m, _ in ood_comps])[kc] + r_ood.standard_normal((n, D)) @ S_ood
        for name, fn in (("Energy", lambda h: energy_score(h, W, b)),
                         ("CTM", lambda h: ctm_score(h, P))):
            res[name].append(auroc(fn(h_id), fn(h_ood)))
    e, c = np.array(res["Energy"]), np.array(res["CTM"])
    d = e - c
    return {"Energy": float(e.mean()), "Energy_se": float(e.std(ddof=1) / np.sqrt(n_batches)),
            "CTM": float(c.mean()), "CTM_se": float(c.std(ddof=1) / np.sqrt(n_batches)),
            "diff_E_minus_C": float(d.mean()), "diff_se": float(d.std(ddof=1) / np.sqrt(n_batches)),
            "resolved": bool(abs(d.mean()) >= 3 * d.std(ddof=1) / np.sqrt(n_batches))}


def self_test() -> None:
    rng = np.random.default_rng(5)
    C, D = 6, 12
    W = rng.standard_normal((C, D)); b = rng.standard_normal(C)
    P = rng.standard_normal((C, D)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    m = rng.standard_normal(D) * 3
    # gradients / Hessians vs finite differences
    for fn, gh in ((lambda x: energy_score(x[None], W, b)[0], lambda x: energy_grad_hess(x, W, b)),
                   (lambda x: ctm_score(x[None], P)[0], lambda x: ctm_grad_hess(x, P, 1.0)[:2])):
        g, H = gh(m)
        eps = 1e-5
        gn = np.array([(fn(m + eps * np.eye(D)[i]) - fn(m - eps * np.eye(D)[i])) / (2 * eps) for i in range(D)])
        assert np.allclose(g, gn, rtol=1e-4, atol=1e-6), np.abs(g - gn).max()
        Hn = np.array([[(fn(m + eps * (np.eye(D)[i] + np.eye(D)[j])) - fn(m + eps * (np.eye(D)[i] - np.eye(D)[j]))
                         - fn(m - eps * (np.eye(D)[i] - np.eye(D)[j])) + fn(m - eps * (np.eye(D)[i] + np.eye(D)[j])))
                        / (4 * eps * eps) for j in range(D)] for i in range(D)])
        assert np.allclose(H, Hn, rtol=1e-3, atol=1e-4), np.abs(H - Hn).max()
    # Taylor moments vs MC on a small-noise Gaussian (surrogate close to the score)
    S = psd_sqrt(0.01 * np.eye(D))
    g, H = energy_grad_hess(m, W, b)
    mt, vt = taylor_moments(energy_score(m[None], W, b)[0], g, H, S)
    draws = energy_score(m + rng.standard_normal((200000, D)) @ S, W, b)
    assert abs(draws.mean() - mt) < 3e-3 and abs(draws.var() / vt - 1) < 0.1, (draws.mean(), mt, draws.var(), vt)
    # determinism + resolved flag + undefined branch
    id_means = [m + rng.standard_normal(D) for _ in range(C)]
    S_id = psd_sqrt(np.eye(D)); S_ood = psd_sqrt(2 * np.eye(D))
    comps = [(m + 4 * rng.standard_normal(D), 0.7), (m - 4 * rng.standard_normal(D), 0.3)]
    r1 = mc_aurocs(id_means, S_id, [1 / C] * C, comps, S_ood, W, b, P, cell_id=7, n_batches=4, n=512)
    r2 = mc_aurocs(id_means, S_id, [1 / C] * C, comps, S_ood, W, b, P, cell_id=7, n_batches=4, n=512)
    assert r1 == r2, "MC streams not deterministic"
    t = taylor_aurocs(id_means, S_id, [1 / C] * C, comps, S_ood, W, b, P, floor_scale=1.0)
    assert 0 <= t["Energy"] <= 1 and (t["CTM"] is None or 0 <= t["CTM"] <= 1)
    try:
        ctm_grad_hess(np.zeros(D), P, 1.0); raise AssertionError("no undefined at zero norm")
    except AnalyticUndefined:
        pass
    print("[gauss] self-test PASS: gradients/Hessians vs finite differences, Taylor vs MC "
          f"(mean {draws.mean():.4f} vs {mt:.4f}), deterministic MC streams, undefined branch raised")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    if ap.parse_args().self_test:
        self_test()

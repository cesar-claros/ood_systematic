"""ETF-EXACT-v2: the literal constructor of plan v3 (carried v2 section
6.3) with exact population measurements of every requested coordinate.

Model: C >= 2 classes with equal priors, equal radii R = s (sigma = 1),
zero bias, an exact centered simplex u_c embedded in R^D by a
deterministic orthonormal basis, an explicit orthonormal complement of
the rank-(C-1) mean span (from the SVD null space, never by removing one
class direction), ID | c ~ N(R u_c, I), OOD ~ N(gamma R u_O, rho^2 I)
with u_O = a u_1 + sqrt(1 - a^2) v, v a unit complement vector, and the
head w_c = t (cos theta u_c + sin theta v_c) with t = L_par / (R cos
theta) and v_c orthonormal complement vectors. Requires D >= 2C + 1 so
that the C head complements and the OOD complement are mutually
orthogonal (declared constraint). Infeasible requests raise
InfeasibleRequest and are never clamped.

Population identities checked by measure(): centering, retained rank
C-1, pairwise cosines -1/(C-1), NC1 = (C-1)^2 / (C R^2), s_dict = R,
maximum alignment = a with every other class at -a/(C-1), realized gamma,
rho, mean target logit L_par, base scale t R, Frobenius self-duality
2(1 - cos theta) (so theta = arccos(1 - sd/2)), head-complement fraction
sin^2 theta, class priors 1/C.

Usage (from code/): python rn18_handoff_replication/theory/etf_constructor.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_ROOT))

REL, ABS = 1e-8, 1e-10


class InfeasibleRequest(ValueError):
    pass


def simplex(c: int) -> np.ndarray:
    """(C, C) exact centered simplex rows, unit norm, pairwise -1/(C-1)."""
    s = (np.eye(c) - 1.0 / c) * np.sqrt(c / (c - 1))
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def construct(C: int, D: int, R: float, a: float, gamma: float, rho: float,
              theta: float, L_par: float, seed: int = 0) -> dict:
    if C < 2 or D < 2 * C + 1:
        raise InfeasibleRequest(f"need C >= 2 and D >= 2C + 1, got C={C}, D={D}")
    if R <= 0 or L_par <= 0 or rho < 0 or gamma <= 0:
        raise InfeasibleRequest("R, L_par, gamma must be positive; rho >= 0")
    if not (0.0 <= a <= 1.0):
        raise InfeasibleRequest(f"a must lie in [0, 1]; negative alignment is "
                                f"incompatible with a centered simplex (a={a})")
    if not (0.0 <= theta < np.pi / 2):
        raise InfeasibleRequest(f"theta must lie in [0, pi/2), got {theta}")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))       # full orthonormal basis
    U = simplex(C) @ Q[:, :C].T                             # (C, D) unit directions
    # complement of the rank-(C-1) mean span via the SVD null space
    _, sv, Vt = np.linalg.svd(U, full_matrices=True)
    rank = int((sv > sv.max() * 1e-12).sum())
    assert rank == C - 1, rank
    comp = Vt[rank:]                                        # (D - C + 1, D) orthonormal
    v_head = comp[:C]                                       # C head complements
    v_ood = comp[C]                                         # OOD complement
    means = R * U
    t = L_par / (R * np.cos(theta))
    W = t * (np.cos(theta) * U + np.sin(theta) * v_head)
    u_o = a * U[0] + np.sqrt(1.0 - a * a) * v_ood
    m_ood = gamma * R * u_o
    return {"C": C, "D": D, "R": R, "a": a, "gamma": gamma, "rho": rho,
            "theta": theta, "L_par": L_par, "L_base": t * R, "t": t,
            "U": U, "means": means, "W": W, "b": np.zeros(C),
            "cov_id": np.eye(D), "cov_ood": rho ** 2 * np.eye(D),
            "m_ood": m_ood, "class_freq": np.full(C, 1.0 / C),
            "complement": comp}


def measure(m: dict) -> dict:
    """Independent population measurements (no sampling) of the model."""
    means, W, C = m["means"], m["W"], m["C"]
    gmean = means.mean(0)
    Mc = means - gmean
    radii = np.linalg.norm(Mc, axis=1)
    R = float(radii.mean())
    sv = np.linalg.svd(Mc, compute_uv=False)
    rank = int((sv > sv.max() * 1e-12).sum())
    Uhat = Mc / radii[:, None]
    cos = Uhat @ Uhat.T
    off = cos[~np.eye(C, dtype=bool)]
    sigma_b = Mc.T @ Mc / C
    sigma_b = (sigma_b + sigma_b.T) / 2
    nc1 = float(np.trace(m["cov_id"] @ np.linalg.pinv(sigma_b, rcond=1e-6,
                                                       hermitian=True)) / C)
    s_dict = float((C - 1) / np.sqrt(C * nc1))
    sigma_iso = float(np.sqrt(np.trace(m["cov_id"]) / m["D"]))
    mo = m["m_ood"]
    align = Uhat @ (mo / np.linalg.norm(mo))
    logit = float(np.mean(np.einsum("cd,cd->c", W, Mc) + m["b"]))
    sd = float(np.sum((W / np.linalg.norm(W) - Mc / np.linalg.norm(Mc)) ** 2))
    Pspan = Uhat.T @ np.linalg.pinv(Uhat @ Uhat.T) @ Uhat
    resid = W - W @ Pspan
    head_comp_frac = float((resid ** 2).sum() / (W ** 2).sum())
    return {"centered": float(np.abs(gmean).max()), "rank": rank,
            "pairwise_cos_max_dev": float(np.abs(off + 1.0 / (C - 1)).max()),
            "R": R, "radius_cv": float(radii.std() / radii.mean()),
            "nc1": nc1, "s_dict": s_dict, "R_over_sigma": R / sigma_iso,
            "a_max": float(align.max()), "a_other_max": float(np.sort(align)[-2]),
            "gamma": float(np.linalg.norm(mo) / R),
            "rho": float(np.sqrt(np.trace(m["cov_ood"]) / m["D"]) / sigma_iso),
            "logit_scale": logit, "L_base": float(np.linalg.norm(W, axis=1).mean() * R),
            "self_duality": sd, "theta_from_sd": float(np.arccos(1 - sd / 2)),
            "head_complement_fraction": head_comp_frac,
            "class_freq_max_dev": float(np.abs(m["class_freq"] - 1 / C).max())}


def check(m: dict) -> dict:
    """Assert every identity at REL/ABS tolerance; return the measurements."""
    x = measure(m)
    C, R = m["C"], m["R"]

    def close(v, ref):
        return abs(v - ref) <= max(ABS, REL * abs(ref))
    assert x["centered"] <= ABS, x["centered"]
    assert x["rank"] == C - 1
    assert x["pairwise_cos_max_dev"] <= 1e-8
    assert close(x["R"], R) and x["radius_cv"] <= ABS
    assert close(x["nc1"], (C - 1) ** 2 / (C * R * R)), (x["nc1"], (C - 1) ** 2 / (C * R * R))
    assert close(x["s_dict"], R) and close(x["R_over_sigma"], R)
    assert close(x["a_max"], m["a"]) and close(x["a_other_max"], -m["a"] / (C - 1))
    assert close(x["gamma"], m["gamma"]) and close(x["rho"], m["rho"])
    assert close(x["logit_scale"], m["L_par"]) and close(x["L_base"], m["L_base"])
    assert close(x["self_duality"], 2 * (1 - np.cos(m["theta"])))
    assert close(x["theta_from_sd"], m["theta"]) or m["theta"] == 0
    assert close(x["head_complement_fraction"], np.sin(m["theta"]) ** 2)
    assert x["class_freq_max_dev"] <= ABS
    return x


def self_test() -> None:
    grid = [(10, 64, 5.0, 0.9, 0.8, 1.0, 0.0, 10.0), (10, 64, 24.0, 0.3, 0.5, 2.0, 0.1, 10.0),
            (19, 64, 8.0, 0.0, 1.0, 0.5, 0.5, 3.0), (100, 512, 7.0, 0.99, 0.2, 1.0, 1.2, 12.0),
            (200, 2048, 11.0, 0.6, 1.5, 0.05, 0.3, 8.0)]
    for C, D, R, a, g, rho, th, L in grid:
        check(construct(C, D, R, a, g, rho, th, L))
    for bad in [dict(C=10, D=20), dict(a=-0.1), dict(a=1.1), dict(theta=np.pi / 2),
                dict(L_par=0.0), dict(gamma=0.0), dict(R=-1.0)]:
        kw = dict(C=10, D=64, R=5.0, a=0.5, gamma=0.5, rho=1.0, theta=0.1, L_par=10.0)
        kw.update(bad)
        try:
            construct(**kw)
            raise AssertionError(f"infeasible request accepted: {bad}")
        except InfeasibleRequest:
            pass
    # sampled agreement with the frozen estimators (smoke, loose tolerance)
    from pilot0.geometry import fit_feature_model, papyan_metrics
    from pilot0.ood_coords import estimate_ood_coords
    m = construct(10, 64, 6.0, 0.7, 0.9, 1.5, 0.2, 10.0)
    rng = np.random.default_rng(1)
    n = 2000
    y = np.repeat(np.arange(10), n)
    h = m["means"][y] + rng.standard_normal((10 * n, 64))
    fm = fit_feature_model(h.astype(np.float32), y, 10)
    pap = papyan_metrics(m["W"], fm)
    co = estimate_ood_coords(m["m_ood"] + m["rho"] * rng.standard_normal((4000, 64)), fm)
    s_hat = 9 / np.sqrt(10 * pap["var_collapse"])
    assert abs(s_hat - 6.0) < 0.15, s_hat
    assert abs(co["a"] - 0.7) < 0.03 and abs(co["gamma"] - 0.9) < 0.03 and abs(co["rho"] - 1.5) < 0.03, co
    print("[etf] self-test PASS: exact identities on 5 grid points, 7 infeasible "
          f"requests rejected, sampled estimators agree (s {s_hat:.3f}, a {co['a']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    if ap.parse_args().self_test:
        self_test()

"""Theorem (head-space indistinguishability at the canonical point), from the
2026-09-11 evaluation (F5): self-dual ETF, zero bias, shared noise
covariance, OOD direction u = a u_1 + sqrt(1-a^2) v with v orthogonal to the
class-mean span; at gamma a = 1 the OOD logit law equals the ID class-1 logit
law, so every permutation-invariant logit score has AUROC exactly 1/2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def etf(C: int, D: int, rng) -> np.ndarray:
    """Rows: unit simplex-ETF directions u_c in a random (C-1)-dim subspace of R^D."""
    M = np.eye(C) - np.ones((C, C)) / C
    Q, _ = np.linalg.qr(rng.standard_normal((D, C - 1)))
    U = M @ np.linalg.qr(M)[0][:, :C - 1] @ Q.T          # C x D, rank C-1
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def auroc(s_id, s_ood):
    x = np.concatenate([s_ood, s_id]); r = rankdata(x); n0, n1 = len(s_ood), len(s_id)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def test_canonical_point_exact_logit_equality_and_chance_auroc():
    rng = np.random.default_rng(0)
    C, D, R, alpha, a, sigma = 7, 40, 3.0, 1.7, 0.4, 0.8
    U = etf(C, D, rng); W = alpha * U
    # v orthogonal to span{u_c}
    P = U.T @ np.linalg.pinv(U.T)                          # projector onto the mean span
    v = rng.standard_normal(D); v = v - P @ v; v /= np.linalg.norm(v)
    u = a * U[0] + np.sqrt(1 - a * a) * v
    gamma = 1 / a
    # exact equality of the projected means
    assert np.allclose(W @ (gamma * R * u), W @ (R * U[0]), atol=1e-12)
    # sampled logits: ID class 1 and OOD share the law; other classes permute it
    n = 200_000
    Z = rng.standard_normal((n, D)) * sigma
    g_id = (R * U[0] + Z) @ W.T; g_ood = (gamma * R * u + Z[::-1]) @ W.T
    mls = lambda g: g.max(1); energy = lambda g: np.log(np.exp(g - g.max(1, keepdims=True)).sum(1)) + g.max(1)
    msr = lambda g: np.exp(g.max(1) - energy(g))
    for name, fn in (("MLS", mls), ("Energy", energy), ("MSR", msr)):
        assert abs(auroc(fn(g_id), fn(g_ood)) - 0.5) < 0.005, name
    # covariance equality of the logit vectors (shared noise covariance): W Sigma W^T for both
    assert np.allclose(np.cov(g_id.T), np.cov(g_ood.T), atol=0.05)
    # permutation: class 2's logit law is a coordinate permutation of class 1's
    g2 = (R * U[1] + Z) @ W.T
    perm = np.argsort(np.argsort(-(W @ (R * U[1]))))
    assert np.allclose(np.sort(g2.mean(0)), np.sort(g_id.mean(0)), atol=0.02)
    # away from the point the score laws differ (sanity): gamma = 2/a
    g_far = (2 * gamma * R * u + Z[::-1]) @ W.T
    assert abs(auroc(mls(g_id), mls(g_far)) - 0.5) > 0.05


def test_tied_profile_is_not_covered():
    """The theorem needs a single-alignment profile: a tied profile is not a class-1 copy."""
    rng = np.random.default_rng(1)
    C, D, R, alpha = 5, 30, 3.0, 1.0
    U = etf(C, D, rng); W = alpha * U
    u = (U[0] + U[1]); u /= np.linalg.norm(u)
    a = float(U[0] @ u)
    assert not np.allclose(W @ (R * u / a), W @ (R * U[0]))

"""Theorem 1(i) repair (2026-09-09): (a) the tied-profile counterexample is
reproduced (exact MLS AUROC far below 1/2 at gamma a = 1); (b) on the
canonical single-alignment profile the exact MLS AUROC at gamma a = 1 is
within the stated switching bound p_ID + p_OOD of the surrogate's 1/2;
(c) at high SNR the bound is tiny and the crossing is sharp."""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pilot0.scores import auroc  # noqa: E402


def simplex(C):
    S = (np.eye(C) - 1 / C) * np.sqrt(C / (C - 1))
    return S / np.linalg.norm(S, axis=1, keepdims=True)


def mls_auroc(C, D, s, u_o_unit, gamma, n, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    U = simplex(C) @ Q[:, :C].T
    means = s * U
    m_o = gamma * s * (u_o_unit(U))
    y = rng.integers(0, C, n)
    h_id = means[y] + rng.standard_normal((n, D))
    h_od = m_o + rng.standard_normal((n, D))
    return auroc((h_id @ U.T).max(1), (h_od @ U.T).max(1))


def switching_bound(C, s, gamma, a):
    k = np.sqrt(C / (2 * (C - 1)))
    return (C - 1) * norm.cdf(-s * k) + (C - 1) * norm.cdf(-gamma * a * s * k)


def test_tied_profile_counterexample():
    A = mls_auroc(3, 32, 4.0, lambda U: (U[0] + U[1]) / np.linalg.norm(U[0] + U[1]), 2.0, 200_000, 0)
    assert A < 0.35, A                      # theorem-as-stated claimed 0.5; limit 0.2902


def test_canonical_profile_within_bound():
    C, s = 10, 6.0
    a = 0.8
    def u_o(U):
        _, sv, Vt = np.linalg.svd(U, full_matrices=True)
        v = Vt[C - 1]                        # complement of the mean span
        return a * U[0] + np.sqrt(1 - a * a) * v
    A = mls_auroc(C, 64, s, u_o, 1.0 / a, 200_000, 1)
    bound = switching_bound(C, s, 1.0 / a, a)
    assert abs(A - 0.5) <= bound + 0.005, (A, bound)


def test_high_snr_sharp_crossing():
    C, s, a = 10, 12.0, 0.8
    def u_o(U):
        _, sv, Vt = np.linalg.svd(U, full_matrices=True)
        return a * U[0] + np.sqrt(1 - a * a) * Vt[C - 1]
    A = mls_auroc(C, 64, s, u_o, 1.0 / a, 200_000, 2)
    assert switching_bound(C, s, 1.0 / a, a) < 1e-6 and abs(A - 0.5) < 0.01, A

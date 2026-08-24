"""Tests for the A2 nullspace geometry math (evaluation doc 4.2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nc_csf_predictivity.interventions.a2_nullspace_extract import (
    nullspace_fractions,
    nullspace_record,
    rowspan_basis,
    self_duality,
)

C, D = 5, 12


def _etf_head(rng: np.random.Generator) -> np.ndarray:
    """Rank-(C-1) simplex-ETF head embedded in D dimensions."""
    simplex = np.eye(C) - np.ones((C, C)) / C
    q, _ = np.linalg.qr(rng.normal(size=(D, C)))
    return simplex @ q.T[:C]


def test_rowspan_rank_of_etf_head():
    w = _etf_head(np.random.default_rng(0))
    assert rowspan_basis(w).shape[0] == C - 1


def test_nullspace_fraction_recovers_construction():
    rng = np.random.default_rng(1)
    w = _etf_head(rng)
    basis = rowspan_basis(w)
    _, _, vt_full = np.linalg.svd(w, full_matrices=True)
    null = vt_full[C - 1:]
    leak = rng.normal(size=(C, null.shape[0])) @ null
    m = w + leak
    eta, per_class = nullspace_fractions(m, basis)
    expected = (leak ** 2).sum() / (m ** 2).sum()
    assert abs(eta - expected) < 1e-10
    assert per_class.shape == (C,)
    assert eta > 0.3  # the construction leaks substantially


def test_projected_self_duality_separates_leakage():
    rng = np.random.default_rng(2)
    w = _etf_head(rng)
    _, _, vt_full = np.linalg.svd(w, full_matrices=True)
    null = vt_full[C - 1:]
    m = w + 2.0 * rng.normal(size=(C, null.shape[0])) @ null
    rec = nullspace_record(w, m)
    # Full-space self-duality looks broken; projected is exact (the
    # in-span component IS W), so the discrepancy is pure nullspace leak.
    assert rec["self_duality_full"] > 0.1
    assert rec["self_duality_proj"] < 1e-20
    assert rec["rank_w"] == C - 1
    assert rec["eta_perp"] > 0.5


def test_self_duality_zero_on_identical_matrices():
    rng = np.random.default_rng(3)
    w = rng.normal(size=(C, D))
    assert self_duality(w, 3.0 * w) < 1e-30  # scale-invariant metric


def test_rank_sensitivity_table():
    from nc_csf_predictivity.interventions.a2_nullspace_extract import (
        rank_sensitivity,
    )
    rng = np.random.default_rng(4)
    w = _etf_head(rng)
    _, _, vt_full = np.linalg.svd(w, full_matrices=True)
    null = vt_full[C - 1:]
    m = w + rng.normal(size=(C, null.shape[0])) @ null
    sens = rank_sensitivity(w, m, fixed_ranks=(C - 1,),
                            rtols=(1e-2, 1e-6))
    fixed = sens[f"rank_{C - 1}"]
    assert fixed["rank"] == C - 1
    # At the exact ETF rank the projection is clean: proj self-duality ~ 0.
    assert fixed["self_duality_proj"] < 1e-10
    # A strict relative threshold recovers the same rank and metrics.
    assert sens["rtol_1e-06"]["rank"] == C - 1
    assert abs(sens["rtol_1e-06"]["eta_perp"] - fixed["eta_perp"]) < 1e-12
    assert len(sens["singular_value_tail"]) == 3

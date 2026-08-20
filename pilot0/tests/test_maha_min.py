"""Tests for the min-statistic Maha predictor (B-axis protocol section 7)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot0.theory import predicted_maha_auroc, predicted_maha_auroc_min

C, D = 10, 32


def _setup(noise_sd: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.standard_normal((C, D))
    means = 10.0 * means / np.linalg.norm(means, axis=1, keepdims=True)
    cov = noise_sd**2 * np.eye(D)
    precision = np.eye(D) / noise_sd**2
    freq = np.full(C, 1.0 / C)
    return means, freq, precision, cov


def test_matches_closed_form_when_no_switching():
    # Well-separated regime: the argmin approximation is exact, so the
    # MC min-statistic must reproduce the frozen closed form.
    means, freq, precision, cov = _setup(noise_sd=0.3)
    m_ood = means[0] + 1.0 * means[1] / np.linalg.norm(means[1])
    closed = predicted_maha_auroc(means, precision, cov, m_ood, cov)
    mc, diag = predicted_maha_auroc_min(
        means, freq, precision, cov, m_ood, cov,
        n_samples=8000, seed=0, diagnostics=True)
    assert diag["id_switch_rate"] < 0.01
    assert diag["ood_nearest_share"] > 0.9
    assert abs(mc - closed) < 0.03


def test_diverges_from_closed_form_under_heavy_switching():
    # High-scatter regime (the A2 pattern): switching is rampant and the
    # argmin approximation no longer tracks the true min statistic.
    means, freq, precision, cov = _setup(noise_sd=5.0)
    m_ood = means.mean(0)
    closed = predicted_maha_auroc(means, precision, cov, m_ood, cov)
    mc, diag = predicted_maha_auroc_min(
        means, freq, precision, cov, m_ood, cov,
        n_samples=8000, seed=0, diagnostics=True)
    assert diag["id_switch_rate"] > 0.2
    # The OOD argmin distribution collapses toward uniform over classes.
    assert diag["ood_nearest_share"] < 0.3
    assert abs(mc - closed) > 0.05


def test_seeded_determinism_and_mc_stability():
    means, freq, precision, cov = _setup(noise_sd=1.5)
    m_ood = means.mean(0) + 1.0
    args = (means, freq, precision, cov, m_ood, cov)
    a = predicted_maha_auroc_min(*args, n_samples=4000, seed=0)
    b = predicted_maha_auroc_min(*args, n_samples=4000, seed=0)
    c = predicted_maha_auroc_min(*args, n_samples=4000, seed=1)
    assert a == b
    assert abs(a - c) < 0.02  # MC error, not model disagreement


def test_rank_auroc_bounds_and_degenerate_covariance():
    # Rank-deficient OOD covariance (small OOD sets) must sample fine.
    means, freq, precision, cov = _setup(noise_sd=0.5)
    cov_ood = np.zeros((D, D))
    cov_ood[0, 0] = 0.25
    m_ood = means[0] + 3.0
    auc = predicted_maha_auroc_min(
        means, freq, precision, cov, m_ood, cov_ood,
        n_samples=2000, seed=0)
    assert 0.0 <= auc <= 1.0

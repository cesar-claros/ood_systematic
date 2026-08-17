"""Synthetic end-to-end test of the Pilot 0 harness.

Runs the full protocol on a small ETF checkpoint where the X1 model is
exactly true, and asserts the pre-registered gates pass: exact feature-side
invariance, majority sign agreement on material cells, theory beating the
constant-response baseline, near-perfect H-estimator recovery, and the
AUGRC/failure-AUROC ranking identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot0.run_pilot0 import analyze_checkpoint, make_synthetic_cache


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> dict:
    rng = np.random.default_rng(7)
    cache = make_synthetic_cache(rng, n_classes=10, dim=64, snr=8.0)
    out = tmp_path_factory.mktemp("pilot0_out")
    (out / "pilot0_report.md").write_text("# test\n")
    return analyze_checkpoint(cache, out)


def test_logit_consistency(result: dict) -> None:
    assert result["gates"]["G0_logit_consistency"] < 1e-2


def test_feature_invariance_exact(result: dict) -> None:
    assert result["gates"]["G1_feature_invariance_max_abs"] == 0.0


def test_sign_agreement_both_arms(result: dict) -> None:
    for arm in ("iso", "emp"):
        assert result["gates"][f"G2_material_cells_{arm}"] > 10
        assert result["gates"][f"G2_sign_agreement_{arm}"] >= 0.8


def test_theory_beats_constant_on_responses(result: dict) -> None:
    for arm in ("iso", "emp"):
        assert (result["gates"][f"G3a_response_mae_{arm}"]
                < result["gates"]["G3a_response_mae_constant"])


def test_level_accurate_on_isotropic_truth(result: dict) -> None:
    # The synthetic checkpoint has isotropic noise, so both arms must
    # agree and both must be level-accurate.
    assert result["gates"]["G3b_level_mae_iso"] < 0.03
    assert result["gates"]["G3b_level_mae_emp"] < 0.03


def test_identity(result: dict) -> None:
    assert result["gates"]["G5_rank_agreement_worst_spearman"] > 0.99
    assert result["gates"]["G5_identity_max_abs_dev"] < 5e-3


def test_estimator_recovery(result: dict) -> None:
    for rec in result["h_validation"]:
        assert abs(rec["est_gamma"] - rec["true_gamma"]) < 0.05
        assert abs(rec["est_a"] - rec["true_a"]) < 0.05
        assert abs(rec["est_rho"] - rec["true_rho"]) < 0.05
        assert abs(rec["est_w_perp"] - rec["true_w_perp"]) < 0.05

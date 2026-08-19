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


def test_new_feature_nulls_exact(result: dict) -> None:
    # PCA_RE and Residual join the invariance null set; G1 must stay exact.
    assert result["gates"]["G1_feature_invariance_max_abs"] == 0.0


def test_fdbd_divergence(result: dict) -> None:
    # tier-3 prediction: the fDBD advantage over head-CTM grows with
    # away-rotation angle; on the exact-ETF synthetic the trend must be
    # positive in a majority of OOD sets and near-unit rank correlation
    # must hold between the two scores at the (near-collapsed) baseline.
    pos, total = result["gates"]["G7_fdbd_divergence_positive_sets"].split("/")
    assert int(pos) * 2 > int(total)
    assert result["fdbd_ctm_score_spearman_baseline"] > 0.9


def test_stage2b_feature_predictors_match_mc() -> None:
    # Maha and mean-CTM predictors (stage 2b additions) against Monte
    # Carlo on the exact-ETF synthetic: binormal predictions must land
    # within a few AUROC points of the rank-based empirical values.
    from pilot0.geometry import fit_feature_model
    from pilot0.run_pilot0 import make_synthetic_cache
    from pilot0.scores import MahalanobisScorer, ctm
    from pilot0.theory import predicted_ctm_mean_auroc, predicted_maha_auroc
    from pilot0.scores import auroc as emp_auroc

    cache = make_synthetic_cache(np.random.default_rng(9), n_classes=10,
                                 dim=64, snr=8.0)
    n_classes = cache["meta"]["n_classes"]
    model = fit_feature_model(cache["h_train"], cache["y_train"], n_classes)
    means_unc = model.class_means + model.global_mean
    maha = MahalanobisScorer(cache["h_train"], cache["y_train"], n_classes)
    precision = maha.precision
    cov_iso = model.sigma_iso**2 * np.eye(cache["h_train"].shape[1])

    for name in cache["meta"]["ood_sets"]:
        h_o = cache[f"h_{name}"].astype(np.float64)
        m_o = h_o.mean(0)
        resid = h_o - m_o
        cov_o = resid.T @ resid / len(resid)

        emp_m = emp_auroc(maha(cache["h_iid_test"]), maha(h_o))
        pred_m = predicted_maha_auroc(maha.means, precision, cov_iso,
                                      m_o, cov_o)
        assert abs(emp_m - pred_m) < 0.04, (name, emp_m, pred_m)

        emp_c = emp_auroc(ctm(cache["h_iid_test"], means_unc),
                          ctm(h_o, means_unc))
        pred_c = predicted_ctm_mean_auroc(means_unc, model.class_freq,
                                          cov_iso, m_o, cov_o)
        assert abs(emp_c - pred_c) < 0.04, (name, emp_c, pred_c)


def test_wperp_level_ordering() -> None:
    # On the exact-ETF synthetic the reconstruction-level proposition is
    # near-exact: the predicted AUROC must rank the OOD sets perfectly
    # (tilt g1.25 > tilt g1.0 > midpoint at w_perp = 0 ~ chance) and track
    # the empirical levels closely.
    from pilot0.diagnose_pilot0 import wperp_level_check
    from pilot0.run_pilot0 import make_synthetic_cache

    cache = make_synthetic_cache(np.random.default_rng(5), n_classes=10,
                                 dim=64, snr=8.0)
    out = wperp_level_check(cache)
    for family in ("PCA_RE", "Residual"):
        rec = out[family]
        assert rec["spearman_pred_emp"] > 0.99
        err = max(abs(p - e) for p, e in zip(rec["pred"], rec["emp"]))
        assert err < 0.06


def test_per_score_table(result: dict) -> None:
    # All four head scores are reported per score; on the exact-ETF
    # synthetic even the non-registered MSR diagnostic passes.
    assert set(result["per_score"]) == {"MSR", "MLS", "Energy", "CTM_head"}
    msr = result["per_score"]["MSR"]["emp"]
    assert msr["response_mae"] < result["per_score"]["MSR"][
        "response_mae_constant"]


def test_identity(result: dict) -> None:
    assert result["gates"]["G5_rank_agreement_worst_spearman"] > 0.99
    assert result["gates"]["G5_identity_max_abs_dev"] < 5e-3


def test_estimator_recovery(result: dict) -> None:
    for rec in result["h_validation"]:
        assert abs(rec["est_gamma"] - rec["true_gamma"]) < 0.05
        assert abs(rec["est_a"] - rec["true_a"]) < 0.05
        assert abs(rec["est_rho"] - rec["true_rho"]) < 0.05
        assert abs(rec["est_w_perp"] - rec["true_w_perp"]) < 0.05

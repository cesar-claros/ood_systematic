"""Tests for the Maha repair validation (operators + bounded calibration)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import nc_csf_predictivity.interventions.maha_repair_validation as mv

SETS = ["ood_a", "ood_b", "ood_c", "ood_d"]
RUNS = [1, 2, 3, 4]
DOSE_RAW = {"-0.1": 0.02, "0.3": -0.01, "1.0": -0.03, "hard": 0.334}
SAT = 0.15  # true saturation cap of the observed response
BASE_AUROC = 0.85


def raw_response(lam: str, set_name: str) -> float:
    tilt = 1.0 + 0.1 * SETS.index(set_name)
    return DOSE_RAW[lam] * tilt


def obs_response(lam: str, run: int, set_name: str) -> float:
    wiggle = 2e-4 * (((run * 5 + SETS.index(set_name)) % 7) - 3)
    return SAT * np.tanh(raw_response(lam, set_name) / SAT) + wiggle


def _repair() -> dict:
    repair = {}
    for lam in ("0.0", *mv.ARM_LAMS.values()):
        for run in RUNS:
            sets = {}
            for s in SETS:
                raw = 0.0 if lam == "0.0" else raw_response(lam, s)
                # dL_pred = pred_base - pred_arm, so arm preds sit at
                # BASE - raw (min operator) and BASE - 3*raw (old, the
                # 3x amplitude bias being repaired).
                sets[s] = {"pred_old": BASE_AUROC - 3.0 * raw,
                           "pred_min": BASE_AUROC - raw,
                           "emp_auroc": BASE_AUROC - (
                               0.0 if lam == "0.0"
                               else obs_response(lam, 1, s)),
                           "mc_diag": {
                               "id_switch_rate":
                                   0.6 if lam == "hard" else 0.05,
                               "ood_score_mean": -100.0,
                               "ood_score_var": 50.0,
                               "id_score_mean": -30.0,
                               "id_score_var": 20.0,
                               "ood_nearest_share": 0.2},
                           "emp_ood_score_mean": -110.0,
                           "emp_ood_score_var": 55.0,
                           "n_ood": 2000}
            repair[(lam, run)] = {
                "lam": lam, "run": run, "kind": "etfreg",
                "id_test": {"n": 10000, "n_correct": 7000,
                            "emp_switch_rate":
                                0.5 if lam == "hard" else 0.04,
                            "emp_score_mean": -30.0,
                            "emp_score_var": 20.0},
                "sets": sets}
    return repair


def _table() -> pd.DataFrame:
    rows = []
    for lam in ("0.0", *mv.ARM_LAMS.values()):
        for run in RUNS:
            for s in SETS:
                d = 0.0 if lam == "0.0" else obs_response(lam, run, s)
                rows.append({"lam": lam, "run": run, "set_name": s,
                             "method": "Maha",
                             "auroc_f": BASE_AUROC - d,
                             "augrc": (1 - BASE_AUROC + d) * 0.2})
    return pd.DataFrame(rows)


def test_operator_comparison_detects_repair():
    result = mv.run_validation(_table(), _repair())
    ops = result["operators"]
    # The old operator carries the 3x bias on both groups; the repaired
    # one matches A1 (linear region) and only saturation remains on A2.
    assert ops["A1"]["repair_improves"] and ops["A2"]["repair_improves"]
    assert ops["A1"]["mae_d_old"] > 2.0 * ops["A1"]["mae_d_min"]
    assert abs(ops["A1"]["mean_d_min"] - ops["A1"]["mean_d_obs"]) < 0.002
    assert ops["A2"]["mean_d_old"] > ops["A2"]["mean_d_obs"]
    assert ops["extraction_sanity_corr"] > 0.99


def test_calibration_selects_saturating_form():
    result = mv.run_validation(_table(), _repair())
    cal = result["calibration"]
    assert cal["selected_form"] in ("cap", "slope_cap")
    chosen = cal["forms"][cal["selected_form"]]
    assert chosen["a2_mae"] < cal["forms"]["linear"]["a2_mae"]
    assert chosen["a2_mae"] < cal["no_change"]["a2_mae"]
    assert chosen["a2_sign_agreement"] == 1.0
    # The cap parameter recovers the constructed saturation level.
    cap_value = abs(cal["forms"]["cap"]["params"][0])
    assert abs(cap_value - SAT) < 0.03


def test_diagnostics_summarize_switching():
    result = mv.run_validation(_table(), _repair())
    d = result["diagnostics"]
    assert d["A2"]["median_mc_id_switch"] > 5 * d["baseline"]["median_mc_id_switch"]
    assert d["A2"]["median_emp_id_switch"] > 0.4


def test_render_smoke():
    text = mv.render(mv.run_validation(_table(), _repair()))
    assert "Selected form" in text and "audit 5.4" in text
    assert "design data" in text

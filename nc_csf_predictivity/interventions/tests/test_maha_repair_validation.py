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


def true_response(lam: str, set_name: str) -> float:
    return SAT * np.tanh(raw_response(lam, set_name) / SAT)


def obs_response(lam: str, run: int, set_name: str) -> float:
    wiggle = 2e-4 * (((run * 5 + SETS.index(set_name)) % 7) - 3)
    return true_response(lam, set_name) + wiggle


def _repair(with_val: bool = False) -> dict:
    repair = {}
    for lam in ("0.0", *mv.ARM_LAMS.values()):
        for run in RUNS:
            sets = {}
            for s in SETS:
                raw = 0.0 if lam == "0.0" else raw_response(lam, s)
                truth = 0.0 if lam == "0.0" else true_response(lam, s)
                # dL_pred = pred_base - pred_arm: old carries a 3x
                # amplitude bias, min carries the raw (unsaturated) R.
                rec = {"pred_old": BASE_AUROC - 3.0 * raw,
                       "pred_min": BASE_AUROC - raw,
                       "emp_auroc": BASE_AUROC - truth,
                       "mc_diag": {
                           "id_switch_rate":
                               0.0,
                           "ood_score_mean": -100.0,
                           "ood_score_var": 50.0,
                           "id_score_mean": -30.0,
                           "id_score_var": 20.0,
                           "ood_nearest_share": 0.2},
                       "emp_ood_score_mean": -110.0,
                       "emp_ood_score_var": 55.0,
                       "n_ood": 2000}
                if with_val:
                    # Iteration 2: the val-fit operator tracks the truth.
                    rec["pred_old_val"] = BASE_AUROC - 1.4 * truth
                    rec["pred_min_val"] = BASE_AUROC - truth
                    rec["mc_diag_val"] = dict(rec["mc_diag"],
                                              id_switch_rate=0.25)
                sets[s] = rec
            model = {"lam": lam, "run": run, "kind": "etfreg",
                     "id_test": {"n": 10000, "n_correct": 7000,
                                 "emp_switch_rate":
                                     0.26 if lam == "hard" else 0.24,
                                 "emp_score_mean": -30.0,
                                 "emp_score_var": 20.0},
                     "sets": sets}
            if with_val:
                model["val"] = {"n": 5000, "emp_switch_rate": 0.25,
                                "emp_score_mean": -32.0,
                                "emp_score_var": 21.0}
            repair[(lam, run)] = model
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


def test_operator_comparison_ranks_operators():
    result = mv.run_validation(_table(), _repair())
    ops = result["operators"]
    assert ops["operators"] == ["d_old", "d_min"]
    for group in ("A1", "A2"):
        assert ops[group]["best_operator"] == "d_min"
        assert ops[group]["mae_d_old"] > ops[group]["mae_d_min"]
    assert abs(ops["A1"]["mean_d_min"] - ops["A1"]["mean_d_obs"]) < 0.002
    assert ops["A2"]["mean_d_old"] > ops["A2"]["mean_d_obs"]
    assert ops["extraction_sanity_corr"] > 0.99


def test_calibration_selects_saturating_form():
    result = mv.run_validation(_table(), _repair())
    cal = result["calibration"]
    sel = cal["selected"]
    assert sel["input"] == "d_min" and sel["form"] in ("cap", "slope_cap")
    forms = cal["inputs"]["d_min"]
    assert sel["a2_mae"] < forms["linear"]["a2_mae"]
    assert sel["a2_mae"] < cal["no_change"]["a2_mae"]
    # The cap parameter recovers the constructed saturation level.
    assert abs(abs(forms["cap"]["params"][0]) - SAT) < 0.03


def test_val_fit_operator_wins_when_present():
    result = mv.run_validation(_table(), _repair(with_val=True))
    ops = result["operators"]
    assert set(ops["operators"]) == {"d_old", "d_min", "d_old_val",
                                     "d_min_val"}
    for group in ("A1", "A2"):
        assert ops[group]["best_operator"] == "d_min_val"
    assert result["calibration"]["selected"]["input"] == "d_min_val"
    diag = result["diagnostics"]["A2"]
    # Falsifiable iteration-2 check: val-fit sampled switching matches
    # the empirical rate where the train-fit model said zero.
    assert diag["median_mc_id_switch"] == 0.0
    assert abs(diag["median_mc_val_id_switch"] - 0.25) < 1e-9
    assert abs(diag["median_val_emp_switch"] - 0.25) < 1e-9


def test_diagnostics_and_render_smoke():
    result = mv.run_validation(_table(), _repair(with_val=True))
    text = mv.render(result)
    assert "Selected (fresh-confirmation candidate): d_min_val" in text
    assert "val-fit" in text and "design data" in text
    text_noval = mv.render(mv.run_validation(_table(), _repair()))
    assert "d_min" in text_noval

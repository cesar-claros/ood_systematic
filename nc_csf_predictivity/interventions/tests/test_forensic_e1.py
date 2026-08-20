"""Tests for the E1 forensic reanalysis (evaluation doc 5.1-5.4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import nc_csf_predictivity.interventions.forensic_e1 as fx

SETS = ["ood_a", "ood_b", "ood_c", "ood_d"]
RUNS = [1, 2, 3, 4]
LAMS = ["0.0", "-0.1", "0.3", "1.0", "hard"]
DOSE = {"0.0": 0.0, "-0.1": -0.1, "0.3": 0.3, "1.0": 1.0}
TRUE_SLOPE = 1.5


def pred_auroc(score: str, lam: str, set_name: str) -> float:
    tilt = 0.003 * SETS.index(set_name)
    if score == "MLS":
        base = 0.90 + tilt
        return base + (0.005 * DOSE[lam] if lam != "hard" else 0.02)
    base = 0.85 + tilt
    return base + (0.02 * DOSE[lam] if lam != "hard" else -0.30)


def seed_wiggle(lam: str, run: int) -> float:
    if lam == "0.0":
        return 0.0
    return 2e-4 * (((run * 3 + LAMS.index(lam)) % 5) - 2)


def obs_auroc(score: str, lam: str, run: int, set_name: str) -> float:
    loss = 0.001 + TRUE_SLOPE * (1.0 - pred_auroc(score, lam, set_name))
    if score == "Maha":
        loss += seed_wiggle(lam, run)
    return 1.0 - loss


def _table() -> pd.DataFrame:
    rows = []
    for lam in LAMS:
        for run in RUNS:
            for s in SETS:
                for score in ("MLS", "Maha"):
                    a = obs_auroc(score, lam, run, s)
                    rows.append({"name": f"x_run{run}_lam{lam}",
                                 "lam": lam, "run": run, "set_name": s,
                                 "method": score, "auroc_f": a,
                                 "augrc": (1.0 - a) * 0.2})
    return pd.DataFrame(rows)


def _stage2b() -> dict:
    out = {}
    for lam in LAMS:
        for run in RUNS:
            out[(lam, run)] = {"lam": lam, "run": run, "sets": {
                s: {"preds": {"emp": {
                    "MLS": pred_auroc("MLS", lam, s),
                    "Maha": pred_auroc("Maha", lam, s)}}}
                for s in SETS}}
    return out


def _committed_e1(stage2b: dict) -> dict:
    committed = {}
    for label, lam in fx.ARM_LAMS.items():
        cells = {}
        for s in SETS:
            r_hat = fx.pred_response(stage2b, lam, 1, s)
            cells[s] = {"sign": int(np.sign(r_hat)),
                        "material": abs(r_hat) >= 0.001}
        committed[label] = {"cells": cells}
    return committed


def _geometry() -> dict:
    rng = np.random.default_rng(3)
    return {(lam, run): {f: float(rng.normal()) for f in fx.Q_FIELDS}
            for lam in LAMS for run in RUNS}


def _run_all() -> dict:
    table, s2b = _table(), _stage2b()
    committed = _committed_e1(s2b)
    runs = RUNS
    return {
        "a1_only": fx.a1_only_analysis(table, committed, runs),
        "decompose": fx.decompose(table, runs, SETS),
        "response_transport": fx.response_transport(
            table, s2b, _geometry(), runs, SETS),
        "amplitude": fx.amplitude_attribution(table, s2b, runs, SETS),
    }


def test_a1_only_seed_level_statistics():
    r = _run_all()["a1_only"]
    pooled = r["A1_pooled"]
    assert pooled["mean"] > 0
    assert pooled["n_seeds_positive"] == 4
    assert pooled["sign_test_p"] == 0.0625  # exact floor at four seeds
    assert pooled["t_p_one_sided"] < 0.05
    assert pooled["cell_agreement"].split("/")[0] == \
        pooled["cell_agreement"].split("/")[1]
    # A2 is reported but its effect dwarfs A1 (never pooled).
    assert r["A2"]["mean"] > 10 * pooled["mean"]


def test_decomposition_attributes_channels():
    r = _run_all()["decompose"]
    # A2's Mahalanobis crater dominates the gap response.
    assert r["A2"]["maha_share"] > 0.8
    # The A1 co-movement construction is also Maha-dominated.
    assert r["A1+"]["maha_share"] > 0.5


def test_response_transport_coupled_world():
    r = _run_all()["response_transport"]
    for rec in r["per_arm_raw"].values():
        assert rec["sign_agreement"] == 1.0
    a12 = r["a1_to_a2"]
    assert abs(a12["fit"]["beta"] - TRUE_SLOPE) < 0.05
    assert a12["mae"]["calibrated_plugin"] < a12["mae"]["raw_plugin"]
    assert a12["mae"]["calibrated_plugin"] < a12["mae"]["no_change"]
    assert a12["mae"]["calibrated_plugin"] < a12["mae"]["delta_nuisance"]
    assert a12["sign_agreement_calibrated"] == 1.0
    for rec in r["within_a1_loo_dose"].values():
        assert rec["sign_agreement"] == 1.0
        assert rec["mae_calibrated"] < rec["mae_no_change"]


def test_amplitude_attribution_localizes_to_maha():
    r = _run_all()["amplitude"]
    assert r["A2"]["maha_error_share"] > 0.8
    assert r["A2"]["Maha"]["mae_pred"] > r["A2"]["MLS"]["mae_pred"]


def test_render_smoke():
    text = fx.render(_run_all())
    assert "A1-only E1 (5.1)" in text
    assert "POST HOC" in text
    assert "0.0625" in text

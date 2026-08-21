"""Tests for the B-axis dose-search gates and selection."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import nc_csf_predictivity.interventions.b_dose_report as bd

RUNS = [1, 2, 3, 4]
B_RUNS = [1, 2]


def _ref_record(lam: str, run: int, rng: np.random.Generator) -> dict:
    vc = {"0.0": 0.119, "-0.1": 0.128, "0.3": 0.115, "1.0": 0.111}[lam]
    return {"kind": "etfreg", "lam": lam, "run": run,
            "var_collapse": vc + 0.0004 * rng.normal(),
            "self_duality": {"0.0": 0.014, "-0.1": 0.050,
                             "0.3": 0.004, "1.0": 0.003}[lam]
            + 0.0002 * rng.normal(),
            "logit_scale": 9.8 + 0.05 * rng.normal(),
            "eig_max_over_mean": 21.0 + 0.3 * rng.normal(),
            "head_residual_fraction": 0.006 + 0.0005 * rng.normal(),
            "val_acc": 0.70 + 0.002 * rng.normal()}


def _b_record(kind: str, lam: str, run: int, vc: float, acc: float,
              rng: np.random.Generator, logit_scale: float = 9.8) -> dict:
    return {"kind": kind, "lam": lam, "run": run,
            "var_collapse": vc + 0.0003 * rng.normal(),
            "self_duality": 0.014 + 0.0005 * rng.normal(),
            "logit_scale": logit_scale + 0.05 * rng.normal(),
            "eig_max_over_mean": 21.0 + 0.3 * rng.normal(),
            "head_residual_fraction": 0.006 + 0.0005 * rng.normal(),
            "val_acc": acc + 0.001 * rng.normal()}


def _geometry() -> dict:
    rng = np.random.default_rng(0)
    geo = {}
    for lam in bd.REF_LAMS:
        for run in RUNS:
            geo[("etfreg", lam, run)] = _ref_record(lam, run, rng)
    for run in B_RUNS:
        # Matched dose: lands at the A1++ var_collapse level, acc intact.
        geo[("varreg", "1.0", run)] = _b_record(
            "varreg", "1.0", run, vc=0.1095, acc=0.699, rng=rng)
        # Strong dose: big movement but blows logit scale out of the span.
        geo[("varreg", "3.0", run)] = _b_record(
            "varreg", "3.0", run, vc=0.050, acc=0.697, rng=rng,
            logit_scale=25.0)
        # Accuracy-violating dose.
        geo[("ctrreg", "0.01", run)] = _b_record(
            "ctrreg", "0.01", run, vc=0.100, acc=0.66, rng=rng)
        # Immaterial dose: barely moves the target.
        geo[("ctrreg", "0.0003", run)] = _b_record(
            "ctrreg", "0.0003", run, vc=0.1185, acc=0.700, rng=rng)
    return geo


def test_gates_and_selection():
    result = bd.run_report(_geometry(), nulls={})
    doses = result["doses"]
    matched = doses["varreg"]["1.0"]
    assert matched["all_pass"] and matched["on_support"]
    # GB2 is anchored to the A1++ displacement (amendment 2026-08-21).
    assert abs(matched["median_d_vc"]) >= 0.008
    strong = doses["varreg"]["3.0"]
    assert not strong["gates"]["GB3_selectivity"]
    assert not strong["on_support"]
    acc_fail = doses["ctrreg"]["0.01"]
    assert not acc_fail["gates"]["GB1_accuracy"]
    weak = doses["ctrreg"]["0.0003"]
    assert not weak["gates"]["GB2_material"]
    assert result["recommended"]["varreg"] == "1.0"
    assert result["recommended"]["ctrreg"] is None
    assert result["overall_recommendation"] == ("varreg", "1.0")


def test_leakage_gate_uses_nullspace_records():
    geo = _geometry()
    nulls = {k: {"eta_perp": 0.005} for k in geo if k[0] == "etfreg"}
    for run in B_RUNS:
        nulls[("varreg", "1.0", run)] = {"eta_perp": 0.004}   # fine
        nulls[("varreg", "3.0", run)] = {"eta_perp": 0.30}    # extreme
    result = bd.run_report(geo, nulls)
    assert result["doses"]["varreg"]["1.0"]["gates"]["GB4_leakage"]
    assert not result["doses"]["varreg"]["3.0"]["gates"]["GB4_leakage"]
    # ctrreg has no nullspace records -> GB4 not evaluated, not failed.
    assert "GB4_leakage" not in result["doses"]["ctrreg"]["0.01"]["gates"]


def test_single_seed_fails_completeness():
    geo = _geometry()
    del geo[("varreg", "1.0", 2)]
    result = bd.run_report(geo, nulls={})
    rec = result["doses"]["varreg"]["1.0"]
    assert not rec["gates"]["GB5_complete"] and not rec["all_pass"]


def test_loader_merges_dirs_and_requires_kind(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir(), d2.mkdir()
    rec = _ref_record("0.0", 1, np.random.default_rng(1))
    (d1 / "etfreg_bbvgg13_do0_run1_lam0.0__last.json").write_text(
        json.dumps(rec))
    brec = _b_record("varreg", "1.0", 1, 0.11, 0.7,
                     np.random.default_rng(2))
    (d2 / "varreg_bbvgg13_do0_run1_lam1.0__last.json").write_text(
        json.dumps(brec))
    (d2 / "nokind__last.json").write_text(json.dumps({"lam": "0.0"}))
    out = bd.load_records([d1, d2, tmp_path / "missing"])
    assert set(out) == {("etfreg", "0.0", 1), ("varreg", "1.0", 1)}


def test_render_smoke():
    text = bd.render(bd.run_report(_geometry(), nulls={}))
    assert "Overall geometry-matched pick: ('varreg', '1.0')" in text
    assert "QUALIFIES" in text and "rejected" in text

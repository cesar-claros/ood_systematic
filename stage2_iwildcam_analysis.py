"""iWildCam Stage-2 analysis (amendment 7 registered readouts; the
committed FIRST READER of the stage-2 records, per the voluntarily
adopted GR-5 discipline).

Readouts (all DESCRIPTIVE at n = 5; no claim upgrade is available):
- code-asserted denominators (5 records, 1 forwarded set each, clean-ID
  construction fields present and consistent);
- re-measured orientation AUROCs per score;
- material-cell counts, raw and prevalence-balanced;
- the frozen corrected-dictionary arm (ICML section-8.1 recipe,
  IDENTICAL arithmetic via import from icml_campaign_analysis):
  winner-sign agreement on material cells, Spearman(|M|, |gap|),
  direct-AUROC level MAE (P00 only; no mixture stats are extracted for
  this source).

Usage (from code/):
    python stage2_iwildcam_analysis.py [--self-test]
Input:  pilot0/stage2_iwildcam_coords/
Output: nc_csf_predictivity/outputs/track1/stage2_iwildcam_report.md/.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from crossing_robustness_audit import OUT_DIR
from icml_campaign_analysis import frozen_margin, pred_auroc

CODE = Path(__file__).resolve().parent
IN_DIR = CODE / "pilot0/stage2_iwildcam_coords"
EXPECTED_N = 5
OOD_NAME = "wilds_animals_ood_test"
SCORES = ("Energy", "CTM", "MSR", "MLS", "Maha", "fDBD")


def analyze(records: list[dict]) -> dict:
    assert len(records) == EXPECTED_N, (
        f"expected {EXPECTED_N} records, got {len(records)}")
    rows = []
    for r in records:
        e = r["ood"][OOD_NAME]
        assert "error" not in e, (r["slug"], e)
        it = r["iid_test"]
        assert it["n_excluded_devries_val"] == 1000, it
        assert it["n_clean"] == it["n_after_pre_slice"] - 1000, it
        l_e, l_c = frozen_margin(r, e)
        rows.append({
            "cell": f"{r['paradigm']}/rew{r['reward']}",
            "n_id_clean": it["n_clean"], "pre_slice": it["pre_slice"],
            "id_error_rate": round(it["id_error_rate"], 4),
            "auroc": {s: e[f"auroc_id_vs_ood_{s}"] for s in SCORES},
            "gap_raw": e["gap_raw"], "gap_balanced": e["gap_balanced"],
            "material_raw": bool(abs(e["gap_raw"]) >= 0.01),
            "material_balanced": bool(e["material"]),
            "k_balanced": e["k_balanced"],
            "l_E": l_e, "l_C": l_c, "M": l_e - l_c,
            "sign_match_balanced": (bool(np.sign(l_e - l_c)
                                         == np.sign(e["gap_balanced"]))
                                    if e["gap_balanced"] != 0 else None),
            "pred_auroc_Energy": round(pred_auroc(l_e), 4),
            "pred_auroc_CTM": round(pred_auroc(l_c), 4),
            "level_abs_err_Energy": round(abs(
                pred_auroc(l_e) - e["auroc_id_vs_ood_Energy"]), 4),
            "level_abs_err_CTM": round(abs(
                pred_auroc(l_c) - e["auroc_id_vs_ood_CTM"]), 4),
        })
    mat_bal = [r for r in rows if r["material_balanced"]]
    m_vals = [r["M"] for r in rows]
    g_vals = [r["gap_balanced"] for r in rows]
    rho = (float(spearmanr(np.abs(m_vals), np.abs(g_vals)).statistic)
           if len(set(np.abs(g_vals))) > 1 else None)
    return {
        "denominators": {
            "n_records": len(records),
            "n_id_clean": sorted({r["n_id_clean"] for r in rows}),
            "pre_slice": sorted({r["pre_slice"] for r in rows}),
            "n_ood": 42791},
        "cells": rows,
        "n_material_raw": sum(r["material_raw"] for r in rows),
        "n_material_balanced": len(mat_bal),
        "sign_agreement_material_balanced": (
            [r["sign_match_balanced"] for r in mat_bal] or None),
        "spearman_absM_absgap_balanced_descriptive_n5": rho,
        "level_mae_Energy": round(float(np.mean(
            [r["level_abs_err_Energy"] for r in rows])), 4),
        "level_mae_CTM": round(float(np.mean(
            [r["level_abs_err_CTM"] for r in rows])), 4),
    }


def self_test() -> None:
    recs = []
    for i in range(EXPECTED_N):
        recs.append({
            "slug": f"s{i}", "paradigm": "dg", "reward": str(i),
            "n_classes": 182, "dim": 2048,
            "papyan": {"var_collapse": 1.5 + 0.1 * i,
                       "self_duality": 0.9},
            "geometry": {"logit_scale": 11.0,
                         "class_mean_radius_cv": 0.29},
            "iid_test": {"n_clean": 7154, "pre_slice": "None",
                         "n_after_pre_slice": 8154,
                         "n_excluded_devries_val": 1000,
                         "id_error_rate": 0.35},
            "ood": {OOD_NAME: {
                "gamma": 0.22, "a": 0.7, "rho": 1.4,
                "gap_raw": 0.02 * (1 if i % 2 else -1),
                "gap_balanced": 0.02 * (1 if i % 2 else -1),
                "material": True, "k_balanced": 7154,
                **{f"auroc_id_vs_ood_{s}": 0.5 + 0.02 * i
                   for s in SCORES}}}})
    rep = analyze(recs)
    assert rep["denominators"]["n_records"] == 5
    assert rep["n_material_balanced"] == 5
    assert all(np.isfinite(r["M"]) for r in rep["cells"])
    assert rep["level_mae_Energy"] >= 0
    print("[s2-iwc-analysis] self-test PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    records = []
    for p in sorted(IN_DIR.glob("*.json")):
        if p.name.startswith("FAILED_"):
            raise SystemExit(f"failed extraction present: {p.name}")
        records.append(json.loads(p.read_text()))
    rep = analyze(records)
    (OUT_DIR / "stage2_iwildcam_report.json").write_text(
        json.dumps(rep, indent=1, default=float))
    lines = ["# iWildCam Stage-2 readout (amendment 7; descriptive, "
             "n = 5)", "", "```", json.dumps(rep, indent=1,
                                             default=float), "```", ""]
    (OUT_DIR / "stage2_iwildcam_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

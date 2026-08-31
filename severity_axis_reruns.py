"""Severity-axis amendment reruns: the complete severity-consuming
pipeline under d^K (KID, amended primary) and d^F (FD, mandatory
robustness), per [[KID_FD_severity_axis_recommendations_2026-08-28]]
sections 6.5-6.6 and the 2026-08-31 contract amendment (declared BEFORE
this script ran).

EVIDENCE CLASS: post-registration methodological amendment for the
historical pool (the composite outcomes were inspected); both axes are
reported regardless of outcome; the composite results stay in the
appendix as the registered analysis.

FROZEN SPEC: axes d^K = z_s(kid), d^F = z_s(fd) via the audited
severity_map single-axis construction (per-source standardization over
the frozen suite; the csv's text_align column plays no role). Everything
else unchanged: checkpoints, sources, shifts, Energy-CTM pair,
materiality |gap| >= 10 milli, estimators, censoring rules, corrected
NC1 panel (v2 convention) for every stratification and geometry
feature. Seeds: pooled/case-study bands 1301, held-out geometry boot
1311, corrected-arm boot 1321, folds 2027, joint audit as registered
(20260826). Analyses per axis, mapped onto the post-audit-11 objects
(the pooled tertile ordering is retired; the ordering object is the
CIFAR-100 within-source case study):
 A. pooled crossing suite: four estimators, pava B=2000 bands,
    sign-change counts, leave-one-OOD-set-out and leave-one-source-out
    pooled-location ranges (denominators asserted);
 B. CIFAR-100 case study: within-source corrected-panel tertiles, four
    estimators (pava banded), leave-one-set-out (n asserted);
 C. paradigm-stratified crossings + equal-paradigm pooled curve
    (corrected panel);
 D. held-out suite: severity-only baseline, geometry and flexible
    models, both fold modes, geometry-minus-severity and
    corrected-dictionary-minus-severity paired checkpoint bootstraps
    (B=2000);
 E. joint confounding audit (M0/M0+/M1...) with the amended severity.
GATES (declared here; A and B of the document are PENDING the HPC
regeneration and are so reported):
 GC: under d^K, the cifar100 case-study ordering is retained (pava) AND
     a pooled first up-crossing exists.
 GD: under d^F, the cifar100 ordering is retained (weak delayed or
     absent reading per the frozen censoring-aware rule).
 GE: per axis, ordering retained on 4/4 estimators and on at least
     n_loo - 1 of the n_loo case-study set-deletions (n_loo asserted
     from the actual suite; expected 8).
 GF: per axis, held-out geometry-minus-severity (sign accuracy, ckpt5)
     keeps a positive point with CI excluding zero.

Usage (from code/): python severity_axis_reruns.py
Output: outputs/track1/severity_axis_reruns_report.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossing_robustness_audit import (ESTIMATORS, OUT_DIR, PARQUET,
                                       attach_d, build_cells,
                                       load_severity_rows,
                                       ordering_retained, severity_map,
                                       stratified, tertiles)
from corrected_dictionary_audit import CACHE as CORR_CACHE
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, load_coords,
                                       load_outcomes, run_folds)
from joint_confound_audit import (add_metadata, joint_summary,
                                  paradigm_crossings, run_joint)

B, SEED_BAND = 2000, 1301
SEED_GEO, SEED_CORR = 1311, 1321
AXES = {"K": ("kid",), "F": ("fd",)}


def summar(res):
    out = {k: {"x": res[k].get("first_up_crossing"),
               "tie": res[k].get("tie_region"),
               "n_sign": res[k].get("n_sign_changes")}
           for k in res if isinstance(res[k], dict)}
    out["ordering_retained"] = bool(ordering_retained(res))
    return out


def crossing_block(cells_d, vc_corr):
    """Pooled suite + cifar100 case study for one axis."""
    out = {}
    rng = np.random.default_rng(SEED_BAND)
    pooled = {est: stratified(cells_d, {}, est, B if est == "pava" else 0,
                              rng)["pooled"] for est in ESTIMATORS}
    out["pooled"] = {est: {"x": p.get("first_up_crossing"),
                           "tie": p.get("tie_region"),
                           "n_sign": p.get("n_sign_changes")}
                     for est, p in pooled.items()}
    loo_sets = sorted(cells_d.eval_dataset.unique())
    out["pooled_loo_locations"] = {
        e: stratified(cells_d[cells_d.eval_dataset != e], {}, "pava", 0,
                      np.random.default_rng(SEED_BAND)
                      )["pooled"].get("first_up_crossing")
        for e in loo_sets}
    out["pooled_loso_locations"] = {
        s: stratified(cells_d[cells_d.source != s], {}, "pava", 0,
                      np.random.default_rng(SEED_BAND)
                      )["pooled"].get("first_up_crossing")
        for s in sorted(cells_d.source.unique())}
    c100 = cells_d[cells_d.source == "cifar100"].copy()
    c100["var_collapse"] = c100.cell.map(vc_corr)
    strata = tertiles(c100)
    est_res = {}
    for est in ESTIMATORS:
        rng = np.random.default_rng(SEED_BAND)
        est_res[est] = summar(stratified(c100, strata, est,
                                         B if est == "pava" else 0, rng))
    n_loo = len(sorted(c100.eval_dataset.unique()))
    loo = {}
    for e in sorted(c100.eval_dataset.unique()):
        sub = c100[c100.eval_dataset != e]
        loo[e] = summar(stratified(sub, tertiles(sub), "pava", 0,
                                   np.random.default_rng(SEED_BAND)))
    out["c100_estimators"] = est_res
    out["c100_loo"] = {e: v["ordering_retained"] for e, v in loo.items()}
    out["c100_loo_n"] = n_loo
    out["gate_E_inputs"] = {
        "estimators_retained": sum(est_res[e]["ordering_retained"]
                                   for e in ESTIMATORS),
        "loo_retained": sum(loo[e]["ordering_retained"] for e in loo),
        "n_loo": n_loo}
    return out


def heldout_block(cells_d):
    """Held-out suite for one axis (severity baseline changes with d;
    theory = corrected-dictionary sign from the frozen cache)."""
    corr = pd.read_parquet(CORR_CACHE)[["cell", "eval_dataset", "m_corr"]]
    cells = cells_d.merge(corr, on=["cell", "eval_dataset"], how="left")
    theory = pd.Series(np.sign(cells.m_corr.values), index=cells.index)
    out = {}
    for mode in ("ckpt5", "loso"):
        fr = run_folds(cells, theory, mode,
                       np.random.default_rng(FOLD_SEED))
        m = fr[np.abs(fr.gap) >= MATERIALITY].copy()
        o = m.gap.values
        arms = {a: {"sign": round(accuracy(np.sign(m[a].values), o), 4),
                    "balanced": round(balanced_accuracy(
                        np.sign(m[a].values), o), 4)}
                for a in ("theory", "severity", "geometry", "flexible",
                          "mean")}
        cks = np.array(sorted(m.cell.unique()))
        gps = {c: g for c, g in m.groupby("cell")}

        def boot(col_a, col_b, seed):
            rng = np.random.default_rng(seed)
            d = np.empty(B)
            for i in range(B):
                b = pd.concat([gps[c] for c in
                               rng.choice(cks, len(cks), replace=True)])
                d[i] = (accuracy(np.sign(b[col_a].values), b.gap.values)
                        - accuracy(np.sign(b[col_b].values), b.gap.values))
            point = (accuracy(np.sign(m[col_a].values), o)
                     - accuracy(np.sign(m[col_b].values), o))
            return {"point": round(float(point), 4),
                    "ci95": [round(float(np.quantile(d, q)), 4)
                             for q in (0.025, 0.975)]}
        out[mode] = {"n_material": int(len(m)), "arms": arms,
                     "geometry_minus_severity": boot("geometry",
                                                     "severity", SEED_GEO),
                     "corrected_minus_severity": boot("theory", "severity",
                                                      SEED_CORR)}
    return out


def main() -> None:
    df = pd.read_parquet(PARQUET)
    sev_rows = load_severity_rows()
    coords, _ = load_coords(Path("pilot0/pool_coords"))
    base = build_cells(df)
    vc_corr = {c: coords[c]["papyan"]["var_collapse"]
               for c in base.cell.unique()}
    ho_base = load_outcomes(df)
    report = {}
    for axis, metrics in AXES.items():
        dm = severity_map(sev_rows, metrics)
        cells_d = attach_d(base, dm)
        print(f"[sev-{axis}] crossing suite ...", flush=True)
        blk = crossing_block(cells_d, vc_corr)
        ho = attach_d(ho_base, dm)
        print(f"[sev-{axis}] held-out suite ...", flush=True)
        blk["heldout"] = heldout_block(ho)
        meta = ho.copy()
        meta["var_collapse"] = meta.cell.map(vc_corr)
        meta = add_metadata(meta, coords)
        print(f"[sev-{axis}] paradigm crossings ...", flush=True)
        pc = paradigm_crossings(meta)
        blk["paradigm"] = {p: {"strata": {k: v["strata"][k].get(
            "first_up_crossing") for k in ("strong", "middle", "weak")}}
            for p, v in pc.items() if p != "equal_paradigm_pooled"}
        blk["equal_paradigm_pooled_x"] = pc["equal_paradigm_pooled"].get(
            "first_up_crossing")
        print(f"[sev-{axis}] joint audit ...", flush=True)
        js = joint_summary(run_joint(meta))
        blk["joint"] = {k: js[k] for k in ("arms", "differences")
                        if k in js}
        # gates
        ge = blk["gate_E_inputs"]
        assert ge["n_loo"] == len(blk["c100_loo"])
        pava_ok = blk["c100_estimators"]["pava"]["ordering_retained"]
        pooled_x = blk["pooled"]["pava"]["x"]
        blk["gates"] = {
            "GC_or_GD": bool(pava_ok and (pooled_x is not None)),
            "GE": bool(ge["estimators_retained"] == 4
                       and ge["loo_retained"] >= ge["n_loo"] - 1),
            "GF": bool(blk["heldout"]["ckpt5"]
                       ["geometry_minus_severity"]["ci95"][0] > 0),
        }
        report[axis] = blk
    report["gates_pending_hpc"] = {
        "GateA_provenance": "recorded; full pass requires the CLIP "
                            "regeneration (ICML extraction job)",
        "GateB_kid_uncertainty": "requires features; pending same job"}
    (OUT_DIR / "severity_axis_reruns_report.json").write_text(
        json.dumps(report, indent=1, default=str))
    L = ["# Severity-axis amendment reruns (d^K primary, d^F robustness)",
         "", "Post-registration amendment; spec and gates in the script "
         "header; both axes reported.", "", "```",
         json.dumps(report, indent=1, default=str), "```", ""]
    (OUT_DIR / "severity_axis_reruns_report.md").write_text("\n".join(L))
    for axis in AXES:
        print(axis, json.dumps(report[axis]["gates"]))


if __name__ == "__main__":
    main()

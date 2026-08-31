"""Audit-11 R11.2: the convention-robust panel-B primary (cifar100
within-source collapse tertiles, decision D2 = case study) and its
scoped robustness suite. Gate G11.B lives here.

FROZEN SPEC (declared before outcomes): subset = cifar100 checkpoints
(70) with the registered full-suite severity axis; strata = within-
source var_collapse tertiles by the frozen thirds rule, computed on the
CORRECTED panel (pool_coords pilot0 papyan; the old panel agrees within
source at Spearman 0.905); primary estimator pava with B = 2000
simultaneous bands, seed 1101; robustness: all four frozen estimators
(B = 2000 each), 9 leave-one-OOD-set-out refits (pava, point verdicts),
9 severity-definition variants (full composite, four single-axis, four
without-axis; pava, point verdicts). Ordering verdict = the frozen
ordering_retained rule. GATE G11.B PASSES iff the ordering is retained
on 4/4 estimators, >= 8/9 leave-one-OOD-set-out refits, and >= 7/9
severity variants. If G11.B fails, the ordering claim is DROPPED (not
softened) per the audit-11 decision document.

Also exports the primary strata curves (fine grid, pava curve, band q95,
first up-crossings) for the hero panel-B rebuild.

POST-RUN NOTE (2026-08-30, before any consequence was applied): the
declared LOO threshold used denominator 9, which is impossible for
cifar100's 8-set suite (the 9-set count is the pool-wide union), so
as written the gate demands zero failing refits; the observed result
is 7/8 (one failure: dropping 'lsun cropped' swaps strong and middle
by 0.10). RULING (strict execution): G11.B FAIL as written; the
ordering claim is dropped from the paper's claims per the audit-11
consequence, the case study stays descriptive with the exceptions
disclosed, and the corrected-count reading (7/8 = at most one
failure) is recorded for the user's Option-C decision. The report
strings' '/9' denominators are this defect, left as produced.

Usage (from code/): python panelb_cifar100_audit.py
Outputs: outputs/track1/panelb_cifar100_report.md/.json
         outputs/track1/panelb_cifar100_curves.json  (hero input)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossing_robustness_audit import (ESTIMATORS, METRICS, OUT_DIR,
                                       PARQUET, attach_d, build_cells,
                                       curve, load_severity_rows,
                                       make_data, ordering_retained,
                                       severity_map, stratified, tertiles)
from heldout_theory_validation import load_coords

B, SEED = 2000, 1101
NAMES = ("strong", "middle", "weak")


def summar(res):
    out = {k: {"x": res[k].get("first_up_crossing"),
               "tie": res[k].get("tie_region"),
               "n_sign": res[k].get("n_sign_changes")}
           for k in ("pooled",) + NAMES if k in res}
    out["ordering_retained"] = bool(ordering_retained(res))
    return out


def main() -> None:
    cells = build_cells(pd.read_parquet(PARQUET))
    sev_rows = load_severity_rows()
    dmap = severity_map(sev_rows, METRICS)
    coords, _ = load_coords(Path("pilot0/pool_coords"))

    cd = attach_d(cells, dmap)
    cd = cd[cd.source == "cifar100"].copy()
    vc_new = {c: coords[c]["papyan"]["var_collapse"]
              for c in cd.cell.unique()}
    cd["var_collapse"] = cd.cell.map(vc_new)
    strata = tertiles(cd)

    # primary + estimator robustness (full bands)
    est_res = {}
    for est in ESTIMATORS:
        rng = np.random.default_rng(SEED)
        est_res[est] = summar(stratified(cd, strata, est, B, rng))
    est_pass = sum(est_res[e]["ordering_retained"] for e in ESTIMATORS)

    # leave-one-OOD-set-out (pava, point verdicts)
    loo = {}
    for e in sorted(cd.eval_dataset.unique()):
        sub = cd[cd.eval_dataset != e]
        rng = np.random.default_rng(SEED)
        loo[e] = summar(stratified(sub, tertiles(sub), "pava", 0, rng))
    loo_pass = sum(v["ordering_retained"] for v in loo.values())

    # severity-definition variants (pava, point verdicts)
    variants = {"full_composite": dmap}
    for m in METRICS:
        variants[f"only_{m}"] = severity_map(sev_rows, (m,))
        keep = tuple(x for x in METRICS if x != m)
        variants[f"without_{m}"] = severity_map(sev_rows, keep)
    var_res = {}
    for name, dm in variants.items():
        sub = attach_d(cells[cells.source == "cifar100"], dm)
        sub = sub.copy()
        sub["var_collapse"] = sub.cell.map(vc_new)
        rng = np.random.default_rng(SEED)
        var_res[name] = summar(stratified(sub, tertiles(sub), "pava", 0,
                                          rng))
    var_pass = sum(v["ordering_retained"] for v in var_res.values())

    gate = (est_pass == 4 and loo_pass >= 8 and var_pass >= 7)
    report = {
        "n_checkpoints": int(cd.cell.nunique()),
        "strata_sizes": {n: len(strata[n]) for n in NAMES},
        "estimators": est_res,
        "estimators_retained": f"{est_pass}/4",
        "leave_one_oodset": {e: {"retained": v["ordering_retained"],
                                 "strong_x": v["strong"]["x"]}
                             for e, v in loo.items()},
        "loo_retained": f"{loo_pass}/9",
        "severity_variants": {n: {"retained": v["ordering_retained"],
                                  "strong_x": v["strong"]["x"]}
                              for n, v in var_res.items()},
        "variants_retained": f"{var_pass}/9",
        "G11B": "PASS" if gate else "FAIL",
    }
    (OUT_DIR / "panelb_cifar100_report.json").write_text(
        json.dumps(report, indent=1, default=str))
    L = ["# Audit-11 R11.2: cifar100 case-study panel B (gate G11.B)", "",
         "Frozen spec in header; corrected panel; ordering verdicts by "
         "the frozen rule.", "", "```",
         json.dumps(report, indent=1, default=str), "```", ""]
    (OUT_DIR / "panelb_cifar100_report.md").write_text("\n".join(L))

    # hero panel-B export: primary pava curves with band q95 per stratum
    data, active, fine = make_data(cd)
    rng = np.random.default_rng(SEED)
    curves = {"fine": [round(float(x), 4) for x in fine]}
    for name in NAMES:
        sub = [c for c in active if c in strata[name]]
        g0 = curve("pava", data, sub, fine)
        devs = np.empty(B)
        for i in range(B):
            boot = list(rng.choice(sub, len(sub), replace=True))
            devs[i] = np.nanmax(np.abs(curve("pava", data, boot, fine)
                                       - g0))
        curves[name] = {
            "curve": [round(float(v), 4) for v in g0],
            "band_q95": round(float(np.quantile(devs, 0.95)), 4),
            "first_up_crossing": est_res["pava"][name]["x"],
        }
    (OUT_DIR / "panelb_cifar100_curves.json").write_text(
        json.dumps(curves, default=str))
    print(json.dumps({k: report[k] for k in
                      ("estimators_retained", "loo_retained",
                       "variants_retained", "G11B")}, indent=1))
    print("estimator primary (pava):",
          json.dumps(est_res["pava"], default=str))


if __name__ == "__main__":
    main()

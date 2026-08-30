"""NC1-normalization impact check 1: pooled collapse-tertile composition
and the tertile ordering under the corrected panel (option (c),
2026-08-30; feeds the next audit round; no manuscript change here).

Background: the frozen tertile rule pools all 280 checkpoints and takes
thirds of the parquet var_collapse (= Papyan NC1 / C). The /C factor is
per-source constant, so WITHIN-source orderings are unaffected up to the
pinv-conditioning noise between the two implementations, but the POOLED
tertile boundaries mix sources and can recompose under the corrected
panel (pilot0 NC1, the paper's stated definition).

FROZEN CHECK SPEC (declared before outcomes): identical frozen machinery
(build_cells, severity attach, tertile thirds rule, pava estimator,
B = 2000 simultaneous bands, seed 1071 for both runs); the corrected
panel is the pool_coords pilot0 papyan var_collapse. Reported:
- within-source Spearman between the two panels (pinv-noise gauge);
- 3x3 tertile membership overlap and per-tertile source composition;
- stratified first-up-crossings, tie regions, and the frozen
  ordering_retained verdict under BOTH panels (the old-panel run is the
  replication reference for -1.204 / -1.189 / none).

Usage (from code/): python nc1_tertile_check.py
Output: nc_csf_predictivity/outputs/track1/nc1_tertile_check.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crossing_robustness_audit import (METRICS, OUT_DIR, PARQUET,
                                       attach_d, build_cells,
                                       load_severity_rows, ordering_retained,
                                       severity_map, stratified, tertiles)
from heldout_theory_validation import load_coords

B, SEED = 2000, 1071


def main() -> None:
    cells = build_cells(pd.read_parquet(PARQUET))
    cells_d = attach_d(cells, severity_map(load_severity_rows(), METRICS))
    coords, _ = load_coords(Path("pilot0/pool_coords"))

    per = (cells_d.groupby("cell")
           .agg(source=("source", "first"), vc_old=("var_collapse", "first"))
           .reset_index())
    per["vc_new"] = [coords[c]["papyan"]["var_collapse"] for c in per.cell]

    within = {s: round(float(spearmanr(g.vc_old, g.vc_new).statistic), 4)
              for s, g in per.groupby("source")}

    old_t = tertiles(cells_d)
    cells_new = cells_d.copy()
    vc_map = dict(zip(per.cell, per.vc_new))
    cells_new["var_collapse"] = cells_new.cell.map(vc_map)
    new_t = tertiles(cells_new)

    names = ("strong", "middle", "weak")
    overlap = {f"{a}->{b}": len(old_t[a] & new_t[b])
               for a in names for b in names if len(old_t[a] & new_t[b])}
    retained = sum(len(old_t[n] & new_t[n]) for n in names)
    src_of = dict(zip(per.cell, per.source))

    def comp(t):
        return {n: dict(pd.Series([src_of[c] for c in t[n]])
                        .value_counts().sort_index()) for n in names}

    rng = np.random.default_rng(SEED)
    res_old = stratified(cells_d, old_t, "pava", B, rng)
    rng = np.random.default_rng(SEED)
    res_new = stratified(cells_d, new_t, "pava", B, rng)

    def summar(res):
        out = {}
        for k in ("pooled",) + names:
            r = res[k]
            out[k] = {"first_up_crossing": r.get("first_up_crossing"),
                      "tie_region": r.get("tie_region"),
                      "n_sign_changes": r.get("n_sign_changes")}
        out["ordering_retained"] = bool(ordering_retained(res))
        return out

    report = {
        "within_source_spearman_old_vs_new_panel": within,
        "tertile_membership_retained": f"{retained}/280",
        "membership_overlap_counts": overlap,
        "source_composition_old": comp(old_t),
        "source_composition_new": comp(new_t),
        "strata_old_panel_reference": summar(res_old),
        "strata_new_panel": summar(res_new),
    }
    (OUT_DIR / "nc1_tertile_check.json").write_text(
        json.dumps(report, indent=1, default=str))
    L = ["# NC1 impact check 1: tertile composition and ordering under "
         "the corrected panel", "",
         "Frozen machinery, seed 1071, B=2000; no manuscript change; "
         "feeds the next audit round.", "", "```",
         json.dumps(report, indent=1, default=str), "```", ""]
    (OUT_DIR / "nc1_tertile_check.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

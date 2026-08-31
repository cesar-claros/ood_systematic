"""Audit-11 R11.5 + R11.6: E2 held-out strata ordering, within-paradigm
crossings, and the joint confounding sensitivity, all re-run with
var_collapse substituted from the corrected (pilot0) panel. Local.

EVIDENCE CLASS: post-outcome confirmation runs (audit-11 rules). Only
the var_collapse column is substituted (the /K bug touches Sigma_W
only; self-duality, equinorm, and equiangularity are Sigma_W-free).
Machinery, seeds, folds, and rules are the frozen implementations,
imported unchanged: stage2_closure.e2_gate3 (bands B = 2000),
joint_confound_audit.run_joint/joint_summary (fold seed 20260826),
joint_confound_audit.paradigm_crossings. Frozen reference values are
restated from the registered reports for the delta comparison.

Usage (from code/): python nc1_joint_e2_check.py
Output: outputs/track1/nc1_joint_e2_check.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import load_coords
from joint_confound_audit import (add_metadata, joint_summary,
                                  paradigm_crossings, run_joint)
from stage2_closure import build_cells_with_severity, e2_gate3

FROZEN = {
    "joint_primary_M1_minus_M0plus": "+0.007 [-0.001, 0.017]",
    "e2": "retained cifar10/cifar100/tinyimagenet; reversed "
          "supercifar100; verdict INCONCLUSIVE",
    "paradigm_crossings": "RETAINED dg + devries; NOT-RETAINED confidnet "
                          "(left-censored middle)",
}


def main() -> None:
    cells = build_cells_with_severity()
    coords, _ = load_coords(Path("pilot0/pool_coords"))
    vc_new = {c: coords[c]["papyan"]["var_collapse"]
              for c in cells.cell.unique()}
    cells = cells.copy()
    cells["var_collapse"] = cells.cell.map(vc_new)

    meta = add_metadata(cells, coords)
    print("[nc1] E2 under corrected panel ...", flush=True)
    e2 = e2_gate3(meta)
    print("[nc1] paradigm crossings ...", flush=True)
    pc = paradigm_crossings(meta)
    print("[nc1] joint models ...", flush=True)
    fitted = run_joint(meta)
    js = joint_summary(fitted)
    joint_slim = {k: v for k, v in js.items()
                  if "M1" in str(k) or "M0" in str(k) or "macro" in str(k)
                  or "primary" in str(k)} or js
    report = {"frozen_reference": FROZEN,
              "e2_corrected": e2,
              "paradigm_crossings_corrected": pc,
              "joint_corrected": joint_slim}
    (OUT_DIR / "nc1_joint_e2_check.json").write_text(
        json.dumps(report, indent=1, default=str))
    L = ["# Audit-11 R11.5/R11.6: E2, paradigm crossings, joint audit "
         "under the corrected panel", "", "```",
         json.dumps(report, indent=1, default=str), "```", ""]
    (OUT_DIR / "nc1_joint_e2_check.md").write_text("\n".join(L))
    print(json.dumps(report, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()

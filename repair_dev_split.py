"""Frozen development/validation split for the mixture-aware anisotropic
repair campaign (analytic-saturation plan 2026-08-28, section 9.2,
"acceptable diagnostic: nested development split").

Committed BEFORE any repair specification is implemented, as the plan
requires ("divide the existing pool at the checkpoint level before
implementing alternatives"). The whole future exercise stays labeled post
hoc because the research question arose after observing the Stage-2
failure.

FROZEN RULE: the 280 pool checkpoints are split 50/50 into development
and validation at the CHECKPOINT level, stratified within source x
paradigm cells (odd cells give the extra checkpoint to development), by
one seeded permutation per cell; seed 20260830. No detector outcome, no
geometry value, and no coordinate enters the split. The split file's
sha256 is printed and the file is committed; every P00/P01/P10/P11
development decision uses the development half only, and the validation
half is evaluated once, after the single repaired specification is
frozen.

Usage (from code/): python repair_dev_split.py
Output: nc_csf_predictivity/outputs/track1/repair_dev_split.json
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from crossing_robustness_audit import OUT_DIR
from stage2_closure import build_cells_with_severity

SEED = 20260830


def main() -> None:
    cells = build_cells_with_severity()
    ckpt = (cells[["cell", "source"]].drop_duplicates("cell")
            .assign(paradigm=lambda f: f.cell.str.split("|").str[0]))
    rng = np.random.default_rng(SEED)
    dev, val = [], []
    for (_, _), g in ckpt.groupby(["source", "paradigm"]):
        names = sorted(g.cell)
        perm = rng.permutation(len(names))
        half = (len(names) + 1) // 2
        dev += [names[i] for i in perm[:half]]
        val += [names[i] for i in perm[half:]]
    out = {"seed": SEED, "rule": "50/50 checkpoint-level, stratified "
                                 "within source x paradigm; odd cells "
                                 "favor development",
           "n_dev": len(dev), "n_val": len(val),
           "development": sorted(dev), "validation": sorted(val)}
    path = OUT_DIR / "repair_dev_split.json"
    text = json.dumps(out, indent=1)
    path.write_text(text)
    print(f"dev {len(dev)} / val {len(val)}; sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()

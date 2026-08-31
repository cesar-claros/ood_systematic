"""ICML protocol roster C: manifest for the repair VALIDATION half
(140 checkpoints, seed-20260830 split; spent ONLY on the P10 level
endpoint E3 per the frozen protocol section 8.2). Local; metadata only;
no outcome consulted.

Usage (from code/): python make_repair_valhalf_manifest.py
Output: nc_csf_predictivity/outputs/track1/repair_valhalf_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import load_coords


def main() -> None:
    split = json.loads((OUT_DIR / "repair_dev_split.json").read_text())
    coords, problems = load_coords(Path("pilot0/pool_coords"))
    assert not problems, problems
    pool = []
    for cell in sorted(split["validation"]):
        rec = coords[cell]
        pool.append({"cell": cell, "model_path": rec["model_path"],
                     "source": rec["source"]})
    manifest = {"plan": "ICML retarget protocol section 8.2 roster C",
                "rule": "the full validation half of the seed-20260830 "
                        "repair split; P10 level endpoint (E3) only",
                "n_pool": len(pool), "pool": pool, "breeds": []}
    text = json.dumps(manifest, indent=1)
    path = OUT_DIR / "repair_valhalf_manifest.json"
    path.write_text(text)
    print(f"{len(pool)} validation-half checkpoints; sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()

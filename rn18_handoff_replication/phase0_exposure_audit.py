"""RN18 handoff-replication plan v3, PHASE 0: exposure audit (HPC;
metadata only; opens no logit, score, or outcome).

For every expected cell (manifests/expected_panel.json) resolve the
experiment dir under EXPERIMENT_ROOT_DIR (bare, then 'fd-shifts/'),
and record: existence, hydra/config.yaml, checkpoint files, native
test-output files (test_results/), the newest output mtime, and the
config's declared query_studies text. Assert that none of the four new
shifts (mnist, fashionmnist, kmnist, stl10) appears in any config.
Also list EXTRA ResNet-18 dirs not in the expected panel.

The ledger separates EXISTENCE from EXPOSURE: the human_inspection
fields stay null until the project owner fills them ('never_computed',
'computed_not_viewed', or 'viewed', with who and when). Unknown
exposure is not blindness (v2 section 2.1).

Usage (HPC, .env loaded, from code/):
    python rn18_handoff_replication/phase0_exposure_audit.py
Output: rn18_handoff_replication/manifests/exposure_ledger.json (rsync back)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

MANIFEST = Path("rn18_handoff_replication/manifests/expected_panel.json")
OUT = Path("rn18_handoff_replication/manifests/exposure_ledger.json")
PREFIXES = ("", "fd-shifts/")
NEW_SHIFT_TOKENS = ("mnist", "fashionmnist", "kmnist", "stl10")
NATIVE_OUTPUTS = ("raw_logits.npz", "raw_logits_dist.npz", "raw_output.npz",
                  "raw_output_dist.npz", "external_confids.npz",
                  "external_confids_dist.npz")


def audit_dir(d: Path) -> dict:
    rec = {"exists": d.is_dir(), "has_config": (d / "hydra" / "config.yaml").is_file(),
           "checkpoints": [], "native_outputs": [], "other_test_files": [],
           "newest_output_mtime": None, "query_studies_text": None,
           "new_shift_token_in_config": False}
    if not rec["exists"]:
        return rec
    for r, _dirs, files in os.walk(d):
        for f in files:
            if f.endswith(".ckpt") and "last" in f:
                rec["checkpoints"].append(str(Path(r, f).relative_to(d)))
    tr = d / "test_results"
    if tr.is_dir():
        mt = 0.0
        for f in sorted(tr.rglob("*")):
            if f.is_file():
                (rec["native_outputs"] if f.name in NATIVE_OUTPUTS
                 else rec["other_test_files"]).append(str(f.relative_to(d)))
                mt = max(mt, f.stat().st_mtime)
        rec["newest_output_mtime"] = (time.strftime("%Y-%m-%dT%H:%M:%S",
                                                    time.localtime(mt))
                                      if mt else None)
    if rec["has_config"]:
        text = (d / "hydra" / "config.yaml").read_text()
        m = re.search(r"query_studies:(.*?)(\n\S|\Z)", text, re.S)
        rec["query_studies_text"] = m.group(1).strip() if m else None
        low = text.lower()
        rec["new_shift_token_in_config"] = any(t in low for t in NEW_SHIFT_TOKENS)
    return rec


def main() -> None:
    root = Path(os.environ["EXPERIMENT_ROOT_DIR"])
    man = json.loads(MANIFEST.read_text())
    comps = {"paradigm_pool": [], "standalone_ce": []}
    n_found = 0
    for cell in man["cells"]:
        resolved = None
        for p in PREFIXES:
            d = root / p / cell["model_path"]
            if d.is_dir():
                resolved = d
                break
        rec = dict(cell, resolved_path=(str(resolved) if resolved else None))
        rec.update(audit_dir(resolved) if resolved else audit_dir(root / "__missing__"))
        assert not rec["new_shift_token_in_config"], (
            f"new-shift token in config of {cell['model_path']}")
        n_found += int(rec["exists"])
        comps[cell["component"]].append(rec)
    expected = {c["model_path"] for c in man["cells"]}
    extra = []
    for p in PREFIXES:
        for d in sorted((root / p).glob("*_paper_sweep/*_bbresnet18_*")):
            rel = str(d.relative_to(root / p))
            if rel not in expected and d.is_dir():
                extra.append(rel)
    ledger = {
        "plan": "RN18 handoff-replication plan v3, phase 0 exposure audit",
        "rule": "existence is not exposure; unknown exposure is not "
                "blindness; the four-shift outcomes of every cell below "
                "have never been computed by any registered pipeline "
                "(asserted: no new-shift token in any config)",
        "experiment_root": str(root),
        "components": {
            name: {"cells": cells,
                   "n_expected": len(cells),
                   "n_found": sum(c["exists"] for c in cells),
                   "n_with_config": sum(c["has_config"] for c in cells),
                   "n_with_checkpoint": sum(bool(c["checkpoints"]) for c in cells),
                   "n_with_native_outputs": sum(bool(c["native_outputs"]) for c in cells),
                   "human_inspection_four_shift_outcomes": None,
                   "human_inspection_registered_suite_outputs": None,
                   "declared_by": None, "declared_on": None,
                   "allowed_values": ["never_computed", "computed_not_viewed", "viewed"]}
            for name, cells in comps.items()},
        "extra_resnet18_dirs_not_in_panel": extra,
        "n_found_total": n_found,
    }
    text = json.dumps(ledger, indent=1)
    OUT.write_text(text)
    for name, c in ledger["components"].items():
        print(f"[phase0] {name}: found {c['n_found']}/{c['n_expected']}, "
              f"config {c['n_with_config']}, ckpt {c['n_with_checkpoint']}, "
              f"native outputs {c['n_with_native_outputs']}")
    print(f"[phase0] extra RN18 dirs not in panel: {len(extra)}")
    print(f"[phase0] wrote {OUT}; sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()}")
    print("[phase0] NEXT: rsync the ledger back; the project owner fills the "
          "human_inspection fields per component before phase 1 is read.")


if __name__ == "__main__":
    main()

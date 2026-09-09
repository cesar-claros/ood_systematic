"""RN18 handoff-replication plan v3, PHASE 5: the freeze artifact.
Writes FREEZE.json (+ FREEZE.md) binding, BEFORE the single readout:
the panel of record, the licensed endpoints with their multipliers and
alpha, every seed, the sha256 of every manifest / simulation artifact /
analysis module the readout depends on, the git HEAD, the expected
denominators, and the mechanical inventory (name, size, sha256) of the
still-unread extraction outputs. No output is parsed here.

Usage (from code/): python rn18_handoff_replication/phase5_freeze.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("rn18_handoff_replication")
CODE_FILES = [
    "rn18_handoff_replication/rn18_analysis.py", "rn18_handoff_replication/comparators.py",
    "rn18_handoff_replication/extract_fourshift_rn18.py", "rn18_handoff_replication/phase1_nc1_remeasure.py",
    "rn18_handoff_replication/phase1_leverage_gate.py", "rn18_handoff_replication/phase4_qualification.py",
    "rn18_handoff_replication/theory/etf_constructor.py", "rn18_handoff_replication/theory/gaussian_diagnostics.py",
    "rn18_handoff_replication/theory/coordinate_roundtrip.py",
    "icml_campaign_analysis.py", "crossing_robustness_audit.py", "heldout_theory_validation.py",
    "tail_space_audit.py", "mc_phase_audit.py", "pilot0/geometry.py", "pilot0/ood_coords.py",
    "pilot0/scores.py", "pilot0/repair_stats.py", "pilot0/extract_stage2_expansion.py",
    "pilot0/extract_roster_b_newshifts.py", "pilot0/clip_severity_v2.csv" if False else "pilot0/clip_severity_v2.csv",
]
MANIFESTS = ["rn18_handoff_replication/manifests/expected_panel.json",
             "rn18_handoff_replication/manifests/vgg_bridge_panel.json",
             "rn18_handoff_replication/manifests/exposure_ledger.json",
             "rn18_handoff_replication/simulations/design_manifest.json",
             "rn18_handoff_replication/simulations/development_critical_values.json",
             "rn18_handoff_replication/simulations/qualification_report.json",
             "rn18_handoff_replication/outputs/phase1_leverage_report.json"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inventory(d: Path) -> list[dict]:
    return [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(d.glob("*.json")) if not p.name.startswith("FAILED_")]


def main() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "rn18_handoff_replication", "icml_campaign_analysis.py"],
                           capture_output=True, text=True).stdout.strip()
    panel = json.loads(Path(MANIFESTS[0]).read_text())
    q = json.loads(Path(MANIFESTS[5]).read_text())
    crit = json.loads(Path(MANIFESTS[4]).read_text())
    rn18 = inventory(ROOT / "outputs/fourshift_rn18")
    vgg = inventory(ROOT / "outputs/fourshift_vgg_bridge")
    freeze = {
        "plan": "RN18 handoff-replication plan v3 (documentation/companion_phase_diagram_rn18_handoff_replication_plan_v3_2026-09-08.md)",
        "frozen_on": "2026-09-09", "git_head": head, "uncommitted_changes_in_scope": dirty.splitlines(),
        "panel_of_record": {"n_eligible": panel["panel_of_record_2026_09_09"]["n_eligible"],
                            "n_ineligible": panel["panel_of_record_2026_09_09"]["n_ineligible"],
                            "components": {"paradigm_pool": 56, "standalone_ce": 40},
                            "vgg_bridge": 20},
        "endpoints": {
            "HO": {"class": "registered, gate-based, no alpha", "panel": "full 96", "rule": "GR-3 rule + all single-shift deletions >= n-1 + every delete-one-checkpoint recomputation + informativeness; global at >= 3 of 4 sources under d^K, d^F reported", "min_stratum": 5, "bands_seed": 1211, "B": 2000},
            "SEL": {"class": "registered, alpha 0.025 (Bonferroni over 9 comparators)", "panel": "standalone CE, families = (dropout, run) blocks", "N_f": 10, "multiplier": crit["multipliers"]["SEL_Nf10"], "licensed": q["licenses"]["SEL_Nf10"]["licensed"], "reference": "vgg_matched_scalar_ridge", "epsilon": 0.002, "expected_class": "inferior to the reference or unresolved (superiority and equivalence have zero simulated power)"},
            "LEVEL": {"class": "registered, alpha 0.025", "panel": "standalone CE", "N_f": 10, "multiplier": crit["multipliers"]["LEVEL_Nf10"], "licensed": q["licenses"]["LEVEL_Nf10"]["licensed"], "margin": 0.01},
            "sensitivities": {"SEL/LEVEL dropout-off N_f=5": {"multipliers": [crit["multipliers"]["SEL_Nf5"], crit["multipliers"]["LEVEL_Nf5"]], "licensed": [q["licenses"]["SEL_Nf5"]["licensed"], q["licenses"]["LEVEL_Nf5"]["licensed"]]},
                              "SEL AUG-VIEW comparators": "descriptive", "ORG": "descriptive, no alpha, ce_do0 N_f=5 jackknife 95%, equivalence 0.003", "E4": "descriptive"}},
        "seeds": {"HO_bands": 1211, "phase1_bootstrap": 1201, "aug_view": 20260908, "balance_rng": 20260827,
                  "sim_dev": 2401, "sim_audit": 2402, "mc_master": 2201},
        "expected_denominators": {"rn18_records": 96, "rn18_cells": 384, "ce_families": 10, "ce_do0_families": 5,
                                  "vgg_bridge_records": 20, "sets_per_source": 4},
        "hashes": {"code": {f: sha(Path(f)) for f in CODE_FILES if Path(f).exists()},
                   "manifests": {f: sha(Path(f)) for f in MANIFESTS}},
        "unread_outputs": {"fourshift_rn18": {"n": len(rn18), "files": rn18},
                           "fourshift_vgg_bridge": {"n": len(vgg), "files": vgg},
                           "attestation": "inventory computed mechanically (name, size, sha256); no field parsed"},
        "first_reader": "rn18_handoff_replication/rn18_analysis.py, run exactly once after this freeze is committed",
    }
    text = json.dumps(freeze, indent=1)
    (ROOT / "FREEZE.json").write_text(text)
    md = ["# FREEZE (phase 5)", "", f"git HEAD `{head}`; frozen 2026-09-09; panel of record 96 (+20 VGG bridge).",
          f"Licensed: SEL N_f=10 m={crit['multipliers']['SEL_Nf10']}, LEVEL N_f=10 m={crit['multipliers']['LEVEL_Nf10']}; sensitivities at N_f=5 m=1.1.",
          f"Unread outputs: {len(rn18)} RN18 + {len(vgg)} VGG bridge records, inventory hashed.",
          f"FREEZE.json sha256 `{hashlib.sha256(text.encode()).hexdigest()}`", ""]
    (ROOT / "FREEZE.md").write_text("\n".join(md))
    print("\n".join(md))
    assert len(rn18) == 96 and len(vgg) == 20


if __name__ == "__main__":
    main()

"""Frozen Phase-1 subset manifest for the repair campaign (saturation
plan 2026-08-28, section 12 Phase 1). Local; deterministic; outcome-free.

FROZEN SELECTION RULE (declared before any repair statistic exists; no
detector outcome, score, gap, or winner is read here; the only measured
input is per-checkpoint variability collapse, a geometry coordinate):
- Pool checkpoints come EXCLUSIVELY from the frozen development half of
  the repair split (outputs/track1/repair_dev_split.json, seed 20260830;
  the validation half stays untouched). Per source x paradigm cell:
  the development checkpoints with the lowest and the highest
  var_collapse (ties broken by cell-key sort). Per source, additionally
  the DeepGamblers development checkpoint with median var_collapse
  (lower median, sorted index (n-1)//2, the stage-1 convention).
  Duplicates collapse. This spans strong/middle/weak collapse in every
  cell and covers all four sources and all three paradigms; coverage of
  both empirical Energy-CTM directions follows structurally from the
  known opposite source base rates and is NOT checked against outcomes.
- Lower-SNR external case: five BREEDS checkpoints by the stage-1
  deterministic rule on metadata only: dg do1 run1 at the lowest,
  median, and highest reward, plus confidnet do1 run1 rew2.2 and
  devries do1 run1 rew2.2, with the stage-1 substitution ladder
  (do1_run1 -> do0_run1 -> do1_run2 -> do0_run2) if a slug is absent.
  The roster is enumerated from the committed stage-2 coordinate
  filenames (metadata only; no record content is read).

Usage (from code/): python repair_phase1_manifest.py
Output: nc_csf_predictivity/outputs/track1/repair_phase1_manifest.json
        (sha256 printed; commit before extraction)
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import load_coords
from stage2_closure import build_cells_with_severity

BREEDS_RE = re.compile(
    r"^breeds_paper_sweep__(?P<paradigm>[a-z]+)_bbresnet50_do(?P<do>\d)"
    r"_run(?P<run>\d+)_rew(?P<rew>[\d.]+)\.json$")
LADDER = [(1, 1), (0, 1), (1, 2), (0, 2)]


def pool_selection(dev: set, coords: dict) -> list[dict]:
    cells = build_cells_with_severity()
    per_ckpt = (cells.groupby("cell")
                .agg(source=("source", "first"),
                     var_collapse=("var_collapse", "first"))
                .reset_index())
    per_ckpt["paradigm"] = per_ckpt.cell.str.split("|").str[0]
    per_ckpt = per_ckpt[per_ckpt.cell.isin(dev)].sort_values("cell")
    chosen: dict[str, dict] = {}

    def add(row, why):
        rec = coords.get(row.cell)
        assert rec is not None, f"no coords record for {row.cell}"
        entry = chosen.setdefault(row.cell, {
            "cell": row.cell, "model_path": rec["model_path"],
            "source": row.source, "paradigm": row.paradigm,
            "var_collapse": float(row.var_collapse), "why": []})
        entry["why"].append(why)

    for (_, _), g in per_ckpt.groupby(["source", "paradigm"]):
        g = g.sort_values(["var_collapse", "cell"],
                          kind="mergesort").reset_index(drop=True)
        add(g.iloc[0], "cell_min_collapse")
        add(g.iloc[-1], "cell_max_collapse")
    for _, g in per_ckpt[per_ckpt.paradigm == "dg"].groupby("source"):
        g = g.sort_values(["var_collapse", "cell"],
                          kind="mergesort").reset_index(drop=True)
        add(g.iloc[(len(g) - 1) // 2], "source_dg_median_collapse")
    return sorted(chosen.values(), key=lambda e: e["cell"])


def breeds_selection() -> list[dict]:
    slugs = {}
    for p in sorted(Path("pilot0/stage2_expansion_coords").glob(
            "breeds_*.json")):
        m = BREEDS_RE.match(p.name)
        if m:
            slugs[(m["paradigm"], int(m["do"]), int(m["run"]),
                   m["rew"])] = p.stem
    rewards = sorted({float(k[3]) for k in slugs if k[0] == "dg"})
    assert rewards, "no BREEDS dg checkpoints found"
    targets = [("dg", f"{rewards[0]:g}", "dg_lowest_reward"),
               ("dg", f"{rewards[(len(rewards) - 1) // 2]:g}",
                "dg_median_reward"),
               ("dg", f"{rewards[-1]:g}", "dg_highest_reward"),
               ("confidnet", "2.2", "confidnet"),
               ("devries", "2.2", "devries")]
    out = []
    for paradigm, rew, why in targets:
        for do, run in LADDER:
            key = (paradigm, do, run, rew)
            if key in slugs:
                out.append({"slug": slugs[key],
                            "model_path": slugs[key].replace("__", "/", 1)
                            .replace("breeds_paper_sweep/",
                                     "breeds_paper_sweep/"),
                            "paradigm": paradigm, "reward": rew,
                            "dropout": do, "run": run, "why": why})
                break
        else:
            out.append({"why": why, "paradigm": paradigm, "reward": rew,
                        "error": "no checkpoint on the ladder"})
    return out


def main() -> None:
    split = json.loads(
        (OUT_DIR / "repair_dev_split.json").read_text())
    dev = set(split["development"])
    coords, _ = load_coords(Path("pilot0/pool_coords"))
    pool = pool_selection(dev, coords)
    breeds = breeds_selection()
    manifest = {
        "plan": "analytic_saturation_diagnosis_and_repair_plan_2026-08-28 "
                "section 12 Phase 1",
        "rule": __doc__.split("FROZEN SELECTION RULE")[1]
                .split("Usage")[0].strip(),
        "dev_split_seed": split["seed"],
        "n_pool": len(pool), "n_breeds": len(breeds),
        "pool": pool, "breeds": breeds,
        "note": "no detector outcome consulted; var_collapse is the only "
                "measured input; validation half untouched",
    }
    for e in pool:
        assert e["cell"] in dev
    text = json.dumps(manifest, indent=1)
    path = OUT_DIR / "repair_phase1_manifest.json"
    path.write_text(text)
    counts = {}
    for e in pool:
        counts[(e["source"], e["paradigm"])] = counts.get(
            (e["source"], e["paradigm"]), 0) + 1
    print(f"pool {len(pool)} checkpoints "
          f"({sorted((f'{s}/{p}', n) for (s, p), n in counts.items())}); "
          f"breeds {sum('error' not in b for b in breeds)}/5; sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()

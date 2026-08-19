"""Stage 2b, local side: the committed directional-prediction table.

Aggregates the per-model JSONs from `stage2b_extract_predict.py` into the
per-(endpoint pair, arm, OOD set) predicted gap-change signs that the
manifest's E1/E2/E4 analyses test after unblinding. Gap convention:
gap = L_A - L_B with L = 1 - AUROC, so a NEGATIVE predicted delta means
"A gains on B" relative to the paired same-seed baseline. Materiality:
|mean paired delta| >= 2 * se_gap with se_gap = sqrt(se_A^2 + se_B^2),
Hanley-McNeil SEs at the baseline predicted AUROCs (n_id = 10000).

This table is COMMITTED before any detector outcome is computed
(manifest Addendum A item 3).

Usage (from code/):
    python nc_csf_predictivity/interventions/stage2b_signs.py \
        [--stage2b_dir nc_csf_predictivity/interventions/stage2b] \
        [--out nc_csf_predictivity/interventions/stage2b_predictions.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pilot0.theory import hanley_mcneil_se

PAIRS = {"E1": ("MLS", "Maha"), "E2": ("CTM_head", "CTM_mean"),
         "E4": ("Energy", "MLS")}
ARM_LABEL = {"-0.1": "A1-", "0.3": "A1+", "1.0": "A1++", "hard": "A2"}
N_ID = 10_000


def load_models(stage2b_dir: Path) -> list[dict]:
    records = [json.loads(p.read_text())
               for p in sorted(stage2b_dir.glob("*.json"))
               if not p.name.endswith("FAILED.json")]
    if not any(r["lam"] == "0.0" for r in records):
        raise ValueError("no baseline models found in stage2b dir")
    return records


def gap(preds: dict[str, float], pair: tuple[str, str]) -> float:
    a, b = pair
    return (1.0 - preds[a]) - (1.0 - preds[b])


def evaluate(records: list[dict], arm_name: str = "emp") -> dict:
    """Per (endpoint, arm, set): mean paired gap delta, sign, materiality."""
    baselines = {r["run"]: r for r in records if r["lam"] == "0.0"}
    set_names = sorted(next(iter(baselines.values()))["sets"])
    out: dict = {}
    for endpoint, pair in PAIRS.items():
        out[endpoint] = {}
        for lam, label in ARM_LABEL.items():
            arm_records = [r for r in records if r["lam"] == lam]
            if not arm_records:
                continue
            cells = {}
            for set_name in set_names:
                deltas = []
                for r in arm_records:
                    base = baselines.get(r["run"])
                    if base is None:
                        continue
                    deltas.append(
                        gap(r["sets"][set_name]["preds"][arm_name], pair)
                        - gap(base["sets"][set_name]["preds"][arm_name],
                              pair))
                base0 = next(iter(baselines.values()))
                n_ood = base0["sets"][set_name]["n_ood"]
                preds0 = base0["sets"][set_name]["preds"][arm_name]
                se_gap = float(np.sqrt(
                    hanley_mcneil_se(preds0[pair[0]], N_ID, n_ood) ** 2
                    + hanley_mcneil_se(preds0[pair[1]], N_ID, n_ood) ** 2))
                mean_delta = float(np.mean(deltas))
                cells[set_name] = {
                    "mean_delta": mean_delta,
                    "sign": int(np.sign(mean_delta)),
                    "material": bool(abs(mean_delta) >= 2.0 * se_gap),
                    "se_gap": se_gap, "n_seeds": len(deltas),
                }
            material = [c for c in cells.values() if c["material"]]
            signs = [c["sign"] for c in material]
            out[endpoint][label] = {
                "cells": cells,
                "n_material": len(material),
                "majority_sign": (int(np.sign(np.sum(signs)))
                                  if signs else 0),
            }
    return out


def render(table: dict, arm_name: str) -> str:
    lines = ["# Stage 2b: committed directional predictions "
             f"(frozen {arm_name} plug-in)", ""]
    lines.append("Gap = L_A - L_B (L = 1 - AUROC); negative delta = A "
                 "gains on B vs the paired baseline. Committed before any "
                 "detector outcome exists (manifest Addendum A item 3).")
    lines.append("")
    for endpoint, pair in PAIRS.items():
        lines.append(f"## {endpoint}: {pair[0]} vs {pair[1]}")
        lines.append("")
        lines.append("| arm | majority sign (material cells) "
                     "| material / total | per-set signed deltas |")
        lines.append("|---|---|---|---|")
        for label, rec in table[endpoint].items():
            arrow = {1: "+ (A loses ground)", -1: f"- ({pair[0]} gains)",
                     0: "none material"}[rec["majority_sign"]]
            per_set = "; ".join(
                f"{s.replace('ood_nsncs_', '').replace('ood_', '')} "
                f"{c['mean_delta']:+.4f}{'*' if c['material'] else ''}"
                for s, c in rec["cells"].items())
            lines.append(f"| {label} | {arrow} | {rec['n_material']}/"
                         f"{len(rec['cells'])} | {per_set} |")
        lines.append("")
    lines.append("`*` = material (|delta| >= 2 se_gap). Unblinding of "
                 "detector outcomes is permitted only after this file is "
                 "committed.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2b directional-prediction table")
    parser.add_argument(
        "--stage2b_dir", type=str,
        default="nc_csf_predictivity/interventions/stage2b")
    parser.add_argument(
        "--out", type=str,
        default="nc_csf_predictivity/interventions/stage2b_predictions.md")
    parser.add_argument("--arm", type=str, default="emp",
                        choices=["emp", "iso"])
    args = parser.parse_args()
    records = load_models(Path(args.stage2b_dir))
    table = evaluate(records, args.arm)
    Path(args.out).write_text(render(table, args.arm))
    json_out = Path(args.out).with_suffix(".json")
    json_out.write_text(json.dumps(table, indent=1))
    for endpoint, arms in table.items():
        summary = ", ".join(
            f"{label}: {rec['majority_sign']:+d}@{rec['n_material']}"
            for label, rec in arms.items())
        print(f"{endpoint}: {summary}")
    print(f"wrote {args.out} and {json_out}")


if __name__ == "__main__":
    main()

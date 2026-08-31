"""Audit-11 R11.4: NC1 pinv-fragility quantification (tinyimagenet
verdict; all Phase-1 checkpoints for context). Local; no new extraction.

Method: NC1 = Tr(Sigma_W Sigma_B^+)/C reduces exactly to the class-mean
span: with G = M M' (C x C Gram of centered class means) = R A_id (rows
scaled by radii) and S = M Sigma_W M' = R CROSS_id, eigendecompose
G = sum_i Lambda_i w_i w_i'; then NC1 = sum_i w_i' S w_i / Lambda_i^2
over the eigenvalues kept by the rcond rule (Lambda_i >= rcond *
Lambda_max). Both matrices are stored in the Phase-1 NPZ files, so the
sweep is exact and local.

FROZEN SPEC (declared before outcomes): rcond grid {1e-8, 1e-7, 1e-6,
1e-5, 1e-4, 1e-3}; sanity gate = the rcond 1e-6 value must match the
extractor's stored papyan var_collapse to relative 1e-3 (same
definition, span-reduced arithmetic); verdict tolerance (contract
amendment 2026-08-30): per checkpoint, ratio = max/min NC1 over the
grid; source verdict from the median ratio of its checkpoints:
ratio <= 2 -> stable; 2 < ratio <= 10 -> keep with range + caveat;
ratio > 10 -> drop tinyimagenet NC1-based statements.

Usage (from code/): python nc1_tinyimagenet_fragility.py
Output: outputs/track1/nc1_tinyimagenet_fragility.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crossing_robustness_audit import OUT_DIR

STATS_DIR = Path("pilot0/repair_phase1_stats")
RCONDS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def nc1_sweep(path: Path) -> dict:
    r = json.loads(path.read_text())
    z = np.load(path.with_suffix(".npz"))
    radii = z["id__radii"]
    g = radii[:, None] * z["id__A_id"]
    s_mm = radii[:, None] * z["id__CROSS_id"]
    g = (g + g.T) / 2.0
    s_mm = (s_mm + s_mm.T) / 2.0
    lam, w = np.linalg.eigh(g)
    lam, w = lam[::-1], w[:, ::-1]
    quad = np.einsum("ij,jk,ki->i", w.T, s_mm, w)
    vals = {}
    for rc in RCONDS:
        keep = lam >= rc * lam[0]
        vals[f"{rc:g}"] = float((quad[keep] / lam[keep] ** 2).sum())
    v = np.array(list(vals.values()))
    c = r["n_classes"]
    stored = r["papyan"]["var_collapse"]
    v6 = vals["1e-06"]
    return {"slug": r["slug"], "kind": r["kind"], "C": c, "D": r["dim"],
            "source": r["slug"].split("_paper_sweep")[0],
            "nc1_by_rcond": {k: round(x, 6) for k, x in vals.items()},
            "ratio_max_min": round(float(v.max() / v.min()), 3),
            "s_range": [round(float((c - 1) / np.sqrt(c * v.max())), 2),
                        round(float((c - 1) / np.sqrt(c * v.min())), 2)],
            "sanity_rel_dev_vs_stored": round(
                abs(v6 - stored) / stored, 6)}


def main() -> None:
    rows = [nc1_sweep(p) for p in sorted(STATS_DIR.glob("*.json"))
            if not p.name.startswith("FAILED")]
    sane = all(r["sanity_rel_dev_vs_stored"] < 1e-3 for r in rows)
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r["ratio_max_min"])
    verdicts = {}
    for s, v in by_src.items():
        med = float(np.median(v))
        verdicts[s] = {
            "median_ratio": round(med, 3),
            "max_ratio": round(float(np.max(v)), 3),
            "verdict": ("stable" if med <= 2 else
                        "keep_with_range_and_caveat" if med <= 10
                        else "drop")}
    report = {"sanity_all_match_stored_papyan": bool(sane),
              "per_source_verdicts": verdicts,
              "checkpoints": rows}
    (OUT_DIR / "nc1_tinyimagenet_fragility.json").write_text(
        json.dumps(report, indent=1, default=float))
    head = {k: report[k] for k in
            ("sanity_all_match_stored_papyan", "per_source_verdicts")}
    L = ["# Audit-11 R11.4: NC1 pinv-fragility (rcond sweep)", "", "```",
         json.dumps(head, indent=1), "```", ""]
    (OUT_DIR / "nc1_tinyimagenet_fragility.md").write_text("\n".join(L))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()

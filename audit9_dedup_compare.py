"""Audit-9 duplicate-exclusion sensitivity comparison (submission-close
plan sections 6.3-6.4 and gates A/B). Local; consumes the dedup
re-extractions.

Compares the frozen baseline records against the dedup records (identical
pipeline; the certificate's duplicate ID-test images excluded before
scoring). Analytic margins are train-fit and unchanged by construction;
they are reused from the baseline side. Reports per-checkpoint changes in
every reported metric, the recomputed BREEDS rank statistics (Spearman,
checkpoint bootstrap B=2000 seed 941, permutation B=10000 seed 991,
leave-one-reward influence), and the ImageNet-200 qualitative check (sign
pattern and the CTM-advantage-vs-gamma*a trend). Prints the gate verdicts:

Gate A (BREEDS): PASS iff direction and material-cell conclusions are
unchanged, the point correlation stays materially positive, the bootstrap
lower bound stays above zero, and the reward-influence conclusion holds.
Gate B (ImageNet-200): PASS iff the sign pattern and the increasing CTM
advantage remain unchanged.

Usage (from code/): python audit9_dedup_compare.py
Inputs: pilot0/stage2_expansion_coords[_dedup]/breeds*.json
        pilot0/stage3_imagenet200_coords[_dedup]/*.json
Output: nc_csf_predictivity/outputs/track1/audit9_dedup_report.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audit8_checks import expansion_cells
from crossing_robustness_audit import OUT_DIR
from pilot0.extract_stage2_expansion import SCORE_NAMES


def rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def load(dirpath: str, pattern: str) -> dict:
    out = {}
    for p in sorted(Path(dirpath).glob(pattern)):
        if p.name.startswith("FAILED"):
            continue
        r = json.loads(p.read_text())
        out[r.get("slug") or r.get("run")] = r
    return out


def breeds_compare() -> dict:
    base = load("pilot0/stage2_expansion_coords", "breeds*.json")
    dedup = load("pilot0/stage2_expansion_coords_dedup", "breeds*.json")
    assert len(dedup) == len(base) == 28, (len(base), len(dedup))
    margins = {r.slug: r.margin for r in
               expansion_cells().query("source=='breeds'").itertuples()}
    rows, max_changes = [], {}
    for slug, b in base.items():
        d = dedup[slug]
        eb = list(b["ood"].values())[0]
        ed = list(d["ood"].values())[0]
        changes = {"id_error": abs(b["iid_test"]["id_error_rate"]
                                   - d["iid_test"]["id_error_rate"]),
                   "gap_balanced": abs(eb["gap_balanced"]
                                       - ed["gap_balanced"])}
        for s in SCORE_NAMES:
            changes[f"auroc_{s}"] = abs(eb[f"auroc_id_vs_ood_{s}"]
                                        - ed[f"auroc_id_vs_ood_{s}"])
        for k, v in changes.items():
            max_changes[k] = max(max_changes.get(k, 0.0), v)
        rows.append({"slug": slug, "paradigm": b["paradigm"],
                     "reward": float(b["reward"]),
                     "margin": margins[slug],
                     "gap_base": eb["gap_balanced"],
                     "gap_dedup": ed["gap_balanced"],
                     "mat_base": eb["material"], "mat_dedup": ed["material"],
                     "n_excluded": d.get("n_idtest_excluded")})
    sign_flips = sum(1 for r in rows
                     if np.sign(r["gap_base"]) != np.sign(r["gap_dedup"]))
    mat_changes = sum(1 for r in rows if r["mat_base"] != r["mat_dedup"])
    x = np.array([r["margin"] for r in rows])
    y = np.array([abs(r["gap_dedup"]) for r in rows])
    point = rho(x, y)
    rng = np.random.default_rng(941)
    boots = np.array([rho(x[i], y[i]) for i in
                      (rng.choice(28, 28, replace=True)
                       for _ in range(2000))])
    rngp = np.random.default_rng(991)
    perms = np.array([rho(rngp.permutation(x), y) for _ in range(10000)])
    p_perm = float((1 + (perms >= point).sum()) / 10001)
    infl = {}
    for rw in sorted({r["reward"] for r in rows
                      if r["paradigm"] == "dg"}):
        keep = [i for i, r in enumerate(rows)
                if not (r["paradigm"] == "dg" and r["reward"] == rw)]
        infl[f"drop_rew{rw:g}"] = round(rho(x[keep], y[keep]), 3)
    ci = [round(float(np.quantile(boots, 0.025)), 3),
          round(float(np.quantile(boots, 0.975)), 3)]
    n_mat = sum(r["mat_dedup"] for r in rows)
    pos = sum(1 for r in rows if r["mat_dedup"] and r["gap_dedup"] > 0)
    gate_a = (sign_flips == 0 and mat_changes == 0 and point > 0.5
              and ci[0] > 0
              and all(v > 0.5 for v in infl.values()))
    return {"n_excluded_per_ckpt": rows[0]["n_excluded"],
            "max_changes": {k: round(v, 5) for k, v in max_changes.items()},
            "winner_sign_flips": sign_flips,
            "materiality_changes": mat_changes,
            "n_material_dedup": int(n_mat),
            "frac_positive_material_dedup": (round(pos / n_mat, 3)
                                             if n_mat else None),
            "spearman_dedup": round(point, 3), "boot_ci95": ci,
            "perm_p": round(p_perm, 5),
            "leave_one_reward_rho": infl,
            "GATE_A": "PASS" if gate_a else "REVIEW"}


def imagenet200_compare() -> dict:
    base = load("pilot0/stage3_imagenet200_coords", "*.json")
    dedup = load("pilot0/stage3_imagenet200_coords_dedup", "*.json")
    assert len(dedup) == len(base) == 3
    rows, max_change = [], 0.0
    trend_ok, sign_same = True, True
    for run, b in base.items():
        d = dedup[run]
        gaps_b, gaps_d, gas = [], [], []
        for name, eb in b["ood"].items():
            ed = d["ood"][name]
            max_change = max(max_change, abs(eb["gap_balanced"]
                                             - ed["gap_balanced"]))
            if abs(eb["gap_balanced"]) > 5e-4 or abs(
                    ed["gap_balanced"]) > 5e-4:
                if np.sign(eb["gap_balanced"]) != np.sign(
                        ed["gap_balanced"]):
                    sign_same = False
            gaps_d.append(ed["gap_balanced"])
            gas.append(ed["gamma"] * ed["a"])
        if rho(gas, gaps_d) <= 0:
            trend_ok = False
        rows.append({"run": run, "n_excluded": d.get("n_idtest_excluded"),
                     "trend_rho_ga_vs_gap": round(rho(gas, gaps_d), 3)})
    gate_b = sign_same and trend_ok
    return {"per_run": rows, "max_gap_change": round(max_change, 5),
            "sign_pattern_unchanged": sign_same,
            "ctm_advantage_trend_retained": trend_ok,
            "GATE_B": "PASS" if gate_b else "REVIEW"}


def main() -> None:
    out = {"breeds": breeds_compare(),
           "imagenet200": imagenet200_compare()}
    L = ["# Audit-9 duplicate-exclusion sensitivity (gates A and B)", "",
         "Baseline vs dedup re-extraction; analytic margins are train-fit "
         "and unchanged by construction.", "",
         "## BREEDS (Gate A)", "```",
         json.dumps(out["breeds"], indent=1), "```", "",
         "## ImageNet-200 (Gate B)", "```",
         json.dumps(out["imagenet200"], indent=1), "```", ""]
    (OUT_DIR / "audit9_dedup_report.md").write_text("\n".join(L))
    (OUT_DIR / "audit9_dedup_report.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""TARGET-ALIGNED AUDIT (post-outcome, descriptive): the same detector
comparison on three prediction targets (recommendation A of the 2026-09-11
evaluation, finding F1).

Targets, all as CTM-minus-Energy gaps with positive favoring CTM:
  dA  = ID-versus-OOD AUROC gap (the theory's and LEVEL's target);
  dF  = failure-detection AUROC gap (correct-ID vs incorrect-ID + OOD);
  dG  = AUGRC gap (Energy - CTM), the registered E1/E2/HO target; affine in
        dF at fixed failure prevalence pi (Traub et al. 2024, Prop. in the paper).
Per-example arrays of the version-2 ResNet-18 panel give dF directly (raw and
prevalence-balanced); for the new-shift VGG roster (rounded version-1
records) dF is recovered exactly from AUGRC and pi = (n_id e + n_ood)/(n_id + n_ood).

Reports: (1) sign disagreement among targets, all cells and per-target
material cells (|gap| >= 0.01 in its own units); (2) frozen-arm winner
accuracy per target, with the always-CTM constant policy; (3) the handoff
gate (unchanged code, b = 0: the verdict does not depend on b) recomputed
per target on the ResNet-18 panel and the E2 ordering on the VGG roster;
(4) the E1 contrast recomputed on the AUROC target with the severity
baseline refitted (checkpoint folds and source folds; checkpoint bootstrap
with refitting, B = 500). Registered results are untouched.

Usage (from code/): python rn18_handoff_replication/target_aligned_audit.py
Output: rn18_handoff_replication/outputs/target_aligned_audit.json/.md
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from heldout_theory_validation import MATERIALITY, accuracy
from icml_campaign_analysis import frozen_margin, record_params, run_folds_e1, severity_axes
from rn18_handoff_replication import rn18_analysis as v1
from rn18_handoff_replication.rn18_analysis_v2 import choice_prob, regret

OUT = Path("rn18_handoff_replication/outputs")
V2 = OUT / "fourshift_v2_rn18"
VGG_GLOB = "pilot0/icml_roster_b_coords/*.json"
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
TARGETS = ("dA", "dF", "dG")
LABEL = "TARGET-ALIGNED AUDIT (post-outcome, descriptive)"
MAT = 0.01


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.concatenate([neg, pos]); r = rankdata(x); n0, n1 = len(neg), len(pos)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def failure_auroc_from_augrc(augrc: float, pi: float) -> float:
    return 1.0 - (augrc - pi * pi / 2) / (pi * (1 - pi))


# ---------------------------------------------------------------------------
# Cell tables.
# ---------------------------------------------------------------------------

def rn18_cells(axes: dict) -> pd.DataFrame:
    rows = []
    for p in sorted(V2.glob("*.json")):
        if p.name.startswith("FAILED_"):
            continue
        r = json.loads(p.read_text()); z = np.load(V2 / f"{p.stem}.npz")
        c, d, s, theta, logit, eta = record_params(r)
        correct = z["id__correct"].astype(bool); e = float(1 - correct.mean())
        fam = (f"do{int(r['dropout'])}_run{int(r['run_label'])}" if r["component"] == "standalone_ce" else "paradigm_block")
        for sname in SHIFTS:
            o = r["ood"][sname]; u = r["v2"]["ood_unrounded"][sname]
            l_e, l_c = frozen_margin(r, o)
            row = dict(cell=r["slug"], component=r["component"], paradigm=r["paradigm"], dropout=int(r["dropout"]), source=r["source"],
                       ood_set=sname, family=fam, nc1=float(r["papyan"]["var_collapse"]), g=np.log(float(r["papyan"]["var_collapse"])),
                       dK=axes[(r["source"], sname, "dK")], dF_axis=axes[(r["source"], sname, "dF")], M=l_e - l_c, id_error=e,
                       pi_raw=u["pi_raw"], pi_bal=u["pi_balanced"])
            for sc in ("Energy", "CTM"):
                sid, sood = z[f"id__score__{sc}"], z[f"set__{sname}__score__{sc}"]
                idx_id, idx_ood = z[f"set__{sname}__balance_id_idx"], z[f"set__{sname}__balance_ood_idx"]
                row[f"A_{sc}"] = u["auroc_id_vs_ood"][sc]
                row[f"F_{sc}"] = auroc(sid[correct], np.concatenate([sid[~correct], sood]))
                cb = correct[idx_id]; sid_b, sood_b = sid[idx_id], sood[idx_ood]
                row[f"Fbal_{sc}"] = auroc(sid_b[cb], np.concatenate([sid_b[~cb], sood_b]))
                row[f"G_{sc}"] = u["augrc_raw"][sc]; row[f"Gbal_{sc}"] = u["augrc_balanced"][sc]
            row["dA"] = row["A_CTM"] - row["A_Energy"]; row["dF"] = row["F_CTM"] - row["F_Energy"]; row["dFbal"] = row["Fbal_CTM"] - row["Fbal_Energy"]
            row["dG"] = row["G_Energy"] - row["G_CTM"]; row["dGbal"] = row["Gbal_Energy"] - row["Gbal_CTM"]
            # identity check: dG = pi(1-pi)(F_CTM - F_Energy)
            row["identity_resid"] = row["dG"] - row["pi_raw"] * (1 - row["pi_raw"]) * row["dF"]
            rows.append(row)
    return pd.DataFrame(rows)


def vgg_cells(axes: dict) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(VGG_GLOB)):
        r = json.load(open(p))
        if "dim" not in r:
            pc = Path("pilot0/pool_coords") / f"{r['model_path'].replace('/', '__')}.json"
            r["dim"] = int(json.loads(pc.read_text())["dim"])
        e = r["iid_test"].get("id_error_rate")
        run = int(re.search(r"_run(\d+)_", r["model_path"]).group(1))
        for sname, o in r["ood"].items():
            if sname not in SHIFTS or "error" in o:
                continue
            l_e, l_c = frozen_margin(r, o)
            row = dict(cell=r["slug"], source=r["source"], ood_set=sname, family=f"run{run}", nc1=float(r["papyan"]["var_collapse"]),
                       g=np.log(float(r["papyan"]["var_collapse"])), dK=axes[(r["source"], sname, "dK")], dF_axis=axes[(r["source"], sname, "dF")],
                       M=l_e - l_c, id_error=e, n_id=o["n_id"], n_ood=o["n_ood"])
            row["dA"] = o["auroc_id_vs_ood_CTM"] - o["auroc_id_vs_ood_Energy"]; row["dG"] = o["gap_raw"]; row["dGbal"] = o["gap_balanced"]
            if e is not None:
                pi = (o["n_id"] * e + o["n_ood"]) / (o["n_id"] + o["n_ood"])
                row["pi_raw"] = pi
                row["dF"] = failure_auroc_from_augrc(o["augrc_raw_CTM"], pi) - failure_auroc_from_augrc(o["augrc_raw_Energy"], pi)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries.
# ---------------------------------------------------------------------------

def disagreement(df: pd.DataFrame, targets) -> dict:
    out = {}
    for a in targets:
        for b in targets:
            if a >= b or a not in df or b not in df:
                continue
            m = np.isfinite(df[a]) & np.isfinite(df[b])
            opp = int((np.sign(df[a][m]) != np.sign(df[b][m])).sum())
            mat_a = m & (df[a].abs() >= MAT); mat_b = m & (df[b].abs() >= MAT)
            out[f"{a}_vs_{b}"] = {"n": int(m.sum()), "opposite": opp, "opposite_frac": opp / max(int(m.sum()), 1),
                                  f"opposite_among_{a}_material": [int((np.sign(df[a][mat_a]) != np.sign(df[b][mat_a])).sum()), int(mat_a.sum())],
                                  f"opposite_among_{b}_material": [int((np.sign(df[a][mat_b]) != np.sign(df[b][mat_b])).sum()), int(mat_b.sum())]}
    return out


def winner_accuracy(df: pd.DataFrame, targets) -> dict:
    out = {}
    p00 = choice_prob(df.M.values)
    for t in targets:
        if t not in df:
            continue
        y = df[t].values; ok = np.isfinite(y)
        mat = ok & (np.abs(y) >= MAT)
        out[t] = {"frozen_arm_all_nonzero": accuracy(df.M.values[ok & (y != 0)], y[ok & (y != 0)]), "n_all_nonzero": int((ok & (y != 0)).sum()),
                  "frozen_arm_material": (accuracy(df.M.values[mat], y[mat]) if mat.any() else None), "n_material": int(mat.sum()),
                  "always_ctm_all": float(np.mean(y[ok] > 0)), "always_ctm_material": (float(np.mean(y[mat] > 0)) if mat.any() else None),
                  "frozen_arm_regret": float(regret(y[ok], p00[ok]).mean()), "always_ctm_regret": float(regret(y[ok], np.ones(ok.sum())).mean()),
                  "always_energy_regret": float(regret(y[ok], np.zeros(ok.sum())).mean()), "p00_tie_fraction": float(np.mean(p00[ok] == 0.5))}
    return out


def ho_per_target(df: pd.DataFrame, targets) -> dict:
    out = {}
    for t in targets:
        if t not in df:
            continue
        sub = df.copy(); sub["dG"] = sub[t]; sub["dF"] = sub["dF_axis"]
        out[t] = {ax: {s: {"verdict": (r := v1.ho_source(g, ax, 0))["verdict"], "informative": r.get("informative"),
                          "crossings": ({k: c.get("first_up_crossing") for k, c in r["full_suite"].items()} if "full_suite" in r else None)}
                       for s, g in sub.groupby("source")} for ax in ("dK", "dF")}
    return out


def e1_target(df: pd.DataFrame, target: str, b: int, seed: int = 4401) -> dict:
    """E1 statistic (theory minus refitted severity, material cells of the target) with
    checkpoint-bootstrap refitting. Uses the frozen fold code with gap := 1000 * target."""
    cells = df.rename(columns={"ood_set": "eval_dataset"})[["cell", "source", "eval_dataset", "M", "dK", target]].copy()
    cells = cells.rename(columns={"M": "m"}); cells["gap"] = 1000.0 * cells[target]; cells["d"] = cells["dK"]
    cells = cells.dropna(subset=["gap"]).reset_index(drop=True)

    def stat(cf: pd.DataFrame, mode: str) -> float:
        folded = run_folds_e1(cf, mode)
        m = folded[np.abs(folded.gap) >= MATERIALITY]
        return accuracy(m.m.values, m.gap.values) - accuracy(m.severity.values, m.gap.values) if len(m) else float("nan")
    out = {}
    rng = np.random.default_rng(seed)
    ck = np.array(sorted(cells.cell.unique()))
    for mode in ("ckpt5", "loso"):
        point = stat(cells, mode)
        boots = []
        for i in range(b):
            draw = rng.choice(ck, len(ck), replace=True)
            parts = [cells[cells.cell == c].assign(cell=f"{c}#b{j}") for j, c in enumerate(draw)]
            boots.append(stat(pd.concat(parts, ignore_index=True), mode))
        boots = np.array(boots)
        folded = run_folds_e1(cells, mode); m = folded[np.abs(folded.gap) >= MATERIALITY]
        out[mode] = {"theory_acc": accuracy(m.m.values, m.gap.values), "severity_acc": accuracy(m.severity.values, m.gap.values), "n_material": int(len(m)),
                     "diff_point": point, "diff_ci95_refit_bootstrap": [float(np.nanquantile(boots, .025)), float(np.nanquantile(boots, .975))], "B": b}
    return out


def main() -> None:
    axes = v1.severity_axes()
    rn = rn18_cells(axes); vg = vgg_cells(axes)
    ce = rn[rn.component == "standalone_ce"]
    rep = {"label": LABEL, "materiality": MAT,
           "identity_check_rn18_max_abs_resid": float(np.abs(rn.identity_resid).max()),
           "rn18": {"n_cells": int(len(rn)), "id_error_range": [float(rn.id_error.min()), float(rn.id_error.max())],
                    "pi_raw_range": [float(rn.pi_raw.min()), float(rn.pi_raw.max())],
                    "disagreement_all": disagreement(rn, ("dA", "dF", "dG")), "disagreement_ce": disagreement(ce, ("dA", "dF", "dG")),
                    "winner_accuracy_all": winner_accuracy(rn, TARGETS), "winner_accuracy_ce": winner_accuracy(ce, TARGETS),
                    "ho_per_target": ho_per_target(rn, ("dG", "dA", "dF", "dGbal", "dFbal"))},
           "vgg_new_shifts": {"n_cells": int(len(vg)), "id_error_available": bool(vg.id_error.notna().all()),
                              "disagreement": disagreement(vg, tuple(t for t in ("dA", "dF", "dG") if t in vg)),
                              "winner_accuracy": winner_accuracy(vg, tuple(t for t in TARGETS if t in vg)),
                              "e2_per_target": ho_per_target(vg.assign(component="vgg", paradigm="confidnet", dropout=0), tuple(t for t in ("dG", "dA", "dF") if t in vg)),
                              "e1_registered_target_dG_reproduced": e1_target(vg, "dG", b=500),
                              "e1_on_dA": e1_target(vg, "dA", b=500),
                              **({"e1_on_dF": e1_target(vg, "dF", b=500)} if "dF" in vg else {})}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "target_aligned_audit.json").write_text(json.dumps(rep, indent=1, default=str))
    small = json.loads(json.dumps(rep, default=str)); small["rn18"]["ho_per_target"] = {t: {ax: {s: v["verdict"] for s, v in d.items()} for ax, d in tt.items()} for t, tt in rep["rn18"]["ho_per_target"].items()}
    small["vgg_new_shifts"]["e2_per_target"] = {t: {ax: {s: (v["verdict"], v["informative"]) for s, v in d.items()} for ax, d in tt.items()} for t, tt in rep["vgg_new_shifts"]["e2_per_target"].items()}
    (OUT / "target_aligned_audit.md").write_text("# Target-aligned audit\n\n```\n" + json.dumps(small, indent=1, default=str) + "\n```\n")
    print(json.dumps(small, indent=1, default=str))


if __name__ == "__main__":
    main()

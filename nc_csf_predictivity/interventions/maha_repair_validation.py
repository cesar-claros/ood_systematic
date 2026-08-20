"""Maha amplitude-operator repair: validation on Pilot 1 (local side).

B-axis protocol section 7 prerequisite, run on the outputs of
`maha_repair_extract.py` plus the Pilot 1 stats CSVs. Pilot 1 outcomes
(A2 included) are DESIGN DATA (audit section 7): everything chosen here
is frozen into the fresh-confirmation manifest before any fresh model is
scored.

Three sections:
  1. Operator comparison. Paired Maha responses dL = L(arm) - L(baseline)
     per (model, OOD set): observed vs the frozen closed form (pred_old)
     vs the min-statistic repair (pred_min). Reported per group (A1 pool,
     A2): means, MAEs, amplification factors. The repair succeeds
     structurally if it shrinks the A2 over-prediction and does not
     degrade A1.
  2. Mechanism diagnostics (audit 5.4): sampled vs empirical
     nearest-prototype switching rates and score moments (Gaussianity
     check), plus the extraction sanity check (rank-AUROC from actual
     features vs the pipeline's observed AUROC_f).
  3. Bounded response calibration. Four forms fit on the A1 cells and
     evaluated on A2, input = the repaired predictor's raw response:
     linear (a + bR), slope (bR), cap (A tanh(R/A)), slope_cap
     (A tanh(bR/A)). Guard: a form is admissible only if its within-A1
     leave-one-dose-out CV MAE beats the no-change predictor. Selection:
     lowest A2 MAE among admissible forms. The selected form is the
     candidate for the fresh-confirmation manifest.

Usage (from code/):
    python nc_csf_predictivity/interventions/maha_repair_validation.py \
        --stats_root $EXPERIMENT_ROOT_DIR/cifar100_intervention
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nc_csf_predictivity.interventions.outcome_analysis import (
    _loss,
    load_long,
)

ARM_LAMS = {"A1-": "-0.1", "A1+": "0.3", "A1++": "1.0", "A2": "hard"}
A1_LABELS = ("A1-", "A1+", "A1++")
BASE_LAM = "0.0"


def load_repair(repair_dir: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for path in sorted(repair_dir.glob("*.json")):
        if path.name.endswith("FAILED.json"):
            continue
        rec = json.loads(path.read_text())
        out[(rec["lam"], int(rec["run"]))] = rec
    if not out:
        raise FileNotFoundError(f"no repair JSONs in {repair_dir}")
    return out


def build_cells(table: pd.DataFrame, repair: dict) -> pd.DataFrame:
    """One row per (arm label, run, set): observed and predicted paired
    Maha responses dL = L(arm) - L(baseline); dL_pred = a_base - a_arm."""
    runs = sorted({r for _, r in repair})
    rows = []
    for label, lam in ARM_LAMS.items():
        for run in runs:
            arm, base = repair.get((lam, run)), repair.get((BASE_LAM, run))
            if arm is None or base is None:
                continue
            for set_name, s_arm in arm["sets"].items():
                s_base = base["sets"][set_name]
                d_obs = (_loss(table, lam, run, set_name, "Maha", "auroc_f")
                         - _loss(table, BASE_LAM, run, set_name, "Maha",
                                 "auroc_f"))
                rows.append({
                    "label": label, "lam": lam, "run": run,
                    "set_name": set_name, "d_obs": d_obs,
                    "d_old": s_base["pred_old"] - s_arm["pred_old"],
                    "d_min": s_base["pred_min"] - s_arm["pred_min"],
                    "emp_auroc_arm": s_arm["emp_auroc"],
                    "obs_auroc_arm": 1.0 - _loss(table, lam, run, set_name,
                                                 "Maha", "auroc_f"),
                })
    return pd.DataFrame(rows)


def operator_comparison(cells: pd.DataFrame) -> dict:
    out = {}
    for group, labels in (("A1", A1_LABELS), ("A2", ("A2",))):
        sub = cells[cells.label.isin(labels)]
        rec = {"n_cells": len(sub)}
        for col in ("d_obs", "d_old", "d_min"):
            rec[f"mean_{col}"] = float(sub[col].mean())
        for op in ("d_old", "d_min"):
            rec[f"mae_{op}"] = float((sub[op] - sub["d_obs"]).abs().mean())
            denom = rec["mean_d_obs"]
            rec[f"factor_{op}"] = (float(rec[f"mean_{op}"] / denom)
                                   if abs(denom) > 1e-12 else float("nan"))
        rec["repair_improves"] = bool(rec["mae_d_min"] < rec["mae_d_old"])
        out[group] = rec
    out["extraction_sanity_corr"] = float(np.corrcoef(
        cells["emp_auroc_arm"], cells["obs_auroc_arm"])[0, 1])
    return out


def diagnostics_summary(repair: dict) -> dict:
    out = {}
    for label, lam in (("baseline", BASE_LAM), *ARM_LAMS.items()):
        recs = [r for (l, _), r in repair.items() if l == lam]
        if not recs:
            continue
        mc_switch = [s["mc_diag"]["id_switch_rate"]
                     for r in recs for s in r["sets"].values()]
        emp_switch = [r["id_test"]["emp_switch_rate"] for r in recs]
        gap = [abs(s["mc_diag"]["ood_score_mean"] - s["emp_ood_score_mean"])
               / max(abs(s["emp_ood_score_mean"]), 1e-9)
               for r in recs for s in r["sets"].values()]
        out[label] = {
            "median_mc_id_switch": float(np.median(mc_switch)),
            "median_emp_id_switch": float(np.median(emp_switch)),
            "median_ood_mean_rel_gap": float(np.median(gap)),
        }
    return out


FORMS = ("linear", "slope", "cap", "slope_cap")


def _predict(form: str, params: np.ndarray, r: np.ndarray) -> np.ndarray:
    if form == "linear":
        return params[0] + params[1] * r
    if form == "slope":
        return params[0] * r
    if form == "cap":
        a = abs(params[0]) + 1e-9
        return a * np.tanh(r / a)
    a = abs(params[0]) + 1e-9
    return a * np.tanh(params[1] * r / a)


def _fit(form: str, r: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import least_squares
    x0 = {"linear": [0.0, 1.0], "slope": [1.0],
          "cap": [0.1], "slope_cap": [0.1, 1.0]}[form]
    res = least_squares(lambda p: _predict(form, p, r) - y, x0,
                        method="lm", max_nfev=10_000)
    return res.x


def calibration(cells: pd.DataFrame, input_col: str = "d_min") -> dict:
    a1 = cells[cells.label.isin(A1_LABELS)]
    a2 = cells[cells.label == "A2"]
    out: dict = {"input": input_col, "forms": {}}
    no_change_a1 = float(a1["d_obs"].abs().mean())
    no_change_a2 = float(a2["d_obs"].abs().mean())
    out["no_change"] = {"a1_mae": no_change_a1, "a2_mae": no_change_a2}
    for form in FORMS:
        params = _fit(form, a1[input_col].values, a1["d_obs"].values)
        cv_errs = []
        for held in A1_LABELS:
            tr = a1[a1.label != held]
            te = a1[a1.label == held]
            p = _fit(form, tr[input_col].values, tr["d_obs"].values)
            cv_errs.extend(np.abs(_predict(form, p, te[input_col].values)
                                  - te["d_obs"].values))
        pred_a2 = _predict(form, params, a2[input_col].values)
        out["forms"][form] = {
            "params": [float(v) for v in params],
            "a1_cv_mae": float(np.mean(cv_errs)),
            "admissible": bool(np.mean(cv_errs) <= no_change_a1),
            "a2_mae": float(np.abs(pred_a2 - a2["d_obs"].values).mean()),
            "a2_sign_agreement": float(
                np.mean(np.sign(pred_a2) == np.sign(a2["d_obs"].values))),
        }
    admissible = {f: r for f, r in out["forms"].items() if r["admissible"]}
    out["selected_form"] = (min(admissible,
                                key=lambda f: admissible[f]["a2_mae"])
                            if admissible else None)
    return out


def render(result: dict) -> str:
    lines = [("# Maha amplitude-operator repair: Pilot 1 validation "
             "(design data)"), ""]
    lines.append("## 1. Operator comparison (paired responses, "
                 "L = 1 - AUROC_f)")
    lines.append("")
    lines.append("| group | mean obs | mean old | mean min | MAE old "
                 "| MAE min | repair improves |")
    lines.append("|---|---|---|---|---|---|---|")
    for group in ("A1", "A2"):
        r = result["operators"][group]
        lines.append(
            f"| {group} | {r['mean_d_obs']:+.4f} | {r['mean_d_old']:+.4f} "
            f"| {r['mean_d_min']:+.4f} | {r['mae_d_old']:.4f} "
            f"| {r['mae_d_min']:.4f} | {r['repair_improves']} |")
    lines.append("")
    lines.append(f"- extraction sanity: corr(rank-AUROC from features, "
                 f"pipeline AUROC_f) = "
                 f"{result['operators']['extraction_sanity_corr']:.3f}")
    lines.append("")
    lines.append("## 2. Mechanism diagnostics (audit 5.4)")
    lines.append("")
    lines.append("| arm | MC id switch | empirical id switch "
                 "| OOD score-mean rel gap |")
    lines.append("|---|---|---|---|")
    for label, r in result["diagnostics"].items():
        lines.append(f"| {label} | {r['median_mc_id_switch']:.3f} "
                     f"| {r['median_emp_id_switch']:.3f} "
                     f"| {r['median_ood_mean_rel_gap']:.3f} |")
    lines.append("")
    cal = result["calibration"]
    lines.append(f"## 3. Bounded calibration (input {cal['input']}; "
                 f"fit on A1, evaluated on A2)")
    lines.append("")
    lines.append(f"No-change MAE: A1 {cal['no_change']['a1_mae']:.4f}, "
                 f"A2 {cal['no_change']['a2_mae']:.4f}.")
    lines.append("")
    lines.append("| form | params | A1 LOO-dose CV MAE | admissible "
                 "| A2 MAE | A2 sign |")
    lines.append("|---|---|---|---|---|---|")
    for form, r in cal["forms"].items():
        params = ", ".join(f"{v:+.4f}" for v in r["params"])
        lines.append(f"| {form} | {params} | {r['a1_cv_mae']:.4f} "
                     f"| {r['admissible']} | {r['a2_mae']:.4f} "
                     f"| {r['a2_sign_agreement']:.3f} |")
    lines.append("")
    lines.append(f"**Selected form (fresh-confirmation candidate): "
                 f"{cal['selected_form']}**")
    lines.append("")
    return "\n".join(lines)


def run_validation(table: pd.DataFrame, repair: dict) -> dict:
    cells = build_cells(table, repair)
    return {"operators": operator_comparison(cells),
            "diagnostics": diagnostics_summary(repair),
            "calibration": calibration(cells)}


def main() -> None:
    base = "nc_csf_predictivity/interventions"
    parser = argparse.ArgumentParser(description="Maha repair validation")
    parser.add_argument("--stats_root", type=str, required=True)
    parser.add_argument("--repair_dir", type=str,
                        default=f"{base}/maha_repair")
    parser.add_argument("--out", type=str,
                        default=f"{base}/maha_repair_report.md")
    args = parser.parse_args()
    table = load_long(Path(args.stats_root))
    repair = load_repair(Path(args.repair_dir))
    result = run_validation(table, repair)
    Path(args.out).write_text(render(result))
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(result, indent=1, default=float))
    ops = result["operators"]
    print(f"A1: MAE old {ops['A1']['mae_d_old']:.4f} -> min "
          f"{ops['A1']['mae_d_min']:.4f}; A2: {ops['A2']['mae_d_old']:.4f} "
          f"-> {ops['A2']['mae_d_min']:.4f}; selected calibration: "
          f"{result['calibration']['selected_form']}; wrote {args.out}")


if __name__ == "__main__":
    main()

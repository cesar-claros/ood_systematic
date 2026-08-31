"""Audit-11 R11.1 + R11.3: corrected theory cache with hero-A inputs and
E4-style diagnostics, plus the corrected held-out suite completion
(LOSO, leave-one-OOD-set influence, baselines table, dev/val halves).

EVIDENCE CLASS: post-outcome correctness audit (contract amendment
2026-08-30, audit-11 rules). Margins come UNCHANGED from the frozen-spec
corrected-dictionary audit cache; this script adds coordinates and
evaluation modes, fits nothing, and never overwrites a frozen artifact.

FROZEN SPEC: folds seed 2027 (identical machinery/folds as the frozen
held-out run and the corrected audit); paired theory-vs-severity
checkpoint bootstrap B = 2000 seed 1091 (both fold modes); materiality
|gap| >= 10 milli (frozen); ga = clip(gamma*a, 1e-4) from the same
pool_coords records the corrected audit used; E4-style diagnostics use
A = 1 - exp(l) (exact); "both above 0.99" = both l < ln(0.01);
CTM-material displayed side = A_C - A_E >= 0.01. GATE G11.A: the
material sign accuracy, balanced accuracy, and resolvable fraction
recomputed here must equal the corrected-dictionary report exactly.

Usage (from code/): python nc1_corrected_suite.py
Outputs: outputs/track1/theory_cell_predictions_corrected.parquet
         outputs/track1/nc1_corrected_suite_report.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from corrected_dictionary_audit import CACHE as CORR_CACHE
from corrected_dictionary_audit import RESOLVE_L
from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, load_coords,
                                       map_ood_names, run_folds)

B_BOOT, BOOT_SEED = 2000, 1091
NEW_CACHE = OUT_DIR / "theory_cell_predictions_corrected.parquet"
LN_099 = float(np.log(0.01))


def rho(x, y) -> float:
    return float(spearmanr(x, y, nan_policy="omit").statistic)


def main() -> None:
    cells = pd.read_parquet(CORR_CACHE)
    coords, _ = load_coords(Path("pilot0/pool_coords"))
    # run_folds' geometry/flexible baselines need the geometry columns;
    # merge them from the frozen cells table, with var_collapse
    # substituted from the corrected panel (round-consistent; declared).
    from stage2_closure import build_cells_with_severity
    geo = build_cells_with_severity()[
        ["cell", "eval_dataset", "var_collapse", "self_duality",
         "equinorm_uc", "max_equiangular_wc"]]
    cells = cells.merge(geo, on=["cell", "eval_dataset"], how="left")
    cells["var_collapse"] = cells.cell.map(
        {c: coords[c]["papyan"]["var_collapse"]
         for c in cells.cell.unique()})
    ga = np.full(len(cells), np.nan)
    rho_c = np.full(len(cells), np.nan)
    for idx, row in cells.iterrows():
        rec = coords.get(row.cell)
        sets = map_ood_names(rec, set(cells[cells.cell == row.cell]
                                      .eval_dataset))
        co = sets.get(row.eval_dataset)
        if co is not None:
            ga[idx] = max(co["gamma"] * co["a"], 1e-4)
            rho_c[idx] = co["rho"]
    cells["ga"] = ga
    cells["rho"] = rho_c
    a_e = 1.0 - np.exp(cells.l_E)
    a_c = 1.0 - np.exp(cells.l_C)
    cells["auroc_E_corr"] = a_e
    cells["auroc_C_corr"] = a_c
    cells.to_parquet(NEW_CACHE)

    mat_mask = np.abs(cells.gap) >= MATERIALITY
    e4 = {
        "frac_direct_zero_equiv": round(float(
            (~cells.resolvable_direct).mean()), 4),
        "abs_margin_quantiles_50_90_95_99": [
            round(float(np.quantile(cells.m_corr.abs(), q)), 4)
            for q in (0.5, 0.9, 0.95, 0.99)],
        "frac_both_pred_auroc_above_099": round(float(
            ((cells.l_E < LN_099) & (cells.l_C < LN_099)).mean()), 4),
        "frac_ctm_material_side_displayed": round(float(
            ((a_c - a_e) >= 0.01).mean()), 4),
        "frac_energy_material_side_displayed": round(float(
            ((a_e - a_c) >= 0.01).mean()), 4),
        "spearman_absM_absgap_all": round(
            rho(cells.m_corr.abs(), cells.gap.abs()), 4),
        "spearman_absM_absgap_material": round(
            rho(cells[mat_mask].m_corr.abs(),
                cells[mat_mask].gap.abs()), 4),
    }

    # G11.A fidelity against the corrected-dictionary report
    ref = json.loads((OUT_DIR / "corrected_dictionary_report.json")
                     .read_text())
    theory_sign = pd.Series(np.sign(cells.m_corr.values),
                            index=cells.index)
    ck = run_folds(cells, theory_sign, "ckpt5",
                   np.random.default_rng(FOLD_SEED))
    lo = run_folds(cells, theory_sign, "loso",
                   np.random.default_rng(FOLD_SEED))
    for fr in (ck, lo):
        fr["half"] = cells.half.values
        fr["m_corr"] = cells.m_corr.values
    mat_ck = ck[np.abs(ck.gap) >= MATERIALITY]
    g11a = {
        "sign_acc_here": round(accuracy(mat_ck.theory.values,
                                        mat_ck.gap.values), 4),
        "sign_acc_ref": ref["pooled"]["sign_acc"],
        "balanced_here": round(balanced_accuracy(
            mat_ck.theory.values, mat_ck.gap.values), 4),
        "balanced_ref": ref["pooled"]["balanced_acc"],
        "frac_resolvable_here": round(float(
            cells.resolvable_direct.mean()), 4),
        "frac_resolvable_ref": ref["frac_resolvable_direct"],
    }
    g11a["PASS"] = (g11a["sign_acc_here"] == g11a["sign_acc_ref"]
                    and g11a["balanced_here"] == g11a["balanced_ref"]
                    and g11a["frac_resolvable_here"]
                    == g11a["frac_resolvable_ref"])

    def summar(fr, mode):
        m = fr[np.abs(fr.gap) >= MATERIALITY].copy()
        o = m.gap.values
        arms = ["theory", "severity", "geometry", "flexible", "mean"] + (
            ["src_id"] if mode == "ckpt5" else [])
        out = {"n_material": int(len(m)), "arms": {}}
        for a in arms:
            out["arms"][a] = {
                "sign": round(accuracy(np.sign(m[a].values), o), 4),
                "balanced": round(balanced_accuracy(
                    np.sign(m[a].values), o), 4)}
        rng = np.random.default_rng(BOOT_SEED)
        cks = np.array(sorted(m.cell.unique()))
        gps = {c: g for c, g in m.groupby("cell")}
        diffs = np.empty(B_BOOT)
        for i in range(B_BOOT):
            b = pd.concat([gps[c] for c in
                           rng.choice(cks, len(cks), replace=True)])
            diffs[i] = (accuracy(b.theory.values, b.gap.values)
                        - accuracy(np.sign(b.severity.values),
                                   b.gap.values))
        out["theory_minus_severity"] = {
            "point": round(accuracy(m.theory.values, o)
                           - accuracy(np.sign(m.severity.values), o), 4),
            "ci95": [round(float(np.quantile(diffs, q)), 4)
                     for q in (0.025, 0.975)]}
        out["per_half"] = {
            h: {"sign": round(accuracy(g.theory.values, g.gap.values), 4),
                "n": int(len(g))} for h, g in m.groupby("half")}
        if mode == "loso":
            out["per_source"] = {
                s: round(accuracy(g.theory.values, g.gap.values), 3)
                for s, g in m.groupby("source")}
        out["leave_one_oodset_sign"] = {
            e: round(accuracy(m[m.eval_dataset != e].theory.values,
                              m[m.eval_dataset != e].gap.values), 3)
            for e in sorted(m.eval_dataset.unique())}
        return out

    report = {"E4_diagnostics_corrected": e4, "G11A": g11a,
              "ckpt5": summar(ck, "ckpt5"), "loso": summar(lo, "loso")}
    (OUT_DIR / "nc1_corrected_suite_report.json").write_text(
        json.dumps(report, indent=1, default=float))
    L = ["# Audit-11 R11.1/R11.3: corrected cache + held-out suite", "",
         "Post-outcome correctness audit; margins unchanged from the "
         "corrected-dictionary cache; nothing fitted.", "", "```",
         json.dumps(report, indent=1, default=float), "```", ""]
    (OUT_DIR / "nc1_corrected_suite_report.md").write_text("\n".join(L))
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()

"""NC1-normalization audit: the frozen theory arm re-evaluated under the
paper's own stated dictionary definition (post-outcome; discovered
2026-08-30 by the repair-campaign factorial's P00r-vs-P00d divergence).

THE DISCREPANCY (verified in code and data before this audit ran):
- The paper's dictionary states NC1 = Tr(Sigma_W Sigma_B^+)/C, giving
  s = (C-1)/sqrt(C*NC1); for the isotropic simplex ETF this identity is
  exact (Sigma_B eigenvalues R^2/(C-1) on the span, so
  NC1 = sigma^2 (C-1)^2/(C R^2) = (C-1)^2/(C s^2)).
- src/neural_collapse.py line 133 normalizes the within-class scatter by
  1/(N*K) instead of 1/N, so the harmonized parquet's var_collapse
  column equals NC1/C (measured per-source ratios pilot0/parquet:
  cifar10 10.0, supercifar100 19.1, cifar100 99.9, tinyimagenet ~200
  with pinv noise at D=2048; C = 10/19/100/200).
- The frozen Stage-2 theory arm consumed the parquet column, inflating
  every pool dictionary SNR by sqrt(C) (per-source medians 29/31/108/224
  -> approximately 7-16 under the correct definition, the same range as
  the BREEDS/SVHN/ImageNet-200 external cases, which used the correct
  pilot0 convention throughout).

FROZEN SPECIFICATION for this audit (declared before any corrected
outcome was inspected): identical machinery to the frozen arm and the
audit-10 tail audit (same coords records, clamps, seeds, materiality,
folds), with ONE mechanical change: s and theta come from each
checkpoint's pilot0 papyan panel in pool_coords (var_collapse,
self_duality; the paper's stated definition) instead of the parquet
columns. Stable tail evaluation (fast CTM path, equivalence 5e-13
established). Endpoints: material-cell sign and balanced accuracy with
checkpoint-clustered B=2000 bootstrap (seed 1061); comparison against
fold-fitted severity-only and train-fold mean (run_folds, seed 2027);
Spearman(|M|, |gap|); per-source table; fraction of cells whose direct
float64 margins become resolvable (both log-errors above -36.7, the
1-ulp-below-one threshold); ALL endpoints reported separately for the
frozen repair development and validation halves (seed-20260830 split):
the correction was identified from code inspection plus the
development-subset factorial, so the validation half functions as a
quasi-confirmatory readout and is flagged as such.

EVIDENCE CLASS: post-outcome correctness audit of a measurement-pipeline
normalization; no parameter is fitted; the frozen Stage-2 artifacts stay
untouched; claim upgrades remain gated on the contract's repair-campaign
rules and external evidence.

Usage (from code/): python corrected_dictionary_audit.py
Output: outputs/track1/corrected_dictionary_report.md/.json + cells cache
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, load_coords,
                                       map_ood_names, run_folds)
from stage2_closure import build_cells_with_severity
from tail_space_audit import frozen_cfg, tail_pair

B_BOOT, BOOT_SEED = 2000, 1061
RESOLVE_L = -36.7365   # log(2^-53): both above -> direct margin nonzero
CACHE = OUT_DIR / "corrected_dictionary_cells.parquet"


def rho(x, y) -> float:
    return float(spearmanr(x, y, nan_policy="omit").statistic)


def main() -> None:
    cells = build_cells_with_severity()
    coords, _ = load_coords(Path("pilot0/pool_coords"))
    res = np.full((len(cells), 2), np.nan)
    s_used = np.full(len(cells), np.nan)
    for idx, row in cells.iterrows():
        rec = coords.get(row.cell)
        if rec is None:
            continue
        sets = map_ood_names(rec, set(cells[cells.cell == row.cell]
                                      .eval_dataset))
        co = sets.get(row.eval_dataset)
        if co is None:
            continue
        c = int(rec["n_classes"])
        vc = rec["papyan"]["var_collapse"]
        sd = rec["papyan"]["self_duality"]
        s = float((c - 1) / np.sqrt(c * max(vc, 1e-9)))
        theta = float(np.degrees(np.arccos(
            np.clip(1.0 - sd / 2.0, -1.0, 1.0))))
        cfg = frozen_cfg(s, theta, rec["geometry"]["logit_scale"],
                         rec["geometry"]["class_mean_radius_cv"],
                         co["gamma"], co["a"], co["rho"])
        _, _, l_e, l_c = tail_pair(c, int(rec["dim"]), cfg, fast_ctm=True)
        res[idx] = (l_e, l_c)
        s_used[idx] = s
        if idx % 400 == 0:
            print(f"[nc1] {idx}/{len(cells)}", flush=True)
    cells = cells.copy()
    cells[["l_E", "l_C"]] = res
    cells["m_corr"] = cells.l_E - cells.l_C
    cells["s_corrected"] = s_used
    cells["resolvable_direct"] = (cells.l_E > RESOLVE_L) & (
        cells.l_C > RESOLVE_L)

    split = json.loads((OUT_DIR / "repair_dev_split.json").read_text())
    half = {c: "dev" for c in split["development"]}
    half.update({c: "val" for c in split["validation"]})
    cells["half"] = cells.cell.map(half)

    folded = run_folds(cells, pd.Series(np.sign(cells.m_corr.values),
                                        index=cells.index), "ckpt5",
                       np.random.default_rng(FOLD_SEED))
    folded["m_corr"] = cells.m_corr.values
    folded["half"] = cells.half.values
    folded["s_corrected"] = cells.s_corrected.values
    folded["resolvable_direct"] = cells.resolvable_direct.values
    mat = folded[np.abs(folded.gap) >= MATERIALITY].copy()
    rng = np.random.default_rng(BOOT_SEED)

    def block(fr):
        m = fr[np.abs(fr.gap) >= MATERIALITY]
        o = m.gap.values
        ck = np.array(sorted(m.cell.unique()))
        gp = {c: g for c, g in m.groupby("cell")}
        accs = np.empty(B_BOOT)
        diffs = np.empty(B_BOOT)
        for i in range(B_BOOT):
            b = pd.concat([gp[c] for c in
                           rng.choice(ck, len(ck), replace=True)])
            ob = b.gap.values
            accs[i] = accuracy(b.theory.values, ob)
            diffs[i] = (accuracy(b.theory.values, ob)
                        - accuracy(b.severity.values, ob))
        q = lambda x: [round(float(np.quantile(x, p)), 3)
                       for p in (.025, .975)]
        return {
            "n_material": int(len(m)),
            "sign_acc": round(accuracy(m.theory.values, o), 4),
            "sign_acc_ci95": q(accs),
            "balanced_acc": round(balanced_accuracy(m.theory.values, o),
                                  4),
            "severity_sign_acc": round(accuracy(m.severity.values, o), 4),
            "trainfold_mean_sign_acc": round(
                accuracy(m["mean"].values, o), 4),
            "corr_minus_severity": {
                "point": round(accuracy(m.theory.values, o)
                               - accuracy(m.severity.values, o), 4),
                "ci95": q(diffs)},
            "spearman_absM_absgap_material": round(
                rho(m.m_corr.abs(), np.abs(o)), 4),
            "per_source_sign_acc": {
                s: round(accuracy(g.theory.values, g.gap.values), 3)
                for s, g in m.groupby("source")},
        }

    report = {
        "s_corrected_per_source_median": {
            s: round(float(g.s_corrected.median()), 2)
            for s, g in folded.groupby("source")},
        "frac_resolvable_direct": round(
            float(folded.resolvable_direct.mean()), 4),
        "frac_resolvable_frozen_reference": 0.197,
        "spearman_absM_absgap_all": round(
            rho(folded.m_corr.abs(), folded.gap.abs()), 4),
        "pooled": block(folded),
        "development_half": block(folded[folded.half == "dev"]),
        "validation_half_quasi_confirmatory": block(
            folded[folded.half == "val"]),
    }
    keep = ["cell", "source", "eval_dataset", "half", "gap", "d",
            "l_E", "l_C", "m_corr", "s_corrected", "resolvable_direct"]
    cells[keep].to_parquet(CACHE)
    (OUT_DIR / "corrected_dictionary_report.json").write_text(
        json.dumps(report, indent=1, default=float))
    L = ["# Corrected-dictionary audit (NC1 normalization; paper's "
         "stated definition)", "",
         "Post-outcome correctness audit; frozen Stage-2 artifacts "
         "untouched; spec in the script header.", "", "```",
         json.dumps(report, indent=1), "```", ""]
    (OUT_DIR / "corrected_dictionary_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

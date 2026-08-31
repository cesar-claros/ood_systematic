"""Severity-axis amendment, supplement reruns (document section 6.6
completion): E1-style held-out leave-one-OOD-set-out, the categorical
reward control (audit-8 R2 variant), and the hero panel-B curve exports,
all under d^K and d^F. Same evidence class and frozen conventions as
severity_axis_reruns.py (post-registration amendment; both axes
reported; corrected NC1 panel for geometry features and strata).

Declared seeds (new; the registered composite runs keep theirs):
LOO boots none (E1 convention: per-held-set points + range), categorical
boots 1341 (macro) / 1342 (two-class) / 1343 (row), hero bands 1301.

Usage (from code/): python severity_axis_supplement.py
Outputs: outputs/track1/severity_axis_supplement_report.md/.json
         outputs/track1/hero_curves_dK.json, hero_curves_dF.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit8_checks import (boot_diff, fit_cat, macro_two_class,
                           two_class_cells)
from crossing_robustness_audit import (OUT_DIR, PARQUET, attach_d,
                                       build_cells, curve,
                                       load_severity_rows, make_data,
                                       severity_map, tertiles)
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, load_coords,
                                       load_outcomes, run_folds)
from heldout_theory_validation import GEO_COLS
from joint_confound_audit import (H_COLS, SOURCES, add_metadata,
                                  group_stats, macro_metric, run_joint)

B = 2000
AXES = {"K": ("kid",), "F": ("fd",)}


def loo_heldout(df, sev_rows, metrics) -> dict:
    """E1-style: per held-out OOD set, severity standardization
    recomputed without it, baselines refit, geometry-minus-severity
    material sign point."""
    base = load_outcomes(df)
    out = {}
    for e in sorted(base.eval_dataset.unique()):
        dm = severity_map(sev_rows, metrics, exclude_set=e)
        cells = attach_d(base[base.eval_dataset != e], dm)
        fr = run_folds(cells, pd.Series(np.nan, index=cells.index),
                       "ckpt5", np.random.default_rng(FOLD_SEED))
        m = fr[np.abs(fr.gap) >= MATERIALITY]
        out[e] = round(accuracy(np.sign(m.geometry.values), m.gap.values)
                       - accuracy(np.sign(m.severity.values),
                                  m.gap.values), 4)
    vals = list(out.values())
    return {"per_set": out, "range": [min(vals), max(vals)]}


def categorical_control(df, sev_rows, metrics, vc_corr, coords) -> dict:
    dm = severity_map(sev_rows, metrics)
    cells = attach_d(load_outcomes(df), dm)
    cells = cells.copy()
    cells["var_collapse"] = cells.cell.map(vc_corr)
    cells = add_metadata(cells, coords)
    fitted = run_joint(cells)
    reward_levels = {
        s: sorted(cells[(cells.source == s)
                        & (cells.paradigm == "dg")].reward.unique())
        for s in SOURCES}
    fitted["M0pluscat"] = np.nan
    fitted["M1cat"] = np.nan
    for k in range(5):
        te = fitted.fold == k
        train, test = fitted[~te], fitted[te]
        gstats = group_stats(train, GEO_COLS)
        hstats = group_stats(train, H_COLS)
        for mname in ("M0pluscat", "M1cat"):
            fitted.loc[te, mname] = fit_cat(train, test, mname, gstats,
                                            hstats, reward_levels)
    mat = fitted[np.abs(fitted.gap) >= MATERIALITY].dropna(
        subset=["M0plus", "M1", "M0pluscat", "M1cat"]).copy()
    cells2 = two_class_cells(mat)

    def stat_macro(fr, arm):
        return macro_metric(fr, arm, balanced_accuracy)

    def stat_2c(fr, arm):
        return macro_two_class(fr, arm, cells2)

    return {
        "M1cat_minus_M0pluscat_macro": boot_diff(mat, "M1cat",
                                                 "M0pluscat", stat_macro,
                                                 1341),
        "M1cat_minus_M0pluscat_two_class": boot_diff(mat, "M1cat",
                                                     "M0pluscat", stat_2c,
                                                     1342),
        "M0pluscat_macro": round(stat_macro(mat, "M0pluscat"), 3),
    }


def hero_curves(df, sev_rows, metrics, vc_corr, tag: str) -> None:
    dm = severity_map(sev_rows, metrics)
    cells_d = attach_d(build_cells(df), dm)
    data, active, fine = make_data(cells_d)
    rng = np.random.default_rng(1301)
    g0 = curve("pava", data, active, fine)
    devs = np.empty(B)
    for i in range(B):
        boot = list(rng.choice(active, len(active), replace=True))
        devs[i] = np.nanmax(np.abs(curve("pava", data, boot, fine) - g0))
    out = {"fine": [round(float(x), 4) for x in fine],
           "pooled": {"curve": [round(float(v), 4) for v in g0],
                      "band_q95": round(float(np.quantile(devs, 0.95)),
                                        4)}}
    c100 = cells_d[cells_d.source == "cifar100"].copy()
    c100["var_collapse"] = c100.cell.map(vc_corr)
    strata = tertiles(c100)
    data_c, active_c, fine_c = make_data(c100)
    out["fine_c100"] = [round(float(x), 4) for x in fine_c]
    for name in ("strong", "middle", "weak"):
        sub = [c for c in active_c if c in strata[name]]
        gc = curve("pava", data_c, sub, fine_c)
        devs = np.empty(B)
        for i in range(B):
            boot = list(rng.choice(sub, len(sub), replace=True))
            devs[i] = np.nanmax(np.abs(curve("pava", data_c, boot, fine_c)
                                       - gc))
        out[name] = {"curve": [round(float(v), 4) for v in gc],
                     "band_q95": round(float(np.quantile(devs, 0.95)), 4)}
    (OUT_DIR / f"hero_curves_{tag}.json").write_text(
        json.dumps(out, default=str))


def main() -> None:
    df = pd.read_parquet(PARQUET)
    sev_rows = load_severity_rows()
    coords, _ = load_coords(Path("pilot0/pool_coords"))
    vc_corr = {c: coords[c]["papyan"]["var_collapse"]
               for c in build_cells(df).cell.unique()}
    report = {}
    for axis, metrics in AXES.items():
        print(f"[supp-{axis}] LOO held-out ...", flush=True)
        loo = loo_heldout(df, sev_rows, metrics)
        print(f"[supp-{axis}] categorical control ...", flush=True)
        cat = categorical_control(df, sev_rows, metrics, vc_corr, coords)
        print(f"[supp-{axis}] hero curves ...", flush=True)
        hero_curves(df, sev_rows, metrics, vc_corr, f"d{axis}")
        report[axis] = {"heldout_loo_geo_minus_sev": loo,
                        "categorical": cat}
    (OUT_DIR / "severity_axis_supplement_report.json").write_text(
        json.dumps(report, indent=1, default=str))
    L = ["# Severity-axis supplement (LOO held-out, categorical control, "
         "hero curves)", "", "```",
         json.dumps(report, indent=1, default=str), "```", ""]
    (OUT_DIR / "severity_axis_supplement_report.md").write_text(
        "\n".join(L))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()

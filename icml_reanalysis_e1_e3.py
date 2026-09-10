"""ICML campaign E1 / E3: METHODOLOGICAL REANALYSIS (repair item P1-E;
status-review finding F8). POST-OUTCOME and DESCRIPTIVE by construction:
the registered results of the registered procedures stand unchanged in
`icml_campaign_report.json`; this file adds, beside them,

E1 (roster B, prospective sign accuracy of the frozen arm vs the severity
baseline, both axes, ckpt5 and loso folds):
  - the registered point and 95% percentile interval, REPRODUCED and
    asserted equal to the readout of record (the original bootstrap
    resampled material rows of previously fitted folds);
  - a checkpoint bootstrap WITH REFIT (the fold assignment and the
    severity baseline are refitted inside every replicate);
  - a run-label FAMILY bootstrap with refit (units = the five FD-Shifts
    run labels; every checkpoint of a resampled family enters; the
    support of a five-unit bootstrap is coarse and is reported);
  - a delete-one-family jackknife with t_{4} (N_f = 5) on the refit
    statistic, with the five leave-one-out values and the per-family
    material sign accuracies;
E3 (P10 vs P00 predicted-AUROC error, rosters A and C):
  - the registered per-checkpoint percentile interval, REPRODUCED;
  - roster C: family bootstrap and family jackknife t_{4} over the five
    run labels (25/26/29/24/36 checkpoints), per-family contributions;
  - roster A: all six checkpoint contributions listed (two datasets x
    three seeds), the exact count of positive contributions, a t_{5}
    interval over the six checkpoints, and the seed-index grouping
    (s0/s1/s2 across datasets) as a three-family sensitivity.
Nominal levels are labeled on every interval (95% two-sided); none of
these intervals carries a simulation license; they are companions to the
registered results, not replacements.

Usage (from code/):
    python icml_reanalysis_e1_e3.py [--b 2000] [--self-test]
Output: nc_csf_predictivity/outputs/track1/icml_e1_e3_reanalysis.json/.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student

from icml_campaign_analysis import (DIR_A_CIFAR, DIR_A_IN200, DIR_B, DIR_C_OUT, DIR_C_STATS, DIR_POOL, E1_SEED, E3_SEED, OUT_DIR, SEV_V2,
                                    _load_dir, _synth_cells, attach_axis, cells_from_records, e1_block, e3_endpoint,
                                    pred_auroc, roster_c_cells, run_folds_e1, severity_axes)
from heldout_theory_validation import MATERIALITY, accuracy

REPORT = OUT_DIR / "icml_campaign_report.json"
LABEL = "METHODOLOGICAL REANALYSIS (post-outcome, descriptive; nominal 95% two-sided; no simulation license)"
SEED_RE = 3301


def family_of(cell: str) -> str:
    m = re.search(r"_run(\d+)_", cell)
    if m:
        return f"run{m.group(1)}"
    m = re.search(r"__s(\d+)$", cell)
    return f"seed{m.group(1)}" if m else "single"


def _tci(vals: np.ndarray, level=0.95) -> dict:
    v = np.asarray(vals, float); n = len(v)
    se = float(v.std(ddof=1) / np.sqrt(n)); q = student.ppf(1 - (1 - level) / 2, n - 1)
    return {"mean": float(v.mean()), "se": se, "ci": [float(v.mean() - q * se), float(v.mean() + q * se)], "df": n - 1, "nominal": level}


def _jack(units: list, stat) -> dict:
    """Delete-one jackknife with t_{N-1}; stat(units_kept) -> float."""
    N = len(units)
    full = float(stat(units))
    loo = np.array([stat([u for u in units if u != x]) for x in units], float)
    if not np.all(np.isfinite(loo)):
        return {"estimate": full, "N_f": N, "not_estimable": "non-finite leave-one-out value", "loo": dict(zip(units, loo.tolist()))}
    se = float(np.sqrt((N - 1) / N * ((loo - loo.mean()) ** 2).sum()))
    q = student.ppf(0.975, N - 1)
    return {"estimate": full, "se": se, "N_f": N, "ci": [full - q * se, full + q * se], "df": N - 1, "nominal": 0.95,
            "loo": dict(zip(units, loo.tolist())), "not_estimable": None if se > 0 else "zero jackknife SE"}


# ---------------------------------------------------------------------------
# E1 with refit.
# ---------------------------------------------------------------------------

def e1_diff_refit(cells: pd.DataFrame, mode: str) -> float:
    """The E1 statistic with the folds and the severity baseline fitted on
    exactly the cells passed in (refit)."""
    folded = run_folds_e1(cells, mode)
    m = folded[np.abs(folded.gap) >= MATERIALITY]
    if not len(m):
        return float("nan")
    return accuracy(m.m.values, m.gap.values) - accuracy(m.severity.values, m.gap.values)


def _resample_units(cells: pd.DataFrame, col: str, units: np.ndarray, rng) -> pd.DataFrame:
    """Resample units with replacement; a unit drawn k times contributes k
    copies with distinct identities (so refit folds treat copies as separate
    checkpoints of the same data, the standard cluster-bootstrap convention)."""
    draw = rng.choice(units, len(units), replace=True)
    parts = []
    for i, u in enumerate(draw):
        part = cells[cells[col] == u].copy()
        part["cell"] = part["cell"] + f"#b{i}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def e1_reanalysis(cells: pd.DataFrame, axis_map: dict, mode: str, b: int, registered: dict) -> dict:
    withd = attach_axis(cells, axis_map)
    withd["family"] = withd.cell.map(family_of)
    # registered procedure, reproduced
    reg = e1_block(run_folds_e1(withd, mode), b)
    assert reg["diff_point"] == registered["diff_point"] and reg["diff_ci95"] == registered["diff_ci95"], (mode, reg["diff_point"], registered["diff_point"])
    out = {"label": LABEL, "registered_reproduced": {"diff_point": reg["diff_point"], "diff_ci95_percentile_no_refit": reg["diff_ci95"],
                                                      "n_material": reg["n_material"], "n_checkpoints_material": reg["n_checkpoints_material"]}}
    point = e1_diff_refit(withd, mode)
    out["point_refit_full_panel"] = point
    ck = np.array(sorted(withd.cell.unique())); fams = np.array(sorted(withd.family.unique()))
    rng = np.random.default_rng([SEED_RE, 1])
    boots = np.array([e1_diff_refit(_resample_units(withd, "cell", ck, rng), mode) for _ in range(b)])
    out["checkpoint_bootstrap_with_refit"] = {"B": b, "ci95_percentile": [float(np.nanquantile(boots, .025)), float(np.nanquantile(boots, .975))],
                                              "n_nonfinite": int(np.isnan(boots).sum()), "unit": "checkpoint (seed reuse across configurations not audited)"}
    rng = np.random.default_rng([SEED_RE, 2])
    fb = np.array([e1_diff_refit(_resample_units(withd, "family", fams, rng), mode) for _ in range(b)])
    out["family_bootstrap_with_refit"] = {"B": b, "N_f": int(len(fams)), "ci95_percentile": [float(np.nanquantile(fb, .025)), float(np.nanquantile(fb, .975))],
                                          "n_nonfinite": int(np.isnan(fb).sum()), "unit": "run label (all its checkpoints)",
                                          "caveat": "five-unit bootstrap: 126 distinct resamples; percentile bounds are coarse"}
    jk = _jack(list(fams), lambda keep: e1_diff_refit(withd[withd.family.isin(keep)], mode))
    out["family_jackknife_t4"] = jk
    per_fam = {}
    folded = run_folds_e1(withd, mode)
    for f, g in folded.groupby("family"):
        m = g[np.abs(g.gap) >= MATERIALITY]
        per_fam[f] = {"n_material": int(len(m)), "theory_sign_acc": (accuracy(m.m.values, m.gap.values) if len(m) else None),
                      "severity_sign_acc": (accuracy(m.severity.values, m.gap.values) if len(m) else None)}
    out["per_family_material_sign_accuracy_full_panel_fit"] = per_fam
    return out


# ---------------------------------------------------------------------------
# E3 with family units.
# ---------------------------------------------------------------------------

def e3_per_checkpoint(cells: pd.DataFrame) -> pd.DataFrame:
    sub = cells.dropna(subset=["auroc_E", "auroc_C", "l_E_p10"])
    pa = lambda vals: np.array([pred_auroc(v) for v in vals])
    rows = []
    for cell, g in sub.groupby("cell"):
        e00 = np.concatenate([np.abs(pa(g.l_E) - g.auroc_E.values.astype(float)), np.abs(pa(g.l_C) - g.auroc_C.values.astype(float))])
        e10 = np.concatenate([np.abs(pa(g.l_E_p10) - g.auroc_E.values.astype(float)), np.abs(pa(g.l_C_p10) - g.auroc_C.values.astype(float))])
        rows.append(dict(cell=cell, family=family_of(cell), source=g.source.iloc[0], n_cells=int(len(g)),
                         mae_P00=float(e00.mean()), mae_P10=float(e10.mean()), diff=float(e00.mean() - e10.mean())))
    return pd.DataFrame(rows)


def e3_reanalysis(cells: pd.DataFrame, b: int, registered: dict, roster: str) -> dict:
    reg = e3_endpoint(cells, b)
    assert reg["diff_point"] == registered["diff_point"] and reg["diff_ci95"] == registered["diff_ci95"], (roster, reg, registered)
    df = e3_per_checkpoint(cells)
    out = {"label": LABEL, "registered_reproduced": {"diff_point": reg["diff_point"], "diff_ci95_percentile_checkpoints": reg["diff_ci95"],
                                                      "n_checkpoints": reg["n_checkpoints"], "unit": "checkpoint"},
           "n_positive_contributions": [int((df["diff"] > 0).sum()), int(len(df))]}
    fams = sorted(df.family.unique())
    fam_mean = df.groupby("family")["diff"].mean()
    out["per_family"] = {f: {"n_checkpoints": int((df.family == f).sum()), "mean_diff": float(fam_mean[f]),
                             "min_diff": float(df[df.family == f]["diff"].min()), "max_diff": float(df[df.family == f]["diff"].max())} for f in fams}
    if len(fams) >= 3:
        rng = np.random.default_rng([SEED_RE, 3])
        fb = []
        for _ in range(b):
            draw = rng.choice(fams, len(fams), replace=True)
            fb.append(float(np.concatenate([df[df.family == f]["diff"].values for f in draw]).mean()))
        fb = np.array(fb)
        out["family_bootstrap"] = {"B": b, "N_f": len(fams), "ci95_percentile": [float(np.quantile(fb, .025)), float(np.quantile(fb, .975))],
                                   "statistic": "mean of per-checkpoint differences over the resampled families' checkpoints",
                                   "caveat": (f"{len(fams)}-unit bootstrap: coarse support" if len(fams) <= 6 else None)}
        out["family_jackknife_t"] = _jack(fams, lambda keep: float(df[df.family.isin(keep)]["diff"].mean()))
    if roster == "A":
        out["six_contributions"] = df[["cell", "source", "family", "n_cells", "mae_P00", "mae_P10", "diff"]].to_dict("records")
        out["t_interval_over_six_checkpoints"] = _tci(df["diff"].values)
        out["exact_sign_count"] = {"positive": int((df["diff"] > 0).sum()), "of": int(len(df)),
                                   "two_sided_sign_test_p": float(2 * min(sum(__import__('math').comb(len(df), k) for k in range(0, int((df['diff'] > 0).sum()) + 1)),
                                                                          sum(__import__('math').comb(len(df), k) for k in range(int((df['diff'] > 0).sum()), len(df) + 1))) / 2 ** len(df))}
        out["seed_index_grouping"] = {"note": "s0/s1/s2 across the two datasets are separate trainings; grouped only as a sensitivity",
                                      "jackknife_t2": _jack(fams, lambda keep: float(df[df.family.isin(keep)]["diff"].mean()))}
    return out


# ---------------------------------------------------------------------------
# Run / self-test.
# ---------------------------------------------------------------------------

def run(b: int) -> None:
    rep = json.loads(REPORT.read_text())
    sev = pd.read_csv(SEV_V2)
    recs_a, _ = _load_dir(DIR_A_CIFAR, "schema_icml_a")
    recs_a3, _ = _load_dir(DIR_A_IN200, "schema_stage3")
    recs_b, _ = _load_dir(DIR_B, "schema_icml_b")
    dim_map = {}                                   # roster-B schema-1 records omit `dim`: the frozen reader's pool_coords join
    for p in sorted(DIR_POOL.glob("*.json")):
        if p.name.startswith("FAILED"):
            continue
        r = json.loads(p.read_text())
        if r.get("schema") == 2:                   # exactly the frozen reader's condition
            dim_map[r["model_path"]] = int(r["dim"])
    for r in recs_b:
        if "dim" not in r:
            r["dim"] = dim_map[r["model_path"]]
    cells_a = cells_from_records(recs_a + recs_a3, "gap_balanced", with_p10=True)
    cells_b = cells_from_records(recs_b, "gap_raw", with_p10=False)
    cells_c = roster_c_cells(DIR_C_STATS, DIR_C_OUT)
    axes_b = {ax: severity_axes(sev, "new_shifts")[ax] for ax in ("K", "F")}
    out = {"label": LABEL, "registered_readout": {"path": str(REPORT), "unchanged": True},
           "E1_roster_b": {ax: {mode: e1_reanalysis(cells_b, axes_b[ax], mode, b, rep["E1_roster_b"][ax][mode]) for mode in ("ckpt5", "loso")} for ax in ("K", "F")},
           "E3_roster_c": e3_reanalysis(cells_c, b, rep["E3_roster_c"], "C"),
           "E3_roster_a": e3_reanalysis(cells_a, b, rep["E3_roster_a"], "A"),
           "family_structure": {"roster_b": cells_b.cell.map(family_of).value_counts().to_dict(),
                                "roster_c": pd.Series([family_of(c) for c in cells_c.cell.unique()]).value_counts().to_dict(),
                                "roster_a": pd.Series([family_of(c) for c in cells_a.cell.unique()]).value_counts().to_dict()}}
    (OUT_DIR / "icml_e1_e3_reanalysis.json").write_text(json.dumps(out, indent=1, default=str))
    (OUT_DIR / "icml_e1_e3_reanalysis.md").write_text("# ICML E1/E3 methodological reanalysis (post-outcome, descriptive)\n\n```\n" + json.dumps(out, indent=1, default=str) + "\n```\n")
    summary = {"E1_B_K_ckpt5": {k: out["E1_roster_b"]["K"]["ckpt5"][k] for k in ("registered_reproduced", "checkpoint_bootstrap_with_refit", "family_bootstrap_with_refit")},
               "E1_B_K_ckpt5_family_jackknife_ci": out["E1_roster_b"]["K"]["ckpt5"]["family_jackknife_t4"].get("ci"),
               "E3_C": {k: out["E3_roster_c"][k] for k in ("registered_reproduced", "n_positive_contributions", "family_bootstrap")},
               "E3_C_family_jackknife_ci": out["E3_roster_c"]["family_jackknife_t"].get("ci"),
               "E3_A": {k: out["E3_roster_a"][k] for k in ("registered_reproduced", "exact_sign_count", "t_interval_over_six_checkpoints")}}
    print(json.dumps(summary, indent=1, default=str))


def self_test() -> None:
    rng = np.random.default_rng(3)
    cells = _synth_cells(rng)
    cells["cell"] = [f"{c}_run{(i % 5) + 1}_x" for i, c in enumerate(cells.cell)]
    axis = {(s, e): float(i) for s in cells.source.unique() for i, e in enumerate(sorted(cells.eval_dataset.unique()))}
    withd = attach_axis(cells, axis); withd["family"] = withd.cell.map(family_of)
    assert set(withd.family) == {f"run{i}" for i in range(1, 6)}
    d = e1_diff_refit(withd, "ckpt5"); assert np.isfinite(d)
    jk = _jack(sorted(withd.family.unique()), lambda keep: e1_diff_refit(withd[withd.family.isin(keep)], "ckpt5"))
    assert jk["N_f"] == 5 and jk["df"] == 4
    if "l_E_p10" in cells:
        df = e3_per_checkpoint(cells); assert len(df) == cells.cell.nunique()
    print("[e1e3-reanalysis] self-test PASS: families parsed, refit statistic finite, family jackknife df 4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    self_test() if a.self_test else run(a.b)


if __name__ == "__main__":
    main()

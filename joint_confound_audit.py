"""Post-hoc, audit-requested source-, training-paradigm-, and
training-hyperparameter-confounding sensitivity (audit #7, 2026-08-26,
sections 3.4-3.8; documentation/
companion_phase_diagram_evaluation_and_source_expansion_plan_2026-08-26.md).

STATUS: post hoc. The specification below is FROZEN in this header before
execution; nothing here may be tuned after the joint-audit outcomes are
inspected. The frozen Stage-2 results (heldout_theory_report,
stage2_closure_report) remain the evidence of record and are not modified.

FROZEN SPECIFICATION
====================

Data: registered cell table (stage2_closure.build_cells_with_severity),
280 VGG-13 checkpoints, cell = paradigm|source|run|reward|dropout;
materiality |gap| >= 10 (AUGRC x 1000, registered). Geometry features
G = (var_collapse, self_duality, equinorm_uc, max_equiangular_wc)
(registered GEO_COLS). Head-state features H = (logit_scale,
head_residual_fraction, equinorm_wc) from pilot0/pool_coords (schema 2).
ID accuracy and ID calibration are NOT present in the harmonized data and
are therefore excluded from H; this is recorded, not a choice made on
outcomes. Severity d = registered CLIP composite.

Folds: checkpoint-grouped five-fold, stratified within source x paradigm
cells (per-cell permutation, round-robin deal). Fold seed 20260826.

Models (audit sections 3.5-3.6), all fitted with the registered ridge
logistic (lam = 1.0, IRLS) on train-fold-standardized features, predicting
sign(gap) > 0; all centering/scaling statistics computed from TRAINING-fold
checkpoints only (one value per checkpoint for checkpoint-level features):
  M0   : d, d^2, source one-hots, paradigm one-hots, source x paradigm
         one-hots, d x source, d x paradigm.
  M0+  : M0 + dropout indicator + DG reward basis nested within DG and
         source: z_s(log reward) and z_s(log reward)^2 per source
         (z from training-fold DG checkpoints of that source; zero for
         non-DG rows).
  M1   : M0+ + Gt + d x Gt, where Gt = G centered and scaled within each
         source x paradigm cell using training-fold checkpoint statistics.
  MH   : M0+ + Ht + d x Ht (head channel), Ht centered like Gt.
  MHG  : MH + Gt + d x Gt.
Reference arms (continuity with Stage 2 / audit section 3.3): pooled
severity isotonic (registered severity_only) and source-specific severity
isotonic (same estimator fitted per source on training folds).

Endpoints: material-cell balanced accuracy (PRIMARY; winner prevalence is
extreme) and sign accuracy (secondary, Stage-2 continuity). Primary pooled
summary: MACRO over source x paradigm cells (equal-weight mean of per-cell
metrics over cells with >= 10 material evaluation cells; cells with a
single observed winner class are included but flagged, and a two-class-only
macro is reported as sensitivity). Row-weighted summaries are secondary.

Primary estimand: Delta_geometry|S,L = BAcc(M1) - BAcc(M0+), macro.

Uncertainty: paired checkpoint-cluster bootstrap, B = 2000, resampling
checkpoints with replacement WITHIN each source x paradigm stratum
(preserves the 12-cell composition), computed on the pooled out-of-fold
predictions. Bootstrap seeds: 811 (model-difference CIs; +1 per listed
comparison in order), 91 (crossing bands), reported per table.

Support (audit section 3.4): per source x paradigm cell and per DG
source x reward stratum: checkpoint count, material-cell count, winner
prevalence, 5/50/95% quantiles of each G feature, distinct-severity count,
dropout composition, H medians. Pairwise paradigm contrasts within source:
standardized mean difference (pooled SD) and range-overlap fraction
(|intersection| / |union| of [min, max]) per G feature. Out-of-support
rule: an evaluation checkpoint is outside training support if ANY raw G
feature falls outside the training-fold [min, max] of its source x
paradigm cell; report the fraction of material evaluation cells on
out-of-support checkpoints.

Paradigm-stratified crossings (audit section 3.7): per paradigm, collapse
tertiles from that paradigm's checkpoints only (registered tertiles);
registered pava estimator, cluster-bootstrap simultaneous bands (B = 2000),
registered censoring rules (crossing_value / ordering_retained).
FEASIBILITY (declared): a paradigm is eligible only with >= 20
checkpoints, >= 5 distinct severities, material cells in both winner
classes, and >= 5 checkpoints per tertile; otherwise report INFEASIBLE
with the reason, never a failed crossing. Equal-paradigm-weighted pooled
curve: unweighted mean of the three per-paradigm pooled pava curves on the
common severity grid, bootstrap resampling checkpoints within paradigm
(B = 2000). Saturation by paradigm from the frozen theory cache
(theory_cell_predictions.parquet): analytic sign accuracy on material
cells, fraction of exactly zero margins, fraction with both predicted
AUROCs > 0.99, gamma*a quantiles. No theory rerun, no recalibration.

Leave-one-paradigm-out transport (audit section 3.8): train the registered
severity_only and geometry_model (non-flexible) arms on two paradigms,
freeze, evaluate on the third; three results reported individually (never
bootstrapped as a paradigm population), with the fraction of held-out
checkpoints outside the training G support (within source) as the
extrapolation flag.

Interpretation rule (audit section 3.9), declared before results:
"geometry adds within source x paradigm" = macro Delta(M1 - M0+) > 0 with
clustered 95% CI excluding zero; "objective-held-out transport succeeds" =
geometry beats severity in balanced accuracy on >= 2 of 3 held-out
paradigms with no held-out paradigm below -0.05.

Usage (from code/): python joint_confound_audit.py
Outputs: nc_csf_predictivity/outputs/track1/joint_confound_audit_report.md
         + .json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossing_robustness_audit import (OUT_DIR, analyze_curve,
                                       crossing_display, crossing_value,
                                       ordering_retained, pava_inc)
from heldout_theory_validation import (GEO_COLS, MATERIALITY, accuracy,
                                       balanced_accuracy, geometry_model,
                                       load_coords, ridge_logistic,
                                       severity_only)
from stage2_closure import CACHE, build_cells_with_severity

FOLD_SEED_J = 20260826
B_BOOT = 2000
SOURCES = ("cifar10", "cifar100", "supercifar100", "tinyimagenet")
PARADIGMS = ("confidnet", "devries", "dg")
H_COLS = ["logit_scale", "head_residual_fraction", "equinorm_wc"]
MODEL_ORDER = ["M0", "M0plus", "M1", "MH", "MHG", "sev_pooled", "sev_source"]
COMPARISONS = [("M0plus", "M0"), ("M1", "M0plus"), ("M1", "M0"),
               ("MH", "M0plus"), ("MHG", "MH"), ("M1", "MH"),
               ("M0plus", "sev_source")]


# ---------------------------------------------------------------------------
# Metadata + features.
# ---------------------------------------------------------------------------

def add_metadata(cells: pd.DataFrame, coords: dict) -> pd.DataFrame:
    parts = cells.cell.str.split("|")
    cells = cells.copy()
    cells["paradigm"] = parts.str[0]
    cells["reward"] = parts.str[3].astype(float)
    cells["dropout"] = (parts.str[4] == "True").astype(float)
    cells["sp"] = cells.source + "/" + cells.paradigm
    for h in H_COLS:
        cells[h] = np.nan
    for cell, rec in coords.items():
        m = cells.cell == cell
        cells.loc[m, "logit_scale"] = rec["geometry"]["logit_scale"]
        cells.loc[m, "head_residual_fraction"] = (
            rec["geometry"]["head_residual_fraction"])
        cells.loc[m, "equinorm_wc"] = rec["papyan"]["equinorm_wc"]
    assert not cells[H_COLS].isna().any().any(), "missing head features"
    return cells


def ckpt_frame(fr: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return fr.groupby("cell")[["sp", "source", "paradigm", "reward",
                               "dropout"] + cols].first()


def group_stats(train: pd.DataFrame, cols: list[str]) -> dict:
    """source x paradigm -> (mu, sd, mn, mx) per column, from training-fold
    checkpoints (one value each); fallback = source-level stats."""
    ck = ckpt_frame(train, cols)
    stats = {}
    for key, g in ck.groupby("sp"):
        v = g[cols].values
        stats[key] = (v.mean(0), v.std(0) + 1e-12, v.min(0), v.max(0))
    for key, g in ck.groupby("source"):
        v = g[cols].values
        stats[("src", key)] = (v.mean(0), v.std(0) + 1e-12, v.min(0),
                               v.max(0))
    return stats


def centered(fr: pd.DataFrame, cols: list[str], stats: dict) -> np.ndarray:
    out = np.empty((len(fr), len(cols)))
    for i, (sp, src) in enumerate(zip(fr.sp.values, fr.source.values)):
        mu, sd, _, _ = stats.get(sp, stats[("src", src)])
        out[i] = (fr[cols].values[i] - mu) / sd
    return out


def reward_stats(train: pd.DataFrame) -> dict:
    ck = ckpt_frame(train, [])
    out = {}
    for src, g in ck[ck.paradigm == "dg"].groupby("source"):
        lr = np.log(g.reward.values)
        out[src] = (lr.mean(), lr.std() + 1e-12)
    return out


def features(fr: pd.DataFrame, model: str, gstats: dict, hstats: dict,
             rstats: dict) -> np.ndarray:
    d = fr.d.values
    feats = [d, d * d]
    src_oh = {s: (fr.source == s).values.astype(float) for s in SOURCES}
    par_oh = {p: (fr.paradigm == p).values.astype(float) for p in PARADIGMS}
    feats += list(src_oh.values()) + list(par_oh.values())
    feats += [src_oh[s] * par_oh[p] for s in SOURCES for p in PARADIGMS]
    feats += [d * src_oh[s] for s in SOURCES]
    feats += [d * par_oh[p] for p in PARADIGMS]
    if model != "M0":
        feats.append(fr.dropout.values)
        is_dg = par_oh["dg"]
        lr = np.where(fr.paradigm == "dg", np.log(fr.reward.values), 0.0)
        for s in SOURCES:
            mu, sd = rstats.get(s, (0.0, 1.0))
            z = np.where((fr.source == s).values, (lr - mu) / sd, 0.0) * is_dg
            feats += [z, z * z]
    if model in ("MH", "MHG"):
        ht = centered(fr, H_COLS, hstats)
        feats += [ht[:, j] for j in range(ht.shape[1])]
        feats += [d * ht[:, j] for j in range(ht.shape[1])]
    if model in ("M1", "MHG"):
        gt = centered(fr, GEO_COLS, gstats)
        feats += [gt[:, j] for j in range(gt.shape[1])]
        feats += [d * gt[:, j] for j in range(gt.shape[1])]
    return np.column_stack(feats)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, model: str,
                gstats: dict, hstats: dict, rstats: dict) -> np.ndarray:
    xtr = features(train, model, gstats, hstats, rstats)
    xte = features(test, model, gstats, hstats, rstats)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    beta = ridge_logistic((xtr - mu) / sd,
                          (train.gap.values > 0).astype(float), lam=1.0)
    return beta[0] + ((xte - mu) / sd) @ beta[1:]


def source_severity(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    out = np.full(len(test), np.nan)
    for src in test.source.unique():
        tr, te = train[train.source == src], test[test.source == src]
        if len(tr):
            out[np.array(test.source == src)] = severity_only(tr, te)
    return out


# ---------------------------------------------------------------------------
# Section 3.4: support / positivity audit.
# ---------------------------------------------------------------------------

def support_audit(cells: pd.DataFrame) -> dict:
    ck = ckpt_frame(cells, GEO_COLS + H_COLS)
    mat = cells[np.abs(cells.gap) >= MATERIALITY]
    out = {"cells": {}, "dg_reward": {}, "contrasts": {}}
    for sp, g in ck.groupby("sp"):
        rows = cells[cells.sp == sp]
        m = mat[mat.sp == sp]
        rec = {"n_ckpt": int(len(g)), "n_material": int(len(m)),
               "frac_positive": (round(float((m.gap > 0).mean()), 3)
                                 if len(m) else None),
               "single_winner_class": bool(len(m) and
                                           m.gap.gt(0).nunique() == 1),
               "n_severities": int(rows.d.nunique()),
               "dropout_frac": round(float(g.dropout.mean()), 2),
               "rewards": sorted(g.reward.unique().tolist())
               if (g.paradigm == "dg").all() else None}
        for c in GEO_COLS:
            rec[c] = [round(float(np.quantile(g[c], q)), 4)
                      for q in (0.05, 0.5, 0.95)]
        for c in H_COLS:
            rec[f"{c}_median"] = round(float(g[c].median()), 4)
        out["cells"][sp] = rec
    dg = ck[ck.paradigm == "dg"]
    for (src, rew), g in dg.groupby(["source", "reward"]):
        m = mat[(mat.source == src) & (mat.paradigm == "dg")
                & (mat.reward == rew)]
        out["dg_reward"][f"{src}/rew{rew:g}"] = {
            "n_ckpt": int(len(g)), "n_material": int(len(m)),
            "frac_positive": (round(float((m.gap > 0).mean()), 3)
                              if len(m) else None),
            "var_collapse_median": round(float(g.var_collapse.median()), 4)}
    for src in SOURCES:
        for i, p1 in enumerate(PARADIGMS):
            for p2 in PARADIGMS[i + 1:]:
                g1 = ck[(ck.source == src) & (ck.paradigm == p1)]
                g2 = ck[(ck.source == src) & (ck.paradigm == p2)]
                rec = {}
                for c in GEO_COLS:
                    v1, v2 = g1[c].values, g2[c].values
                    pooled = np.sqrt((v1.var() + v2.var()) / 2) + 1e-12
                    smd = (v1.mean() - v2.mean()) / pooled
                    lo = max(v1.min(), v2.min())
                    hi = min(v1.max(), v2.max())
                    union = max(v1.max(), v2.max()) - min(v1.min(), v2.min())
                    ov = max(0.0, hi - lo) / (union + 1e-12)
                    rec[c] = {"smd": round(float(smd), 2),
                              "range_overlap": round(float(ov), 2)}
                out["contrasts"][f"{src}:{p1}-vs-{p2}"] = rec
    return out


# ---------------------------------------------------------------------------
# Section 3.5 + 3.6: joint models, out-of-fold.
# ---------------------------------------------------------------------------

def make_folds(cells: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    fold = pd.Series(-1, index=sorted(cells.cell.unique()))
    for _, g in ckpt_frame(cells, []).groupby("sp"):
        ids = rng.permutation(np.array(sorted(g.index)))
        for i, c in enumerate(ids):
            fold[c] = i % 5
    return fold


def run_joint(cells: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(FOLD_SEED_J)
    fold = make_folds(cells, rng)
    cells = cells.copy()
    cells["fold"] = cells.cell.map(fold)
    for m in MODEL_ORDER:
        cells[m] = np.nan
    cells["out_of_support"] = False
    for k in range(5):
        te = cells.fold == k
        train, test = cells[~te], cells[te]
        gstats = group_stats(train, GEO_COLS)
        hstats = group_stats(train, H_COLS)
        rstats = reward_stats(train)
        for m in ("M0", "M0plus", "M1", "MH", "MHG"):
            cells.loc[te, m] = fit_predict(train, test, m, gstats, hstats,
                                           rstats)
        cells.loc[te, "sev_pooled"] = severity_only(train, test)
        cells.loc[te, "sev_source"] = source_severity(train, test)
        oos = np.zeros(int(te.sum()), bool)
        gvals = test[GEO_COLS].values
        for i, (sp, src) in enumerate(zip(test.sp.values, test.source.values)):
            _, _, mn, mx = gstats.get(sp, gstats[("src", src)])
            oos[i] = bool(np.any(gvals[i] < mn) or np.any(gvals[i] > mx))
        cells.loc[te, "out_of_support"] = oos
        print(f"[joint] fold {k} done", flush=True)
    return cells


def macro_metric(mat: pd.DataFrame, arm: str, metric,
                 two_class_only: bool = False) -> float:
    vals = []
    for _, g in mat.groupby("sp"):
        if len(g) < 10:
            continue
        if two_class_only and g.gap.gt(0).nunique() < 2:
            continue
        vals.append(metric(g[arm].values, g.gap.values))
    return float(np.mean(vals)) if vals else float("nan")


def paired_boot(mat: pd.DataFrame, arm_a: str, arm_b: str, metric, seed: int,
                macro: bool) -> dict:
    def stat(fr):
        if macro:
            return (macro_metric(fr, arm_a, metric)
                    - macro_metric(fr, arm_b, metric))
        return metric(fr[arm_a].values, fr.gap.values) - metric(
            fr[arm_b].values, fr.gap.values)
    strata = {sp: {c: g for c, g in gg.groupby("cell")}
              for sp, gg in mat.groupby("sp")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(B_BOOT)
    for i in range(B_BOOT):
        parts = []
        for sp, groups in strata.items():
            ids = np.array(sorted(groups))
            parts += [groups[c] for c in rng.choice(ids, len(ids),
                                                    replace=True)]
        diffs[i] = stat(pd.concat(parts))
    return {"point": round(stat(mat), 3),
            "ci95": [round(float(np.quantile(diffs, 0.025)), 3),
                     round(float(np.quantile(diffs, 0.975)), 3)]}


def joint_summary(cells: pd.DataFrame) -> dict:
    mat = cells[np.abs(cells.gap) >= MATERIALITY].dropna(
        subset=list(MODEL_ORDER)).copy()
    out = {"n_material": int(len(mat)),
           "frac_positive": round(float((mat.gap > 0).mean()), 3),
           "frac_material_out_of_support": round(
               float(mat.out_of_support.mean()), 3),
           "arms": {}, "differences": {}, "per_source": {},
           "per_paradigm": {}, "influence": {}}
    for m in MODEL_ORDER:
        out["arms"][m] = {
            "bal_macro": round(macro_metric(mat, m, balanced_accuracy), 3),
            "bal_macro_two_class": round(
                macro_metric(mat, m, balanced_accuracy, True), 3),
            "bal_row": round(balanced_accuracy(mat[m].values,
                                               mat.gap.values), 3),
            "sign_macro": round(macro_metric(mat, m, accuracy), 3),
            "sign_row": round(accuracy(mat[m].values, mat.gap.values), 3)}
    seed = 811
    for a, b in COMPARISONS:
        out["differences"][f"{a}-{b}"] = {
            "bal_macro": paired_boot(mat, a, b, balanced_accuracy, seed,
                                     True),
            "bal_row": paired_boot(mat, a, b, balanced_accuracy, seed + 1,
                                   False),
            "sign_row": paired_boot(mat, a, b, accuracy, seed + 2, False)}
        seed += 3
    for key, col in (("per_source", "source"), ("per_paradigm", "paradigm")):
        for v, g in mat.groupby(col):
            out[key][v] = {
                "n_material": int(len(g)),
                "frac_positive": round(float((g.gap > 0).mean()), 3),
                "M0plus_bal": round(balanced_accuracy(g.M0plus.values,
                                                      g.gap.values), 3),
                "M1_bal": round(balanced_accuracy(g.M1.values,
                                                  g.gap.values), 3),
                "MH_bal": round(balanced_accuracy(g.MH.values,
                                                  g.gap.values), 3),
                "MHG_bal": round(balanced_accuracy(g.MHG.values,
                                                   g.gap.values), 3)}
    for col in ("source", "paradigm"):
        for v in mat[col].unique():
            sub = mat[mat[col] != v]
            out["influence"][f"drop_{v}"] = {
                "M1-M0plus_bal_macro": round(
                    macro_metric(sub, "M1", balanced_accuracy)
                    - macro_metric(sub, "M0plus", balanced_accuracy), 3),
                "M1-M0plus_bal_row": round(
                    balanced_accuracy(sub.M1.values, sub.gap.values)
                    - balanced_accuracy(sub.M0plus.values, sub.gap.values),
                    3)}
    return out


# ---------------------------------------------------------------------------
# Section 3.7: paradigm-stratified crossings + saturation.
# ---------------------------------------------------------------------------

def paradigm_crossings(cells: pd.DataFrame) -> dict:
    out = {}
    rng = np.random.default_rng(91)
    fine_all = np.linspace(cells.d.min(), cells.d.max(), 301)
    pooled_curves = {}
    for p in PARADIGMS:
        sub = cells[cells.paradigm == p]
        ck = sub.groupby("cell").var_collapse.first().sort_values()
        mat = sub[np.abs(sub.gap) >= MATERIALITY]
        rec = {"n_ckpt": int(len(ck)),
               "vc_range": [round(float(ck.min()), 4),
                            round(float(ck.max()), 4)],
               "n_severities": int(sub.d.nunique()),
               "n_material": int(len(mat)),
               "frac_positive": round(float((mat.gap > 0).mean()), 3)}
        reasons = []
        if len(ck) < 20:
            reasons.append("fewer than 20 checkpoints")
        if sub.d.nunique() < 5:
            reasons.append("fewer than 5 severities")
        if mat.gap.gt(0).nunique() < 2:
            reasons.append("single winner class among material cells")
        if len(ck) // 3 < 5:
            reasons.append("fewer than 5 checkpoints per tertile")
        data: dict[str, list] = {}
        for r in sub.itertuples():
            data.setdefault(r.cell, []).append((float(r.d), float(r.gap)))
        pooled_curves[p] = data
        if reasons:
            rec["verdict"] = "INFEASIBLE: " + "; ".join(reasons)
            out[p] = rec
            continue
        ids = ck.index.to_list()
        n = len(ids)
        strata = {"strong": ids[: n // 3], "middle": ids[n // 3: 2 * n // 3],
                  "weak": ids[2 * n // 3:]}
        res = {}
        for name, members in strata.items():
            d2 = {c: data[c] for c in members}
            res[name] = analyze_curve("pava", d2, sorted(d2), fine_all,
                                      B_BOOT, rng)
        rec["pooled"] = analyze_curve("pava", data, sorted(data), fine_all,
                                      B_BOOT, rng)
        rec["strata"] = {k: {"first_up_crossing": v.get("first_up_crossing"),
                             "display": crossing_display(v),
                             "tie_region": v.get("tie_region"),
                             "n_ckpt": len(strata[k])}
                         for k, v in res.items()}
        rec["ordering_retained"] = bool(ordering_retained(res))
        rec["verdict"] = ("RETAINED" if rec["ordering_retained"]
                          else "NOT-RETAINED")
        out[p] = rec
        print(f"[crossing] {p} done", flush=True)
    # equal-paradigm-weighted pooled curve
    def eq_curve(datasets: dict[str, dict]) -> np.ndarray:
        curves = []
        for p, data in datasets.items():
            acc: dict[float, list[float]] = {}
            for c in data:
                for d, g in data[c]:
                    acc.setdefault(d, []).append(g)
            ds = np.array(sorted(acc))
            w = np.array([len(acc[d]) for d in ds], float)
            ym = np.array([float(np.mean(acc[d])) for d in ds])
            curves.append(np.interp(fine_all, ds, pava_inc(ym, w)))
        return np.mean(curves, axis=0)

    g0 = eq_curve(pooled_curves)
    devs = np.empty(B_BOOT)
    for i in range(B_BOOT):
        boot = {}
        for p, data in pooled_curves.items():
            ids = np.array(sorted(data))
            sel = rng.choice(ids, len(ids), replace=True)
            boot[p] = {f"{c}#{j}": data[c] for j, c in enumerate(sel)}
        devs[i] = np.nanmax(np.abs(eq_curve(boot) - g0))
    q = float(np.quantile(devs, 0.95))
    sgn = np.sign(g0)
    ups = [float(fine_all[i] + (fine_all[i + 1] - fine_all[i])
                 * g0[i] / (g0[i] - g0[i + 1]))
           for i in range(len(sgn) - 1)
           if sgn[i] < 0 and sgn[i + 1] > 0]
    inside = np.abs(g0) <= q
    out["equal_paradigm_pooled"] = {
        "first_up_crossing": round(ups[0], 3) if ups else None,
        "band_q95": round(q, 3),
        "tie_region": [round(float(fine_all[inside].min()), 3),
                       round(float(fine_all[inside].max()), 3)]
        if inside.any() else None}
    return out


def paradigm_saturation(cells: pd.DataFrame) -> dict:
    fr = pd.read_parquet(CACHE).dropna(subset=["pred_gap"]).copy()
    fr["paradigm"] = fr.cell.str.split("|").str[0]
    out = {}
    for p, g in fr.groupby("paradigm"):
        margin = np.abs(g.auroc_E - g.auroc_C)
        mat = g[np.abs(g.obs_gap) >= MATERIALITY]
        out[p] = {
            "n_cells": int(len(g)), "n_material": int(len(mat)),
            "analytic_sign_acc_material": round(
                accuracy(mat.pred_gap.values, mat.obs_gap.values), 3),
            "frac_margin_zero": round(float((margin == 0).mean()), 3),
            "frac_both_above_099": round(float(
                ((g.auroc_E > 0.99) & (g.auroc_C > 0.99)).mean()), 3),
            "ga_quantiles": [round(float(np.quantile(g.ga, q)), 3)
                             for q in (0.05, 0.5, 0.95)]}
    return out


# ---------------------------------------------------------------------------
# Section 3.8: leave-one-paradigm-out transport.
# ---------------------------------------------------------------------------

def lopo(cells: pd.DataFrame) -> dict:
    out = {}
    for held in PARADIGMS:
        te = cells.paradigm == held
        train, test = cells[~te], cells[te]
        sev = severity_only(train, test)
        geo = geometry_model(train, test, flexible=False)
        mat_mask = np.abs(test.gap.values) >= MATERIALITY
        obs = test.gap.values[mat_mask]
        tr_ck = ckpt_frame(train, GEO_COLS)
        te_ck = ckpt_frame(test, GEO_COLS)
        oos = []
        for src in te_ck.source.unique():
            tr_s = tr_ck[tr_ck.source == src][GEO_COLS]
            for _, r in te_ck[te_ck.source == src].iterrows():
                oos.append(bool(np.any(r[GEO_COLS].values < tr_s.min().values)
                                or np.any(r[GEO_COLS].values
                                          > tr_s.max().values)))
        out[held] = {
            "n_material": int(mat_mask.sum()),
            "frac_positive": round(float((obs > 0).mean()), 3),
            "severity_bal": round(balanced_accuracy(sev[mat_mask], obs), 3),
            "geometry_bal": round(balanced_accuracy(geo[mat_mask], obs), 3),
            "severity_sign": round(accuracy(sev[mat_mask], obs), 3),
            "geometry_sign": round(accuracy(geo[mat_mask], obs), 3),
            "geo_minus_sev_bal": round(
                balanced_accuracy(geo[mat_mask], obs)
                - balanced_accuracy(sev[mat_mask], obs), 3),
            "frac_ckpt_out_of_support": round(float(np.mean(oos)), 3)}
    wins = [p for p, r in out.items() if r["geo_minus_sev_bal"] > 0]
    fails = [p for p, r in out.items() if r["geo_minus_sev_bal"] < -0.05]
    out["declared_rule"] = ("SUCCEEDS" if len(wins) >= 2 and not fails
                            else "FAILS")
    return out


# ---------------------------------------------------------------------------

def render(sup, js, cr, sat, lp) -> str:
    L = ["# Post-hoc source-, training-paradigm-, and "
         "training-hyperparameter-confounding sensitivity",
         "",
         "Audit #7 (2026-08-26) sections 3.4-3.8. POST HOC: specified after "
         "Stage-2 outcomes were inspected; the frozen specification is in "
         "joint_confound_audit.py's header and was not tuned afterward. The "
         "frozen Stage-2 results remain the evidence of record.", ""]
    L += ["## Section 3.4: support and positivity", "",
          "| source/paradigm | ckpt | mat | frac+ | 1-class | sev | do | "
          "var_collapse 5/50/95 |", "|---|---|---|---|---|---|---|---|"]
    for sp, r in sup["cells"].items():
        L.append(f"| {sp} | {r['n_ckpt']} | {r['n_material']} | "
                 f"{r['frac_positive']} | "
                 f"{'Y' if r['single_winner_class'] else ''} | "
                 f"{r['n_severities']} | {r['dropout_frac']} | "
                 f"{r['var_collapse']} |")
    L += ["", "DG reward strata (source/reward: ckpt, material, frac+, "
          "median var_collapse):"]
    for k, r in sup["dg_reward"].items():
        L.append(f"- {k}: {r['n_ckpt']} ckpt, {r['n_material']} mat, "
                 f"frac+ {r['frac_positive']}, vc {r['var_collapse_median']}")
    L += ["", "Paradigm contrasts within source (per G feature: "
          "standardized mean difference / range overlap):"]
    for k, rec in sup["contrasts"].items():
        parts = [f"{c}: {v['smd']}/{v['range_overlap']}"
                 for c, v in rec.items()]
        L.append(f"- {k}: " + ", ".join(parts))
    L += ["", "## Sections 3.5-3.6: joint models (out-of-fold, "
          "material cells)", "",
          f"n material {js['n_material']}, frac positive "
          f"{js['frac_positive']}, material cells on out-of-support "
          f"checkpoints {js['frac_material_out_of_support']}", "",
          "| model | bal macro | bal macro 2-class | bal row | sign macro | "
          "sign row |", "|---|---|---|---|---|---|"]
    for m in MODEL_ORDER:
        a = js["arms"][m]
        L.append(f"| {m} | {a['bal_macro']} | {a['bal_macro_two_class']} | "
                 f"{a['bal_row']} | {a['sign_macro']} | {a['sign_row']} |")
    L += ["", "| comparison | bal macro (CI95) | bal row (CI95) | "
          "sign row (CI95) |", "|---|---|---|---|"]
    for k, r in js["differences"].items():
        L.append(f"| {k} | {r['bal_macro']['point']:+.3f} "
                 f"{r['bal_macro']['ci95']} | {r['bal_row']['point']:+.3f} "
                 f"{r['bal_row']['ci95']} | {r['sign_row']['point']:+.3f} "
                 f"{r['sign_row']['ci95']} |")
    for key, title in (("per_source", "Per source"),
                       ("per_paradigm", "Per paradigm")):
        L += ["", f"{title} (balanced accuracy):", "",
              "| group | n mat | frac+ | M0+ | M1 | MH | MHG |",
              "|---|---|---|---|---|---|---|"]
        for v, r in js[key].items():
            L.append(f"| {v} | {r['n_material']} | {r['frac_positive']} | "
                     f"{r['M0plus_bal']} | {r['M1_bal']} | {r['MH_bal']} | "
                     f"{r['MHG_bal']} |")
    L += ["", "Influence (M1 - M0+ after dropping each group; "
          "macro / row):"]
    for k, r in js["influence"].items():
        L.append(f"- {k}: {r['M1-M0plus_bal_macro']:+.3f} / "
                 f"{r['M1-M0plus_bal_row']:+.3f}")
    L += ["", "## Section 3.7: paradigm-stratified crossings", ""]
    for p in PARADIGMS:
        r = cr[p]
        L.append(f"### {p}: {r['verdict']}")
        L.append(f"- ckpt {r['n_ckpt']}, vc range {r['vc_range']}, "
                 f"severities {r['n_severities']}, material "
                 f"{r['n_material']}, frac+ {r['frac_positive']}")
        if "strata" in r:
            for k, v in r["strata"].items():
                L.append(f"- {k}: crossing {v['display']}, tie "
                         f"{v['tie_region']}, ckpt {v['n_ckpt']}")
            L.append(f"- pooled first up-crossing "
                     f"{r['pooled'].get('first_up_crossing')}, tie "
                     f"{r['pooled'].get('tie_region')}")
        L.append("")
    ep = cr["equal_paradigm_pooled"]
    L += [f"Equal-paradigm-weighted pooled curve: first up-crossing "
          f"{ep['first_up_crossing']}, tie region {ep['tie_region']} "
          f"(band q95 {ep['band_q95']}).", "",
          "Saturation by paradigm (frozen theory cache):", "",
          "| paradigm | cells | mat | sign acc | zero margin | both>0.99 | "
          "gamma*a 5/50/95 |", "|---|---|---|---|---|---|---|"]
    for p, r in sat.items():
        L.append(f"| {p} | {r['n_cells']} | {r['n_material']} | "
                 f"{r['analytic_sign_acc_material']} | "
                 f"{r['frac_margin_zero']} | {r['frac_both_above_099']} | "
                 f"{r['ga_quantiles']} |")
    L += ["", "## Section 3.8: leave-one-paradigm-out transport", "",
          "| held-out paradigm | n mat | frac+ | sev bal | geo bal | "
          "G-S bal | sev sign | geo sign | out-of-support |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in PARADIGMS:
        r = lp[p]
        L.append(f"| {p} | {r['n_material']} | {r['frac_positive']} | "
                 f"{r['severity_bal']} | {r['geometry_bal']} | "
                 f"{r['geo_minus_sev_bal']:+.3f} | {r['severity_sign']} | "
                 f"{r['geometry_sign']} | {r['frac_ckpt_out_of_support']} |")
    L += ["", f"Declared transport rule: {lp['declared_rule']} (three "
          "results reported individually; never a paradigm population).", ""]
    return "\n".join(L)


def main() -> None:
    cells = build_cells_with_severity()
    coords, problems = load_coords(Path("pilot0/pool_coords"))
    assert not problems, problems
    cells = add_metadata(cells, coords)
    print(f"cells {len(cells)}, checkpoints {cells.cell.nunique()}",
          flush=True)

    sup = support_audit(cells)
    print("[3.4] support audit done", flush=True)
    fitted = run_joint(cells)
    js = joint_summary(fitted)
    print("[3.5/3.6] joint models done", flush=True)
    cr = paradigm_crossings(cells)
    sat = paradigm_saturation(cells)
    print("[3.7] crossings done", flush=True)
    lp = lopo(cells)
    print("[3.8] transport done", flush=True)

    md = render(sup, js, cr, sat, lp)
    (OUT_DIR / "joint_confound_audit_report.md").write_text(md)
    (OUT_DIR / "joint_confound_audit_report.json").write_text(
        json.dumps({"support": sup, "joint": js, "crossings": cr,
                    "saturation": sat, "lopo": lp}, indent=1, default=str))
    print(md)


if __name__ == "__main__":
    main()

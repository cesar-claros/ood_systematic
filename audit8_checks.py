"""Audit #8 submission-blocking checks R1, R2, R4, R5, R7 (evaluation doc
companion_phase_diagram_paper_report_evaluation_and_next_steps_2026-08-28,
sections 10 and 13; R6 runs on the HPC via
pilot0/split_integrity_certificate.py).

All checks are POST HOC sensitivities. They reuse frozen artifacts and the
registered machinery unchanged. Nothing replaces a frozen primary result.

R1 (two-class-only joint-audit interval): rebuild the joint-audit
out-of-fold predictions (identical folds, seed 20260826), restrict the
macro to the source x paradigm cells with BOTH observed material winner
classes, and report M1 - M0+ with a paired checkpoint bootstrap
(stratified within cells, B = 2000, seed 911).

R2 (categorical DG-reward sensitivity): replace the frozen quadratic
per-source log-reward basis with source-specific categorical DG reward
indicators; keep folds, ridge lam, all other features, materiality, and
bootstrap machinery unchanged. Report M0+cat - M0 and M1cat - M0+cat
(all-cell macro, two-class macro, row), plus leave-one-reward-level
influence on the M1cat - M0+cat point estimate. Both models are reported;
neither replaces the frozen quadratic primary.

R4 (BREEDS rank uncertainty): for the 28 BREEDS cells, Spearman between
|analytic margin| and |balanced gap| with a checkpoint bootstrap
(B = 2000, seed 921), a checkpoint-unit permutation test (B = 10000, seed
977, one-sided for positive association), a DG-only estimate with
bootstrap interval, leave-one-DG-reward-level-out influence, and a
paradigm table with ConfidNet and DeVries marked underpowered (n = 4).

R5 (analytic-margin numerical stability): margins recomputed after (a)
rounding all measured inputs to float32 and (b) relative +-1e-6 coordinate
perturbations (seed 931); compare BREEDS sign and rank stability, and
compare margin magnitudes against a conservative numerical-error scale
(1e-12). ImageNet-200 margins (~1e-9) stay classified as degenerate.

R7 (matched correlation decomposition): within-shift/across-checkpoint
rank correlations per (source, shift) pair, and within-checkpoint/
across-shift correlations per checkpoint, computed separately for the
original pool (theory cache; |obs raw gap|), SVHN, and BREEDS (balanced
gaps); ImageNet-200 reported per checkpoint across its five shifts. Pairs
need >= 10 nonzero-margin units; others are reported as degenerate. BREEDS
is compared only against the pool's within-shift distribution.

Usage (from code/): python audit8_checks.py
Output: nc_csf_predictivity/outputs/track1/audit8_checks_report.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import (GEO_COLS, MATERIALITY,
                                       balanced_accuracy, load_coords,
                                       ridge_logistic)
from joint_confound_audit import (H_COLS, PARADIGMS, SOURCES, add_metadata,
                                  centered, group_stats, macro_metric,
                                  make_folds, run_joint)
from stage2_closure import CACHE, build_cells_with_severity
from stage2_expansion_analysis import theory_pair

B_BOOT = 2000
B_PERM = 10000


# ---------------------------------------------------------------------------
# R1 + R2: joint-audit sensitivities.
# ---------------------------------------------------------------------------

def features_cat(fr, model, gstats, hstats, reward_levels) -> np.ndarray:
    d = fr.d.values
    feats = [d, d * d]
    src_oh = {s: (fr.source == s).values.astype(float) for s in SOURCES}
    par_oh = {p: (fr.paradigm == p).values.astype(float) for p in PARADIGMS}
    feats += list(src_oh.values()) + list(par_oh.values())
    feats += [src_oh[s] * par_oh[p] for s in SOURCES for p in PARADIGMS]
    feats += [d * src_oh[s] for s in SOURCES]
    feats += [d * par_oh[p] for p in PARADIGMS]
    is_dg = par_oh["dg"]
    feats.append(fr.dropout.values)
    for s in SOURCES:
        for r in reward_levels[s]:
            feats.append(((fr.source == s).values
                          & (fr.reward == r).values).astype(float) * is_dg)
    if model == "M1cat":
        gt = centered(fr, GEO_COLS, gstats)
        feats += [gt[:, j] for j in range(gt.shape[1])]
        feats += [d * gt[:, j] for j in range(gt.shape[1])]
    return np.column_stack(feats)


def fit_cat(train, test, model, gstats, hstats, reward_levels):
    xtr = features_cat(train, model, gstats, hstats, reward_levels)
    xte = features_cat(test, model, gstats, hstats, reward_levels)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    beta = ridge_logistic((xtr - mu) / sd,
                          (train.gap.values > 0).astype(float), lam=1.0)
    return beta[0] + ((xte - mu) / sd) @ beta[1:]


def two_class_cells(mat: pd.DataFrame) -> set:
    return {sp for sp, g in mat.groupby("sp")
            if len(g) >= 10 and g.gap.gt(0).nunique() == 2}


def macro_two_class(mat, arm, cells2):
    vals = [balanced_accuracy(g[arm].values, g.gap.values)
            for sp, g in mat.groupby("sp") if sp in cells2]
    return float(np.mean(vals)) if vals else float("nan")


def boot_diff(mat, arm_a, arm_b, stat, seed) -> dict:
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
        fr = pd.concat(parts)
        diffs[i] = stat(fr, arm_a) - stat(fr, arm_b)
    return {"point": round(stat(mat, arm_a) - stat(mat, arm_b), 3),
            "ci95": [round(float(np.quantile(diffs, 0.025)), 3),
                     round(float(np.quantile(diffs, 0.975)), 3)]}


def r1_r2() -> tuple[dict, dict, pd.DataFrame]:
    cells = build_cells_with_severity()
    coords, problems = load_coords(Path("pilot0/pool_coords"))
    assert not problems, problems
    cells = add_metadata(cells, coords)
    fitted = run_joint(cells)
    # categorical-reward variants on the SAME folds
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
        for m in ("M0pluscat", "M1cat"):
            fitted.loc[te, m] = fit_cat(train, test, m, gstats, hstats,
                                        reward_levels)
        print(f"[r2] fold {k} done", flush=True)
    mat = fitted[np.abs(fitted.gap) >= MATERIALITY].dropna(
        subset=["M0plus", "M1", "M0pluscat", "M1cat"]).copy()
    cells2 = two_class_cells(mat)

    def stat_2c(fr, arm):
        return macro_two_class(fr, arm, cells2)

    def stat_macro(fr, arm):
        return macro_metric(fr, arm, balanced_accuracy)

    def stat_row(fr, arm):
        return balanced_accuracy(fr[arm].values, fr.gap.values)

    r1 = {"two_class_cells": sorted(cells2),
          "M1-M0plus_two_class_macro": boot_diff(mat, "M1", "M0plus",
                                                 stat_2c, 911)}
    r2 = {"arms": {m: {"bal_macro": round(stat_macro(mat, m), 3),
                       "bal_two_class_macro": round(stat_2c(mat, m), 3),
                       "bal_row": round(stat_row(mat, m), 3)}
                   for m in ("M0", "M0plus", "M1", "M0pluscat", "M1cat")},
          "M0pluscat-M0": {
              "bal_macro": boot_diff(mat, "M0pluscat", "M0", stat_macro,
                                     912),
              "bal_row": boot_diff(mat, "M0pluscat", "M0", stat_row, 913)},
          "M1cat-M0pluscat": {
              "bal_macro": boot_diff(mat, "M1cat", "M0pluscat", stat_macro,
                                     914),
              "bal_two_class_macro": boot_diff(mat, "M1cat", "M0pluscat",
                                               stat_2c, 915),
              "bal_row": boot_diff(mat, "M1cat", "M0pluscat", stat_row,
                                   916)}}
    infl = {}
    for r in sorted(mat[mat.paradigm == "dg"].reward.unique()):
        keep = ~((mat.paradigm == "dg") & (mat.reward == r))
        sub = mat[keep]
        infl[f"drop_rew{r:g}"] = round(
            stat_macro(sub, "M1cat") - stat_macro(sub, "M0pluscat"), 3)
    r2["leave_one_reward_influence_M1cat-M0pluscat_macro"] = infl
    return r1, r2, mat


# ---------------------------------------------------------------------------
# R4 + R5 + R7 helpers: expansion cells with margins.
# ---------------------------------------------------------------------------

def expansion_cells(perturb: str | None = None,
                    seed: int = 931) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cache: dict = {}
    for d_, forced_src in (("pilot0/stage2_expansion_coords", None),
                           ("pilot0/stage3_imagenet200_coords",
                            "imagenet200")):
        for p in sorted(Path(d_).glob("*.json")):
            if p.name.startswith("FAILED"):
                continue
            r = json.loads(p.read_text())
            src = forced_src or r["source"]
            c, dim = int(r["n_classes"]), int(r["dim"])
            vc = r["papyan"]["var_collapse"]
            sd = r["papyan"]["self_duality"]
            lsc = r["geometry"]["logit_scale"]
            eta = r["geometry"]["class_mean_radius_cv"]
            for name, e in r["ood"].items():
                if "error" in e:
                    continue
                vals = [vc, sd, lsc, eta, e["gamma"], e["a"], e["rho"]]
                if perturb == "float32":
                    vals = [float(np.float32(v)) for v in vals]
                elif perturb == "rel1e-6":
                    vals = [v * (1 + rng.uniform(-1e-6, 1e-6))
                            for v in vals]
                vc_, sd_, lsc_, eta_, g_, a_, rho_ = vals
                s_dict = (c - 1) / np.sqrt(c * max(vc_, 1e-9))
                theta = float(np.degrees(np.arccos(
                    np.clip(1 - sd_ / 2, -1, 1))))
                ae, ac = theory_pair(c, dim, s_dict, theta, lsc_, eta_,
                                     g_, a_, rho_, cache)
                rows.append({
                    "source": src,
                    "paradigm": r.get("paradigm", "crossentropy"),
                    "reward": float(r.get("reward", 0) or 0),
                    "slug": r.get("slug") or r.get("run"), "set": name,
                    "margin": abs(ae - ac), "pred_sign": np.sign(-(ae - ac)),
                    "obs": e["gap_balanced"]})
    return pd.DataFrame(rows)


def rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def r4(br: pd.DataFrame) -> dict:
    x, y = br.margin.values, np.abs(br.obs.values)
    point = rho(x, y)
    rng = np.random.default_rng(921)
    boots = np.array([rho(x[i], y[i]) for i in
                      (rng.choice(len(x), len(x), replace=True)
                       for _ in range(B_BOOT))])
    rngp = np.random.default_rng(977)
    perms = np.array([rho(rngp.permutation(x), y) for _ in range(B_PERM)])
    p_perm = float((1 + (perms >= point).sum()) / (B_PERM + 1))
    dg = br[br.paradigm == "dg"]
    xd, yd = dg.margin.values, np.abs(dg.obs.values)
    boots_dg = np.array([rho(xd[i], yd[i]) for i in
                         (rng.choice(len(xd), len(xd), replace=True)
                          for _ in range(B_BOOT))])
    infl = {}
    for r_ in sorted(dg.reward.unique()):
        keep = ~((br.paradigm == "dg") & (br.reward == r_))
        infl[f"drop_rew{r_:g}"] = round(
            rho(br[keep].margin.values, np.abs(br[keep].obs.values)), 3)
    par = {}
    for p_, g in br.groupby("paradigm"):
        par[p_] = {"n": int(len(g)),
                   "rho": round(rho(g.margin.values, np.abs(g.obs.values)),
                                3),
                   "underpowered": bool(len(g) < 10)}
    return {"n": int(len(br)), "spearman": round(point, 3),
            "boot_ci95": [round(float(np.quantile(boots, 0.025)), 3),
                          round(float(np.quantile(boots, 0.975)), 3)],
            "perm_p_one_sided": round(p_perm, 5),
            "dg_only": {"n": int(len(dg)),
                        "rho": round(rho(xd, yd), 3),
                        "boot_ci95": [
                            round(float(np.quantile(boots_dg, 0.025)), 3),
                            round(float(np.quantile(boots_dg, 0.975)), 3)]},
            "leave_one_reward_rho": infl, "per_paradigm": par}


def r5(base: pd.DataFrame) -> dict:
    out = {}
    for tag in ("float32", "rel1e-6"):
        alt = expansion_cells(perturb=tag)
        m = base.merge(alt, on=["slug", "set"], suffixes=("", "_alt"))
        br = m[m.source == "breeds"]
        out[tag] = {
            "breeds_sign_flips": int((br.pred_sign
                                      != br.pred_sign_alt).sum()),
            "breeds_rank_spearman_base_vs_alt": round(
                rho(br.margin.values, br.margin_alt.values), 4),
            "max_rel_margin_change_breeds": round(float(
                (np.abs(br.margin - br.margin_alt)
                 / np.clip(br.margin, 1e-30, None)).max()), 4)}
    br = base[base.source == "breeds"]
    out["magnitude_vs_error"] = {
        "breeds_min_margin": float(br.margin.min()),
        "numerical_error_scale": 1e-12,
        "imagenet200_median_margin": float(
            base[base.source == "imagenet200"].margin.median()),
        "verdict": "BREEDS margins >> numerical error; ImageNet-200 "
                   "margins remain classified degenerate"}
    return out


def r7(exp: pd.DataFrame) -> dict:
    pool = pd.read_parquet(CACHE).dropna(subset=["pred_gap"])
    pool["margin"] = (pool.auroc_E - pool.auroc_C).abs()
    out: dict = {}

    def within_shift(fr, obs_col) -> list:
        vals = []
        for (s, e), g in fr.groupby(["source", "set"]
                                    if "set" in fr else
                                    ["source", "eval_dataset"]):
            nz = g[g.margin > 0]
            if len(nz) >= 10:
                vals.append(round(rho(nz.margin.values,
                                      np.abs(nz[obs_col].values)), 3))
        return vals

    def within_ckpt(fr, obs_col, id_col) -> list:
        vals = []
        for _, g in fr.groupby(id_col):
            nz = g[g.margin > 0]
            if len(nz) >= 5:
                vals.append(round(rho(nz.margin.values,
                                      np.abs(nz[obs_col].values)), 3))
        return vals

    ws = within_shift(pool, "obs_gap")
    wc = within_ckpt(pool, "obs_gap", "cell")
    out["pool"] = {
        "within_shift_across_ckpt": {
            "n_pairs_usable": len(ws), "n_pairs_total": 32,
            "median": round(float(np.median(ws)), 3) if ws else None,
            "iqr": ([round(float(np.quantile(ws, q)), 3)
                     for q in (0.25, 0.75)] if ws else None),
            "values": ws},
        "within_ckpt_across_shift": {
            "n_ckpt_usable": len(wc), "n_ckpt_total": 280,
            "median": round(float(np.median(wc)), 3) if wc else None}}
    sv = exp[exp.source == "svhn"]
    out["svhn_within_shift_across_ckpt"] = within_shift(sv, "obs")
    out["breeds_within_shift_across_ckpt"] = [
        round(rho(exp[exp.source == "breeds"].margin.values,
                  np.abs(exp[exp.source == "breeds"].obs.values)), 3)]
    inn = exp[exp.source == "imagenet200"]
    out["imagenet200_per_ckpt_across_5_shifts"] = {
        s: round(rho(g.margin.values, np.abs(g.obs.values)), 3)
        for s, g in inn.groupby("slug")}
    return out


def main() -> None:
    r1, r2_, _ = r1_r2()
    exp = expansion_cells()
    br = exp[exp.source == "breeds"].reset_index(drop=True)
    r4_ = r4(br)
    r5_ = r5(exp)
    r7_ = r7(exp)
    out = {"R1": r1, "R2": r2_, "R4": r4_, "R5": r5_, "R7": r7_}
    L = ["# Audit-8 checks R1/R2/R4/R5/R7 (post-hoc sensitivities; frozen "
         "primaries unchanged)", ""]
    L += [f"## R1 two-class-only joint-audit interval",
          f"- two-class cells: {r1['two_class_cells']}",
          f"- M1 - M0+ two-class macro: "
          f"{r1['M1-M0plus_two_class_macro']}", ""]
    L += ["## R2 categorical DG-reward sensitivity", "",
          "| model | bal macro | bal 2-class macro | bal row |",
          "|---|---|---|---|"]
    for m, v in r2_["arms"].items():
        L.append(f"| {m} | {v['bal_macro']} | {v['bal_two_class_macro']} | "
                 f"{v['bal_row']} |")
    for k in ("M0pluscat-M0", "M1cat-M0pluscat"):
        L.append(f"- {k}: " + ", ".join(
            f"{kk} {vv['point']:+.3f} {vv['ci95']}"
            for kk, vv in r2_[k].items()))
    L.append(f"- leave-one-reward influence (M1cat-M0pluscat macro): "
             f"{r2_['leave_one_reward_influence_M1cat-M0pluscat_macro']}")
    L += ["", "## R4 BREEDS rank uncertainty",
          f"- Spearman {r4_['spearman']} (n={r4_['n']}), boot CI95 "
          f"{r4_['boot_ci95']}, permutation p {r4_['perm_p_one_sided']}",
          f"- DG-only: {r4_['dg_only']}",
          f"- leave-one-reward rho: {r4_['leave_one_reward_rho']}",
          f"- per paradigm: {r4_['per_paradigm']}", ""]
    L += ["## R5 numerical stability", f"- {json.dumps(r5_, indent=1)}", ""]
    L += ["## R7 matched correlation decomposition",
          f"- pool within-shift/across-ckpt: "
          f"{r7_['pool']['within_shift_across_ckpt']}",
          f"- pool within-ckpt/across-shift: "
          f"{r7_['pool']['within_ckpt_across_shift']}",
          f"- svhn within-shift values: "
          f"{r7_['svhn_within_shift_across_ckpt']}",
          f"- breeds within-shift value: "
          f"{r7_['breeds_within_shift_across_ckpt']}",
          f"- imagenet200 per-ckpt: "
          f"{r7_['imagenet200_per_ckpt_across_5_shifts']}", ""]
    (OUT_DIR / "audit8_checks_report.md").write_text("\n".join(L))
    (OUT_DIR / "audit8_checks_report.json").write_text(
        json.dumps(out, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()

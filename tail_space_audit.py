"""Phase-0 stable tail-space audit of the frozen Stage-2 theory arm
(analytic-saturation diagnosis plan 2026-08-28, sections 4, 5, and 12).

EVIDENCE CLASS (plan section 5.1): post-outcome numerical-correctness and
sensitivity audit. The stable expression is mathematically equivalent to
the frozen formula comparison; no hyperparameter is selected; the frozen
Stage-2 arm, cache, materiality, and folds are reused unchanged. This
audit determines whether the reported exact-zero winner margins are
finite-precision CDF ties (Phi(z) rounding to 1.0 before subtraction) or
genuine model saturation.

FROZEN SPECIFICATION (declared in this header before any outcome of the
tail computation was inspected):
- Cells: the frozen 2,240 checkpoint-shift cells with severity
  (stage2_closure.build_cells_with_severity), frozen coordinates
  (pilot0/pool_coords), and the exact stage2_closure.theory_full clamps
  (s >= 3, theta in [0, 85], logit >= 1e-3, eta in [0, 0.5], ga >= 1e-4,
  a in [1e-3, 0.999], rho >= 0.05); build_config_model(seed=0).
- Tail statistic: l_j = logsumexp_y(log freq_y + log_ndtr(-z_jy)) per
  detector from the component z values that reproduce the frozen mixture
  AUROCs term by term; M_tail = l_E - l_C, whose sign equals the
  predicted AUGRC-gap sign (positive favors CTM) in exact arithmetic.
- Materiality and folds: |gap| >= 10 AUGRC-milli units (frozen), grouped
  5-fold by checkpoint with seed 2027, run_folds/summarize machinery of
  the frozen held-out script for the severity-only and train-fold-mean
  comparators.
- Uncertainty: checkpoint-cluster bootstrap B = 2000, seed 1009.
- Numerical sensitivity (plan section 5.4): (a) full recomputation from
  float32-rounded measured inputs; (b) K = 20 draws of independent
  multiplicative relative perturbations x -> x (1 + 1e-6 u),
  u ~ Uniform(-1, 1), seed 1013, applied to the raw measured inputs
  (s, theta, logit scale, eta, gamma, a, rho) before clamping. The
  perturbation passes use an isotropic-equivalent CTM component path
  (algebraically identical under cov = sigma^2 I; its unperturbed signs
  are verified against the full path and any disagreement is reported).
  Per-cell stability = fraction of draws whose M_tail sign equals the
  unperturbed sign; selective accuracy is reported on the declared,
  untuned grid tau in {0.5, 0.8, 0.95, 1.0}. Extraction-derived
  scientific uncertainty is NOT available per cell without re-extraction
  and is recorded as unavailable.
- High precision: mpmath dps = 50 recomputation of l_E, l_C, M_tail for
  the 10 material cells with smallest |M_tail| plus 10 random cells
  (seed 1021).
- Continuity: recomputed direct AUROCs must match the frozen cache
  theory_cell_predictions.parquet exactly; the max deviation is reported.

Decision rules are the plan's section 5.5 (N1-N4); they are applied after
the numbers exist, not encoded here.

Usage (from code/):
  python tail_space_audit.py --self_test   # section-12 minimum unit tests
  python tail_space_audit.py               # full audit (~30-60 min)
Outputs: nc_csf_predictivity/outputs/track1/tail_space_audit_report.md/.json
         + tail_space_audit_cells.parquet (new cache; the frozen
         theory_cell_predictions.parquet is never overwritten)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, dictionary_params,
                                       load_coords, map_ood_names, run_folds)
from mc_phase_audit import BASE, build_config_model
from pilot0.theory import (HeadContext, NoiseModel, ctm_mean_z_components,
                           ctm_stats, head_z_components,
                           log_error_probability, predicted_aurocs,
                           predicted_ctm_mean_auroc)
from stage2_closure import CACHE as FROZEN_CACHE
from stage2_closure import build_cells_with_severity

from pathlib import Path

TAIL_CACHE = OUT_DIR / "tail_space_audit_cells.parquet"
B_BOOT, BOOT_SEED, PERT_SEED, HP_SEED, K_PERT = 2000, 1009, 1013, 1021, 20
TAU_GRID = (0.5, 0.8, 0.95, 1.0)


def frozen_cfg(s, theta, logit, eta, gamma, a, rho) -> dict:
    """The exact stage2_closure clamp set."""
    return dict(BASE, s=max(float(s), 3.0),
                theta_deg=float(np.clip(theta, 0, 85)),
                logit_target=max(float(logit), 1e-3),
                eta_std=float(np.clip(eta, 0.0, 0.5)),
                ga=float(np.clip(gamma * a, 1e-4, None)),
                a=float(np.clip(a, 1e-3, 0.999)),
                rho=float(np.clip(rho, 0.05, None)))


def tail_pair(c: int, d: int, cfg: dict, fast_ctm: bool = False):
    """(A_E, A_C, l_E, l_C) at one cell's coordinates.

    fast_ctm uses the isotropic-equivalent CTM component path (no DxD
    matrices); the direct AUROCs are then omitted (None).
    """
    m = build_config_model(c, d, cfg, seed=0)
    ctx = HeadContext.from_head(m["w"], m["b"])
    dim = m["means"].shape[1]
    nid = NoiseModel.isotropic(m["sigma"], ctx, dim)
    nood = NoiseModel.isotropic(cfg["rho"] * m["sigma"], ctx, dim)
    wz = head_z_components(m["means"], m["class_freq"], nid, m["m_ood"],
                           nood, ctx)
    w_e, z_e = wz["Energy"]
    l_e = log_error_probability(w_e, z_e)
    if fast_ctm:
        w_c, z_c = ctm_iso_z(m["means"], m["class_freq"], m["sigma"],
                             cfg["rho"] * m["sigma"], m["m_ood"], dim)
        return None, None, l_e, log_error_probability(w_c, z_c)
    a_e = float(predicted_aurocs(m["means"], m["class_freq"], nid,
                                 m["m_ood"], nood, ctx)["Energy"])
    a_c = float(predicted_ctm_mean_auroc(m["means"], m["class_freq"],
                                         m["cov_id"], m["m_ood"],
                                         m["cov_ood"]))
    w_c, z_c = ctm_mean_z_components(m["means"], m["class_freq"],
                                     m["cov_id"], m["m_ood"], m["cov_ood"])
    return a_e, a_c, l_e, log_error_probability(w_c, z_c)


def ctm_iso_z(class_means, class_freq, sig_id, sig_ood, m_ood, dim):
    """Isotropic-equivalent CTM components (cov = sigma^2 I, no matrices)."""
    mu_hat = class_means / np.linalg.norm(class_means, axis=1, keepdims=True)
    n = len(class_means)
    m_o, v_o = ctm_stats(m_ood, sig_ood**2 * dim, dim, mu_hat,
                         sig_ood**2 * np.ones(n))
    z = np.empty(n)
    for i, mu_y in enumerate(class_means):
        m_y, v_y = ctm_stats(mu_y, sig_id**2 * dim, dim, mu_hat,
                             sig_id**2 * np.ones(n))
        z[i] = (m_y - m_o) / np.sqrt(v_y + v_o)
    return np.asarray(class_freq, dtype=float), z


def raw_inputs(cells: pd.DataFrame, coords: dict) -> pd.DataFrame:
    """Per-cell raw measured inputs (pre-clamp), replicating the frozen
    lookup path."""
    rows = []
    for idx, row in cells.iterrows():
        rec = coords.get(row.cell)
        out = {"idx": idx}
        if rec is not None:
            sets = map_ood_names(rec, set(cells[cells.cell == row.cell]
                                          .eval_dataset))
            co = sets.get(row.eval_dataset)
            if co is not None:
                s, theta = dictionary_params(row, rec["n_classes"])
                out.update({"C": int(rec["n_classes"]),
                            "D": int(rec["dim"]), "s": s, "theta": theta,
                            "logit": rec["geometry"]["logit_scale"],
                            "eta": rec["geometry"]["class_mean_radius_cv"],
                            "gamma": co["gamma"], "a_": co["a"],
                            "rho": co["rho"]})
        rows.append(out)
    return pd.DataFrame(rows).set_index("idx")


def main_pass(cells, raw, cast=None, fast=False, pert=None):
    """One full evaluation pass; `cast` optionally rounds inputs (e.g.
    np.float32); `pert` is a per-cell multiplicative factor array (7,)."""
    out = np.full((len(cells), 4), np.nan)
    for idx in raw.index:
        r = raw.loc[idx]
        if not np.isfinite(r.get("s", np.nan)):
            continue
        vals = np.array([r.s, r.theta, r.logit, r.eta, r.gamma, r.a_, r.rho])
        if cast is not None:
            vals = vals.astype(cast).astype(np.float64)
        if pert is not None:
            vals = vals * pert[idx]
        cfg = frozen_cfg(*vals)
        out[idx] = np.array(tail_pair(int(r.C), int(r.D), cfg,
                                      fast_ctm=fast), dtype=float)
    return out


def clustered_ci(mat, col_pred, rng):
    ckpts = np.array(sorted(mat.cell.unique()))
    groups = {c: g for c, g in mat.groupby("cell")}
    accs, diffs = np.empty(B_BOOT), np.empty(B_BOOT)
    for i in range(B_BOOT):
        fr = pd.concat([groups[c] for c in
                        rng.choice(ckpts, len(ckpts), replace=True)])
        o = fr.gap.values
        accs[i] = accuracy(fr[col_pred].values, o)
        diffs[i] = (accuracy(fr[col_pred].values, o)
                    - accuracy(fr["severity"].values, o))
    q = lambda x: [round(float(np.quantile(x, p)), 3) for p in (.025, .975)]
    return q(accs), q(diffs)


def self_test() -> None:
    from scipy.stats import norm
    import mpmath
    w = np.array([0.5, 0.5])
    # equal z -> exactly zero preference
    z = np.array([3.0, 5.0])
    assert log_error_probability(w, z) == log_error_probability(w, z.copy())
    # ordering: larger z (better detector) -> smaller log error
    assert (log_error_probability(w, z + 1.0)
            < log_error_probability(w, z))
    # moderate z: direct and tail agree in value and sign
    for za, zb in ((np.array([1.0, 2.0]), np.array([1.5, 2.5])),):
        qa = 1.0 - float(w @ norm.cdf(za))
        assert abs(np.exp(log_error_probability(w, za)) - qa) < 1e-12
        direct = float(w @ norm.cdf(zb)) - float(w @ norm.cdf(za))
        tail = (log_error_probability(w, za)
                - log_error_probability(w, zb))
        assert np.sign(direct) == np.sign(tail)
    # huge z: direct CDF saturates to a zero margin, tail stays ordered
    za, zb = np.array([50.0, 52.0]), np.array([60.0, 62.0])
    assert float(w @ norm.cdf(zb)) - float(w @ norm.cdf(za)) == 0.0
    assert log_error_probability(w, zb) < log_error_probability(w, za) < 0
    assert np.isfinite(log_error_probability(w, np.array([300.0, 400.0])))
    # class-mixture log-tail vs mpmath dps=50
    mpmath.mp.dps = 50
    z = np.array([8.0, 41.0, 13.0])
    w3 = np.array([0.2, 0.5, 0.3])
    q_mp = sum(mpmath.mpf(float(wi)) * mpmath.erfc(zi / mpmath.sqrt(2)) / 2
               for wi, zi in zip(w3, z))
    assert abs(log_error_probability(w3, z) - float(mpmath.log(q_mp))) < 1e-10
    # invariance to class order and to weight normalization
    p = np.array([2, 0, 1])
    assert abs(log_error_probability(w3[p], z[p])
               - log_error_probability(w3, z)) < 1e-12
    assert abs(log_error_probability(10 * w3, z)
               - log_error_probability(w3, z)) < 1e-12
    print("self_test: all section-12 minimum tests PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    cells = build_cells_with_severity()
    coords, _ = load_coords(Path("pilot0/pool_coords"))
    raw = raw_inputs(cells, coords)
    print(f"[tail] {len(cells)} cells; inputs resolved for "
          f"{int(np.isfinite(raw.get('s', pd.Series(dtype=float))).sum())}",
          flush=True)

    # --- unperturbed full pass (exact frozen arithmetic) ---------------
    res = main_pass(cells, raw)
    cells = cells.copy()
    cells[["auroc_E", "auroc_C", "l_E", "l_C"]] = res
    cells["m_tail"] = cells.l_E - cells.l_C   # >0 favors CTM = gap sign
    cells["direct_margin"] = cells.auroc_C - cells.auroc_E
    cells["direct_zero"] = cells.direct_margin == 0.0
    cells["tail_nonzero"] = cells.m_tail != 0.0
    cells["paradigm"] = cells.cell.str.split("|").str[0]

    # continuity vs the frozen cache
    frz = pd.read_parquet(FROZEN_CACHE)[["cell", "eval_dataset", "auroc_E",
                                         "auroc_C"]]
    mg = cells.merge(frz, on=["cell", "eval_dataset"],
                     suffixes=("", "_frz"))
    cont = {"max_abs_dev_E": float((mg.auroc_E - mg.auroc_E_frz).abs()
                                   .max()),
            "max_abs_dev_C": float((mg.auroc_C - mg.auroc_C_frz).abs()
                                   .max())}
    print(f"[tail] continuity: {cont}", flush=True)

    # --- fast-path verification + perturbation stability ---------------
    fastres = main_pass(cells, raw, fast=True)
    base_fast_sign = np.sign(fastres[:, 2] - fastres[:, 3])
    fast_disagree = int(np.nansum(base_fast_sign
                                  != np.sign(cells.m_tail.values)))
    rngp = np.random.default_rng(PERT_SEED)
    signs = np.zeros((len(cells), K_PERT))
    for k in range(K_PERT):
        pert = 1.0 + 1e-6 * rngp.uniform(-1, 1, size=(len(cells), 7))
        pr = main_pass(cells, raw, fast=True, pert=pert)
        signs[:, k] = np.sign(pr[:, 2] - pr[:, 3])
        print(f"[tail] perturbation draw {k + 1}/{K_PERT}", flush=True)
    stability = (signs == base_fast_sign[:, None]).mean(1)
    cells["stability"] = stability

    # --- float32 sensitivity (full path) --------------------------------
    r32 = main_pass(cells, raw, cast=np.float32)
    m32 = r32[:, 2] - r32[:, 3]
    flip32 = int(np.nansum(np.sign(m32) != np.sign(cells.m_tail.values)))

    # --- high-precision spot check --------------------------------------
    import mpmath
    mpmath.mp.dps = 50
    mat_mask = np.abs(cells.gap) >= MATERIALITY
    ok = cells.m_tail.notna()
    small = cells[ok & mat_mask].reindex(
        cells[ok & mat_mask].m_tail.abs().sort_values().index)[:10]
    rand = cells[ok].sample(10, random_state=HP_SEED)
    hp = []
    for idx in list(small.index) + list(rand.index):
        r = raw.loc[idx]
        cfg = frozen_cfg(r.s, r.theta, r.logit, r.eta, r.gamma, r.a_, r.rho)
        m = build_config_model(int(r.C), int(r.D), cfg, seed=0)
        ctx = HeadContext.from_head(m["w"], m["b"])
        dim = m["means"].shape[1]
        nid = NoiseModel.isotropic(m["sigma"], ctx, dim)
        nood = NoiseModel.isotropic(cfg["rho"] * m["sigma"], ctx, dim)
        w_e, z_e = head_z_components(m["means"], m["class_freq"], nid,
                                     m["m_ood"], nood, ctx)["Energy"]
        w_c, z_c = ctm_mean_z_components(m["means"], m["class_freq"],
                                         m["cov_id"], m["m_ood"],
                                         m["cov_ood"])

        def lmp(w, z):
            qq = sum(mpmath.mpf(float(wi)) / mpmath.mpf(float(w.sum()))
                     * mpmath.erfc(mpmath.mpf(float(zi)) / mpmath.sqrt(2))
                     / 2 for wi, zi in zip(w, z))
            return mpmath.log(qq)
        m_mp = float(lmp(w_e, z_e) - lmp(w_c, z_c))
        hp.append({"idx": int(idx), "m_tail_f64": float(cells.m_tail[idx]),
                   "m_tail_mp50": m_mp,
                   "sign_agrees": bool(np.sign(cells.m_tail[idx])
                                       == np.sign(m_mp))})
    hp_summary = {
        "n_checked": len(hp),
        "n_sign_agree": int(sum(h["sign_agrees"] for h in hp)),
        "max_rel_diff": float(max(
            abs(h["m_tail_f64"] - h["m_tail_mp50"])
            / max(abs(h["m_tail_mp50"]), 1e-300) for h in hp)),
    }

    # --- evaluation on frozen material cells + comparators --------------
    tail_sign = pd.Series(np.sign(cells.m_tail.values), index=cells.index)
    folded = run_folds(cells, tail_sign, "ckpt5",
                       np.random.default_rng(FOLD_SEED))
    folded["m_tail"] = cells.m_tail.values
    folded["stability"] = cells.stability.values
    folded["paradigm"] = cells.paradigm.values
    mat = folded[np.abs(folded.gap) >= MATERIALITY].copy()
    obs = mat.gap.values
    rng = np.random.default_rng(BOOT_SEED)
    acc_ci, diff_ci = clustered_ci(mat, "theory", rng)
    dz = cells.direct_zero & cells.m_tail.notna()
    report = {
        "continuity": cont,
        "n_cells": int(len(cells)), "n_material": int(len(mat)),
        "frac_direct_zero": round(float(cells.direct_zero.mean()), 4),
        "frac_direct_zero_with_nonzero_tail": round(float(
            cells[dz].tail_nonzero.mean()), 4),
        "sign_agreement_direct_nonzero_vs_tail": round(float(
            (np.sign(cells[~cells.direct_zero].direct_margin)
             == np.sign(cells[~cells.direct_zero].m_tail)).mean()), 4),
        "material_sign_acc_tail": round(accuracy(mat.theory.values, obs), 4),
        "material_sign_acc_ci95": acc_ci,
        "material_balanced_acc_tail": round(
            balanced_accuracy(mat.theory.values, obs), 4),
        "frozen_material_sign_acc_direct": 0.099,
        "tail_minus_severity": {
            "point": round(accuracy(mat.theory.values, obs)
                           - accuracy(mat.severity.values, obs), 4),
            "ci95": diff_ci},
        "severity_sign_acc": round(accuracy(mat.severity.values, obs), 4),
        "trainfold_mean_sign_acc": round(accuracy(mat["mean"].values, obs),
                                         4),
        "spearman_abs_mtail_vs_abs_gap_material": round(float(
            spearmanr(mat.m_tail.abs(), np.abs(obs)).statistic), 4),
        "spearman_abs_mtail_vs_abs_gap_all": round(float(
            spearmanr(cells.m_tail.abs(), cells.gap.abs(),
                      nan_policy="omit").statistic), 4),
        "per_source": {
            s: {"sign_acc": round(accuracy(g.theory.values, g.gap.values),
                                  3),
                "n_material": int(len(g))}
            for s, g in mat.groupby("source")},
        "per_paradigm": {
            p: {"sign_acc": round(accuracy(g.theory.values, g.gap.values),
                                  3),
                "n_material": int(len(g))}
            for p, g in mat.groupby("paradigm")},
        "leave_one_source_sign_acc": {
            s: round(accuracy(mat[mat.source != s].theory.values,
                              mat[mat.source != s].gap.values), 3)
            for s in sorted(mat.source.unique())},
        "leave_one_oodset_sign_acc": {
            e: round(accuracy(mat[mat.eval_dataset != e].theory.values,
                              mat[mat.eval_dataset != e].gap.values), 3)
            for e in sorted(mat.eval_dataset.unique())},
        "float32_sign_flips": flip32,
        "fast_path_sign_disagreements": fast_disagree,
        "stability": {
            "mean": round(float(np.nanmean(stability)), 4),
            "frac_fully_stable": round(float(
                np.nanmean(stability == 1.0)), 4)},
        "selective": {str(t): {
            "coverage": round(float((mat.stability >= t).mean()), 4),
            "sign_acc": round(accuracy(
                mat[mat.stability >= t].theory.values,
                mat[mat.stability >= t].gap.values), 4)}
            for t in TAU_GRID},
        "high_precision": hp_summary,
        "scientific_uncertainty": "extraction-derived per-cell coordinate "
                                  "uncertainty unavailable without "
                                  "re-extraction; recorded per plan 5.4",
        "high_precision_cells": hp,
    }
    keep = ["cell", "source", "eval_dataset", "paradigm", "gap", "d",
            "auroc_E", "auroc_C", "l_E", "l_C", "m_tail", "direct_margin",
            "direct_zero", "tail_nonzero", "stability"]
    cells[keep].to_parquet(TAIL_CACHE)
    (OUT_DIR / "tail_space_audit_report.json").write_text(
        json.dumps(report, indent=1, default=float))
    lines = ["# Phase-0 stable tail-space audit (saturation plan sections "
             "4-5)", "",
             "Post-outcome numerical-correctness and sensitivity audit; "
             "mathematically equivalent stable representation of the "
             "frozen formulas; frozen primaries unchanged.", "", "```",
             json.dumps({k: v for k, v in report.items()
                         if k != "high_precision_cells"}, indent=1,
                        default=float), "```", ""]
    (OUT_DIR / "tail_space_audit_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

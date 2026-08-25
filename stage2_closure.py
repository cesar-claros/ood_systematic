"""Stage-2 scientific closure (audit #6, section 9: E1-E4).

FROZEN RULES, declared before execution:

E1 (leave-one-OOD-set-out): hold out one OOD set name at a time; severity
standardization recomputed per fold (per-source mean/sd over the seven
training sets, applied to all eight); baselines refitted on training cells
only; theory signs are the frozen Stage-2 predictions (no fitting). Report
every held-out set, paired geometry-minus-severity differences, material
counts and sign prevalence, and the pooled advantage after removing each set.

E2 (Gate 3, held-out strata ordering): for each held-out ID source, tertile
boundaries are computed from the TRAINING sources' checkpoints only (frozen
rule: thirds of per-checkpoint var_collapse, ascending; strong = lowest
third) and applied to the held-out source's checkpoints; per-stratum
direct-gap isotonic curves with B=2000 simultaneous checkpoint-cluster bands
use the registered full-suite severity axis. Per-source outcome: RETAINED if
crossing_robustness_audit.ordering_retained holds; REVERSED if weak (or
middle) crosses at least 0.05 earlier than strong, or strong shows no
crossing while weak crosses; else INCONCLUSIVE. Overall verdict: PASS if at
least three sources RETAINED and none REVERSED; FAIL if at least two
REVERSED; else INCONCLUSIVE.

E3 (publication-grade geometry-vs-severity): paired sign-accuracy and
balanced-accuracy differences with checkpoint-clustered 95% intervals
(B=2000) in both registered fold modes; per-source accuracies; pooled
source-held-out result after removing each source in turn; material counts
and class prevalence. Four sources are reported individually, never
bootstrapped as a source population.

E4 (coordinate-support diagnostic): per-cell frozen theory evaluation
(cached to outputs/track1/theory_cell_predictions.parquet): distributions of
gamma*a, rho, and dictionary SNR by source; fraction of cells on the
CTM-material side of the displayed analytic boundary (analytic gap <= -0.01);
analytic winner-margin distribution; fraction of cells with both predicted
AUROCs above 0.99; observed |gap| against analytic margin. Analysis only; no
post-outcome recalibration of the theory arm.

Usage (from code/): python stage2_closure.py
Outputs: nc_csf_predictivity/outputs/track1/stage2_closure_report.md/.json
         + theory_cell_predictions.parquet (per-cell cache, incl. hero overlay
         coordinates)
Seeds: folds 2027 (as Stage 2), bootstraps 21 (E2 bands), 31/32 (E3), 41 (E1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossing_robustness_audit import (METRICS, OUT_DIR, PARQUET,
                                       analyze_curve, crossing_value,
                                       load_severity_rows, ordering_retained,
                                       severity_map)
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, dictionary_params,
                                       geometry_model, load_coords,
                                       load_outcomes, map_ood_names,
                                       run_folds, severity_only)

CACHE = OUT_DIR / "theory_cell_predictions.parquet"


# ---------------------------------------------------------------------------
# Shared: cells + full per-cell theory evaluation (cached).
# ---------------------------------------------------------------------------

def build_cells_with_severity() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    cells = load_outcomes(df)
    sev = severity_map(load_severity_rows(), METRICS)
    cells["d"] = [sev.get((s, e)) for s, e in
                  zip(cells.source, cells.eval_dataset)]
    return cells.dropna(subset=["d"]).reset_index(drop=True)


def theory_full(cells: pd.DataFrame, coords: dict) -> pd.DataFrame:
    done: dict[tuple, dict] = {}
    if CACHE.exists():
        cached = pd.read_parquet(CACHE)
        if len(cached) == len(cells):
            return cached
        done = {(r.cell, r.eval_dataset): r._asdict()
                for r in cached.itertuples(index=False)}
        print(f"[theory] warm start: {len(done)} cells cached", flush=True)
    from mc_phase_audit import BASE, build_config_model
    from pilot0.theory import (HeadContext, NoiseModel, predicted_aurocs,
                               predicted_ctm_mean_auroc)
    rows = []
    for idx, row in cells.iterrows():
        if (row.cell, row.eval_dataset) in done:
            rec_c = dict(done[(row.cell, row.eval_dataset)])
            rec_c["idx"] = idx
            rows.append(rec_c)
            continue
        if idx % 100 == 0:
            print(f"[theory] {idx}/{len(cells)}", flush=True)
        if idx % 400 == 0 and rows:
            pd.DataFrame(rows).to_parquet(CACHE)
        rec = coords.get(row.cell)
        out = {"idx": idx, "cell": row.cell, "source": row.source,
               "eval_dataset": row.eval_dataset, "obs_gap": row.gap,
               "d": row.d}
        if rec is not None:
            sets = map_ood_names(rec,
                                 set(cells[cells.cell == row.cell]
                                     .eval_dataset))
            co = sets.get(row.eval_dataset)
            if co is not None:
                s, theta = dictionary_params(row, rec["n_classes"])
                cfg = dict(BASE, s=max(s, 3.0),
                           theta_deg=float(np.clip(theta, 0, 85)),
                           logit_target=max(rec["geometry"]["logit_scale"],
                                            1e-3),
                           eta_std=float(np.clip(
                               rec["geometry"]["class_mean_radius_cv"],
                               0.0, 0.5)),
                           ga=float(np.clip(co["gamma"] * co["a"], 1e-4,
                                            None)),
                           a=float(np.clip(co["a"], 1e-3, 0.999)),
                           rho=float(np.clip(co["rho"], 0.05, None)))
                m = build_config_model(rec["n_classes"], rec["dim"], cfg,
                                       seed=0)
                ctx = HeadContext.from_head(m["w"], m["b"])
                dim = m["means"].shape[1]
                nid = NoiseModel.isotropic(m["sigma"], ctx, dim)
                nood = NoiseModel.isotropic(cfg["rho"] * m["sigma"], ctx, dim)
                head = predicted_aurocs(m["means"], m["class_freq"], nid,
                                        m["m_ood"], nood, ctx)
                ctm = predicted_ctm_mean_auroc(m["means"], m["class_freq"],
                                               m["cov_id"], m["m_ood"],
                                               m["cov_ood"])
                out.update({"auroc_E": float(head["Energy"]),
                            "auroc_C": float(ctm),
                            "pred_gap": -(float(head["Energy"]) - float(ctm)),
                            "ga": cfg["ga"], "rho": cfg["rho"], "s_dict": s})
        rows.append(out)
    fr = pd.DataFrame(rows)
    fr.to_parquet(CACHE)
    return fr


# ---------------------------------------------------------------------------
# E1: leave-one-OOD-set-out.
# ---------------------------------------------------------------------------

def severity_foldaware(rows: list[dict], held_set: str) -> dict:
    """Per-source standardization from the seven training sets, applied to
    all eight (train params only; the held set is scored, not fitted)."""
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    out = {}
    for source, rr in by_source.items():
        train = [r for r in rr if r["eval_dataset"] != held_set]
        mat_t = np.array([[float(r[m]) for m in METRICS] for r in train])
        mu, sd = mat_t.mean(0), mat_t.std(0) + 1e-12
        for r in rr:
            z = (np.array([float(r[m]) for m in METRICS]) - mu) / sd
            out[(source, r["eval_dataset"])] = float(z.mean())
    return out


def e1_loo_ood(cells: pd.DataFrame, theory: pd.Series,
               sev_rows: list[dict]) -> dict:
    sets = sorted(cells.eval_dataset.unique())
    per_set = {}
    pooled_rows = []
    for held in sets:
        smap = severity_foldaware(sev_rows, held)
        fold = cells.copy()
        fold["d"] = [smap.get((s, e)) for s, e in
                     zip(fold.source, fold.eval_dataset)]
        fold = fold.dropna(subset=["d"])
        te = fold.eval_dataset == held
        train, test = fold[~te], fold[te]
        preds = {"severity": severity_only(train, test),
                 "geometry": geometry_model(train, test, flexible=False),
                 "flexible": geometry_model(train, test, flexible=True),
                 "mean": np.full(len(test), float(train.gap.mean()))}
        test = test.assign(theory=theory[test.index], **preds)
        mat = test[np.abs(test.gap) >= MATERIALITY]
        obs = mat.gap.values
        rec = {"n_material": int(len(mat)),
               "frac_positive": float((obs > 0).mean()) if len(mat) else None}
        for arm in ("theory", "severity", "geometry", "flexible", "mean"):
            rec[arm] = accuracy(mat[arm].values, obs)
            rec[f"{arm}_bal"] = balanced_accuracy(mat[arm].values, obs)
        rec["g_minus_s"] = (rec["geometry"] - rec["severity"]
                            if rec["n_material"] else None)
        per_set[held] = rec
        pooled_rows.append(mat.assign(held_set=held))
    pooled = pd.concat(pooled_rows)
    obs = pooled.gap.values
    res = {"per_set": per_set,
           "pooled": {arm: accuracy(pooled[arm].values, obs)
                      for arm in ("theory", "severity", "geometry",
                                  "flexible", "mean")}}
    res["pooled"]["g_minus_s"] = (res["pooled"]["geometry"]
                                  - res["pooled"]["severity"])
    infl = {}
    for held in sets:
        sub = pooled[pooled.held_set != held]
        infl[held] = (accuracy(sub.geometry.values, sub.gap.values)
                      - accuracy(sub.severity.values, sub.gap.values))
    res["pooled_g_minus_s_dropping_set"] = infl
    return res


# ---------------------------------------------------------------------------
# E2: Gate 3 on held-out sources.
# ---------------------------------------------------------------------------

def e2_gate3(cells: pd.DataFrame, b: int = 2000) -> dict:
    fine_n = 301
    out = {"per_source": {}}
    outcomes = []
    for held in sorted(cells.source.unique()):
        train_cells = cells[cells.source != held]
        vc = train_cells.groupby("cell")["var_collapse"].first().sort_values()
        b1 = float(vc.iloc[len(vc) // 3 - 1])
        b2 = float(vc.iloc[2 * len(vc) // 3 - 1])
        held_cells = cells[cells.source == held]
        vc_h = held_cells.groupby("cell")["var_collapse"].first()
        strata = {"strong": set(vc_h[vc_h <= b1].index),
                  "middle": set(vc_h[(vc_h > b1) & (vc_h <= b2)].index),
                  "weak": set(vc_h[vc_h > b2].index)}
        res = {}
        rng = np.random.default_rng(21)
        for name, members in strata.items():
            sub = held_cells[held_cells.cell.isin(members)]
            if sub.cell.nunique() < 3:
                res[name] = {"n_ckpt": int(sub.cell.nunique()),
                             "note": "too few checkpoints"}
                continue
            data: dict[str, list] = {}
            for r in sub.itertuples():
                data.setdefault(r.cell, []).append((float(r.d), float(r.gap)))
            fine = np.linspace(sub.d.min(), sub.d.max(), fine_n)
            rec = analyze_curve("pava", data, sorted(data), fine, b, rng)
            rec["n_ckpt"] = int(sub.cell.nunique())
            res[name] = rec
        retained = ordering_retained(res)
        cv = {k: crossing_value(res.get(k, {})) for k in
              ("strong", "middle", "weak")}
        reversed_ = bool(
            (np.isfinite(cv["weak"]) and cv["weak"] < cv["strong"] - 0.05)
            or (np.isfinite(cv["middle"]) and cv["middle"] < cv["strong"] - 0.05)
            or (not np.isfinite(cv["strong"])
                and (np.isfinite(cv["weak"]) or np.isfinite(cv["middle"]))))
        outcome = ("RETAINED" if retained
                   else "REVERSED" if reversed_ else "INCONCLUSIVE")
        outcomes.append(outcome)
        out["per_source"][held] = {
            "tertile_bounds_from_train": [b1, b2], "strata": res,
            "crossing_values": {k: (None if not np.isfinite(v) else round(v, 3))
                                if not (v == -np.inf) else "<=range-min"
                                for k, v in cv.items()},
            "outcome": outcome}
    n_ret = outcomes.count("RETAINED")
    n_rev = outcomes.count("REVERSED")
    out["verdict"] = ("PASS" if n_ret >= 3 and n_rev == 0
                      else "FAIL" if n_rev >= 2 else "INCONCLUSIVE")
    out["outcomes"] = outcomes
    return out


# ---------------------------------------------------------------------------
# E3: geometry-vs-severity uncertainty.
# ---------------------------------------------------------------------------

def paired_ci(mat: pd.DataFrame, arm_a: str, arm_b: str, metric,
              seed: int, b: int = 2000) -> dict:
    ck = np.array(sorted(mat.cell.unique()))
    groups = {c: g for c, g in mat.groupby("cell")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(b)
    for i in range(b):
        fr = pd.concat([groups[c] for c in rng.choice(ck, len(ck),
                                                      replace=True)])
        diffs[i] = (metric(fr[arm_a].values, fr.gap.values)
                    - metric(fr[arm_b].values, fr.gap.values))
    point = (metric(mat[arm_a].values, mat.gap.values)
             - metric(mat[arm_b].values, mat.gap.values))
    return {"point": round(float(point), 3),
            "ci95": [round(float(np.quantile(diffs, 0.025)), 3),
                     round(float(np.quantile(diffs, 0.975)), 3)]}


def e3_uncertainty(cells: pd.DataFrame, theory: pd.Series) -> dict:
    out = {}
    for mode, seed in (("ckpt5", 31), ("loso", 32)):
        fitted = run_folds(cells, theory, mode,
                           np.random.default_rng(FOLD_SEED))
        mat = fitted[np.abs(fitted.gap) >= MATERIALITY].dropna(
            subset=["geometry", "severity"])
        rec = {"n_material": int(len(mat)),
               "frac_positive": round(float((mat.gap > 0).mean()), 3),
               "g_minus_s_sign": paired_ci(mat, "geometry", "severity",
                                           accuracy, seed),
               "g_minus_s_balanced": paired_ci(mat, "geometry", "severity",
                                               balanced_accuracy, seed + 100)}
        per_src = {}
        for s, g in mat.groupby("source"):
            per_src[s] = {
                "n_material": int(len(g)),
                "frac_positive": round(float((g.gap > 0).mean()), 3),
                "geometry": round(accuracy(g.geometry.values, g.gap.values), 3),
                "severity": round(accuracy(g.severity.values, g.gap.values), 3),
                "geometry_bal": round(balanced_accuracy(g.geometry.values,
                                                        g.gap.values), 3),
                "severity_bal": round(balanced_accuracy(g.severity.values,
                                                        g.gap.values), 3)}
        rec["per_source"] = per_src
        infl = {}
        for s in mat.source.unique():
            sub = mat[mat.source != s]
            infl[s] = round(accuracy(sub.geometry.values, sub.gap.values)
                            - accuracy(sub.severity.values, sub.gap.values), 3)
        rec["g_minus_s_dropping_source"] = infl
        out[mode] = rec
    return out


# ---------------------------------------------------------------------------
# E4: coordinate-support diagnostic.
# ---------------------------------------------------------------------------

def e4_support(theory_fr: pd.DataFrame) -> dict:
    fr = theory_fr.dropna(subset=["pred_gap"]).copy()
    out = {"n": int(len(fr))}
    for col in ("ga", "rho", "s_dict"):
        out[f"{col}_quantiles_by_source"] = {
            s: [round(float(np.quantile(g[col], q)), 3)
                for q in (0.05, 0.5, 0.95)]
            for s, g in fr.groupby("source")}
    margin = np.abs(fr.auroc_E - fr.auroc_C)
    out["analytic_margin_quantiles"] = [
        round(float(np.quantile(margin, q)), 5)
        for q in (0.5, 0.9, 0.95, 0.99)]
    out["frac_margin_zero"] = round(float((margin == 0).mean()), 3)
    out["frac_both_above_099"] = round(float(
        ((fr.auroc_E > 0.99) & (fr.auroc_C > 0.99)).mean()), 3)
    out["frac_ctm_material_side"] = round(float(
        ((fr.auroc_E - fr.auroc_C) <= -0.01).mean()), 3)
    mat = fr[np.abs(fr.obs_gap) >= MATERIALITY]
    from scipy.stats import spearmanr
    nz = mat[np.abs(mat.auroc_E - mat.auroc_C) > 0]
    out["spearman_obsgap_vs_margin_nonzero_cells"] = (
        round(float(spearmanr(np.abs(nz.auroc_E - nz.auroc_C),
                              np.abs(nz.obs_gap)).statistic), 3)
        if len(nz) > 10 else None)
    out["median_abs_obs_gap_material"] = round(float(
        np.abs(mat.obs_gap).median()), 1)
    return out


# ---------------------------------------------------------------------------

def render(e1, e2, e3, e4) -> str:
    L = ["# Stage-2 scientific closure (audit #6 E1-E4; frozen rules in "
         "stage2_closure.py docstring)", ""]
    L += ["## E1: leave-one-OOD-set-out", "",
          "| held-out set | n mat | frac + | theory | severity | geometry | flexible | G-S |",
          "|---|---|---|---|---|---|---|---|"]
    for s, r in e1["per_set"].items():
        L.append(f"| {s} | {r['n_material']} | {r['frac_positive']:.2f} | "
                 f"{r['theory']:.3f} | {r['severity']:.3f} | "
                 f"{r['geometry']:.3f} | {r['flexible']:.3f} | "
                 f"{r['g_minus_s']:+.3f} |")
    p = e1["pooled"]
    L += ["", f"Pooled LOO-OOD: theory {p['theory']:.3f}, severity "
          f"{p['severity']:.3f}, geometry {p['geometry']:.3f}, G-S "
          f"{p['g_minus_s']:+.3f}.",
          "Pooled G-S after dropping each set: "
          + ", ".join(f"{k}: {v:+.3f}"
                      for k, v in e1["pooled_g_minus_s_dropping_set"].items()),
          ""]
    L += ["## E2: Gate 3 (held-out strata ordering)", "",
          "| held-out source | strong | middle | weak | outcome |",
          "|---|---|---|---|---|"]
    for s, r in e2["per_source"].items():
        cv = r["crossing_values"]
        L.append(f"| {s} | {cv['strong']} | {cv['middle']} | {cv['weak']} | "
                 f"{r['outcome']} |")
    L += ["", f"**Gate 3 verdict: {e2['verdict']}** "
          f"(outcomes: {e2['outcomes']})", ""]
    L += ["## E3: geometry-vs-severity uncertainty", ""]
    for mode, r in e3.items():
        L += [f"### {mode}: n material {r['n_material']}, frac positive "
              f"{r['frac_positive']}",
              f"- sign-acc G-S: {r['g_minus_s_sign']['point']:+.3f}, CI95 "
              f"{r['g_minus_s_sign']['ci95']}",
              f"- balanced-acc G-S: {r['g_minus_s_balanced']['point']:+.3f}, "
              f"CI95 {r['g_minus_s_balanced']['ci95']}",
              "",
              "| source | n mat | frac + | geometry | severity | geo bal | sev bal |",
              "|---|---|---|---|---|---|---|"]
        for s, v in r["per_source"].items():
            L.append(f"| {s} | {v['n_material']} | {v['frac_positive']} | "
                     f"{v['geometry']} | {v['severity']} | "
                     f"{v['geometry_bal']} | {v['severity_bal']} |")
        L += ["", "G-S after dropping each source: "
              + ", ".join(f"{k}: {v:+.3f}"
                          for k, v in r["g_minus_s_dropping_source"].items()),
              ""]
    L += ["## E4: coordinate-support diagnostic", "",
          f"- cells with theory evaluation: {e4['n']}",
          f"- gamma*a quantiles (5/50/95) by source: {e4['ga_quantiles_by_source']}",
          f"- rho quantiles by source: {e4['rho_quantiles_by_source']}",
          f"- dictionary SNR quantiles by source: {e4['s_dict_quantiles_by_source']}",
          f"- analytic winner-margin quantiles (50/90/95/99): {e4['analytic_margin_quantiles']}",
          f"- fraction with margin exactly zero: {e4['frac_margin_zero']}",
          f"- fraction with both AUROCs > 0.99: {e4['frac_both_above_099']}",
          f"- fraction on the CTM-material side of the displayed boundary (gap <= -0.01): {e4['frac_ctm_material_side']}",
          f"- median |observed gap| on material cells: {e4['median_abs_obs_gap_material']} (AUGRC x 1000)",
          f"- Spearman(|analytic margin|, |observed gap|) on nonzero-margin material cells: {e4['spearman_obsgap_vs_margin_nonzero_cells']}",
          ""]
    return "\n".join(L)


def main() -> None:
    cells = build_cells_with_severity()
    coords, problems = load_coords(Path("pilot0/pool_coords"))
    assert not problems, problems
    theory_fr = theory_full(cells, coords)
    theory = pd.Series(theory_fr.pred_gap.values, index=cells.index)
    sev_rows = load_severity_rows()

    e1 = e1_loo_ood(cells, theory, sev_rows)
    e2 = e2_gate3(cells)
    e3 = e3_uncertainty(cells, theory)
    e4 = e4_support(theory_fr)

    md = render(e1, e2, e3, e4)
    (OUT_DIR / "stage2_closure_report.md").write_text(md)
    (OUT_DIR / "stage2_closure_report.json").write_text(
        json.dumps({"e1": e1, "e2": e2, "e3": e3, "e4": e4}, indent=1,
                   default=str))
    print(md)


if __name__ == "__main__":
    main()

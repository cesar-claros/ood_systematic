"""Corrected ladder summaries (step 4 of the record-audit analysis, 2026-09-18). Leaves ladder_readout.py untouched.

Reads the same per-run files (ledger.json, measurements.csv, outcomes.csv) and, if present, accuracy_splits.csv (from
split_accuracy.py: fit, validation and test accuracy per checkpoint). Reports, in this order:
  1. the realized run grid (steps, shifts, realized OOD counts from the ledger provenance, detectors, missing cells, last batch loss);
  2. severe-degradation arms (mean test-accuracy change below --severe), reported separately and excluded from the main statistics;
  3. shift classes by the best reference detector at step 0 (pNML excluded);
  4. losses per cell: the oracle-envelope loss (best reference detector minus best adapted detector) beside detector-specific losses for
     detectors fixed without the target shift (MahaPP for the density family and NCI for the logit family, the choices of the task-B rule R,
     made on the development task) and the worst fixed detector; maxima over accuracy-neutral cells on reference-separable shifts;
  5. Spearman correlations of the losses with the retention measures, with the accuracy change (validation if available, else test,
     descriptive) and with the step; within recipe and pooled; with and without the severe arms;
  6. fusion at a fixed detector: combined minus adapted and combined minus reference, per detector, with the fraction of cells where the
     fusion loses more than 0.005 to the adapted detector;
  7. per-arm means.
Writes <root>/ladder_readout_v2.md and <root>/ladder_readout_v2_rows.csv.

  python x8_pool_a/ladder_readout_v2.py documentation/adaptation_results/ladder
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DENSITY, LOGIT = "MahaPP", "NCI"   # rule R's fixed choices, made on task A before task B


def md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f": d[c] = d[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.astype(str).values]
    return "\n".join(lines)


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[ok], y[ok]).correlation if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0 else np.nan


def load(root: pathlib.Path, keep_pnml: bool):
    runs = {}
    for d in sorted(p for p in root.iterdir() if (p / "outcomes.csv").exists()):
        L = json.load(open(d / "ledger.json")); cfg = L["config"]; o = pd.read_csv(d / "outcomes.csv"); m = pd.read_csv(d / "measurements.csv").set_index("step")
        if not keep_pnml: o = o[o.detector != "pNML"].reset_index(drop=True)
        acc = pd.read_csv(d / "accuracy_splits.csv").set_index("step") if (d / "accuracy_splits.csv").exists() else None
        task = cfg["data"]; arm = d.name[len(task) + 1:].rsplit("_seed", 1)[0]
        runs[d.name] = dict(task=task, arm=arm, cfg=cfg, L=L, o=o, m=m, acc=acc)
    return runs


def main(root: pathlib.Path, acc_tol: float, severe: float, sep: float, keep_pnml: bool, fusion_tol: float):
    runs = load(root, keep_pnml); out = []
    P = out.append
    P(f"# Corrected ladder readout (v2), {pd.Timestamp.today():%Y-%m-%d}\n")
    P(f"Root `{root}`; runs {len(runs)}; pNML {'kept' if keep_pnml else 'excluded (numerically unstable when validation rows exceed feature dimensions; validity audit 2026-09-18)'}; accuracy-neutral = |accuracy change| <= {acc_tol}; severe degradation = mean test-accuracy change over post-reference checkpoints < {severe}; separable = best reference AUROC at step 0 >= {sep}.\n")
    have_val = all(r["acc"] is not None for r in runs.values())
    P(f"Accuracy used for neutrality and as the competing predictor: {'validation (permitted; test reported alongside, descriptive)' if have_val else 'TEST accuracy (descriptive; validation accuracy not available: run split_accuracy.py on the HPC and resync accuracy_splits.csv)'}.\n")

    # 1. grid
    grid = []
    for name, r in runs.items():
        o, m, L, cfg = r["o"], r["m"], r["L"], r["cfg"]; prov = L.get("data_provenance", {}); shifts = sorted(o.ood_set.unique()); steps = sorted(o.step.unique())
        n_ood = {s: min(cfg["n_ood"], prov.get(f"{s}_n_total", cfg["n_ood"])) for s in shifts}
        dets = o.detector.nunique(); variants = sorted(o.variant.unique())
        expected = len(shifts) * dets * (1 + (len(steps) - 1) * len(variants)); missing = int(o.auroc_allid.isna().sum())
        acc0, accT = m.id_test_acc.loc[0], m.id_test_acc.loc[max(steps)]
        grid.append(dict(run=name, steps=len(steps), shifts=",".join(f"{s}:{n_ood[s]}" for s in shifts), n_test=cfg["n_test"], n_val=cfg["n_val"], detectors=dets, rows=len(o), expected=expected, missing_auroc=missing,
                         last_batch_loss=L.get("last_loss"), test_acc_ref=acc0, test_acc_final=accT, mean_acc_change=float(m.id_test_acc.loc[steps[1:]].mean() - acc0)))
    G = pd.DataFrame(grid); P("## 1. Realized run grid\n"); P(md(G)); P("")
    P(f"Missing AUROC cells in total: {int(G.missing_auroc.sum())}. Row counts equal the expected counts in {int((G.rows == G.expected).sum())} of {len(G)} runs.\n")

    # 2. severe arms
    sev = G[G.mean_acc_change < severe]; sev_runs = set(sev.run)
    P("## 2. Severe classification degradation (excluded from sections 3 to 7; cause not diagnosed here)\n")
    if len(sev):
        rows = []
        for name in sev.run:
            m = runs[name]["m"]; rows.append(dict(run=name, **{f"acc@{int(s)}": round(float(a), 3) for s, a in m.id_test_acc.items()}, last_batch_loss=runs[name]["L"].get("last_loss"), chance=round(1 / runs[name]["cfg"]["n_cls"], 3)))
        P(md(pd.DataFrame(rows))); P("")
    else: P("none\n")

    # 3 and 4. per-cell rows
    cells = []
    for name, r in runs.items():
        o, m, task, arm = r["o"], r["m"], r["task"], r["arm"]; recipe = arm.split("_")[0]; steps = sorted(o.step.unique()); acc0 = m.id_test_acc.loc[0]
        for shift, g in o.groupby("ood_set"):
            A = g[g.variant == "adapted"].pivot(index="step", columns="detector", values="auroc_allid")
            R = g[g.variant == "reference"].pivot(index="step", columns="detector", values="auroc_allid"); R.loc[0] = A.loc[0]; R = R.sort_index()
            C = {v: g[g.variant == v].pivot(index="step", columns="detector", values="auroc_allid") for v in ("combined", "combined_w25", "combined_w75") if (g.variant == v).any()}
            ref_best = A.loc[0].max(); d_ref = A.loc[0].idxmax(); cls = "separable" if ref_best >= sep else "confused"
            for s in steps[1:]:
                acc_t = m.id_test_acc.loc[s] - acc0
                val_t = (r["acc"].val_acc.loc[s] - r["acc"].val_acc.loc[0]) if r["acc"] is not None else np.nan
                acc_used = val_t if have_val else acc_t
                cc, cm, ck = m.cc_cka_ref.loc[s], m.cm_cka_ref.loc[s], m.paired_cka_fit.loc[s]
                w = 0.25 if cc >= 0.9 else (0.5 if cc >= 0.8 else 0.75); wtag = {0.25: "combined_w25", 0.5: "combined", 0.75: "combined_w75"}[w]
                det_loss = (A.loc[0] - A.loc[s])   # same-detector loss, reference minus adapted
                row = dict(run=name, task=task, arm=arm, recipe=recipe, shift=shift, cls=cls, step=s, severe=name in sev_runs, ref_best=ref_best, ref_best_det=d_ref,
                           acc_change_test=acc_t, acc_change_val=val_t, acc_change=acc_used, acc_neutral=abs(acc_used) <= acc_tol,
                           one_minus_cc=1 - cc, one_minus_cm=1 - cm, one_minus_cka=1 - ck,
                           loss_envelope=A.loc[0].max() - A.loc[s].max(), loss_density=det_loss.get(DENSITY, np.nan), loss_logit=det_loss.get(LOGIT, np.nan),
                           loss_refbest_det=det_loss.get(d_ref, np.nan), loss_worst_det=det_loss.max(), worst_det=det_loss.idxmax(), loss_median_det=det_loss.median(),
                           fusion_w=w)
                for det in (DENSITY, LOGIT):
                    if det in A.columns and "combined" in C:
                        row[f"fus_minus_adapted_{det}"] = C["combined"].loc[s, det] - A.loc[s, det]; row[f"fus_minus_reference_{det}"] = C["combined"].loc[s, det] - A.loc[0, det]
                        row[f"r2_minus_better_{det}"] = C[wtag].loc[s, det] - max(A.loc[s, det], A.loc[0, det])
                cells.append(row)
                for det in A.columns:   # per-detector fusion rows
                    if "combined" in C: cells[-1].setdefault("_fus", {})[det] = (C["combined"].loc[s, det] - A.loc[s, det], C["combined"].loc[s, det] - A.loc[0, det])
    fus_rows = [dict(run=c["run"], task=c["task"], arm=c["arm"], shift=c["shift"], cls=c["cls"], step=c["step"], severe=c["severe"], detector=d, fus_minus_adapted=v[0], fus_minus_reference=v[1]) for c in cells for d, v in c.pop("_fus", {}).items()]
    df = pd.DataFrame(cells); F = pd.DataFrame(fus_rows); df.to_csv(root / "ladder_readout_v2_rows.csv", index=False)
    main_df = df[~df.severe]

    P("## 3. Shift classes (best reference detector at step 0)\n")
    P(md(df.groupby(["task", "shift"]).agg(ref_best=("ref_best", "first"), detector=("ref_best_det", "first"), cls=("cls", "first")).reset_index())); P("")

    P("## 4. Losses: oracle envelope beside fixed detectors (severe arms excluded)\n")
    P(f"loss_envelope = best reference detector minus best adapted detector (per cell); loss_density = same-detector loss of {DENSITY}; loss_logit = same-detector loss of {LOGIT}; loss_refbest_det = same-detector loss of the detector that was best at the reference on that shift (selected with OOD data; descriptive); loss_worst_det = largest same-detector loss over the inventory. Maxima are over post-reference cells; 'neutral' = accuracy-neutral cells only.\n")
    def agg_losses(d):
        n = d[d.acc_neutral]
        return pd.Series(dict(n_cells=len(d), n_neutral=len(n), envelope_max=d.loss_envelope.max(), envelope_max_neutral=n.loss_envelope.max(),
                              density_max_neutral=n.loss_density.max(), logit_max_neutral=n.loss_logit.max(), refbest_max_neutral=n.loss_refbest_det.max(),
                              worst_det_max_neutral=n.loss_worst_det.max(), worst_det_name=(n.loc[n.loss_worst_det.idxmax(), "worst_det"] if len(n) else ""), median_det_max_neutral=n.loss_median_det.max()))
    T4 = pd.DataFrame([dict(task=t, cls=c, **agg_losses(g)) for (t, c), g in main_df.groupby(["task", "cls"])]); P(md(T4)); P("")
    sepn = main_df[(main_df.cls == "separable") & main_df.acc_neutral]
    P(f"Accuracy-neutral, reference-separable cells: {len(sepn)}. Maximum loss: envelope {sepn.loss_envelope.max():.4f}; {DENSITY} {sepn.loss_density.max():.4f}; {LOGIT} {sepn.loss_logit.max():.4f}; reference-best detector {sepn.loss_refbest_det.max():.4f}; worst fixed detector {sepn.loss_worst_det.max():.4f} ({sepn.loc[sepn.loss_worst_det.idxmax(), 'worst_det'] if len(sepn) else ''}, {sepn.loc[sepn.loss_worst_det.idxmax(), 'run'] if len(sepn) else ''}, {sepn.loc[sepn.loss_worst_det.idxmax(), 'shift'] if len(sepn) else ''}). The pre-declared criterion (envelope below 0.02) is an oracle-availability statement; the fixed-detector maxima are the deployed-detector statements.\n")
    P("Worst fixed detector on accuracy-neutral separable cells, by arm (largest same-detector loss and which detector):\n")
    W = pd.DataFrame([dict(task=t, arm=a, n=len(d), worst_loss=d.loss_worst_det.max(), detector=d.loc[d.loss_worst_det.idxmax(), "worst_det"], shift=d.loc[d.loss_worst_det.idxmax(), "shift"], density_loss_max=d.loss_density.max(), logit_loss_max=d.loss_logit.max()) for (t, a), d in sepn.groupby(["task", "arm"])])
    P(md(W) if len(W) else "no accuracy-neutral separable cells"); P("")

    P("## 5. Spearman correlations of the losses with candidate predictors\n")
    P("Predictors: one minus class-centred CKA (cc), one minus class-mean CKA (cm), one minus paired CKA (cka), minus the accuracy change (acc; validation if available, else test, descriptive), and the step. 'all arms' includes the severe arms (comparable with the original readout); 'main' excludes them; 'within recipe' pools the arms of one recipe. n is the number of cells.\n")
    preds = {"cc": "one_minus_cc", "cm": "one_minus_cm", "cka": "one_minus_cka", "acc": "acc_change", "step": "step"}
    def corr_table(d, label):
        rows = []
        for (task, cls), g in d.groupby(["task", "cls"]):
            for lname, lcol in (("envelope", "loss_envelope"), (DENSITY, "loss_density"), (LOGIT, "loss_logit")):
                r = dict(scope=label, task=task, cls=cls, loss=lname, n=len(g))
                for k, col in preds.items(): r[f"rho_{k}"] = rho((-g[col] if k == "acc" else g[col]), g[lcol])
                r["rho_cc_vs_acc"] = rho(g.one_minus_cc, -g.acc_change); rows.append(r)
        return pd.DataFrame(rows)
    T5 = pd.concat([corr_table(df, "all arms"), corr_table(main_df, "main")]); P(md(T5, "{:.2f}")); P("")
    P("Within recipe (main arms):\n")
    rows = []
    for (task, cls, recipe), g in main_df.groupby(["task", "cls", "recipe"]):
        r = dict(task=task, cls=cls, recipe=recipe, n=len(g), arms=g.arm.nunique())
        for k, col in preds.items(): r[f"rho_{k}"] = rho((-g[col] if k == "acc" else g[col]), g.loss_envelope)
        r[f"rho_cc_{DENSITY}"] = rho(g.one_minus_cc, g.loss_density); r["rho_cc_vs_acc"] = rho(g.one_minus_cc, -g.acc_change); rows.append(r)
    P(md(pd.DataFrame(rows), "{:.2f}")); P("")

    P("## 6. Fusion at a fixed detector (severe arms excluded; fusion = rank mean with weight one half on the reference)\n")
    if len(F):
        Fm = F[~F.severe]
        T6 = Fm.groupby("detector").agg(n=("step", "size"), fus_minus_adapted_mean=("fus_minus_adapted", "mean"), fus_minus_adapted_min=("fus_minus_adapted", "min"),
                                        frac_loses_to_adapted=("fus_minus_adapted", lambda x: float((x < -fusion_tol).mean())), fus_minus_reference_mean=("fus_minus_reference", "mean"), fus_minus_reference_min=("fus_minus_reference", "min"),
                                        frac_loses_to_reference=("fus_minus_reference", lambda x: float((x < -fusion_tol).mean()))).reset_index().sort_values("fus_minus_adapted_min")
        P(md(T6)); P("")
        r2 = main_df[[c for c in main_df.columns if c.startswith("r2_minus_better_")]].agg(["mean", "min"]).T.reset_index().rename(columns={"index": "quantity"})
        P(f"ID-weighted fusion (weight 0.25 / 0.5 / 0.75 on the reference by class-centred CKA >= 0.9 / >= 0.8 / < 0.8) minus the better of adapted and reference, same detector:\n"); P(md(r2)); P("")
    P("## 7. Per-arm means over post-reference checkpoints (all arms; severe flagged)\n")
    T7 = df.groupby(["task", "arm"]).agg(severe=("severe", "first"), acc_change=("acc_change", "mean"), one_minus_cc=("one_minus_cc", "mean"), one_minus_cka=("one_minus_cka", "mean"),
                                        envelope_loss=("loss_envelope", "mean"), envelope_loss_max=("loss_envelope", "max"), density_loss_max=("loss_density", "max"), logit_loss_max=("loss_logit", "max"), worst_det_loss_max=("loss_worst_det", "max")).reset_index()
    P(md(T7)); P("")
    text = "\n".join(out); (root / "ladder_readout_v2.md").write_text(text); print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--acc-tol", type=float, default=0.01); ap.add_argument("--severe", type=float, default=-0.1)
    ap.add_argument("--sep", type=float, default=0.95); ap.add_argument("--keep-pnml", action="store_true"); ap.add_argument("--fusion-tol", type=float, default=0.005)
    a = ap.parse_args(); main(pathlib.Path(a.root), a.acc_tol, a.severe, a.sep, a.keep_pnml, a.fusion_tol)

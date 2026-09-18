"""Pre-declared readout of the H1 retention ladder (written before any ladder trajectory is run; 2026-09-17).

Inputs: every run directory under a ladder root (outcomes.csv, measurements.csv, ledger.json). Definitions, fixed now:
- Reference AUROC of a (run, shift) = best adapted detector at step 0. A shift is REFERENCE-SEPARABLE if that value is >= 0.95,
  REFERENCE-CONFUSED otherwise.
- Loss at a checkpoint = (best reference-variant AUROC) - (best adapted AUROC), positive = detectability lost relative to the frozen
  reference. Accuracy-neutral checkpoint = ID test accuracy within 0.01 of the reference accuracy.
- Retention measures (ID only): cm_cka_ref (class-mean CKA), cc_cka_ref (class-centred paired CKA), paired_cka_fit (raw paired CKA).
- H1 test: Spearman rank correlation between (1 - retention) and loss over all checkpoints and arms, per task and shift class.
  Prediction: rho >= 0.8 on reference-separable shifts; no positive relation on reference-confused shifts.
- Kill criterion: max loss over accuracy-neutral checkpoints on reference-separable shifts < 0.02 AUROC -> observation, not a paper.
- ID-weighted fusion rule R2: reference weight w = 0.25 if cc_cka_ref >= 0.9, 0.5 if 0.8 <= cc_cka_ref < 0.9, 0.75 if cc_cka_ref < 0.8,
  applied to the detector that was best at the reference; compared with the best adapted detector, the frozen reference, the fixed
  0.5 fusion and the per-checkpoint oracle. Success: mean regret against the better of reference and adapted <= 0.005 on every shift.
Run: .venv/bin/python x8_pool_a/ladder_readout.py <ladder_root> > <ladder_root>/ladder_readout.md
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEP_THRESHOLD, ACC_TOL, KILL, R2_TOL = 0.95, 0.01, 0.02, 0.005
W_OF = lambda cc: 0.25 if cc >= 0.9 else (0.5 if cc >= 0.8 else 0.75)


def load_runs(root: pathlib.Path):
    runs = {}
    for d in sorted(p for p in root.iterdir() if (p / "outcomes.csv").exists()):
        L = json.load(open(d / "ledger.json")); runs[d.name] = (pd.read_csv(d / "outcomes.csv"), pd.read_csv(d / "measurements.csv").set_index("step"), L["config"])
    return runs


def main(root):
    # Missing metric cells (NaN AUROC, written by the driver when a raw score was nonfinite) must never enter a maximum silently.
    runs = load_runs(root); rows = []
    # pNML is excluded from every maximum: its score is numerically unstable when the validation set has more rows than feature
    # dimensions (the kernel-range residual is float noise, so the recorded GPU values and a CPU recomputation differ by up to 0.9;
    # validity audit of 2026-09-18). Pass --keep-pnml to reproduce the original readout of 2026-09-17.
    if not KEEP_PNML:
        runs = {k: (o[o.detector != "pNML"].reset_index(drop=True), m, c) for k, (o, m, c) in runs.items()}; print("pNML excluded from all maxima (numerically unstable; see score_validity audit)\n")
    n_missing = sum(int(o.auroc_allid.isna().sum()) for o, _, _ in runs.values())
    if n_missing: print(f"WARNING: {n_missing} missing AUROC cells (nonfinite raw scores); every maximum below ignores them, so the stopping statistic is not certified until they are resolved\n")
    for name, (o, m, cfg) in runs.items():
        task, arm = cfg["data"], f'{cfg["method"]}_{cfg.get("lora_rank") if cfg["method"] == "lora" else cfg["lr"]}'
        acc0 = m.id_test_acc.loc[0]
        for ood in sorted(o.ood_set.unique()):
            A = o[(o.ood_set == ood) & (o.variant == "adapted")].pivot(index="step", columns="detector", values="auroc_allid")
            R = o[(o.ood_set == ood) & (o.variant == "reference")].pivot(index="step", columns="detector", values="auroc_allid")
            V = {v: o[(o.ood_set == ood) & (o.variant == v)].pivot(index="step", columns="detector", values="auroc_allid") for v in ("combined", "combined_w25", "combined_w75")}
            ref_auroc = A.loc[0].max(); d0 = A.loc[0].idxmax(); cls = "separable" if ref_auroc >= SEP_THRESHOLD else "confused"
            for s in [s for s in A.index if s > 0]:
                cc = m.cc_cka_ref.loc[s] if "cc_cka_ref" in m.columns else np.nan; w = W_OF(cc) if not np.isnan(cc) else 0.5
                r2 = {0.25: V["combined_w25"], 0.5: V["combined"], 0.75: V["combined_w75"]}[w].loc[s, d0]
                better = max(R.loc[s, d0], A.loc[s, d0])
                rows.append(dict(run=name, task=task, arm=arm, shift=ood, cls=cls, step=s, ref_auroc=ref_auroc, best_adapted=A.loc[s].max(), best_reference=R.loc[s].max(),
                                 loss=R.loc[s].max() - A.loc[s].max(), acc_neutral=abs(m.id_test_acc.loc[s] - acc0) <= ACC_TOL, acc_delta=m.id_test_acc.loc[s] - acc0,
                                 one_minus_cm=1 - m.cm_cka_ref.loc[s], one_minus_cc=1 - cc, one_minus_cka=1 - m.paired_cka_fit.loc[s], w_R2=w,
                                 R2=r2, fixed_fusion=V["combined"].loc[s, d0], reference_d0=R.loc[s, d0], adapted_d0=A.loc[s, d0], regret_R2=better - r2, oracle=A.loc[s].max()))
    df = pd.DataFrame(rows); df.to_csv(root / "ladder_readout_rows.csv", index=False)
    print("# H1 retention ladder readout\n"); print(f"runs: {len(runs)}; rows: {len(df)}\n")
    print("## Shift classes by task (reference AUROC of the best detector at step 0)\n"); print(df.groupby(["task", "shift"]).ref_auroc.first().round(3).to_string(), "\n")
    print("## H1: Spearman rho between (1 - retention) and loss, per task and shift class (all arms and checkpoints pooled)\n")
    print("| Task | Class | n | rho(1-cm_cka) | rho(1-cc_cka) | rho(1-paired_cka) | max loss | max loss (accuracy-neutral) |"); print("|---|---|---:|---:|---:|---:|---:|---:|")
    kill_max = []
    for (task, cls), g in df.groupby(["task", "cls"]):
        rh = lambda c: spearmanr(g[c], g.loss).correlation if g[c].notna().sum() > 2 else np.nan
        mx_neutral = g[g.acc_neutral].loss.max() if g.acc_neutral.any() else np.nan
        if cls == "separable": kill_max.append(mx_neutral)
        print(f"| {task} | {cls} | {len(g)} | {rh('one_minus_cm'):.2f} | {rh('one_minus_cc'):.2f} | {rh('one_minus_cka'):.2f} | {g.loss.max():.3f} | {mx_neutral:.3f} |")
    km = np.nanmax(kill_max) if kill_max else np.nan
    print(f"\nKill criterion: max accuracy-neutral loss on reference-separable shifts = {km:.3f}; threshold {KILL}. Verdict: {'KILL (observation, not a paper)' if km < KILL else 'SURVIVES'}\n")
    print("## Rule R2 (ID-weighted fusion of the reference-best detector) against baselines, mean AUROC per task and shift over arms and checkpoints\n")
    t = df.groupby(["task", "shift"]).agg(R2=("R2", "mean"), fixed_fusion=("fixed_fusion", "mean"), reference=("reference_d0", "mean"), adapted=("adapted_d0", "mean"), best_adapted=("best_adapted", "mean"), regret_R2=("regret_R2", "mean")).round(4)
    print(t.to_string()); print(f"\nR2 success (mean regret <= {R2_TOL} on every shift): {'MEETS' if (t.regret_R2 <= R2_TOL).all() else 'DOES NOT MEET'}\n")
    print("## Per-arm summary (mean over checkpoints)\n"); print(df.groupby(["task", "arm", "shift"]).agg(loss=("loss", "mean"), acc_delta=("acc_delta", "mean"), one_minus_cc=("one_minus_cc", "mean"), one_minus_cka=("one_minus_cka", "mean")).round(4).to_string())


KEEP_PNML = False

if __name__ == "__main__":
    KEEP_PNML = "--keep-pnml" in sys.argv[2:]
    main(pathlib.Path(sys.argv[1]))

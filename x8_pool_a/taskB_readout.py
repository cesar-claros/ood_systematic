"""Single readout of the adaptation pilot (task A and task B) against the frozen protocol of
documentation/adaptation_pilot_taskA_synthesis_2026-09-16.md (P1, P2, P3, rule R; primary sets SVHN and Textures; secondary sets after).
Run: .venv/bin/python x8_pool_a/taskB_readout.py > ../documentation/adaptation_results/taskB_readout.md
"""
import json
import pathlib

import numpy as np
import pandas as pd

R = pathlib.Path(__file__).resolve().parents[2] / "documentation" / "adaptation_results"
RUNS = ["A_lora_seed0", "A_lora_seed1", "A_full_seed0", "A_full_seed1", "B_lora_seed0", "B_lora_seed1", "B_full_seed0", "B_full_seed1"]
LOGIT = ["MSR", "MLS", "Energy", "PE", "GEN", "REN", "GE", "PCE", "GradNorm", "pNML", "NCI"]
DENSITY = ["Maha", "MahaPP", "Residual", "PCA RecError global", "ViM", "NeCo", "NNGuide", "CTM", "fDBD"]
PRIMARY = ["svhn", "dtd"]; SECONDARY = ["places365", "food101"]; LAST = 3515
pd.set_option("display.width", 220)


def load(run):
    o = pd.read_csv(R / run / "outcomes.csv"); m = pd.read_csv(R / run / "measurements.csv").set_index("step"); L = json.load(open(R / run / "ledger.json")); return o, m, L


def auroc_table(o, ood, variant="adapted"):
    return o[(o.ood_set == ood) & (o.variant == variant)].pivot(index="step", columns="detector", values="auroc_allid")


print("# Adaptation pilot readout against the frozen protocol (2026-09-17)\n")
print("Eight trajectories: task A (CIFAR-100) and task B (Oxford-IIIT Pets), LoRA rank 8 at 2e-4 and full fine-tuning at 2e-5, seeds 0 and 1, 3,515 steps, checkpoints 0, 350, 700, 1400, 2100, 3515. All-ID versus OOD AUROC. Predictions and rule R are judged on the primary sets (SVHN, Textures); secondary sets (Places365, Food-101) follow descriptively.\n")
data = {r: load(r) for r in RUNS}

print("## Measurements at step 350 and at the end (task B in bold rows)\n")
print("| Run | Acc 0 / 350 / end | NC1 0 / end | Residue 0 / end | Norm CV 0 / 350 / end | CKA 350 / end | Drift end | Train loss end |")
print("|---|---|---|---|---|---|---|---|")
for r in RUNS:
    o, m, L = data[r]; cv = m.get("feat_norm_cv")
    cvs = f"{cv.loc[0]:.3f} / {cv.loc[350]:.3f} / {cv.loc[LAST]:.3f}" if cv is not None and not np.isnan(cv.loc[0]) else "n/a (not recorded)"
    print(f"| {r} | {m.id_test_acc.loc[0]:.3f} / {m.id_test_acc.loc[350]:.3f} / {m.id_test_acc.loc[LAST]:.3f} | {m.nc_var_collapse.loc[0]:.2f} / {m.nc_var_collapse.loc[LAST]:.2f} | {m.residue_energy_projector.loc[0]:.3f} / {m.residue_energy_projector.loc[LAST]:.3f} | {cvs} | {m.paired_cka_fit.loc[350]:.3f} / {m.paired_cka_fit.loc[LAST]:.3f} | {m.total_drift_fit.loc[LAST]:.2f} | {L.get('last_loss', float('nan')):.4f} |")

print("\n## P1: norm mechanism (judged on SVHN)\n")
print("Full fine-tuning: norm CV at step 350 > 1.5 x reference; unnormalized Maha and Residual AUROC at step 350 below reference; MahaPP not below reference. LoRA: norm CV at step 350 < reference; unnormalized Maha at the end above reference.\n")
print("| Run | Norm CV ratio at 350 | Maha 0 -> 350 -> end | Residual 0 -> 350 -> end | MahaPP 0 -> 350 -> end | P1 verdict |")
print("|---|---|---|---|---|---|")
for r in RUNS:
    o, m, L = data[r]; T = auroc_table(o, "svhn"); cv = m.get("feat_norm_cv"); ratio = (cv.loc[350] / cv.loc[0]) if cv is not None and not np.isnan(cv.loc[0]) else np.nan
    full = "full" in r
    if full:
        ok = (np.isnan(ratio) or ratio > 1.5) and T.loc[350, "Maha"] < T.loc[0, "Maha"] and T.loc[350, "Residual"] < T.loc[0, "Residual"] and T.loc[350, "MahaPP"] >= T.loc[0, "MahaPP"]
        verdict = ("HOLDS" if ok else "FAILS") + ("" if not np.isnan(ratio) else " (CV not recorded; detector part only)")
    else:
        ok = (np.isnan(ratio) or ratio < 1.0) and T.loc[LAST, "Maha"] > T.loc[0, "Maha"]; verdict = ("HOLDS" if ok else "FAILS") + ("" if not np.isnan(ratio) else " (CV not recorded; detector part only)")
    print(f"| {r} | {ratio:.2f} | {T.loc[0,'Maha']:.3f} -> {T.loc[350,'Maha']:.3f} -> {T.loc[LAST,'Maha']:.3f} | {T.loc[0,'Residual']:.3f} -> {T.loc[350,'Residual']:.3f} -> {T.loc[LAST,'Residual']:.3f} | {T.loc[0,'MahaPP']:.3f} -> {T.loc[350,'MahaPP']:.3f} -> {T.loc[LAST,'MahaPP']:.3f} | {verdict} |")

print("\n## P2: reference retention (judged at the final checkpoint)\n")
print("Full fine-tuning: on Textures the frozen reference density detector beats the adapted one; the reference-plus-adapted combination is the best variant for the logit family on both primary shifts. LoRA: neither holds.\n")
print("| Run | Textures Maha adapted / reference | Textures MahaPP adapted / reference | Combined best for logit family (Energy, MLS, NCI) on SVHN / Textures | P2 verdict |")
print("|---|---|---|---|---|")
for r in RUNS:
    o, m, L = data[r]; F = o[o.step == LAST]
    def var(ood, det): return F[(F.ood_set == ood) & (F.detector == det)].set_index("variant").auroc_allid
    ma, mp = var("dtd", "Maha"), var("dtd", "MahaPP")
    ref_beats = (ma["reference"] > ma["adapted"]) and (mp["reference"] > mp["adapted"])
    comb = {ood: all(var(ood, d)["combined"] >= max(var(ood, d)["adapted"], var(ood, d)["reference"]) for d in ["Energy", "MLS", "NCI"]) for ood in PRIMARY}
    full = "full" in r; ok = (ref_beats and comb["svhn"] and comb["dtd"]) if full else (not ref_beats and not (comb["svhn"] and comb["dtd"]))
    print(f"| {r} | {ma['adapted']:.3f} / {ma['reference']:.3f} | {mp['adapted']:.3f} / {mp['reference']:.3f} | {comb['svhn']} / {comb['dtd']} | {'HOLDS' if ok else 'FAILS'} |")

print("\n## P3: no family switch; always-reference-best regret at most 0.02 AUROC per checkpoint\n")
print("| Run | Shift | Family winner by checkpoint (L = logit incl. NCI, D = density) | Best-at-reference detector | Max regret of always-that-detector | P3 verdict |")
print("|---|---|---|---|---|---|")
for r in RUNS:
    o, m, L = data[r]
    for ood in PRIMARY:
        T = auroc_table(o, ood); fam = ["L" if T[LOGIT].max(axis=1).loc[s] >= T[DENSITY].max(axis=1).loc[s] else "D" for s in T.index]
        d0 = T.loc[0].idxmax(); reg = (T.max(axis=1) - T[d0]).max()
        ok = len(set(fam)) == 1 and reg <= 0.02
        print(f"| {r} | {ood} | {''.join(fam)} | {d0} | {reg:.3f} | {'HOLDS' if ok else 'FAILS'} |")

print("\n## Rule R: executed AUROC per family against the baselines (primary shifts, all checkpoints after the reference)\n")
print("R: density family MahaPP, logit family NCI; combine with the frozen reference by rank mean when paired CKA < 0.85, else adapted alone. Baselines: always the detector best at the reference (adapted), the frozen reference version of that detector, the per-checkpoint oracle over all adapted detectors. Thresholds: gain 0.01 pooled, harm 0.01 on any shift.\n")
print("| Run | Shift | R density (MahaPP) | R logit (NCI) | Always-ref-best (adapted) | Frozen reference detector | Oracle | R best-family minus always-ref-best |")
print("|---|---|---|---|---|---|---|---|")
pooled = {"R_best": [], "base": [], "harm": {}}
for r in RUNS:
    o, m, L = data[r]
    for ood in PRIMARY:
        T = auroc_table(o, ood); Tc = auroc_table(o, ood, "combined"); Tr = auroc_table(o, ood, "reference"); steps = [s for s in T.index if s > 0]
        d0 = T.loc[0].idxmax()
        def r_val(det): return np.mean([(Tc.loc[s, det] if m.paired_cka_fit.loc[s] < 0.85 else T.loc[s, det]) for s in steps])
        rd, rl = r_val("MahaPP"), r_val("NCI"); base = np.mean([T.loc[s, d0] for s in steps]); frozen = np.mean([Tr.loc[s, d0] for s in steps]); orc = np.mean([T.loc[s].max() for s in steps])
        best = max(rd, rl); pooled["R_best"].append(best); pooled["base"].append(base); pooled["harm"][(r, ood)] = best - base
        print(f"| {r} | {ood} | {rd:.3f} | {rl:.3f} | {base:.3f} ({d0}) | {frozen:.3f} | {orc:.3f} | {best-base:+.3f} |")
gain = np.mean(pooled["R_best"]) - np.mean(pooled["base"]); worst = min(pooled["harm"].values())
print(f"\nPooled R (best family, which needs the shift type) minus always-reference-best: {gain:+.4f} AUROC; worst shift-level difference {worst:+.4f}. Threshold: gain >= 0.01 and no shift below -0.01. Verdict: {'MEETS' if gain >= 0.01 and worst >= -0.01 else 'DOES NOT MEET'} (and the family choice is not ID-observable).")

print("\n## Secondary sets (descriptive): Places365 and Food-101, task B\n")
print("| Run | Shift | Best detector at reference (AUROC) | Best at end (AUROC) | Maha 0 -> end | MahaPP 0 -> end | Energy 0 -> end | NCI 0 -> end | Reference-best detector at end (reference features) |")
print("|---|---|---|---|---|---|---|---|---|")
for r in [x for x in RUNS if x.startswith("B_")]:
    o, m, L = data[r]
    for ood in SECONDARY:
        if ood not in set(o.ood_set): continue
        T = auroc_table(o, ood); Tr = auroc_table(o, ood, "reference"); d0 = T.loc[0].idxmax(); dl = T.loc[LAST].idxmax()
        print(f"| {r} | {ood} | {d0} ({T.loc[0,d0]:.3f}) | {dl} ({T.loc[LAST,dl]:.3f}) | {T.loc[0,'Maha']:.3f} -> {T.loc[LAST,'Maha']:.3f} | {T.loc[0,'MahaPP']:.3f} -> {T.loc[LAST,'MahaPP']:.3f} | {T.loc[0,'Energy']:.3f} -> {T.loc[LAST,'Energy']:.3f} | {T.loc[0,'NCI']:.3f} -> {T.loc[LAST,'NCI']:.3f} | {Tr.loc[LAST].max():.3f} |")
print("\n## Full task-B detector trajectories on the primary shifts (adapted)\n")
for r in [x for x in RUNS if x.startswith("B_")]:
    o, m, L = data[r]
    for ood in PRIMARY:
        print(f"\n{r}, {ood}:\n"); print(auroc_table(o, ood)[["MSR", "Energy", "NCI", "CTM", "Maha", "MahaPP", "NNGuide", "ViM", "Residual", "NeCo", "GradNorm"]].round(3).to_markdown())

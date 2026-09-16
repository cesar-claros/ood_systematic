"""Headroom decomposition and regime-free executable policies (2026-09-15, descriptive, unregistered).

Follows the 2026-09-15 review of outputs/43_selection_headroom_diagnostic.md. Changes relative to that script:
(1) one global fixed rule (CTM, chosen on all VGG13 OOD rows) on every pool, including the probes;
(2) the review's exact decomposition L0 - Lrow = (L0 - Lfixed) + (Lfixed - Lcheckpoint) + (Lcheckpoint - Lrow);
(3) the exploratory ridge loss policy fitted once on all VGG13 OOD rows (no regime channel), one detector per checkpoint;
(4) a per-paradigm metadata rule; (5) per-checkpoint abstention with a threshold sweep and achieved coverage.
Units: AUGRC x1e3; regret vs the per-row oracle. Development data; no claim attaches.
Run: ../.venv/bin/python evaluation/selection_headroom_decomposition.py > outputs/44_selection_headroom_decomposition.md
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
NC = ['var_collapse', 'equiangular_uc', 'equiangular_wc', 'equinorm_uc', 'equinorm_wc', 'max_equiangular_uc', 'max_equiangular_wc', 'self_duality']
REG = ["near", "mid", "far"]
L = pd.read_parquet("outputs/track1/dataset/long.parquet"); L = L[L.regime.isin(REG)].copy()
L["dropout"] = L["dropout"].astype(int); L["reward"] = L["reward"].astype(float)
L["model_id"] = (L.architecture + "|" + L.paradigm + "|" + L.source + "|" + L.run.astype(int).astype(str) + "|" + L.dropout.astype(str) + "|" + L.reward.map(lambda r: "%g" % r))
piv = L.pivot_table(index=["architecture", "model_id", "eval_dataset", "regime"], columns="csf", values="augrc", aggfunc="first")
feat = L.drop_duplicates("model_id").set_index("model_id")[NC + ["architecture", "paradigm"]]


def rows(arch):
    T = piv.xs(arch, level="architecture", drop_level=False); return T[[c for c in T.columns if T[c].notna().all()]]


P = pd.read_parquet("outputs/pool_a/long_pool_a_harmonized.parquet"); P = P[P.regime.isin(REG)]
Pp = P.pivot_table(index=["paradigm", "source", "run", "eval_dataset", "regime"], columns="csf", values="augrc", aggfunc="first")
Pp = Pp[[c for c in Pp.columns if Pp[c].notna().all()]]
VGG = rows("VGG13")
print("# Headroom decomposition and regime-free executable policies (descriptive, 2026-09-15)\n")
print("Global fixed rule chosen on all VGG13 OOD rows:", VGG.mean().idxmin(), "\n")
print("## Global CTM regret by regime\n")
for label, T in [("ResNet18", rows("ResNet18")), ("ViT", rows("ViT")), ("probes (harmonized)", Pp)]:
    o = T.min(axis=1); r = (T["CTM"] - o).groupby(level="regime").mean().round(2).to_dict(); lv = o.groupby(level="regime").mean().round(1).to_dict()
    print(f"- {label}: regret near/mid/far {r['near']} / {r['mid']} / {r['far']}; oracle level {lv['near']} / {lv['mid']} / {lv['far']}")
print("\n## Decomposition (all OOD rows pooled; checkpoint = one trained model)\n")


def decomp(T, ck, label):
    o = T.min(axis=1); L0 = T["CTM"].mean(); Lf = T.mean().min(); fx = T.mean().idxmin()
    means = T.groupby(level=ck).mean(); best = means.idxmin(axis=1)
    key = T.index.to_frame(index=False)[ck].apply(tuple, axis=1) if len(ck) > 1 else T.index.get_level_values(ck[0])
    Lck = float(np.mean([T.iloc[i][best.loc[k]] for i, k in enumerate(key)])); Lr = o.mean()
    print(f"- {label}: rows {len(T)}, detectors {T.shape[1]}; total {L0-Lr:.3f} = fixed-switch ({fx}) {L0-Lf:.3f} + checkpoint-specific {Lf-Lck:.3f} + shift-specific {Lck-Lr:.3f}; one-detector-per-checkpoint ceiling over CTM {L0-Lck:.3f}")


decomp(rows("ResNet18"), ["model_id"], "ResNet18"); decomp(rows("ViT"), ["model_id"], "ViT"); decomp(Pp, ["paradigm", "source", "run"], "probes")
print("\nIllustrative AUROC-equivalent (Traub et al. 2024, Eq. 7, at failure fraction 0.5): one displayed unit = 0.004 AUROC.\n")


def ridge_fit(X, y, a=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))]); return np.linalg.solve(Xb.T @ Xb + a * np.diag([1] * X.shape[1] + [0]), Xb.T @ y)


print("## Regime-free exploratory ridge loss policy (one head set on all VGG13 OOD rows; one detector per checkpoint; alpha = 1)\n")
for arch in ["ResNet18", "ViT"]:
    T = rows(arch); common = [c for c in T.columns if c in VGG.columns]; V = VGG[common]; T = T[common]
    RV = V.sub(V.min(axis=1), axis=0).groupby(level="model_id").mean(); RT = T.sub(T.min(axis=1), axis=0)
    fv = feat.loc[RV.index, NC].values.astype(float); tm = T.index.get_level_values("model_id"); fm = feat.loc[tm.unique(), NC].values.astype(float)
    for st in ["train_only", "target_pool"]:
        if st == "train_only":
            mu, sd = fv.mean(0), fv.std(0) + 1e-12; a, b = (fv - mu) / sd, (fm - mu) / sd
        else:
            a = (fv - fv.mean(0)) / (fv.std(0) + 1e-12); b = (fm - fm.mean(0)) / (fm.std(0) + 1e-12)
        pred = pd.DataFrame({c: np.hstack([b, np.ones((len(b), 1))]) @ ridge_fit(a, RV[c].values.astype(float)) for c in common}, index=tm.unique())
        sel = pred.idxmin(axis=1); r = np.array([RT.iloc[i][sel.loc[m]] for i, m in enumerate(tm)])
        per = pd.Series(r, index=T.index).groupby(level="regime").mean().round(2).to_dict()
        print(f"- {arch}, {st}: pooled {r.mean():.2f} (near/mid/far {per['near']} / {per['mid']} / {per['far']}); CTM pooled {RT['CTM'].mean():.2f}")
    tot = []
    for i, m in enumerate(tm):
        par = feat.loc[m, "paradigm"]; Vp = V[V.index.get_level_values("model_id").map(lambda x: x.split("|")[1] == par)]
        tot.append(RT.iloc[i][Vp.mean().idxmin() if len(Vp) else V.mean().idxmin()])
    print(f"- {arch}, per-paradigm metadata rule (best VGG13 detector of the same paradigm, CTM if none): pooled {np.mean(tot):.2f}")
print("\n## Per-checkpoint abstention on ResNet18 (one decision per checkpoint; fallback CTM; thresholds not chosen on training-side data)\n")
for cfg in ["none_nr_marginal", "source_nr_marginal"]:
    Pd = pd.read_parquet(f"outputs/ablations/calib_cliques_regime_free/track1/xarch/{cfg}/preds.parquet")
    pc = [c for c in Pd.columns if c.startswith("p_")][0]; Q = Pd.pivot_table(index="model_id", columns="csf", values=pc, aggfunc="first")
    T = rows("ResNet18"); Q = Q.reindex(columns=T.columns); RT = T.sub(T.min(axis=1), axis=0); tm = T.index.get_level_values("model_id")
    top = Q.idxmax(axis=1); conf = Q.max(axis=1); out = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        acc = conf >= tau; r = np.array([RT.iloc[i][top.loc[m]] if acc.loc[m] else RT.iloc[i]["CTM"] for i, m in enumerate(tm)])
        out.append(dict(config=cfg, tau=tau, coverage_checkpoints=round(float(acc.reindex(tm.unique()).mean()), 2), policy_pooled=round(float(r.mean()), 2), CTM_pooled=round(float(RT["CTM"].mean()), 2)))
    print(pd.DataFrame(out).to_string(index=False), "\n")

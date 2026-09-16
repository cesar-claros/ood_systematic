"""Selection-headroom and executable-policy diagnostic (2026-09-14, descriptive, unregistered).

Reads only saved artifacts: outputs/track1/dataset/long.parquet, outputs/pool_a/long_pool_a_harmonized.parquet,
outputs/track1/cliques/cliques.parquet, and the stored selector predictions under outputs/ablations/.
Per-row regret = AUGRC(policy detector) - min over available detectors (AUGRC in x1e-3 units, the paper's convention).
Empty shortlists are imputed at the worst available detector (paper convention). Development data; no claim attaches.
Run: ../.venv/bin/python evaluation/selection_headroom_diagnostic.py > outputs/43_selection_headroom_diagnostic.md
"""
import glob
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
NC = ['var_collapse', 'equiangular_uc', 'equiangular_wc', 'equinorm_uc', 'equinorm_wc',
      'max_equiangular_uc', 'max_equiangular_wc', 'self_duality']
REG = ["near", "mid", "far"]

L = pd.read_parquet("outputs/track1/dataset/long.parquet")
L = L[L.regime.isin(REG)].copy()
L["dropout"] = L["dropout"].astype(int); L["reward"] = L["reward"].astype(float)
L["model_id"] = (L.architecture + "|" + L.paradigm + "|" + L.source + "|" + L.run.astype(int).astype(str)
                 + "|" + L.dropout.astype(str) + "|" + L.reward.map(lambda r: "%g" % r))
piv = L.pivot_table(index=["architecture", "model_id", "eval_dataset", "regime"], columns="csf", values="augrc", aggfunc="first")
feat = L.drop_duplicates("model_id").set_index("model_id")[NC + ["architecture", "paradigm", "source"]]
arch_of = feat.architecture


def rows(arch, reg=None):
    T = piv.xs(arch, level="architecture", drop_level=False)
    if reg:
        T = T.xs(reg, level="regime", drop_level=False)
    return T[[c for c in T.columns if T[c].notna().all()]]


def headroom(target, train, label):
    out = []
    for reg in REG:
        T = target.xs(reg, level="regime", drop_level=False); T = T[[c for c in T.columns if T[c].notna().all()]]
        o = T.min(axis=1)
        TR = train.xs(reg, level="regime", drop_level=False)[[c for c in T.columns if c in train.columns]].dropna(axis=1, how="any")
        ftr, fhs = TR.mean().idxmin(), T.mean().idxmin()
        out.append(dict(pool=label, regime=reg, n_rows=len(T), n_det=T.shape[1], oracle_level=round(o.mean(), 1),
                        train_fixed=ftr, regret_train_fixed=round((T[ftr] - o).mean(), 2),
                        hindsight_fixed=fhs, regret_hindsight=round((T[fhs] - o).mean(), 2),
                        regret_random=round(T.sub(o, axis=0).mean(axis=1).mean(), 2),
                        rel_headroom_pct=round(100 * (T[ftr] - o).mean() / o.mean(), 2)))
    return pd.DataFrame(out)


print("# Selection headroom and executable-policy diagnostic (descriptive, 2026-09-14)\n")
print("AUGRC scale by architecture (mean over OOD rows):", L.groupby("architecture").augrc.mean().round(1).to_dict(), "\n")
VGG = rows("VGG13")
tabs = [headroom(rows("ResNet18"), VGG, "ResNet18<-VGG13"), headroom(rows("ViT"), VGG, "ViT<-VGG13")]
for src in ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]:
    tabs.append(headroom(VGG.xs(src, level=None, drop_level=False) if False else VGG[VGG.index.get_level_values("model_id").str.contains("|" + src + "|", regex=False)],
                         VGG[~VGG.index.get_level_values("model_id").str.contains("|" + src + "|", regex=False)], f"VGG lodo:{src}"))
print("## Headroom: per-regime training-chosen fixed detector vs per-row oracle\n")
print(pd.concat(tabs).to_string(index=False), "\n")
for arch in ["ResNet18", "ViT"]:
    T = rows(arch); V = VGG[[c for c in T.columns if c in VGG.columns]]; d = V.mean().idxmin(); o = T.min(axis=1)
    per = (T[d] - o).groupby(level="regime").mean().round(2).to_dict()
    print(f"One-detector fixed rule chosen on all VGG13 OOD rows, applied to {arch}: {d}; regret near/mid/far = {per['near']} / {per['mid']} / {per['far']}")
print()

P = pd.read_parquet("outputs/pool_a/long_pool_a_harmonized.parquet"); P = P[P.regime.isin(REG)]
Pp = P.pivot_table(index=["paradigm", "source", "run", "eval_dataset", "regime"], columns="csf", values="augrc", aggfunc="first")
rws = []
for enc in sorted(P.paradigm.unique()) + ["both"]:
    Q = Pp if enc == "both" else Pp.xs(enc, level="paradigm", drop_level=False)
    for reg in REG:
        T = Q.xs(reg, level="regime", drop_level=False); T = T[[c for c in T.columns if T[c].notna().all()]]; o = T.min(axis=1)
        TR = VGG.xs(reg, level="regime", drop_level=False)[[c for c in T.columns if c in VGG.columns]].dropna(axis=1, how="any")
        ftr, fhs = TR.mean().idxmin(), T.mean().idxmin()
        rws.append(dict(pool=f"probes:{enc}", regime=reg, n_rows=len(T), n_det=T.shape[1], oracle_level=round(o.mean(), 1), train_fixed=ftr,
                        regret_train_fixed=round((T[ftr] - o).mean(), 2), hindsight_fixed=fhs, regret_hindsight=round((T[fhs] - o).mean(), 2),
                        regret_random=round(T.sub(o, axis=0).mean(axis=1).mean(), 2), rel_headroom_pct=round(100 * (T[ftr] - o).mean() / o.mean(), 2)))
print("## Probe pool headroom (harmonized 21-detector table)\n"); print(pd.DataFrame(rws).to_string(index=False), "\n")

cl = pd.read_parquet("outputs/track1/cliques/cliques.parquet")
if "architecture" in cl.columns: cl = cl[cl.architecture == "VGG13"]
if "regime" in cl.columns: cl = cl[cl.regime.isin(REG)]
pi = cl.groupby("csf")["in_top_clique"].mean()


def evaluate(pred_path, arch, fallback, fold=None, label=""):
    Pd = pd.read_parquet(pred_path)
    if fold is not None and "fold_label" in Pd.columns: Pd = Pd[Pd.fold_label == fold]
    pc = [c for c in Pd.columns if c.startswith("p_") or c in ("proba", "prob")][0]
    Pd = Pd[Pd.regime.isin(REG)] if ("regime" in Pd.columns and Pd.regime.isin(REG).any()) else Pd.assign(regime="all")
    ids = set(Pd.model_id) & set(arch_of.index[arch_of == arch])
    print(f"### {label} ({pred_path.split('outputs/')[1]}, fold={fold}, models={len(ids)})\n")
    out = []
    for reg in REG:
        T = piv.xs(reg, level="regime", drop_level=False); T = T[T.index.get_level_values("model_id").isin(ids)]
        T = T[[c for c in T.columns if T[c].notna().all()]]; o = T.min(axis=1); w = T.max(axis=1)
        Q = Pd[Pd.regime.isin([reg, "all"])].pivot_table(index="model_id", columns="csf", values=pc, aggfunc="first").reindex(columns=T.columns)
        Q = Q.reindex(T.index.get_level_values("model_id"))
        top1 = Q.idxmax(axis=1); r_top1 = np.array([T.iloc[i][top1.iloc[i]] - o.iloc[i] for i in range(len(T))])
        pv = pi.reindex(T.columns).fillna(pi.mean()).values
        Pc = (pv * Q) / (pv * Q + (1 - pv) * (1 - Q)); topc = Pc.idxmax(axis=1)
        r_topc = np.array([T.iloc[i][topc.iloc[i]] - o.iloc[i] for i in range(len(T))])
        S = Q > 0.5; rb, rr, sz = [], [], []
        for i in range(len(T)):
            m = [c for c in T.columns if bool(S.iloc[i][c])]; sz.append(len(m))
            rb.append((T.iloc[i][m].min() if m else w.iloc[i]) - o.iloc[i]); rr.append((T.iloc[i][m].mean() if m else w.iloc[i]) - o.iloc[i])
        r_fb = (T[fallback] - o).values; conf = Q.max(axis=1).values; order = np.argsort(-conf); cov = {}
        for c in (1.0, 0.8, 0.6, 0.4, 0.2):
            k = int(round(c * len(T))); acc = np.zeros(len(T), bool); acc[order[:k]] = True
            cov[f"policy@{int(c*100)}%"] = round(float(np.where(acc, r_top1, r_fb).mean()), 2)
        out.append(dict(regime=reg, n=len(T), fixed=fallback, regret_fixed=round(r_fb.mean(), 2), top1_raw=round(r_top1.mean(), 2),
                        top1_prior_corrected=round(r_topc.mean(), 2), best_member=round(np.mean(rb), 2), random_member=round(np.mean(rr), 2),
                        set_size=round(np.mean(sz), 1), empty=round(float(np.mean(np.array(sz) == 0)), 2), **cov))
    print(pd.DataFrame(out).to_string(index=False), "\n")


print("## Executable policies from the stored heads (fixed = one-detector rule chosen on VGG13; abstention falls back to it)\n")
evaluate("outputs/ablations/calib_cliques_regime_free/track1/xarch/none_nr_marginal/preds.parquet", "ResNet18", "CTM", label="ResNet18, NC only, regime-free")
evaluate("outputs/ablations/calib_cliques_regime_free/track1/xarch/source_nr_marginal/preds.parquet", "ResNet18", "CTM", label="ResNet18, NC+source, regime-free")
evaluate("outputs/ablations/calib_cliques/track1/xarch/source/preds.parquet", "ResNet18", "CTM", label="ResNet18, paper arm with regime input")
evaluate("outputs/ablations/calib_cliques_regime_free/track1/lopo/none_nr_marginal/preds.parquet", "ViT", "CTM", fold="lopo_modelvit", label="ViT, NC only, regime-free")


def ridge_fit(X, y, alpha=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))]); A = Xb.T @ Xb + alpha * np.diag([1] * X.shape[1] + [0]); return np.linalg.solve(A, Xb.T @ y)


def loss_regressor(arch, standardize):
    T = rows(arch); V = VGG[[c for c in T.columns if c in VGG.columns]]; T = T[V.columns]
    RVm = V.sub(V.min(axis=1), axis=0).groupby(level=["model_id", "regime"]).mean(); RT = T.sub(T.min(axis=1), axis=0)
    fv = feat.loc[RVm.index.get_level_values("model_id"), NC].values.astype(float); ft = feat.loc[T.index.get_level_values("model_id"), NC].values.astype(float)
    if standardize == "train_only":
        mu, sd = fv.mean(0), fv.std(0) + 1e-12; fv, ft = (fv - mu) / sd, (ft - mu) / sd
    else:
        fa = feat.loc[feat.architecture == arch, NC].values.astype(float)
        fv = (fv - fv.mean(0)) / (fv.std(0) + 1e-12); ft = (ft - fa.mean(0)) / (fa.std(0) + 1e-12)
    out = {}
    for reg in REG:
        m = RVm.index.get_level_values("regime") == reg
        Pm = pd.DataFrame({c: np.hstack([ft, np.ones((len(ft), 1))]) @ ridge_fit(fv[m], RVm.loc[m, c].values.astype(float)) for c in V.columns}, index=T.index)
        sel = Pm.idxmin(axis=1); idx = np.where(T.index.get_level_values("regime") == reg)[0]
        out[reg] = round(float(np.mean([RT.iloc[i][sel.iloc[i]] for i in idx])), 2)
    return out


print("## Exploratory per-detector ridge loss regressor on the 8 NC features (alpha = 1, trained on VGG13), top-1 regret\n")
for arch in ["ResNet18", "ViT"]:
    for st in ["train_only", "target_pool"]:
        print(f"- {arch}, {st} standardization: {loss_regressor(arch, st)}")

"""Post-outcome correctness sensitivity of the matched screen (2026-09-15): NC1 replaced by the corrected measurement; per-source and
alternative-weight rescoring of the M1 decisions. Same procedure as within_family_matched_screen.py; only the NC1 feature changes.
Run: ../.venv/bin/python evaluation/within_family_matched_screen_nc1.py > outputs/47_matched_screen_nc1_sensitivity.md
"""
import numpy as np
import pandas as pd

REG = ["near", "mid", "far"]
NC = ['var_collapse', 'equiangular_uc', 'equiangular_wc', 'equinorm_uc', 'equinorm_wc', 'max_equiangular_uc', 'max_equiangular_wc', 'self_duality']
GROUPS = {"cifar10": ["cifar10"], "cifar100+supercifar100": ["cifar100", "supercifar100"], "tinyimagenet": ["tinyimagenet"]}
NCLS = {"cifar10": 10, "cifar100": 100, "supercifar100": 20, "tinyimagenet": 200}


def load(nc1_col):
    L = pd.read_parquet("outputs/track1/dataset/long_harmonized_v2.parquet"); L = L[L.architecture == "VGG13"].copy()
    L["dropout"] = L["dropout"].astype(int); L["reward"] = L["reward"].astype(float)
    L["model_id"] = L.paradigm + "|" + L.source + "|" + L.run.astype(int).astype(str) + "|" + L.dropout.astype(str) + "|" + L.reward.map(lambda r: "%g" % r)
    L["var_collapse"] = L[nc1_col].astype(float)
    O = L[L.regime.isin(REG)]; ID = L[L.regime == "test"]
    piv = O.pivot_table(index=["model_id", "source", "paradigm", "eval_dataset"], columns="csf", values="augrc", aggfunc="first")
    idpiv = ID.pivot_table(index="model_id", columns="csf", values="augrc", aggfunc="first")
    meta = O.drop_duplicates("model_id").set_index("model_id")[NC + ["source", "paradigm", "dropout", "reward"]]
    return piv, idpiv, meta


def ridge_fit(X, y, a):
    Xb = np.hstack([X, np.ones((len(X), 1))]); return np.linalg.solve(Xb.T @ Xb + a * np.diag([1] * X.shape[1] + [0]), Xb.T @ y)


def features(meta, idpiv, ids, kind, dets, stats=None):
    m = meta.loc[ids]; cols = []
    if "M" in kind:
        cols.append(pd.DataFrame({"logC": np.log10(m.source.map(NCLS).astype(float)), "id_msr": idpiv.loc[ids, "MSR"].values, "id_mean": idpiv.loc[ids, dets].mean(axis=1).values,
                                  "p_confidnet": (m.paradigm == "confidnet").astype(float), "p_devries": (m.paradigm == "devries").astype(float), "dropout": m.dropout.astype(float), "reward": m.reward}, index=ids))
        if kind.startswith("M2"): cols.append(idpiv.loc[ids, dets].add_prefix("id_"))
    if "G" in kind: cols.append(m[NC].astype(float))
    X = pd.concat(cols, axis=1).astype(float)
    if stats is None: stats = (X.mean(0), X.std(0).replace(0, 1.0) + 1e-12)
    return ((X - stats[0]) / stats[1]).values, stats


def decide(meta, idpiv, train_piv, test_piv, kind, alpha):
    dets = list(train_piv.columns); tr = train_piv.index.get_level_values("model_id").unique(); te = test_piv.index.get_level_values("model_id").unique()
    RTr = train_piv.sub(train_piv.min(axis=1), axis=0).groupby(level="model_id").mean().loc[tr]
    Xtr, st = features(meta, idpiv, tr, kind, dets); Xte, _ = features(meta, idpiv, te, kind, dets, st)
    pred = pd.DataFrame({c: np.hstack([Xte, np.ones((len(Xte), 1))]) @ ridge_fit(Xtr, RTr[c].values, alpha) for c in dets}, index=te)
    return pred.idxmin(axis=1)


def regret_of(sel, test_piv):
    RTe = test_piv.sub(test_piv.min(axis=1), axis=0); return pd.Series([RTe.iloc[i][sel.loc[k]] for i, k in enumerate(test_piv.index.get_level_values("model_id"))], index=test_piv.index)


def run(piv, idpiv, meta, kinds=("M1", "M1+G", "M2", "M2+G", "G"), alphas=(0.1, 1, 10, 100, 1000)):
    out = {}; sels = {}
    for g, srcs in GROUPS.items():
        src = piv.index.get_level_values("source"); ev = piv.index.get_level_values("eval_dataset")
        T = piv[src.isin(srcs)]; TR = piv[(~src.isin(srcs)) & (~ev.isin(srcs))]
        RT = T.sub(T.min(axis=1), axis=0); out[(g, "FIXED")] = RT[TR.mean().idxmin()]
        for k in kinds:
            best_a, best_v = None, np.inf
            for a in alphas:
                v = 0.0; n = 0
                for ig in [gg for gg in GROUPS if gg != g]:
                    isrc = GROUPS[ig]; s2 = TR.index.get_level_values("source"); e2 = TR.index.get_level_values("eval_dataset")
                    Tin = TR[s2.isin(isrc)]; TRin = TR[(~s2.isin(isrc)) & (~e2.isin(isrc))]
                    v += regret_of(decide(meta, idpiv, TRin, Tin, k, a), Tin).sum(); n += len(Tin)
                if v / n < best_v: best_v, best_a = v / n, a
            sel = decide(meta, idpiv, TR, T, k, best_a); out[(g, k)] = regret_of(sel, T); sels[(g, k)] = sel
    return out, sels


def pooled(out, kinds):
    return {k: float(pd.concat([out[(g, k)] for g in GROUPS]).mean()) for k in ["FIXED"] + list(kinds)}


print("# Matched screen: NC1 correctness sensitivity and M1 rescoring (descriptive, 2026-09-15)\n")
Lh = pd.read_parquet("outputs/track1/dataset/long_harmonized_v2.parquet"); Lh = Lh[Lh.architecture == "VGG13"]
prov = [c for c in Lh.columns if "corrected" in c or "provenance" in c or "nc1" in c.lower()]
print("columns with corrected/provenance:", prov)
m = Lh.drop_duplicates(["paradigm", "source", "run", "dropout", "reward"])
print("\n| Source | Median legacy NC1 | Median corrected NC1 | Median ratio |\n|---|---:|---:|---:|")
for s in ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]:
    ms = m[m.source == s]; print(f"| {s} | {ms.var_collapse.median():.5f} | {ms.var_collapse_corrected.median():.5f} | {(ms.var_collapse_corrected / ms.var_collapse).median():.2f} |")
kinds = ("M1", "M1+G", "M2", "M2+G", "G")
res = {}
for label, col in [("legacy NC1", "var_collapse"), ("corrected NC1", "var_collapse_corrected")]:
    piv, idpiv, meta = load(col)
    for inv, P in [("20 detectors", piv), ("without Confidence", piv.drop(columns=["Confidence"]))]:
        out, sels = run(P, idpiv, meta); res[(label, inv)] = (out, sels, pooled(out, kinds))
print("\n## Pooled regret (AUGRC x1e3)\n\n| Policy | Legacy, 20 | Corrected, 20 | Legacy, no Conf | Corrected, no Conf |\n|---|---:|---:|---:|---:|")
for k in ["FIXED"] + list(kinds):
    print(f"| {k} | " + " | ".join(f"{res[(l, i)][2][k]:.3f}" for i in ["20 detectors", "without Confidence"] for l in ["legacy NC1", "corrected NC1"]) + " |")
out, sels, _ = res[("corrected NC1", "20 detectors")]
print(f"\nCorrected NC1, 20 detectors: M1+G on tinyimagenet {out[('tinyimagenet','M1+G')].mean():.3f}; on the grouped pair {out[('cifar100+supercifar100','M1+G')].mean():.3f}; M1 on tinyimagenet {out[('tinyimagenet','M1')].mean():.3f}")
out, sels, _ = res[("legacy NC1", "20 detectors")]
fx = pd.concat([out[(g, "FIXED")] for g in GROUPS]); m1 = pd.concat([out[(g, "M1")] for g in GROUPS])
src = fx.index.get_level_values("source"); par = fx.index.get_level_values("paradigm")
print("\n## M1 versus fixed by source (legacy run; M1 does not use NC1)\n\n| Source | Fixed | M1 | Improvement |\n|---|---:|---:|---:|")
for s in ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]:
    print(f"| {s} | {fx[src == s].mean():.3f} | {m1[src == s].mean():.3f} | {fx[src == s].mean() - m1[src == s].mean():.3f} |")
grp = pd.Series(src, index=fx.index).map({"cifar10": "g1", "cifar100": "g2", "supercifar100": "g2", "tinyimagenet": "g3"})
def wmean(v, key): return float(v.groupby(key.values).mean().mean())
print("\n| Weighting | Fixed | M1 | Improvement |\n|---|---:|---:|---:|")
for name, key in [("rows", None), ("equal named sources", pd.Series(src, index=fx.index)), ("equal image-source groups", grp), ("equal source/paradigm", pd.Series(src, index=fx.index) + "|" + pd.Series(par, index=fx.index))]:
    a = fx.mean() if key is None else wmean(fx, key); b = m1.mean() if key is None else wmean(m1, key); print(f"| {name} | {a:.3f} | {b:.3f} | {a-b:.3f} |")
for g in GROUPS:
    s = sels[(g, "M1")]; print(f"- M1 selections on {g}: " + ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))

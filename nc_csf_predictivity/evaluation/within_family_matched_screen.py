"""Grouped within-family decomposition (reproduces the 2026-09-15 source-transfer review) and a matched development screen.

DESIGN FROZEN BEFORE EXECUTION (one run; results reported as produced; no post-hoc changes):
Pool VGG-13 (280 checkpoints), OOD rows near/mid/far, regret vs the per-row oracle over the inventory (20 detectors; sensitivity: 19 without Confidence).
Grouped folds: {cifar10}, {cifar100, supercifar100}, {tinyimagenet}. Cross-role exclusion: training rows whose eval_dataset is the held-out
group's dataset name are dropped. Target per (checkpoint, detector): mean regret over the checkpoint's OOD rows.
Descriptor sets, standardized with training-fold statistics only:
  M1 = log10(n_classes), ID-test AUGRC of MSR, mean ID-test AUGRC over detectors, paradigm one-hot, dropout, reward   (cheap; no candidate fitting)
  M2 = M1 + per-detector ID-test AUGRC vector                                                                       (needs every candidate fitted on ID)
  G  = 8 Papyan NC metrics
Policies: FIXED (single detector chosen on the training fold), M1, M1+G, M2, M2+G, G. Model: per-detector ridge (own intercept and coefficients),
alpha in {0.1, 1, 10, 100, 1000} chosen by inner leave-one-training-group-out executed-policy regret, refit on the full training fold,
argmin predicted regret per held-out checkpoint. References: hindsight fixed, source-by-paradigm oracle, checkpoint oracle.
Units AUGRC x1e3. Development data inspected many times; a screen, not confirmation.
Run: ../.venv/bin/python evaluation/within_family_matched_screen.py > outputs/46_within_family_matched_screen.md
"""
import numpy as np
import pandas as pd

REG = ["near", "mid", "far"]
NC = ['var_collapse', 'equiangular_uc', 'equiangular_wc', 'equinorm_uc', 'equinorm_wc', 'max_equiangular_uc', 'max_equiangular_wc', 'self_duality']
GROUPS = {"cifar10": ["cifar10"], "cifar100+supercifar100": ["cifar100", "supercifar100"], "tinyimagenet": ["tinyimagenet"]}
NCLS = {"cifar10": 10, "cifar100": 100, "supercifar100": 20, "tinyimagenet": 200}
L = pd.read_parquet("outputs/track1/dataset/long.parquet"); L = L[L.architecture == "VGG13"].copy()
L["dropout"] = L["dropout"].astype(int); L["reward"] = L["reward"].astype(float)
L["model_id"] = L.paradigm + "|" + L.source + "|" + L.run.astype(int).astype(str) + "|" + L.dropout.astype(str) + "|" + L.reward.map(lambda r: "%g" % r)
O = L[L.regime.isin(REG)]; ID = L[L.regime == "test"]
piv_all = O.pivot_table(index=["model_id", "source", "paradigm", "eval_dataset"], columns="csf", values="augrc", aggfunc="first")
idpiv = ID.pivot_table(index="model_id", columns="csf", values="augrc", aggfunc="first")
meta = O.drop_duplicates("model_id").set_index("model_id")[NC + ["source", "paradigm", "dropout", "reward"]]


def ck_oracle(T):
    m = T.groupby(level="model_id").mean(); b = m.idxmin(axis=1)
    return float(np.mean([T.iloc[i][b.loc[k]] for i, k in enumerate(T.index.get_level_values("model_id"))]))


def sp_oracle(T):
    m = T.groupby(level=["source", "paradigm"]).mean(); b = m.idxmin(axis=1)
    key = list(zip(T.index.get_level_values("source"), T.index.get_level_values("paradigm")))
    return float(np.mean([T.iloc[i][b.loc[k]] for i, k in enumerate(key)]))


def decomposition(piv, label):
    print(f"\n## Grouped decomposition, {label} ({piv.shape[1]} detectors)\n")
    print("| Held-out group | Rows | Training rule | Regret | Source-fixed gain | Beyond source-fixed | Beyond source-by-paradigm | Ceiling over training rule |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for g, srcs in GROUPS.items():
        src = piv.index.get_level_values("source"); ev = piv.index.get_level_values("eval_dataset")
        T = piv[src.isin(srcs)]; TR = piv[(~src.isin(srcs)) & (~ev.isin(srcs))]
        d0 = TR.mean().idxmin(); o = T.min(axis=1); L0 = T[d0].mean(); Lf = T.mean().min(); Lck = ck_oracle(T); Lsp = sp_oracle(T); Lr = o.mean()
        print(f"| {g} | {len(T)} | {d0} | {L0-Lr:.3f} | {L0-Lf:.3f} | {Lf-Lck:.3f} | {Lsp-Lck:.3f} | {L0-Lck:.3f} |")


def ridge_fit(X, y, a):
    Xb = np.hstack([X, np.ones((len(X), 1))]); return np.linalg.solve(Xb.T @ Xb + a * np.diag([1] * X.shape[1] + [0]), Xb.T @ y)


def features(ids, kind, dets, stats=None):
    m = meta.loc[ids]; cols = []
    if "M" in kind:
        base = pd.DataFrame({"logC": np.log10(m.source.map(NCLS).astype(float)), "id_msr": idpiv.loc[ids, "MSR"].values, "id_mean": idpiv.loc[ids, dets].mean(axis=1).values,
                             "p_confidnet": (m.paradigm == "confidnet").astype(float), "p_devries": (m.paradigm == "devries").astype(float), "dropout": m.dropout.astype(float), "reward": m.reward}, index=ids)
        cols.append(base)
        if kind.startswith("M2"): cols.append(idpiv.loc[ids, dets].add_prefix("id_"))
    if "G" in kind: cols.append(m[NC].astype(float))
    X = pd.concat(cols, axis=1).astype(float)
    if stats is None: stats = (X.mean(0), X.std(0).replace(0, 1.0) + 1e-12)
    return ((X - stats[0]) / stats[1]).values, stats


def policy_regret(train_piv, test_piv, kind, alpha):
    dets = list(train_piv.columns); tr_ids = train_piv.index.get_level_values("model_id").unique(); te_ids = test_piv.index.get_level_values("model_id").unique()
    RTr = train_piv.sub(train_piv.min(axis=1), axis=0).groupby(level="model_id").mean().loc[tr_ids]
    Xtr, st = features(tr_ids, kind, dets); Xte, _ = features(te_ids, kind, dets, st)
    pred = pd.DataFrame({c: np.hstack([Xte, np.ones((len(Xte), 1))]) @ ridge_fit(Xtr, RTr[c].values, alpha) for c in dets}, index=te_ids)
    sel = pred.idxmin(axis=1); RTe = test_piv.sub(test_piv.min(axis=1), axis=0)
    return float(np.mean([RTe.iloc[i][sel.loc[k]] for i, k in enumerate(test_piv.index.get_level_values("model_id"))]))


def screen(piv, label):
    print(f"\n## Matched development screen, {label} ({piv.shape[1]} detectors); alpha by inner leave-one-training-group-out\n")
    kinds = ["M1", "M1+G", "M2", "M2+G", "G"]; alphas = [0.1, 1, 10, 100, 1000]
    print("| Held-out group | Rows | FIXED (training) | " + " | ".join(f"{k} (alpha)" for k in kinds) + " | Hindsight fixed | Checkpoint oracle |"); print("|---|---:|---:|" + "---:|" * len(kinds) + "---:|---:|")
    pooled = {k: 0.0 for k in ["FIXED"] + kinds + ["HF", "CK"]}; n_tot = 0
    for g, srcs in GROUPS.items():
        src = piv.index.get_level_values("source"); ev = piv.index.get_level_values("eval_dataset")
        T = piv[src.isin(srcs)]; TR = piv[(~src.isin(srcs)) & (~ev.isin(srcs))]
        o = T.min(axis=1); Lr = o.mean(); fixed = T[TR.mean().idxmin()].mean() - Lr; hf = T.mean().min() - Lr; ck = ck_oracle(T) - Lr
        row = {"FIXED": fixed, "HF": hf, "CK": ck}; cells = []
        inner_groups = [gg for gg in GROUPS if gg != g]
        for k in kinds:
            best_a, best_v = None, np.inf
            for a in alphas:
                v = 0.0; n = 0
                for ig in inner_groups:
                    isrc = GROUPS[ig]; s2 = TR.index.get_level_values("source"); e2 = TR.index.get_level_values("eval_dataset")
                    Tin = TR[s2.isin(isrc)]; TRin = TR[(~s2.isin(isrc)) & (~e2.isin(isrc))]
                    v += policy_regret(TRin, Tin, k, a) * len(Tin); n += len(Tin)
                if v / n < best_v: best_v, best_a = v / n, a
            r = policy_regret(TR, T, k, best_a); row[k] = r; cells.append(f"{r:.3f} ({best_a:g})")
        print(f"| {g} | {len(T)} | {fixed:.3f} | " + " | ".join(cells) + f" | {hf:.3f} | {ck:.3f} |")
        for k in pooled: pooled[k] += row[k] * len(T)
        n_tot += len(T)
    print("| pooled | " + str(n_tot) + " | " + " | ".join(f"{pooled[k]/n_tot:.3f}" for k in ["FIXED"] + kinds + ["HF", "CK"]) + " |")


print("# Grouped within-family decomposition and matched development screen (descriptive, 2026-09-15)\n")
decomposition(piv_all, "20 detectors"); decomposition(piv_all.drop(columns=["Confidence"]), "without Confidence")
screen(piv_all, "20 detectors"); screen(piv_all.drop(columns=["Confidence"]), "without Confidence")

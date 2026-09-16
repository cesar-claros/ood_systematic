"""Within-family new-dataset headroom and reproduction of the 2026-09-15 commitment-criteria checks (descriptive, unregistered).

(1) Exhaustive raw-confidence threshold audit of the two stored regime-free ResNet-18 policies (fallback CTM).
(2) Probe-pool decomposition with a hindsight fixed detector per encoder.
(3) NEW: VGG-13 source-held-out decomposition (the review's "new ID dataset within a represented family" target):
    L0 = one detector chosen on the other three sources; metadata = per-paradigm choice on the other sources;
    hindsight fixed, checkpoint oracle and row oracle on the held-out source. Units AUGRC x1e3; regret vs the per-row oracle.
Development data; no claim attaches. Run: ../.venv/bin/python evaluation/within_family_headroom.py > outputs/45_within_family_headroom.md
"""
import numpy as np
import pandas as pd

REG = ["near", "mid", "far"]
L = pd.read_parquet("outputs/track1/dataset/long.parquet"); L = L[L.regime.isin(REG)].copy()
L["dropout"] = L["dropout"].astype(int); L["reward"] = L["reward"].astype(float)
L["model_id"] = (L.architecture + "|" + L.paradigm + "|" + L.source + "|" + L.run.astype(int).astype(str) + "|" + L.dropout.astype(str) + "|" + L.reward.map(lambda r: "%g" % r))
piv = L.pivot_table(index=["architecture", "model_id", "eval_dataset", "regime"], columns="csf", values="augrc", aggfunc="first")


def rows(arch):
    T = piv.xs(arch, level="architecture", drop_level=False); return T[[c for c in T.columns if T[c].notna().all()]]


def ck_oracle(T, levels):
    means = T.groupby(level=levels).mean(); best = means.idxmin(axis=1)
    key = T.index.to_frame(index=False)[levels].apply(tuple, axis=1) if len(levels) > 1 else T.index.get_level_values(levels[0])
    return float(np.mean([T.iloc[i][best.loc[k]] for i, k in enumerate(key)]))


VGG = rows("VGG13"); R = rows("ResNet18"); RT = R.sub(R.min(axis=1), axis=0); tm = R.index.get_level_values("model_id")
print("# Within-family headroom and commitment-criteria checks (descriptive, 2026-09-15)\n")
print("## (1) Exhaustive raw-confidence threshold audit, ResNet-18, fallback CTM\n")
for cfg in ["none_nr_marginal", "source_nr_marginal"]:
    Pd = pd.read_parquet(f"outputs/ablations/calib_cliques_regime_free/track1/xarch/{cfg}/preds.parquet"); pc = [c for c in Pd.columns if c.startswith("p_")][0]
    Q = Pd.pivot_table(index="model_id", columns="csf", values=pc, aggfunc="first").reindex(columns=R.columns); top = Q.idxmax(axis=1); conf = Q.max(axis=1)
    r_top = np.array([RT.iloc[i][top.loc[m]] for i, m in enumerate(tm)]); r_ctm = RT["CTM"].values; cm = conf.reindex(tm).values
    best = min(((np.where(cm >= t, r_top, r_ctm).mean(), float((conf >= t).mean()), t) for t in sorted(set(conf.values))))
    print(f"- {cfg}: best policy with nonzero coverage {best[0]:.4f} at checkpoint coverage {best[1]:.3f}; all-fallback CTM {r_ctm.mean():.4f}; distinct confidences {conf.nunique()}")
print("\n## (2) Probe pool with a hindsight fixed detector per encoder\n")
P = pd.read_parquet("outputs/pool_a/long_pool_a_harmonized.parquet"); P = P[P.regime.isin(REG)]
Pp = P.pivot_table(index=["paradigm", "source", "run", "eval_dataset", "regime"], columns="csf", values="augrc", aggfunc="first"); Pp = Pp[[c for c in Pp.columns if Pp[c].notna().all()]]
enc = Pp.index.get_level_values("paradigm"); Lenc = sum(Pp[enc == e][Pp[enc == e].mean().idxmin()].sum() for e in enc.unique()) / len(Pp)
for e in enc.unique():
    Te = Pp[enc == e]; print(f"- {e}: hindsight fixed {Te.mean().idxmin()} at {Te[Te.mean().idxmin()].mean():.4f} over {len(Te)} rows")
L0, Lf, Lck, Lr = Pp["CTM"].mean(), Pp.mean().min(), ck_oracle(Pp, ["paradigm", "source", "run"]), Pp.min(axis=1).mean()
print(f"- decomposition {L0-Lr:.4f} = {L0-Lf:.4f} (CTM to pool fixed {Pp.mean().idxmin()}) + {Lf-Lenc:.4f} (pool fixed to encoder fixed) + {Lenc-Lck:.4f} (encoder fixed to checkpoint oracle) + {Lck-Lr:.4f} (checkpoint to row oracle)")
print("\n## (3) VGG-13 source-held-out: new ID dataset within a represented family\n")
print("| Held-out source | Rows | Training rule (other sources) | Regret | Metadata rule regret | Hindsight fixed | Regret | Checkpoint-oracle regret | Ceiling over training rule |")
print("|---|---:|---|---:|---:|---|---:|---:|---:|")
for src in ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]:
    mask = VGG.index.get_level_values("model_id").map(lambda x: x.split("|")[2] == src); T = VGG[mask]; TR = VGG[~mask]
    d0 = TR.mean().idxmin(); o = T.min(axis=1); L0 = T[d0].mean(); Lf = T.mean().min(); fx = T.mean().idxmin(); Lck = ck_oracle(T, ["model_id"]); Lr = o.mean()
    meta = np.mean([T.iloc[i][TR[TR.index.get_level_values("model_id").map(lambda x: x.split("|")[1] == m.split("|")[1])].mean().idxmin()] for i, m in enumerate(T.index.get_level_values("model_id"))])
    print(f"| {src} | {len(T)} | {d0} | {L0-Lr:.2f} | {meta-Lr:.2f} | {fx} | {Lf-Lr:.2f} | {Lck-Lr:.2f} | {L0-Lck:.2f} |")
print("\nCaveats: four sources of which CIFAR-100 and SuperCIFAR-100 share images; the 'Confidence' detector is each paradigm's learned readout; development data inspected many times.")

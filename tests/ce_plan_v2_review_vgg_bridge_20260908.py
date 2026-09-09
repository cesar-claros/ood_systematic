"""Consumed VGG bridge (roster-B readout, already registered 2026-09-01):
v2 section 7.1 estimator on the 20 confidnet do0 VGG13 backbones, plus the
nuisance-scale decomposition needed for the section 10 power check."""
import json, glob, re
import numpy as np
from scipy import stats

SHIFTS = ["mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new"]
SRC = {"cifar10": "cifar10", "cifar100": "cifar100",
       "supercifar": "supercifar", "tiny-imagenet-200": "tinyimagenet"}
files = sorted(glob.glob("pilot0/icml_roster_b_coords/*confidnet_bbvgg13_do0_*.json"))
d0 = json.load(open(files[0]))
ctm_key = [k for k in d0["ood"]["mnist_new"] if k.startswith("augrc_raw") and "Energy" not in k]
print("CTM augrc key(s):", ctm_key, "| geometry keys:", list(d0["geometry"]))
ck = ctm_key[0]

rows = []
for f in files:
    d = json.load(open(f))
    m = re.search(r"(.+?)_paper_sweep__confidnet_bbvgg13_do0_run(\d)_", f.split("/")[-1])
    src, run = SRC[m.group(1)], int(m.group(2))
    gaps = [d["ood"][o]["augrc_raw_Energy"] - d["ood"][o][ck] for o in SHIFTS]
    rows.append((src, run, d["geometry"]["snr"], gaps))
sources = ["cifar10", "cifar100", "supercifar", "tinyimagenet"]
S, O = 4, 4
g = np.zeros((S, 5)); D = np.zeros((S, 5, O))
for src, run, snr, gaps in rows:
    s = sources.index(src); g[s, run - 1] = -snr   # lower NC1 <=> higher snr <=> rank 1
    D[s, run - 1] = gaps
print("checkpoints loaded:", len(rows))

def T_table(g, D):
    n = g.shape[1]; T = np.empty((S, O))
    for s in range(S):
        x = (stats.rankdata(g[s]) - 0.5) / n - 0.5; xc = x - x.mean()
        for o in range(O):
            d = D[s, :, o]; T[s, o] = -0.8 * (xc * (d - d.mean())).sum() / (xc**2).sum()
    return T
T = T_table(g, D)
np.set_printoptions(precision=4, suppress=True)
print("\nT[s,o] (positive = lower NC1 favors CTM), rows =", sources, "cols =", SHIFTS)
print(T)
print("A^G_VGG =", round(T.mean(), 4), "| A_s =", T.mean(1).round(4), "| B_o =", T.mean(0).round(4))
print("gap Delta^G mean per source-shift:\n", D.mean(1).round(4))

# nuisance scales
m_sr = D.mean(2)                                   # checkpoint mean over shifts
sd_family_within_src = np.sqrt(np.mean(m_sr.var(1, ddof=1)))
resid = D - m_sr[:, :, None] - D.mean(1)[:, None, :] + D.mean((1, 2))[:, None, None]
sd_cell = np.sqrt(resid.var(ddof=1) * (5 * 4) / ((5 - 1) * (4 - 1)))
# shared-run component across sources (only meaningful if run index = shared seed)
c = m_sr - m_sr.mean(1, keepdims=True)
cross = np.mean([np.cov(c[i], c[j])[0, 1] for i in range(S) for j in range(i + 1, S)])
print(f"\nsd of checkpoint-mean gap within source (family+shared): {sd_family_within_src:.4f}")
print(f"sd of source x run x shift residual (cell):              {sd_cell:.4f}")
print(f"mean cross-source covariance of run means (shared seed?): {cross:+.6f}  "
      f"(as sd: {np.sqrt(max(cross,0)):.4f})")
print("sd across the 20 checkpoints of the raw gap, per shift:", D.transpose(2,0,1).reshape(O,-1).std(1, ddof=1).round(4))

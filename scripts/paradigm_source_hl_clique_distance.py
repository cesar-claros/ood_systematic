"""(Paradigm x source) HL-recipe centroid vs clique-pattern distances.

Companion to ``paradigm_source_nc_clique_distance.py`` but uses the
20-feature head-side recipe identified in the regression analysis
(NC-on-logits + Norm-scaled + Confidence-scaled) as the centroid feature
instead of the 8 Papyan NC metrics.

Question this answers: do head-side metric distances between (paradigm,
source) cells track clique-pattern distances - especially head-side clique
distances, where the Papyan NC centroid was uncorrelated (rho ~ 0.08, n.s.)?

Outputs in code/ood_eval_outputs/paradigm_source_hl_clique_distance/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from stats_eval import ALL_PARADIGMS_CONFIGS, _compute_members_for_config  # noqa: E402

# 20-feature recipe: NC-on-logits + Norm-scaled + Confidence-scaled.
HL_NC_LOGITS = [
    "logit_classmean_cv_norm", "logit_classmean_cos_mse",
    "logit_classmean_cos_max_dev", "logit_classmean_cos_mean",
    "logit_within_between_ratio", "logit_classmean_mean_norm",
    "logit_cov_participation", "logit_cov_effrank",
]
HL_NORM_SCALED = ["scaled_logitnorm_mean", "scaled_logitnorm_std",
                  "scaled_logitnorm_p50"]
HL_CONF_SCALED = [
    "scaled_entropy_mean", "scaled_entropy_std", "scaled_entropy_p50",
    "scaled_maxprob_mean", "scaled_maxprob_std", "scaled_maxprob_p50",
    "scaled_kl_uniform_mean", "scaled_kl_uniform_std", "scaled_kl_uniform_p50",
]
HL_RECIPE = HL_NC_LOGITS + HL_NORM_SCALED + HL_CONF_SCALED

PARADIGM_LABELS = {"confidnet": "ConfidNet", "devries": "DeVries",
                   "dg": "Deep Gamblers", "vit": "ViT"}

HEAD_SIDE_CSFS = {"REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
                  "GradNorm", "Energy", "Confidence", "pNML"}
FEATURE_SIDE_CSFS = {"PCA RecError global", "NeCo", "NNGuide", "CTM",
                     "ViM", "Maha", "fDBD", "KPCA RecError global", "Residual"}

SOURCES = ["cifar10", "supercifar100", "cifar100", "tinyimagenet"]
PARADIGMS = ["confidnet", "devries", "dg", "vit"]
OUT = CODE_DIR / "ood_eval_outputs" / "paradigm_source_hl_clique_distance"
OUT.mkdir(parents=True, exist_ok=True)


def load_hl_centroids() -> dict[tuple[str, str], np.ndarray]:
    hl = pd.read_csv(CODE_DIR / "head_logit_metrics" / "hl_metrics.csv",
                     index_col=0)
    # hl_metrics uses 'supercifar', clique panels use 'supercifar100'.
    hl = hl.rename(columns={"dataset": "source", "study": "paradigm"})
    hl["source"] = hl["source"].replace({"supercifar": "supercifar100"})
    # ResNet18 is not in clique data; drop to match.
    hl = hl[hl["architecture"] != "ResNet18"].copy()
    excl = (
        (hl["paradigm"] == "dg")
        & (hl["source"] == "supercifar100")
        & np.isclose(hl["reward"].astype(float), 2.2)
    )
    hl = hl.loc[~excl].copy()

    feats = [c for c in HL_RECIPE if c in hl.columns]
    missing = [c for c in HL_RECIPE if c not in hl.columns]
    if missing:
        print(f"WARNING: missing HL columns: {missing}")
    zs = (hl[feats] - hl[feats].mean()) / hl[feats].std(ddof=0)
    hl[feats] = zs

    centroids: dict[tuple[str, str], np.ndarray] = {}
    for p in PARADIGMS:
        for s in SOURCES:
            rows = hl[(hl["paradigm"] == p) & (hl["source"] == s)]
            if rows.empty:
                continue
            centroids[(p, s)] = rows[feats].mean().to_numpy()
    return centroids, feats


def load_clique_panels() -> dict[tuple[str, str], pd.DataFrame]:
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    for backbone, model_filter, _label in ALL_PARADIGMS_CONFIGS:
        key_p = model_filter[0]
        if key_p == "modelvit":
            key_p = "vit"
        for src in SOURCES:
            try:
                res = _compute_members_for_config(
                    backbone=backbone, model_filter=model_filter,
                    metric=["AUGRC", "AURC"], mcd_flag="False",
                    filter_methods=True, clip_dir="clip_scores",
                    alpha=0.05, sources_all=[src],
                )
                panels[(key_p, src)] = res["members_all"]
            except Exception as e:
                print(f"  skip ({key_p}, {src}): {e}")
    return panels


def flatten_panels(panels, csf_subset=None):
    union_csfs = {c for m in panels.values() for c in m.columns}
    cols = sorted(union_csfs if csf_subset is None
                  else (union_csfs & csf_subset), key=str.casefold)
    regimes = ["test", "near", "mid", "far"]
    flat = {}
    for k, m in panels.items():
        m2 = m.copy()
        m2.index = [r.split("->")[-1] for r in m2.index]
        m2 = m2.reindex(index=regimes, columns=cols, fill_value=False)
        flat[k] = m2.to_numpy().astype(bool).flatten()
    return flat


def pairwise_euclid(centroids, order):
    n = len(order); d = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            d[i, j] = float(np.linalg.norm(centroids[a] - centroids[b]))
    return pd.DataFrame(d, index=order, columns=order)


def pairwise_jaccard(flat, order):
    n = len(order); d = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = flat[a], flat[b]
            inter = int(np.logical_and(va, vb).sum())
            union = int(np.logical_or(va, vb).sum())
            d[i, j] = 0.0 if union == 0 else 1.0 - inter / union
    return pd.DataFrame(d, index=order, columns=order)


def _nice_labels(order):
    s = {"cifar10": "C10", "cifar100": "C100",
         "supercifar100": "sC100", "tinyimagenet": "TIN"}
    return [f"{PARADIGM_LABELS[p]}-{s[src]}" for p, src in order]


def _paradigm_blocks(order):
    return [i for i in range(1, len(order)) if order[i][0] != order[i-1][0]]


def _draw(d, ax, title, cmap, cbar_label, vmin=None, vmax=None):
    labels = _nice_labels(list(d.index))
    sns.heatmap(d, cmap=cmap, ax=ax,
                cbar_kws={"label": cbar_label}, square=True,
                vmin=vmin, vmax=vmax, linewidths=0.2, linecolor="white",
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    for b in _paradigm_blocks(list(d.index)):
        ax.axhline(b, color="white", linewidth=2.0)
        ax.axvline(b, color="white", linewidth=2.0)


def main():
    print("Loading HL-recipe centroids per (paradigm, source)...")
    centroids, feats = load_hl_centroids()
    print(f"  cells={len(centroids)}  features ({len(feats)}): "
          f"{feats[:3]}...{feats[-3:]}")

    print("Loading clique panels per (paradigm, source)...")
    panels = load_clique_panels()

    order = [(p, s) for p in PARADIGMS for s in SOURCES
             if (p, s) in centroids and (p, s) in panels]
    print(f"  joint cells ({len(order)}): {order}")

    flat_all = flatten_panels(panels)
    flat_head = flatten_panels(panels, HEAD_SIDE_CSFS)
    flat_feat = flatten_panels(panels, FEATURE_SIDE_CSFS)

    d_hl = pairwise_euclid(centroids, order)
    d_all = pairwise_jaccard(flat_all, order)
    d_head = pairwise_jaccard(flat_head, order)
    d_feat = pairwise_jaccard(flat_feat, order)

    str_index = [f"{p}|{s}" for p, s in order]
    for d, name in ((d_hl, "hl_centroid_distance"),
                    (d_all, "clique_jaccard_distance_all"),
                    (d_head, "clique_jaccard_distance_head"),
                    (d_feat, "clique_jaccard_distance_feature")):
        d2 = d.copy(); d2.index = d2.columns = str_index
        d2.to_csv(OUT / f"{name}.csv")

    iu = np.triu_indices(len(order), k=1)
    hl_off = d_hl.to_numpy()[iu]
    rho = {n: spearmanr(hl_off, d.to_numpy()[iu])
           for n, d in (("all", d_all), ("head", d_head), ("feature", d_feat))}
    n_pairs = len(iu[0])
    print(f"\nOff-diagonal Spearman vs HL-recipe (n={n_pairs} pairs):")
    for k, (r, p) in rho.items():
        print(f"  {k:>8}: rho={r:+.3f}, p={p:.4f}")

    # NC reference for direct comparison: re-load NC centroids on same cells.
    nc = pd.read_csv(CODE_DIR / "neural_collapse_metrics" / "nc_metrics.csv",
                     index_col=0).rename(columns={"dataset": "source",
                                                  "study": "paradigm"})
    nc["source"] = nc["source"].replace({"supercifar": "supercifar100"})
    NC_M = ["var_collapse", "equiangular_uc", "equiangular_wc",
            "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
            "max_equiangular_wc", "self_duality"]
    excl = ((nc["paradigm"] == "dg") & (nc["source"] == "supercifar100")
            & np.isclose(nc["reward"].astype(float), 2.2))
    nc = nc.loc[~excl].copy()
    nc[NC_M] = (nc[NC_M] - nc[NC_M].mean()) / nc[NC_M].std(ddof=0)
    nc_centroids = {(p, s): nc[(nc.paradigm == p) & (nc.source == s)][NC_M].mean().to_numpy()
                    for p, s in order
                    if not nc[(nc.paradigm == p) & (nc.source == s)].empty}
    if len(nc_centroids) == len(order):
        d_nc = pairwise_euclid(nc_centroids, order)
        nc_off = d_nc.to_numpy()[iu]
        print("\nReference: NC-Papyan vs same clique distances:")
        for k, d in (("all", d_all), ("head", d_head), ("feature", d_feat)):
            r, p = spearmanr(nc_off, d.to_numpy()[iu])
            print(f"  {k:>8}: rho={r:+.3f}, p={p:.4f}")

    # --- Figure 1: HL | head-Jaccard | feature-Jaccard ---
    head_n = len(set().union(*[set(m.columns) for m in panels.values()])
                  & HEAD_SIDE_CSFS)
    feat_n = len(set().union(*[set(m.columns) for m in panels.values()])
                  & FEATURE_SIDE_CSFS)
    rh, _ = rho["head"]; rf, _ = rho["feature"]; ra, _ = rho["all"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    _draw(d_hl, axes[0],
          f"HL-recipe centroid distance\n"
          f"(20 z-scored head-side coords; $\\rho$ vs all-clique = {ra:+.2f})",
          cmap="viridis", cbar_label="Euclidean on z-HL")
    _draw(d_head, axes[1],
          f"Clique Jaccard - head-side CSFs\n"
          f"(n={head_n}; $\\rho$ vs HL = {rh:+.2f})",
          cmap="magma", cbar_label="Jaccard distance", vmin=0, vmax=1)
    _draw(d_feat, axes[2],
          f"Clique Jaccard - feature-side CSFs\n"
          f"(n={feat_n}; $\\rho$ vs HL = {rf:+.2f})",
          cmap="magma", cbar_label="Jaccard distance", vmin=0, vmax=1)
    fig.suptitle("HL-recipe (NC-on-logits + Norm-scaled + Conf-scaled) vs "
                 "clique distances across (paradigm, source) cells",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "paradigm_source_distances_headfeat.pdf",
                bbox_inches="tight")
    fig.savefig(OUT / "paradigm_source_distances_headfeat.png",
                bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"\nSaved figures to {OUT}/")


if __name__ == "__main__":
    main()

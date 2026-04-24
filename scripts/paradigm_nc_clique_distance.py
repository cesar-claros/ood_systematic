"""Paradigm-level NC centroid vs clique-pattern distances.

For each of {ConfidNet, DeVries, Deep Gamblers, ViT}:
 1. NC centroid: mean of 8 z-scored Papyan metrics over that paradigm's slices
    (excluding the supercifar100 x dg @ reward 2.2 slice, matching the
    intervention test convention).
 2. Clique pattern: flattened bool over (source, regime, CSF), taken from the
    existing stats_eval.py clique pipeline with --filter-methods.

Pairwise distance matrices (Euclidean for NC centroids, Jaccard for clique
vectors) are written to CSV and plotted side-by-side. Off-diagonal Spearman
correlation is reported as a descriptive consistency check (n=6 pairs).

Outputs in code/ood_eval_outputs/paradigm_nc_clique_distance/.
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

NC_METRICS = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc",
    "max_equiangular_uc", "max_equiangular_wc", "self_duality",
]

PARADIGM_LABELS = {
    "confidnet": "ConfidNet",
    "devries":   "DeVries",
    "dg":        "Deep Gamblers",
    "vit":       "ViT",
}

# Raw column names (pre-display-rename) classified by where the score reads from.
# Head-side: operates on logits / softmax / predictive-distribution statistics.
# Feature-side: operates on penultimate features (subspaces, prototypes,
# reconstruction errors, feature-boundary distances).
HEAD_SIDE_CSFS = {
    "REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
    "GradNorm", "Energy", "Confidence", "pNML",
}
FEATURE_SIDE_CSFS = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}

SOURCES = ["cifar10", "supercifar100", "cifar100", "tinyimagenet"]
OUT = CODE_DIR / "ood_eval_outputs" / "paradigm_nc_clique_distance"
OUT.mkdir(parents=True, exist_ok=True)


def load_nc_centroids() -> dict[str, np.ndarray]:
    nc = pd.read_csv(CODE_DIR / "neural_collapse_metrics" / "nc_metrics.csv", index_col=0)
    nc = nc.rename(columns={"dataset": "source", "study": "paradigm"})
    excl = (
        (nc["paradigm"] == "dg")
        & (nc["source"] == "supercifar100")
        & np.isclose(nc["reward"].astype(float), 2.2)
    )
    nc = nc.loc[~excl].copy()

    zs = nc[NC_METRICS].copy()
    zs = (zs - zs.mean()) / zs.std(ddof=0)
    nc[NC_METRICS] = zs

    centroids: dict[str, np.ndarray] = {}
    for p in PARADIGM_LABELS:
        rows = nc[nc["paradigm"] == p]
        if rows.empty:
            continue
        centroids[p] = rows[NC_METRICS].mean().to_numpy()
    return centroids


def load_clique_panels() -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for backbone, model_filter, _label in ALL_PARADIGMS_CONFIGS:
        res = _compute_members_for_config(
            backbone=backbone,
            model_filter=model_filter,
            metric=["AUGRC", "AURC"],
            mcd_flag="False",
            filter_methods=True,
            clip_dir="clip_scores",
            alpha=0.05,
            sources_all=SOURCES,
        )
        key = model_filter[0]
        if key == "modelvit":
            key = "vit"
        panels[key] = res["members_all"]
    return panels


def flatten_panels(panels: dict[str, pd.DataFrame],
                   csf_subset: set[str] | None = None) -> dict[str, np.ndarray]:
    all_cells = sorted({i for m in panels.values() for i in m.index})
    union_csfs = {c for m in panels.values() for c in m.columns}
    if csf_subset is None:
        cols = sorted(union_csfs, key=str.casefold)
    else:
        cols = sorted(union_csfs & csf_subset, key=str.casefold)
    flat: dict[str, np.ndarray] = {}
    for p, m in panels.items():
        m2 = m.reindex(index=all_cells, columns=cols, fill_value=False)
        flat[p] = m2.to_numpy().astype(bool).flatten()
    return flat


def pairwise_euclid(centroids: dict[str, np.ndarray], order: list[str]) -> pd.DataFrame:
    n = len(order)
    d = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            d[i, j] = float(np.linalg.norm(centroids[a] - centroids[b]))
    return pd.DataFrame(d, index=order, columns=order)


def pairwise_jaccard(flat: dict[str, np.ndarray], order: list[str]) -> pd.DataFrame:
    n = len(order)
    d = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = flat[a], flat[b]
            inter = int(np.logical_and(va, vb).sum())
            union = int(np.logical_or(va, vb).sum())
            d[i, j] = 0.0 if union == 0 else 1.0 - inter / union
    return pd.DataFrame(d, index=order, columns=order)


def main() -> None:
    print("Loading NC centroids...")
    centroids = load_nc_centroids()
    print(f"  paradigms with NC data: {list(centroids.keys())}")

    print("Loading clique panels (runs the Friedman/clique pipeline per paradigm)...")
    panels = load_clique_panels()
    print(f"  paradigms with clique data: {list(panels.keys())}")

    # Sanity check: which CSFs were seen, and are all classified?
    all_csfs_seen = sorted({c for m in panels.values() for c in m.columns}, key=str.casefold)
    unclassified = [c for c in all_csfs_seen
                    if c not in HEAD_SIDE_CSFS and c not in FEATURE_SIDE_CSFS]
    if unclassified:
        print(f"WARNING: unclassified CSFs (will be dropped from subsets): {unclassified}")
    head_present = [c for c in all_csfs_seen if c in HEAD_SIDE_CSFS]
    feat_present = [c for c in all_csfs_seen if c in FEATURE_SIDE_CSFS]
    print(f"  head-side ({len(head_present)}): {head_present}")
    print(f"  feature-side ({len(feat_present)}): {feat_present}")

    flat_all = flatten_panels(panels, csf_subset=None)
    flat_head = flatten_panels(panels, csf_subset=HEAD_SIDE_CSFS)
    flat_feat = flatten_panels(panels, csf_subset=FEATURE_SIDE_CSFS)

    order = [p for p in ["confidnet", "devries", "dg", "vit"]
             if p in centroids and p in flat_all]
    labels = [PARADIGM_LABELS[p] for p in order]

    d_nc = pairwise_euclid(centroids, order)
    d_all = pairwise_jaccard(flat_all, order)
    d_head = pairwise_jaccard(flat_head, order)
    d_feat = pairwise_jaccard(flat_feat, order)
    for m in (d_nc, d_all, d_head, d_feat):
        m.index = m.columns = labels

    d_nc.to_csv(OUT / "nc_centroid_distance.csv")
    d_all.to_csv(OUT / "clique_jaccard_distance_all.csv")
    d_head.to_csv(OUT / "clique_jaccard_distance_head.csv")
    d_feat.to_csv(OUT / "clique_jaccard_distance_feature.csv")

    iu = np.triu_indices(len(order), k=1)
    nc_off = d_nc.to_numpy()[iu]
    offdiag_rho = {}
    for name, d in (("all", d_all), ("head", d_head), ("feature", d_feat)):
        offdiag_rho[name] = spearmanr(nc_off, d.to_numpy()[iu])

    print("\nNC centroid distance:\n", d_nc.round(2))
    print("\nClique Jaccard — all CSFs:\n", d_all.round(2))
    print("\nClique Jaccard — head-side only:\n", d_head.round(2))
    print("\nClique Jaccard — feature-side only:\n", d_feat.round(2))
    print("\nOff-diagonal Spearman against NC (n=6 pairs):")
    for name, (rho, p) in offdiag_rho.items():
        print(f"  {name:>8}: rho={rho:+.3f}, p={p:.3f}")

    # --- Figure 1: NC + all (kept for backward compatibility) ---
    rho_all, p_all = offdiag_rho["all"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.heatmap(d_nc, annot=True, fmt=".2f", cmap="viridis",
                ax=axes[0], cbar_kws={"label": "Euclidean on z-NC"},
                square=True, linewidths=0.4, linecolor="white")
    axes[0].set_title("NC centroid distance\n(mean of 8 z-scored Papyan coords)")
    sns.heatmap(d_all, annot=True, fmt=".2f", cmap="magma",
                ax=axes[1], cbar_kws={"label": "Jaccard distance"},
                square=True, linewidths=0.4, linecolor="white")
    axes[1].set_title("Clique pattern distance\n(bool over source x regime x CSF)")
    fig.suptitle(
        f"Paradigm-level NC vs clique distances  "
        f"(off-diagonal Spearman $\\rho$={rho_all:+.2f}, n=6 pairs)",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    fig.savefig(OUT / "paradigm_distances.pdf", bbox_inches="tight")
    fig.savefig(OUT / "paradigm_distances.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- Figure 2: NC | head Jaccard | feature Jaccard ---
    rho_head, _ = offdiag_rho["head"]
    rho_feat, _ = offdiag_rho["feature"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    sns.heatmap(d_nc, annot=True, fmt=".2f", cmap="viridis", ax=axes[0],
                cbar_kws={"label": "Euclidean on z-NC"}, square=True,
                linewidths=0.4, linecolor="white")
    axes[0].set_title("NC centroid distance\n(mean of 8 z-scored Papyan coords)")
    sns.heatmap(d_head, annot=True, fmt=".2f", cmap="magma", ax=axes[1],
                cbar_kws={"label": "Jaccard distance"}, square=True, vmin=0, vmax=1,
                linewidths=0.4, linecolor="white")
    axes[1].set_title(f"Clique Jaccard — head-side CSFs\n"
                      f"(n={len(head_present)}; $\\rho$ vs NC = {rho_head:+.2f})")
    sns.heatmap(d_feat, annot=True, fmt=".2f", cmap="magma", ax=axes[2],
                cbar_kws={"label": "Jaccard distance"}, square=True, vmin=0, vmax=1,
                linewidths=0.4, linecolor="white")
    axes[2].set_title(f"Clique Jaccard — feature-side CSFs\n"
                      f"(n={len(feat_present)}; $\\rho$ vs NC = {rho_feat:+.2f})")
    fig.suptitle("Head-side vs feature-side clique distances across paradigms",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(OUT / "paradigm_distances_headfeat.pdf", bbox_inches="tight")
    fig.savefig(OUT / "paradigm_distances_headfeat.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"\nSaved figures to {OUT}/paradigm_distances{{,_headfeat}}.pdf")


if __name__ == "__main__":
    main()

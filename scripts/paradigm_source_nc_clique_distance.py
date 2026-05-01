"""(Paradigm x source)-stratified NC vs clique-pattern distances.

Companion to ``paradigm_nc_clique_distance.py`` but computed at the
(paradigm, source) level instead of paradigm-only. With 4 paradigms x
4 sources we get up to 16 cells; pairwise NC and clique distances are
plotted as 16x16 heatmaps with paradigm-block separators.

For each (paradigm, source) cell we compute:

1. NC centroid - mean of 8 z-scored Papyan metrics across all configs in
   that cell (excluding the dg x supercifar100 x reward-2.2 slice, matching
   the intervention-test convention).
2. Clique pattern - boolean over (regime in {test,near,mid,far}) x CSF
   from running the Friedman/Conover-Holm pipeline on that single source.
   The 4 regime labels are common across sources, so cells from different
   sources are comparable on the regime axis even though the underlying
   OOD datasets differ.

Outputs in code/ood_eval_outputs/paradigm_source_nc_clique_distance/.
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

HEAD_SIDE_CSFS = {
    "REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
    "GradNorm", "Energy", "Confidence", "pNML",
}
FEATURE_SIDE_CSFS = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}

SOURCES = ["cifar10", "supercifar100", "cifar100", "tinyimagenet"]
PARADIGMS = ["confidnet", "devries", "dg", "vit"]
OUT = CODE_DIR / "ood_eval_outputs" / "paradigm_source_nc_clique_distance"
OUT.mkdir(parents=True, exist_ok=True)


def load_nc_centroids() -> dict[tuple[str, str], np.ndarray]:
    nc = pd.read_csv(CODE_DIR / "neural_collapse_metrics" / "nc_metrics.csv",
                     index_col=0)
    nc = nc.rename(columns={"dataset": "source", "study": "paradigm"})
    # nc_metrics.csv uses "supercifar"; clique panels use "supercifar100".
    nc["source"] = nc["source"].replace({"supercifar": "supercifar100"})
    excl = (
        (nc["paradigm"] == "dg")
        & (nc["source"] == "supercifar100")
        & np.isclose(nc["reward"].astype(float), 2.2)
    )
    nc = nc.loc[~excl].copy()

    zs = (nc[NC_METRICS] - nc[NC_METRICS].mean()) / nc[NC_METRICS].std(ddof=0)
    nc[NC_METRICS] = zs

    centroids: dict[tuple[str, str], np.ndarray] = {}
    for p in PARADIGMS:
        for s in SOURCES:
            rows = nc[(nc["paradigm"] == p) & (nc["source"] == s)]
            if rows.empty:
                continue
            centroids[(p, s)] = rows[NC_METRICS].mean().to_numpy()
    return centroids


def load_clique_panels() -> dict[tuple[str, str], pd.DataFrame]:
    panels: dict[tuple[str, str], pd.DataFrame] = {}
    for backbone, model_filter, _label in ALL_PARADIGMS_CONFIGS:
        key_p = model_filter[0]
        if key_p == "modelvit":
            key_p = "vit"
        for src in SOURCES:
            try:
                res = _compute_members_for_config(
                    backbone=backbone,
                    model_filter=model_filter,
                    metric=["AUGRC", "AURC"],
                    mcd_flag="False",
                    filter_methods=True,
                    clip_dir="clip_scores",
                    alpha=0.05,
                    sources_all=[src],
                )
                panels[(key_p, src)] = res["members_all"]
            except Exception as e:
                print(f"  skip ({key_p}, {src}): {e}")
    return panels


def flatten_panels(panels: dict[tuple[str, str], pd.DataFrame],
                   csf_subset: set[str] | None = None
                   ) -> dict[tuple[str, str], np.ndarray]:
    """Flatten each panel to a boolean vector aligned on (regime, CSF).

    Each panel's index is "{src}->{regime}" - we strip the source prefix so
    cells from different sources share the same row ordering (4 regimes).
    """
    union_csfs = {c for m in panels.values() for c in m.columns}
    if csf_subset is None:
        cols = sorted(union_csfs, key=str.casefold)
    else:
        cols = sorted(union_csfs & csf_subset, key=str.casefold)
    regimes = ["test", "near", "mid", "far"]

    flat: dict[tuple[str, str], np.ndarray] = {}
    for key, m in panels.items():
        m2 = m.copy()
        m2.index = [r.split("->")[-1] for r in m2.index]
        m2 = m2.reindex(index=regimes, columns=cols, fill_value=False)
        flat[key] = m2.to_numpy().astype(bool).flatten()
    return flat


def pairwise_euclid(centroids: dict[tuple[str, str], np.ndarray],
                    order: list[tuple[str, str]]) -> pd.DataFrame:
    n = len(order)
    d = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            d[i, j] = float(np.linalg.norm(centroids[a] - centroids[b]))
    return pd.DataFrame(d, index=order, columns=order)


def pairwise_jaccard(flat: dict[tuple[str, str], np.ndarray],
                     order: list[tuple[str, str]]) -> pd.DataFrame:
    n = len(order)
    d = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            va, vb = flat[a], flat[b]
            inter = int(np.logical_and(va, vb).sum())
            union = int(np.logical_or(va, vb).sum())
            d[i, j] = 0.0 if union == 0 else 1.0 - inter / union
    return pd.DataFrame(d, index=order, columns=order)


def _nice_labels(order: list[tuple[str, str]]) -> list[str]:
    short_src = {"cifar10": "C10", "cifar100": "C100",
                 "supercifar100": "sC100", "tinyimagenet": "TIN"}
    return [f"{PARADIGM_LABELS[p]}-{short_src[s]}" for p, s in order]


def _paradigm_blocks(order: list[tuple[str, str]]) -> list[int]:
    """Indices where the paradigm changes (for separator lines)."""
    breaks = []
    for i in range(1, len(order)):
        if order[i][0] != order[i - 1][0]:
            breaks.append(i)
    return breaks


def _draw_heatmap(d: pd.DataFrame, ax, title: str, cmap: str,
                  cbar_label: str, vmin=None, vmax=None) -> None:
    labels = _nice_labels(list(d.index))
    sns.heatmap(d, annot=False, cmap=cmap, ax=ax,
                cbar_kws={"label": cbar_label}, square=True,
                vmin=vmin, vmax=vmax,
                linewidths=0.2, linecolor="white",
                xticklabels=labels, yticklabels=labels)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    for b in _paradigm_blocks(list(d.index)):
        ax.axhline(b, color="white", linewidth=2.0)
        ax.axvline(b, color="white", linewidth=2.0)


def main() -> None:
    print("Loading NC centroids per (paradigm, source)...")
    centroids = load_nc_centroids()
    print(f"  cells: {sorted(centroids.keys())}")

    print("Loading clique panels per (paradigm, source)...")
    panels = load_clique_panels()
    print(f"  cells: {sorted(panels.keys())}")

    # Order: paradigm-major, source-minor; only keep cells present in BOTH
    order: list[tuple[str, str]] = [
        (p, s) for p in PARADIGMS for s in SOURCES
        if (p, s) in centroids and (p, s) in panels
    ]
    print(f"  joint cells ({len(order)}): {order}")

    flat_all = flatten_panels(panels, csf_subset=None)
    flat_head = flatten_panels(panels, csf_subset=HEAD_SIDE_CSFS)
    flat_feat = flatten_panels(panels, csf_subset=FEATURE_SIDE_CSFS)

    d_nc = pairwise_euclid(centroids, order)
    d_all = pairwise_jaccard(flat_all, order)
    d_head = pairwise_jaccard(flat_head, order)
    d_feat = pairwise_jaccard(flat_feat, order)

    # Save raw matrices with composite keys.
    str_index = [f"{p}|{s}" for p, s in order]
    for d, name in ((d_nc, "nc_centroid_distance"),
                    (d_all, "clique_jaccard_distance_all"),
                    (d_head, "clique_jaccard_distance_head"),
                    (d_feat, "clique_jaccard_distance_feature")):
        d2 = d.copy()
        d2.index = d2.columns = str_index
        d2.to_csv(OUT / f"{name}.csv")

    # Off-diagonal Spearman.
    iu = np.triu_indices(len(order), k=1)
    nc_off = d_nc.to_numpy()[iu]
    rho = {}
    for name, d in (("all", d_all), ("head", d_head), ("feature", d_feat)):
        rho[name] = spearmanr(nc_off, d.to_numpy()[iu])
    n_pairs = len(iu[0])
    print(f"\nOff-diagonal Spearman vs NC (n={n_pairs} pairs):")
    for name, (r, p) in rho.items():
        print(f"  {name:>8}: rho={r:+.3f}, p={p:.4f}")

    # --- Figure 1: NC | all CSFs ---
    rho_all, p_all = rho["all"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _draw_heatmap(d_nc, axes[0],
                  "NC centroid distance\n(8 z-scored Papyan coords)",
                  cmap="viridis", cbar_label="Euclidean on z-NC")
    _draw_heatmap(d_all, axes[1],
                  "Clique pattern distance\n(bool over regime x CSF)",
                  cmap="magma", cbar_label="Jaccard distance",
                  vmin=0, vmax=1)
    fig.suptitle(
        f"(Paradigm x source) NC vs clique distances  "
        f"(off-diagonal Spearman $\\rho$={rho_all:+.2f}, n={n_pairs} pairs)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT / "paradigm_source_distances.pdf", bbox_inches="tight")
    fig.savefig(OUT / "paradigm_source_distances.png", bbox_inches="tight",
                dpi=150)
    plt.close(fig)

    # --- Figure 2: NC | head Jaccard | feature Jaccard ---
    head_present = sorted({c for m in panels.values() for c in m.columns
                           if c in HEAD_SIDE_CSFS}, key=str.casefold)
    feat_present = sorted({c for m in panels.values() for c in m.columns
                           if c in FEATURE_SIDE_CSFS}, key=str.casefold)
    rho_head, _ = rho["head"]
    rho_feat, _ = rho["feature"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    _draw_heatmap(d_nc, axes[0],
                  "NC centroid distance",
                  cmap="viridis", cbar_label="Euclidean on z-NC")
    _draw_heatmap(d_head, axes[1],
                  f"Clique Jaccard - head-side CSFs\n"
                  f"(n={len(head_present)}; $\\rho$ vs NC = {rho_head:+.2f})",
                  cmap="magma", cbar_label="Jaccard distance",
                  vmin=0, vmax=1)
    _draw_heatmap(d_feat, axes[2],
                  f"Clique Jaccard - feature-side CSFs\n"
                  f"(n={len(feat_present)}; $\\rho$ vs NC = {rho_feat:+.2f})",
                  cmap="magma", cbar_label="Jaccard distance",
                  vmin=0, vmax=1)
    fig.suptitle("Head-side vs feature-side clique distances "
                 "across (paradigm, source) cells",
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

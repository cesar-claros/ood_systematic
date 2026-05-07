"""Coefficient heatmaps for the clique-rule (b+c) calibrated predictor.

Generates one heatmap per (split, config) for the headline predictor:
  - L2 LogisticRegressionCV(Cs=50, cv=5, class_weight='balanced')
  - NC features pre-standardized per architecture (passthrough)
  - Clique label rule (per-(paradigm, source, dropout, reward, regime)
    Friedman-Conover top cliques)

Configs:
  source     — NC + source one-hot + regime one-hot
  n_classes  — NC + n_classes (scaled) + regime one-hot
  none       — NC + regime one-hot only

Splits: xarch + lopo (the two transfer headlines).

Outputs (per split × config):
  outputs/figures/clique_coefficients_heatmap_<split>_<config>.{pdf,png}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]
HEAD_SIDE_CSFS = [
    "Confidence", "Energy", "GE", "GEN", "GradNorm",
    "MLS", "MSR", "PCE", "PE", "REN", "pNML",
]
FEATURE_SIDE_CSFS = [
    "CTM", "fDBD", "KPCA RecError global", "Maha", "NeCo",
    "NNGuide", "PCA RecError global", "Residual", "ViM",
]
SPLITS = ["xarch", "lopo"]
CONFIGS = ["source", "n_classes", "none"]


def order_features(present: list[str]) -> list[str]:
    nc_order = [f for f in NC_PRIMARY if f in present]
    nclass = [f for f in present if f == "n_classes"]
    src = sorted([f for f in present if f.startswith("source_")])
    reg = sorted([f for f in present if f.startswith("regime_")])
    rest = [f for f in present
            if f not in nc_order and f not in nclass
            and not f.startswith(("source_", "regime_"))]
    return nc_order + nclass + src + reg + rest


def order_csfs(present: set[str]) -> list[str]:
    feat = [c for c in FEATURE_SIDE_CSFS if c in present]
    head = [c for c in HEAD_SIDE_CSFS if c in present]
    return feat + head


def plot_heatmap(coefs: pd.DataFrame, title: str, out_base: Path) -> None:
    coefs = coefs[coefs["feature"] != "(intercept)"].copy()
    avg = coefs.groupby(["csf", "feature"])["coefficient"].mean().reset_index()
    pivot = avg.pivot(index="feature", columns="csf", values="coefficient")
    feature_order = order_features(list(pivot.index))
    csf_order = order_csfs(set(pivot.columns))
    pivot = pivot.reindex(index=feature_order, columns=csf_order)
    n_features = len(feature_order)
    n_csfs = len(csf_order)
    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 1.0
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(max(n_csfs * 0.7 + 3, 12),
                                     max(n_features * 0.5 + 3, 7)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(n_csfs))
    ax.set_xticklabels(csf_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_order, fontsize=9)
    for i in range(n_features):
        for j in range(n_csfs):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            colour = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=colour)

    sep_after_nc = sum(1 for f in feature_order if f in NC_PRIMARY) - 0.5
    sep_after_nclass = sep_after_nc + sum(1 for f in feature_order if f == "n_classes")
    sep_after_src = sep_after_nclass + sum(1 for f in feature_order if f.startswith("source_"))
    if 0 < sep_after_nc < n_features - 1:
        ax.axhline(sep_after_nc, color="black", lw=0.8, alpha=0.7)
    if sep_after_nc < sep_after_nclass < n_features - 1:
        ax.axhline(sep_after_nclass, color="black", lw=0.8, alpha=0.7)
    if sep_after_nclass < sep_after_src < n_features - 1:
        ax.axhline(sep_after_src, color="black", lw=0.8, alpha=0.7)
    n_feat_csfs = sum(1 for c in csf_order if c in FEATURE_SIDE_CSFS)
    if 0 < n_feat_csfs < n_csfs:
        ax.axvline(n_feat_csfs - 0.5, color="black", lw=0.8, alpha=0.7)

    fig.text(0.5 * (n_feat_csfs - 0.5) / (n_csfs - 1) * 0.78 + 0.11,
             0.92, "feature-side CSFs",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)
    fig.text((n_feat_csfs + (n_csfs - 1 - n_feat_csfs) * 0.5) / (n_csfs - 1) * 0.78 + 0.11,
             0.92, "head-side CSFs",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("L2 logistic coefficient (Cs=50, balanced, per-arch std)\n"
                   "(NC features on per-arch z-score scale; "
                   "categoricals on standardized scale)", fontsize=8)
    ax.set_xlabel("CSF (output dimension)", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    fig.suptitle(title, fontsize=11, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for config in CONFIGS:
            cp = (out_root / "ablations" / "calib_cliques" / "track1"
                  / split / config / "coefficients.parquet")
            if not cp.exists():
                print(f"  missing {cp}")
                continue
            coefs = pq.read_table(cp).to_pandas()
            split_label = ("xarch — VGG13 → ResNet18, 1 fold" if split == "xarch"
                           else f"{split} — averaged across folds")
            config_label = {
                "source": "NC + source one-hot + regime",
                "n_classes": "NC + n_classes + regime",
                "none": "NC + regime only",
            }[config]
            title = ("Per-CSF logistic coefficients (clique label rule, "
                     f"calibrated +b+c)\n{config_label}; {split_label}")
            base = fig_dir / f"clique_coefficients_heatmap_{split}_{config}"
            plot_heatmap(coefs, title=title, out_base=base)
            print(f"  wrote {base}.pdf")


if __name__ == "__main__":
    main()

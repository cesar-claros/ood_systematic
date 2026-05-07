"""Coefficient heatmap for the headline classifier.

Loads per-CSF binary logistic regression coefficients (within_eps_rank label
rule, with-source one-hot encoding — the headline predictor) from step 12
and produces a heatmap with:
  - x-axis: CSFs grouped by side (feature-side then head-side, alphabetical
    within each side)
  - y-axis: features grouped (8 NC metrics, then source one-hot levels, then
    regime one-hot levels)
  - cell color: standardized-feature-scale coefficient (diverging colormap
    centered at zero — blue = positive, red = negative)
  - cell text: rounded coefficient value

Outputs:
  outputs/figures/coefficients_heatmap_xarch.{pdf,png}
  outputs/figures/coefficients_heatmap_lopo.{pdf,png}  (averaged across folds)
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


def load_coefficients(out_root: Path, split: str) -> pd.DataFrame:
    p = (out_root / "track1" / split / "per_csf_binary"
         / "within_eps_rank" / "coefficients.parquet")
    if not p.exists():
        raise FileNotFoundError(p)
    coefs = pq.read_table(p).to_pandas()
    # Drop the intercept; keep one row per (csf, feature, fold_id)
    coefs = coefs[coefs["feature"] != "(intercept)"].copy()
    return coefs


def aggregate_across_folds(coefs: pd.DataFrame) -> pd.DataFrame:
    """Average coefficients across folds for the same (csf, feature)."""
    return (coefs.groupby(["csf", "feature"])["coefficient"]
            .mean().reset_index())


def order_features(present_features: list[str]) -> list[str]:
    """Group: NC first (in NC_PRIMARY order), then source one-hots, then regime."""
    nc_order = [f for f in NC_PRIMARY if f in present_features]
    src = sorted([f for f in present_features if f.startswith("source_")])
    reg = sorted([f for f in present_features if f.startswith("regime_")])
    rest = [f for f in present_features
            if f not in nc_order and not f.startswith(("source_", "regime_"))]
    return nc_order + src + reg + rest


def order_csfs(present_csfs: set[str]) -> list[str]:
    """Feature-side first (alphabetical), then head-side (alphabetical)."""
    feat = [c for c in FEATURE_SIDE_CSFS if c in present_csfs]
    head = [c for c in HEAD_SIDE_CSFS if c in present_csfs]
    return feat + head


def plot_heatmap(coefs: pd.DataFrame, title: str, out_base: Path) -> None:
    pivot = coefs.pivot(index="feature", columns="csf", values="coefficient")
    feature_order = order_features(list(pivot.index))
    csf_order = order_csfs(set(pivot.columns))
    pivot = pivot.reindex(index=feature_order, columns=csf_order)

    n_features = len(feature_order)
    n_csfs = len(csf_order)

    # Diverging colormap centered at 0
    vmax = np.nanmax(np.abs(pivot.values))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(max(n_csfs * 0.7 + 3, 12),
                                     max(n_features * 0.5 + 3, 7)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   aspect="auto")

    ax.set_xticks(range(n_csfs))
    ax.set_xticklabels(csf_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_order, fontsize=9)

    # Cell text values
    for i in range(n_features):
        for j in range(n_csfs):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            colour = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=colour)

    # Faint group separators on y-axis
    sep_after_nc = sum(1 for f in feature_order if f in NC_PRIMARY) - 0.5
    sep_after_src = sep_after_nc + sum(1 for f in feature_order if f.startswith("source_"))
    if 0 < sep_after_nc < n_features - 1:
        ax.axhline(sep_after_nc, color="black", lw=0.8, alpha=0.7)
    if sep_after_nc < sep_after_src < n_features - 1:
        ax.axhline(sep_after_src, color="black", lw=0.8, alpha=0.7)

    # Group separator on x-axis between feature- and head-side CSFs
    n_feature_csfs = sum(1 for c in csf_order if c in FEATURE_SIDE_CSFS)
    if 0 < n_feature_csfs < n_csfs:
        ax.axvline(n_feature_csfs - 0.5, color="black", lw=0.8, alpha=0.7)

    # Move side labels OUTSIDE the plot area (above the title-free top edge)
    ax.set_xlim(-0.5, n_csfs - 0.5)
    ax.set_ylim(n_features - 0.5, -0.5)
    # Use figure-relative coords for labels above the heatmap
    fig.text(0.5 * (0 + n_feature_csfs - 0.5) / (n_csfs - 1) * 0.78 + 0.11,
             0.92, "feature-side CSFs",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)
    fig.text((n_feature_csfs + (n_csfs - 1 - n_feature_csfs) * 0.5) / (n_csfs - 1) * 0.78 + 0.11,
             0.92, "head-side CSFs",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Logistic regression coefficient\n(standardized-feature scale)",
                   fontsize=9)

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

    # xarch: 1 fold
    print("xarch ...")
    coefs = load_coefficients(out_root, "xarch")
    coefs_agg = aggregate_across_folds(coefs)
    plot_heatmap(coefs_agg,
                 title=("Per-CSF binary logistic regression coefficients\n"
                        "(headline predictor: per_csf_binary, label_rule = within_eps_rank, "
                        "split = xarch — VGG13 → ResNet18, 1 fold)"),
                 out_base=fig_dir / "coefficients_heatmap_xarch")
    print(f"  wrote {fig_dir / 'coefficients_heatmap_xarch.pdf'}")

    # lopo: 4 folds, average
    print("lopo (averaged across 4 folds) ...")
    coefs = load_coefficients(out_root, "lopo")
    coefs_agg = aggregate_across_folds(coefs)
    plot_heatmap(coefs_agg,
                 title=("Per-CSF binary logistic regression coefficients\n"
                        "(headline predictor: per_csf_binary, label_rule = within_eps_rank, "
                        "split = lopo — averaged across 4 folds)"),
                 out_base=fig_dir / "coefficients_heatmap_lopo")
    print(f"  wrote {fig_dir / 'coefficients_heatmap_lopo.pdf'}")


if __name__ == "__main__":
    main()

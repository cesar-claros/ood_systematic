"""Side-by-side Conv vs ViT top-clique heatmap.

Rows   : (source, regime) cells, grouped by regime (test, near, mid, far, all).
Columns: the 20 base CSFs (config-locked).
Cell   : color-coded membership
         - shared         : both architectures include the CSF in the top clique
         - Conv-only
         - ViT-only
         - (blank)        : neither

Outputs:
- conv_vs_vit_heatmap.pdf / .png
- conv_vs_vit_summary.json  (aggregate overlap stats)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
VIT_JSON = REPO / "ood_eval_outputs" / "vit_cliques" / "top_cliques_ViT_False_RC_cliques.json"
CONV_JSON = REPO / "ood_eval_outputs" / "conv_cliques_pooled" / "top_cliques_Conv_False_RC_cliques.json"
CONFIG_PATH = REPO / "configs" / "intervention_config.yaml"
OUT_DIR = REPO / "ood_eval_outputs" / "vit_cliques"

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
REGIMES = ["test", "near", "mid", "far", "all"]

COLOR_SHARED = "#2ca02c"    # green
COLOR_CONV = "#1f77b4"      # blue
COLOR_VIT = "#ff7f0e"       # orange
COLOR_NONE = "#f2f2f2"      # light grey


def load_csfs() -> list[str]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return sorted(cfg["csfs"], key=str.casefold)


def build_membership_grid(
    conv: dict, vit: dict, csfs: list[str]
) -> tuple[np.ndarray, list[str]]:
    """0 = none, 1 = conv-only, 2 = vit-only, 3 = shared."""
    rows = [f"{src}→{reg}" for reg in REGIMES for src in SOURCES]
    grid = np.zeros((len(rows), len(csfs)), dtype=int)
    for i, reg in enumerate(REGIMES):
        for j, src in enumerate(SOURCES):
            row = i * len(SOURCES) + j
            conv_members = set(conv.get(src, {}).get(reg, []))
            vit_members = set(vit.get(src, {}).get(reg, []))
            for k, csf in enumerate(csfs):
                in_conv = csf in conv_members
                in_vit = csf in vit_members
                if in_conv and in_vit:
                    grid[row, k] = 3
                elif in_conv:
                    grid[row, k] = 1
                elif in_vit:
                    grid[row, k] = 2
    return grid, rows


def main() -> None:
    conv = json.loads(CONV_JSON.read_text())
    vit = json.loads(VIT_JSON.read_text())
    csfs = load_csfs()

    grid, row_labels = build_membership_grid(conv, vit, csfs)

    cmap = {0: COLOR_NONE, 1: COLOR_CONV, 2: COLOR_VIT, 3: COLOR_SHARED}

    n_rows, n_cols = grid.shape
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * n_cols + 2), max(5, 0.35 * n_rows + 2)))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()

    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(
                mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=cmap[grid[i, j]],
                    edgecolor="white", linewidth=0.5,
                )
            )
    # Regime separator lines
    for reg_idx in range(1, len(REGIMES)):
        y = reg_idx * len(SOURCES) - 0.5
        ax.axhline(y, color="black", linewidth=0.8)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(csfs, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("CSF")
    ax.set_title("Top-clique membership: Conv (VGG-13) vs ViT")

    handles = [
        mpatches.Patch(color=COLOR_SHARED, label="shared (Conv ∩ ViT)"),
        mpatches.Patch(color=COLOR_CONV, label="Conv-only"),
        mpatches.Patch(color=COLOR_VIT, label="ViT-only"),
        mpatches.Patch(color=COLOR_NONE, label="neither"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=4, frameon=False, fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "conv_vs_vit_heatmap"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", bbox_inches="tight", dpi=150)

    # Aggregate summary
    n_shared = int((grid == 3).sum())
    n_conv = int((grid == 1).sum())
    n_vit = int((grid == 2).sum())
    cells_with_conv = {
        src: [reg for reg in REGIMES if conv.get(src, {}).get(reg)]
        for src in SOURCES
    }
    cells_with_vit = {
        src: [reg for reg in REGIMES if vit.get(src, {}).get(reg)]
        for src in SOURCES
    }
    summary = {
        "grid_shape": list(grid.shape),
        "membership_counts": {
            "shared": n_shared,
            "conv_only": n_conv,
            "vit_only": n_vit,
        },
        "csfs_shared_anywhere": sorted({
            csfs[j] for i in range(n_rows) for j in range(n_cols) if grid[i, j] == 3
        }),
        "csfs_conv_only_anywhere": sorted({
            csfs[j] for i in range(n_rows) for j in range(n_cols) if grid[i, j] == 1
        }),
        "csfs_vit_only_anywhere": sorted({
            csfs[j] for i in range(n_rows) for j in range(n_cols) if grid[i, j] == 2
        }),
        "cells_with_conv": cells_with_conv,
        "cells_with_vit": cells_with_vit,
    }
    (OUT_DIR / "conv_vs_vit_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {out}.pdf and .png")
    print(f"shared cells (both architectures pick same CSF): {n_shared}")
    print(f"Conv-only memberships : {n_conv}")
    print(f"ViT-only memberships  : {n_vit}")
    print(f"shared CSFs anywhere : {summary['csfs_shared_anywhere']}")
    print(f"Conv-only CSFs        : {summary['csfs_conv_only_anywhere']}")
    print(f"ViT-only CSFs         : {summary['csfs_vit_only_anywhere']}")


if __name__ == "__main__":
    main()

"""Side-stratified regret figure for the clique (b+c) headline predictor.

Generates two figures (xarch + lopo). Each figure has 3 side panels (all,
feature, head). Per panel: 3 regime groups (near, mid, far). Per regime
group: 4 bars (NC source / NC n_classes / NC none / best baseline) with
bootstrap 95% CI error bars.

Inputs:
  outputs/clique_bc/track1/<split>/<config>/metrics/aggregate.parquet
  outputs/track1/<split>/baselines/aggregate.parquet  (for baseline numbers)

Outputs:
  outputs/figures/regret_by_side_clique_bc_xarch.{pdf,png}
  outputs/figures/regret_by_side_clique_bc_lopo.{pdf,png}
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

SIDES = ["all", "feature", "head"]
REGIMES = ["near", "mid", "far"]
CONFIGS = ["source", "n_classes", "none"]
SPLITS = ["xarch", "lopo"]
CONFIG_LABELS = {"source": "NC + source", "n_classes": "NC + n_classes",
                 "none": "NC alone"}
CONFIG_COLORS = {"source": "#2ca02c", "n_classes": "#1f77b4",
                 "none": "#9467bd"}
BASELINE_COLOR = "#d62728"


def load_predictor_aggregate(out_root: Path, split: str, config: str
                             ) -> pd.DataFrame | None:
    p = (out_root / "clique_bc" / "track1" / split / config
         / "metrics" / "aggregate.parquet")
    if not p.exists():
        return None
    return pq.read_table(p).to_pandas()


def best_baseline(out_root: Path, split: str, regime: str, side: str
                  ) -> tuple[str, float, float, float] | None:
    p = (out_root / "track1" / split / "baselines" / "aggregate.parquet")
    if not p.exists():
        return None
    bl = pq.read_table(p).to_pandas()
    bl = bl[(bl["comparator_kind"] == "baseline")
            & (bl["regime"] == regime) & (bl["side"] == side)]
    if bl.empty:
        return None
    row = bl.loc[bl["regret_raw_mean"].idxmin()]
    return (str(row["comparator_name"]), float(row["regret_raw_mean"]),
            float(row["regret_raw_ci_lo"]), float(row["regret_raw_ci_hi"]))


def plot_split(out_root: Path, split: str, fig_dir: Path) -> None:
    aggs = {c: load_predictor_aggregate(out_root, split, c) for c in CONFIGS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    bar_w = 0.18
    x_centers = np.arange(len(REGIMES))
    config_offsets = {
        "source":    -1.5 * bar_w,
        "n_classes": -0.5 * bar_w,
        "none":       0.5 * bar_w,
        "baseline":   1.5 * bar_w,
    }

    for ax, side in zip(axes, SIDES):
        # NC predictor bars per config
        for cfg in CONFIGS:
            agg = aggs.get(cfg)
            if agg is None:
                continue
            means, lo, hi = [], [], []
            for regime in REGIMES:
                sel = agg[(agg["regime"] == regime) & (agg["side"] == side)]
                if sel.empty:
                    means.append(np.nan); lo.append(0); hi.append(0)
                    continue
                m = float(sel["set_regret_imputed_mean"].iloc[0])
                l = float(sel["set_regret_imputed_ci_lo"].iloc[0])
                h = float(sel["set_regret_imputed_ci_hi"].iloc[0])
                means.append(m); lo.append(m - l); hi.append(h - m)
            ax.bar(x_centers + config_offsets[cfg], means, bar_w,
                   yerr=[lo, hi], capsize=3,
                   label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg],
                   edgecolor="black", linewidth=0.5)

        # Baseline bar (best per cell)
        bl_means, bl_lo, bl_hi, bl_names = [], [], [], []
        for regime in REGIMES:
            res = best_baseline(out_root, split, regime, side)
            if res is None:
                bl_means.append(np.nan); bl_lo.append(0); bl_hi.append(0); bl_names.append("")
                continue
            name, m, l, h = res
            bl_means.append(m); bl_lo.append(m - l); bl_hi.append(h - m)
            bl_names.append(name.replace("Always-", "A-").replace("Oracle-on-train", "OoT"))
        ax.bar(x_centers + config_offsets["baseline"], bl_means, bar_w,
               yerr=[bl_lo, bl_hi], capsize=3,
               label="best baseline", color=BASELINE_COLOR,
               edgecolor="black", linewidth=0.5)
        # Annotate baseline bar with name
        for i, name in enumerate(bl_names):
            if name and not np.isnan(bl_means[i]):
                ax.text(x_centers[i] + config_offsets["baseline"],
                        bl_means[i] + bl_hi[i] + 0.02 * max(ax.get_ylim()[1], 1),
                        name, ha="center", va="bottom", fontsize=6, rotation=0)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(REGIMES)
        ax.set_xlabel("OOD regime")
        if side == "all":
            ax.set_ylabel("Mean set-regret (raw AUGRC, imputed)\n95% bootstrap CI")
        ax.set_title(f"side = {side}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    split_label = ("xarch — VGG13 → ResNet18 (cross-architecture)"
                   if split == "xarch"
                   else "lopo — leave-one-paradigm-out (cross-paradigm)")
    fig.suptitle(f"Clique (b+c) headline predictor vs baselines — {split_label}\n"
                 "Per-CSF binary, L2 Cs=50, class_weighted, per-arch NC std, "
                 "clique label rule",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    base = fig_dir / f"regret_by_side_clique_bc_{split}"
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {base}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        plot_split(out_root, split, fig_dir)


if __name__ == "__main__":
    main()

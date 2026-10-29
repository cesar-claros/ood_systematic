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

With --regime-free: xarch only, bars for the regime-free NC-only and
NC + source arms (outputs/regimefree_wilcoxon/xarch_aggregate.csv) and the
best baseline, no figure title:
  outputs/figures/regret_by_side_regime_free_xarch.{pdf,png}
With --regime-free --roster22: the same for the 22-CSF heads and baselines
(outputs/newcsf_heads/{aggregate,baseline_aggregate}.csv):
  outputs/figures/regret_by_side_22csf_xarch.{pdf,png}
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
REGIME_FREE_ARMS = ["none_nr_marginal", "source_nr_marginal"]
REGIME_FREE_LABELS = {"none_nr_marginal": "NC only",
                      "source_nr_marginal": "NC + source"}
REGIME_FREE_COLORS = {"none_nr_marginal": CONFIG_COLORS["none"],
                      "source_nr_marginal": CONFIG_COLORS["source"]}


def load_predictor_aggregate(out_root: Path, split: str, config: str
                             ) -> pd.DataFrame | None:
    p = (out_root / "clique_bc" / "track1" / split / config
         / "metrics" / "aggregate.parquet")
    if not p.exists():
        return None
    return pq.read_table(p).to_pandas()


def load_baselines(out_root: Path, split: str) -> pd.DataFrame | None:
    """Baseline aggregate written by baselines.py for one split (20-CSF roster)."""
    p = (out_root / "track1" / split / "baselines" / "aggregate.parquet")
    return pq.read_table(p).to_pandas() if p.exists() else None


def best_baseline(bl: pd.DataFrame | None, regime: str, side: str
                  ) -> tuple[str, float, float, float] | None:
    if bl is None:
        return None
    bl = bl[(bl["comparator_kind"] == "baseline")
            & (bl["regime"] == regime) & (bl["side"] == side)]
    if bl.empty:
        return None
    row = bl.loc[bl["regret_raw_mean"].idxmin()]
    return (str(row["comparator_name"]), float(row["regret_raw_mean"]),
            float(row["regret_raw_ci_lo"]), float(row["regret_raw_ci_hi"]))


def load_regime_free_aggregates(path: Path) -> dict[str, pd.DataFrame]:
    """Per-arm aggregates (regimefree_wilcoxon.py or newcsf_heads.py layout)."""
    agg = pd.read_csv(path)
    return {arm: agg[agg["arm"] == arm] for arm in REGIME_FREE_ARMS}


def plot_split(baselines: pd.DataFrame | None, aggs: dict[str, pd.DataFrame | None],
               labels: dict[str, str], colors: dict[str, str], out_base: Path,
               suptitle: str | None) -> None:
    """Three side panels of mean set-regret bars: predictor arms + best baseline.

    Args:
        baselines: Baseline aggregate (baselines.py layout) for the best-baseline
            bar of each cell.
        aggs: Arm name -> per-(regime, side) aggregate in the clique_bc layout.
        labels: Arm name -> legend label.
        colors: Arm name -> bar colour.
        out_base: Output path without extension (.pdf and .png are written).
        suptitle: Figure title, or None for no title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    n_bars = len(aggs) + 1
    bar_w = 0.72 / n_bars
    x_centers = np.arange(len(REGIMES))
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_w
    config_offsets = dict(zip([*aggs, "baseline"], offsets))

    for ax, side in zip(axes, SIDES):
        for cfg in aggs:
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
                   label=labels[cfg], color=colors[cfg],
                   edgecolor="black", linewidth=0.5)

        # Baseline bar (best per cell)
        bl_means, bl_lo, bl_hi, bl_names = [], [], [], []
        for regime in REGIMES:
            res = best_baseline(baselines, regime, side)
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

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_base}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--regime-free", action="store_true",
                        help="plot the regime-free arms (xarch only, no title)")
    parser.add_argument("--roster22", action="store_true",
                        help="with --regime-free: the 22-CSF heads and baselines "
                             "of evaluation/newcsf_heads.py")
    args = parser.parse_args()
    out_root = Path(args.out_root)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if args.regime_free:
        if args.roster22:
            aggs = load_regime_free_aggregates(out_root / "newcsf_heads" / "aggregate.csv")
            baselines = pd.read_csv(out_root / "newcsf_heads" / "baseline_aggregate.csv")
            name = "regret_by_side_22csf_xarch"
        else:
            aggs = load_regime_free_aggregates(
                out_root / "regimefree_wilcoxon" / "xarch_aggregate.csv")
            baselines = load_baselines(out_root, "xarch")
            name = "regret_by_side_regime_free_xarch"
        plot_split(baselines, aggs, REGIME_FREE_LABELS, REGIME_FREE_COLORS,
                   fig_dir / name, suptitle=None)
        return
    for split in SPLITS:
        split_label = ("xarch — VGG13 → ResNet18 (cross-architecture)"
                       if split == "xarch"
                       else "lopo — leave-one-paradigm-out (cross-paradigm)")
        plot_split(load_baselines(out_root, split),
                   {c: load_predictor_aggregate(out_root, split, c)
                    for c in CONFIGS},
                   CONFIG_LABELS, CONFIG_COLORS,
                   fig_dir / f"regret_by_side_clique_bc_{split}",
                   suptitle=(f"Clique (b+c) headline predictor vs baselines — "
                             f"{split_label}\nPer-CSF binary, L2 Cs=50, "
                             "class_weighted, per-arch NC std, clique label rule"))


if __name__ == "__main__":
    main()

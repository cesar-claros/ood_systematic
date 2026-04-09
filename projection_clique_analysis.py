"""
Clique analysis with best projection variants substituted for base methods.

For each base method, replaces its scores with those of the best projection
variant (determined by paired AUGRC analysis) and re-runs the full
Friedman → Conover → Bron-Kerbosch clique pipeline. This answers:
"If we use the best variant of each method, which methods enter top cliques
that were previously excluded?"

Compares three configurations:
  1. base-only   – original base methods (--filter-methods behavior)
  2. best-swap   – base methods replaced by their best significant variant
  3. all-methods – all methods including all projection variants

Usage:
    python projection_clique_analysis.py --backbone Conv
    python projection_clique_analysis.py --backbone ViT
"""

import argparse
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.utils_stats import (
    load_all_scores,
    choose_baseline_rows,
    friedman_blocked,
    conover_posthoc_from_pivot,
    maximal_cliques_from_pmatrix,
    rank_cliques,
    greedy_exclusive_layers,
)

warnings.filterwarnings("ignore")

SOURCES = ["cifar10", "supercifar100", "cifar100", "tinyimagenet"]
DISTANCE_DICT = {"0": "test", "1": "near", "2": "mid", "3": "far"}

# Best projection variants from paired AUGRC analysis (projection_filtering_analysis.py).
# Only significant improvements (p < 0.05, positive mean_diff) are included.
BEST_VARIANTS = {
    "Conv": {
        "PCA RecError global": "PCA RecError class pred",
        "Maha": "Maha global",
        "pNML": "pNML class pred",
        "GradNorm": "GradNorm class pred",
        "CTMmean": "CTMmean class avg",
        "CTM": "CTM global",
        "fDBD": "fDBD global",
        "NNGuide": "NNGuide global",
        "REN": "REN global",
        "PCE": "PCE class",
        "MSR": "MSR global",
        "GEN": "GEN global",
        "PE": "PE global",
        "GE": "GE global",
    },
    "ViT": {
        "PCA RecError global": "PCA RecError class pred",
        "GradNorm": "GradNorm class pred",
        "REN": "REN class pred",
        "MLS": "MLS global",
        "GE": "GE global",
        "Energy": "Energy global",
        "PE": "PE global",
        "MSR": "MSR global",
        "PCE": "PCE global",
        "GEN": "GEN global",
        "GradNorm": "GradNorm class pred",
    },
}

# Methods to keep even when filtering projection variants (same as stats_eval.py)
KEEP_EXCEPTIONS = {
    "KPCA RecError global",
    "PCA RecError global",
    "MCD-KPCA RecError global",
    "MCD-PCA RecError global",
}


def load_data(backbone: str, clip_dir: str = "clip_scores") -> pd.DataFrame:
    """Load scores for all sources and merge CLIP groups."""
    clip_names = {"Unnamed: 0_level_1": "dataset", "Unnamed: 5_level_1": "group"}
    frames = []
    for source in SOURCES:
        config = {
            "AUGRC": f"scores_risk/scores_all_AUGRC_MCD-False_{backbone}_{source}.csv",
            "AURC": f"scores_risk/scores_all_AURC_MCD-False_{backbone}_{source}.csv",
            "CLIP_FILE": f"{clip_dir}/clip_distances_{source}.csv",
            "OUTDIR": ".",
            "ALPHA": 0.05,
            "N_BOOT": 0,
        }
        if not os.path.exists(config["AUGRC"]):
            logger.warning(f"Missing {config['AUGRC']}, skipping {source}")
            continue
        try:
            df = load_all_scores(config)
            df["source"] = source
            clip_path = config["CLIP_FILE"]
            if os.path.exists(clip_path):
                clip = pd.read_csv(clip_path, header=[0, 1])
                clip.columns = clip.columns.droplevel(0)
                clip = clip.rename(clip_names, axis="columns")
                if "group" in clip.columns:
                    df = df.merge(clip[["dataset", "group"]], on="dataset", how="left")
                    df["group"] = df["group"].apply(
                        lambda x: str(int(x)) if pd.notna(x) else x
                    )
            frames.append(choose_baseline_rows(df))
        except Exception as e:
            logger.error(f"Error loading {source}: {e}")
    return pd.concat(frames, axis=0, ignore_index=True)


def filter_base_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only base methods (same logic as stats_eval.py --filter-methods)."""
    mask = df["methods"].str.contains("global|class", case=False, na=False)
    mask &= ~df["methods"].isin(KEEP_EXCEPTIONS)
    return df[~mask].copy()


def swap_best_variants(df: pd.DataFrame, backbone: str) -> pd.DataFrame:
    """Replace base method scores with their best projection variant scores.

    For each base→variant mapping:
      1. Find rows matching the variant name.
      2. Rename them to the base name.
      3. Drop original base rows and insert the renamed variant rows.
    Methods without a significant best variant keep their original scores.
    """
    variant_map = BEST_VARIANTS.get(backbone, {})
    df_out = df.copy()

    for base_name, variant_name in variant_map.items():
        # Find variant rows
        variant_rows = df_out[df_out["methods"] == variant_name].copy()
        if variant_rows.empty:
            logger.warning(f"Variant '{variant_name}' not found, keeping base '{base_name}'")
            continue

        # Rename variant to base name (so it competes under the base name)
        variant_rows["methods"] = base_name

        # Drop original base rows
        df_out = df_out[df_out["methods"] != base_name]

        # Insert renamed variant rows
        df_out = pd.concat([df_out, variant_rows], ignore_index=True)

    # Now filter to base-method names only (the swapped ones + unchanged bases)
    df_out = filter_base_only(df_out)
    return df_out


def run_clique_pipeline(
    df: pd.DataFrame,
    backbone: str,
    metric: list[str] = ["AUGRC", "AURC"],
    alpha: float = 0.05,
) -> dict:
    """Run Friedman → Conover → Bron-Kerbosch for each source × group.

    Returns: {source: {group_label: {"members": [...], "avg_ranks": {...}}}}
    """
    rank_group = ["dataset", "model", "metric", "group", "run"]
    blocks = ["dataset", "model", "metric", "group", "run"]

    results = {}
    for source in SOURCES:
        df_src = df[df["source"] == source].copy()
        if backbone == "ViT":
            df_src = df_src[df_src["methods"] != "Confidence"]

        df_met = df_src[df_src["metric"].isin(metric)].copy()
        if df_met.empty:
            continue

        df_met["score_std_rank"] = df_met.groupby(rank_group)["score_std"].rank(
            ascending=False, method="average", pct=True
        )

        results[source] = {}
        if "group" not in df_met.columns:
            continue

        for group_id, g in df_met.groupby("group"):
            sub = g.copy()
            sub["block"] = sub[blocks].astype(str).agg("|".join, axis=1)
            try:
                stat, p, pivot = friedman_blocked(
                    sub, entity_col="methods", block_col="block", value_col="score_std"
                )
                if isinstance(stat, float) and not math.isnan(stat):
                    ph = conover_posthoc_from_pivot(pivot)
                    ranks_ = pivot.rank(axis=1, ascending=False)
                    avg_ranks_ = ranks_.mean(axis=0).sort_values()
                    cliques = maximal_cliques_from_pmatrix(ph, alpha)
                    scored = rank_cliques(cliques, list(avg_ranks_.index), avg_ranks_)
                    layers = greedy_exclusive_layers(scored)
                    if layers:
                        group_label = DISTANCE_DICT.get(str(group_id), str(group_id))
                        results[source][group_label] = {
                            "members": layers[0]["members"],
                            "mean_rank": layers[0]["mean_rank"],
                            "avg_ranks": avg_ranks_.to_dict(),
                        }
            except Exception as e:
                logger.error(f"Error in {source}/{group_id}: {e}")

    return results


def clique_membership_summary(cliques: dict) -> pd.DataFrame:
    """Build a boolean DataFrame: rows = source→group, columns = methods."""
    rows = []
    index = []
    for source in SOURCES:
        if source not in cliques:
            continue
        for group in ["test", "near", "mid", "far"]:
            if group not in cliques[source]:
                continue
            members = cliques[source][group]["members"]
            rows.append({m: True for m in members})
            index.append(f"{source}->{group}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, index=index).fillna(False)
    df = df[sorted(df.columns, key=str.casefold)]
    return df


def compare_configurations(base_cliques, swap_cliques, all_cliques) -> pd.DataFrame:
    """Compare clique counts across configurations."""
    base_mem = clique_membership_summary(base_cliques)
    swap_mem = clique_membership_summary(swap_cliques)
    all_mem = clique_membership_summary(all_cliques)

    all_methods = sorted(
        set(base_mem.columns) | set(swap_mem.columns) | set(all_mem.columns),
        key=str.casefold,
    )

    records = []
    for m in all_methods:
        base_count = int(base_mem[m].sum()) if m in base_mem.columns else 0
        swap_count = int(swap_mem[m].sum()) if m in swap_mem.columns else 0
        all_count = int(all_mem[m].sum()) if m in all_mem.columns else 0
        records.append(
            {
                "method": m,
                "cliques_base": base_count,
                "cliques_best_swap": swap_count,
                "cliques_all": all_count,
                "gain_swap_vs_base": swap_count - base_count,
            }
        )

    df = pd.DataFrame(records).sort_values("gain_swap_vs_base", ascending=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Clique analysis with best projection variants swapped in"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["Conv", "ViT"],
        default=None,
        help="Backbone to analyze (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="projection_clique_outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--clip-dir",
        type=str,
        default="clip_scores",
        help="Directory with CLIP distance CSVs",
    )
    args = parser.parse_args()

    backbones = [args.backbone] if args.backbone else ["Conv", "ViT"]
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)

    for backbone in backbones:
        print(f"\n{'='*80}")
        print(f"  {backbone}")
        print(f"{'='*80}")

        logger.info(f"Loading data for {backbone}...")
        df_all = load_data(backbone, clip_dir=args.clip_dir)
        logger.info(f"Loaded {len(df_all)} rows, {df_all['methods'].nunique()} methods")

        # Configuration 1: base-only
        logger.info("Running base-only clique analysis...")
        df_base = filter_base_only(df_all)
        base_cliques = run_clique_pipeline(df_base, backbone)

        # Configuration 2: best-swap (replace base with best variant)
        logger.info("Running best-swap clique analysis...")
        df_swap = swap_best_variants(df_all, backbone)
        swap_cliques = run_clique_pipeline(df_swap, backbone)

        # Configuration 3: all methods (no filtering)
        logger.info("Running all-methods clique analysis...")
        all_cliques = run_clique_pipeline(df_all, backbone)

        # Compare
        comparison = compare_configurations(base_cliques, swap_cliques, all_cliques)
        comparison.to_csv(outdir / f"clique_comparison_{backbone}.csv", index=False)

        # Save clique JSONs
        for label, cliques in [
            ("base", base_cliques),
            ("best_swap", swap_cliques),
            ("all", all_cliques),
        ]:
            # Simplify for JSON (drop avg_ranks)
            export = {}
            for src, groups in cliques.items():
                export[src] = {}
                for grp, info in groups.items():
                    export[src][grp] = info["members"]
            with open(outdir / f"cliques_{backbone}_{label}.json", "w") as f:
                json.dump(export, f, indent=2)

        # Print summary
        print(f"\n--- Clique membership comparison ({backbone}) ---")
        print(f"{'Method':<30s} {'Base':>6s} {'Swap':>6s} {'All':>6s} {'Gain':>6s}")
        print("-" * 60)
        for _, row in comparison.iterrows():
            gain = row["gain_swap_vs_base"]
            marker = " +" if gain > 0 else " -" if gain < 0 else "  "
            print(
                f"{row['method']:<30s} {row['cliques_base']:>6d} "
                f"{row['cliques_best_swap']:>6d} {row['cliques_all']:>6d} "
                f"{gain:>+5d}{marker}"
            )

        # Highlight methods that enter cliques only after swap
        new_entrants = comparison[
            (comparison["cliques_base"] == 0) & (comparison["cliques_best_swap"] > 0)
        ]
        if not new_entrants.empty:
            print(f"\n--- NEW entrants (entered cliques only after best-variant swap) ---")
            for _, row in new_entrants.iterrows():
                variant = BEST_VARIANTS.get(backbone, {}).get(row["method"], "?")
                print(
                    f"  {row['method']:<25s} → {variant:<30s} "
                    f"cliques: 0 → {row['cliques_best_swap']}"
                )

        # Methods that gained cliques after swap
        gainers = comparison[comparison["gain_swap_vs_base"] > 0]
        if not gainers.empty:
            print(f"\n--- Methods that GAINED cliques after swap ---")
            for _, row in gainers.iterrows():
                variant = BEST_VARIANTS.get(backbone, {}).get(row["method"], "unchanged")
                print(
                    f"  {row['method']:<25s} → {variant:<30s} "
                    f"cliques: {row['cliques_base']} → {row['cliques_best_swap']} "
                    f"(+{row['gain_swap_vs_base']})"
                )

        # Methods that lost cliques after swap
        losers = comparison[comparison["gain_swap_vs_base"] < 0]
        if not losers.empty:
            print(f"\n--- Methods that LOST cliques after swap ---")
            for _, row in losers.iterrows():
                variant = BEST_VARIANTS.get(backbone, {}).get(row["method"], "unchanged")
                print(
                    f"  {row['method']:<25s} → {variant:<30s} "
                    f"cliques: {row['cliques_base']} → {row['cliques_best_swap']} "
                    f"({row['gain_swap_vs_base']})"
                )


if __name__ == "__main__":
    main()

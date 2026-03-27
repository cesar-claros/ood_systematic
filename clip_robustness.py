"""
Robustness analysis of CLIP-based OOD proximity rankings across CLIP backbones.

Computes Spearman and Kendall rank correlations of OOD dataset ordering across
different CLIP models to validate that the near/mid/far grouping is not an
artifact of the specific embedding space or distance metrics.

Usage:
    # After running clip_proximity.py for each backbone:
    python clip_proximity.py --iid_dataset cifar10 --model_name ViT-B-32 --output_dir clip_proximity_results
    python clip_proximity.py --iid_dataset cifar10 --model_name ViT-B-16 --output_dir clip_proximity_results
    python clip_proximity.py --iid_dataset cifar10 --model_name ViT-L-14 --output_dir clip_proximity_results
    # ... repeat for each dataset

    # Then run robustness analysis:
    python clip_robustness.py \\
        --input-dir clip_proximity_results \\
        --models ViT-B-32 ViT-B-16 ViT-L-14 \\
        --output-dir clip_robustness
"""

import os
import json
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from loguru import logger


DATASETS = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]

# Distance metrics available in clip_proximity JSON output
DISTANCE_METRICS = {
    "fid": ("global", "fid"),
    "kid_mean": ("global", "kid_mean"),
    "img_centroid_dist_mean": ("class_aware", "img_centroid_dist_mean"),
    "text_alignment_mean": ("class_aware", "text_alignment_mean"),
}


def load_proximity_json(input_dir: str, dataset: str, model_name: str) -> dict | None:
    """Load clip_proximity JSON for a given dataset and model."""
    model_tag = model_name.replace("/", "-")
    path = os.path.join(input_dir, f"clip_proximity_{dataset}_{model_tag}.json")
    if not os.path.exists(path):
        # Fallback: no model tag (backward compatibility with ViT-B-32 default)
        path = os.path.join(input_dir, f"clip_proximity_{dataset}.json")
    if not os.path.exists(path):
        logger.warning(f"Not found: clip_proximity_{dataset}_{model_tag}.json")
        return None
    with open(path) as f:
        return json.load(f)


def extract_distances(results: dict, source_dataset: str, metric: str) -> pd.Series:
    """Extract a distance metric for all OOD sets, return as Series."""
    group, key = DISTANCE_METRICS[metric]
    values = {}
    for ood_set, metrics in results.items():
        if ood_set == source_dataset:
            continue
        try:
            values[ood_set] = float(metrics[group][key])
        except (KeyError, TypeError):
            pass
    return pd.Series(values, name=metric).sort_values()


def main():
    parser = argparse.ArgumentParser(
        description="Robustness analysis of CLIP proximity rankings across backbones"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing clip_proximity JSON files")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", required=True,
                        help="CLIP model names to compare (e.g., ViT-B-32 ViT-B-16 ViT-L-14)")
    parser.add_argument("--metric", type=str, default="fid",
                        choices=list(DISTANCE_METRICS.keys()),
                        help="Distance metric for ranking (default: fid)")
    parser.add_argument("--output-dir", type=str, default="clip_robustness")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load all proximity results ────────────────────────────────────────
    all_distances = {}  # (dataset, model) -> Series of distances
    for dataset in args.datasets:
        for model in args.models:
            results = load_proximity_json(args.input_dir, dataset, model)
            if results is None:
                continue
            distances = extract_distances(results, dataset, args.metric)
            all_distances[(dataset, model)] = distances
            logger.info(f"{dataset} / {model}: "
                        f"{list(distances.index)} ({args.metric})")

    if not all_distances:
        logger.error("No data loaded. Check --input-dir and --models.")
        return

    # ── Build full distance table ─────────────────────────────────────────
    dist_rows = []
    for (dataset, model), distances in all_distances.items():
        for ood_set, dist_val in distances.items():
            dist_rows.append({
                "source_dataset": dataset,
                "model": model,
                "ood_set": ood_set,
                args.metric: dist_val,
                "rank": distances.rank()[ood_set],
            })
    dist_df = pd.DataFrame(dist_rows)
    dist_path = os.path.join(args.output_dir,
                             f"ood_rankings_by_backbone_{args.metric}.csv")
    dist_df.to_csv(dist_path, index=False)
    logger.info(f"Saved full rankings: {dist_path}")

    # ── Pairwise rank correlations per source dataset ─────────────────────
    corr_rows = []
    for dataset in args.datasets:
        models_with_data = [m for m in args.models
                            if (dataset, m) in all_distances]
        if len(models_with_data) < 2:
            continue

        for m1, m2 in itertools.combinations(models_with_data, 2):
            d1 = all_distances[(dataset, m1)]
            d2 = all_distances[(dataset, m2)]
            common = d1.index.intersection(d2.index)
            if len(common) < 3:
                continue
            rho, p_rho = spearmanr(d1[common].values, d2[common].values)
            tau, p_tau = kendalltau(d1[common].values, d2[common].values)
            corr_rows.append({
                "source_dataset": dataset,
                "model_1": m1,
                "model_2": m2,
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "kendall_tau": tau,
                "kendall_p": p_tau,
                "n_ood_sets": len(common),
            })

    if not corr_rows:
        logger.warning("Not enough paired data for correlation analysis.")
        return

    corr_df = pd.DataFrame(corr_rows)
    corr_path = os.path.join(args.output_dir,
                             f"rank_correlations_{args.metric}.csv")
    corr_df.to_csv(corr_path, index=False)
    logger.success(f"Saved correlations: {corr_path}")

    # ── Ordinal consistency check ─────────────────────────────────────────
    logger.info(f"\n=== Ordinal Stability ({args.metric}) ===")
    logger.info(f"Mean Spearman rho: {corr_df['spearman_rho'].mean():.4f}")
    logger.info(f"Min  Spearman rho: {corr_df['spearman_rho'].min():.4f}")
    logger.info(f"Mean Kendall tau:  {corr_df['kendall_tau'].mean():.4f}")

    for dataset in args.datasets:
        sub = corr_df[corr_df["source_dataset"] == dataset]
        if sub.empty:
            continue
        logger.info(f"\n  {dataset}:")
        logger.info(f"    Spearman rho: "
                     f"{sub['spearman_rho'].mean():.4f} "
                     f"(range {sub['spearman_rho'].min():.4f}–"
                     f"{sub['spearman_rho'].max():.4f})")

    # ── Check if ordinal ordering is identical ────────────────────────────
    logger.info("\n=== OOD Set Ordering per Backbone ===")
    for dataset in args.datasets:
        orders = {}
        for model in args.models:
            if (dataset, model) not in all_distances:
                continue
            d = all_distances[(dataset, model)]
            orders[model] = tuple(d.index)  # already sorted by distance

        if len(orders) < 2:
            continue

        unique_orders = set(orders.values())
        status = "IDENTICAL" if len(unique_orders) == 1 else "DIFFERS"
        logger.info(f"  {dataset}: ordering {status} across {len(orders)} backbones")
        for model, order in orders.items():
            logger.info(f"    {model}: {' < '.join(order)}")

    # ── Leave-one-metric-out analysis ─────────────────────────────────────
    # Also compute a composite distance using all 4 metrics and check
    # how robust the ranking is to dropping each metric.
    logger.info("\n=== Leave-One-Metric-Out (composite ranking) ===")
    all_metrics = list(DISTANCE_METRICS.keys())
    loo_rows = []

    for dataset in args.datasets:
        # Use only the first model for LOO analysis
        model = args.models[0]
        results = load_proximity_json(args.input_dir, dataset, model)
        if results is None:
            continue

        # Full composite: rank by each metric, average ranks
        full_ranks = {}
        for m in all_metrics:
            d = extract_distances(results, dataset, m)
            # Higher text_alignment = more similar (closer), so invert
            if m == "text_alignment_mean":
                d = -d
            full_ranks[m] = d.rank()

        full_rank_df = pd.DataFrame(full_ranks)
        full_composite = full_rank_df.mean(axis=1).sort_values()
        full_order = tuple(full_composite.index)

        for drop_metric in all_metrics:
            kept = [m for m in all_metrics if m != drop_metric]
            loo_composite = full_rank_df[kept].mean(axis=1).sort_values()
            loo_order = tuple(loo_composite.index)
            # Compute Spearman between full and LOO
            common = full_composite.index.intersection(loo_composite.index)
            rho, _ = spearmanr(full_composite[common].values,
                               loo_composite[common].values)
            same_order = full_order == loo_order
            loo_rows.append({
                "source_dataset": dataset,
                "model": model,
                "dropped_metric": drop_metric,
                "spearman_vs_full": rho,
                "order_identical": same_order,
            })
            logger.info(f"  {dataset} drop {drop_metric}: "
                        f"rho={rho:.4f}, order={'SAME' if same_order else 'CHANGED'}")

    if loo_rows:
        loo_df = pd.DataFrame(loo_rows)
        loo_path = os.path.join(args.output_dir,
                                f"leave_one_metric_out_{args.metric}.csv")
        loo_df.to_csv(loo_path, index=False)
        logger.success(f"Saved LOO analysis: {loo_path}")


if __name__ == "__main__":
    main()

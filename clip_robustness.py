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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
        for model in args.models:
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
                logger.info(f"  {dataset}/{model} drop {drop_metric}: "
                            f"rho={rho:.4f}, order={'SAME' if same_order else 'CHANGED'}")

    if loo_rows:
        loo_df = pd.DataFrame(loo_rows)
        loo_path = os.path.join(args.output_dir,
                                "leave_one_metric_out.csv")
        loo_df.to_csv(loo_path, index=False)
        logger.success(f"Saved LOO analysis: {loo_path}")

    # ── Silhouette analysis for k selection ──────────────────────────────
    logger.info("\n=== Silhouette Analysis (k=2..6) ===")
    k_range = range(2, 7)
    sil_rows = []

    for dataset in args.datasets:
        for model in args.models:
            results = load_proximity_json(args.input_dir, dataset, model)
            if results is None:
                continue

            # Build feature matrix from all 4 metrics
            all_series = {}
            for m in all_metrics:
                d = extract_distances(results, dataset, m)
                if m == "text_alignment_mean":
                    d = -d  # invert so higher = farther
                all_series[m] = d

            feat_df = pd.DataFrame(all_series).dropna()
            if len(feat_df) < 3:
                logger.warning(f"  {dataset}/{model}: too few OOD sets ({len(feat_df)}), skipping")
                continue

            X = StandardScaler().fit_transform(feat_df.values)

            for k in k_range:
                if k >= len(feat_df):
                    continue
                km = KMeans(n_clusters=k, n_init=50, random_state=0).fit(X)
                sil = silhouette_score(X, km.labels_)
                sil_samples = silhouette_samples(X, km.labels_)
                # Per-cluster mean silhouette
                per_cluster = []
                for c in range(k):
                    mask = km.labels_ == c
                    per_cluster.append(sil_samples[mask].mean())
                min_cluster_sil = min(per_cluster)

                sil_rows.append({
                    "source_dataset": dataset,
                    "model": model,
                    "k": k,
                    "silhouette_score": sil,
                    "min_cluster_silhouette": min_cluster_sil,
                    "n_ood_sets": len(feat_df),
                })
                logger.info(f"  {dataset}/{model} k={k}: "
                            f"silhouette={sil:.4f}, min_cluster={min_cluster_sil:.4f}")

    if sil_rows:
        sil_df = pd.DataFrame(sil_rows)
        sil_path = os.path.join(args.output_dir, "silhouette_analysis.csv")
        sil_df.to_csv(sil_path, index=False)
        logger.success(f"Saved silhouette analysis: {sil_path}")

        # Summary: best k per dataset/model
        logger.info("\n=== Best k per Dataset/Model ===")
        for (dataset, model), grp in sil_df.groupby(["source_dataset", "model"]):
            best = grp.loc[grp["silhouette_score"].idxmax()]
            logger.info(f"  {dataset}/{model}: best k={int(best['k'])} "
                        f"(silhouette={best['silhouette_score']:.4f})")

        # Summary: mean silhouette per k across all datasets/models
        logger.info("\n=== Mean Silhouette per k ===")
        mean_sil = sil_df.groupby("k")["silhouette_score"].agg(["mean", "std", "min", "max"])
        for k, row in mean_sil.iterrows():
            logger.info(f"  k={k}: mean={row['mean']:.4f} ± {row['std']:.4f} "
                        f"(range {row['min']:.4f}–{row['max']:.4f})")


    # ── Hierarchical clustering + dendrogram ────────────────────────────
    logger.info("\n=== Hierarchical Clustering (Ward linkage) ===")
    hc_rows = []

    for dataset in args.datasets:
        for model in args.models:
            results = load_proximity_json(args.input_dir, dataset, model)
            if results is None:
                continue

            # Build feature matrix from all 4 metrics
            all_series = {}
            for m in all_metrics:
                d = extract_distances(results, dataset, m)
                if m == "text_alignment_mean":
                    d = -d
                all_series[m] = d

            feat_df = pd.DataFrame(all_series).dropna()
            if len(feat_df) < 3:
                continue

            X = StandardScaler().fit_transform(feat_df.values)
            labels = list(feat_df.index)

            # Ward linkage
            Z = linkage(X, method="ward")

            # Record merge distances (for gap analysis)
            merge_distances = Z[:, 2]
            gaps = np.diff(merge_distances)
            # Number of clusters = n - index_of_largest_gap
            # The largest gap in merge distance suggests the natural cut
            if len(gaps) > 0:
                largest_gap_idx = np.argmax(gaps)
                suggested_k = len(labels) - largest_gap_idx - 1
            else:
                suggested_k = 1

            # Also get cluster assignments for k=2,3,4
            for k in [2, 3, 4]:
                if k >= len(labels):
                    continue
                cluster_labels = fcluster(Z, t=k, criterion="maxclust")
                # Map clusters to OOD sets
                cluster_map = {lbl: int(cl) for lbl, cl in zip(labels, cluster_labels)}
                hc_rows.append({
                    "source_dataset": dataset,
                    "model": model,
                    "k": k,
                    "cluster_assignments": str(cluster_map),
                    "suggested_k_gap": suggested_k,
                })

            logger.info(f"  {dataset}/{model}: suggested k (largest gap) = {suggested_k}")
            logger.info(f"    Merge distances: {', '.join(f'{d:.3f}' for d in merge_distances)}")
            logger.info(f"    Gaps:            {', '.join(f'{g:.3f}' for g in gaps)}")

            # Save dendrogram figure
            fig, ax = plt.subplots(figsize=(8, 5))
            dendrogram(
                Z,
                labels=labels,
                leaf_rotation=45,
                leaf_font_size=9,
                ax=ax,
            )
            ax.set_title(f"Hierarchical Clustering: {dataset} / {model}")
            ax.set_ylabel("Ward distance")

            # Draw horizontal line at the cut for k=3
            if len(merge_distances) >= 2:
                # Cut between the 2nd and 3rd last merges gives k=3
                cut_height = (merge_distances[-2] + merge_distances[-3]) / 2 if len(merge_distances) >= 3 else merge_distances[-1]
                ax.axhline(y=cut_height, color="red", linestyle="--",
                           linewidth=1, label=f"k=3 cut")
                ax.legend(fontsize=9)

            model_tag = model.replace("/", "-")
            for ext in ["pdf", "jpeg"]:
                fig_path = os.path.join(args.output_dir,
                                        f"dendrogram_{dataset}_{model_tag}.{ext}")
                fig.savefig(fig_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info(f"    Saved dendrogram: dendrogram_{dataset}_{model_tag}.pdf")

    if hc_rows:
        hc_df = pd.DataFrame(hc_rows)
        hc_path = os.path.join(args.output_dir, "hierarchical_clustering.csv")
        hc_df.to_csv(hc_path, index=False)
        logger.success(f"Saved hierarchical clustering: {hc_path}")

        # Summary: suggested k counts
        suggested = hc_df.drop_duplicates(subset=["source_dataset", "model"])
        k_counts = suggested["suggested_k_gap"].value_counts().sort_index()
        logger.info("\n=== Suggested k (largest-gap method) frequency ===")
        for k, count in k_counts.items():
            logger.info(f"  k={k}: {count}/{len(suggested)} configurations")


if __name__ == "__main__":
    main()

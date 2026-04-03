"""
Generate k-means group assignments for all backbone/dataset combinations.

Reads clip_proximity JSONs from clip_proximity_results/ and produces
clip_distances_{dataset}_{model_tag}.csv files in the output directory.

Usage:
    python clip_clustering_all_backbones.py \
        --input-dir clip_proximity_results \
        --output-dir clip_scores \
        --n-clusters 3
"""
import os
import json
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger


DATASETS = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]


def cluster_one(input_path, dataset, n_clusters=3):
    """Run k-means clustering on one proximity JSON. Returns DataFrame with group assignments."""
    with open(input_path, "r") as f:
        clip_distances = json.load(f)

    distances_df = pd.DataFrame.from_dict(clip_distances, orient='index')

    dist_list = []
    for col in distances_df.columns:
        dist_df = distances_df[col].apply(pd.Series)
        dist_df.columns = dist_df.columns.str.replace('_', ' ')
        dist_list.append(dist_df)
    distances_df = pd.concat(dist_list, keys=['global', 'class-aware'], axis=1)
    distances_df.index = distances_df.index.str.replace('_', ' ')

    # Calculate inverse text alignment mean
    distances_df[('class-aware', 'inv text alignment mean')] = (
        1 - distances_df[('class-aware', 'text alignment mean')]
    )

    metrics = [
        ('global', 'kid mean'),
        ('global', 'fid'),
        ('class-aware', 'inv text alignment mean'),
        ('class-aware', 'img centroid dist mean'),
    ]
    metrics_show = metrics + [('group', '')]

    # Handle test row
    test_row = None
    dataset_index = dataset.replace('_', ' ')
    if dataset_index in distances_df.index:
        distances_df.rename(index={dataset_index: 'test'}, inplace=True)
    if dataset in distances_df.index:
        distances_df.rename(index={dataset: 'test'}, inplace=True)

    if 'test' in distances_df.index:
        test_row = distances_df.loc[['test']].copy()
        test_row['cluster_id'] = -1
        test_row['group'] = "0"
        distances_df = distances_df.drop('test', axis='rows')

    X = StandardScaler().fit_transform(distances_df[metrics].values)

    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=0).fit(X)
    distances_df["cluster_id"] = km.labels_

    # Order clusters by mean FID (near=lowest, far=highest)
    cluster_order = (
        distances_df.groupby("cluster_id")[[('global', 'fid')]]
        .mean()
        .sort_values(by=('global', 'fid'))
        .index.to_list()
    )

    id_to_name = {"0": "0"}
    id_to_name_dist = {"0": "ID"}
    for i, cid in enumerate(cluster_order):
        id_to_name[cid] = str(i + 1)
    group_names = ["Near", "Mid", "Far"]
    for i, cid in enumerate(cluster_order):
        id_to_name_dist[cid] = group_names[i] if i < len(group_names) else f"Group{i+1}"

    distances_df["group"] = distances_df["cluster_id"].map(id_to_name)
    if test_row is not None:
        distances_df = pd.concat([test_row, distances_df], axis=0)

    distances_df["group_name"] = distances_df["cluster_id"].map(id_to_name_dist)
    distances_df = distances_df.sort_values(by='group', ascending=True)

    return distances_df, metrics_show


def main():
    parser = argparse.ArgumentParser(
        description="Cluster CLIP proximity metrics for all backbones."
    )
    parser.add_argument("--input-dir", type=str, default="clip_proximity_results")
    parser.add_argument("--output-dir", type=str, default="clip_scores")
    parser.add_argument("--n-clusters", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in DATASETS:
        for model in MODELS:
            model_tag = model.replace("/", "-")
            input_path = os.path.join(
                args.input_dir, f"clip_proximity_{dataset}_{model_tag}.json"
            )
            if not os.path.exists(input_path):
                logger.warning(f"Missing: {input_path}, skipping")
                continue

            logger.info(f"Clustering {dataset} / {model_tag} (k={args.n_clusters})")
            distances_df, metrics_show = cluster_one(
                input_path, dataset, args.n_clusters
            )

            csv_path = os.path.join(
                args.output_dir, f"clip_distances_{dataset}_{model_tag}.csv"
            )
            distances_df[metrics_show].to_csv(csv_path)
            logger.success(f"  Saved {csv_path}")

            # Log assignments
            ood_rows = distances_df[distances_df.get("group", pd.Series()) != "0"]
            if "group_name" in distances_df.columns:
                for idx, row in distances_df.iterrows():
                    if idx == "test":
                        continue
                    logger.info(f"    {idx}: group={row.get('group_name', '?')}")


if __name__ == "__main__":
    main()

import os
import json
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="Cluster CLIP proximity metrics.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--n-clusters", type=int, default=3, help="Number of clusters for KMeans")
    parser.add_argument("--input-dir", type=str, default=".", help="Directory containing the input JSON file")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the output files")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX output in a text file")

    args = parser.parse_args()

    dataset = args.dataset
    n_clusters = args.n_clusters
    input_dir = args.input_dir
    output_dir = args.output_dir

    logger.info(f"Starting clustering for dataset: {dataset}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Number of clusters: {n_clusters}")

    metrics_show = [('global','kid mean'),('global','fid'),
                    ('class-aware','inv text alignment mean'),
                    ('class-aware','img centroid dist mean'),
                    ('group_name','')]
    metrics = [('global','kid mean'),('global','fid'),('class-aware','inv text alignment mean'),('class-aware','img centroid dist mean')]

    input_path = os.path.join(input_dir, f'clip_proximity_{dataset}.json')
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}")
    with open(input_path, "r") as f:
        clip_distances = json.load(f)

    distances_df = pd.DataFrame.from_dict(clip_distances, orient='index')

    dist_list = []
    for col in distances_df.columns:
        dist_df = distances_df[col].apply(pd.Series)
        dist_df.columns = dist_df.columns.str.replace('_',' ')
        dist_list.append(dist_df)
    distances_df = pd.concat(dist_list, keys=['global','class-aware'], axis=1)
    distances_df.index = distances_df.index.str.replace('_',' ')

    # Calculate inverse text alignment mean
    distances_df[('class-aware','inv text alignment mean')] = 1 - distances_df[('class-aware','text alignment mean')]
    
    # Handle test row
    if f'{dataset}' in distances_df.index:
        distances_df.rename(index={f'{dataset}': 'test'}, inplace=True)
    elif 'test' not in distances_df.index:
        logger.warning(f"Dataset name '{dataset}' not found in index, and 'test' not present. Check input data.")
        # Proceeding might fail if 'test' row is expected for reference, but let's try to verify
        if len(distances_df) > 0:
             # Assuming the first one might be it or just creating a placeholder if needed, 
             # but typical behavior is to split 'test' from others.
             # If exact match index isn't found, we skip the test_row extraction specific logic 
             # or error out. Let's error out safely if critical.
             pass

    if 'test' in distances_df.index:
        test_row = distances_df.loc[['test']].copy()
        test_row['cluster_id'] = -1
        test_row['group'] = '0'
        distances_df = distances_df.drop('test',axis='rows')
    else:
        logger.warning("No 'test' row found/extracted. Proceeding with clustering on all available data.")
        test_row = None

    distances_metrics_df = distances_df[metrics].values
    X = StandardScaler().fit_transform(distances_metrics_df)

    logger.info(f"Fitting KMeans with {n_clusters} clusters")
    km = KMeans(n_clusters=n_clusters, n_init=50, random_state=0).fit(X)
    distances_df["cluster_id"] = km.labels_

    # Ordering clusters by FID
    cluster_order = (
        distances_df.groupby("cluster_id")[[('global','fid')]]
        .mean()
        .sort_values(by=('global','fid'))
        .index.to_list()
    )

    id_to_name = {cluster_order[0]:"1", cluster_order[1]:"2", cluster_order[2]:"3",}
    id_to_name_dist = {cluster_order[0]:"Near", cluster_order[1]:"Mid", cluster_order[2]:"Far",'0':"ID"}
    
    if test_row is not None:
        distances_df = pd.concat([test_row,distances_df],axis=0)
    
    distances_df["group"] = distances_df["cluster_id"].map(id_to_name)
    distances_df["group_name"] = distances_df["cluster_id"].map(id_to_name_dist)
    distances_df = distances_df.sort_values(by='group', ascending=True)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'clip_distances_{dataset}.csv')
    distances_df[metrics_show].to_csv(csv_path)
    logger.success(f"Saved CSV results to {csv_path}")

    if args.latex:
        latex_output = distances_df[metrics_show]\
            .style\
            .set_caption(f"CLIP-based distance metrics. Dataset={dataset}")\
            .background_gradient(axis=0, cmap="coolwarm")\
            .format(precision=4)\
            .to_latex(environment='table', convert_css=True, column_format='r|p{2.5cm}|p{2.5cm}|p{2.5cm}|p{2.5cm}|c')
        
        txt_path = os.path.join(output_dir, f'clip_distances_{dataset}_latex.txt')
        with open(txt_path, "w") as f:
            f.write(latex_output)
        logger.success(f"Saved LaTeX output to {txt_path}")

if __name__ == "__main__":
    main()

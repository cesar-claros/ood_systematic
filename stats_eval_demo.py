import os, math, warnings
import pandas as pd
import numpy as np
import argparse
from loguru import logger
from src.utils_stats import (
    load_all_scores, 
    parse_method_variation, 
    friedman_blocked, 
    conover_posthoc_from_pivot,
    maximal_cliques_from_pmatrix, 
    rank_cliques, 
    greedy_exclusive_layers
)

# Set Default Environment Variables
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 200)

def main():
    parser = argparse.ArgumentParser(description="Step-by-step demonstration of OOD Stats Evaluation")
    parser.add_argument("--source", type=str, default='cifar10', help="Dataset source to use (e.g., cifar10)")
    parser.add_argument("--backbone", type=str, default='Conv', choices=['Conv', 'ViT'], help="Backbone type")
    parser.add_argument("--mcd", action="store_true", help="Set MCD flag")
    parser.add_argument("--metric", type=str, default='AURC', help="Single metric to demonstrate (e.g., AURC)")
    parser.add_argument("--output-dir", type=str, default="demo_outputs", help="Directory to save intermediate results")
    
    args = parser.parse_args()
    
    SOURCE = args.source
    BACKBONE = args.backbone
    MCD_flag = str(args.mcd)
    METRIC = args.metric
    OUTDIR = args.output_dir
    
    os.makedirs(OUTDIR, exist_ok=True)
    logger.add(os.path.join(OUTDIR, "demo.log"))
    
    logger.info("=== STEP 1: LOAD RAW SCORES ===")
    # Simulating the config used in stats_eval.py but for a single source/metric context
    # We load adjacent metrics if possible to match load_all_scores expectations or just mocked dict
    # load_all_scores expects a dict with metric names as keys and file paths as values
    
    # We will try to find the file for the requested metric.
    # Pattern: scores/scores_all_{METRIC}_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv
    filename = f"scores/scores_all_{METRIC}_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv"
    logger.info(f"Target file: {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}. Please ensure 'stats_eval.py' logic would find it.")
        return

    CONFIG = {
        METRIC: filename,
        "ALPHA": 0.05
    }
    
    # Data Loading
    df = load_all_scores(CONFIG)
    df['source'] = SOURCE
    
    logger.info(f"Loaded {len(df)} rows.")
    sample_raw = df.head(10)
    logger.info(f"Raw Scores Sample:\n{sample_raw}")
    sample_raw.to_csv(os.path.join(OUTDIR, "1_raw_scores_sample.csv"))
    
    logger.info("=== STEP 2: PREPROCESSING & FILTERING ===")
    # Filter by source (already done conceptually by loading only one, but ensuring consistnecy)
    df_ = df[df['source'] == SOURCE].copy()
    
    # Filter confidence baseline if ViT
    if BACKBONE == 'ViT':
        logger.info("Removing 'Confidence' method (ViT specific adjustment)")
        df_ = df_[df_['methods']!='Confidence']
    
    # Filter for the specific metric of interest
    df_met = df_[df_['metric'] == METRIC].copy()
    
    # Add CLIP groupings if available (Optional for this demo, focusing on stats pipeline)
    # We will simulate a 'group' column if it doesn't exist, assigning all to 'test' group for simplicity 
    # unless we load the clip file. Let's try to load clip file if present to make it realistic.
    clip_file = f"clip_scores/clip_distances_{SOURCE}.csv"
    if os.path.exists(clip_file):
        logger.info(f"Loading CLIP groups from {clip_file}")
        clip = pd.read_csv(clip_file, header=[0,1])
        clip.columns = clip.columns.droplevel(0)
        clip = clip.rename({'Unnamed: 0_level_1':'dataset', 'Unnamed: 5_level_1':'group'}, axis='columns')
        if "group" in clip.columns:
            df_met = df_met.merge(clip[["dataset", "group"]], on="dataset", how="left")
            logger.info("Merged CLIP groups.")
    else:
        logger.info("CLIP file not found. Assigning default group 'All'.")
        df_met['group'] = 'All'
        
    logger.info(f"Data ready for ranking. Rows: {len(df_met)}")
    df_met.to_csv(os.path.join(OUTDIR, "2_preprocessed_data.csv"), index=False)

    logger.info("=== STEP 3: RANKING ===")
    # Rank methods within each block (dataset, model, metric, group)
    # Lower score is better? usually yes for error metrics, but `load_all_scores` might invert/normalize?
    # stats_eval.py uses: ascending=False (implicitly depending on metric, usually score_std is normalized)
    # Let's stick strictly to stats_eval.py logic:
    # df_["rank"] = df_.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
    # rank_group = ["dataset", 'model', "metric", 'group']
    
    rank_group = ["dataset", 'model', "metric", 'group']
    df_met["rank"] = df_met.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
    
    # Show ranks for a specific dataset/model block
    example_block = df_met.groupby(rank_group).first().index[0]
    logger.info(f"Showing ranks for block: {example_block}")
    block_data = df_met.set_index(rank_group).loc[example_block].sort_values("rank")
    logger.info(f"Block Ranks:\n{block_data[['methods', 'score_std', 'rank']].head(10)}")
    block_data.to_csv(os.path.join(OUTDIR, "3_block_ranking_example.csv"))

    logger.info("=== STEP 4: AGGREGATE RANKS (Global/Group) ===")
    # Average rank per method within each group
    avg_rank = df_met.groupby(['group', "methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
    logger.info(f"Average Ranks per Group:\n{avg_rank.head(10)}")
    avg_rank.to_csv(os.path.join(OUTDIR, "4_aggregated_ranks.csv"))

    logger.info("=== STEP 5: FRIEDMAN TEST & POST-HOC ANALYSIS ===")
    alpha = CONFIG["ALPHA"]
    
    # Process each group separately
    unique_groups = df_met['group'].unique()
    
    combined_cliques_df = []
    
    for grp in unique_groups:
        logger.info(f"Analyzing Group: {grp}")
        sub = df_met[df_met['group'] == grp].copy()
        
        # Block definition for Friedman: dataset|model|metric
        blocks = ['dataset',"model",'metric']
        sub["block"] = sub[blocks].astype(str).agg("|".join, axis=1)
        
        # Friedman Test
        stat, p, pivot = friedman_blocked(sub, entity_col="methods", block_col="block", value_col="score_std")
        logger.info(f"Friedman Test (Group {grp}): Statistic={stat:.4f}, p-value={p:.4e}")
        
        with open(os.path.join(OUTDIR, f"5_{grp}_friedman_result.txt"), "w") as f:
            f.write(f"Friedman Statistic: {stat}\nP-value: {p}\nNum Blocks: {pivot.shape[0]}\nNum Methods: {pivot.shape[1]}\n")
            
        if isinstance(stat, float) and not math.isnan(stat):
            # Nemenyi/Conover Post-hoc
            # This generates a matrix of p-values for pairwise comparisons
            ph = conover_posthoc_from_pivot(pivot)
            logger.info(f"Post-hoc Matrix Shape: {ph.shape}")
            ph.to_csv(os.path.join(OUTDIR, f"6_{grp}_posthoc_pvalues.csv"))
            
            # Average ranks for this group (needed for clique finding)
            ranks_ = pivot.rank(axis=1, ascending=False)
            avg_ranks_ = ranks_.mean(axis=0).sort_values()
            
            logger.info("=== STEP 6: FINDING MAXIMAL CLIQUES ===")
            logger.info("Finding cliques of methods that are not significantly different...")
            cliques = maximal_cliques_from_pmatrix(ph, alpha)
            logger.info(f"Found {len(cliques)} maximal cliques.")
            
            # Save raw cliques
            with open(os.path.join(OUTDIR, f"7_{grp}_raw_cliques.txt"), "w") as f:
                for i, c in enumerate(cliques):
                    f.write(f"Clique {i}: {list(c)}\n")
            
            logger.info("=== STEP 7: RANKING CLIQUES & LAYERING ===")
            # Rank cliques based on the average rank of their members
            scored = rank_cliques(cliques, list(avg_ranks_.index), avg_ranks_)
            
            # Greedy exclusive layering: Top-1, Top-2...
            layers = greedy_exclusive_layers(scored)
            logger.info(f"Identified {len(layers)} layers.")
            
            # Save layers info
            layer_data = []
            for i, layer in enumerate(layers):
                members = layer['members']
                mean_rank = layer['mean_rank']
                logger.debug(f"Layer {i+1} (Mean Rank {mean_rank:.4f}): {members[:3]}...")
                layer_data.append({
                    "layer": i+1,
                    "mean_rank": mean_rank,
                    "members": ", ".join(members)
                })
            
            pd.DataFrame(layer_data).to_csv(os.path.join(OUTDIR, f"8_{grp}_final_layers.csv"), index=False)
            
            top_layer_members = layers[0]['members']
            logger.info(f"Top Layer Members for {grp}: {top_layer_members}")

    logger.success("Demonstration complete. Check intermediate files in output directory.")

if __name__ == "__main__":
    main()

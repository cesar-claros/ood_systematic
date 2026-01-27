import os, re, math, json, itertools, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from src.utils_stats import *
from loguru import logger
import argparse

# Set Default Environment Variables if not present
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)

def main():
    parser = argparse.ArgumentParser(description="OOD Systematic Eval Analysis")
    parser.add_argument("--mcd", action="store_true", help="Set MCD flag (default: False)")
    parser.add_argument("--backbone", type=str, required=True, choices=['Conv', 'ViT'], help="Backbone type")
    parser.add_argument("--metric-group", type=str, required=True, choices=['RC', 'ROC'], help="Metric group: RC=['AUGRC', 'AURC'], ROC=['AUROC_f', 'FPR@95TPR']")
    parser.add_argument("--output-dir", type=str, default="ood_eval_outputs", help="Output directory")

    args = parser.parse_args()

    MCD_flag = str(args.mcd) # 'True' or 'False' as string for file paths if that's the convention
    BACKBONE = args.backbone
    OUTDIR = args.output_dir
    
    if args.metric_group == 'RC':
        metric = ['AUGRC','AURC']
    else:
        metric = ['AUROC_f','FPR@95TPR']
    
    logger.info(f"Starting stats eval with: MCD={MCD_flag}, Backbone={BACKBONE}, MetricGroup={args.metric_group} ({metric})")
    logger.info(f"Output directory: {OUTDIR}")

    os.makedirs(OUTDIR, exist_ok=True)

    clip_names = {'Unnamed: 0_level_1':'dataset', 
                    'Unnamed: 5_level_1':'group'}
    distance_dict = {'0':'test','1':'near','2':'mid','3':'far'}

    df_all = []
    
    # =========================
    # CONFIG: edit as needed
    # =========================
    sources = ['cifar10', 'supercifar100', 'cifar100', 'tinyimagenet']
    for SOURCE in sources:
        logger.info(f"Processing source: {SOURCE}")
        CONFIG = {
            # Map metric -> CSV path. You can point these to your real files per metric
            # NOTE: Assuming file naming convention matches what was in script. 
            # If MCD flag in filename uses 'True'/'False' string, this works.
            "FPR@95TPR": f"scores/scores_all_FPR@95TPR_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AUROC_f":   f"scores/scores_all_AUROC_f_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AUGRC":     f"scores/scores_all_AUGRC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AURC":      f"scores/scores_all_AURC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            # Optional CLIP distances / groupings file (columns: dataset, features..., e.g., 'group', 'clip_dist_id_ood', etc.)
            "CLIP_FILE": f"clip_scores/clip_distances_{SOURCE}.csv",  # set to None if not available
            # Output dir
            "OUTDIR": OUTDIR,
            # Alpha for significance
            "ALPHA": 0.05,
            # Bootstraps for CIs
            "N_BOOT": 2000
        }

        alpha = CONFIG["ALPHA"]
        try:
            df = load_all_scores(CONFIG)
            df['source'] = SOURCE

            if os.path.exists(CONFIG["CLIP_FILE"]):
                clip = pd.read_csv(CONFIG["CLIP_FILE"],header=[0,1])
                clip.columns = clip.columns.droplevel(0)
                clip = clip.rename(clip_names,axis='columns')
                # Expect columns: dataset, group (optional), and numeric features like clip_dist_to_ID, etc.
                # Keep only columns present
                numeric_features = [c for c in clip.columns if c not in ["dataset", "group"]]
                if len(numeric_features) == 0:
                   logger.warning("CLIP file found but no numeric features detected; skipping LODO.")
                   merged = df # fallback
                else:
                    # Stability: variance within vs between clusters if 'group' present
                    if "group" in clip.columns:
                        merged = df.merge(clip[["dataset", "group"]], on="dataset", how="left")
                    else:
                        merged = df
            else:
                 logger.warning(f"CLIP file not found: {CONFIG['CLIP_FILE']}")
                 merged = df

            df_all.append(choose_baseline_rows(merged))
        except Exception as e:
            logger.error(f"Error processing {SOURCE}: {e}")
            continue

    if not df_all:
        logger.error("No data loaded. Exiting.")
        return

    rank_group = ["dataset", 'model', "metric", 'group']
    blocks = ['dataset',"model",'metric']

    members_list = []
    
    # Combined Dataframe
    df_combined = pd.concat(df_all,axis=0)

    for source_ in sources:
        logger.info(f"Computing ranks for source: {source_}")
        
        # Filter for backbone specifics if needed
        # Assuming df_all logic was correct, re-applying checks
        
        df_ = df_combined[df_combined['source']==source_].copy()
        
        if BACKBONE=='ViT':
             df_ = df_[df_['methods']!='Confidence']
             
        if df_.empty:
            logger.warning(f"No data for source {source_} after filtering.")
            continue

        df_["rank"] = df_.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
        # Average rank per method
        avg_rank = df_.groupby(["methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
        # Average rank per method per metric
        df_met = df_[(df_['metric']==metric[0])|(df_['metric']==metric[1])].copy()
        
        if df_met.empty:
             logger.warning(f"No data for metrics {metric} in source {source_}.")
             continue
             
        df_met["rank"] = df_met.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
        # avg_rank_group_metric = df_met.groupby(['group','metric',"methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
        # parsed = avg_rank_group_metric["methods"].apply(parse_method_variation)
        # avg_rank_group_metric["method_base"] = parsed.apply(lambda x: x[0])
        # avg_rank_group_metric["variation"] = parsed.apply(lambda x: x[1])
        
        friedman_results = []
        layered_cliques = {}
        layered_ranks = []
        
        # Group by 'group' column (merged from CLIP)
        if 'group' not in df_met.columns:
             logger.warning(f"Group column missing in data for {source_}. Skipping friedman analysis.")
             continue

        for dataset_group, g in df_met.groupby('group'):
            sub = g.copy()
            sub["block"] = sub[blocks].astype(str).agg("|".join, axis=1)
            try:
                stat, p, pivot = friedman_blocked(sub, entity_col="methods", block_col="block", value_col="score_std")
                friedman_results.append({"metric": metric, "friedman_stat": stat, "p": p, "n_blocks": pivot.shape[0], "n_methods": pivot.shape[1]})
                
                if isinstance(stat, float) and not math.isnan(stat):
                    ph = conover_posthoc_from_pivot(pivot)
                    ranks_ = pivot.rank(axis=1, ascending=False)
                    avg_ranks_ = ranks_.mean(axis=0).sort_values()
                    avg_ranks_.name = dataset_group
                    cliques = maximal_cliques_from_pmatrix(ph, alpha)
                    scored = rank_cliques(cliques, list(avg_ranks_.index), avg_ranks_)
                    layers = greedy_exclusive_layers(scored)   # disjoint “Top-1, Top-2, …” layers
                    layered_cliques.update({f'{dataset_group}':layers})
                    layered_ranks.append(avg_ranks_)
            except Exception as e:
                logger.error(f"Error in Friedman/Posthoc for {source_} group {dataset_group}: {e}")

        clique_members = []
        clique_avg = []
        datasets_order = ['0','1','2','3'] # mapped to test, near, mid, far
        
        # Verify if all groups exist
        all_groups_present = True
        for c in datasets_order:
             if c not in layered_cliques:
                  all_groups_present = False
                  logger.warning(f"Group {c} ({distance_dict.get(c,c)}) missing in layers for {source_}")
        
        if all_groups_present:
            for c in datasets_order:
                clique_members.append( layered_cliques[c][0]['members'] )
                clique_avg.append(layered_cliques[c][0]['mean_rank'])
            
            member_df = pd.DataFrame([{name: True for name in names} for names in clique_members])
            member_df = member_df.where(member_df==True, False)
            member_df.index = [source_+'->'+distance_dict[d] for d in datasets_order]
            members_list.append(member_df)
        else:
             logger.warning(f"Skipping members plot for {source_} due to missing groups.")

    if not members_list:
        logger.error("No members lists created. Exiting.")
        return

    reorder_index = [
     'cifar10->test',
     'supercifar100->test',
     'cifar100->test',
     'tinyimagenet->test',
     'cifar10->near',
     'supercifar100->near',
     'cifar100->near',
     'tinyimagenet->near',
     'cifar10->mid',
     'supercifar100->mid',
     'cifar100->mid',
     'tinyimagenet->mid',
     'cifar10->far',
     'supercifar100->far',
     'cifar100->far',
     'tinyimagenet->far',
     ]

    members_all = pd.concat(members_list,axis=0)
    members_all = members_all.where(members_all==True, False)
    # Filter to existing indices from reorder_index
    valid_reorder = [idx for idx in reorder_index if idx in members_all.index]
    members_all = members_all.loc[valid_reorder]

    # Plotting Logic
    figsize = (5,8)
    rects_list = []
    rows_shading = []
    
    # Configure plot based on args
    if args.metric_group == 'ROC' and BACKBONE=='ViT':
        figsize = (5,8)
        cols_show = [x for x in members_all.columns if 'class' not in x]
        cols_show.extend(['MLS class pred','MSR class pred','PCA RecError class pred','PCE class pred'])
        cols_show = [c for c in cols_show if c in members_all.columns] # safety
        members_all = members_all[cols_show]
        members_all = members_all[sorted(members_all.columns)]
        rows_shading = [(0,1),(4,4),(7,8),(11,12),(14,16),(20,20),(23,23),(26,28),(31,32),(34,34),(37,38)]
        # Add rects definition (simplified for refactor, keeping original structure)
        rects_list = [[(-0.5, 8.5),4,2,'dotted','black','wheat'],
                        [(-0.5, 16.5),4,3,'dotted','black','wheat'],
                        [(-0.5, 25.5),4,3,'dotted','black','wheat'],
                        [(-0.5, 28.5),4,2,'dotted','black','wheat'],
                        [(-0.5, 30.5),4,2,'dotted','black','wheat'],
                        [(4.5, 10.5),3,2,'dotted','black','plum'],
                        [(8.5, 10.5),3,2,'dotted','black','plum'],
                        [(12.5, 10.5),3,2,'dotted','black','plum'],
                        [(3.5, 23.5),3,2,'dotted','black','plum'],
                        [(7.5, 23.5),3,2,'dotted','black','plum'],
                        [(11.5, 23.5),3,2,'dotted','black','plum'],
                        ]
    elif args.metric_group == 'RC' and BACKBONE=='ViT':
        figsize = (5,8)
        cols_show = [x for x in members_all.columns if 'class' not in x]
        cols_show.extend(['GradNorm class pred','KPCA RecError class pred','MLS class'])
        cols_show = [c for c in cols_show if c in members_all.columns]
        members_all = members_all[cols_show]
        members_all = members_all[sorted(members_all.columns)]
        rows_shading = [(0,1),(4,4),(7,8),(11,13),(16,18),(21,21),(24,24),(26,27),(30,31),(33,33),(36,37)]
        rects_list = [[(-0.5, 8.5),4,2,'dotted','black','wheat'],
                        [(-0.5, 18.5),4,2,'dotted','black','wheat'],
                        [(-0.5, 25.5),4,2,'dotted','black','wheat'],
                        [(-0.5, 29.5),4,2,'dotted','black','wheat'],
                        [(4.5, 10.5),3,3,'dotted','black','plum'],
                        [(8.5, 10.5),3,3,'dotted','black','plum'],
                        [(12.5, 10.5),3,3,'dotted','black','plum'],
                        [(3.5, 13.5),3,2,'dotted','black','plum'],
                        [(7.5, 13.5),3,2,'dotted','black','plum'],
                        [(11.5, 13.5),3,2,'dotted','black','plum'],
                        ]

    elif args.metric_group == 'ROC' and BACKBONE=='Conv': 
        figsize = (5,8)
        cols_show = [x for x in members_all.columns if 'class' not in x]
        cols_show = [c for c in cols_show if c in members_all.columns]
        members_all = members_all[cols_show]
        members_all = members_all[sorted(members_all.columns)]
        rows_shading = [(0,1),(4,4),(6,7),(10,11),(14,15),(18,18),(21,22),(25,26)]
        rects_list = [[(6.5, -0.5),1,2,'dotted','black','skyblue'],
                    [(2.5, -0.5),1,2,'dotted','black','skyblue'],
                    [(10.5, -0.5),1,2,'dotted','black','skyblue'],
                    [(14.5, -0.5),1,2,'dotted','black','skyblue'],
                    [(-0.5, 9.5),4,2,'dotted','black','wheat'],
                    [(-0.5, 13.5),4,2,'dotted','black','wheat'],
                    [(-0.5, 18.5),4,2,'dotted','black','wheat'],
                    [(-0.5, 22.5),4,2,'dotted','black','wheat'],
                    [(3.5, 5.5),3,2,'dotted','black','plum'],
                    [(7.5, 5.5),3,2,'dotted','black','plum'],
                    [(11.5, 5.5),3,2,'dotted','black','plum'],
                    [(3.5, 11.5),3,2,'dotted','black','plum'],
                    [(7.5, 11.5),3,2,'dotted','black','plum'],
                    [(11.5, 11.5),3,2,'dotted','black','plum'],
                    [(3.5, 17.5),3,1,'dotted','black','plum'],
                    [(7.5, 17.5),3,1,'dotted','black','plum'],
                    [(11.5, 17.5),3,1,'dotted','black','plum'],
                    ]
    elif args.metric_group == 'RC' and BACKBONE=='Conv':
        figsize = (5,8)
        cols_show = [x for x in members_all.columns if 'class' not in x]
        cols_show = [c for c in cols_show if c in members_all.columns]
        members_all = members_all[cols_show]
        members_all = members_all[sorted(members_all.columns)]
        rows_shading = [(0,1),(4,4),(6,7),(10,11),(14,15),(18,19),(21,22),(25,26),(29,29)]
        rects_list = [[(6.5, -0.5),1,2,'dotted','black','deepskyblue'],
                    [(2.5, -0.5),1,2,'dotted','black','deepskyblue'],
                    [(10.5, -0.5),1,2,'dotted','black','deepskyblue'],
                    [(14.5, -0.5),1,2,'dotted','black','deepskyblue'],
                    [(-0.5, 9.5),4,2,'dotted','black','wheat'],
                    [(-0.5, 15.5),4,2,'dotted','black','wheat'],
                    [(3.5, 5.5),3,2,'dotted','black','plum'],
                    [(7.5, 5.5),2,2,'dotted','black','plum'],
                    [(11.5, 5.5),1,2,'dotted','black','plum'],
                    [(3.5, 13.5),3,2,'dotted','black','plum'],
                    [(7.5, 13.5),2,2,'dotted','black','plum'],
                    [(11.5, 13.5),1,2,'dotted','black','plum'],
                    [(3.5, 17.5),3,2,'dotted','black','plum'],
                    [(7.5, 17.5),2,2,'dotted','black','plum'],
                    [(11.5, 17.5),1,2,'dotted','black','plum'],
                    [(5.5, 26.5),1,2,'dotted','black','forestgreen'],
                    [(9.5, 26.5),1,2,'dotted','black','forestgreen'],
                    [(13.5, 26.5),1,2,'dotted','black','forestgreen'],
                    ]

    logger.info("Generating plot...")
    fig, ax = plt.subplots(1,2,figsize=figsize,width_ratios=[0.8,0.2],sharey='all')
    plt.subplots_adjust(wspace=0.05)
    
    # Adjust c_list length if necessary, simplistic palette reuse
    c_list = ['tab:green'] * 4 + ['tab:blue'] * 4 + ['tab:red'] * 4 + ['tab:orange'] * 4
    # Extend if more rows
    if len(members_all) > len(c_list):
        c_list = (c_list * (len(members_all)//len(c_list) + 1))[:len(members_all)]


    try:
        plot_grid(members_all, color_dotline=c_list, ax=ax[0], zorder=10)
    except Exception as e:
        logger.error(f"Error plotting grid: {e}")
        return

    for (y1,y2) in rows_shading:
        ax[0].axhspan(-.5+y1, .5+y2, facecolor='lightgray', alpha=0.5)
        
    ax[0].set_title(f'Top cliques\n(Backbone:{BACKBONE}, Metrics={metric})')
    
    sns.barplot(x=members_all.sum(axis=0), y=members_all.columns, color='gray', ax=ax[1])
    # For modern matplotlib versions
    try:
        ax[1].bar_label(ax[1].containers[0])
    except:
        pass # Skip if old matplotlib
        
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    for x in ["top", "bottom", "right"]:
        ax[1].spines[x].set_visible(False)

    for xy,width,height,ls,lcolor,bcolor in rects_list:
        rect = Rectangle(xy, width, height,
                        linewidth=2,
                        edgecolor=lcolor,
                        facecolor=bcolor,
                        linestyle=ls,
                        alpha=0.5)
        ax[0].add_patch(rect)
    
    out_filename = f'top_cliques_{BACKBONE}_{MCD_flag}_{args.metric_group}.pdf'
    out_path = os.path.join(OUTDIR, out_filename)
    fig.savefig(out_path, bbox_inches='tight')
    logger.success(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()

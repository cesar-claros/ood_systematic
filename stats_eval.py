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
    parser.add_argument("--metric-group", type=str, required=True, choices=['RC', 'ROC', 'CE', 'CE_BOUND'], help="Metric group: RC=['AUGRC', 'AURC'], ROC=['AUROC_f', 'FPR@95TPR'], CE=['ECE_L1','ECE_L2'], CE_BOUND=['ECE_L1_BOUND','ECE_L2_BOUND']")
    parser.add_argument("--output-dir", type=str, default="ood_eval_outputs", help="Output directory")
    parser.add_argument("--filter-methods", action="store_true", help="Exclude methods containing 'global' or 'class' (except PCA/KPCA RecError global variants)")
    parser.add_argument("--model", type=str, nargs='+', default=None,
                        help="Filter by model(s). Conv options: confidnet, devries, dg. ViT options: modelvit. Default: all models.")
    parser.add_argument("--clip-dir", type=str, default="clip_scores",
                        help="Directory containing clip_distances_{source}.csv files (default: clip_scores)")

    args = parser.parse_args()

    # Validate --model choices against --backbone
    if args.model is not None:
        valid_conv = {'confidnet', 'devries', 'dg'}
        valid_vit = {'modelvit'}
        valid = valid_vit if args.backbone == 'ViT' else valid_conv
        invalid = set(args.model) - valid
        if invalid:
            parser.error(f"Invalid model(s) {invalid} for backbone {args.backbone}. Valid options: {valid}")

    MCD_flag = str(args.mcd) # 'True' or 'False' as string for file paths if that's the convention
    BACKBONE = args.backbone
    OUTDIR = args.output_dir
    
    if args.metric_group == 'RC':
        metric = ['AUGRC','AURC']
    elif args.metric_group == 'ROC':
        metric = ['AUROC_f','FPR@95TPR']
    elif args.metric_group == 'CE_BOUND':
        metric = ['ECE_L1_BOUND','ECE_L2_BOUND']
    elif args.metric_group == 'CE':
        metric = ['ECE_L1','ECE_L2']
    else:
        raise ValueError(f"Unknown metric group: {args.metric_group}")

    logger.info(f"Starting stats eval with: MCD={MCD_flag}, Backbone={BACKBONE}, MetricGroup={args.metric_group} ({metric}), CLIP dir={args.clip_dir}")
    logger.info(f"Output directory: {OUTDIR}")

    os.makedirs(OUTDIR, exist_ok=True)

    clip_names = {'Unnamed: 0_level_1':'dataset',
                    'Unnamed: 5_level_1':'group'}
    distance_dict_full = {'0':'test','1':'near','2':'mid','3':'far'}
    distance_dict = distance_dict_full  # may be narrowed after data loading

    df_all = []
    
    # =========================
    # CONFIG: edit as needed
    # =========================
    # Derive sources from available CLIP CSV files in the clip directory
    _all_sources = ['cifar10', 'supercifar100', 'cifar100', 'tinyimagenet']
    sources = [s for s in _all_sources if os.path.exists(os.path.join(args.clip_dir, f"clip_distances_{s}.csv"))]
    logger.info(f"Active sources (from {args.clip_dir}): {sources}")
    for SOURCE in sources:
        logger.info(f"Processing source: {SOURCE}")
        CONFIG = {
            # Map metric -> CSV path. You can point these to your real files per metric
            # NOTE: Assuming file naming convention matches what was in script. 
            # If MCD flag in filename uses 'True'/'False' string, this works.
            "FPR@95TPR":       f"scores_risk/scores_all_FPR@95TPR_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AUROC_f":         f"scores_risk/scores_all_AUROC_f_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AUGRC":           f"scores_risk/scores_all_AUGRC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AURC":            f"scores_risk/scores_all_AURC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L1":          f"scores_calibration/scores_all_ECE_L1_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L2":          f"scores_calibration/scores_all_ECE_L2_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "MCE":             f"scores_calibration/scores_all_MCE_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L1_BOUND":    f"scores_calibration/scores_all_ECE_L1_BOUND_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L2_BOUND":    f"scores_calibration/scores_all_ECE_L2_BOUND_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            # Optional CLIP distances / groupings file (columns: dataset, features..., e.g., 'group', 'clip_dist_id_ood', etc.)
            "CLIP_FILE": f"{args.clip_dir}/clip_distances_{SOURCE}.csv",  # set to None if not available
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

            if args.filter_methods:
                keep_exceptions = {
                    "KPCA RecError global",
                    "PCA RecError global",
                    "MCD-KPCA RecError global",
                    "MCD-PCA RecError global",
                }
                mask = df['methods'].str.contains('global|class', case=False, na=False)
                mask &= ~df['methods'].isin(keep_exceptions)
                n_removed = mask.sum()
                df = df[~mask]
                logger.info(f"Filtered {n_removed} rows with 'global'/'class' methods for {SOURCE}")

            if args.model is not None:
                df = df[df['model'].isin(args.model)]
                logger.info(f"Filtered to models {args.model} for {SOURCE} ({len(df)} rows remaining)")

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
                        # Convert group to string to avoid float keys from NaN promotion
                        merged["group"] = merged["group"].apply(lambda x: str(int(x)) if pd.notna(x) else x)
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

    rank_group = ["dataset", 'model', "metric", 'group', 'run' ]
    blocks = ['dataset', 'model', 'metric', 'group', 'run']

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

        # Determine which groups are present in the data for this source
        available_groups = sorted(layered_cliques.keys())
        # Build datasets_order from available groups (e.g. ['0','1','2','3'] or ['0','1','3'])
        datasets_order = [g for g in ['0','1','2','3'] if g in available_groups]

        # Verify all available groups have cliques
        all_groups_present = True
        for c in datasets_order:
             if c not in layered_cliques:
                  all_groups_present = False
                  logger.warning(f"Group {c} ({distance_dict.get(c,c)}) missing in layers for {source_}")

        if all_groups_present and len(datasets_order) > 0:
            for c in datasets_order:
                clique_members.append( layered_cliques[c][0]['members'] )
                clique_avg.append(layered_cliques[c][0]['mean_rank'])

            member_df = pd.DataFrame([{name: True for name in names} for names in clique_members])
            member_df = member_df.where(member_df==True, False)
            member_df.index = [source_+'->'+distance_dict.get(d, d) for d in datasets_order]
            members_list.append(member_df)
        else:
             logger.warning(f"Skipping members plot for {source_} due to missing groups.")

    if not members_list:
        logger.error("No members lists created. Exiting.")
        return

    # Build reorder_index dynamically from the groups present in the data
    group_order = ['test', 'near', 'mid', 'far']
    # Detect which group labels are actually present in the members_list indices
    all_present_labels = set()
    for mdf in members_list:
        for idx in mdf.index:
            label = idx.split('->')[-1]
            all_present_labels.add(label)
    active_groups = [g for g in group_order if g in all_present_labels]

    reorder_index = []
    for group_label in active_groups:
        for src in sources:
            key = f'{src}->{group_label}'
            reorder_index.append(key)

    members_all = pd.concat(members_list,axis=0)
    
    valid_reorder = [idx for idx in reorder_index if idx in members_all.index]
    members_all = members_all.loc[valid_reorder]
    
    members_all = members_all.where(members_all==True, False)
    # Filter to existing indices from reorder_index
    members_all = members_all[sorted(members_all.columns, key=str.casefold)]

    members_all = members_all.loc[:, members_all.sum(axis=0) > 0]
    # Plotting Logic — auto-size based on data
    n_rows = len(members_all)
    n_cols = len(members_all.columns)
    figsize = (max(4, 0.25 * n_cols + 2), max(4, 0.35 * n_rows + 2))

    logger.info("Generating plot...")
    fig, ax = plt.subplots(1,2,figsize=figsize,width_ratios=[0.8,0.2],sharey='all')
    plt.subplots_adjust(wspace=0.05)
    
    # Build c_list dynamically: one color per group, 4 sources per group
    group_colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']
    n_sources = len(sources)
    c_list = []
    for i, _ in enumerate(active_groups):
        c_list.extend([group_colors[i % len(group_colors)]] * n_sources)
    # Extend if more rows than expected
    if len(members_all) > len(c_list):
        c_list = (c_list * (len(members_all)//len(c_list) + 1))[:len(members_all)]


    try:
        plot_grid(members_all, color_dotline=c_list, ax=ax[0], zorder=10)
    except Exception as e:
        logger.error(f"Error plotting grid: {e}")
        return

    grouping_label = f', Grouping: {os.path.basename(args.clip_dir)}' if args.clip_dir != 'clip_scores' else ''
    ax[0].set_title(f'Top cliques\n(Backbone:{f"Convolutional" if BACKBONE=="Conv" else "Transformer"},\nMetrics={metric}{grouping_label})')
    
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

    model_suffix = f'_{"_".join(args.model)}' if args.model is not None else ''
    clip_suffix = f'_{os.path.basename(args.clip_dir)}' if args.clip_dir != 'clip_scores' else ''
    out_filename = f'top_cliques_{BACKBONE}_{MCD_flag}_{args.metric_group}{model_suffix}{clip_suffix}'
    out_path = os.path.join(OUTDIR, out_filename)
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    fig.savefig(out_path + '.jpeg', bbox_inches='tight')
    logger.success(f"Saved plot to: {out_path}")

    # Export top cliques as JSON for use by multinomial_analysis.py
    # Format: {source: {group_name: [method1, method2, ...]}}
    cliques_export: dict[str, dict[str, list[str]]] = {}
    for idx in members_all.index:
        source, group_name = idx.split('->')
        methods = [col for col in members_all.columns if members_all.loc[idx, col]]
        cliques_export.setdefault(source, {})[group_name] = methods

    # Add "all" cliques (pooled OOD groups) — not shown in plot but needed by multinomial_analysis.py
    for source_ in sources:
        df_src = df_combined[df_combined['source'] == source_].copy()
        if BACKBONE == 'ViT':
            df_src = df_src[df_src['methods'] != 'Confidence']
        df_src_met = df_src[(df_src['metric'] == metric[0]) | (df_src['metric'] == metric[1])].copy()
        ood_groups = [g for g in df_src_met['group'].unique() if g != '0']
        if not ood_groups:
            continue
        sub_all = df_src_met[df_src_met['group'].isin(ood_groups)].copy()
        sub_all["block"] = sub_all[blocks].astype(str).agg("|".join, axis=1)
        try:
            stat, p, pivot = friedman_blocked(sub_all, entity_col="methods", block_col="block", value_col="score_std")
            if isinstance(stat, float) and not math.isnan(stat):
                ph = conover_posthoc_from_pivot(pivot)
                ranks_ = pivot.rank(axis=1, ascending=False)
                avg_ranks_ = ranks_.mean(axis=0).sort_values()
                all_cliques = maximal_cliques_from_pmatrix(ph, alpha)
                scored = rank_cliques(all_cliques, list(avg_ranks_.index), avg_ranks_)
                layers = greedy_exclusive_layers(scored)
                if layers:
                    cliques_export.setdefault(source_, {})["all"] = layers[0]["members"]
        except Exception:
            pass

    cliques_path = out_path + '_cliques.json'
    with open(cliques_path, 'w') as f:
        json.dump(cliques_export, f, indent=2)
    logger.success(f"Saved cliques JSON to: {cliques_path}")

if __name__ == "__main__":
    main()

#%%
import os, re, math, json, itertools, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import matplotlib.lines as mlines
# from upsetplot import generate_counts, plot
# from upsetplot import from_contents, from_memberships
# from upsetplot import UpSet
from matplotlib.patches import Rectangle
from src.utils_stats import *
#%%
# Set Default Environment Variables if not present
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')
#%%
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)
#%%
# SOURCE = 'tinyimagenet'
clip_names = {'Unnamed: 0_level_1':'dataset', 
                'Unnamed: 5_level_1':'group'}
distance_dict = {'0':'test','1':'near','2':'mid','3':'far'}
#%%
df_all = []

MCD_flag = 'False'
BACKBONE = 'Conv'
# =========================
# CONFIG: edit as needed
# =========================
sources = ['cifar10', 'supercifar100', 'cifar100', 'tinyimagenet']
for SOURCE in sources:
    CONFIG = {
        # Map metric -> CSV path. You can point these to your real files per metric
        "FPR@95TPR": f"scores/scores_all_FPR@95TPR_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
        "AUROC_f":   f"scores/scores_all_AUROC_f_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
        "AUGRC":     f"scores/scores_all_AUGRC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
        "AURC":      f"scores/scores_all_AURC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
        # Optional CLIP distances / groupings file (columns: dataset, features..., e.g., 'group', 'clip_dist_id_ood', etc.)
        "CLIP_FILE": f"clip_scores/clip_distances_{SOURCE}.csv",  # set to None if not available
        # Output dir
        "OUTDIR": f"ood_eval_outputs/{SOURCE}_{BACKBONE}_MCD-{MCD_flag}",
        # Alpha for significance
        "ALPHA": 0.05,
        # Bootstraps for CIs
        "N_BOOT": 2000
    }

    os.makedirs(CONFIG["OUTDIR"], exist_ok=True)
    alpha = CONFIG["ALPHA"]
    df = load_all_scores(CONFIG)
    df['source'] = SOURCE

    clip = pd.read_csv(CONFIG["CLIP_FILE"],header=[0,1])
    clip.columns = clip.columns.droplevel(0)
    clip = clip.rename(clip_names,axis='columns')
    # Expect columns: dataset, group (optional), and numeric features like clip_dist_to_ID, etc.
    # Keep only columns present
    numeric_features = [c for c in clip.columns if c not in ["dataset", "group"]]
    if len(numeric_features) == 0:
        print("CLIP file found but no numeric features detected; skipping LODO.")
    else:
        # Stability: variance within vs between clusters if 'group' present
        if "group" in clip.columns:
            merged = df.merge(clip[["dataset", "group"]], on="dataset", how="left")
    df_all.append(choose_baseline_rows(merged))
    # df.to_csv(os.path.join(CONFIG["OUTDIR"], f"all_scores_long_MCD-{MCD_flag}_{SOURCE}.csv"), index=False)
#%%
metric = ['AUGRC','AURC']
rank_group = ["dataset", 'model', "metric", 'group']
blocks = ['dataset',"model",'metric']
#%%
# source_ ='cifar10'
df_ = pd.concat(df_all,axis=0)
#%%
# df = df[df['is_baseline']]
# if BACKBONE=='ViT':
#     df_ = df_[df_['methods']!='Confidence']
# df_ = df_[df_['source']==source_].copy()
# df_["rank"] = df_.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
# avg_rank = df_.groupby(["methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
# df_met = df_[(df_['metric']==metric[0])|(df_['metric']==metric[1])].copy()
# rank_group = ["dataset", 'model', "metric", 'group']
# Average rank per method
# %%
# significance_level = 0.05
# metric = ['AURC','AUGRC']
# metric = ['AUROC_f','FPR@95TPR']
# source_ = 'tinyimagenet'

#%%
members_list = []
for source_ in sources:
    df_ = pd.concat(df_all,axis=0)
    # df = df[df['is_baseline']]
    if BACKBONE=='ViT':
        df_ = df_[df_['methods']!='Confidence']

    df_ = df_[df_['source']==source_].copy()
    df_["rank"] = df_.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
    # Average rank per method
    avg_rank = df_.groupby(["methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
    # Average rank per method per metric
    df_met = df_[(df_['metric']==metric[0])|(df_['metric']==metric[1])].copy()
    df_met["rank"] = df_met.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
    avg_rank_group_metric = df_met.groupby(['group','metric',"methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
    parsed = avg_rank_group_metric["methods"].apply(parse_method_variation)
    avg_rank_group_metric["method_base"] = parsed.apply(lambda x: x[0])
    avg_rank_group_metric["variation"] = parsed.apply(lambda x: x[1])
    # top_K = 22
    
    
    # blocks = ['dataset',"model"]
    friedman_results = []
    layered_cliques = {}
    layered_ranks = []
    # for (metric, dataset), g in df_met.groupby(['dataset']):
    for dataset, g in df_met.groupby('group'):
    # for metric, g in df_met.groupby(["metric",'dataset']):
        # print(metric,dataset)
        sub = g.copy()
        sub["block"] = sub[blocks].astype(str).agg("|".join, axis=1)
        stat, p, pivot = friedman_blocked(sub, entity_col="methods", block_col="block", value_col="score_std")
        friedman_results.append({"metric": metric, "friedman_stat": stat, "p": p, "n_blocks": pivot.shape[0], "n_methods": pivot.shape[1]})
        # Posthoc Nemenyi
        if isinstance(stat, float) and not math.isnan(stat):
            ph = conover_posthoc_from_pivot(pivot)
            ranks_ = pivot.rank(axis=1, ascending=False)
            avg_ranks_ = ranks_.mean(axis=0).sort_values()
            avg_ranks_.name = dataset
            cliques = maximal_cliques_from_pmatrix(ph,alpha)
            scored = rank_cliques(cliques, list(avg_ranks_.index), avg_ranks_)
            layers = greedy_exclusive_layers(scored)   # disjoint “Top-1, Top-2, …” layers
            layered_cliques.update({f'{dataset}':layers})
            layered_ranks.append(avg_ranks_)
            # break
            # # ph.to_csv(os.path.join(CONFIG["OUTDIR"], f"Q3_posthoc_nemenyi_{metric}.csv"))
            # top_avg_ranks = avg_rank_dataset_metric[avg_rank_dataset_metric['dataset']==f'{dataset}'][['methods','method_base','avg_rank']]
            # # break
            # top_avg_ranks_best_method = pd.concat([top_avg_ranks[top_avg_ranks['method_base']==y].sort_values(by='avg_rank').iloc[[0],:] for y in sorted(list(set(df_met['method_base'])))])
            # top_avg_ranks_best_method = top_avg_ranks_best_method.set_index('methods').iloc[:,1].sort_values()
            # top_index = top_avg_ranks_best_method.index
            # best = avg_ranks_.index[0]
            # # top_clique = pd.Series([m for m in avg_ranks_.index if (m==best) or (ph.loc[m, best] >= significance_level)], name=dataset)
            # # cliques.append(top_clique)
            # # print(dataset, top_clique)
            # if len(top_index)>top_K:
            #     top_index = top_index[:top_K]
            #     top_avg_ranks_best_method = top_avg_ranks_best_method[:top_K]
            # fig,ax = plt.subplots(1,1,figsize=(6,3))
            # sp.critical_difference_diagram(top_avg_ranks_best_method, ph.loc[top_index,top_index],
            #                                 # label_props={'color': 'black', 'fontweight': 'bold'},
            #                                 alpha = significance_level,
            #                                 crossbar_props={'color': None, 'marker': 'o'},
            #                                 elbow_props={'color': 'black'},
            #                                 ax=ax)
            # ax.set_title(f'Critical Difference Diagram ({dataset})')
    clique_members = []
    clique_avg = []
    datasets_order = ['0','1','2','3']
    for c in datasets_order:
        clique_members.append( layered_cliques[c][0]['members'] )
        clique_avg.append(layered_cliques[c][0]['mean_rank'])
    member_df = pd.DataFrame([{name: True for name in names} for names in clique_members])
    member_df = member_df.where(member_df==True, False)
    member_df.index = [source_+'->'+distance_dict[d] for d in datasets_order]
    members_list.append(member_df)
#%%
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
#%%
members_all = pd.concat(members_list,axis=0)
members_all = members_all.where(members_all==True, False)
# members_all = members_all[sorted(members_all.columns)]
members_all = members_all.loc[reorder_index]
#%%
if metric == ['AUROC_f','FPR@95TPR'] and BACKBONE=='ViT':
    figsize = (5,8)
    cols_show = [x for x in members_all.columns if 'class' not in x]
    cols_show.extend(['MLS class pred','MSR class pred','PCA RecError class pred','PCE class pred'])
    members_all = members_all[cols_show]
    members_all = members_all[sorted(members_all.columns)]
    rows_shading = [(0,1),(4,4),(7,8),(11,12),(14,16),(20,20),(23,23),(26,28),(31,32),(34,34),(37,38)]
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
elif metric == ['AUGRC','AURC'] and BACKBONE=='ViT':
    figsize = (5,8)
    cols_show = [x for x in members_all.columns if 'class' not in x]
    cols_show.extend(['GradNorm class pred','KPCA RecError class pred','MLS class'])
    members_all = members_all[cols_show]
    members_all = members_all[sorted(members_all.columns)]
    rows_shading = [(0,1),(4,4),(7,8),(11,13),(16,18),(21,21),(24,24),(26,27),(30,31),(33,33),(36,37)]
    # rows_shading = [(0,1),(7,7),(13,17),(21,24),(29,33),(39,40),(45,45),(50,54),(60,62),(64,64),(68,71)]
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

elif metric == ['AUROC_f','FPR@95TPR'] and BACKBONE=='Conv': 
    figsize = (5,8)
    cols_show = [x for x in members_all.columns if 'class' not in x]
    members_all = members_all[cols_show]
    members_all = members_all[sorted(members_all.columns)]
    rows_shading = [(0,1),(4,4),(6,7),(10,11),(14,15),(18,18),(21,22),(25,26)]
    members_all = members_all[sorted(members_all.columns)]
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
                # [(11.5, 5.5),1,2,'dotted','black','plum'],
                # [(3.5, 13.5),3,2,'dotted','black','plum'],
                # [(7.5, 13.5),2,2,'dotted','black','plum'],
                # [(11.5, 13.5),1,2,'dotted','black','plum'],
                # [(3.5, 17.5),3,2,'dotted','black','plum'],
                # [(7.5, 17.5),2,2,'dotted','black','plum'],
                # [(11.5, 17.5),1,2,'dotted','black','plum'],
                ]
elif metric == ['AUGRC','AURC'] and BACKBONE=='Conv':
    figsize = (5,8)
    cols_show = [x for x in members_all.columns if 'class' not in x]
    members_all = members_all[cols_show]
    members_all = members_all[sorted(members_all.columns)]
    rows_shading = [(0,1),(4,4),(6,7),(10,11),(14,15),(18,19),(21,22),(25,26),(29,29)]
    # rows_shading = [(0,4),(10,10),(12,16),(22,25),(29,33),(38,38),(43,43),(46,50),(56,60),(65,66)]
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
    # rects_list = [[(6.5, -0.5),1,5,'dotted','black','skyblue'],
    #             [(10.5, -0.5),1,5,'dotted','black','skyblue'],
    #             [(14.5, -0.5),1,5,'dotted','black','skyblue'],
    #             [(2.5, -0.5),1,5,'dotted','black','skyblue'],
    #             [(-0.5, 21.5),4,4,'dotted','black','wheat'],
    #             [(3.5, 11.5),3,5,'dotted','black','plum'],
    #             [(7.5, 11.5),2,5,'dotted','black','plum'],
    #             [(11.5, 11.5),1,5,'dotted','black','plum'],
    #             [(3.5, 28.5),3,5,'dotted','black','plum'],
    #             [(7.5, 28.5),2,5,'dotted','black','plum'],
    #             [(11.5, 28.5),1,5,'dotted','black','plum'],
    #             [(3.5, 38.5),3,4,'dotted','black','plum'],
    #             [(7.5, 38.5),2,4,'dotted','black','plum'],
    #             [(11.5, 38.5),1,4,'dotted','black','plum'],
    #             ]

fig, ax = plt.subplots(1,2,figsize=figsize,width_ratios=[0.8,0.2],sharey='all',)
plt.subplots_adjust(wspace=0.05)
# c_list = ['tab:green','tab:blue','tab:red','tab:orange',
#             'tab:green','tab:blue','tab:red','tab:orange',
#             'tab:green','tab:blue','tab:red','tab:orange',
#             'tab:green','tab:blue','tab:red','tab:orange',]
c_list = ['tab:green','tab:green','tab:green','tab:green',
            'tab:blue','tab:blue','tab:blue','tab:blue',
            'tab:red','tab:red','tab:red','tab:red',
            'tab:orange','tab:orange','tab:orange','tab:orange',
            ]
# c_list = ['tab:green','tab:green','tab:green',
#             'tab:blue','tab:blue','tab:blue',
#             'tab:red','tab:red','tab:red',
#             'tab:orange','tab:orange','tab:orange',
#             ]
plot_grid(members_all, color_dotline=c_list, ax=ax[0], zorder=10)
# t = np.arange(-1.0, 17, 0.01)

for (y1,y2) in rows_shading:
    # ax.fill_between(t, -.5+y1,.5+y2,  facecolor='gray', alpha=.25)
    ax[0].axhspan(-.5+y1, .5+y2, facecolor='lightgray', alpha=0.5)
ax[0].set_title(f'Top cliques\n(Backbone:{BACKBONE}, Metrics={metric})')
sns.barplot(x=members_all.sum(axis=0), y=members_all.columns, color='gray', ax=ax[1])
# sns.countplot(x=members_all.sum(axis=0), y=members_all.columns, ax=ax[1])
ax[1].bar_label(ax[1].containers[0])
ax[1].xaxis.set_visible(False)
ax[1].yaxis.set_visible(False)
for x in ["top", "bottom", "right"]:
    ax[1].spines[x].set_visible(False)

# Define the properties of the rectangle
# xy = (6.5, -0.5)  # Lower-left corner (x, y)
# width = 1
# height = 5
# ls = 'dotted'
# lcolor = 'black'
for xy,width,height,ls,lcolor,bcolor in rects_list:
    # Create the Rectangle patch
    # Set 'fill=False' to only draw the border, not fill the rectangle
    # Set 'linestyle='dotted'' for a dotted line
    rect = Rectangle(xy, width, height,
                    linewidth=2,  # Line thickness
                    edgecolor=lcolor,  # Line color
                    facecolor=bcolor,  # No fill color
                    linestyle=ls,
                    alpha=0.5)
    ax[0].add_patch(rect)
fig.savefig(f'top_cliques_{BACKBONE}_{MCD_flag}_{metric}.pdf', bbox_inches='tight')


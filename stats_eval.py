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
        "CLIP_FILE": f"scores/clip_distances_{SOURCE}.csv",  # set to None if not available
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
source_ ='cifar10'
df_ = pd.concat(df_all,axis=0)
# df = df[df['is_baseline']]
if BACKBONE=='ViT':
    df_ = df_[df_['methods']!='Confidence']
df_ = df_[df_['source']==source_].copy()
df_["rank"] = df_.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
avg_rank = df_.groupby(["methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
df_met = df_[(df_['metric']==metric[0])|(df_['metric']==metric[1])].copy()
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
# Add the rectangle to the axes
#%%
# Get all y-axis tick labels
# ytick_labels = ax.get_yticklabels()
# # Define a bbox style for the background color
# bbox_style = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", lw=1, alpha=0.7)
# # Apply the background color to specific labels (e.g., 'Label B' and 'Label D')
# for i, label in enumerate(y_values):
#     if label == 'Label B' or label == 'Label D':
#         ytick_labels[i].set_bbox(bbox_style)
# ax.fill_betwwen()
#%%
# example = example.to_frame(name='value')
# ax_dict = UpSet(example,sort_by='input',show_counts=True,subset_size='count',).plot()
# plot_result = plot(example)
# ax_dict['intersections'].clear()
# # ax_dict['intersections'].xaxis.set_visible(True)
# ax_dict['intersections'].scatter(range(len(clique_avg)),clique_avg, c='k')
# ax_dict["intersections"].set_ylabel("Average rank")
# ax_dict["intersections"].grid()
# # ax_dict['intersections'].set_xticks(range(len(datasets_order)))
# # ax_dict['intersections'].set_xticklabels([]) 

# ax_dict['matrix'].xaxis.set_visible(True)
# ax_dict['matrix'].set_xlabel('Datasets')
# ax_dict['matrix'].set_xticks(range(len(datasets_order)))
# ax_dict['matrix'].set_xticklabels(datasets_order, rotation=90, ha='right') 
# # plot_result["totals"].set_xlabel("Frequency")
# plt.show()
#%%
# def ns_adjacency(avg_ranks: pd.Series, p_adj: pd.DataFrame, alpha=0.05, use_cd=False, CD=None):
#     methods = avg_ranks.index.tolist()
#     A = (p_adj.loc[methods, methods].values >= alpha).astype(int)
#     np.fill_diagonal(A, 1)
#     keep = np.ones(len(methods), dtype=bool)
#     if use_cd and CD is not None:
#         rmin = avg_ranks.min()
#         keep = (avg_ranks.values <= rmin + CD)
#     return methods, A, keep

# # Bron–Kerbosch with pivot to enumerate maximal cliques
# def bron_kerbosch_pivot(R, P, X, neigh):
#     if not np.any(P) and not np.any(X):
#         yield R.copy(); return
#     # choose pivot u in P∪X with max degree
#     UX = np.where(P | X)[0]
#     u = UX[np.argmax(neigh[UX].sum(axis=1))]
#     for v in np.where(P & ~neigh[u])[0]:
#         R_new = R | {v}
#         P_new = P & neigh[v]
#         X_new = X & neigh[v]
#         yield from bron_kerbosch_pivot(R_new, P_new, X_new, neigh)
#         P[v] = False
#         X[v] = True

# def maximal_cliques(A):
#     n = A.shape[0]
#     neigh = A.astype(bool)
#     P = np.ones(n, dtype=bool); X = np.zeros(n, dtype=bool)
#     clqs = []
#     for C in bron_kerbosch_pivot(set(), P.copy(), X.copy(), neigh):
#         clqs.append(sorted(C))
#     # dedupe
#     out = []
#     seen = set()
#     for C in clqs:
#         key = tuple(C)
#         if key not in seen:
#             out.append(C); seen.add(key)
#     return out

# %%
# use_cd = False  # set True to restrict by CD band
# CD = None
# methods, A, keep_mask = ns_adjacency(avg_ranks_, ph, alpha=significance_level, use_cd=use_cd, CD=CD)
# A_sub = A[np.ix_(keep_mask, keep_mask)]
# meth_sub = [m for m,k in zip(methods, keep_mask) if k]

# cliques = maximal_cliques(A_sub)
# scored = rank_cliques(cliques, meth_sub, avg_ranks_)

# top_L = scored[:5]                         # multiple (overlapping) top cliques
# layers = greedy_exclusive_layers(scored)   # disjoint “Top-1, Top-2, …” layers
#%%
# cliques = maximal_cliques_from_pmatrix(ph,significance_level)
# scored = rank_cliques(cliques, meth_sub, avg_ranks_)
# layers = greedy_exclusive_layers(scored)   # disjoint “Top-1, Top-2, …” layers
# %%
import numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def build_ns_graph(p_adj: pd.DataFrame, avg_ranks: pd.Series, alpha=0.05):
    meth = avg_ranks.index.tolist()
    P = p_adj.loc[meth, meth].copy()
    P = P.combine(P.T, np.maximum)                    # symmetrize
    A = (P.values >= alpha) & ~np.eye(len(meth), dtype=bool)
    G = nx.Graph()
    for m in meth:
        G.add_node(m, rank=float(avg_ranks[m]))
    for i in range(len(meth)):
        for j in range(i+1, len(meth)):
            if A[i, j]:
                # map p to an edge weight/width (stronger when more non-sig)
                w = float((P.iat[i, j] - alpha) / max(1e-9, 1.0 - alpha))
                G.add_edge(meth[i], meth[j], p=float(P.iat[i, j]), weight=w)
    return G

def choose_primary_clique(cliques, avg_ranks):
    # pick, for each node, the clique with best (lowest) mean rank
    best_for = {}
    for C in cliques:
        mr = avg_ranks.loc[C].mean()
        for m in C:
            cur = best_for.get(m, (np.inf, None))
            if mr < cur[0]:
                best_for[m] = (mr, tuple(C))
    return {m: c for m, (_, c) in best_for.items()}

def draw_ns_clique_network(p_adj, avg_ranks, cliques, alpha=0.05,
                           k=None, hulls=True, out_png="ns_clique_network.png"):
    G = build_ns_graph(p_adj, avg_ranks, alpha=alpha)
    if k is None: k = 0.8/np.sqrt(max(1, G.number_of_nodes()))  # spring scale
    pos = nx.spring_layout(G, seed=0, k=k, weight="weight")

    # node sizes (better rank -> larger)
    r = pd.Series({n: G.nodes[n]["rank"] for n in G.nodes()})
    r_min, r_max = r.min(), r.max()
    size = ((r_max - r) + 1.0) / ((r_max - r_min) + 1.0)
    node_sizes = (300 + 1200*size).values

    # assign a primary clique per node
    primary = choose_primary_clique(cliques, avg_ranks)
    # palette
    uniq_cliques = list({tuple(sorted(C)) for C in cliques})
    cmap = plt.cm.tab20
    colors = {}
    for i, C in enumerate(uniq_cliques):
        for m in C:
            # if node in several cliques, keep the best (already chosen)
            pass
    for m in G.nodes():
        C = primary.get(m, None)
        idx = uniq_cliques.index(tuple(sorted(C))) if C else -1
        colors[m] = cmap(idx % 20) if idx >= 0 else (0.7,0.7,0.7,1.0)

    # edges width by “non-sig strength”
    widths = [1.0 + 4.0*G.edges[e]["weight"] for e in G.edges()]

    # draw
    plt.figure(figsize=(9, 7))
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35)
    nx.draw_networkx_nodes(G, pos,
                           node_color=[colors[n] for n in G.nodes()],
                           node_size=node_sizes, edgecolors="black", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, font_size=9)

    # optional convex hulls per clique
    if hulls:
        for C in uniq_cliques:
            pts = np.array([pos[m] for m in C if m in pos])
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                plt.fill(poly[:,0], poly[:,1], alpha=0.08)

    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[OK] wrote {out_png}")

# %%
draw_ns_clique_network(ph, avg_ranks_, cliques, alpha=0.05)
# %%
def plot_layered_cd(avg_ranks: pd.Series, layers, title="Layered cliques (α=0.05)"):
    avg_ranks = avg_ranks.sort_values()
    y0 = 0.0
    plt.figure(figsize=(10, 2 + 0.35*len(layers)))

    # draw method ticks
    for m, x in avg_ranks.items():
        plt.plot([x, x], [0, 0.05], color="k", lw=1)
        plt.text(x, 0.09, m, rotation=90, ha="right", va="bottom", fontsize=8)

    # layered bars beneath the axis
    for i, layer in enumerate(layers, 1):
        members = layer["members"]
        xs = avg_ranks.loc[[m for m in members if m in avg_ranks.index]]
        if xs.empty: 
            continue
        x_min, x_max = xs.min(), xs.max()
        y = -(0.25*i)
        plt.hlines(y, x_min, x_max, lw=6, alpha=0.5)
        plt.text((x_min+x_max)/2, y-0.07, f"Layer {i}: {', '.join(members)}",
                 ha="center", va="top", fontsize=8)

    plt.axhline(0, color="k", lw=1)
    plt.yticks([])
    plt.xlabel("Average rank (lower is better)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
# %%
import numpy as np, pandas as pd, networkx as nx, matplotlib.pyplot as plt

def draw_layered_ns_network(p_adj, avg_ranks, layers, alpha=0.05,
                            show_cross=False, out_png="layered_cliques.png"):
    # Build non-significance graph (edge iff adjusted p >= alpha)
    methods = avg_ranks.index.tolist()
    P = p_adj.loc[methods, methods].copy().combine(p_adj.T, np.maximum)  # symmetrize
    A = (P.values >= alpha) & ~np.eye(len(methods), dtype=bool)

    G = nx.Graph()
    for m in methods:
        G.add_node(m, rank=float(avg_ranks[m]))
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            if A[i, j]:
                G.add_edge(methods[i], methods[j], p=float(P.iat[i, j]))

    # Map node -> layer (0 = unassigned)
    layer_of = {m: 0 for m in methods}
    for i, L in enumerate(layers, 1):
        for m in L["members"]:
            if m in layer_of: layer_of[m] = i
    Lmax = max(layer_of.values()) if layer_of else 0

    # Coordinates: x = average rank (left = better), y = -layer (downward)
    r = avg_ranks.astype(float)
    rng = np.random.default_rng(0)
    pos = {m: (float(r[m]), -layer_of[m] - 0.05*rng.normal())
           for m in methods}

    # Edge sets
    intra = [(u,v) for u,v in G.edges() if layer_of[u]==layer_of[v] and layer_of[u]>0]
    cross = [(u,v) for u,v in G.edges() if (layer_of[u]!=layer_of[v]) or (layer_of[u]==0 or layer_of[v]==0)]

    plt.figure(figsize=(10, 1.5 + 0.6*max(1, Lmax)))
    if show_cross:
        nx.draw_networkx_edges(G, pos, edgelist=cross, alpha=0.08, width=0.6, style="dotted", edge_color="gray")
    nx.draw_networkx_edges(G, pos, edgelist=intra, alpha=0.35, width=1.6, edge_color="gray")

    # Node color/size by layer & rank
    cmap = plt.cm.tab10
    sizes = [(700*(r.max()-r[n]+1)/(r.max()-r.min()+1e-9) + 220) for n in G.nodes()]
    colors = [cmap((layer_of[n]-1) % 10) if layer_of[n]>0 else (0.8,0.8,0.8,1) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Layer bars (span min–max rank within each layer)
    for i, L in enumerate(layers, 1):
        xs = [r[m] for m in L["members"] if m in r.index]
        if xs:
            y = -i
            plt.hlines(y, min(xs), max(xs), colors=cmap((i-1)%10), linewidth=6, alpha=0.15)

    plt.yticks([-i for i in range(0, Lmax+1)], ["unassigned"] + [f"Layer {i}" for i in range(1, Lmax+1)])
    plt.xlabel("Average rank (lower is better)"); plt.title("Non-significance network with layered cliques")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.show()

# %%
draw_layered_ns_network(ph, avg_ranks_, layers)
# %%
import pandas as pd
import numpy as np

def sankey_from_layers(avg_ranks: pd.Series, layers, weight="inverse_rank"):
    """
    avg_ranks: pd.Series index=method, value=avg rank (lower=better)
    layers:    list of dicts from greedy_exclusive_layers, each with "members"
    weight:    "inverse_rank" | "uniform"
    Returns node labels and link arrays for a Sankey.
    """
    # nodes: layers first, then methods (disjoint by construction)
    layer_labels = [f"Layer {i} (n={len(L['members'])})" for i, L in enumerate(layers, 1)]
    methods = []
    for L in layers:
        for m in L["members"]:
            if m not in methods: methods.append(m)

    node_labels = layer_labels + methods
    idx_layer  = {i: i for i in range(len(layer_labels))}
    idx_method = {m: len(layer_labels)+j for j, m in enumerate(methods)}

    # link values
    def w(m):
        if weight == "uniform":  return 1.0
        v = float(avg_ranks.get(m, np.nan))
        return 0.0 if np.isnan(v) else 1.0 / max(v, 1e-9)

    sources, targets, values = [], [], []
    for i, L in enumerate(layers, 0):  # 0-based for source indices
        for m in L["members"]:
            if m in idx_method:
                sources.append(idx_layer[i])
                targets.append(idx_method[m])
                values.append(w(m))

    return node_labels, sources, targets, values

# --- Plotly Sankey (interactive HTML) ---
def plot_sankey_layers(avg_ranks, layers, out_html="layered_cliques_sankey.html", weight="inverse_rank"):
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError("plotly is required for this Sankey") from e

    labels, s, t, v = sankey_from_layers(avg_ranks, layers, weight=weight)
    fig = go.Figure(go.Sankey(
        node=dict(label=labels),
        link=dict(source=s, target=t, value=v)
    ))
    fig.update_layout(title="Layered cliques → methods (width ∝ weight)", font_size=12)
    fig.show()
    # fig.write_html(out_html)
    # print(f"[OK] wrote {out_html}")

# %%
def layer_map(layers):
    d = {}
    for i, L in enumerate(layers, 1):
        for m in L["members"]:
            d[m] = i
    return d
#%%
def plot_sankey_layer_to_layer(avg_ranks, layers_A, layers_B, out_html="layers_A_to_B.html"):
    import plotly.graph_objects as go
    mapA, mapB = layer_map(layers_A), layer_map(layers_B)
    methods = sorted(set(mapA) | set(mapB))

    nodes_A = [f"A:Layer {i}" for i in range(1, 1+len(layers_A))]
    nodes_B = [f"B:Layer {i}" for i in range(1, 1+len(layers_B))]
    labels  = nodes_A + nodes_B
    offsetB = len(nodes_A)

    # aggregate flows per (layerA, layerB)
    from collections import Counter
    flows = Counter()
    for m in methods:
        a = mapA.get(m, None); b = mapB.get(m, None)
        if a is not None and b is not None:
            flows[(a, b)] += 1  # or use 1/avg_ranks[m] for weighting

    sources, targets, values = [], [], []
    for (a, b), c in sorted(flows.items()):
        sources.append(a-1)
        targets.append(offsetB + b-1)
        values.append(c)

    fig = go.Figure(go.Sankey(node=dict(label=labels),
                              link=dict(source=sources, target=targets, value=values)))
    fig.update_layout(title="Layer membership shifts (A → B)", font_size=12)
    fig.show()
# %%
# Multi-layering Sankey: methods → Layering A → Layering B → Layering C ...
# layers_dict: {"A": layers_A, "B": layers_B, ...} where each layers_X is the
#              output of greedy_exclusive_layers(scored): a list of dicts with key "members"
# ranks_dict:  optional dict {"A": avg_ranks_A (pd.Series), "B": avg_ranks_B, ...}
#              used for link weights (inverse rank). If omitted, uses uniform weight=1.
def plot_multi_layering_sankey(layers_dict, ranks_dict=None,
                               weight="uniform", out_html="methods_to_layerings.html"):
    import numpy as np, pandas as pd
    import plotly.graph_objects as go

    # ---- collect nodes ----
    # Methods are the union over all layerings’ members
    all_methods = []
    for L in layers_dict.values():
        for d in L:
            for m in d["members"]:
                if m not in all_methods:
                    all_methods.append(m)
    all_methods = sorted(all_methods)

    # Build layer-node labels per layering (Layer 1, Layer 2, …), keep index maps
    layerings = list(layers_dict.keys())
    layer_labels = []
    for key in layerings:
        L = layers_dict[key]
        labels = [f"{key}:Layer {i} (n={len(L[i-1]['members'])})" for i in range(1, len(L)+1)]
        layer_labels.append(labels)

    # Node indexing: [methods] + [A layers] + [B layers] + ...
    node_labels = list(all_methods)
    layer_offsets = {}
    offset = len(node_labels)
    for key, labels in zip(layerings, layer_labels):
        layer_offsets[key] = offset
        node_labels.extend(labels)
        offset += len(labels)

    # ---- helper: method → layer index for a given layering
    def method_to_layer_idx(method, layering_key):
        L = layers_dict[layering_key]
        for i, d in enumerate(L):  # i = 0..L-1
            if method in d["members"]:
                return layer_offsets[layering_key] + i
        return None  # method not present in this layering (shouldn't happen if union is correct)

    # ---- link weights
    def link_weight(method, layering_key):
        if weight == "uniform" or ranks_dict is None or layering_key not in ranks_dict:
            return 1.0
        r = ranks_dict[layering_key]
        v = float(r.get(method, np.nan))
        return 0.0 if np.isnan(v) else 1.0 / max(v, 1e-9)

    # ---- build links: methods → each layering column
    sources, targets, values, colors = [], [], [], []
    # color per layering
    palette = ["rgba(31,119,180,0.35)","rgba(255,127,14,0.35)","rgba(44,160,44,0.35)",
               "rgba(214,39,40,0.35)","rgba(148,103,189,0.35)","rgba(140,86,75,0.35)"]
    color_map = {key: palette[i % len(palette)] for i, key in enumerate(layerings)}

    for m_idx, m in enumerate(all_methods):
        for key in layerings:
            t_idx = method_to_layer_idx(m, key)
            if t_idx is None:
                continue
            sources.append(m_idx)
            targets.append(t_idx)
            values.append(link_weight(m, key))
            colors.append(color_map[key])

    # ---- optional: fix x/y positions to create columns (methods at x=0, each layering spaced to the right)
    n_nodes = len(node_labels)
    x = np.zeros(n_nodes); y = np.zeros(n_nodes)
    # methods column x=0.0, y spread evenly
    for i in range(len(all_methods)):
        x[i] = 0.0
        y[i] = (i + 0.5) / len(all_methods)

    # layer columns: evenly spaced from 0.4 → 0.95
    if layerings:
        xs = np.linspace(0.4, 0.95, num=len(layerings))
        for col, key in enumerate(layerings):
            start = layer_offsets[key]
            count = len(layers_dict[key])
            for j in range(count):
                x[start + j] = xs[col]
                y[start + j] = (j + 0.5) / count

    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=node_labels,
            x=x, y=y,
            pad=10, thickness=14,
            color=["rgba(200,200,200,0.9)"]*len(all_methods) +  # methods
                  [color_map[key].replace("0.35","0.8") for key, labels in zip(layerings, layer_labels) for _ in labels]
        ),
        link=dict(
            source=sources, target=targets, value=values, color=colors
        )
    ))
    fig.update_layout(title="Methods → Layered cliques across conditions", font_size=12)
    fig.show()
    fig.write_html(out_html)
    print(f"[OK] wrote {out_html}")

# %%
layers_dict = {"A": layered_cliques[0], "B": layered_cliques[1], "C": layered_cliques[2]}

# Optional: average ranks per condition for weighting (pd.Series, lower=better)
ranks_dict = {"A": layered_ranks[0], "B": layered_ranks[1], "C": layered_ranks[2]}

plot_multi_layering_sankey(layers_dict, ranks_dict)
# %%
# methods_to_layerings_sankey.py
import numpy as np
import pandas as pd

def _distinct_colors(n):
    """Generate n distinct hex colors (cycles H in HSV)."""
    import colorsys
    hues = np.linspace(0, 1, num=n, endpoint=False)
    cols = []
    for h in hues:
        r,g,b = colorsys.hsv_to_rgb(h, 0.6, 0.9)
        cols.append("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
    return cols

def _build_method_order(layers_dict, methods):
    """
    Order methods to reduce crossings: sort by their (layer index per layering) tuple, then name.
    Missing → large sentinel so they appear last.
    """
    keys = list(layers_dict.keys())
    # method -> layer idx per layering (1..L, or big sentinel if absent)
    sentinel = 1_000_000
    m2tuple = {}
    for m in methods:
        tup = []
        for key in keys:
            idx = None
            for i, L in enumerate(layers_dict[key], 1):
                if m in L["members"]:
                    idx = i; break
            tup.append(idx if idx is not None else sentinel)
        m2tuple[m] = tuple(tup)
    return sorted(methods, key=lambda x: (m2tuple[x], x))

def plot_methods_to_layerings_sankey(layers_dict, out_html="methods_to_layerings.html",
                                     method_colors=None, link_value=1.0):
    """
    layers_dict: Ordered mapping like {"A": layers_A, "B": layers_B, "C": layers_C, ...}
                 where each layers_X is a list of dicts produced by greedy_exclusive_layers,
                 and each dict has "members": [method1, method2, ...]
    out_html:    output HTML file (interactive)
    method_colors: optional dict {method: "#RRGGBB"}; auto-generated if None
    link_value:  width per method per step (keep 1.0 for uniform thickness)
    """
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise RuntimeError("This function requires plotly>=5") from e

    # ---- collect universe of methods
    all_methods = []
    for Ls in layers_dict.values():
        for L in Ls:
            for m in L["members"]:
                if m not in all_methods:
                    all_methods.append(m)
    # stable, low-crossing order
    all_methods = _build_method_order(layers_dict, all_methods)

    # ---- colors per method
    if method_colors is None:
        palette = _distinct_colors(max(1, len(all_methods)))
        method_colors = {m: palette[i % len(palette)] for i, m in enumerate(all_methods)}

    # ---- node list: methods column + each layering's layer nodes
    node_labels = list(all_methods)
    node_colors = [method_colors[m] for m in all_methods]  # color method nodes by their color
    node_x, node_y = [], []

    # Methods positions (x=0), evenly spaced y
    nM = len(all_methods)
    y_methods = {m: (i + 0.5) / max(1, nM) for i, m in enumerate(all_methods)}
    node_x += [0.0] * nM
    node_y += [y_methods[m] for m in all_methods]

    layer_offsets = {}  # start index of each layering's nodes in node list
    x_cols = np.linspace(0.35, 0.95, num=len(layers_dict))  # column x for each layering

    # Build per-layering layer nodes, y at mean of member methods' y to reduce crossings
    for col, (key, layers) in enumerate(layers_dict.items()):
        layer_offsets[key] = len(node_labels)
        for i, L in enumerate(layers, 1):
            members = [m for m in L["members"] if m in y_methods]
            if members:
                y = float(np.mean([y_methods[m] for m in members]))
            else:
                y = (i - 0.5) / max(1, len(layers))
            label = f"{key}:Layer {i} (n={len(L['members'])})"
            node_labels.append(label)
            # neutral color for layer nodes
            node_colors.append("rgba(180,180,180,0.9)")
            node_x.append(float(x_cols[col]))
            node_y.append(y)

    # ---- links: one link per method per step so color persists
    sources, targets, values, colors, hovers = [], [], [], [], []

    # Helper: method -> layer node index for a given layering key
    def layer_node_index(method, key):
        start = layer_offsets[key]
        for i, L in enumerate(layers_dict[key], 1):
            if method in L["members"]:
                return start + (i - 1)
        return None

    # Step 1: methods -> first layering
    keys = list(layers_dict.keys())
    if not keys:
        raise ValueError("layers_dict is empty.")
    first = keys[0]
    for m_idx, m in enumerate(all_methods):
        t = layer_node_index(m, first)
        if t is None:  # skip if method not in first layering
            continue
        sources.append(m_idx)
        targets.append(t)
        values.append(link_value)
        colors.append(method_colors[m].replace("#", "rgba(") + ",0.55)") if method_colors[m].startswith("#") else method_colors[m]
        hovers.append(f"{m} → {first} L{t - layer_offsets[first] + 1}")

    # Subsequent transitions: Layering k -> Layering k+1, one link per method
    for a, b in zip(keys[:-1], keys[1:]):
        for m in all_methods:
            s = layer_node_index(m, a)
            t = layer_node_index(m, b)
            if s is None or t is None:
                continue
            sources.append(s)
            targets.append(t)
            values.append(link_value)
            # slightly more transparent to distinguish from first hop
            col = method_colors[m]
            rgba = col.replace("#", "rgba(") + ",0.35)" if col.startswith("#") else col
            colors.append(rgba)
            hovers.append(f"{m}: {a} L{s - layer_offsets[a] + 1} → {b} L{t - layer_offsets[b] + 1}")

    # ---- build Sankey (fixed positions so columns stay aligned)
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            label=node_labels,
            color=node_colors,
            x=node_x, y=node_y,
            pad=10, thickness=14
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers
        )
    ))
    fig.update_layout(
        title="Methods → layered cliques across conditions (color persists per method)",
        font_size=12
    )
    fig.write_html(out_html)
    fig.show()
    print(f"[OK] wrote {out_html}")

# %%
plot_methods_to_layerings_sankey(layers_dict)
# %%
# --- color helpers ---
def _hex_to_rgb(hex_str):
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:  # e.g., #abc → #aabbcc
        s = "".join([ch*2 for ch in s])
    if len(s) != 6:
        raise ValueError(f"Bad hex color: {hex_str}")
    r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    return r, g, b

def _to_rgba(color, alpha):
    """Accepts '#rrggbb' | '#rgb' | 'rgb(r,g,b)' | 'rgba(r,g,b,a)' and returns 'rgba(r,g,b,alpha)'."""
    c = color.strip()
    if c.startswith("#"):
        r, g, b = _hex_to_rgb(c)
        return f"rgba({r},{g},{b},{alpha})"
    if c.startswith("rgba"):
        # replace alpha (assumes form rgba(r,g,b,a))
        head, _ = c.rsplit(",", 1)
        return head + f",{alpha})"
    if c.startswith("rgb("):
        # rgb(r,g,b) → rgba(r,g,b,alpha)
        return c.replace("rgb(", "rgba(").rstrip(")") + f",{alpha})"
    # last resort: try a few named colors via matplotlib if available
    try:
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(c)  # floats 0..1
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"
    except Exception:
        # fallback to a neutral gray
        return f"rgba(128,128,128,{alpha})"

# %%
# --- corrected Sankey with persistent method colors across layers ---
import numpy as np

def _distinct_colors(n):
    import colorsys
    hues = np.linspace(0, 1, num=n, endpoint=False)
    return ["#%02x%02x%02x" % tuple(int(c*255) for c in colorsys.hsv_to_rgb(h, 0.6, 0.9))
            for h in hues]

def _build_method_order(layers_dict, methods):
    keys = list(layers_dict.keys()); sentinel = 10**9
    def layer_tuple(m):
        tup = []
        for key in keys:
            pos = sentinel
            for i, L in enumerate(layers_dict[key], 1):
                if m in L["members"]: pos = i; break
            tup.append(pos)
        return tuple(tup)
    return sorted(methods, key=lambda m: (layer_tuple(m), m))

def plot_methods_to_layerings_sankey(layers_dict, out_html="methods_to_layerings.html",
                                     method_colors=None, link_value=1.0):
    import plotly.graph_objects as go

    # collect & order methods
    all_methods = []
    for Ls in layers_dict.values():
        for L in Ls:
            for m in L["members"]:
                if m not in all_methods: all_methods.append(m)
    all_methods = _build_method_order(layers_dict, all_methods)

    # assign method colors (hex is fine for node colors)
    if method_colors is None:
        palette = _distinct_colors(len(all_methods))
        method_colors = {m: palette[i] for i, m in enumerate(all_methods)}

    # nodes: methods + each layering's layers
    node_labels = list(all_methods)
    node_colors = [method_colors[m] for m in all_methods]
    node_x, node_y = [], []

    # methods at x=0
    nM = len(all_methods)
    y_methods = {m: (i + 0.5)/max(1,nM) for i, m in enumerate(all_methods)}
    node_x += [0.0]*nM
    node_y += [y_methods[m] for m in all_methods]

    # layer nodes by column
    layer_offsets = {}
    x_cols = np.linspace(0.35, 0.95, num=len(layers_dict))
    for col, (key, layers) in enumerate(layers_dict.items()):
        layer_offsets[key] = len(node_labels)
        for i, L in enumerate(layers, 1):
            members = [m for m in L["members"] if m in y_methods]
            y = float(np.mean([y_methods[m] for m in members])) if members else (i-0.5)/max(1,len(layers))
            node_labels.append(f"{key}:Layer {i} (n={len(L['members'])})")
            node_colors.append("rgba(180,180,180,0.9)")
            node_x.append(float(x_cols[col]))
            node_y.append(y)

    # helper: index of a layer node for a method in a given layering
    def layer_node_index(method, key):
        start = layer_offsets[key]
        for i, L in enumerate(layers_dict[key], 1):
            if method in L["members"]:
                return start + (i-1)
        return None

    # build links with correct RGBA colors per method
    sources, targets, values, colors, hovers = [], [], [], [], []
    keys = list(layers_dict.keys())
    first = keys[0]

    # methods → first layering
    for m_idx, m in enumerate(all_methods):
        t = layer_node_index(m, first)
        if t is None: continue
        sources.append(m_idx); targets.append(t); values.append(link_value)
        colors.append(_to_rgba(method_colors[m], 0.55))   # FIX: real rgba
        hovers.append(f"{m} → {first} L{t - layer_offsets[first] + 1}")

    # chaining across layerings
    for a, b in zip(keys[:-1], keys[1:]):
        for m in all_methods:
            s = layer_node_index(m, a); t = layer_node_index(m, b)
            if s is None or t is None: continue
            sources.append(s); targets.append(t); values.append(link_value)
            colors.append(_to_rgba(method_colors[m], 0.35))  # FIX: real rgba
            hovers.append(f"{m}: {a} L{s - layer_offsets[a] + 1} → {b} L{t - layer_offsets[b] + 1}")

    import plotly.graph_objects as go
    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(label=node_labels, color=node_colors, x=node_x, y=node_y,
                  pad=10, thickness=14),
        link=dict(source=sources, target=targets, value=values,
                  color=colors, customdata=hovers,
                  hovertemplate="%{customdata}<extra></extra>")
    ))
    fig.update_layout(title="Methods → layered cliques (persistent per-method colors)", font_size=12)
    fig.show()
    fig.write_html(out_html)
    print(f"[OK] wrote {out_html}")

# %%
plot_methods_to_layerings_sankey(layers_dict)
# %%
layered_cliques[0]
# %%
import plotly.graph_objects as go
import colorsys
import numpy as np

def evenly_spaced_colors(n, s=0.62, l=0.52):
    """Return n distinct hex colors using evenly spaced H hues in HLS."""
    cols = []
    for i in range(n):
        h = (i / max(1, n)) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        cols.append('#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)))
    return cols

def hex_to_rgba(hex_color, alpha=0.55):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def build_group_maps(configs):
    """
    For each configuration, build:
      - group_labels: label per group (layer) like 'C1–L1 (n=9)'
      - method_to_group: mapping method -> group_index
    """
    group_labels_per_config = []
    method_to_group_per_config = []
    for ci, groups in enumerate(configs, start=1):
        labels = []
        m2g = {}
        for gi, g in enumerate(groups, start=1):
            members = g['members']
            labels.append(f'C{ci}–L{gi} (n={len(members)})')
            for m in members:
                m2g[m] = gi - 1  # 0-based index
        group_labels_per_config.append(labels)
        method_to_group_per_config.append(m2g)
    return group_labels_per_config, method_to_group_per_config

def sankey_from_layered_configs(methods, configs, title="Method rearrangements across configurations"):
    """
    methods: iterable of method names (set/list)
    configs: list of configurations; each configuration is a list of dicts with key 'members':[...]
    """
    methods = sorted(list(methods))  # stable order
    n_methods = len(methods)
    if len(configs) == 0:
        raise ValueError("Provide at least one configuration.")
    
    # Colors per method (stable)
    method_colors_hex = dict(zip(methods, evenly_spaced_colors(n_methods)))
    method_colors_rgba = {m: hex_to_rgba(c, 0.55) for m, c in method_colors_hex.items()}

    # Build group labels + method→group maps for each config
    group_labels_per_config, method_to_group_per_config = build_group_maps(configs)

    # Build nodes: one per (config, group)
    node_labels = []
    node_x = []
    node_y = []
    node_ids = {}  # (config_index, group_index) -> node_id

    n_configs = len(configs)
    for ci, labels in enumerate(group_labels_per_config):  # ci = 0..C-1
        x = 0.5 if n_configs == 1 else ci / (n_configs - 1)  # spread columns
        # vertical placement: evenly spaced within [0.05, 0.95]
        gcount = len(labels)
        if gcount == 1:
            ys = [0.5]
        else:
            ys = np.linspace(0.05, 0.95, gcount)
        for gi, lab in enumerate(labels):  # gi = 0..G-1
            node_ids[(ci, gi)] = len(node_labels)
            node_labels.append(lab)
            node_x.append(x)
            node_y.append(ys[gi])

    # Build links: for each adjacent pair of configs, one link per method
    sources, targets, values, link_colors, link_hovers = [], [], [], [], []
    for ci in range(n_configs - 1):
        m2g_src = method_to_group_per_config[ci]
        m2g_tgt = method_to_group_per_config[ci + 1]
        for m in methods:
            if m not in m2g_src and m not in m2g_tgt:
                # Method absent in both—skip
                continue
            if m not in m2g_src or m not in m2g_tgt:
                # If it appears/disappears, skip or route to a placeholder node (optional)
                # For simplicity we skip; you can add a "New" or "Dropped" bucket if needed.
                continue
            gi_src = m2g_src[m]
            gi_tgt = m2g_tgt[m]
            s = node_ids[(ci, gi_src)]
            t = node_ids[(ci + 1, gi_tgt)]
            sources.append(s)
            targets.append(t)
            values.append(1)  # each method contributes unit flow
            link_colors.append(method_colors_rgba[m])
            link_hovers.append(f"{m}<br>C{ci+1} → C{ci+2}")

    # Node colors (neutral)
    node_color = ['#e6e9ef'] * len(node_labels)

    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        node=dict(
            label=node_labels,
            pad=50,
            thickness=5,
            color=node_color,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            # show method name on hover
            customdata=link_hovers,
            hovertemplate='%{customdata}<extra></extra>'
        )
    )])

    # Add a legend mapping method → color (via invisible scatter points)
    for m in methods:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=method_colors_hex[m]),
            name=m,
            showlegend=True
        ))

    fig.update_layout(
        title=title,
        font=dict(size=12),
        legend=dict(title="Methods", itemsizing='constant', traceorder='normal')
    )
    return fig

# -----------------------
# Example usage:
methods = {
    'GE','CTMmean','REN','pNML','Confidence','fDBD','GradNorm','NNGuide','MSR',
    'NeCo','Residual','Maha','MLS','PCA RecError global','ViM','CTMmeanOC','PE',
    'CTM','Energy','PCE','GEN','KPCA RecError global'
}

config1 = [
    {'members': ['CTM', 'GEN', 'MLS', 'MSR', 'NNGuide', 'NeCo', 'PCE', 'PE', 'REN'],
     'size': 9, 'best_rank': 6.777777777777778, 'mean_rank': 7.580246913580246},
    {'members': ['CTMmean', 'CTMmeanOC', 'Confidence', 'Energy', 'GE', 'GradNorm', 'KPCA RecError global', 'fDBD'],
     'size': 8, 'best_rank': 9.88888888888889, 'mean_rank': 10.805555555555555},
    {'members': ['PCA RecError global', 'ViM', 'pNML'],
     'size': 3, 'best_rank': 18.055555555555557, 'mean_rank': 18.444444444444446},
]
# To visualize rearrangements you need ≥2 configs. Example: a toy second config
# where a few methods move across layers (replace this with your real config2..N):
config2 = [
    {'members': ['CTM', 'MSR', 'NeCo', 'MLS', 'GEN', 'REN', 'PE'], 'size': 7},
    {'members': ['CTMmean', 'CTMmeanOC', 'Confidence', 'Energy', 'GE', 'GradNorm', 'KPCA RecError global', 'fDBD', 'PCE'], 'size': 9},
    {'members': ['PCA RecError global', 'ViM', 'pNML', 'NNGuide'], 'size': 4},
]

fig = sankey_from_layered_configs(methods, layered_cliques,
                                  title="Method layering and rearrangements")
fig.show()

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import colorsys

# ---------- utilities
def distinct_palette(methods, lightness=0.52, saturation=0.62):
    """Stable distinct hex colors via evenly spaced H in HLS."""
    methods = sorted(methods)
    colors = {}
    n = len(methods)
    for i, m in enumerate(methods):
        h = (i / max(1, n)) % 1.0
        r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
        colors[m] = '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
    return colors

def method_layer_index(config, method):
    """1-based layer index for a method in a config; None if absent."""
    for i, g in enumerate(config):
        if method in g['members']:
            return i + 1
    return None

# ---------- main plot
def plot_rearrangement_dotgrid(methods, configs, title="Method rearrangement across configurations",
                               jitter=0.18, seed=0, dot_size=10, show_layer_bands=True):
    """
    methods: iterable of method names
    configs: list of configurations; each is a list of dicts with key 'members': [...]
             Top layer must be the first dict, second layer next, etc.
    """
    methods = sorted(set(methods))
    C = len(configs)
    if C == 0:
        raise ValueError("Provide at least one configuration.")
        
    # Palette: fixed color per method
    color_map = distinct_palette(methods)
    rng = np.random.default_rng(seed)

    # Build figure (one subplot per configuration)
    fig = make_subplots(rows=1, cols=C, horizontal_spacing=0.06,
                        subplot_titles=[f"Config {i+1}" for i in range(C)])
    
    # Add dots (one trace per method per config; legend only on first column)
    for ci, cfg in enumerate(configs):
        maxL = len(cfg)

        # optional background bands for layer rows
        if show_layer_bands:
            for li in range(1, maxL+1):
                fig.add_hrect(
                    y0=li-0.5, y1=li+0.5,
                    line_width=0, fillcolor="rgba(0,0,0,0.04)",
                    row=1, col=ci+1
                )

        for mi, m in enumerate(methods):
            li = method_layer_index(cfg, m)
            if li is None:
                continue
            # x is just jitter to separate dots inside the same layer
            x = rng.uniform(-jitter, jitter)
            fig.add_trace(
                go.Scatter(
                    x=[x], y=[li],
                    mode="markers",
                    marker=dict(size=dot_size, color=color_map[m],
                                line=dict(color="rgba(0,0,0,0.35)", width=0.5)),
                    name=m,
                    showlegend=(ci == 0),  # legend once
                    hovertemplate=(f"<b>{m}</b><br>"
                                   f"Config {ci+1}<br>"
                                   f"Layer {li}"
                                   "<extra></extra>")
                ),
                row=1, col=ci+1
            )

        # Axes per column
        ticktexts = [f"Layer {i+1} (n={len(cfg[i]['members'])})" for i in range(maxL)]
        fig.update_yaxes(
            range=[maxL+0.5, 0.5],  # top layer at top
            tickmode="array",
            tickvals=list(range(1, maxL+1)),
            ticktext=ticktexts,
            title_text="Layer",
            row=1, col=ci+1
        )
        fig.update_xaxes(visible=False, row=1, col=ci+1)

    fig.update_layout(
        title=title,
        height=max(350, 120 + 90 * max(len(cfg) for cfg in configs)),
        legend=dict(title="Methods", itemsizing="constant"),
        margin=dict(l=40, r=20, t=60, b=30)
    )
    return fig

# -----------------------
# Example with your set and first configuration
methods = {
    'GE', 'CTMmean', 'REN', 'pNML', 'Confidence', 'fDBD', 'GradNorm', 'NNGuide', 'MSR',
    'NeCo', 'Residual', 'Maha', 'MLS', 'PCA RecError global', 'ViM', 'CTMmeanOC', 'PE',
    'CTM', 'Energy', 'PCE', 'GEN', 'KPCA RecError global'
}

config1 = [
    {'members': ['CTM', 'GEN', 'MLS', 'MSR', 'NNGuide', 'NeCo', 'PCE', 'PE', 'REN'],
     'size': 9, 'best_rank': 6.777777777777778, 'mean_rank': 7.580246913580246},
    {'members': ['CTMmean', 'CTMmeanOC', 'Confidence', 'Energy', 'GE', 'GradNorm', 'KPCA RecError global', 'fDBD'],
     'size': 8, 'best_rank': 9.88888888888889, 'mean_rank': 10.805555555555555},
    {'members': ['PCA RecError global', 'ViM', 'pNML'],
     'size': 3, 'best_rank': 18.055555555555557, 'mean_rank': 18.444444444444446},
]

# Add your additional configurations as config2, config3, ... in the same format:
# Example (just illustrative; replace with your real groupings):
config2 = [
    {'members': ['CTM', 'MSR', 'NeCo', 'MLS', 'GEN', 'REN', 'PE']},
    {'members': ['CTMmean', 'CTMmeanOC', 'Confidence', 'Energy', 'GE', 'GradNorm',
                 'KPCA RecError global', 'fDBD', 'PCE']},
    {'members': ['PCA RecError global', 'ViM', 'pNML', 'NNGuide']},
]

fig = plot_rearrangement_dotgrid(methods, layered_cliques,
                                 title="Layer placement per configuration (no flow)")
fig.show()

    # %%

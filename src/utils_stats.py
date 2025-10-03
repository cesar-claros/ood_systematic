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
import networkx as nx
import matplotlib.lines as mlines
import matplotlib
from matplotlib import colors, patches
#%%
# Known metadata columns in score tables; all other columns are assumed to be dataset columns
KNOWN_META_COLS = ["model", "drop out", "reward", "methods"]

# Metric direction (True = higher is better on raw scale)
HIGHER_BETTER = {
    "AUROC_f": True,
    "FPR@95TPR": False,
    "AUGRC": False,
    "AURC": False
}

# Recognize variations; "global" counts as the baseline variation for most
VARIATION_TAGS = ["class avg", "class pred", "class", "global"]
#%%
def _multiply_alpha(c, mult):
    r, g, b, a = colors.to_rgba(c)
    a *= mult
    return colors.to_hex((r, g, b, a), keep_alpha=True)
#%%
def styles_grid(inclusion:np.array,other_dots_color:float,style_columns:dict):
    n_cats = inclusion.shape[1]
    bgcolor = matplotlib.rcParams.get("axes.facecolor", "white")
    r, g, b, a = colors.to_rgba(bgcolor)
    lightness = colors.rgb_to_hsv((r, g, b))[-1] * a
    facecolor = "black" if lightness >= 0.5 else "white"
    _facecolor = facecolor
    _other_dots_color = (
        _multiply_alpha(facecolor, other_dots_color)
        if isinstance(other_dots_color, float)
        else other_dots_color
    )

    subset_styles = [
        {"facecolor": facecolor} for i in range(len(inclusion))
    ]
    styles = [
        [
            subset_styles[i]
            if inclusion[i, j]
            else {"facecolor": _other_dots_color, "linewidth": 0}
            for j in range(n_cats)
        ]
        for i in range(len(inclusion))
    ]
    styles = sum(styles, [])  # flatten nested list
    
    styles = (
        pd.DataFrame(styles)
        .reindex(columns=style_columns.keys())
        .astype(
            {
                "facecolor": "O",
                "edgecolor": "O",
                "linewidth": float,
                "linestyle": "O",
                "hatch": "O",
            }
        )
    )
    # df[col] = df[col].method(value)
    styles["linewidth"] = styles["linewidth"].fillna(1,)
    styles["facecolor"] = styles["facecolor"].fillna(_facecolor,)
    styles["edgecolor"] = styles["edgecolor"].fillna(styles["facecolor"],)
    styles["linestyle"] = styles["linestyle"].fillna("solid",)
    del styles["hatch"]

    cs = pd.Series(
        [
            style.get("edgecolor", style.get("facecolor", _facecolor))
            for style in subset_styles
        ],
        name="color",
    )
    return styles, cs
#%%
def plot_grid(member_df, other_dots_color:float = 0.18, color_dotline=None, ax=None, s:int=100, zorder:int=10):
    style_columns = {
        "facecolor": "facecolors",
        "edgecolor": "edgecolors",
        "linewidth": "linewidths",
        "linestyle": "linestyles",
        "hatch": "hatch",
    }
    # methods_order = member_df.sum().sort_values(ascending=True).index
    # methods_order = sorted(member_df.columns)
    # member_df = member_df[methods_order]
    inclusion = member_df.values
    n_cats = inclusion.shape[1]
    x = np.repeat(np.arange(len(inclusion)), n_cats)
    y = np.tile(np.arange(n_cats), len(inclusion))
    styles, cs = styles_grid(inclusion, other_dots_color, style_columns)
    if color_dotline!=None and isinstance(color_dotline,list) and len(color_dotline)==len(member_df):
        for k,c_dt in enumerate(color_dotline):
            rows = int(styles.shape[0]/len(member_df))
            styles.iloc[rows*k:rows*(k+1),:] = styles.iloc[rows*k:rows*(k+1),:].replace({'black':c_dt}) 
            cs[k] = c_dt
    # not supported in matrix (currently)
    if ax == None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(
                x, y,
                s=s,
                zorder=zorder,
                # facecolor='black',
                **styles.rename(columns=style_columns),
            )

    idx = np.flatnonzero(inclusion)
    line_data = (
        pd.Series(y[idx], index=x[idx])
        .groupby(level=0)
        .aggregate(["min", "max"])
    )
    
    line_data = line_data.join(cs)
    ax.vlines(
        line_data.index.values,
        line_data["min"],
        line_data["max"],
        lw=2,
        colors=line_data["color"],
        zorder=5,
    )
    # Ticks and axes
    ax.yaxis.set_ticks(np.arange(n_cats))
    ax.yaxis.set_ticklabels(
        member_df.columns, rotation=0)
    ax.xaxis.set_ticks(np.arange(len(member_df.index)))
    ax.xaxis.set_ticklabels(
        member_df.index, rotation=90)
    # ax.xaxis.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    # ax.yaxis.set_ticks_position("top")
    ax.set_frame_on(False)
    ax.set_ylim(-0.7, y[-1] + 0.7, auto=False)
    ax.set_xlim(-0.7, x[-1] + 0.7, auto=False)
    ax.grid(True)
#%%
# Method parsing helpers -------------------------------------------------------
def parse_method_variation(method_full: str):
    """
    Split 'PCE class pred' -> base='PCE', variation='class pred'
    'ViM' -> base='ViM', variation='(none)'
    For PCA RecError / KPCA RecError: treat 'global' as implicit base.
    """
    m = method_full.strip()
    # Normalize spaces
    m = re.sub(r"\s+", " ", m)
    parts = m.split(" ")
    # Try to detect variation by scanning known tags at the end
    for tag in sorted(VARIATION_TAGS, key=len, reverse=True):
        if m.lower().endswith(tag):
            base = m[:-(len(tag))].strip()
            var = tag
            break
    else:
        base, var = m, "(none)"
    # Normalize PCA/KPCA special note for interpretation (baseline is 'global')
    return base, var

def standardize_metric(metric, values):
    """Return standardized values so that higher is better for all metrics."""
    if HIGHER_BETTER.get(metric, True):
        return values
    return -1.0 * values

def inv_standardize(metric, std_values):
    """Back to original direction."""
    if HIGHER_BETTER.get(metric, True):
        return std_values
    return -1.0 * std_values
#%%
# IO & reshape ----------------------------------------------------------------
def load_metric_csv(metric_name, csv_path):
    df = pd.read_csv(csv_path)
    path_info = csv_path.split('.')[0].split('_')
    mcd = path_info[-3] 
    backbone = path_info[-2]
    n_classes = path_info[-1]
    mcd_dict = {'MCD-False':'0', 'MCD-True':'1'}
    classes_dict = {'cifar10':10,'cifar100':100,'supercifar100':20,'tinyimagenet':200}

    # Ensure required cols exist; if some are missing, create defaults
    for col in ["model", "drop out", "methods"]:
        if col not in df.columns:
            raise ValueError(f"CSV {csv_path} must contain column '{col}'")
    if "reward" not in df.columns:
        df["reward"] = np.nan

    # infer dataset columns
    ds_cols = [c for c in df.columns if c not in KNOWN_META_COLS]
    if len(ds_cols) == 0:
        raise ValueError(f"No dataset columns found in {csv_path}")

    # melt to long
    long = df.melt(id_vars=[c for c in df.columns if c not in ds_cols],
                   value_vars=ds_cols, var_name="dataset", value_name="score")
    long["metric"] = metric_name
    # # parse method base/variation
    parsed = long["methods"].apply(parse_method_variation)
    long["method_base"] = parsed.apply(lambda x: x[0])
    long["variation"] = parsed.apply(lambda x: x[1])
    # # normalize dropout
    # long["drop out"] = long["drop out"].astype(str).str.lower()
    # long["drop out"] = long["drop out"].replace({"do0": "0", "do1": "1"})
    # # normalize reward
    long["reward"] = long["reward"].astype(str).str.lower().map(lambda x: x.split('rew')[1])
    long["reward"] = long["reward"].astype(float)
    # Add csv information
    long['backbone'] = backbone
    long['n_classes'] = n_classes
    long['n_classes'] = long['n_classes'].map(classes_dict).astype(int)
    long['MCD'] = mcd
    long['MCD'] = long['MCD'].map(mcd_dict)
    return long
#%%
def load_all_scores(config):
    frames = []
    for metric, path in config.items():
        if metric in ["CLIP_FILE", "OUTDIR", "ALPHA", "N_BOOT"]:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for {metric}: {path}")
        frames.append(load_metric_csv(metric, path))
    df = pd.concat(frames, ignore_index=True)
    # Standardized score (higher is better)
    df["score_std"] = df.apply(lambda r: standardize_metric(r["metric"], r["score"]), axis=1)
    # Clean dataset names (optional)
    # ds_map = {
    #     "cifar10_test": "CIFAR-10",
    #     "cifar100_test": "CIFAR-100",
    #     "tinyimagenet_test": "TinyImageNet",
    #     "lsun_resize": "LSUN-resize",
    #     "lsun_cropped": "LSUN-crop",
    #     "isun": "iSUN",
    #     "places365": "Places365",
    #     "textures": "Textures",
    #     "svhn": "SVHN"
    # }
    # df["dataset"] = df["dataset"].map(lambda x: ds_map.get(x, x))
    return df
#%%
# Baselines & deltas -----------------------------------------------------------
def choose_baseline_rows(df):
    """
    Define the baseline row for each (method_base, training, dropout, reward, dataset, metric).
    We prefer 'global' if present; else fallback to '(none)' if some methods don't have variations.
    """
    # flag baseline
    df = df.copy()
    df["is_baseline"] = False

    # For groups lacking any 'global', choose the sole variant as baseline if unique
    group_cols = ["method_base", "model", "drop out", "reward", "dataset", "metric"]
    for _, g in df.groupby(group_cols):
        if not g["is_baseline"].any():
            # if any '(none)' present, mark as baseline
            idx_none = g.index[g["variation"].eq("(none)")]
            if len(idx_none) > 0:
                df.loc[idx_none, "is_baseline"] = True
            # else:
            #     # otherwise pick the most common variation as pragmatic baseline
            #     counts = g["variation"].value_counts()
            #     df.loc[g.index[g["variation"] == counts.index[0]], "is_baseline"] = True

    # # KPCA and PCA Preference: 'global'
    PCA_condition = (df["method_base"].str.lower().eq("pca recerror")) | (df["method_base"].str.lower().eq("kpca recerror"))
    mask_global = df["variation"].str.lower().eq("global") & PCA_condition 
    # # Second: '(none)' for methods without variations
    # # We'll later fill remaining baselines per group if only one variant exists
    # # Mark potential global baselines:
    df.loc[mask_global, "is_baseline"] = True

    
    return df
#%%
def compute_variation_deltas(df):
    """
    For each non-baseline variation, compute Δ relative to its matched baseline
    within (method_base, training, dropout, reward, dataset, metric).
    """
    df = choose_baseline_rows(df)
    group_cols = ["method_base", "model", "drop out", "reward", "dataset", "metric"]
    # build baseline lookups
    base_df = df[df["is_baseline"]][group_cols + ["score_std"]].rename(columns={"score_std": "score_std_baseline"})
    merged = df.merge(base_df, on=group_cols, how="left")
    merged["delta_vs_baseline"] = merged["score_std"] - merged["score_std_baseline"]
    # filter to non-baseline rows
    deltas = merged[~merged["is_baseline"]].copy()
    return deltas
#%%
# Wilcoxon + bootstrap CI for HL median shift ---------------------------------
def bootstrap_ci(x, n_boot=2000, stat_fn=np.median, alpha=0.05, random_state=0):
    rng = np.random.default_rng(random_state)
    stats_boot = []
    x = np.asarray(x)
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        stats_boot.append(stat_fn(samp))
    lo = np.percentile(stats_boot, 100*alpha/2)
    hi = np.percentile(stats_boot, 100*(1-alpha/2))
    return float(np.median(x)), float(lo), float(hi)

def signed_rank_test(x):
    # Wilcoxon signed rank against 0
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan, np.nan
    stat, p = stats.wilcoxon(x, zero_method="wilcox", alternative="greater")  # testing Δ>0
    return stat, p
#%%
# Mixed effects helpers --------------------------------------------------------
def fit_mixedlm(df, formula, group_cols, re_formula='1', vc_formula=None):
    """
    Fit mixed model with multiple grouping random intercepts via patsy trick:
    we add categorical random effects as separate RE terms by concatenating.
    """
    # Create a single grouping factor by crossing groups (approximate)
    df = df.copy()
    if isinstance(group_cols, (list, tuple)):
        df["grp"] = df[group_cols].astype(str).agg("|".join, axis=1)
        group = "grp"
    else:
        group = group_cols
    try:
        model = smf.mixedlm(formula, data=df, groups=df[group], re_formula=re_formula, vc_formula=vc_formula)
        res = model.fit(method="lbfgs", reml=False, maxiter=200)
        return res
    except Exception as e:
        print("MixedLM failed:", e)
        return None
#%%
# Friedman + posthoc -----------------------------------------------------------
def friedman_blocked(df, entity_col, block_col, value_col):
    """
    Friedman test across entities with blocks.
    df rows must be complete blocks over entity_col within each block.
    """
    # pivot blocks × entities
    pivot = df.pivot_table(index=block_col, columns=entity_col, values=value_col, aggfunc='mean')
    pivot = pivot.dropna(axis=0, how='any')
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return np.nan, np.nan, pivot
    stat, p = stats.friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    return stat, p, pivot
#%%
def nemenyi_posthoc_from_pivot(pivot):
    """
    Run Nemenyi posthoc on pivot (blocks × entities) required by scikit_posthocs.
    """
    # scikit_posthocs expects raw matrix
    data = pivot.values
    ph = sp.posthoc_nemenyi_friedman(data)
    ph.index = pivot.columns
    ph.columns = pivot.columns
    return ph
#%%
def conover_posthoc_from_pivot(pivot):
    """
    Run Nemenyi posthoc on pivot (blocks × entities) required by scikit_posthocs.
    """
    # scikit_posthocs expects raw matrix
    data = pivot.values
    ph = sp.posthoc_conover_friedman(data)
    ph.index = pivot.columns
    ph.columns = pivot.columns
    return ph
#%%
# Ranking utilities ------------------------------------------------------------
def average_ranks(values_by_entity):
    """
    Given a dict entity -> list of values (higher better), return avg rank per entity.
    """
    df = pd.DataFrame(values_by_entity)
    # rank within each row
    ranks = df.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()
#%%
def kendalls_w(rank_matrix):
    """
    rank_matrix: numpy array shape (m items, n raters)
    """
    R = rank_matrix
    m, n = R.shape
    R_bar = R.mean(axis=0)
    S = np.sum((R.sum(axis=1) - n*(m+1)/2.0)**2)
    W = 12*S / (n**2 * (m**3 - m))
    return W

#%%
def pareto_non_dominated(x_fpr: np.ndarray, y_auroc: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of non-dominated points (Pareto frontier).
    Here: lower x (FPR) is better, higher y (AUROC) is better.
    Point i is dominated if ∃ j with:
      x_j <= x_i and y_j >= y_i, and at least one strict inequality.
    """
    pts = np.column_stack([x_fpr, y_auroc])
    n = len(pts)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: 
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1]) and \
               (pts[j, 0] < pts[i, 0] or  pts[j, 1] >  pts[i, 1]):
                nd[i] = False
                break
    return nd
#%%
def plot_pareto_front(df, metric_auroc="AUROC_f", metric_fpr="FPR@95TPR", outpath=None, title="Pareto Front: AUROC vs FPR@95"):
    sub = df[(df["metric"].isin([metric_auroc, metric_fpr]))]
    # Make wide per (method, dataset) averaging over training/do/reward
    agg = sub.groupby(["methods", "dataset", "metric"])["score"].mean().reset_index()
    wide = agg.pivot_table(index=["methods", "dataset"], columns="metric", values="score")
    wide = wide.dropna()
    ## Global
    # points = list(zip(wide[metric_fpr].values, wide[metric_auroc].values))
    # nd = pareto_non_dominated(points)
    nd = pareto_non_dominated(wide[metric_fpr].values,wide[metric_auroc].values)
    plt.figure(figsize=(7,5))
    plt.scatter(wide[metric_fpr], wide[metric_auroc], alpha=0.4, label="All")
    plt.scatter(wide[metric_fpr].values[nd], wide[metric_auroc].values[nd], s=30, label="Pareto", marker='x')
    plt.xlabel(metric_fpr + " (lower better)")
    plt.ylabel(metric_auroc + " (higher better)")
    plt.title(title)
    plt.legend()
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath, dpi=160)
        plt.close()
    # Per dataset
    units_by_dataset = group_units_by_dataset(wide)
    # Observed frontier
    obs_frontier = compute_observed_frontiers(wide)
    return obs_frontier
#%%
def group_units_by_dataset(wide: pd.DataFrame) -> dict:
    """
    Return mapping: dataset -> { method: array of shape (n_units, 2) with columns [AUROC, FPR] }.
    If a method appears multiple times (e.g., different dropout/reward), each is a unit.
    """
    units_by_dataset = {}
    for dataset, g in wide.groupby("dataset"):
        by_m = {}
        for method, gm in g.groupby("methods"):
            # store as [AUROC, FPR]
            by_m[method] = gm[["AUROC_f", "FPR@95TPR"]].to_numpy()
        units_by_dataset[dataset] = by_m
    return units_by_dataset
#%%
def group_units(wide: pd.DataFrame) -> dict:
    """
    Return mapping: dataset -> { method: array of shape (n_units, 2) with columns [AUROC, FPR] }.
    If a method appears multiple times (e.g., different dropout/reward), each is a unit.
    """
    # units_by_dataset = {}
    # for dataset, g in wide.groupby("dataset"):
    by_m = {}
    for method, gm in wide.groupby("methods"):
        # store as [AUROC, FPR]
        by_m[method] = gm[["AUROC_f", "FPR@95TPR"]].to_numpy()
    # units_by_dataset[dataset] = by_m
    return by_m
#%%
def compute_observed_frontiers(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute observed method points = mean across units, then frontier per dataset.
    """
    obs_points = []
    for (dataset, method), g in wide.groupby(["dataset", "methods"]):
        arr = g[["AUROC_f", "FPR@95TPR"]].to_numpy()
        mu = arr.mean(axis=0)
        obs_points.append({
            "dataset": dataset, "method": method,
            "FPR@95TPR": float(mu[1]), "AUROC_f": float(mu[0]),
            "n_units": int(arr.shape[0])
        })
    obs_points = pd.DataFrame(obs_points)
    rows = []
    for dataset, g in obs_points.groupby("dataset"):
        if len(g) < 2: 
            continue
        nd = pareto_non_dominated(g["FPR@95TPR"].values, g["AUROC_f"].values)
        tmp = g.copy()
        tmp["on_frontier"] = nd
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame([])
#%%
def bootstrap_frontier_stability(units_by_dataset: dict, B=600, seed=0) -> pd.DataFrame:
    """
    For each dataset, resample units WITHIN each method (size=original #units) and
    recompute mean point; mark if the method lies on the Pareto frontier.
    Return per (dataset, method): frontier_prob in [0,1].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, by_m in units_by_dataset.items():
        methods = sorted([m for m, arr in by_m.items() if arr.shape[0] >= 1])
        if len(methods) < 2: 
            continue
        counts = {m: 0 for m in methods}
        n_units = {m: by_m[m].shape[0] for m in methods}
        for b in range(B):
            recs = []
            for m in methods:
                arr = by_m[m]
                idx = rng.integers(0, arr.shape[0], size=arr.shape[0]) if arr.shape[0] > 0 else np.array([], dtype=int)
                samp = arr[idx] if arr.shape[0] > 0 else np.empty((0, 2))
                if samp.shape[0] == 0:
                    continue
                mu = samp.mean(axis=0)   # [AUROC, FPR]
                recs.append((m, float(mu[1]), float(mu[0])))
            if len(recs) < 2:
                continue
            names = [r[0] for r in recs]
            fprs  = np.array([r[1] for r in recs])
            aurocs= np.array([r[2] for r in recs])
            nd = pareto_non_dominated(fprs, aurocs)
            for k, m in enumerate(names):
                if nd[k]:
                    counts[m] += 1
        for m in methods:
            rows.append({
                "dataset": dataset,
                "method": m,
                "frontier_prob": counts[m] / B if B > 0 else np.nan,
                "n_units": n_units[m],
                "B": B
            })
    return pd.DataFrame(rows)


#%%
def constraints_from_frontier_points(frontier_df: pd.DataFrame, taus, phis_use) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build observed constraint tables from FRONTIER points:
      A: min FPR given AUROC >= tau
      B: max AUROC given FPR <= phi
    """
    outA, outB = [], []
    for dataset, g in frontier_df.groupby("dataset"):
        front = g[g["on_frontier"]].copy()
        # A: min FPR at AUROC >= tau
        for tau in taus:
            cand = front[front["AUROC_f"] >= tau]
            if cand.empty:
                outA.append({"dataset": dataset, "tau": tau, "min_FPR_at_tau": np.nan, "support": 0})
            else:
                outA.append({"dataset": dataset, "tau": tau,
                             "min_FPR_at_tau": float(cand["FPR@95TPR"].min()),
                             "support": int(len(cand))})
        # B: max AUROC at FPR <= phi
        for phi in phis_use:
            cand = front[front["FPR@95TPR"] <= phi]
            if cand.empty:
                outB.append({"dataset": dataset, "phi": phi, "max_AUROC_at_phi": np.nan, "support": 0})
            else:
                outB.append({"dataset": dataset, "phi": phi,
                             "max_AUROC_at_phi": float(cand["AUROC_f"].max()),
                             "support": int(len(cand))})
    return pd.DataFrame(outA), pd.DataFrame(outB)
#%%
# Caterpillar (forest) plot for Δ ----------------------------------------------
def caterpillar_plot(deltas_df, group_cols, delta_col="delta_vs_baseline", label_col="label", outpath=None, title="Δ vs baseline (HL median with 95% CI)"):
    """
    Expect deltas_df aggregated to one row per label, with columns:
    label, hl_median, lo, hi
    """
    plot_df = deltas_df.copy()
    # plot_df = plot_df.sort_values("hl_median")
    y = np.arange(len(plot_df))
    plt.figure(figsize=(8, max(4, 0.3*len(y))))
    plt.hlines(y, plot_df["lo"], plot_df["hi"])
    plt.plot(plot_df["hl_median"], y, "o")
    plt.yticks(y, plot_df[label_col])
    plt.axvline(0, color="k", linewidth=1, linestyle="--")
    plt.xlabel("HL median Δ (standardized; >0 favors variation)")
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=160)
        plt.close()
#%%
def bootstrap_constraints(units_by_dataset: dict, taus, phis_use, B=600, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap CIs for constraints:
      For each dataset and replicate:
        - resample units within each method to get mean point per method
        - compute frontier and constraints
      Then summarize per tau/phi: median, 95% CI.
    """
    rng = np.random.default_rng(seed)
    rowsA, rowsB = [], []
    for dataset, by_m in units_by_dataset.items():
        methods = [m for m, arr in by_m.items() if arr.shape[0] >= 1]
        if len(methods) < 2:
            continue
        valsA = {tau: [] for tau in taus}
        valsB = {phi: [] for phi in phis_use}
        for b in range(B):
            recs = []
            for m in methods:
                arr = by_m[m]
                idx = rng.integers(0, arr.shape[0], size=arr.shape[0]) if arr.shape[0] > 0 else np.array([], dtype=int)
                samp = arr[idx] if arr.shape[0] > 0 else np.empty((0, 2))
                if samp.shape[0] == 0:
                    continue
                mu = samp.mean(axis=0)   # [AUROC, FPR]
                recs.append((m, float(mu[1]), float(mu[0])))
            if len(recs) < 2:
                continue
            fprs  = np.array([r[1] for r in recs])
            aurocs= np.array([r[2] for r in recs])
            nd = pareto_non_dominated(fprs, aurocs)
            front_f = fprs[nd]
            front_a = aurocs[nd]
            # A: min FPR at AUROC >= tau
            for tau in taus:
                mask = front_a >= tau
                valsA[tau].append(float(np.min(front_f[mask])) if np.any(mask) else np.nan)
            # B: max AUROC at FPR <= phi
            for phi in phis_use:
                mask = front_f <= phi
                valsB[phi].append(float(np.max(front_a[mask])) if np.any(mask) else np.nan)
        # summarize
        for tau, arr in valsA.items():
            v = np.array([x for x in arr if not math.isnan(x)])
            if v.size == 0:
                rowsA.append({"dataset": dataset, "tau": tau, "median": np.nan, "lo": np.nan, "hi": np.nan, "B_eff": 0})
            else:
                rowsA.append({"dataset": dataset, "tau": tau,
                              "median": float(np.nanmedian(v)),
                              "lo": float(np.nanpercentile(v, 2.5)),
                              "hi": float(np.nanpercentile(v, 97.5)),
                              "B_eff": int(v.size)})
        for phi, arr in valsB.items():
            v = np.array([x for x in arr if not math.isnan(x)])
            if v.size == 0:
                rowsB.append({"dataset": dataset, "phi": phi, "median": np.nan, "lo": np.nan, "hi": np.nan, "B_eff": 0})
            else:
                rowsB.append({"dataset": dataset, "phi": phi,
                              "median": float(np.nanmedian(v)),
                              "lo": float(np.nanpercentile(v, 2.5)),
                              "hi": float(np.nanpercentile(v, 97.5)),
                              "B_eff": int(v.size)})
    return pd.DataFrame(rowsA), pd.DataFrame(rowsB)
#%%
# Top-10 stability -------------------------------------------------------------
def rbo(S, T, p=0.9, k=None):
    """
    Rank Biased Overlap for two ranked lists S, T.
    """
    if k is None:
        k = max(len(S), len(T))
    S = S[:k]; T = T[:k]
    ss = set(); tt = set()
    A = 0.0
    for d in range(1, k+1):
        ss.add(S[d-1] if d-1 < len(S) else None)
        tt.add(T[d-1] if d-1 < len(T) else None)
        Xd = len(ss.intersection(tt))
        A += (Xd / d) * (p ** (d-1))
    return (1-p) * A

def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a|b) == 0:
        return np.nan
    return len(a & b) / len(a | b)

# CLIP LODO --------------------------------------------------------------------
def lodo_regression(perf_df, clip_df, feature_cols, target_col="score_std", id_col="dataset", method_col="methods"):
    """
    Leave-one-dataset-out ridge predicting target from CLIP features + method dummies.
    Returns R2 per left-out dataset and overall summary.
    """
    # Merge
    df = perf_df.merge(clip_df, on="dataset", how="left")
    # One-hot method
    cat_cols = [method_col]
    numeric = feature_cols
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    model = Pipeline([("pre", pre), ("ridge", RidgeCV(alphas=np.logspace(-3,3,13)))])
    results = []
    datasets = df[id_col].dropna().unique()
    for hold in datasets:
        train = df[df[id_col] != hold]
        test = df[df[id_col] == hold]
        if test.empty or train.empty:
            continue
        Xtr = train[numeric + cat_cols]
        ytr = train[target_col]
        Xte = test[numeric + cat_cols]
        yte = test[target_col]
        model.fit(Xtr, ytr)
        r2 = model.score(Xte, yte)
        mae = np.mean(np.abs(model.predict(Xte) - yte))
        results.append({"left_out": hold, "R2": r2, "MAE": mae, "n": len(test)})
    res_df = pd.DataFrame(results)
    return res_df
#%%
def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n1, n2 = len(x), len(y)
    if n1==0 or n2==0: return np.nan
    allv = np.concatenate([x,y])
    ranks = stats.rankdata(allv, method="average")
    r1 = ranks[:n1].sum()
    U1 = r1 - n1*(n1+1)/2.0
    return float((2*U1)/(n1*n2) - 1)
#%%
def permute_within_strata(df_in, groups, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    # observed weighted mean diff across strata
    obs, weights = [], []
    for key, g in df_in.groupby(groups):
        a = g[g["drop out"]=="do1"]["score_std"].values
        b = g[g["drop out"]=="do0"]["score_std"].values
        if len(a)==0 or len(b)==0: continue
        obs.append(np.mean(a) - np.mean(b))
        weights.append(len(a)+len(b))
    obs_diff = np.average(obs, weights=weights) if obs else np.nan

    # permute labels within strata
    null = []
    for _ in range(B):
        diffs, wts = [], []
        for key, g in df_in.groupby(groups):
            vals = g["score_std"].values.copy()
            labs = g["drop out"].values.copy()
            # shuffle labels within stratum
            rng.shuffle(labs)
            a = vals[labs=="do1"]; b = vals[labs=="do0"]
            if len(a)==0 or len(b)==0: continue
            diffs.append(np.mean(a) - np.mean(b))
            wts.append(len(a)+len(b))
        if diffs:
            null.append(np.average(diffs, weights=wts))
    null = np.array(null)
    if null.size == 0 or np.isnan(obs_diff):
        return obs_diff, np.nan
    p = (np.sum(np.abs(null) >= abs(obs_diff)) + 1) / (len(null) + 1)
    return obs_diff, p
#%%
def get_dominance_graph_from_df(df, item_col, criteria_cols):
    """
    Constructs a dominance graph from a pandas DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the items and criteria.
        item_col (str): The name of the column identifying each item.
        criteria_cols (list): A list of column names for the criteria.
    Returns:
        nx.DiGraph: A directed graph representing the dominance relations.
    """
    dominance_graph = nx.DiGraph()
    dominance_graph.add_nodes_from(df[item_col])

    # Iterate through all pairs of rows in the DataFrame
    for i, item1 in df.iterrows():
        for j, item2 in df.iterrows():
            if i == j:
                continue

            # Check for dominance: item1 dominates item2
            is_dominant = all(item1[c] <= item2[c] for c in criteria_cols) and \
                          any(item1[c] < item2[c] for c in criteria_cols)

            if is_dominant:
                dominance_graph.add_edge(item1[item_col], item2[item_col])
    return dominance_graph

def get_hierarchical_positions(graph):
    """
    Computes a hierarchical layout for a DAG by assigning levels
    based on the longest path from a minimal node.
    """
    levels = {}
    
    # Calculate the longest path from any source (minimal node) for each node
    # This automatically determines the correct layer for each node
    for node in nx.topological_sort(graph):
        if graph.in_degree(node) == 0:
            levels[node] = 0
        else:
            levels[node] = 1 + max(levels[pred] for pred in graph.predecessors(node))
    
    # Get all nodes sorted by level
    nodes_by_level = {}
    for node, level in levels.items():
        nodes_by_level.setdefault(level, []).append(node)
    
    # Assign positions
    pos = {}
    max_y = max(levels.values()) if levels else 0
    
    for level, nodes in nodes_by_level.items():
        # Distribute nodes horizontally for a clean layout
        num_nodes_in_level = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (num_nodes_in_level - 1) / 2) * 0.5  # Simple horizontal spacing
            # y = max_y - level  # Invert y to place maximal nodes at the top
            y = -(max_y - level)
            pos[node] = (x, y)
    
    return pos

def draw_hasse_diagram(dominance_graph, title="Hasse Diagram of Significant Dominance", ax=None):
    """
    Draws the Hasse diagram (transitive reduction of the dominance graph).
    Args:
        dominance_graph (nx.DiGraph): The full dominance graph.
        title (str): The title for the plot.
    """
    # Compute the transitive reduction to get the Hasse diagram
    hasse_diagram = nx.transitive_reduction(dominance_graph)
    # 1. Identify maximal nodes (nodes with no outgoing edges)
    maximal_nodes = [node for node, out_degree in hasse_diagram.out_degree() if out_degree == 0]
    # 2. Create a dictionary with labels for only the maximal nodes
    maximal_labels = {node: node for node in maximal_nodes}
    
    pos = get_hierarchical_positions(hasse_diagram)
    
    # plt.figure(figsize=(8, 8))
    
    # Draw all nodes and edges, but without any labels
    node_colors = ['red' if node in maximal_nodes else 'lightgrey' for node in hasse_diagram.nodes]
    nx.draw_networkx_edges(hasse_diagram, pos, arrowsize=1, alpha=0.15, style=':', ax=ax)
    nx.draw_networkx_nodes(hasse_diagram, pos, node_size=100, node_color=node_colors, alpha=0.5, ax=ax)
    
    # 3. Draw labels only for the maximal nodes
    maximal_texts = nx.draw_networkx_labels(hasse_diagram, pos, labels=maximal_labels, font_size=8,ax=ax)
    # 2. Iterate through the text objects and set the rotation
    for _, t in maximal_texts.items():
        t.set_rotation(45)
    ax.set_title(title)
    # plt.show()
#%%

def rank_cliques(cliques, methods, avg_ranks):
    scored = []
    for C in cliques:
        names = C
        ranks = avg_ranks.loc[names]
        scored.append({
            "members": names,
            "size": len(names),
            "best_rank": ranks.min(),
            "mean_rank": ranks.mean()
        })
    scored = sorted(scored, key=lambda d: (d["best_rank"], d["mean_rank"], -d["size"]))
    return scored

def greedy_exclusive_layers(scored_cliques, max_layers=None):
    """Pick non-overlapping cliques in layers, from best to worse."""
    layers, used = [], set()
    for sc in scored_cliques:
        if any(m in used for m in sc["members"]): 
            continue
        layers.append(sc)
        used.update(sc["members"])
        if max_layers and len(layers) >= max_layers:
            break
    return layers

def ns_graph_from_pmatrix(p_adj: pd.DataFrame, alpha=0.05):
    meth = p_adj.index.tolist()
    M = p_adj.loc[meth, meth].to_numpy()
    # Build adjacency: edge ⇔ NOT significant (p ≥ alpha)
    A = (M >= alpha)
    A = np.where(np.isnan(A), False, A)   # NaN -> no edge
    np.fill_diagonal(A, False)            # no self-loops
    A = np.logical_and(A, A.T)            # symmetrize
    # neighbor sets
    N = [set(np.where(A[i])[0]) for i in range(A.shape[0])]
    return meth, N

def bron_kerbosch_pivot(R, P, X, N):
    if not P and not X:
        yield R; return
    # pivot u from P∪X with max neighbors
    if P or X:
        u = max(P | X, key=lambda v: len(N[v]))
        cand = P - N[u]
    else:
        cand = set()
    for v in list(cand):
        yield from bron_kerbosch_pivot(R | {v}, P & N[v], X & N[v], N)
        P.remove(v); X.add(v)

def maximal_cliques_from_pmatrix(p_adj: pd.DataFrame, alpha=0.05, min_size=1):
    methods, N = ns_graph_from_pmatrix(p_adj, alpha=alpha)
    P, X = set(range(len(methods))), set()
    cliques = []
    for C in bron_kerbosch_pivot(set(), P, X, N):
        if len(C) >= min_size:
            cliques.append([methods[i] for i in sorted(C)])
    return cliques
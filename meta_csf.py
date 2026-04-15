"""Meta-CSF: NC-metric-guided OOD detection method selector.

Given a model's Neural Collapse metric profile, the meta-CSF selects the
best OOD detection method by partitioning the most predictive NC metric
into segments and assigning the best-performing method to each segment.

Pipeline
--------
1. Load NC metrics + AUGRC scores  →  regression dataset
2. Z-score NC features within each source dataset (removes dataset-scale
   confound so bins reflect geometry, not dataset identity)
3. LODO evaluation — for each held-out dataset:
   a. Select the top NC feature by IV on the training datasets
   b. Optimally bin that feature; assign best method per bin
   c. Apply the bins to the held-out dataset; compare meta-CSF
      recommendation against oracle, best-single method, and clique
4. Visualise: NC metric on x-axis, method ranks as lines, bin boundaries
   as vertical separators, coloured bands showing the meta-CSF selection
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from optbinning import OptimalBinning
from scipy.stats import gaussian_kde
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from loguru import logger

from method_augrc_prediction import (
    PAPYAN_NC_METRICS,
    ARCH_MAP,
    STUDY_MAP,
    SOURCES,
    load_scores,
    build_regression_dataset,
)
from nc_regime_analysis import NC_METRICS, load_nc_metrics


# ── Normalisation ────────────────────────────────────────────────────────────
def zscore_nc_features(
    df: pd.DataFrame,
    nc_features: list[str],
) -> pd.DataFrame:
    """Z-score each NC feature within each source dataset.

    This removes the dataset-scale confound so that bins reflect
    geometry differences rather than dataset identity.
    """
    df = df.copy()
    avail = [f for f in nc_features if f in df.columns]
    for feat in avail:
        df[feat] = df.groupby("dataset")[feat].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
        )
    return df


# ── Feature selection ────────────────────────────────────────────────────────
def select_top_feature(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    top_k: int = 3,
) -> tuple[str, pd.DataFrame]:
    """Select the NC feature with the highest mean IV across methods.

    The binary target is avg_ood_rank <= top_k within each setting.

    Returns
    -------
    best_feature : str
    iv_table : DataFrame  (feature, mean_iv)
    """
    gdf = df[df["group"] == group_label].copy()
    avail = [f for f in nc_features if f in gdf.columns]

    # Binarise
    rank_keys = ["dataset", "group", "study", "dropout", "reward", "run"]
    gdf["setting_rank"] = gdf.groupby(rank_keys)["avg_ood_rank"].rank(
        method="average", ascending=True,
    )
    gdf["is_top"] = (gdf["setting_rank"] <= top_k).astype(int)

    iv_rows = []
    methods = sorted(gdf["method"].unique())
    for method in methods:
        msub = gdf[gdf["method"] == method]
        y = msub["is_top"].values
        if y.sum() == 0 or y.sum() == len(y):
            continue
        for feat in avail:
            x = msub[feat].values
            try:
                ob = OptimalBinning(
                    name=feat, dtype="numerical", solver="cp",
                    min_bin_size=0.05, max_n_bins=6,
                )
                ob.fit(x, y)
                table = ob.binning_table.build()
                total_iv = pd.to_numeric(
                    table[~table["Bin"].isin(["Special", "Missing", "Totals"])]["IV"],
                    errors="coerce",
                ).sum()
                iv_rows.append({
                    "method": method, "feature": feat, "total_iv": total_iv,
                })
            except Exception:
                pass

    iv_df = pd.DataFrame(iv_rows)
    if iv_df.empty:
        logger.warning("  No IV computed; falling back to first feature")
        return avail[0], pd.DataFrame({"feature": avail, "mean_iv": 0})

    mean_iv = (
        iv_df.groupby("feature")["total_iv"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"total_iv": "mean_iv"})
    )
    best = mean_iv.iloc[0]["feature"]
    logger.info(
        f"  Top feature by IV: {best} "
        f"(IV={mean_iv.iloc[0]['mean_iv']:.4f})"
    )
    logger.info(f"  IV ranking:\n{mean_iv.to_string(index=False)}")
    return best, mean_iv


# ── Meta-CSF construction ────────────────────────────────────────────────────
def build_meta_csf(
    df: pd.DataFrame,
    feature: str,
    group_label: str,
    max_n_bins: int = 4,
) -> tuple[list[float], list[str], pd.DataFrame]:
    """Bin the NC feature and assign the best method per bin.

    Returns
    -------
    splits : list[float]
        Bin boundaries (n_bins - 1 values).
    bin_methods : list[str]
        Recommended method for each bin (length = n_bins).
    bin_summary : DataFrame
        Per-bin summary: bin_id, bin_range, best_method, mean_rank,
        n_samples, methods_detail.
    """
    gdf = df[df["group"] == group_label].copy()

    # Pool all methods to find global optimal bins on the feature
    x_all = gdf[feature].values
    # Create a dummy binary target: top-k overall
    rank_keys = ["dataset", "group", "study", "dropout", "reward", "run"]
    gdf["setting_rank"] = gdf.groupby(rank_keys)["avg_ood_rank"].rank(
        method="average", ascending=True,
    )
    y_dummy = (gdf["setting_rank"] <= 3).astype(int).values

    ob = OptimalBinning(
        name=feature, dtype="numerical", solver="cp",
        min_bin_size=0.05, max_n_bins=max_n_bins,
    )
    ob.fit(x_all, y_dummy)
    splits = list(ob.splits)

    # Assign bin id to each row
    bin_edges = [-np.inf] + splits + [np.inf]
    gdf["nc_bin"] = pd.cut(
        gdf[feature], bins=bin_edges, labels=False, include_lowest=True,
    )

    # For each bin, find the method with the lowest mean avg_ood_rank
    bin_methods = []
    summary_rows = []
    for bin_id in sorted(gdf["nc_bin"].dropna().unique()):
        bsub = gdf[gdf["nc_bin"] == bin_id]
        method_ranks = (
            bsub.groupby("method")["avg_ood_rank"]
            .mean()
            .sort_values()
        )
        best = method_ranks.index[0]
        bin_methods.append(best)

        lo = bin_edges[int(bin_id)]
        hi = bin_edges[int(bin_id) + 1]
        bin_range = f"({lo:.3f}, {hi:.3f}]"

        # Top-3 methods detail
        top3 = method_ranks.head(3)
        detail = ", ".join(
            f"{m}({r:.2f})" for m, r in top3.items()
        )
        summary_rows.append({
            "bin_id": int(bin_id),
            "bin_range": bin_range,
            "best_method": best,
            "mean_rank": method_ranks.iloc[0],
            "n_samples": len(bsub),
            "methods_detail": detail,
        })
        logger.info(
            f"  Bin {int(bin_id)} {bin_range}: "
            f"best={best} (rank={method_ranks.iloc[0]:.2f}), "
            f"n={len(bsub)}"
        )

    return splits, bin_methods, pd.DataFrame(summary_rows)


# ── LODO evaluation ──────────────────────────────────────────────────────────
def run_lodo_meta_csf(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    top_k: int = 3,
    max_n_bins: int = 4,
    forced_feature: str | None = None,
    cliques: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Leave-One-Dataset-Out evaluation of the meta-CSF.

    For each held-out dataset:
      1. Select top feature (IV) on training datasets
      2. Build meta-CSF (bin + assign methods) on training datasets
      3. Apply bins to held-out dataset, evaluate recommendation

    Returns
    -------
    eval_df : DataFrame   — one row per (held-out instance)
    fold_summaries : list  — per-fold summary dicts
    """
    gdf = df[df["group"] == group_label].copy()
    datasets = sorted(gdf["dataset"].unique())
    avail = [f for f in nc_features if f in gdf.columns]

    all_results = []
    fold_summaries = []

    for held_out in datasets:
        logger.info(f"  LODO fold: held-out = {held_out}")
        train_df = gdf[gdf["dataset"] != held_out]
        test_df = gdf[gdf["dataset"] == held_out]

        if train_df.empty or test_df.empty:
            continue

        # 1. Feature selection on training data
        if forced_feature:
            best_feature = forced_feature
        else:
            best_feature, _ = select_top_feature(
                train_df.assign(group=group_label),
                avail, group_label, top_k=top_k,
            )
            # select_top_feature filters by group internally,
            # but train_df is already filtered — pass group_label through
        logger.info(f"    Feature: {best_feature}")

        # 2. Build meta-CSF on training data
        splits, bin_methods, bin_summary = build_meta_csf(
            train_df.assign(group=group_label),
            best_feature, group_label,
            max_n_bins=max_n_bins,
        )
        logger.info(f"    Splits: {splits}, Methods: {bin_methods}")

        # 3. Apply to held-out
        bin_edges = [-np.inf] + splits + [np.inf]
        test_copy = test_df.copy()
        test_copy["nc_bin"] = pd.cut(
            test_copy[best_feature], bins=bin_edges,
            labels=False, include_lowest=True,
        )

        # Best single method from training data
        train_overall = (
            train_df.groupby("method")["avg_ood_rank"]
            .mean()
            .sort_values()
        )
        best_single_method = train_overall.index[0]

        # Evaluate per instance in held-out
        instance_keys = ["dataset", "study", "dropout", "reward", "run"]
        for keys, inst in test_copy.groupby(instance_keys):
            key_dict = dict(zip(instance_keys, keys))

            nc_val = inst[best_feature].iloc[0]
            bin_id = inst["nc_bin"].iloc[0]

            augrc_map = dict(zip(inst["method"], inst["augrc"]))

            # Oracle
            oracle_method = min(augrc_map, key=augrc_map.get)
            oracle_augrc = augrc_map[oracle_method]

            # Best single
            single_augrc = augrc_map.get(best_single_method, np.nan)

            # Meta-CSF
            if pd.notna(bin_id) and int(bin_id) < len(bin_methods):
                meta_method = bin_methods[int(bin_id)]
                meta_augrc = augrc_map.get(meta_method, np.nan)
            else:
                meta_method = best_single_method
                meta_augrc = single_augrc

            # If the meta method doesn't exist in this instance, fall back
            if np.isnan(meta_augrc):
                meta_method = best_single_method
                meta_augrc = single_augrc

            meta_regret = meta_augrc - oracle_augrc
            single_regret = single_augrc - oracle_augrc if not np.isnan(single_augrc) else np.nan
            meta_norm = meta_regret / oracle_augrc if oracle_augrc else 0
            single_norm = single_regret / oracle_augrc if oracle_augrc else 0

            # Clique
            clique_hit = False
            if cliques and held_out in cliques:
                clique_hit = meta_method in set(cliques[held_out])

            all_results.append({
                **key_dict,
                "held_out": held_out,
                "feature_used": best_feature,
                best_feature: nc_val,
                "nc_bin": int(bin_id) if pd.notna(bin_id) else -1,
                "oracle_method": oracle_method,
                "oracle_augrc": oracle_augrc,
                "best_single_method": best_single_method,
                "best_single_augrc": single_augrc,
                "best_single_regret": single_regret,
                "best_single_norm_regret": single_norm,
                "meta_method": meta_method,
                "meta_augrc": meta_augrc,
                "meta_regret": meta_regret,
                "meta_norm_regret": meta_norm,
                "clique_hit": clique_hit,
            })

        # Fold summary
        fold_results = [r for r in all_results if r["held_out"] == held_out]
        fold_df_tmp = pd.DataFrame(fold_results)
        meta_wins = (fold_df_tmp["meta_augrc"] <= fold_df_tmp["best_single_augrc"]).mean()
        fold_summaries.append({
            "held_out": held_out,
            "feature": best_feature,
            "n_bins": len(bin_methods),
            "splits": str(splits),
            "bin_methods": "|".join(bin_methods),
            "meta_mean_regret": fold_df_tmp["meta_regret"].mean(),
            "meta_mean_norm_regret": fold_df_tmp["meta_norm_regret"].mean(),
            "single_mean_regret": fold_df_tmp["best_single_regret"].mean(),
            "single_mean_norm_regret": fold_df_tmp["best_single_norm_regret"].mean(),
            "meta_wins_rate": meta_wins,
            "clique_hit_rate": fold_df_tmp["clique_hit"].mean() if cliques else np.nan,
        })

        logger.info(
            f"    {held_out}: meta regret={fold_df_tmp['meta_regret'].mean():.4f} "
            f"({fold_df_tmp['meta_norm_regret'].mean():.1%}), "
            f"single regret={fold_df_tmp['best_single_regret'].mean():.4f} "
            f"({fold_df_tmp['best_single_norm_regret'].mean():.1%}), "
            f"meta_wins={meta_wins:.1%}"
        )

    eval_df = pd.DataFrame(all_results)
    return eval_df, fold_summaries


# ── Smoothing ────────────────────────────────────────────────────────────────
def _nadaraya_watson(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """Nadaraya-Watson kernel regression with Gaussian kernel.

    Parameters
    ----------
    x, y : array-like  — observed points
    x_grid : array-like — points at which to evaluate the smoother
    bandwidth : float or None
        Kernel bandwidth.  If None, uses Silverman's rule of thumb.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)

    if bandwidth is None:
        bandwidth = 1.06 * x.std() * len(x) ** (-1 / 5)
        bandwidth = max(bandwidth, 1e-6)

    # Gaussian kernel weights: K((x_grid - x_i) / h)
    diff = x_grid[:, None] - x[None, :]          # (G, N)
    weights = np.exp(-0.5 * (diff / bandwidth) ** 2)  # (G, N)
    w_sum = weights.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum == 0, 1, w_sum)
    y_hat = (weights * y[None, :]).sum(axis=1) / w_sum.squeeze()
    return y_hat


# ── Visualisation ────────────────────────────────────────────────────────────
def plot_meta_csf(
    df: pd.DataFrame,
    feature: str,
    group_label: str,
    splits: list[float],
    bin_methods: list[str],
    cliques: dict[str, list[str]] | None = None,
    output_dir: str = "meta_csf_outputs",
    tag: str = "",
    bandwidth: float | None = None,
) -> None:
    """Visualise the meta-CSF with Nadaraya-Watson smoothed rank curves.

    X-axis: the NC metric (z-scored).
    Smoothed lines: avg_ood_rank per method via kernel regression.
    Light scatter: raw data points.
    Vertical dashed: bin boundaries.
    Coloured bands: meta-CSF selection per segment.
    """
    gdf = df[df["group"] == group_label].copy()

    # Determine which methods to plot: clique members + meta-CSF selections
    plot_methods = set(bin_methods)
    if cliques:
        for members in cliques.values():
            plot_methods.update(members)
    # Also add overall top-3
    overall_top = (
        gdf.groupby("method")["avg_ood_rank"]
        .mean()
        .sort_values()
        .head(3)
        .index
    )
    plot_methods.update(overall_top)
    plot_methods = sorted(plot_methods)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Coloured background bands for meta-CSF bins
    bin_edges = [-np.inf] + splits + [np.inf]
    x_min = gdf[feature].min()
    x_max = gdf[feature].max()
    band_colors = plt.cm.Set3(np.linspace(0, 1, len(bin_methods)))

    for idx, method in enumerate(bin_methods):
        lo = max(bin_edges[idx], x_min - 0.1)
        hi = min(bin_edges[idx + 1], x_max + 0.1)
        ax.axvspan(lo, hi, alpha=0.15, color=band_colors[idx],
                   label=f"meta→{method}")

    # Vertical bin boundaries
    for s in splits:
        ax.axvline(s, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Smoothing grid
    x_grid = np.linspace(x_min, x_max, 200)

    # Method curves
    cmap = plt.cm.tab10
    for i, method in enumerate(plot_methods):
        msub = gdf[gdf["method"] == method]
        x_raw = msub[feature].values
        y_raw = msub["avg_ood_rank"].values

        if len(x_raw) < 5:
            continue

        color = cmap(i % 10)

        # Raw scatter (faint)
        ax.scatter(
            x_raw, y_raw,
            s=8, alpha=0.15, color=color, edgecolors="none",
        )

        # Nadaraya-Watson smoothed curve
        y_smooth = _nadaraya_watson(x_raw, y_raw, x_grid, bandwidth=bandwidth)
        ax.plot(
            x_grid, y_smooth,
            linewidth=2.2, color=color, label=method, alpha=0.9,
        )

    ax.set_xlabel(f"{feature} (z-scored within dataset)", fontsize=12)
    ax.set_ylabel("avg OOD rank (lower = better)", fontsize=12)
    ax.set_title(
        f"Meta-CSF — {group_label}\n"
        f"Bands = meta-CSF selection per {feature} segment",
        fontsize=12,
    )
    ax.legend(
        fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"meta_csf_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved meta-CSF plot: meta_csf_{tag}.pdf")


def plot_regret_comparison(
    eval_df: pd.DataFrame,
    group_label: str,
    output_dir: str,
    tag: str,
) -> None:
    """Bar chart comparing meta-CSF vs best-single regret per dataset."""
    datasets = sorted(eval_df["dataset"].unique())

    meta_regret = eval_df.groupby("dataset")["meta_norm_regret"].mean()
    single_regret = eval_df.groupby("dataset")["best_single_norm_regret"].mean()

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, [meta_regret.get(d, 0) for d in datasets],
           width, label="Meta-CSF", color="steelblue")
    ax.bar(x + width / 2, [single_regret.get(d, 0) for d in datasets],
           width, label="Best single method", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Normalized regret", fontsize=11)
    ax.set_title(f"Regret comparison (LODO) — {group_label}", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"meta_csf_regret_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved regret comparison: meta_csf_regret_{tag}.pdf")


# ── Feature / bin sweep ──────────────────────────────────────────────────────
def sweep_features_and_bins(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    top_k: int = 3,
    max_bins_range: list[int] | None = None,
    cliques: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Try every (feature, max_bins) combination with LODO and report results.

    Returns a DataFrame with one row per (feature, max_bins) ranked by
    meta_mean_norm_regret.
    """
    if max_bins_range is None:
        max_bins_range = [2, 3, 4]

    avail = [f for f in nc_features if f in df.columns]
    rows = []

    for feat in avail:
        for mb in max_bins_range:
            logger.info(f"  Sweep: feature={feat}, max_bins={mb}")
            eval_df, fold_summaries = run_lodo_meta_csf(
                df, nc_features, group_label,
                top_k=top_k,
                max_n_bins=mb,
                forced_feature=feat,
                cliques=cliques,
            )
            if eval_df.empty:
                continue

            meta_wins = (eval_df["meta_augrc"] <= eval_df["best_single_augrc"]).mean()
            rows.append({
                "feature": feat,
                "max_bins": mb,
                "meta_mean_regret": eval_df["meta_regret"].mean(),
                "meta_mean_norm_regret": eval_df["meta_norm_regret"].mean(),
                "single_mean_regret": eval_df["best_single_regret"].mean(),
                "single_mean_norm_regret": eval_df["best_single_norm_regret"].mean(),
                "meta_wins_rate": meta_wins,
                "clique_hit_rate": eval_df["clique_hit"].mean() if cliques else np.nan,
                "meta_beats_single": (
                    eval_df["meta_norm_regret"].mean()
                    < eval_df["best_single_norm_regret"].mean()
                ),
            })

    sweep_df = pd.DataFrame(rows).sort_values("meta_mean_norm_regret")
    return sweep_df


# ── Multi-feature meta-CSF (decision tree) ──────────────────────────────────
def _compute_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """For each (dataset, study, dropout, reward, run), find the oracle method.

    Returns a per-instance DataFrame with NC features + oracle_method column.
    """
    instance_keys = ["dataset", "study", "dropout", "reward", "run"]
    idx = df.groupby(instance_keys)["avg_ood_rank"].idxmin()
    oracle = df.loc[idx].copy()
    oracle = oracle.rename(columns={"method": "oracle_method"})
    return oracle.reset_index(drop=True)


def build_multi_feature_meta_csf(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    max_depth: int = 3,
    min_samples_leaf: float = 0.05,
    top_methods_to_keep: int = 8,
) -> tuple[DecisionTreeClassifier, dict[int, str], pd.DataFrame, list[str]]:
    """Train a shallow decision tree that partitions NC feature space.

    Strategy
    --------
    1. For each instance, find the oracle method (argmin avg_ood_rank).
    2. Keep only the top-K most frequently winning methods as candidate
       classes; for instances where oracle is not in top-K, relabel to the
       best method within the top-K for that instance.
    3. Fit DecisionTreeClassifier on NC features → relabeled oracle.
    4. For each leaf, re-assign the best method empirically as
       argmin of mean avg_ood_rank among instances in the leaf (using the
       full, long-format df so the assignment matches the actual objective
       and decouples splitting from assignment).

    Returns
    -------
    tree : fitted DecisionTreeClassifier
    leaf_methods : dict[leaf_id → method name]
    leaf_summary : DataFrame with per-leaf stats
    feature_names : list[str] of features used
    """
    gdf = df[df["group"] == group_label].copy()
    avail = [f for f in nc_features if f in gdf.columns]

    # ── 1. Oracle per instance ──
    oracle = _compute_oracle(gdf)

    # ── 2. Top-K most frequent winners ──
    top_methods = (
        oracle["oracle_method"].value_counts().head(top_methods_to_keep).index.tolist()
    )
    logger.info(f"  Top-{top_methods_to_keep} oracle methods: {top_methods}")

    # Relabel: for rows whose oracle is not in top-K, pick the best method
    # within the top-K for that specific instance
    instance_keys = ["dataset", "study", "dropout", "reward", "run"]
    top_set = set(top_methods)
    relabel_map = {}
    for keys, inst in gdf[gdf["method"].isin(top_set)].groupby(instance_keys):
        best = inst.loc[inst["avg_ood_rank"].idxmin(), "method"]
        relabel_map[keys] = best

    def _relabel(row):
        key = tuple(row[k] for k in instance_keys)
        return relabel_map.get(key, row["oracle_method"])

    oracle["oracle_method_topk"] = oracle.apply(_relabel, axis=1)

    # Drop rows that couldn't be relabeled (rare)
    oracle = oracle.dropna(subset=["oracle_method_topk"])

    # ── 3. Train decision tree ──
    X = oracle[avail].values
    y = oracle["oracle_method_topk"].values

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=0,
    )
    tree.fit(X, y)

    # ── 4. Empirical re-assignment per leaf ──
    leaf_ids = tree.apply(X)
    oracle["leaf"] = leaf_ids

    # Join leaf ids back to the long-format df so we can compute per-method
    # mean rank within each leaf using the full data
    key_cols = instance_keys
    leaf_map = oracle.set_index(key_cols)["leaf"].to_dict()

    def _lookup_leaf(row):
        return leaf_map.get(tuple(row[k] for k in key_cols), -1)

    gdf["leaf"] = gdf.apply(_lookup_leaf, axis=1)

    leaf_methods = {}
    leaf_rows = []
    for leaf_id in sorted(gdf["leaf"].unique()):
        if leaf_id < 0:
            continue
        lsub = gdf[gdf["leaf"] == leaf_id]
        method_ranks = (
            lsub.groupby("method")["avg_ood_rank"]
            .mean()
            .sort_values()
        )
        # Restrict the assignment to the top-K candidate methods
        candidates = [m for m in method_ranks.index if m in top_set]
        if not candidates:
            best = method_ranks.index[0]
        else:
            best = candidates[0]  # already sorted
        leaf_methods[int(leaf_id)] = best

        top3 = method_ranks.head(3)
        detail = ", ".join(f"{m}({r:.2f})" for m, r in top3.items())
        leaf_rows.append({
            "leaf_id": int(leaf_id),
            "best_method": best,
            "mean_rank": method_ranks[best],
            "n_instances": lsub[instance_keys].drop_duplicates().shape[0],
            "n_rows": len(lsub),
            "methods_detail": detail,
        })
        logger.info(
            f"  Leaf {int(leaf_id)}: best={best} "
            f"(rank={method_ranks[best]:.2f}), n={len(lsub)}"
        )

    leaf_summary = pd.DataFrame(leaf_rows)
    return tree, leaf_methods, leaf_summary, avail


def run_lodo_multi_feature_meta_csf(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    max_depth: int = 3,
    min_samples_leaf: float = 0.05,
    top_methods_to_keep: int = 8,
    cliques: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """LODO evaluation of the multi-feature (decision tree) meta-CSF."""
    gdf = df[df["group"] == group_label].copy()
    datasets = sorted(gdf["dataset"].unique())

    all_results = []
    fold_summaries = []

    for held_out in datasets:
        logger.info(f"  [multi] LODO fold: held-out = {held_out}")
        train_df = gdf[gdf["dataset"] != held_out]
        test_df = gdf[gdf["dataset"] == held_out]

        if train_df.empty or test_df.empty:
            continue

        tree, leaf_methods, leaf_summary, feat_names = build_multi_feature_meta_csf(
            train_df.assign(group=group_label),
            nc_features, group_label,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            top_methods_to_keep=top_methods_to_keep,
        )

        # Best single method from training data
        train_overall = (
            train_df.groupby("method")["avg_ood_rank"]
            .mean()
            .sort_values()
        )
        best_single_method = train_overall.index[0]

        # Evaluate: for each test instance (unique key set), get the
        # NC feature vector once, predict the leaf, look up the method
        instance_keys = ["dataset", "study", "dropout", "reward", "run"]
        test_instances = test_df[instance_keys + feat_names].drop_duplicates(
            subset=instance_keys
        )
        X_test = test_instances[feat_names].values
        test_leaves = tree.apply(X_test)
        inst_to_leaf = dict(
            zip(
                [tuple(r) for r in test_instances[instance_keys].values],
                test_leaves,
            )
        )

        for keys, inst in test_df.groupby(instance_keys):
            key_dict = dict(zip(instance_keys, keys))
            leaf_id = int(inst_to_leaf.get(keys, -1))

            augrc_map = dict(zip(inst["method"], inst["augrc"]))
            oracle_method = min(augrc_map, key=augrc_map.get)
            oracle_augrc = augrc_map[oracle_method]

            single_augrc = augrc_map.get(best_single_method, np.nan)

            meta_method = leaf_methods.get(leaf_id, best_single_method)
            meta_augrc = augrc_map.get(meta_method, np.nan)
            if np.isnan(meta_augrc):
                meta_method = best_single_method
                meta_augrc = single_augrc

            meta_regret = meta_augrc - oracle_augrc
            single_regret = single_augrc - oracle_augrc if not np.isnan(single_augrc) else np.nan
            meta_norm = meta_regret / oracle_augrc if oracle_augrc else 0
            single_norm = single_regret / oracle_augrc if oracle_augrc else 0

            clique_hit = False
            if cliques and held_out in cliques:
                clique_hit = meta_method in set(cliques[held_out])

            all_results.append({
                **key_dict,
                "held_out": held_out,
                "leaf": leaf_id,
                "oracle_method": oracle_method,
                "oracle_augrc": oracle_augrc,
                "best_single_method": best_single_method,
                "best_single_augrc": single_augrc,
                "best_single_regret": single_regret,
                "best_single_norm_regret": single_norm,
                "meta_method": meta_method,
                "meta_augrc": meta_augrc,
                "meta_regret": meta_regret,
                "meta_norm_regret": meta_norm,
                "clique_hit": clique_hit,
            })

        fold_results = [r for r in all_results if r["held_out"] == held_out]
        fdf = pd.DataFrame(fold_results)
        meta_wins = (fdf["meta_augrc"] <= fdf["best_single_augrc"]).mean()
        # Feature importances (top 3)
        fi = dict(zip(feat_names, tree.feature_importances_))
        top_fi = sorted(fi.items(), key=lambda x: -x[1])[:3]
        fi_str = ", ".join(f"{k}={v:.2f}" for k, v in top_fi if v > 0)
        fold_summaries.append({
            "held_out": held_out,
            "n_leaves": tree.get_n_leaves(),
            "tree_depth": tree.get_depth(),
            "top_features": fi_str,
            "meta_mean_regret": fdf["meta_regret"].mean(),
            "meta_mean_norm_regret": fdf["meta_norm_regret"].mean(),
            "single_mean_regret": fdf["best_single_regret"].mean(),
            "single_mean_norm_regret": fdf["best_single_norm_regret"].mean(),
            "meta_wins_rate": meta_wins,
            "clique_hit_rate": fdf["clique_hit"].mean() if cliques else np.nan,
        })
        logger.info(
            f"    {held_out}: meta regret="
            f"{fdf['meta_regret'].mean():.4f} "
            f"({fdf['meta_norm_regret'].mean():.1%}), "
            f"single regret={fdf['best_single_regret'].mean():.4f} "
            f"({fdf['best_single_norm_regret'].mean():.1%}), "
            f"meta_wins={meta_wins:.1%}, top_feats=[{fi_str}]"
        )

    return pd.DataFrame(all_results), fold_summaries


def plot_tree_meta_csf(
    tree: DecisionTreeClassifier,
    feat_names: list[str],
    leaf_methods: dict[int, str],
    output_dir: str,
    tag: str,
) -> None:
    """Visualise the decision tree with leaf method assignments."""
    fig, ax = plt.subplots(figsize=(14, 8))
    class_names = [str(c) for c in tree.classes_]
    plot_tree(
        tree,
        feature_names=feat_names,
        class_names=class_names,
        filled=True, rounded=True, proportion=True,
        fontsize=8, ax=ax,
    )
    # Title includes leaf → empirical method mapping
    leaf_str = " | ".join(
        f"L{k}→{v}" for k, v in sorted(leaf_methods.items())
    )
    ax.set_title(
        f"Multi-feature meta-CSF tree ({tag})\n"
        f"Empirical leaf assignments: {leaf_str}",
        fontsize=10,
    )
    fig.savefig(
        os.path.join(output_dir, f"meta_csf_tree_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved tree plot: meta_csf_tree_{tag}.pdf")


def plot_decision_surface_2d(
    df: pd.DataFrame,
    tree: DecisionTreeClassifier,
    feat_names: list[str],
    leaf_methods: dict[int, str],
    group_label: str,
    output_dir: str,
    tag: str,
) -> None:
    """2D decision surface using the two most important tree features.

    Coloured by the method assigned to each leaf; held-out scatter overlayed.
    """
    importances = tree.feature_importances_
    if (importances > 0).sum() < 2:
        logger.info("  Skipping 2D surface: tree uses fewer than 2 features")
        return
    top2 = np.argsort(importances)[::-1][:2]
    f1, f2 = feat_names[top2[0]], feat_names[top2[1]]

    gdf = df[df["group"] == group_label].copy()

    # Build a dense grid — only f1, f2 vary; all other features are fixed
    # at the training median so the tree can process a full feature vector
    medians = gdf[feat_names].median().values
    n = 200
    x1 = np.linspace(gdf[f1].min(), gdf[f1].max(), n)
    x2 = np.linspace(gdf[f2].min(), gdf[f2].max(), n)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.tile(medians, (n * n, 1))
    grid[:, top2[0]] = X1.ravel()
    grid[:, top2[1]] = X2.ravel()

    leaves_grid = tree.apply(grid).reshape(n, n)
    # Map leaf → method → int for colouring
    unique_methods = sorted(set(leaf_methods.values()))
    method_to_int = {m: i for i, m in enumerate(unique_methods)}
    method_grid = np.vectorize(
        lambda lid: method_to_int.get(leaf_methods.get(int(lid), ""), -1)
    )(leaves_grid)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.Set3
    im = ax.imshow(
        method_grid,
        extent=(x1.min(), x1.max(), x2.min(), x2.max()),
        origin="lower", aspect="auto", alpha=0.45, cmap=cmap,
        vmin=0, vmax=max(len(unique_methods) - 1, 1),
    )
    # Scatter oracle points coloured by oracle method (only top methods)
    oracle = _compute_oracle(gdf)
    keep = oracle["oracle_method"].isin(unique_methods)
    for m in unique_methods:
        sub = oracle[keep & (oracle["oracle_method"] == m)]
        if sub.empty:
            continue
        ax.scatter(
            sub[f1], sub[f2], s=14,
            color=cmap(method_to_int[m] / max(len(unique_methods) - 1, 1)),
            edgecolors="black", linewidths=0.4,
            label=m,
        )

    ax.set_xlabel(f"{f1} (z-scored)", fontsize=11)
    ax.set_ylabel(f"{f2} (z-scored)", fontsize=11)
    ax.set_title(
        f"Multi-feature meta-CSF decision surface — {group_label}\n"
        f"(other NC features fixed at their median)",
        fontsize=11,
    )
    ax.legend(
        fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1),
        title="oracle / leaf method",
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"meta_csf_surface_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved 2D decision surface: meta_csf_surface_{tag}.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Meta-CSF: NC-metric-guided method selector",
    )
    parser.add_argument(
        "--nc-file", type=str,
        default="neural_collapse_metrics/nc_metrics.csv",
    )
    parser.add_argument(
        "--scores-dir", type=str, default="scores_risk",
    )
    parser.add_argument(
        "--clip-dir", type=str, default="clip_scores",
    )
    parser.add_argument(
        "--backbone", type=str, required=True, choices=["Conv", "ViT"],
    )
    parser.add_argument(
        "--study", type=str, default=None,
    )
    parser.add_argument(
        "--clique-file", type=str, default=None,
    )
    parser.add_argument(
        "--filter-methods", action="store_true",
    )
    parser.add_argument(
        "--papyan-only", action="store_true",
    )
    parser.add_argument(
        "--groups", type=str, nargs="*", default=None,
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Rank threshold for IV feature selection (default: 3)",
    )
    parser.add_argument(
        "--max-bins", type=int, default=4,
        help="Max bins for the meta-CSF partitioning (default: 4)",
    )
    parser.add_argument(
        "--feature", type=str, default=None,
        help="Force a specific NC feature (skip IV selection)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="meta_csf_outputs",
    )
    parser.add_argument(
        "--mcd", type=str, default="False",
    )
    parser.add_argument(
        "--score-metric", type=str, default="AUGRC",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep all features × bin counts (2,3,4) with LODO",
    )
    parser.add_argument(
        "--bandwidth", type=float, default=None,
        help="Nadaraya-Watson bandwidth for plot smoothing (default: Silverman)",
    )
    parser.add_argument(
        "--multi-feature", action="store_true",
        help="Use decision-tree-based multi-feature meta-CSF instead of "
             "single-feature binning",
    )
    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Max depth of the decision tree (multi-feature mode)",
    )
    parser.add_argument(
        "--min-samples-leaf", type=float, default=0.05,
        help="Min fraction of samples per leaf (multi-feature mode)",
    )
    parser.add_argument(
        "--top-methods", type=int, default=8,
        help="Number of top-winning methods to keep as tree classes "
             "(multi-feature mode)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    arch = ARCH_MAP[args.backbone]
    nc = load_nc_metrics(args.nc_file)
    nc = nc[nc["architecture"] == arch]
    if args.study:
        nc_study = STUDY_MAP.get(args.study, args.study)
        nc = nc[nc["study"] == nc_study]
    logger.info(f"NC metrics: {len(nc)} rows (arch={arch})")

    metric_pool = PAPYAN_NC_METRICS if args.papyan_only else NC_METRICS
    nc_features = [m for m in metric_pool if m in nc.columns]
    logger.info(f"NC features ({len(nc_features)}): {nc_features}")

    scores = load_scores(
        args.scores_dir, args.backbone, args.mcd,
        args.clip_dir, SOURCES,
        filter_methods=args.filter_methods,
        study_filter=args.study,
    )
    reg_df = build_regression_dataset(nc, scores, nc_features)

    # ── Z-score NC features within each dataset ──
    logger.info("Z-scoring NC features within each dataset ...")
    reg_df = zscore_nc_features(reg_df, nc_features)

    # ── Cliques ──
    clique_data = None
    if args.clique_file and os.path.exists(args.clique_file):
        with open(args.clique_file) as f:
            raw = json.load(f)
        raw.pop("_ranks", None)
        clique_data = raw

    # ── Run per group ──
    groups_to_run = args.groups or ["near", "mid", "far", "all"]
    study_tag = f"_{args.study}" if args.study else ""
    file_prefix = (
        f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"
    )

    all_fold_summaries = []
    for group_label in groups_to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"OOD group: {group_label}")

        # Cliques for this group
        cliques_for_group = None
        if clique_data:
            cliques_for_group = {}
            for src, gmap in clique_data.items():
                if group_label in gmap:
                    cliques_for_group[src] = gmap[group_label]

        tag = f"{file_prefix}_group_{group_label}"
        if args.multi_feature:
            tag = f"multi_{tag}"

        # ── Multi-feature mode (decision tree) ──
        if args.multi_feature:
            logger.info(f"  [multi-feature] max_depth={args.max_depth}, "
                        f"min_samples_leaf={args.min_samples_leaf}, "
                        f"top_methods={args.top_methods}")

            eval_df_m, fold_summaries_m = run_lodo_multi_feature_meta_csf(
                reg_df, nc_features, group_label,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                top_methods_to_keep=args.top_methods,
                cliques=cliques_for_group,
            )

            if eval_df_m.empty:
                continue

            eval_df_m.to_csv(
                os.path.join(args.output_dir, f"meta_csf_eval_{tag}.csv"),
                index=False,
            )
            folds_df = pd.DataFrame(fold_summaries_m)
            folds_df["group"] = group_label
            folds_df.to_csv(
                os.path.join(args.output_dir, f"meta_csf_folds_{tag}.csv"),
                index=False,
            )
            all_fold_summaries.append(folds_df)

            meta_wins = (eval_df_m["meta_augrc"] <= eval_df_m["best_single_augrc"]).mean()
            logger.info(
                f"  Overall {group_label} (multi): "
                f"meta regret={eval_df_m['meta_regret'].mean():.4f} "
                f"({eval_df_m['meta_norm_regret'].mean():.1%}), "
                f"single regret={eval_df_m['best_single_regret'].mean():.4f} "
                f"({eval_df_m['best_single_norm_regret'].mean():.1%}), "
                f"meta_wins={meta_wins:.1%}"
            )

            # Fit a single tree on ALL data for the illustrative plot
            tree_full, leaf_methods_full, leaf_summary_full, feat_names = (
                build_multi_feature_meta_csf(
                    reg_df, nc_features, group_label,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    top_methods_to_keep=args.top_methods,
                )
            )
            leaf_summary_full.to_csv(
                os.path.join(args.output_dir, f"meta_csf_leaves_{tag}.csv"),
                index=False,
            )
            # Dump human-readable tree rules
            rules_txt = export_text(
                tree_full, feature_names=feat_names, decimals=3,
            )
            rules_path = os.path.join(
                args.output_dir, f"meta_csf_rules_{tag}.txt",
            )
            with open(rules_path, "w") as f:
                f.write(f"Tree rules for {group_label}\n")
                f.write(f"Leaf → method assignments: {leaf_methods_full}\n\n")
                f.write(rules_txt)
            logger.info(f"  Saved rules: {rules_path}")

            plot_tree_meta_csf(
                tree_full, feat_names, leaf_methods_full,
                args.output_dir, tag,
            )
            plot_decision_surface_2d(
                reg_df, tree_full, feat_names, leaf_methods_full,
                group_label, args.output_dir, tag,
            )
            plot_regret_comparison(
                eval_df_m, group_label, args.output_dir, tag,
            )
            continue

        # ── Sweep mode: try all features × bin counts ──
        if args.sweep:
            logger.info("  Running feature × bin sweep ...")
            sweep_df = sweep_features_and_bins(
                reg_df, nc_features, group_label,
                top_k=args.top_k,
                cliques=cliques_for_group,
            )
            sweep_path = os.path.join(
                args.output_dir, f"meta_csf_sweep_{tag}.csv",
            )
            sweep_df.to_csv(sweep_path, index=False)
            logger.info(f"  Sweep results ({group_label}):")
            logger.info(f"\n{sweep_df.head(10).to_string(index=False)}")
            # Use the best (feature, max_bins) for the main evaluation
            best_row = sweep_df.iloc[0]
            best_sweep_feature = best_row["feature"]
            best_sweep_bins = int(best_row["max_bins"])
            logger.info(
                f"  Best sweep: feature={best_sweep_feature}, "
                f"max_bins={best_sweep_bins}, "
                f"meta_norm_regret={best_row['meta_mean_norm_regret']:.1%}"
            )

        # LODO evaluation
        eval_df, fold_summaries = run_lodo_meta_csf(
            reg_df, nc_features, group_label,
            top_k=args.top_k,
            max_n_bins=args.max_bins,
            forced_feature=args.feature,
            cliques=cliques_for_group,
        )

        if eval_df.empty:
            continue

        eval_df.to_csv(
            os.path.join(args.output_dir, f"meta_csf_eval_{tag}.csv"),
            index=False,
        )

        folds_df = pd.DataFrame(fold_summaries)
        folds_df["group"] = group_label
        folds_df.to_csv(
            os.path.join(args.output_dir, f"meta_csf_folds_{tag}.csv"),
            index=False,
        )
        all_fold_summaries.append(folds_df)

        # Overall summary for this group
        meta_wins = (eval_df["meta_augrc"] <= eval_df["best_single_augrc"]).mean()
        logger.info(
            f"  Overall {group_label}: "
            f"meta regret={eval_df['meta_regret'].mean():.4f} "
            f"({eval_df['meta_norm_regret'].mean():.1%}), "
            f"single regret={eval_df['best_single_regret'].mean():.4f} "
            f"({eval_df['best_single_norm_regret'].mean():.1%}), "
            f"meta_wins={meta_wins:.1%}"
        )

        # Plot: build meta-CSF on ALL data for the visualisation
        # (the LODO eval above is the honest evaluation;
        #  this full-data fit is just for the illustrative plot)
        if args.feature:
            plot_feature = args.feature
        else:
            plot_feature, _ = select_top_feature(
                reg_df, nc_features, group_label, top_k=args.top_k,
            )

        splits_full, bin_methods_full, bin_summary_full = build_meta_csf(
            reg_df, plot_feature, group_label,
            max_n_bins=args.max_bins,
        )
        bin_summary_full.to_csv(
            os.path.join(args.output_dir, f"meta_csf_bins_{tag}.csv"),
            index=False,
        )

        plot_meta_csf(
            reg_df, plot_feature, group_label,
            splits_full, bin_methods_full,
            cliques=cliques_for_group,
            output_dir=args.output_dir,
            tag=tag,
        )
        plot_regret_comparison(
            eval_df, group_label, args.output_dir, tag,
        )

    # Global summary
    if all_fold_summaries:
        all_folds = pd.concat(all_fold_summaries, ignore_index=True)
        summary = all_folds.groupby("group").agg(
            meta_mean_norm_regret=("meta_mean_norm_regret", "mean"),
            single_mean_norm_regret=("single_mean_norm_regret", "mean"),
            meta_wins_rate=("meta_wins_rate", "mean"),
            clique_hit_rate=("clique_hit_rate", "mean"),
        ).reset_index()

        summary_path = os.path.join(
            args.output_dir, f"meta_csf_summary_{file_prefix}.csv",
        )
        summary.to_csv(summary_path, index=False)
        logger.info(f"\n{summary.to_string(index=False)}")
        logger.info(f"Saved summary: {summary_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

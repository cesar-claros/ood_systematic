"""Per-method AUGRC regression from Neural Collapse metrics.

Algorithm selection approach: for each OOD detection method, train a
Random Forest regressor that predicts the method's AUGRC from NC metrics
of the underlying model.  Recommend the method(s) with the lowest
predicted AUGRC on held-out source datasets (LODO evaluation).

Pipeline
--------
1. Load NC metrics  (features)
2. Load AUGRC scores from ``scores_all`` files  (targets)
3. Load CLIP groupings to aggregate scores by OOD proximity group
4. Join NC metrics with group-averaged AUGRC
5. LODO evaluation:
   a. Per-method RF regressors trained on N-1 source datasets
   b. Predict AUGRC for every method on the held-out source dataset
   c. Rank methods by predicted AUGRC  (lower = better)
   d. Evaluate against actual ranking / clique membership
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from scipy.stats import spearmanr, kendalltau
from loguru import logger

from nc_regime_analysis import NC_METRICS, load_nc_metrics

# ── Constants ─────────────────────────────────────────────────────────────────
PAPYAN_NC_METRICS = [
    "var_collapse",
    "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc",
    "max_equiangular_uc", "max_equiangular_wc",
    "self_duality",
]

ARCH_MAP = {"Conv": "VGG13", "ViT": "ViT"}

STUDY_MAP = {
    "confidnet": "confidnet",
    "devries": "devries",
    "dg": "dg",
    "modelvit": "vit",
}

GROUP_NAMES = {0: "test", 1: "near", 2: "mid", 3: "far"}

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]

# Methods containing these substrings are excluded unless in the keep set
_FILTER_EXCEPTIONS = {
    "KPCA RecError global",
    "PCA RecError global",
    "MCD-KPCA RecError global",
    "MCD-PCA RecError global",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def _parse_reward(rew_str: str) -> float:
    """Convert score-file reward string to numeric (e.g. 'rew2.2' -> 2.2)."""
    return float(rew_str.replace("rew", ""))


def _parse_dropout(do_str: str) -> bool:
    """Convert score-file dropout string to bool (e.g. 'do1' -> True)."""
    return do_str == "do1"


def load_ood_groups(clip_dir: str, source: str) -> dict[str, list[str]]:
    """Return {group_name: [ood_dataset_col_names]} from CLIP CSV."""
    path = os.path.join(clip_dir, f"clip_distances_{source}.csv")
    clip = pd.read_csv(path, header=[0, 1])
    clip.columns = clip.columns.droplevel(0)
    clip = clip.rename(
        {"Unnamed: 0_level_1": "dataset", "Unnamed: 5_level_1": "group"},
        axis="columns",
    )
    groups: dict[str, list[str]] = {}
    for _, row in clip.iterrows():
        g = int(row["group"])
        name = GROUP_NAMES.get(g, str(g))
        if name == "test":
            continue
        groups.setdefault(name, []).append(row["dataset"])
    groups["all"] = [d for ds in groups.values() for d in ds]
    return groups


def load_scores(
    scores_dir: str,
    backbone: str,
    mcd: str,
    clip_dir: str,
    sources: list[str],
    filter_methods: bool = False,
    study_filter: str | None = None,
) -> pd.DataFrame:
    """Load AUGRC scores, rank methods per OOD dataset, then average ranks.

    Returns long-format DataFrame with columns:
        source_dataset, study, dropout, reward, run, method, group,
        augrc (mean AUGRC over OOD datasets in group),
        avg_ood_rank (mean rank across OOD datasets, scale [1, n_methods])
    """
    frames = []
    for source in sources:
        fname = f"scores_all_AUGRC_MCD-{mcd}_{backbone}_{source}.csv"
        path = os.path.join(scores_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Missing scores file: {path}")
            continue

        df = pd.read_csv(path)
        df = df.rename(columns={"model": "study_raw", "drop out": "dropout_raw"})
        df["study"] = df["study_raw"].map(STUDY_MAP)
        df["dropout"] = df["dropout_raw"].map(_parse_dropout)
        df["reward"] = df["reward"].map(_parse_reward)

        if study_filter:
            nc_study = STUDY_MAP.get(study_filter, study_filter)
            df = df[df["study"] == nc_study]

        if filter_methods:
            mask = df["methods"].str.contains(
                "global|class", case=False, na=False
            )
            mask &= ~df["methods"].isin(_FILTER_EXCEPTIONS)
            df = df[~mask]

        ood_groups = load_ood_groups(clip_dir, source)

        meta_cols = ["study", "dropout", "reward", "run", "methods"]
        ood_cols = [
            c for c in df.columns
            if c not in meta_cols
            and c not in ["study_raw", "dropout_raw", "test"]
        ]

        rank_cols_keys = ["study", "dropout", "reward", "run"]
        for group_name, group_datasets in ood_groups.items():
            valid = [d for d in group_datasets if d in ood_cols]
            if not valid:
                continue
            # Rank methods per OOD dataset, then average ranks
            per_ood_ranks = []
            for ood_ds in valid:
                tmp = df[rank_cols_keys + ["methods", ood_ds]].copy()
                tmp = tmp.rename(columns={ood_ds: "augrc_ood", "methods": "method"})
                tmp["ood_rank"] = tmp.groupby(rank_cols_keys)["augrc_ood"].rank(
                    method="average", ascending=True,
                )
                per_ood_ranks.append(
                    tmp[rank_cols_keys + ["method", "ood_rank"]]
                )
            merged = per_ood_ranks[0]
            for r in per_ood_ranks[1:]:
                merged = merged.merge(
                    r, on=rank_cols_keys + ["method"], suffixes=("", "_dup"),
                )
            rank_columns = [c for c in merged.columns if c.startswith("ood_rank")]
            sub = df[meta_cols].copy()
            sub = sub.rename(columns={"methods": "method"})
            sub["augrc"] = df[valid].mean(axis=1)
            sub["avg_ood_rank"] = merged[rank_columns].mean(axis=1).values
            sub["group"] = group_name
            sub["source_dataset"] = source
            frames.append(sub)

    if not frames:
        raise ValueError("No scores loaded")

    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"methods": "method"})

    # avg_ood_rank is already computed per group in the loop above

    logger.info(
        f"Loaded scores: {len(result)} rows, "
        f"{result['source_dataset'].nunique()} sources, "
        f"{result['method'].nunique()} methods, "
        f"{result['group'].nunique()} groups"
    )
    return result


def build_regression_dataset(
    nc: pd.DataFrame,
    scores: pd.DataFrame,
    nc_features: list[str],
) -> pd.DataFrame:
    """Join NC metrics with AUGRC scores.

    Each row is (model_instance, method, group) with NC features and AUGRC.
    """
    nc_copy = nc.copy()
    nc_copy["dataset"] = nc_copy["dataset"].replace(
        {"supercifar": "supercifar100"}
    )
    nc_copy["dropout"] = nc_copy["dropout"].map(
        {True: True, False: False, "True": True, "False": False}
    )

    join_keys = ["dataset", "study", "dropout", "run", "reward"]

    scores_j = scores.rename(columns={"source_dataset": "dataset"})

    avail_features = [f for f in nc_features if f in nc_copy.columns]
    extra_features = [f for f in avail_features if f not in join_keys]
    nc_subset = nc_copy[join_keys + extra_features].drop_duplicates()

    merged = scores_j.merge(nc_subset, on=join_keys, how="inner")

    merged["dropout"] = merged["dropout"].map(
        {True: 1, False: 0, "True": 1, "False": 0}
    ).astype(int)

    logger.info(
        f"Regression dataset: {len(merged)} rows after join "
        f"({merged['dataset'].nunique()} datasets, "
        f"{merged['method'].nunique()} methods)"
    )
    return merged


# ── Evaluation ────────────────────────────────────────────────────────────────
def _make_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def run_lodo_regression(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    target_col: str = "augrc",
    cliques: dict[str, list[str]] | None = None,
) -> dict:
    """Leave-One-Dataset-Out evaluation for a single OOD group.

    Parameters
    ----------
    target_col : str
        Column to regress on.  ``"augrc"`` for raw scores,
        ``"pct_rank"`` for percentile ranks (scale-free).
    """
    gdf = df[df["group"] == group_label].copy()
    datasets = sorted(gdf["dataset"].unique())

    if len(datasets) < 2:
        logger.warning(f"Only {len(datasets)} dataset(s) for group "
                       f"'{group_label}', skipping LODO")
        return {}

    avail = [f for f in nc_features if f in gdf.columns]

    fold_results = []
    per_method_preds = []

    for held_out in datasets:
        train_df = gdf[gdf["dataset"] != held_out]
        test_df = gdf[gdf["dataset"] == held_out]

        common = sorted(
            set(train_df["method"].unique())
            & set(test_df["method"].unique())
        )
        if len(common) < 2:
            logger.warning(
                f"LODO {held_out}: <2 common methods, skipping"
            )
            continue

        actual_target = {}
        predicted_target = {}
        actual_augrc = {}

        for method in common:
            tr = train_df[train_df["method"] == method]
            te = test_df[test_df["method"] == method]

            X_tr, y_tr = tr[avail].values, tr[target_col].values
            X_te, y_te = te[avail].values, te[target_col].values

            rf = _make_rf()
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_te)

            actual_target[method] = y_te.mean()
            predicted_target[method] = y_pred.mean()
            actual_augrc[method] = te["augrc"].values.mean()

            for i in range(len(te)):
                per_method_preds.append({
                    "dataset": held_out,
                    "method": method,
                    f"actual_{target_col}": y_te[i],
                    f"predicted_{target_col}": y_pred[i],
                    "actual_augrc": te["augrc"].values[i],
                })

        # Rank correlation on the predicted target values
        actual_rank = pd.Series(actual_target).rank()
        predicted_rank = pd.Series(predicted_target).rank()

        spearman_r, spearman_p = spearmanr(actual_rank, predicted_rank)
        kendall_t, kendall_p = kendalltau(actual_rank, predicted_rank)

        # Recommendation: pick method with lowest predicted target
        actual_sorted = sorted(actual_target, key=actual_target.get)
        predicted_sorted = sorted(predicted_target, key=predicted_target.get)

        top1_hit = predicted_sorted[0] == actual_sorted[0]
        top3_actual = set(actual_sorted[:3])
        top3_predicted = set(predicted_sorted[:3])
        top3_overlap = len(top3_actual & top3_predicted) / len(
            top3_actual | top3_predicted
        )

        # Regret always in AUGRC units
        actual_best = actual_sorted[0]
        predicted_best = predicted_sorted[0]
        best_augrc = actual_augrc[actual_best]
        recommended_augrc = actual_augrc[predicted_best]
        regret = recommended_augrc - best_augrc

        # Clique-based evaluation
        clique_hit = False
        clique_top3_hit = False
        clique_regret = np.nan
        clique_norm_regret = np.nan
        within_clique_range = False
        clique_top3_jaccard = np.nan
        if cliques and held_out in cliques:
            clique_methods = set(cliques[held_out])
            clique_hit = predicted_sorted[0] in clique_methods
            clique_top3_hit = bool(top3_predicted & clique_methods)
            clique_augrc_vals = [
                actual_augrc[m] for m in clique_methods
                if m in actual_augrc
            ]
            if clique_augrc_vals:
                mean_clique = np.mean(clique_augrc_vals)
                worst_clique = max(clique_augrc_vals)
                clique_regret = recommended_augrc - mean_clique
                clique_norm_regret = (
                    clique_regret / mean_clique if mean_clique else 0
                )
                within_clique_range = recommended_augrc <= worst_clique
            clique_top3_jaccard = (
                len(top3_predicted & clique_methods)
                / len(top3_predicted | clique_methods)
                if (top3_predicted | clique_methods) else 0
            )

        fold_result = {
            "dataset": held_out,
            "n_methods": len(common),
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "kendall_tau": kendall_t,
            "kendall_p": kendall_p,
            "actual_best": actual_best,
            "predicted_best": predicted_best,
            "top1_hit": top1_hit,
            "top3_jaccard": top3_overlap,
            "regret": regret,
            "norm_regret": regret / best_augrc if best_augrc else 0,
            "clique_hit": clique_hit,
            "clique_top3_hit": clique_top3_hit,
            "clique_regret": clique_regret,
            "clique_norm_regret": clique_norm_regret,
            "within_clique_range": within_clique_range,
            "clique_top3_jaccard": clique_top3_jaccard,
            "actual_top3": "|".join(actual_sorted[:3]),
            "predicted_top3": "|".join(predicted_sorted[:3]),
        }
        fold_results.append(fold_result)

        clique_str = (
            f"cq_regret={clique_regret:.2f} "
            f"({clique_norm_regret:.1%}), "
            f"in_range={'Y' if within_clique_range else 'N'}, "
            f"cq_J={clique_top3_jaccard:.3f}"
            if not np.isnan(clique_regret)
            else "no clique"
        )
        logger.info(
            f"  LODO {held_out}: "
            f"rho={spearman_r:.3f}, tau={kendall_t:.3f}, "
            f"top1={'Y' if top1_hit else 'N'}, "
            f"regret={regret:.2f} ({fold_result['norm_regret']:.1%}), "
            f"clique={'Y' if clique_hit else 'N'}, "
            f"cq_top3={'Y' if clique_top3_hit else 'N'}, "
            f"{clique_str}"
        )

    if not fold_results:
        return {}

    fold_df = pd.DataFrame(fold_results)
    preds_df = pd.DataFrame(per_method_preds)

    has_cliques = fold_df["clique_regret"].notna().any()
    summary = {
        "group": group_label,
        "target": target_col,
        "n_folds": len(fold_df),
        "mean_spearman": fold_df["spearman_r"].mean(),
        "mean_kendall": fold_df["kendall_tau"].mean(),
        "top1_accuracy": fold_df["top1_hit"].mean(),
        "mean_top3_jaccard": fold_df["top3_jaccard"].mean(),
        "mean_regret": fold_df["regret"].mean(),
        "mean_norm_regret": fold_df["norm_regret"].mean(),
        "clique_hit_rate": fold_df["clique_hit"].mean(),
        "clique_top3_hit_rate": (
            fold_df["clique_top3_hit"].mean() if has_cliques else np.nan
        ),
        "mean_clique_regret": (
            fold_df["clique_regret"].mean() if has_cliques else np.nan
        ),
        "mean_clique_norm_regret": (
            fold_df["clique_norm_regret"].mean() if has_cliques else np.nan
        ),
        "within_clique_range_rate": (
            fold_df["within_clique_range"].mean() if has_cliques else np.nan
        ),
        "mean_clique_top3_jaccard": (
            fold_df["clique_top3_jaccard"].mean() if has_cliques else np.nan
        ),
    }

    clique_log = ""
    if has_cliques:
        clique_log = (
            f", cq_regret={summary['mean_clique_regret']:.2f} "
            f"({summary['mean_clique_norm_regret']:.1%}), "
            f"in_range={summary['within_clique_range_rate']:.1%}, "
            f"cq_J={summary['mean_clique_top3_jaccard']:.3f}"
        )
    logger.info(
        f"  Overall ({target_col}): "
        f"rho={summary['mean_spearman']:.3f}, "
        f"tau={summary['mean_kendall']:.3f}, "
        f"top1={summary['top1_accuracy']:.1%}, "
        f"top3_J={summary['mean_top3_jaccard']:.3f}, "
        f"regret={summary['mean_regret']:.2f} "
        f"({summary['mean_norm_regret']:.1%}), "
        f"clique={summary['clique_hit_rate']:.1%}, "
        f"cq_top3={summary['clique_top3_hit_rate']:.1%}"
        f"{clique_log}"
    )

    return {
        "summary": summary,
        "folds": fold_df,
        "predictions": preds_df,
    }


def compute_feature_importance(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
) -> pd.DataFrame:
    """Train RF regressors on full data and report feature importance."""
    gdf = df[df["group"] == group_label]
    avail = [f for f in nc_features if f in gdf.columns]
    methods = sorted(gdf["method"].unique())

    rows = []
    for method in methods:
        sub = gdf[gdf["method"] == method]
        X = sub[avail].values
        y = sub["augrc"].values
        if len(X) < 5:
            continue
        rf = _make_rf()
        rf.fit(X, y)
        for feat, imp in zip(avail, rf.feature_importances_):
            rows.append({
                "method": method,
                "feature": feat,
                "importance": imp,
            })

    imp_df = pd.DataFrame(rows)
    avg_imp = (
        imp_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg_imp.columns = ["feature", "mean_importance"]
    avg_imp["rank"] = range(1, len(avg_imp) + 1)
    return avg_imp


def compute_shap_analysis(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    target_col: str = "avg_ood_rank",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train per-method RFs on full data and compute SHAP values.

    Returns
    -------
    shap_long : DataFrame
        One row per (method, sample, feature) with columns:
        method, feature, shap_value, feature_value
    shap_summary : DataFrame
        One row per (method, feature) with columns:
        method, feature, mean_abs_shap, mean_shap
    """
    gdf = df[df["group"] == group_label]
    avail = [f for f in nc_features if f in gdf.columns]
    methods = sorted(gdf["method"].unique())

    long_rows = []
    summary_rows = []

    for method in methods:
        sub = gdf[gdf["method"] == method]
        X = sub[avail].values
        y = sub[target_col].values
        if len(X) < 5:
            continue

        rf = _make_rf()
        rf.fit(X, y)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)

        for i in range(len(X)):
            for j, feat in enumerate(avail):
                long_rows.append({
                    "method": method,
                    "feature": feat,
                    "shap_value": shap_values[i, j],
                    "feature_value": X[i, j],
                })

        for j, feat in enumerate(avail):
            vals = shap_values[:, j]
            summary_rows.append({
                "method": method,
                "feature": feat,
                "mean_abs_shap": np.mean(np.abs(vals)),
                "mean_shap": np.mean(vals),
            })

    shap_long = pd.DataFrame(long_rows)
    shap_summary = pd.DataFrame(summary_rows)

    # Log top features per method
    for method in methods:
        msub = shap_summary[shap_summary["method"] == method].sort_values(
            "mean_abs_shap", ascending=False,
        )
        top = msub.head(3)
        feats = ", ".join(
            f"{r['feature']}({r['mean_shap']:+.4f})"
            for _, r in top.iterrows()
        )
        logger.info(f"  SHAP {method}: {feats}")

    # Global summary (averaged across methods)
    global_imp = (
        shap_summary.groupby("feature")["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
    )
    logger.info(
        f"  SHAP global top features: "
        + ", ".join(f"{f}={v:.4f}" for f, v in global_imp.head(5).items())
    )

    return shap_long, shap_summary


def compute_partial_dependence(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    target_col: str = "avg_ood_rank",
    grid_resolution: int = 50,
) -> pd.DataFrame:
    """Train per-method RFs on full data and compute partial dependence.

    Returns
    -------
    pdp_df : DataFrame
        Columns: method, feature, grid_value, pdp_value
    """
    gdf = df[df["group"] == group_label]
    avail = [f for f in nc_features if f in gdf.columns]
    methods = sorted(gdf["method"].unique())

    pdp_rows = []
    for method in methods:
        sub = gdf[gdf["method"] == method]
        X = sub[avail].values
        y = sub[target_col].values
        if len(X) < 5:
            continue
        rf = _make_rf()
        rf.fit(X, y)
        for j, feat in enumerate(avail):
            result = partial_dependence(
                rf, X, features=[j], grid_resolution=grid_resolution,
                kind="average",
            )
            grid = result["grid_values"][0]
            values = result["average"][0]
            for g, v in zip(grid, values):
                pdp_rows.append({
                    "method": method,
                    "feature": feat,
                    "grid_value": g,
                    "pdp_value": v,
                })

    return pd.DataFrame(pdp_rows)


def plot_nc_clique_summary(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
    shap_summary: pd.DataFrame,
    cliques: dict[str, list[str]] | None,
    target_col: str = "avg_ood_rank",
    output_dir: str = "regression_outputs",
    tag: str = "",
) -> None:
    """Summary figure linking NC metric profiles to top-clique methods.

    Layout (two panels):
      Left  — Heatmap: NC metrics (rows) × source datasets (cols),
              z-scored mean values.  Clique members annotated below.
      Right — Horizontal bars: mean |SHAP| of clique methods per NC metric,
              coloured by sign (negative = helps method, positive = hurts).
    """
    import seaborn as sns

    gdf = df[df["group"] == group_label]
    avail = [f for f in nc_features if f in gdf.columns]
    datasets = sorted(gdf["dataset"].unique())

    # ── NC metric means per source dataset ──
    nc_means = (
        gdf.groupby("dataset")[avail]
        .mean()
        .reindex(datasets)
    )
    # z-score across datasets for visual comparability
    nc_z = (nc_means - nc_means.mean()) / nc_means.std().replace(0, 1)
    nc_z = nc_z.T  # rows = features, cols = datasets

    # ── Clique members per source dataset ──
    clique_labels = {}
    clique_methods_all = set()
    for ds in datasets:
        if cliques and ds in cliques:
            members = cliques[ds]
            clique_labels[ds] = "\n".join(members)
            clique_methods_all.update(members)
        else:
            clique_labels[ds] = "—"

    # ── SHAP summary for clique methods only ──
    if not shap_summary.empty and clique_methods_all:
        clique_shap = shap_summary[
            shap_summary["method"].isin(clique_methods_all)
        ]
        # Mean signed SHAP and mean |SHAP| per feature across clique methods
        shap_agg = (
            clique_shap.groupby("feature")
            .agg(
                mean_abs_shap=("mean_abs_shap", "mean"),
                mean_shap=("mean_shap", "mean"),
            )
            .reindex(avail)
            .fillna(0)
        )
    else:
        shap_agg = pd.DataFrame(
            {"mean_abs_shap": 0, "mean_shap": 0}, index=avail,
        )

    # ── Figure ──
    fig, (ax_heat, ax_shap) = plt.subplots(
        1, 2, figsize=(4 + 2.2 * len(datasets), 0.6 * len(avail) + 2.5),
        gridspec_kw={"width_ratios": [len(datasets), 2], "wspace": 0.05},
    )

    # Left: NC metric heatmap
    sns.heatmap(
        nc_z,
        ax=ax_heat,
        cmap="RdBu_r",
        center=0,
        annot=nc_means.T.round(2).values,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "z-score", "shrink": 0.7},
        xticklabels=datasets,
        yticklabels=avail,
    )
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", rotation=0, labelsize=10)
    ax_heat.tick_params(axis="y", rotation=0, labelsize=10)

    # Annotate clique members below each column
    for i, ds in enumerate(datasets):
        ax_heat.text(
            i + 0.5, len(avail) + 0.4,
            clique_labels[ds],
            ha="center", va="top", fontsize=8,
            fontstyle="italic",
        )
    # Add a label for the clique row
    ax_heat.text(
        -0.3, len(avail) + 0.4,
        "clique",
        ha="right", va="top", fontsize=9, fontweight="bold",
    )
    ax_heat.set_title("NC metric profiles", fontsize=12)

    # Right: SHAP importance bars (clique methods only)
    colors = [
        "#2166ac" if v < 0 else "#b2182b"
        for v in shap_agg["mean_shap"]
    ]
    y_pos = np.arange(len(avail))
    ax_shap.barh(
        y_pos, shap_agg["mean_abs_shap"].values,
        color=colors, edgecolor="gray", linewidth=0.5,
    )
    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels([])
    ax_shap.invert_yaxis()
    ax_shap.set_xlabel("mean |SHAP|", fontsize=10)
    ax_shap.set_title("Clique method\nfeature importance", fontsize=11)
    ax_shap.tick_params(labelsize=9)

    # Legend for SHAP sign
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166ac", label="lower rank (better)"),
        Patch(facecolor="#b2182b", label="higher rank (worse)"),
    ]
    ax_shap.legend(
        handles=legend_elements, fontsize=8, loc="lower right",
    )

    fig.suptitle(
        f"NC profile → clique association — {group_label}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        os.path.join(output_dir, f"nc_clique_summary_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved NC-clique summary: nc_clique_summary_{tag}.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-method AUGRC regression from NC metrics"
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
        help="Filter to a single study (e.g. confidnet, devries, dg)",
    )
    parser.add_argument(
        "--clique-file", type=str, default=None,
        help="Optional JSON clique file for clique-hit evaluation",
    )
    parser.add_argument(
        "--filter-methods", action="store_true",
        help="Exclude methods with 'global'/'class' "
             "(except PCA/KPCA RecError global)",
    )
    parser.add_argument(
        "--papyan-only", action="store_true",
        help="Restrict NC features to 8 Papyan metrics",
    )
    parser.add_argument(
        "--groups", type=str, nargs="*", default=None,
        help="OOD groups to analyse (default: near mid far all)",
    )
    parser.add_argument(
        "--target", type=str, default="both",
        choices=["augrc", "avg_ood_rank", "both"],
        help="Regression target: raw AUGRC, average OOD rank, or both",
    )
    parser.add_argument(
        "--output-dir", type=str, default="regression_outputs",
    )
    parser.add_argument(
        "--score-metric", type=str, default="AUGRC",
        help="Score metric label for file prefix",
    )
    parser.add_argument(
        "--mcd", type=str, default="False",
        help="MCD label for file prefix",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── NC metrics ──
    logger.info("Loading NC metrics...")
    nc = load_nc_metrics(args.nc_file)
    arch = ARCH_MAP[args.backbone]
    nc = nc[nc["architecture"] == arch]
    if args.study:
        nc_study = STUDY_MAP.get(args.study, args.study)
        nc = nc[nc["study"] == nc_study]
    logger.info(f"NC metrics: {len(nc)} rows (arch={arch})")

    metric_pool = PAPYAN_NC_METRICS if args.papyan_only else NC_METRICS
    nc_features = [m for m in metric_pool if m in nc.columns] + ["dropout"]
    logger.info(f"NC features ({len(nc_features)}): {nc_features}")

    # ── Scores ──
    logger.info("Loading AUGRC scores...")
    active_sources = [
        s for s in SOURCES
        if os.path.exists(
            os.path.join(
                args.scores_dir,
                f"scores_all_AUGRC_MCD-{args.mcd}_{args.backbone}_{s}.csv",
            )
        )
    ]
    scores = load_scores(
        args.scores_dir,
        args.backbone,
        args.mcd,
        args.clip_dir,
        active_sources,
        filter_methods=args.filter_methods,
        study_filter=args.study,
    )

    # ── Join ──
    reg_df = build_regression_dataset(nc, scores, nc_features)
    if reg_df.empty:
        logger.error("Empty regression dataset after join")
        return

    # ── Cliques (optional) ──
    clique_data = None
    if args.clique_file and os.path.exists(args.clique_file):
        with open(args.clique_file) as f:
            raw = json.load(f)
        raw.pop("_ranks", None)
        clique_data = raw

    # ── Groups ──
    available_groups = sorted(reg_df["group"].unique())
    default_groups = ["near", "mid", "far", "all"]
    groups_to_run = (
        args.groups if args.groups is not None
        else [g for g in default_groups if g in available_groups]
    )
    logger.info(f"Groups: {groups_to_run}")

    # ── File prefix ──
    study_tag = f"_{args.study}" if args.study else ""
    file_prefix = (
        f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"
    )

    # ── Target columns ──
    if args.target == "both":
        target_cols = ["augrc", "avg_ood_rank"]
    else:
        target_cols = [args.target]

    # ── Run per group × target ──
    all_summaries = []
    for target_col in target_cols:
        for group_label in groups_to_run:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"OOD group: {group_label}  |  target: {target_col}")

            cliques_for_group = None
            if clique_data:
                cliques_for_group = {}
                for src, gmap in clique_data.items():
                    if group_label in gmap:
                        cliques_for_group[src] = gmap[group_label]

            result = run_lodo_regression(
                reg_df, nc_features, group_label,
                target_col=target_col,
                cliques=cliques_for_group,
            )
            if not result:
                continue

            summary = result["summary"]
            all_summaries.append(summary)

            tag = f"{file_prefix}_{target_col}_group_{group_label}"
            result["folds"].to_csv(
                os.path.join(args.output_dir, f"regression_folds_{tag}.csv"),
                index=False,
            )
            result["predictions"].to_csv(
                os.path.join(args.output_dir, f"regression_preds_{tag}.csv"),
                index=False,
            )

            imp_df = compute_feature_importance(
                reg_df, nc_features, group_label,
            )
            imp_df.to_csv(
                os.path.join(
                    args.output_dir, f"regression_importance_{tag}.csv"
                ),
                index=False,
            )

            # SHAP analysis
            logger.info(f"  Computing SHAP values ...")
            shap_long, shap_summary = compute_shap_analysis(
                reg_df, nc_features, group_label,
                target_col=target_col,
            )
            shap_long.to_csv(
                os.path.join(
                    args.output_dir, f"regression_shap_long_{tag}.csv"
                ),
                index=False,
            )
            shap_summary.to_csv(
                os.path.join(
                    args.output_dir, f"regression_shap_summary_{tag}.csv"
                ),
                index=False,
            )

            # Partial dependence data
            logger.info(f"  Computing partial dependence ...")
            pdp_df = compute_partial_dependence(
                reg_df, nc_features, group_label,
                target_col=target_col,
            )
            pdp_df.to_csv(
                os.path.join(
                    args.output_dir, f"regression_pdp_{tag}.csv"
                ),
                index=False,
            )

            # NC-clique summary figure
            plot_nc_clique_summary(
                reg_df, nc_features, group_label,
                shap_summary=shap_summary,
                cliques=cliques_for_group,
                target_col=target_col,
                output_dir=args.output_dir,
                tag=tag,
            )

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(
            args.output_dir,
            f"regression_summary_{file_prefix}.csv",
        )
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSaved summary: {summary_path}")
        logger.info(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()

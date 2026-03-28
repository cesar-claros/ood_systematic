"""
Approach 5: NC Scorecard — Low-dimensional method-selection tool.

Reduces 13 NC metrics to 2-3 interpretable dimensions via PCA, then fits a
shallow decision tree to produce explicit if/then rules for selecting the best
OOD detection method family.

Outputs:
  - PCA loadings and variance explained
  - Decision tree rules (text + per-leaf scorecard table)
  - PC1-PC2 scatter plot with decision boundaries (paper figure)
  - Feature-set comparison (PCA vs top-k raw features vs full 13)
  - Cross-validation results (k-fold + LODO)

Usage:
    python nc_scorecard.py --backbone Conv [--study confidnet] [--filter-methods]
    python nc_scorecard.py --backbone Conv --ood-group --clip-dir clip_scores
    python nc_scorecard.py --backbone ViT [--filter-methods]
"""

import os
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from loguru import logger

from src.utils_stats import HIGHER_BETTER

warnings.filterwarnings("ignore", category=UserWarning)

# ── Reuse data-loading helpers ──────────────────────────────────────────────
from nc_regime_analysis import (
    NC_METRICS,
    load_nc_metrics,
    load_scores,
    join_nc_scores,
    ood_columns,
    compute_mean_ood_score,
)
from multinomial_analysis import (
    BLOCK_KEYS,
    build_block_dataset,
    _load_ood_groups,
    _normalise,
)


# ── PCA analysis ────────────────────────────────────────────────────────────
def fit_pca_analysis(
    block_df: pd.DataFrame,
    nc_features: list[str],
    n_components: int = 3,
) -> tuple[PCA, StandardScaler, np.ndarray, pd.DataFrame]:
    """
    Fit PCA on standardised NC metrics.

    Returns (pca, scaler, X_pca, loadings_df).
    """
    X = block_df[nc_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Loadings: correlation between original features and PCs
    loadings = pca.components_.T  # (n_features, n_components)
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    loadings_df = pd.DataFrame(loadings, index=nc_features, columns=pc_names)
    loadings_df["feature"] = nc_features

    logger.info(f"PCA: {n_components} components, "
                f"variance explained: {pca.explained_variance_ratio_}")
    logger.info(f"Cumulative: {np.cumsum(pca.explained_variance_ratio_)}")

    return pca, scaler, X_pca, loadings_df


# ── Decision tree ───────────────────────────────────────────────────────────
def fit_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    feature_names: list[str],
    min_class_count: int = 2,
) -> tuple[DecisionTreeClassifier, LabelEncoder, np.ndarray]:
    """
    Fit a shallow decision tree. Returns (tree, label_encoder, filtered_mask).
    """
    le = LabelEncoder()

    # Filter rare classes
    counts = Counter(y)
    rare = {cls for cls, cnt in counts.items() if cnt < min_class_count}
    mask = np.array([yi not in rare for yi in y])
    if rare:
        logger.info(f"Dropping {len(rare)} rare classes (count < {min_class_count}): {rare}")

    X_f, y_f = X[mask], y[mask]
    y_enc = le.fit_transform(y_f)

    tree = DecisionTreeClassifier(
        max_depth=max_depth, random_state=42,
        class_weight="balanced",
    )
    tree.fit(X_f, y_enc)

    train_acc = accuracy_score(y_enc, tree.predict(X_f))
    logger.info(f"Decision tree: depth={tree.get_depth()}, "
                f"leaves={tree.get_n_leaves()}, "
                f"train_acc={train_acc:.3f}, "
                f"classes={list(le.classes_)}")

    return tree, le, mask


def extract_tree_rules(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    le: LabelEncoder,
) -> str:
    """Return human-readable decision tree rules."""
    return export_text(tree, feature_names=feature_names,
                       class_names=list(le.classes_))


# ── Scorecard table ─────────────────────────────────────────────────────────
def build_scorecard_table(
    tree: DecisionTreeClassifier,
    le: LabelEncoder,
    X: np.ndarray,
    y_enc: np.ndarray,
) -> pd.DataFrame:
    """
    Build per-leaf scorecard: recommended method, count, accuracy.
    """
    leaf_ids = tree.apply(X)
    rows = []
    for leaf in sorted(np.unique(leaf_ids)):
        mask = leaf_ids == leaf
        y_leaf = y_enc[mask]
        pred = tree.predict(X[mask])[0]  # all same for a leaf
        acc = (y_leaf == pred).mean()
        dist = Counter(le.inverse_transform(y_leaf))
        rows.append({
            "leaf_id": int(leaf),
            "recommended_method": le.inverse_transform([pred])[0],
            "n_models": int(mask.sum()),
            "leaf_accuracy": float(acc),
            "class_distribution": dict(dist),
        })
    return pd.DataFrame(rows)


# ── Cross-validation ────────────────────────────────────────────────────────
def cross_validate_tree(
    block_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int,
    n_folds: int = 5,
) -> dict:
    """k-fold and LODO cross-validation for the decision tree."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    datasets = block_df["dataset"].values

    results = {}
    most_common_count = Counter(y_enc).most_common(1)[0][1]
    results["baseline_accuracy"] = most_common_count / len(y_enc)

    # k-fold
    min_count = min(Counter(y_enc).values())
    k = min(n_folds, min_count)
    if k >= 2:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        tree_cv = DecisionTreeClassifier(
            max_depth=max_depth, random_state=42, class_weight="balanced")
        y_pred = cross_val_predict(tree_cv, X, y_enc, cv=skf)
        results["kfold_accuracy"] = accuracy_score(y_enc, y_pred)
        results["kfold_balanced_accuracy"] = balanced_accuracy_score(y_enc, y_pred)
        results["kfold_n_folds"] = k
    else:
        results["kfold_accuracy"] = None
        results["kfold_n_folds"] = 0

    # LODO
    unique_ds = np.unique(datasets)
    if len(unique_ds) >= 2:
        y_pred_lodo = np.full_like(y_enc, -1)
        for held_out in unique_ds:
            test_mask = datasets == held_out
            train_mask = ~test_mask
            train_classes = np.unique(y_enc[train_mask])
            if len(train_classes) < 2:
                continue
            tree_lodo = DecisionTreeClassifier(
                max_depth=max_depth, random_state=42, class_weight="balanced")
            tree_lodo.fit(X[train_mask], y_enc[train_mask])
            y_pred_lodo[test_mask] = tree_lodo.predict(X[test_mask])
        valid = y_pred_lodo >= 0
        if valid.sum() > 0:
            results["lodo_accuracy"] = accuracy_score(y_enc[valid], y_pred_lodo[valid])
            results["lodo_balanced_accuracy"] = balanced_accuracy_score(
                y_enc[valid], y_pred_lodo[valid])

    return results


# ── Feature-set comparison ──────────────────────────────────────────────────
def compare_feature_sets(
    block_df: pd.DataFrame,
    nc_features: list[str],
    y: np.ndarray,
    max_depth: int,
    n_folds: int,
    pca: PCA,
    scaler: StandardScaler,
    X_pca: np.ndarray,
) -> pd.DataFrame:
    """
    Compare accuracy of decision tree across different feature sets:
    PCA-2, PCA-3, top-2 RF, top-3 RF, all 13 features.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    min_count = min(Counter(y_enc).values())
    k = min(n_folds, min_count)
    if k < 2:
        logger.warning("Too few samples per class for CV comparison")
        return pd.DataFrame()

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Get RF feature importance for top-k selection
    X_all = StandardScaler().fit_transform(block_df[nc_features].values)
    rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                class_weight="balanced")
    rf.fit(X_all, y_enc)
    imp_order = np.argsort(rf.feature_importances_)[::-1]
    top_features = [nc_features[i] for i in imp_order]

    feature_sets = {}

    # PCA variants
    for n_pc in [2, 3]:
        if n_pc <= X_pca.shape[1]:
            feature_sets[f"PCA-{n_pc}"] = X_pca[:, :n_pc]

    # Top-k RF features
    for top_k in [2, 3]:
        feat_names = top_features[:top_k]
        feature_sets[f"Top-{top_k} RF ({', '.join(feat_names)})"] = (
            StandardScaler().fit_transform(block_df[feat_names].values))

    # All 13
    feature_sets["All 13 (tree)"] = X_all
    feature_sets["All 13 (RF)"] = X_all

    rows = []
    for name, X_fs in feature_sets.items():
        if "RF)" == name[-3:] or name == "All 13 (RF)":
            clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                         class_weight="balanced")
        else:
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42,
                                         class_weight="balanced")
        y_pred = cross_val_predict(clf, X_fs, y_enc, cv=skf)
        rows.append({
            "feature_set": name,
            "n_features": X_fs.shape[1],
            "kfold_accuracy": accuracy_score(y_enc, y_pred),
            "kfold_balanced_accuracy": balanced_accuracy_score(y_enc, y_pred),
        })

    baseline = Counter(y_enc).most_common(1)[0][1] / len(y_enc)
    for row in rows:
        row["baseline"] = baseline

    return pd.DataFrame(rows)


# ── Visualization ───────────────────────────────────────────────────────────
def plot_pca_scatter_with_boundaries(
    X_pca: np.ndarray,
    y: np.ndarray,
    tree: DecisionTreeClassifier,
    le: LabelEncoder,
    pca: PCA,
    output_path: str,
    n_components_used: int = 2,
):
    """
    Scatter plot of models in PC1-PC2 space with decision boundaries.
    """
    y_enc = le.transform(y)
    n_classes = len(le.classes_)

    # Use a colorblind-friendly palette
    palette = sns.color_palette("tab10", n_classes)
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "p"][:n_classes]

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # Decision boundary meshgrid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    if n_components_used >= 3 and X_pca.shape[1] >= 3:
        # Fix PC3 at mean for the meshgrid
        pc3_mean = X_pca[:, 2].mean()
        grid_input = np.c_[xx.ravel(), yy.ravel(),
                           np.full(xx.ravel().shape, pc3_mean)]
    else:
        grid_input = np.c_[xx.ravel(), yy.ravel()]

    Z = tree.predict(grid_input)
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    cmap_bg = matplotlib.colors.ListedColormap(
        [(*c[:3], 0.15) for c in palette])
    ax.contourf(xx, yy, Z, alpha=0.2, levels=np.arange(-0.5, n_classes + 0.5),
                colors=[(*c[:3], 0.15) for c in palette])

    # Plot decision boundaries
    ax.contour(xx, yy, Z, colors="gray", linewidths=0.5,
               levels=np.arange(-0.5, n_classes + 0.5))

    # Scatter points
    for i, cls in enumerate(le.classes_):
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[palette[i]], marker=markers[i],
                   s=80, edgecolors="black", linewidths=0.5,
                   label=cls, zorder=5)

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=12)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_title("NC Scorecard: Method Selection by Neural Collapse Profile",
                 fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".jpeg"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scatter plot: {output_path}")


# ── Pipeline orchestrator ───────────────────────────────────────────────────
def run_scorecard(
    block_df: pd.DataFrame,
    nc_features: list[str],
    label: str,
    output_dir: str,
    file_prefix: str,
    max_depth: int = 3,
    n_components: int = 3,
    n_folds: int = 5,
    min_class_count: int = 2,
):
    """Run the full scorecard pipeline for one configuration."""
    df = block_df.copy()

    if df.empty or df["best_method"].nunique() < 2:
        logger.warning(f"Not enough classes for scorecard ({label})")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"NC Scorecard: {label}")
    logger.info(f"  {len(df)} models, {df['best_method'].nunique()} classes")

    # ── 1. PCA ───────────────────────────────────────────────────────
    pca, scaler, X_pca, loadings_df = fit_pca_analysis(
        df, nc_features, n_components)

    loadings_path = os.path.join(
        output_dir, f"scorecard_pca_loadings_{file_prefix}_{label}.csv")
    loadings_out = loadings_df.copy()
    loadings_out.loc["variance_explained"] = (
        list(pca.explained_variance_ratio_) + [np.nan])
    loadings_out.loc["cumulative_variance"] = (
        list(np.cumsum(pca.explained_variance_ratio_)) + [np.nan])
    loadings_out.to_csv(loadings_path, index=True)
    logger.info(f"Saved PCA loadings: {loadings_path}")

    # ── 2. Decision tree on PCA components ───────────────────────────
    n_pc = min(n_components, X_pca.shape[1])
    pc_names = [f"PC{i+1}" for i in range(n_pc)]

    tree, le, mask = fit_decision_tree(
        X_pca[:, :n_pc], df["best_method"].values,
        max_depth, pc_names, min_class_count)

    X_f = X_pca[mask, :n_pc]
    y_f = df["best_method"].values[mask]
    y_enc_f = le.transform(y_f)
    df_f = df.iloc[mask].reset_index(drop=True)

    if len(le.classes_) < 2:
        logger.warning("Fewer than 2 classes after filtering, skipping")
        return

    # Tree rules
    rules_text = extract_tree_rules(tree, pc_names, le)
    logger.info(f"\nDecision tree rules:\n{rules_text}")

    rules_path = os.path.join(
        output_dir, f"scorecard_tree_rules_{file_prefix}_{label}.txt")
    with open(rules_path, "w") as f:
        f.write(f"# NC Scorecard: {label}\n")
        f.write(f"# {len(df_f)} models, {len(le.classes_)} classes\n")
        f.write(f"# PCA variance explained: {pca.explained_variance_ratio_}\n\n")
        f.write("## PCA Component Interpretation\n\n")
        for i in range(n_pc):
            top_pos = loadings_df[f"PC{i+1}"].nlargest(3)
            top_neg = loadings_df[f"PC{i+1}"].nsmallest(3)
            f.write(f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% var):\n")
            f.write(f"  High: {', '.join(f'{k}({v:.2f})' for k, v in top_pos.items())}\n")
            f.write(f"  Low:  {', '.join(f'{k}({v:.2f})' for k, v in top_neg.items())}\n\n")
        f.write("## Decision Rules\n\n")
        f.write(rules_text)
    logger.info(f"Saved rules: {rules_path}")

    # ── 3. Scorecard table ───────────────────────────────────────────
    scorecard = build_scorecard_table(tree, le, X_f, y_enc_f)
    scorecard_path = os.path.join(
        output_dir, f"scorecard_table_{file_prefix}_{label}.csv")
    scorecard.to_csv(scorecard_path, index=False)
    logger.info(f"Saved scorecard: {scorecard_path}")
    logger.info(f"\n{scorecard.to_string(index=False)}")

    # ── 4. Cross-validation ──────────────────────────────────────────
    cv_results = cross_validate_tree(
        df_f, X_f, y_f, max_depth, n_folds)
    cv_results["label"] = label
    cv_results["n_models"] = len(df_f)
    cv_results["n_classes"] = len(le.classes_)
    cv_results["n_pca_components"] = n_pc
    cv_results["max_depth"] = max_depth

    cv_df = pd.DataFrame([cv_results])
    cv_path = os.path.join(
        output_dir, f"scorecard_cv_{file_prefix}_{label}.csv")
    cv_df.to_csv(cv_path, index=False)
    logger.info(f"Saved CV results: {cv_path}")
    logger.info(f"  k-fold acc: {cv_results.get('kfold_accuracy', 'N/A')}")
    logger.info(f"  k-fold bal: {cv_results.get('kfold_balanced_accuracy', 'N/A')}")
    logger.info(f"  LODO acc:   {cv_results.get('lodo_accuracy', 'N/A')}")
    logger.info(f"  Baseline:   {cv_results.get('baseline_accuracy', 'N/A')}")

    # ── 5. Feature-set comparison ────────────────────────────────────
    comparison = compare_feature_sets(
        df_f, nc_features, y_f, max_depth, n_folds,
        pca, scaler, X_f)
    if not comparison.empty:
        comp_path = os.path.join(
            output_dir, f"scorecard_comparison_{file_prefix}_{label}.csv")
        comparison.to_csv(comp_path, index=False)
        logger.info(f"Saved comparison: {comp_path}")
        logger.info(f"\n{comparison.to_string(index=False)}")

    # ── 6. Scatter plot ──────────────────────────────────────────────
    plot_path = os.path.join(
        output_dir, f"scorecard_pca_scatter_{file_prefix}_{label}.pdf")
    plot_pca_scatter_with_boundaries(
        X_pca[mask, :], y_f, tree, le, pca, plot_path,
        n_components_used=n_pc)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="NC Scorecard: low-dimensional method-selection tool"
    )
    parser.add_argument("--nc-file", type=str,
                        default="neural_collapse_metrics/nc_metrics.csv")
    parser.add_argument("--scores-dir", type=str, default="scores_risk")
    parser.add_argument("--score-metric", type=str, default="AUGRC",
                        choices=["AUGRC", "AURC", "AUROC_f", "FPR@95TPR"])
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["Conv", "ViT"])
    parser.add_argument("--mcd", type=str, default="False",
                        choices=["True", "False"])
    parser.add_argument("--study", type=str, default=None)
    parser.add_argument("--filter-methods", action="store_true")
    parser.add_argument("--min-class-count", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Decision tree max depth (default: 3)")
    parser.add_argument("--n-components", type=int, default=3,
                        help="PCA components to extract (default: 3)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--ood-group", action="store_true")
    parser.add_argument("--clip-dir", type=str, default="clip_scores")
    parser.add_argument("--output-dir", type=str, default="scorecard_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
    ascending = not HIGHER_BETTER.get(args.score_metric, True)

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading NC metrics...")
    nc = load_nc_metrics(args.nc_file)
    nc["dataset"] = nc["dataset"].replace({"supercifar": "supercifar100"})

    logger.info("Loading scores...")
    scores = load_scores(args.scores_dir, args.score_metric,
                         args.backbone, args.mcd, datasets)

    if args.filter_methods:
        keep_exceptions = {
            "KPCA RecError global", "PCA RecError global",
            "MCD-KPCA RecError global", "MCD-PCA RecError global",
        }
        mask = scores["methods"].str.contains("global|class", case=False, na=False)
        mask &= ~scores["methods"].isin(keep_exceptions)
        scores = scores[~mask]
        logger.info(f"After method filter: {scores['methods'].nunique()} methods")

    if args.study:
        scores = scores[scores["study"] == args.study]
        nc = nc[nc["study"] == args.study]
        logger.info(f"Filtered to study={args.study}")

    merged = join_nc_scores(nc, scores)
    if merged.empty:
        logger.error("No rows after join.")
        return

    ood_cols = ood_columns(merged)
    study_tag = f"_{args.study}" if args.study else ""
    file_prefix = f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"
    nc_features = [m for m in NC_METRICS
                   if m in merged.columns or m + "_nc" in merged.columns]

    # ── Run scorecard ────────────────────────────────────────────────
    if args.ood_group:
        ood_group_global = _load_ood_groups(args.clip_dir, datasets)
        if not ood_group_global:
            logger.error(f"No OOD group labels from {args.clip_dir}")
            return

        ood_group_norm = {_normalise(k): v for k, v in ood_group_global.items()}
        group_to_cols: dict[str, list[str]] = {}
        for col in ood_cols:
            group = ood_group_norm.get(_normalise(col))
            if group:
                group_to_cols.setdefault(group, []).append(col)

        for group_label in ["near", "mid", "far", "all"]:
            if group_label == "all":
                cols = ood_cols
            else:
                cols = group_to_cols.get(group_label, [])
            if not cols:
                continue

            group_merged = merged.copy()
            group_merged["group_mean_score"] = group_merged[cols].mean(axis=1)
            block_df = build_block_dataset(group_merged, "group_mean_score", ascending)

            if len(block_df) < 5:
                logger.warning(f"Too few blocks for group {group_label}")
                continue

            run_scorecard(
                block_df, nc_features,
                label=f"group_{group_label}",
                output_dir=args.output_dir,
                file_prefix=file_prefix,
                max_depth=args.max_depth,
                n_components=args.n_components,
                n_folds=args.cv_folds,
                min_class_count=args.min_class_count,
            )
    else:
        merged = compute_mean_ood_score(merged, ood_cols)
        block_df = build_block_dataset(merged, "mean_ood_score", ascending)

        if len(block_df) < 5:
            logger.error(f"Too few blocks ({len(block_df)})")
            return

        run_scorecard(
            block_df, nc_features,
            label="mean_ood",
            output_dir=args.output_dir,
            file_prefix=file_prefix,
            max_depth=args.max_depth,
            n_components=args.n_components,
            n_folds=args.cv_folds,
            min_class_count=args.min_class_count,
        )


if __name__ == "__main__":
    main()

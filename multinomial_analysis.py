"""
Approach 2: Multinomial Classification (NC -> Best Method).

For each model (block = dataset × architecture × study × run), the label is the
rank-1 OOD detection method (by mean AUGRC across OOD sets). The 13 NC metrics
are the features. We fit:
  (a) Multinomial logistic regression (with L2 regularisation)
  (b) Random forest classifier

Evaluation:
  - Leave-one-dataset-out cross-validation (LODO): train on 3 source datasets,
    predict on the held-out one.  Tests generalisation across datasets.
  - Stratified 5-fold CV (as a secondary check within the full dataset).
  - Feature importance: logistic regression coefficients + random forest importances.

Also supports OOD-group stratification: instead of averaging across all OOD sets,
the label is the best method for a specific OOD proximity group (near/mid/far).

Usage:
    python multinomial_analysis.py --backbone Conv
    python multinomial_analysis.py --backbone Conv --study confidnet
    python multinomial_analysis.py --backbone Conv --ood-group --clip-dir clip_scores
    python multinomial_analysis.py --backbone ViT
"""

import os
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger

from src.utils_stats import HIGHER_BETTER

warnings.filterwarnings("ignore", category=UserWarning)

# ── Reuse data-loading helpers from nc_regime_analysis ──────────────────────
from nc_regime_analysis import (
    NC_METRICS,
    load_nc_metrics,
    load_scores,
    join_nc_scores,
    ood_columns,
    compute_mean_ood_score,
    JOIN_KEYS_SC,
)

# Papyan et al. (2020) subset: NC1 (var_collapse), NC2 (equiangularity,
# equinormness, max equiangularity for means & weights), NC3 (self_duality)
PAPYAN_NC_METRICS = [
    "var_collapse",
    "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc",
    "max_equiangular_uc", "max_equiangular_wc",
    "self_duality",
]


BLOCK_KEYS = ["dataset", "architecture", "study", "dropout", "run", "reward_val"]


# ── Build block-level dataset ────────────────────────────────────────────────
def build_block_dataset(
    merged: pd.DataFrame,
    score_col: str,
    ascending: bool,
) -> pd.DataFrame:
    """
    Build a block-level dataset where each row is one model (block) with:
      - 13 NC metric features (averaged across dropout/reward)
      - best_method: the rank-1 method (lowest AUGRC or best score)
      - best_score: the score of the best method

    Parameters
    ----------
    merged : joined NC + scores DataFrame
    score_col : column with the score to rank (e.g. "mean_ood_score")
    ascending : if True, lower is better (e.g. AUGRC)

    Returns
    -------
    DataFrame with BLOCK_KEYS + NC_METRICS + ["best_method", "best_score"]
    """
    # Average score per block × method
    agg_cols = BLOCK_KEYS + ["methods", score_col]
    nc_col_names = [c for c in merged.columns if c in NC_METRICS or c.endswith("_nc")]
    agg_cols_nc = BLOCK_KEYS + nc_col_names

    # Score: mean across dropout/reward per block × method
    block_scores = (
        merged[agg_cols]
        .groupby(BLOCK_KEYS + ["methods"], as_index=False)
        .mean(numeric_only=True)
    )

    # NC metrics: mean across dropout/reward per block
    block_nc = (
        merged[agg_cols_nc]
        .groupby(BLOCK_KEYS, as_index=False)
        .mean(numeric_only=True)
    )

    # For each block, find the best method
    if ascending:
        idx_best = block_scores.groupby(BLOCK_KEYS)[score_col].idxmin()
    else:
        idx_best = block_scores.groupby(BLOCK_KEYS)[score_col].idxmax()

    best = block_scores.loc[idx_best, BLOCK_KEYS + ["methods", score_col]].copy()
    best = best.rename(columns={"methods": "best_method", score_col: "best_score"})

    # Merge NC metrics
    result = best.merge(block_nc, on=BLOCK_KEYS, how="inner")

    # Resolve NC metric column names (may have _nc suffix)
    for m in NC_METRICS:
        if m not in result.columns and m + "_nc" in result.columns:
            result[m] = result[m + "_nc"]

    # Drop _nc suffix columns
    result = result[[c for c in result.columns if not c.endswith("_nc")]]

    # Drop rows with missing NC metrics
    nc_available = [m for m in NC_METRICS if m in result.columns]
    result = result.dropna(subset=nc_available)

    logger.info(f"Block dataset: {len(result)} blocks, "
                f"{result['best_method'].nunique()} unique best methods")
    logger.info(f"Best method distribution:\n{result['best_method'].value_counts().to_string()}")

    return result


# ── Classification ───────────────────────────────────────────────────────────
def run_classification(
    block_df: pd.DataFrame,
    nc_features: list[str],
    label: str,
    output_dir: str,
    file_prefix: str,
    min_class_count: int = 2,
):
    """
    Run multinomial classification: NC features -> best_method.

    Uses:
      1. Leave-one-dataset-out CV (LODO)
      2. Stratified 5-fold CV
      3. Feature importance analysis
    """
    df = block_df.copy()

    # Filter out methods that appear fewer than min_class_count times
    method_counts = df["best_method"].value_counts()
    rare_methods = method_counts[method_counts < min_class_count].index.tolist()
    if rare_methods:
        logger.info(f"Dropping {len(rare_methods)} rare methods (count < {min_class_count}): "
                     f"{rare_methods}")
        df = df[~df["best_method"].isin(rare_methods)]

    if df.empty or df["best_method"].nunique() < 2:
        logger.warning(f"Not enough classes for classification ({label})")
        return

    X = df[nc_features].values
    y = df["best_method"].values
    datasets = df["dataset"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    logger.info(f"\n{'='*60}")
    logger.info(f"Classification: {label}")
    logger.info(f"  {len(df)} samples, {n_classes} classes, {len(nc_features)} features")
    logger.info(f"  Classes: {list(le.classes_)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # ── 1. Leave-One-Dataset-Out CV ──────────────────────────────────────
    unique_datasets = np.unique(datasets)
    if len(unique_datasets) >= 2:
        logger.info("\n--- Leave-One-Dataset-Out CV ---")
        y_pred_lodo_lr = np.full_like(y_enc, -1)
        y_pred_lodo_rf = np.full_like(y_enc, -1)

        for held_out_ds in unique_datasets:
            test_mask = datasets == held_out_ds
            train_mask = ~test_mask

            # Check that train set has at least 2 classes
            train_classes = np.unique(y_enc[train_mask])
            if len(train_classes) < 2:
                logger.warning(f"  LODO {held_out_ds}: only 1 class in train, skipping")
                continue

            # Fit scaler on train only
            scaler_lodo = StandardScaler()
            X_train = scaler_lodo.fit_transform(X[train_mask])
            X_test = scaler_lodo.transform(X[test_mask])

            # Logistic Regression
            lr = LogisticRegression(
                solver="lbfgs",
                max_iter=2000, C=1.0, random_state=42,
            )
            lr.fit(X_train, y_enc[train_mask])
            y_pred_lodo_lr[test_mask] = lr.predict(X_test)

            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42,
                class_weight="balanced",
            )
            rf.fit(X_train, y_enc[train_mask])
            y_pred_lodo_rf[test_mask] = rf.predict(X_test)

            acc_lr_fold = accuracy_score(y_enc[test_mask], y_pred_lodo_lr[test_mask])
            acc_rf_fold = accuracy_score(y_enc[test_mask], y_pred_lodo_rf[test_mask])
            logger.info(f"  Hold out {held_out_ds}: "
                        f"LR={acc_lr_fold:.3f}, RF={acc_rf_fold:.3f} "
                        f"({test_mask.sum()} samples, "
                        f"{len(np.unique(y_enc[test_mask]))} classes)")

        # Overall LODO accuracy (exclude any -1 predictions)
        valid_mask_lr = y_pred_lodo_lr >= 0
        valid_mask_rf = y_pred_lodo_rf >= 0
        lodo_results = {}
        for tag, y_pred_lodo, valid_mask in [
            ("lr", y_pred_lodo_lr, valid_mask_lr),
            ("rf", y_pred_lodo_rf, valid_mask_rf),
        ]:
            if valid_mask.sum() > 0:
                acc = accuracy_score(y_enc[valid_mask], y_pred_lodo[valid_mask])
                bal_acc = balanced_accuracy_score(y_enc[valid_mask], y_pred_lodo[valid_mask])
                lodo_results[tag] = {"acc": acc, "bal_acc": bal_acc,
                                     "n": int(valid_mask.sum()), "preds": y_pred_lodo,
                                     "valid": valid_mask}
                logger.info(f"\n  LODO {tag.upper()} overall: acc={acc:.3f}, "
                            f"balanced_acc={bal_acc:.3f}")

        # Pick best LODO classifier and store results
        if lodo_results:
            best_tag = max(lodo_results, key=lambda t: lodo_results[t]["acc"])
            best = lodo_results[best_tag]
            results["lodo_lr_accuracy"] = lodo_results.get("lr", {}).get("acc", np.nan)
            results["lodo_lr_balanced_accuracy"] = lodo_results.get("lr", {}).get("bal_acc", np.nan)
            results["lodo_rf_accuracy"] = lodo_results.get("rf", {}).get("acc", np.nan)
            results["lodo_rf_balanced_accuracy"] = lodo_results.get("rf", {}).get("bal_acc", np.nan)
            results["lodo_accuracy"] = best["acc"]
            results["lodo_balanced_accuracy"] = best["bal_acc"]
            results["lodo_n_samples"] = best["n"]
            y_pred_lodo = best["preds"]
            valid_mask = best["valid"]

            most_common = Counter(y_enc[valid_mask]).most_common(1)[0]
            baseline_acc = most_common[1] / valid_mask.sum()
            results["lodo_baseline_accuracy"] = baseline_acc
            logger.info(f"  Best LODO: {best_tag.upper()} acc={best['acc']:.3f}")
            logger.info(f"  Baseline (most frequent): acc={baseline_acc:.3f} "
                        f"(class={le.classes_[most_common[0]]})")
        else:
            y_pred_lodo = None
    else:
        logger.info("Only 1 dataset — skipping LODO CV")
        y_pred_lodo = None

    # ── 2. Stratified 5-Fold CV ──────────────────────────────────────────
    logger.info("\n--- Stratified 5-Fold CV ---")

    # Ensure each class has at least 5 samples, otherwise reduce folds
    min_count = min(Counter(y_enc).values())
    n_folds = min(5, min_count)
    if n_folds < 2:
        logger.warning(f"Min class count={min_count}, skipping k-fold CV")
        n_folds = 0

    if n_folds >= 2:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        y_pred_lr = np.full_like(y_enc, -1)
        y_pred_rf = np.full_like(y_enc, -1)
        fold_ids = np.full(len(y_enc), -1, dtype=int)

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_enc)):
            fold_ids[test_idx] = fold_i

            lr_cv = LogisticRegression(
                solver="lbfgs", max_iter=2000, C=1.0, random_state=42,
            )
            rf_cv = RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=42,
                class_weight="balanced",
            )
            lr_cv.fit(X_scaled[train_idx], y_enc[train_idx])
            rf_cv.fit(X_scaled[train_idx], y_enc[train_idx])

            y_pred_lr[test_idx] = lr_cv.predict(X_scaled[test_idx])
            y_pred_rf[test_idx] = rf_cv.predict(X_scaled[test_idx])

            # Per-fold metrics
            fold_acc_lr = accuracy_score(y_enc[test_idx], y_pred_lr[test_idx])
            fold_acc_rf = accuracy_score(y_enc[test_idx], y_pred_rf[test_idx])
            fold_classes = le.inverse_transform(np.unique(y_enc[test_idx]))
            logger.info(f"  Fold {fold_i+1}/{n_folds}: "
                        f"LR={fold_acc_lr:.3f}, RF={fold_acc_rf:.3f} "
                        f"({len(test_idx)} samples, classes: {list(fold_classes)})")

        acc_lr = accuracy_score(y_enc, y_pred_lr)
        acc_rf = accuracy_score(y_enc, y_pred_rf)
        bal_acc_lr = balanced_accuracy_score(y_enc, y_pred_lr)
        bal_acc_rf = balanced_accuracy_score(y_enc, y_pred_rf)

        logger.info(f"\n  Overall LR: acc={acc_lr:.3f}, balanced_acc={bal_acc_lr:.3f}")
        logger.info(f"  Overall RF: acc={acc_rf:.3f}, balanced_acc={bal_acc_rf:.3f}")

        results["kfold_lr_accuracy"] = acc_lr
        results["kfold_lr_balanced_accuracy"] = bal_acc_lr
        results["kfold_rf_accuracy"] = acc_rf
        results["kfold_rf_balanced_accuracy"] = bal_acc_rf
        results["kfold_n_folds"] = n_folds

        # Baseline
        most_common = Counter(y_enc).most_common(1)[0]
        baseline_acc = most_common[1] / len(y_enc)
        results["kfold_baseline_accuracy"] = baseline_acc
        logger.info(f"  Baseline (most frequent): acc={baseline_acc:.3f}")

        # Save per-sample fold predictions
        kfold_pred_df = df[BLOCK_KEYS].copy()
        kfold_pred_df["fold"] = fold_ids
        kfold_pred_df["true_method"] = y
        kfold_pred_df["pred_lr"] = le.inverse_transform(y_pred_lr)
        kfold_pred_df["pred_rf"] = le.inverse_transform(y_pred_rf)
        kfold_pred_df["correct_lr"] = (y_pred_lr == y_enc)
        kfold_pred_df["correct_rf"] = (y_pred_rf == y_enc)

    # ── 3. Feature Importance ────────────────────────────────────────────
    logger.info("\n--- Feature Importance ---")

    # Fit on full data for importance analysis
    lr_full = LogisticRegression(
        solver="lbfgs",
        max_iter=2000, C=1.0, random_state=42,
    )
    lr_full.fit(X_scaled, y_enc)

    rf_full = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42,
        class_weight="balanced",
    )
    rf_full.fit(X_scaled, y_enc)

    # LR: coefficient magnitudes (mean absolute across classes)
    lr_importance = np.abs(lr_full.coef_).mean(axis=0)
    lr_importance_df = pd.DataFrame({
        "feature": nc_features,
        "lr_mean_abs_coef": lr_importance,
    }).sort_values("lr_mean_abs_coef", ascending=False)

    # RF: Gini importance
    rf_importance_df = pd.DataFrame({
        "feature": nc_features,
        "rf_gini_importance": rf_full.feature_importances_,
    }).sort_values("rf_gini_importance", ascending=False)

    # Merge
    importance_df = lr_importance_df.merge(rf_importance_df, on="feature")
    importance_df["lr_rank"] = importance_df["lr_mean_abs_coef"].rank(ascending=False).astype(int)
    importance_df["rf_rank"] = importance_df["rf_gini_importance"].rank(ascending=False).astype(int)
    importance_df = importance_df.sort_values("rf_gini_importance", ascending=False)

    logger.info(f"\n{importance_df.to_string(index=False)}")

    # Per-class LR coefficients
    # Binary case: sklearn returns shape (1, n_features); expand to both classes
    coefs = lr_full.coef_
    if coefs.shape[0] == 1 and len(le.classes_) == 2:
        coefs = np.vstack([coefs, -coefs])
    coef_df = pd.DataFrame(
        coefs,
        index=le.classes_,
        columns=nc_features,
    )
    logger.info(f"\nLR coefficients per class:\n{coef_df.round(3).to_string()}")

    # ── 4. Decision Tree Rules ──────────────────────────────────────────
    # Fit a shallow decision tree on unscaled features to produce
    # interpretable if/then rules with thresholds in original NC units.
    logger.info("\n--- Decision Tree Rules ---")

    best_dt = None
    best_dt_acc = -1
    for depth in [2, 3, 4, 5]:
        dt = DecisionTreeClassifier(
            max_depth=depth, random_state=42,
            class_weight="balanced",
        )
        if n_folds >= 2:
            y_pred_dt = cross_val_predict(dt, X, y_enc, cv=skf)
            dt_acc = accuracy_score(y_enc, y_pred_dt)
        else:
            dt_acc = 0.0
        if dt_acc > best_dt_acc:
            best_dt_acc = dt_acc
            best_dt_depth = depth

    # Refit best depth on full data
    best_dt = DecisionTreeClassifier(
        max_depth=best_dt_depth, random_state=42,
        class_weight="balanced",
    )
    best_dt.fit(X, y_enc)

    # Evaluate DT with CV using same folds
    if n_folds >= 2:
        y_pred_dt_cv = np.full_like(y_enc, -1)
        for train_idx, test_idx in skf.split(X, y_enc):
            dt_fold = DecisionTreeClassifier(
                max_depth=best_dt_depth, random_state=42,
                class_weight="balanced",
            )
            dt_fold.fit(X[train_idx], y_enc[train_idx])
            y_pred_dt_cv[test_idx] = dt_fold.predict(X[test_idx])
        dt_cv_acc = accuracy_score(y_enc, y_pred_dt_cv)
        dt_cv_bal_acc = balanced_accuracy_score(y_enc, y_pred_dt_cv)

        # Add DT predictions to kfold_pred_df
        kfold_pred_df["pred_dt"] = le.inverse_transform(y_pred_dt_cv)
        kfold_pred_df["correct_dt"] = (y_pred_dt_cv == y_enc)
    else:
        dt_cv_acc = dt_cv_bal_acc = np.nan

    results["dt_depth"] = best_dt_depth
    results["dt_cv_accuracy"] = dt_cv_acc
    results["dt_cv_balanced_accuracy"] = dt_cv_bal_acc

    # Train accuracy (how well the tree fits the data)
    dt_train_acc = accuracy_score(y_enc, best_dt.predict(X))

    logger.info(f"  Best depth: {best_dt_depth}")
    logger.info(f"  Train accuracy: {dt_train_acc:.3f}")
    logger.info(f"  CV accuracy:    {dt_cv_acc:.3f} (balanced: {dt_cv_bal_acc:.3f})")
    if n_folds >= 2:
        logger.info(f"  RF CV accuracy: {acc_rf:.3f} (for comparison)")

    # Extract rules in text form
    tree_rules = export_text(
        best_dt,
        feature_names=nc_features,
        class_names=list(le.classes_),
        decimals=4,
    )
    logger.info(f"\nDecision Tree Rules:\n{tree_rules}")

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    # Summary
    results["label"] = label
    results["n_samples"] = len(df)
    results["n_classes"] = n_classes
    results["n_features"] = len(nc_features)
    results["classes"] = ",".join(le.classes_)

    summary_df = pd.DataFrame([results])
    summary_path = os.path.join(output_dir, f"multinomial_summary_{file_prefix}_{label}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")

    # Feature importance
    imp_path = os.path.join(output_dir, f"multinomial_importance_{file_prefix}_{label}.csv")
    importance_df.to_csv(imp_path, index=False)
    logger.info(f"Saved importance: {imp_path}")

    # Per-class coefficients
    coef_path = os.path.join(output_dir, f"multinomial_coefs_{file_prefix}_{label}.csv")
    coef_df.to_csv(coef_path)
    logger.info(f"Saved coefficients: {coef_path}")

    # k-Fold predictions per sample
    if n_folds >= 2:
        kfold_path = os.path.join(output_dir, f"multinomial_kfold_preds_{file_prefix}_{label}.csv")
        kfold_pred_df.to_csv(kfold_path, index=False)
        logger.info(f"Saved k-fold predictions: {kfold_path}")

    # Decision tree rules
    rules_path = os.path.join(output_dir, f"multinomial_tree_rules_{file_prefix}_{label}.txt")
    with open(rules_path, "w") as f:
        f.write(f"Decision Tree Rules ({label})\n")
        f.write(f"Depth: {best_dt_depth}, Train acc: {dt_train_acc:.3f}, "
                f"CV acc: {dt_cv_acc:.3f}, Balanced CV acc: {dt_cv_bal_acc:.3f}\n")
        f.write(f"Classes: {list(le.classes_)}\n")
        f.write(f"Features: {nc_features}\n\n")
        f.write(tree_rules)
    logger.info(f"Saved tree rules: {rules_path}")

    # LODO predictions
    if y_pred_lodo is not None:
        valid_mask = y_pred_lodo >= 0
        if valid_mask.sum() > 0:
            pred_df = df[valid_mask].copy()
            pred_df["predicted_method"] = le.inverse_transform(y_pred_lodo[valid_mask])
            pred_df["correct"] = pred_df["best_method"] == pred_df["predicted_method"]
            pred_path = os.path.join(output_dir, f"multinomial_lodo_preds_{file_prefix}_{label}.csv")
            pred_df[BLOCK_KEYS + ["best_method", "predicted_method", "correct"]].to_csv(
                pred_path, index=False)
            logger.info(f"Saved LODO predictions: {pred_path}")

    return results


# ── OOD group support ────────────────────────────────────────────────────────
def _load_ood_groups(clip_dir: str, datasets: list[str]) -> dict[str, dict[str, str]]:
    """Load source-specific OOD group assignments.

    Returns
    -------
    {source_dataset: {ood_set: group_name}}
    """
    GROUP_NAMES = {1: "near", 2: "mid", 3: "far"}
    per_source: dict[str, dict[str, str]] = {}

    for ds in datasets:
        path = os.path.join(clip_dir, f"clip_distances_{ds}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        ds_groups: dict[str, str] = {}
        for ood_set in df.index:
            try:
                g = int(df.loc[ood_set, ("group", "")])
            except (KeyError, ValueError):
                try:
                    g = int(df.loc[ood_set].iloc[-1])
                except (KeyError, ValueError):
                    continue
            if g in GROUP_NAMES:
                ds_groups[ood_set] = GROUP_NAMES[g]
        per_source[ds] = ds_groups

    return per_source


def _normalise(s: str) -> str:
    return s.lower().strip().replace("_", " ")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Approach 2: Multinomial classification — NC features -> best OOD method"
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
    parser.add_argument("--study", type=str, default=None,
                        help="Filter to a single study (e.g. confidnet, devries, dg, vit)")
    parser.add_argument("--filter-methods", action="store_true",
                        help="Exclude methods containing 'global' or 'class' "
                             "(except PCA/KPCA RecError global)")
    parser.add_argument("--min-class-count", type=int, default=2,
                        help="Minimum samples per class to include in classification")
    parser.add_argument("--ood-group", action="store_true",
                        help="Run per OOD group (near/mid/far) using CLIP labels")
    parser.add_argument("--clip-dir", type=str, default="clip_scores",
                        help="Directory with clip_distances_{dataset}.csv")
    parser.add_argument("--papyan-only", action="store_true",
                        help="Restrict NC metrics to the 8 Papyan et al. (2020) metrics")
    parser.add_argument("--output-dir", type=str, default="multinomial_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
    ascending = not HIGHER_BETTER.get(args.score_metric, True)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading NC metrics...")
    nc = load_nc_metrics(args.nc_file)
    nc["dataset"] = nc["dataset"].replace({"supercifar": "supercifar100"})

    logger.info("Loading scores...")
    scores = load_scores(args.scores_dir, args.score_metric,
                         args.backbone, args.mcd, datasets)

    # ── Optional filters ─────────────────────────────────────────────────
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

    # ── Join ─────────────────────────────────────────────────────────────
    merged = join_nc_scores(nc, scores)
    if merged.empty:
        logger.error("No rows after join.")
        return

    ood_cols = ood_columns(merged)
    logger.info(f"OOD columns: {ood_cols}")

    study_tag = f"_{args.study}" if args.study else ""
    file_prefix = f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"

    metric_pool = PAPYAN_NC_METRICS if args.papyan_only else NC_METRICS
    nc_features = [m for m in metric_pool if m in merged.columns or m + "_nc" in merged.columns]
    logger.info(f"NC features ({len(nc_features)}): {nc_features}")

    # ── Run classification ───────────────────────────────────────────────
    if args.ood_group:
        # Per OOD group (source-specific assignments)
        per_source_groups = _load_ood_groups(args.clip_dir, datasets)
        if not per_source_groups:
            logger.error(f"No OOD group labels from {args.clip_dir}")
            return

        # Build source-specific group → column mapping
        # {source_ds: {group_name: [ood_col, ...]}}
        source_group_cols: dict[str, dict[str, list[str]]] = {}
        for src_ds, ood_map in per_source_groups.items():
            norm_map = {_normalise(k): v for k, v in ood_map.items()}
            g2c: dict[str, list[str]] = {}
            for col in ood_cols:
                group = norm_map.get(_normalise(col))
                if group:
                    g2c.setdefault(group, []).append(col)
            source_group_cols[src_ds] = g2c

        # Log source-specific assignments for transparency
        for src_ds in sorted(source_group_cols):
            for grp in ["near", "mid", "far"]:
                cols = source_group_cols[src_ds].get(grp, [])
                if cols:
                    logger.info(f"  {src_ds} {grp}: {cols}")

        for group_label in ["near", "mid", "far", "all"]:
            group_merged = merged.copy()

            if group_label == "all":
                group_merged["group_mean_score"] = group_merged[ood_cols].mean(axis=1)
                logger.info(f"\n{'='*60}")
                logger.info(f"OOD group: all ({len(ood_cols)} OOD sets)")
            else:
                # Compute per-row group mean using source-specific assignments
                group_merged["group_mean_score"] = np.nan
                for src_ds, g2c in source_group_cols.items():
                    cols = g2c.get(group_label, [])
                    if not cols:
                        continue
                    mask = group_merged["dataset"] == src_ds
                    group_merged.loc[mask, "group_mean_score"] = (
                        group_merged.loc[mask, cols].mean(axis=1)
                    )

                # Drop rows with no group assignment (source has no OOD sets in this group)
                n_before = len(group_merged)
                group_merged = group_merged.dropna(subset=["group_mean_score"])
                if len(group_merged) < n_before:
                    logger.info(f"  Dropped {n_before - len(group_merged)} rows with "
                                f"no {group_label}-OOD sets for their source dataset")

                logger.info(f"\n{'='*60}")
                logger.info(f"OOD group: {group_label} (source-specific assignments)")

            if group_merged.empty:
                logger.warning(f"No rows for group '{group_label}'")
                continue

            block_df = build_block_dataset(group_merged, "group_mean_score", ascending)
            if len(block_df) < 5:
                logger.warning(f"Too few blocks ({len(block_df)}) for group {group_label}")
                continue

            run_classification(
                block_df, nc_features,
                label=f"group_{group_label}",
                output_dir=args.output_dir,
                file_prefix=file_prefix,
                min_class_count=args.min_class_count,
            )
    else:
        # Default: mean across all OOD sets
        merged = compute_mean_ood_score(merged, ood_cols)
        block_df = build_block_dataset(merged, "mean_ood_score", ascending)

        if len(block_df) < 5:
            logger.error(f"Too few blocks ({len(block_df)})")
            return

        run_classification(
            block_df, nc_features,
            label="mean_ood",
            output_dir=args.output_dir,
            file_prefix=file_prefix,
            min_class_count=args.min_class_count,
        )


if __name__ == "__main__":
    main()

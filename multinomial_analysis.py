"""
Approach 2: Multilabel Classification (NC -> Set of Strong Methods).

For each model block (dataset × architecture × study × dropout × run), the 8
Papyan NC metrics are the features and the target is the Friedman/Conover top
clique for the corresponding source dataset and OOD grouping, loaded from a
pre-computed JSON file.

We fit:
  (a) One-vs-rest logistic regression (feature-importance analysis only)
  (b) Multilabel random forest (primary LODO predictor)
  (c) One-vs-rest shallow decision tree for interpretable rules

Evaluation:
  - Leave-one-dataset-out cross-validation (LODO)
  - Stratified k-fold CV using the source dataset as the stratification proxy
  - Exact-match, Jaccard, F1, and clique-hit rates
  - Feature importance, SHAP values, optional scorecard analysis

Usage:
    python multinomial_analysis.py --backbone Conv --clique-file ood_eval_outputs/top_cliques_Conv_False_RC_confidnet_cliques.json --papyan-only --filter-methods
    python multinomial_analysis.py --backbone Conv --study confidnet --clique-file ood_eval_outputs/top_cliques_Conv_False_RC_confidnet_cliques.json --papyan-only --filter-methods
    python multinomial_analysis.py --backbone ViT --clique-file ood_eval_outputs/top_cliques_ViT_False_RC_cliques.json --papyan-only --filter-methods
"""

import os
import json
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from loguru import logger

from nc_regime_analysis import (
    NC_METRICS,
    load_nc_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Papyan et al. (2020) subset: NC1 (var_collapse), NC2 (equiangularity,
# equinormness, max equiangularity for means & weights), NC3 (self_duality)
PAPYAN_NC_METRICS = [
    "var_collapse",
    "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc",
    "max_equiangular_uc", "max_equiangular_wc",
    "self_duality",
]

ARCH_MAP = {"Conv": "VGG13", "ViT": "ViT"}

BLOCK_KEYS = ["dataset", "architecture", "study", "dropout", "run"]

# Groups to analyse from the clique JSON (skip "test")
TARGET_GROUPS = ["near", "mid", "far", "all"]


# ── Build block-level dataset from NC metrics + cliques ─────────────────────
def build_block_dataset_from_nc(
    nc: pd.DataFrame,
    nc_features: list[str],
    cliques: dict[str, list[str]],
) -> pd.DataFrame:
    """Build block dataset directly from NC metrics and pre-computed cliques.

    Each row is one model block (dataset × architecture × study × dropout × run)
    with NC features averaged across reward variants and clique targets looked
    up by source dataset.

    Parameters
    ----------
    nc : NC metrics DataFrame (may have multiple reward rows per block)
    nc_features : list of NC metric column names to use
    cliques : {dataset: [method1, method2, ...]} — top clique per source dataset

    Returns
    -------
    DataFrame with BLOCK_KEYS + nc_features + ["clique_methods"]
    """
    available_features = [f for f in nc_features if f in nc.columns]
    if len(available_features) < len(nc_features):
        missing = set(nc_features) - set(available_features)
        logger.warning(f"NC features not found in data: {missing}")

    # Aggregate NC metrics by block (mean across reward variants)
    block_nc = (
        nc[BLOCK_KEYS + available_features]
        .groupby(BLOCK_KEYS, as_index=False)
        .mean(numeric_only=True)
    )

    # Attach clique members per source dataset
    block_nc["clique_methods"] = block_nc["dataset"].map(
        lambda ds: "|".join(cliques[ds]) if ds in cliques else ""
    )

    # Drop rows with missing NC metrics or no clique
    block_nc = block_nc.dropna(subset=available_features)
    block_nc = block_nc[block_nc["clique_methods"] != ""].reset_index(drop=True)

    logger.info(f"Block dataset: {len(block_nc)} blocks, "
                f"{block_nc['dataset'].nunique()} datasets, "
                f"{block_nc['study'].nunique()} studies")
    return block_nc


# ── Target-set helpers ──────────────────────────────────────────────────────
def _parse_method_set(method_str: str, fallback: str | None = None) -> set[str]:
    """Parse a pipe-separated method string into a set."""
    if pd.notna(method_str) and method_str != "":
        return {m for m in method_str.split("|") if m}
    if fallback:
        return {fallback}
    return set()


def _build_target_sets(df: pd.DataFrame) -> list[set[str]]:
    """Return clique target sets from the clique_methods column."""
    return [
        _parse_method_set(method_str)
        for method_str in df["clique_methods"]
    ]


def _filter_target_sets(
    df: pd.DataFrame,
    target_sets: list[set[str]],
    min_class_count: int,
) -> tuple[pd.DataFrame, list[set[str]], list[str]]:
    """Drop rare methods from the multilabel targets and remove empty samples."""
    method_counts = Counter(method for methods in target_sets for method in methods)
    rare_methods = sorted(
        method for method, count in method_counts.items() if count < min_class_count
    )
    if rare_methods:
        logger.info(
            f"Dropping {len(rare_methods)} rare methods (count < {min_class_count}): "
            f"{rare_methods}"
        )

    keep_methods = {method for method, count in method_counts.items() if count >= min_class_count}
    filtered_targets = [methods & keep_methods for methods in target_sets]

    keep_mask = np.array([len(methods) > 0 for methods in filtered_targets], dtype=bool)
    dropped_samples = int((~keep_mask).sum())
    if dropped_samples > 0:
        logger.info(
            f"Dropping {dropped_samples} samples with no remaining multilabel targets"
        )

    filtered_df = df.loc[keep_mask].reset_index(drop=True)
    return (
        filtered_df,
        [filtered_targets[i] for i in np.where(keep_mask)[0]],
        rare_methods,
    )


# ── Prediction and metric helpers ───────────────────────────────────────────
def _indicator_to_method_sets(
    y_indicator: np.ndarray,
    classes: np.ndarray,
) -> list[set[str]]:
    """Convert a binary indicator matrix to predicted method sets."""
    return [
        {str(classes[j]) for j, value in enumerate(row) if value > 0}
        for row in y_indicator
    ]


def _method_sets_to_strings(method_sets: list[set[str]]) -> list[str]:
    """Serialize method sets to stable pipe-separated strings."""
    return ["|".join(sorted(methods)) for methods in method_sets]


def _set_hit_rate(pred_sets: list[set[str]], target_sets: list[set[str]]) -> float:
    """Fraction of samples with any overlap between predicted and target sets."""
    if not pred_sets:
        return 0.0
    hits = sum(1 for pred, target in zip(pred_sets, target_sets) if pred & target)
    return hits / len(pred_sets)


def _sample_exact_match(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact-match multilabel accuracy."""
    return accuracy_score(y_true, y_pred)


def _sample_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sample-wise Jaccard score."""
    return jaccard_score(y_true, y_pred, average="samples", zero_division=0)


def _sample_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sample-wise F1 score."""
    return f1_score(y_true, y_pred, average="samples", zero_division=0)


def _sample_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sample-wise precision."""
    return precision_score(y_true, y_pred, average="samples", zero_division=0)


def _sample_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sample-wise recall."""
    return recall_score(y_true, y_pred, average="samples", zero_division=0)


def _multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_sets: list[set[str]],
    classes: np.ndarray,
) -> dict[str, float]:
    """Compute multilabel exact-match and set-overlap metrics."""
    pred_sets = _indicator_to_method_sets(y_pred, classes)
    return {
        "accuracy": _sample_exact_match(y_true, y_pred),
        "jaccard": _sample_jaccard(y_true, y_pred),
        "f1": _sample_f1(y_true, y_pred),
        "precision": _sample_precision(y_true, y_pred),
        "recall": _sample_recall(y_true, y_pred),
        "clique_hit": _set_hit_rate(pred_sets, target_sets),
    }


def _predict_indicator_matrix(model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predict a multilabel indicator matrix with non-empty fallback."""
    probs = model.predict_proba(X)
    if isinstance(probs, list):
        pos_probs = np.column_stack(
            [p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel() for p in probs]
        )
    else:
        pos_probs = np.asarray(probs)
        if pos_probs.ndim == 1:
            pos_probs = pos_probs[:, None]

    y_pred = (pos_probs >= threshold).astype(int)
    empty_rows = y_pred.sum(axis=1) == 0
    if empty_rows.any():
        best_idx = np.argmax(pos_probs[empty_rows], axis=1)
        y_pred[empty_rows] = 0
        y_pred[np.where(empty_rows)[0], best_idx] = 1
    return y_pred


# ── Classification ───────────────────────────────────────────────────────────
def run_classification(
    block_df: pd.DataFrame,
    nc_features: list[str],
    label: str,
    output_dir: str,
    file_prefix: str,
    min_class_count: int = 2,
    scorecard: bool = False,
):
    """
    Run multilabel classification: NC features -> set of competitive methods.

    Uses:
      1. Leave-one-dataset-out CV (LODO)
      2. Feature importance analysis
      3. SHAP interpretation
      4. Optional scorecard analysis
      5. Decision tree rules
    """
    df = block_df.copy()

    target_sets = _build_target_sets(df)
    df, target_sets, _ = _filter_target_sets(df, target_sets, min_class_count)

    if df.empty:
        logger.warning(f"Not enough multilabel targets for classification ({label})")
        return

    mlb = MultiLabelBinarizer()
    y_multi = mlb.fit_transform(target_sets)
    n_classes = len(mlb.classes_)
    if n_classes < 2:
        logger.warning(f"Need at least 2 classes for classification ({label})")
        return

    X = df[nc_features].to_numpy()
    datasets = df["dataset"].to_numpy()

    logger.info(f"\n{'='*60}")
    logger.info(f"Classification: {label}")
    logger.info(f"  {len(df)} samples, {n_classes} classes, {len(nc_features)} features")
    logger.info(f"  Classes: {list(mlb.classes_)}")
    clique_sizes = [len(c) for c in target_sets if c]
    if clique_sizes:
        logger.info(f"  Clique sizes: min={min(clique_sizes)}, "
                    f"max={max(clique_sizes)}, mean={np.mean(clique_sizes):.1f}")

    results = {}

    def _make_rf():
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            class_weight="balanced_subsample",
        )

    def _make_dt(depth: int):
        return OneVsRestClassifier(
            DecisionTreeClassifier(
                max_depth=depth,
                random_state=42,
                class_weight="balanced",
            )
        )

    # ── 1. Leave-One-Dataset-Out CV ──────────────────────────────────────
    unique_datasets = np.unique(datasets)
    if len(unique_datasets) >= 2:
        logger.info("\n--- Leave-One-Dataset-Out CV ---")
        y_pred_lodo_rf = np.zeros_like(y_multi)
        y_pred_baseline = np.zeros_like(y_multi)
        valid_mask_rf = np.ones(len(df), dtype=bool)
        all_target_strings = np.array(_method_sets_to_strings(target_sets))

        for held_out_ds in unique_datasets:
            test_mask = datasets == held_out_ds
            train_mask = ~test_mask

            y_train = y_multi[train_mask]
            n_active_classes = y_train.sum(axis=0).astype(bool).sum()

            # Per-fold baseline: predict training-set majority label-set
            train_set_counts = Counter(all_target_strings[train_mask])
            majority_set_str = train_set_counts.most_common(1)[0][0]
            majority_row = np.array([1 if m in majority_set_str.split("|") else 0
                                     for m in mlb.classes_])
            y_pred_baseline[test_mask] = np.tile(majority_row, (test_mask.sum(), 1))

            if n_active_classes < 2:
                # Fallback: use majority prediction for RF too
                y_pred_lodo_rf[test_mask] = y_pred_baseline[test_mask]
                logger.info(f"  LODO {held_out_ds}: only {n_active_classes} class in train, "
                            f"using majority fallback for {test_mask.sum()} test samples")
            else:
                rf = _make_rf()
                rf.fit(X[train_mask], y_train)
                y_pred_lodo_rf[test_mask] = _predict_indicator_matrix(rf, X[test_mask])

            test_indices = np.where(test_mask)[0]
            test_targets = [target_sets[i] for i in test_indices]
            metrics_rf = _multilabel_metrics(
                y_multi[test_mask], y_pred_lodo_rf[test_mask], test_targets, mlb.classes_
            )
            logger.info(f"  Hold out {held_out_ds}: "
                        f"RF acc={metrics_rf['accuracy']:.3f}, "
                        f"jaccard={metrics_rf['jaccard']:.3f}, "
                        f"clique_hit={metrics_rf['clique_hit']:.3f} "
                        f"({test_mask.sum()} samples, "
                        f"{int(y_multi[test_mask].sum(axis=0).astype(bool).sum())} active classes)")

        lodo_metrics = _multilabel_metrics(
            y_multi, y_pred_lodo_rf, target_sets, mlb.classes_
        )
        baseline_metrics = _multilabel_metrics(
            y_multi, y_pred_baseline, target_sets, mlb.classes_
        )
        logger.info(
            f"\n  LODO RF overall: acc={lodo_metrics['accuracy']:.3f}, "
            f"jaccard={lodo_metrics['jaccard']:.3f}, f1={lodo_metrics['f1']:.3f}, "
            f"clique_hit={lodo_metrics['clique_hit']:.3f}"
        )
        logger.info(
            f"  Baseline (per-fold majority): acc={baseline_metrics['accuracy']:.3f}, "
            f"jaccard={baseline_metrics['jaccard']:.3f}, f1={baseline_metrics['f1']:.3f}, "
            f"clique_hit={baseline_metrics['clique_hit']:.3f}"
        )

        results["lodo_accuracy"] = lodo_metrics["accuracy"]
        results["lodo_jaccard"] = lodo_metrics["jaccard"]
        results["lodo_f1"] = lodo_metrics["f1"]
        results["lodo_clique_hit"] = lodo_metrics["clique_hit"]
        results["lodo_n_samples"] = len(df)
        results["lodo_baseline_accuracy"] = baseline_metrics["accuracy"]
        results["lodo_baseline_jaccard"] = baseline_metrics["jaccard"]
        results["lodo_baseline_f1"] = baseline_metrics["f1"]
        results["lodo_baseline_clique_hit"] = baseline_metrics["clique_hit"]
        results["lodo_best_model"] = "RF"
        y_pred_lodo = y_pred_lodo_rf
        valid_mask = valid_mask_rf
    else:
        logger.info("Only 1 dataset — skipping LODO CV")
        y_pred_lodo = None

    # ── 2. Feature Importance ────────────────────────────────────────────
    logger.info("\n--- Feature Importance ---")
    rf_full = _make_rf()
    rf_full.fit(X, y_multi)

    importance_df = pd.DataFrame({
        "feature": nc_features,
        "rf_gini_importance": rf_full.feature_importances_,
    }).sort_values("rf_gini_importance", ascending=False)
    importance_df["rf_rank"] = importance_df["rf_gini_importance"].rank(ascending=False).astype(int)

    logger.info(f"\n{importance_df.to_string(index=False)}")

    # ── 3. SHAP Interpretation ──────────────────────────────────────────
    # Fit per-method binary RFs and compute SHAP values.
    # Per-method RFs are used instead of the multi-output RF because
    # SHAP's TreeExplainer produces correct values for single-output models.
    logger.info("\n--- SHAP Interpretation ---")
    import shap

    shap_rows = []
    shap_values_all = {}  # method_name -> (n_samples, n_features) SHAP array
    for k, method_name in enumerate(mlb.classes_):
        rf_k = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced_subsample"
        )
        rf_k.fit(X, y_multi[:, k])

        explainer_k = shap.TreeExplainer(rf_k)
        sv = explainer_k.shap_values(X)

        # sv shape: (n_samples, n_features, 2) — last dim is [class0, class1]
        sv_pos = sv[:, :, 1]
        shap_values_all[method_name] = sv_pos

        mean_abs = np.abs(sv_pos).mean(axis=0)
        mean_signed = sv_pos.mean(axis=0)
        for j, feat in enumerate(nc_features):
            shap_rows.append({
                "method": method_name,
                "feature": feat,
                "mean_abs_shap": mean_abs[j],
                "mean_shap": mean_signed[j],
            })

        # Log top 3 features for this method
        order = np.argsort(-np.abs(mean_signed))
        top3 = ", ".join(
            f"{nc_features[j]}({'+'if mean_signed[j]>0 else '-'}{mean_abs[j]:.3f})"
            for j in order[:3]
        )
        logger.info(f"  {method_name}: {top3}")

    shap_df = pd.DataFrame(shap_rows)

    # Build heatmap matrix: methods × features (mean signed SHAP)
    shap_pivot = shap_df.pivot(index="method", columns="feature", values="mean_shap")
    shap_pivot = shap_pivot[nc_features]  # consistent column order
    shap_abs_pivot = shap_df.pivot(index="method", columns="feature", values="mean_abs_shap")
    shap_abs_pivot = shap_abs_pivot[nc_features]

    logger.info(f"\nMean |SHAP| per method:\n{shap_abs_pivot.round(4).to_string()}")

    # ── 4. Scorecard Analysis (optbinning) ──────────────────────────────
    if scorecard:
        logger.info("\n--- Scorecard Analysis (optbinning) ---")
        from sklearn.linear_model import LogisticRegression
        from optbinning import BinningProcess, Scorecard as OBScorecard

        X_df = pd.DataFrame(X, columns=nc_features)
        sc_tables = []
        iv_rows = []

        for k, method_name in enumerate(mlb.classes_):
            y_binary = y_multi[:, k]

            # Skip constant targets
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                logger.info(f"  {method_name}: skipped (constant target)")
                continue

            # Optimal binning per feature
            binning_process = BinningProcess(
                variable_names=list(nc_features),
                max_n_bins=3,
                min_bin_size=0.15,
            )
            binning_process.fit(X_df, y_binary)

            # IV summary per feature
            bp_summary = binning_process.summary()
            for _, row in bp_summary.iterrows():
                iv_rows.append({
                    "method": method_name,
                    "feature": row["name"],
                    "iv": row["iv"],
                    "n_bins": int(row.get("n_bins", 0)),
                    "quality_score": row.get("quality_score", np.nan),
                })

            # Per-feature binning tables (WoE per bin)
            for feat in nc_features:
                try:
                    optb = binning_process.get_binned_variable(feat)
                    bt_df = optb.binning_table.build()
                    bt_df = bt_df[bt_df["Bin"] != "Totals"].copy()
                    bt_df["method"] = method_name
                    bt_df["feature"] = feat
                    sc_tables.append(bt_df)
                except Exception as e:
                    logger.warning(f"  {method_name}/{feat}: binning table failed: {e}")

            # Build scorecard (LR on WoE-transformed features)
            try:
                lr_sc = LogisticRegression(
                    solver="lbfgs", max_iter=1000, random_state=42,
                    class_weight="balanced",
                )
                ob_scorecard = OBScorecard(
                    binning_process=binning_process,
                    estimator=lr_sc,
                    scaling_method="min_max",
                    scaling_method_params={"min": 0, "max": 100},
                )
                ob_scorecard.fit(X_df, y_binary)
                sc_detail = ob_scorecard.table(style="detailed").copy()
                sc_detail["method"] = method_name
                sc_tables.append(sc_detail)
                logger.info(f"  {method_name}: scorecard fitted, "
                            f"top IV feature={bp_summary.sort_values('iv', ascending=False).iloc[0]['name']}")
            except Exception as e:
                logger.warning(f"  {method_name}: scorecard fitting failed: {e}")
                logger.info(f"  {method_name}: using binning tables only, "
                            f"top IV feature={bp_summary.sort_values('iv', ascending=False).iloc[0]['name']}")

        # Save IV summary
        iv_df = pd.DataFrame(iv_rows)
        iv_path = os.path.join(output_dir, f"multinomial_iv_{file_prefix}_{label}.csv")
        iv_df.to_csv(iv_path, index=False)
        logger.info(f"Saved IV: {iv_path}")

        # Save binning/scorecard tables
        if sc_tables:
            sc_df = pd.concat(sc_tables, ignore_index=True)
            sc_path = os.path.join(output_dir, f"multinomial_scorecard_{file_prefix}_{label}.csv")
            sc_df.to_csv(sc_path, index=False)
            logger.info(f"Saved scorecard: {sc_path}")

        # Log IV heatmap
        if iv_rows:
            iv_pivot = iv_df.pivot(index="method", columns="feature", values="iv")
            iv_pivot = iv_pivot.reindex(columns=nc_features)
            logger.info(f"\nIV per method:\n{iv_pivot.round(4).to_string()}")

    # ── 5. Decision Tree Rules ──────────────────────────────────────────
    # Fit a shallow decision tree on unscaled features to produce
    # interpretable if/then rules with thresholds in original NC units.
    logger.info("\n--- Decision Tree Rules ---")

    # Select DT depth via LODO Jaccard (same fold structure as RF)
    best_dt_jacc = -1.0
    best_dt_depth = 2
    if len(unique_datasets) >= 2:
        for depth in [2, 3, 4, 5]:
            y_pred_dt = np.zeros_like(y_multi)
            for held_out_ds in unique_datasets:
                test_mask = datasets == held_out_ds
                train_mask = ~test_mask
                dt = _make_dt(depth)
                dt.fit(X[train_mask], y_multi[train_mask])
                y_pred_dt[test_mask] = _predict_indicator_matrix(dt, X[test_mask])
            dt_jacc = _sample_jaccard(y_multi, y_pred_dt)
            if dt_jacc > best_dt_jacc:
                best_dt_jacc = dt_jacc
                best_dt_depth = depth

    best_dt = _make_dt(best_dt_depth)
    best_dt.fit(X, y_multi)

    # Evaluate DT with LODO
    if len(unique_datasets) >= 2:
        y_pred_dt_cv = np.zeros_like(y_multi)
        for held_out_ds in unique_datasets:
            test_mask = datasets == held_out_ds
            train_mask = ~test_mask
            dt_fold = _make_dt(best_dt_depth)
            dt_fold.fit(X[train_mask], y_multi[train_mask])
            y_pred_dt_cv[test_mask] = _predict_indicator_matrix(dt_fold, X[test_mask])
        dt_metrics = _multilabel_metrics(
            y_multi, y_pred_dt_cv, target_sets, mlb.classes_
        )
        dt_cv_acc = dt_metrics["accuracy"]
        dt_cv_jaccard = dt_metrics["jaccard"]
        dt_cv_f1 = dt_metrics["f1"]
        dt_cv_cq = dt_metrics["clique_hit"]
    else:
        dt_cv_acc = dt_cv_jaccard = dt_cv_f1 = np.nan
        dt_cv_cq = np.nan

    results["dt_depth"] = best_dt_depth
    results["dt_cv_accuracy"] = dt_cv_acc
    results["dt_cv_jaccard"] = dt_cv_jaccard
    results["dt_cv_f1"] = dt_cv_f1
    results["dt_cv_clique_hit"] = dt_cv_cq

    dt_train_pred = _predict_indicator_matrix(best_dt, X)
    dt_train_acc = _sample_exact_match(y_multi, dt_train_pred)

    logger.info(f"  Best depth: {best_dt_depth}")
    logger.info(f"  Train accuracy: {dt_train_acc:.3f}")
    logger.info(f"  LODO accuracy:  {dt_cv_acc:.3f} (jaccard: {dt_cv_jaccard:.3f}, "
                f"clique_hit: {dt_cv_cq:.3f})")

    tree_rule_blocks = []
    for class_name, estimator in zip(mlb.classes_, best_dt.estimators_):
        class_rules = export_text(
            estimator,
            feature_names=nc_features,
            class_names=[f"not_{class_name}", str(class_name)],
            decimals=4,
        )
        tree_rule_blocks.append(f"=== {class_name} ===\n{class_rules}")
    tree_rules = "\n".join(tree_rule_blocks)
    logger.info(f"\nDecision Tree Rules:\n{tree_rules}")

    # ── Save outputs ────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    results["label"] = label
    results["n_samples"] = len(df)
    results["n_classes"] = n_classes
    results["n_features"] = len(nc_features)
    results["classes"] = ",".join(mlb.classes_)
    results["target_mode"] = "clique"

    summary_df = pd.DataFrame([results])
    summary_path = os.path.join(output_dir, f"multinomial_summary_{file_prefix}_{label}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")

    imp_path = os.path.join(output_dir, f"multinomial_importance_{file_prefix}_{label}.csv")
    importance_df.to_csv(imp_path, index=False)
    logger.info(f"Saved importance: {imp_path}")

    shap_path = os.path.join(output_dir, f"multinomial_shap_{file_prefix}_{label}.csv")
    shap_df.to_csv(shap_path, index=False)
    logger.info(f"Saved SHAP values: {shap_path}")

    shap_heatmap_path = os.path.join(output_dir, f"multinomial_shap_heatmap_{file_prefix}_{label}.csv")
    shap_pivot.to_csv(shap_heatmap_path)
    logger.info(f"Saved SHAP heatmap: {shap_heatmap_path}")

    rules_path = os.path.join(output_dir, f"multinomial_tree_rules_{file_prefix}_{label}.txt")
    with open(rules_path, "w") as f:
        f.write(f"Decision Tree Rules ({label})\n")
        f.write(f"Depth: {best_dt_depth}, Train acc: {dt_train_acc:.3f}, "
                f"LODO acc: {dt_cv_acc:.3f}, LODO jaccard: {dt_cv_jaccard:.3f}\n")
        f.write(f"Classes: {list(mlb.classes_)}\n")
        f.write(f"Features: {nc_features}\n\n")
        f.write(tree_rules)
    logger.info(f"Saved tree rules: {rules_path}")

    if y_pred_lodo is not None:
        if valid_mask.sum() > 0:
            pred_df = df[valid_mask].copy()
            valid_indices = np.where(valid_mask)[0]
            valid_targets = [target_sets[i] for i in valid_indices]
            pred_sets = _indicator_to_method_sets(y_pred_lodo[valid_mask], mlb.classes_)
            pred_df["true_methods"] = _method_sets_to_strings([target_sets[i] for i in valid_indices])
            pred_df["predicted_methods"] = _method_sets_to_strings(pred_sets)
            pred_df["exact_match"] = np.all(
                y_pred_lodo[valid_mask] == y_multi[valid_mask], axis=1
            )
            pred_df["clique_hit"] = [bool(p & t) for p, t in zip(pred_sets, valid_targets)]
            pred_path = os.path.join(output_dir, f"multinomial_lodo_preds_{file_prefix}_{label}.csv")
            pred_df[
                BLOCK_KEYS + [
                    "true_methods",
                    "clique_methods",
                    "predicted_methods",
                    "exact_match",
                    "clique_hit",
                ]
            ].to_csv(
                pred_path, index=False)
            logger.info(f"Saved LODO predictions: {pred_path}")

    return results


# ── Clique file loading ─────────────────────────────────────────────────────
def _load_clique_file(
    clique_file: str,
) -> dict[str, dict[str, list[str]]]:
    """Load pre-computed cliques from a JSON file exported by stats_eval.py.

    Expected format:
        {source_dataset: {group_name: [method1, method2, ...]}}
    where group_name is one of "test", "near", "mid", "far", "all".
    """
    with open(clique_file) as f:
        data = json.load(f)
    logger.info(f"Loaded cliques from {clique_file}")
    for src in sorted(data):
        for grp in sorted(data[src]):
            logger.info(f"  {src} {grp}: {data[src][grp]}")
    return data


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multilabel classification: NC features -> set of competitive OOD methods"
    )
    parser.add_argument("--nc-file", type=str,
                        default="neural_collapse_metrics/nc_metrics.csv")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["Conv", "ViT"])
    parser.add_argument("--study", type=str, default=None,
                        help="Filter to a single study (e.g. confidnet, devries, dg, vit)")
    parser.add_argument("--clique-file", type=str, required=True,
                        help="Path to a JSON clique file exported by stats_eval.py "
                             "(e.g. top_cliques_Conv_False_RC_confidnet_cliques.json).")
    parser.add_argument("--filter-methods", action="store_true",
                        help="Exclude methods containing 'global' or 'class' from cliques "
                             "(except PCA/KPCA RecError global)")
    parser.add_argument("--min-class-count", type=int, default=2,
                        help="Minimum samples per class to include in classification")
    parser.add_argument("--papyan-only", action="store_true",
                        help="Restrict NC metrics to the 8 Papyan et al. (2020) metrics")
    parser.add_argument("--groups", type=str, nargs="*", default=None,
                        help="OOD groups to analyse (default: near mid far all). "
                             "Must be keys present in the clique JSON.")
    parser.add_argument("--output-dir", type=str, default="multinomial_outputs")
    parser.add_argument("--scorecard", action="store_true",
                        help="Run optbinning scorecard analysis (requires optbinning)")
    # Kept for output file naming compatibility
    parser.add_argument("--score-metric", type=str, default="AUGRC",
                        help="Score metric label for output file prefix (default: AUGRC)")
    parser.add_argument("--mcd", type=str, default="False",
                        help="MCD label for output file prefix (default: False)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load NC metrics ─────────────────────────────────────────────────
    logger.info("Loading NC metrics...")
    nc = load_nc_metrics(args.nc_file)
    nc["dataset"] = nc["dataset"].replace({"supercifar": "supercifar100"})

    # Filter to backbone
    arch = ARCH_MAP[args.backbone]
    nc = nc[nc["architecture"] == arch]
    logger.info(f"Filtered to architecture={arch}: {len(nc)} rows")

    # Filter to study
    if args.study:
        nc = nc[nc["study"] == args.study]
        logger.info(f"Filtered to study={args.study}: {len(nc)} rows")

    if nc.empty:
        logger.error("No NC metrics after filtering.")
        return

    # Determine NC features
    metric_pool = PAPYAN_NC_METRICS if args.papyan_only else NC_METRICS
    nc_features = [m for m in metric_pool if m in nc.columns]
    logger.info(f"NC features ({len(nc_features)}): {nc_features}")

    # ── Load cliques ────────────────────────────────────────────────────
    file_cliques = _load_clique_file(args.clique_file)

    # Optional method filtering on clique contents
    if args.filter_methods:
        keep_exceptions = {
            "KPCA RecError global", "PCA RecError global",
            "MCD-KPCA RecError global", "MCD-PCA RecError global",
        }
        n_removed = 0
        for src_ds in file_cliques:
            for grp in file_cliques[src_ds]:
                methods = file_cliques[src_ds][grp]
                filtered = [
                    m for m in methods
                    if not (
                        ("global" in m.lower() or "class" in m.lower())
                        and m not in keep_exceptions
                    )
                ]
                n_removed += len(methods) - len(filtered)
                file_cliques[src_ds][grp] = filtered
        if n_removed > 0:
            logger.info(f"Filtered {n_removed} method entries from cliques")

    # ── Determine groups to run ─────────────────────────────────────────
    available_groups = set()
    for src_ds, group_map in file_cliques.items():
        available_groups.update(group_map.keys())

    if args.groups is not None:
        groups_to_run = args.groups
        missing = set(groups_to_run) - available_groups
        if missing:
            logger.warning(f"Groups not found in clique file: {missing}")
    else:
        groups_to_run = [g for g in TARGET_GROUPS if g in available_groups]

    logger.info(f"Groups to analyse: {groups_to_run}")

    # ── File prefix ─────────────────────────────────────────────────────
    study_tag = f"_{args.study}" if args.study else ""
    file_prefix = f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"

    # ── Run per group ───────────────────────────────────────────────────
    for group_label in groups_to_run:
        # Extract cliques for this group
        cliques = {}
        for src_ds, group_map in file_cliques.items():
            if group_label in group_map and group_map[group_label]:
                cliques[src_ds] = group_map[group_label]

        if not cliques:
            logger.warning(f"No cliques for group '{group_label}', skipping")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"OOD group: {group_label}")
        for src_ds in sorted(cliques):
            logger.info(f"  {src_ds}: {cliques[src_ds]}")

        block_df = build_block_dataset_from_nc(nc, nc_features, cliques)
        if len(block_df) < 5:
            logger.warning(f"Too few blocks ({len(block_df)}) for group {group_label}")
            continue

        run_classification(
            block_df, nc_features,
            label=f"group_{group_label}",
            output_dir=args.output_dir,
            file_prefix=file_prefix,
            min_class_count=args.min_class_count,
            scorecard=args.scorecard,
        )


if __name__ == "__main__":
    main()

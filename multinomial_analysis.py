"""
Approach 2: Multilabel Classification (NC -> Set of Strong Methods).

For each model block (dataset × architecture × study × run), the 13 NC metrics
are the features and the target is a set of competitive OOD detection methods:
  - top-3 methods by mean score across OOD sets, or
  - the Friedman/Conover top clique for the source dataset.

We fit:
  (a) One-vs-rest logistic regression
  (b) Multilabel random forest
  (c) One-vs-rest shallow decision tree for interpretability

Evaluation:
  - Leave-one-dataset-out cross-validation (LODO)
  - Stratified k-fold CV using the rank-1 method as the stratification proxy
  - Exact-match, Jaccard, F1, and set-overlap hit rates
  - Feature importance and per-class logistic coefficients

Also supports OOD-group stratification: instead of averaging across all OOD sets,
the targets are derived from a specific OOD proximity group (near/mid/far).

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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.multiclass import OneVsRestClassifier

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from loguru import logger

from src.utils_stats import (
    HIGHER_BETTER,
    friedman_blocked,
    conover_posthoc_from_pivot,
    maximal_cliques_from_pmatrix,
    rank_cliques,
    greedy_exclusive_layers,
)

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


BLOCK_KEYS = ["dataset", "architecture", "study", "run"]


# ── Clique computation ──────────────────────────────────────────────────────
def _compute_top_cliques(
    merged: pd.DataFrame,
    score_col: str,
    ascending: bool,
    alpha: float = 0.05,
) -> dict[str, list[str]]:
    """Compute the top Friedman/Conover clique per source dataset.

    For each source dataset, performs a Friedman test across methods with
    model configurations as blocks, then extracts the top clique of
    statistically indistinguishable methods via Conover post-hoc.

    Returns
    -------
    {dataset: [method1, method2, ...]}  — top clique members per source.
    """
    merged = merged.copy()
    # Standardize so higher = better (Friedman ranks descending)
    if ascending:  # lower raw score is better (e.g., AUGRC)
        merged[score_col] = -merged[score_col]

    # Block keys within a dataset (exclude "dataset" since it's the groupby)
    inner_block_keys = [k for k in BLOCK_KEYS if k != "dataset"]

    cliques: dict[str, list[str]] = {}
    for ds, ds_data in merged.groupby("dataset"):
        ds_data = ds_data.copy()
        ds_data["_block"] = (
            ds_data[inner_block_keys].astype(str).agg("|".join, axis=1)
        )

        # Keep only methods present in ALL blocks (complete block design)
        block_methods = ds_data.groupby("_block")["methods"].apply(set)
        common_methods = set.intersection(*block_methods) if len(block_methods) > 0 else set()
        if len(common_methods) < 2:
            logger.warning(f"  Clique {ds}: only {len(common_methods)} common methods "
                           f"across {len(block_methods)} blocks, skipping")
            continue
        ds_data = ds_data[ds_data["methods"].isin(common_methods)]

        n_methods = ds_data["methods"].nunique()
        n_blocks = ds_data["_block"].nunique()
        if n_blocks < 2 or n_methods < 2:
            logger.warning(f"  Clique {ds}: only {n_blocks} blocks or "
                           f"{n_methods} methods, skipping")
            continue

        try:
            stat, p, pivot = friedman_blocked(
                ds_data, entity_col="methods", block_col="_block",
                value_col=score_col,
            )
            if not (isinstance(stat, float) and not np.isnan(stat)):
                logger.warning(f"  Clique {ds}: Friedman stat={stat}, skipping")
                continue

            ph = conover_posthoc_from_pivot(pivot)
            ranks = pivot.rank(axis=1, ascending=False)
            avg_ranks = ranks.mean(axis=0).sort_values()
            clique_list = maximal_cliques_from_pmatrix(ph, alpha)
            scored = rank_cliques(clique_list, list(avg_ranks.index), avg_ranks)
            layers = greedy_exclusive_layers(scored)
            if layers:
                top_members = layers[0]["members"]
                cliques[ds] = top_members
                logger.info(f"  Clique {ds}: {top_members} "
                            f"(mean_rank={layers[0]['mean_rank']:.2f}, "
                            f"Friedman p={p:.4f})")
        except Exception as e:
            logger.warning(f"  Clique computation failed for {ds}: {e}")

    return cliques


# ── Build block-level dataset ────────────────────────────────────────────────
def build_block_dataset(
    merged: pd.DataFrame,
    score_col: str,
    ascending: bool,
    cliques: dict[str, list[str]] | None = None,
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

    # For each block, find the best method and top-3 methods
    if ascending:
        ranked = block_scores.sort_values(BLOCK_KEYS + [score_col], ascending=True)
    else:
        ranked = block_scores.sort_values(BLOCK_KEYS + [score_col], ascending=False)

    # Best method (rank 1)
    best = ranked.groupby(BLOCK_KEYS, as_index=False).first()
    best = best.rename(columns={"methods": "best_method", score_col: "best_score"})

    # Top-3 methods per block (pipe-separated string + individual rank columns)
    def _top3_info(group):
        methods = group["methods"].iloc[:3].tolist()
        return pd.Series({
            "top3_methods": "|".join(methods),
            "rank1_method": methods[0] if len(methods) >= 1 else np.nan,
            "rank2_method": methods[1] if len(methods) >= 2 else np.nan,
            "rank3_method": methods[2] if len(methods) >= 3 else np.nan,
        })

    top3 = ranked.groupby(BLOCK_KEYS).apply(_top3_info).reset_index()
    best = best.merge(top3, on=BLOCK_KEYS, how="left")

    # Add clique members per block (looked up from source dataset)
    if cliques:
        best["clique_methods"] = best["dataset"].map(
            lambda ds: "|".join(cliques[ds]) if ds in cliques else ""
        )
    else:
        best["clique_methods"] = ""

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


def _parse_method_set(method_str: str, fallback: str | None = None) -> set[str]:
    """Parse a pipe-separated method string into a set."""
    if pd.notna(method_str) and method_str != "":
        return {m for m in method_str.split("|") if m}
    if fallback:
        return {fallback}
    return set()


def _build_target_sets(
    df: pd.DataFrame,
    use_cliques: bool,
) -> tuple[list[set[str]], list[set[str]], list[set[str]]]:
    """Return target sets plus auxiliary top-3 and clique sets."""
    top3_sets = [
        _parse_method_set(method_str, fallback=best_method)
        for method_str, best_method in zip(df["top3_methods"], df["best_method"])
    ]
    clique_sets = [
        _parse_method_set(method_str, fallback=best_method)
        for method_str, best_method in zip(df["clique_methods"], df["best_method"])
    ]
    target_sets = clique_sets if use_cliques else top3_sets
    return target_sets, top3_sets, clique_sets


def _filter_target_sets(
    df: pd.DataFrame,
    target_sets: list[set[str]],
    top3_sets: list[set[str]],
    clique_sets: list[set[str]],
    min_class_count: int,
) -> tuple[pd.DataFrame, list[set[str]], list[set[str]], list[set[str]], list[str]]:
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
    filtered_top3 = [methods & keep_methods for methods in top3_sets]
    filtered_cliques = [methods & keep_methods for methods in clique_sets]

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
        [filtered_top3[i] for i in np.where(keep_mask)[0]],
        [filtered_cliques[i] for i in np.where(keep_mask)[0]],
        rare_methods,
    )


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
    top3_sets: list[set[str]],
    clique_sets: list[set[str]],
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
        "top3_hit": _set_hit_rate(pred_sets, top3_sets),
        "clique_hit": _set_hit_rate(pred_sets, clique_sets),
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
    use_cliques: bool = False,
    scorecard: bool = False,
):
    """
    Run multilabel classification: NC features -> set of strong methods.

    Uses:
      1. Leave-one-dataset-out CV (LODO)
      2. Stratified 5-fold CV
      3. Feature importance analysis
    """
    df = block_df.copy()

    target_sets, top3_sets, clique_sets = _build_target_sets(df, use_cliques)
    df, target_sets, top3_sets, clique_sets, _ = _filter_target_sets(
        df, target_sets, top3_sets, clique_sets, min_class_count
    )

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
    primary_labels = df["best_method"].to_numpy()
    primary_counts = Counter(primary_labels)
    min_count = min(primary_counts.values())
    train_mode = "clique" if use_cliques else "top3"

    logger.info(f"\n{'='*60}")
    logger.info(f"Classification: {label} (target={train_mode})")
    logger.info(f"  {len(df)} samples, {n_classes} classes, {len(nc_features)} features")
    logger.info(f"  Classes: {list(mlb.classes_)}")
    if use_cliques:
        clique_sizes = [len(c) for c in clique_sets if c]
        if clique_sizes:
            logger.info(f"  Clique sizes: min={min(clique_sizes)}, "
                        f"max={max(clique_sizes)}, mean={np.mean(clique_sizes):.1f}")

    results = {}

    def _make_lr():
        return OneVsRestClassifier(
            LogisticRegression(solver="lbfgs", max_iter=2000, C=1.0, random_state=42)
        )

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
            test_top3 = [top3_sets[i] for i in test_indices]
            test_cliques = [clique_sets[i] for i in test_indices]
            metrics_rf = _multilabel_metrics(
                y_multi[test_mask], y_pred_lodo_rf[test_mask], test_top3, test_cliques, mlb.classes_
            )
            logger.info(f"  Hold out {held_out_ds}: "
                        f"RF acc={metrics_rf['accuracy']:.3f}, "
                        f"jaccard={metrics_rf['jaccard']:.3f}, "
                        f"top3={metrics_rf['top3_hit']:.3f}, "
                        f"clique={metrics_rf['clique_hit']:.3f} "
                        f"({test_mask.sum()} samples, "
                        f"{int(y_multi[test_mask].sum(axis=0).astype(bool).sum())} active classes)")

        lodo_metrics = _multilabel_metrics(
            y_multi, y_pred_lodo_rf, top3_sets, clique_sets, mlb.classes_
        )
        baseline_metrics = _multilabel_metrics(
            y_multi, y_pred_baseline, top3_sets, clique_sets, mlb.classes_
        )
        logger.info(
            f"\n  LODO RF overall: acc={lodo_metrics['accuracy']:.3f}, "
            f"jaccard={lodo_metrics['jaccard']:.3f}, f1={lodo_metrics['f1']:.3f}, "
            f"top3_hit={lodo_metrics['top3_hit']:.3f}, clique_hit={lodo_metrics['clique_hit']:.3f}"
        )
        logger.info(
            f"  Baseline (per-fold majority): acc={baseline_metrics['accuracy']:.3f}, "
            f"jaccard={baseline_metrics['jaccard']:.3f}, f1={baseline_metrics['f1']:.3f}, "
            f"top3_hit={baseline_metrics['top3_hit']:.3f}, clique_hit={baseline_metrics['clique_hit']:.3f}"
        )

        results["lodo_accuracy"] = lodo_metrics["accuracy"]
        results["lodo_jaccard"] = lodo_metrics["jaccard"]
        results["lodo_f1"] = lodo_metrics["f1"]
        results["lodo_top3_hit"] = lodo_metrics["top3_hit"]
        results["lodo_clique_hit"] = lodo_metrics["clique_hit"]
        results["lodo_n_samples"] = len(df)
        results["lodo_baseline_accuracy"] = baseline_metrics["accuracy"]
        results["lodo_baseline_jaccard"] = baseline_metrics["jaccard"]
        results["lodo_baseline_f1"] = baseline_metrics["f1"]
        results["lodo_baseline_top3_hit"] = baseline_metrics["top3_hit"]
        results["lodo_baseline_clique_hit"] = baseline_metrics["clique_hit"]
        results["lodo_best_model"] = "RF"
        y_pred_lodo = y_pred_lodo_rf
        valid_mask = valid_mask_rf
    else:
        logger.info("Only 1 dataset — skipping LODO CV")
        y_pred_lodo = None

    # ── 2. Stratified K-Fold CV ───────────────────────────────────────────
    logger.info("\n--- Stratified K-Fold CV ---")
    n_folds = min(5, min_count)
    if n_folds < 2:
        logger.warning(f"Min class count={min_count}, skipping k-fold CV")
        n_folds = 0

    if n_folds >= 2:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        y_pred_rf = np.zeros_like(y_multi)
        fold_ids = np.full(len(y_multi), -1, dtype=int)

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, primary_labels)):
            fold_ids[test_idx] = fold_i
            y_train = y_multi[train_idx]

            rf_cv = _make_rf()
            rf_cv.fit(X[train_idx], y_train)
            y_pred_rf[test_idx] = _predict_indicator_matrix(rf_cv, X[test_idx])

            fold_top3 = [top3_sets[i] for i in test_idx]
            fold_cliques = [clique_sets[i] for i in test_idx]
            metrics_rf = _multilabel_metrics(
                y_multi[test_idx], y_pred_rf[test_idx], fold_top3, fold_cliques, mlb.classes_
            )
            fold_classes = sorted({method for idx in test_idx for method in target_sets[idx]})
            logger.info(f"  Fold {fold_i+1}/{n_folds}: "
                        f"RF acc={metrics_rf['accuracy']:.3f}, "
                        f"jaccard={metrics_rf['jaccard']:.3f}, "
                        f"top3={metrics_rf['top3_hit']:.3f}, "
                        f"clique={metrics_rf['clique_hit']:.3f} "
                        f"({len(test_idx)} samples, classes: {fold_classes})")

        overall_rf = _multilabel_metrics(y_multi, y_pred_rf, top3_sets, clique_sets, mlb.classes_)
        logger.info(
            f"\n  Overall RF: acc={overall_rf['accuracy']:.3f}, "
            f"jaccard={overall_rf['jaccard']:.3f}, f1={overall_rf['f1']:.3f}, "
            f"top3={overall_rf['top3_hit']:.3f}, clique={overall_rf['clique_hit']:.3f}"
        )

        results["kfold_rf_accuracy"] = overall_rf["accuracy"]
        results["kfold_rf_jaccard"] = overall_rf["jaccard"]
        results["kfold_rf_f1"] = overall_rf["f1"]
        results["kfold_rf_top3_hit"] = overall_rf["top3_hit"]
        results["kfold_rf_clique_hit"] = overall_rf["clique_hit"]
        results["kfold_n_folds"] = n_folds

        baseline_set = Counter(_method_sets_to_strings(target_sets)).most_common(1)[0][0]
        baseline_acc = np.mean(np.array(_method_sets_to_strings(target_sets)) == baseline_set)
        results["kfold_baseline_accuracy"] = baseline_acc
        logger.info(f"  Baseline (most frequent label-set): acc={baseline_acc:.3f}")

        kfold_pred_df = df[BLOCK_KEYS].copy()
        kfold_pred_df["fold"] = fold_ids
        kfold_pred_df["true_methods"] = _method_sets_to_strings(target_sets)
        kfold_pred_df["top3_methods"] = df["top3_methods"].values
        kfold_pred_df["clique_methods"] = df["clique_methods"].values
        pred_rf_sets = _indicator_to_method_sets(y_pred_rf, mlb.classes_)
        kfold_pred_df["pred_rf"] = _method_sets_to_strings(pred_rf_sets)
        kfold_pred_df["exact_match_rf"] = np.all(y_pred_rf == y_multi, axis=1)
        kfold_pred_df["top3_hit_rf"] = [bool(p & t) for p, t in zip(pred_rf_sets, top3_sets)]
        kfold_pred_df["clique_hit_rf"] = [bool(p & t) for p, t in zip(pred_rf_sets, clique_sets)]

    # ── 3. Feature Importance ────────────────────────────────────────────
    logger.info("\n--- Feature Importance ---")
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)

    lr_full = _make_lr()
    lr_full.fit(X_scaled, y_multi)
    rf_full = _make_rf()
    rf_full.fit(X, y_multi)

    lr_coefs = np.vstack([est.coef_.ravel() for est in lr_full.estimators_])
    lr_importance = np.abs(lr_coefs).mean(axis=0)
    lr_importance_df = pd.DataFrame({
        "feature": nc_features,
        "lr_mean_abs_coef": lr_importance,
    }).sort_values("lr_mean_abs_coef", ascending=False)

    rf_importance_df = pd.DataFrame({
        "feature": nc_features,
        "rf_gini_importance": rf_full.feature_importances_,
    }).sort_values("rf_gini_importance", ascending=False)

    importance_df = lr_importance_df.merge(rf_importance_df, on="feature")
    importance_df["lr_rank"] = importance_df["lr_mean_abs_coef"].rank(ascending=False).astype(int)
    importance_df["rf_rank"] = importance_df["rf_gini_importance"].rank(ascending=False).astype(int)
    importance_df = importance_df.sort_values("rf_gini_importance", ascending=False)

    logger.info(f"\n{importance_df.to_string(index=False)}")

    coef_df = pd.DataFrame(
        lr_coefs,
        index=mlb.classes_,
        columns=nc_features,
    )
    logger.info(f"\nLR coefficients per class:\n{coef_df.round(3).to_string()}")

    # ── 4. SHAP Interpretation ──────────────────────────────────────────
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

    # ── 5. Scorecard Analysis (optbinning) ──────────────────────────────
    if scorecard:
        logger.info("\n--- Scorecard Analysis (optbinning) ---")
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
                sc_detail = ob_scorecard.table(style="detailed")
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

    # ── 6. Decision Tree Rules ──────────────────────────────────────────
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
            y_multi, y_pred_dt_cv, top3_sets, clique_sets, mlb.classes_
        )
        dt_cv_acc = dt_metrics["accuracy"]
        dt_cv_jaccard = dt_metrics["jaccard"]
        dt_cv_f1 = dt_metrics["f1"]
        dt_cv_t3 = dt_metrics["top3_hit"]
        dt_cv_cq = dt_metrics["clique_hit"]
    else:
        dt_cv_acc = dt_cv_jaccard = dt_cv_f1 = np.nan
        dt_cv_t3 = dt_cv_cq = np.nan

    results["dt_depth"] = best_dt_depth
    results["dt_cv_accuracy"] = dt_cv_acc
    results["dt_cv_jaccard"] = dt_cv_jaccard
    results["dt_cv_f1"] = dt_cv_f1
    results["dt_cv_top3_hit"] = dt_cv_t3
    results["dt_cv_clique_hit"] = dt_cv_cq

    dt_train_pred = _predict_indicator_matrix(best_dt, X)
    dt_train_acc = _sample_exact_match(y_multi, dt_train_pred)

    logger.info(f"  Best depth: {best_dt_depth}")
    logger.info(f"  Train accuracy: {dt_train_acc:.3f}")
    logger.info(f"  LODO accuracy:  {dt_cv_acc:.3f} (jaccard: {dt_cv_jaccard:.3f}, "
                f"top3: {dt_cv_t3:.3f}, clique: {dt_cv_cq:.3f})")

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

    os.makedirs(output_dir, exist_ok=True)

    results["label"] = label
    results["n_samples"] = len(df)
    results["n_classes"] = n_classes
    results["n_features"] = len(nc_features)
    results["classes"] = ",".join(mlb.classes_)
    results["target_mode"] = train_mode

    summary_df = pd.DataFrame([results])
    summary_path = os.path.join(output_dir, f"multinomial_summary_{file_prefix}_{label}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")

    imp_path = os.path.join(output_dir, f"multinomial_importance_{file_prefix}_{label}.csv")
    importance_df.to_csv(imp_path, index=False)
    logger.info(f"Saved importance: {imp_path}")

    coef_path = os.path.join(output_dir, f"multinomial_coefs_{file_prefix}_{label}.csv")
    coef_df.to_csv(coef_path)
    logger.info(f"Saved coefficients: {coef_path}")

    if n_folds >= 2:
        kfold_path = os.path.join(output_dir, f"multinomial_kfold_preds_{file_prefix}_{label}.csv")
        kfold_pred_df.to_csv(kfold_path, index=False)
        logger.info(f"Saved k-fold predictions: {kfold_path}")

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
            valid_top3 = [top3_sets[i] for i in valid_indices]
            valid_cliques = [clique_sets[i] for i in valid_indices]
            pred_sets = _indicator_to_method_sets(y_pred_lodo[valid_mask], mlb.classes_)
            pred_df["true_methods"] = _method_sets_to_strings([target_sets[i] for i in valid_indices])
            pred_df["predicted_methods"] = _method_sets_to_strings(pred_sets)
            pred_df["exact_match"] = np.all(
                y_pred_lodo[valid_mask] == y_multi[valid_mask], axis=1
            )
            pred_df["top3_hit"] = [bool(p & t) for p, t in zip(pred_sets, valid_top3)]
            pred_df["clique_hit"] = [bool(p & t) for p, t in zip(pred_sets, valid_cliques)]
            pred_path = os.path.join(output_dir, f"multinomial_lodo_preds_{file_prefix}_{label}.csv")
            pred_df[
                BLOCK_KEYS + [
                    "best_method",
                    "true_methods",
                    "top3_methods",
                    "clique_methods",
                    "predicted_methods",
                    "exact_match",
                    "top3_hit",
                    "clique_hit",
                ]
            ].to_csv(
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


def _load_clique_file(
    clique_file: str,
) -> dict[str, dict[str, list[str]]]:
    """Load pre-computed cliques from a JSON file exported by stats_eval.py.

    Expected format:
        {source_dataset: {group_name: [method1, method2, ...]}}
    where group_name is one of "test", "near", "mid", "far".
    """
    import json
    with open(clique_file) as f:
        data = json.load(f)
    logger.info(f"Loaded cliques from {clique_file}")
    for src in sorted(data):
        for grp in sorted(data[src]):
            logger.info(f"  {src} {grp}: {data[src][grp]}")
    return data


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
    parser.add_argument("--use-cliques", action="store_true",
                        help="Use Friedman/Conover top clique for training expansion "
                             "(instead of top-3 rank-based expansion)")
    parser.add_argument("--clique-alpha", type=float, default=0.05,
                        help="Significance level for Conover post-hoc clique test (default: 0.05)")
    parser.add_argument("--clique-file", type=str, default=None,
                        help="Path to a JSON clique file exported by stats_eval.py "
                             "(e.g. top_cliques_Conv_False_RC_cliques.json). "
                             "When provided, cliques are loaded from this file instead of "
                             "being recomputed, ensuring targets match the clique plots exactly. "
                             "Implies --use-cliques.")
    parser.add_argument("--output-dir", type=str, default="multinomial_outputs")
    parser.add_argument("--scorecard", action="store_true",
                        help="Run optbinning scorecard analysis (requires optbinning)")
    args = parser.parse_args()

    # --clique-file implies --use-cliques
    if args.clique_file:
        args.use_cliques = True

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

        # Load pre-computed cliques from file if provided
        file_cliques = None
        if args.clique_file:
            file_cliques = _load_clique_file(args.clique_file)

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

            # Resolve cliques: from file or recomputed
            cliques = None
            if file_cliques is not None:
                # Extract cliques for this group from the pre-computed file
                cliques = {}
                for src_ds, group_map in file_cliques.items():
                    if group_label in group_map:
                        cliques[src_ds] = group_map[group_label]
                if not cliques:
                    logger.warning(f"No pre-computed cliques for group '{group_label}' "
                                   f"in {args.clique_file}")
                    cliques = None
                else:
                    logger.info(f"Using pre-computed cliques from {args.clique_file} "
                                f"for group '{group_label}':")
                    for src_ds in sorted(cliques):
                        logger.info(f"  {src_ds}: {cliques[src_ds]}")
            elif args.use_cliques:
                logger.info(f"Computing Friedman/Conover cliques (alpha={args.clique_alpha})...")
                cliques = _compute_top_cliques(
                    group_merged, "group_mean_score", ascending,
                    alpha=args.clique_alpha,
                )

            block_df = build_block_dataset(
                group_merged, "group_mean_score", ascending, cliques=cliques,
            )
            if len(block_df) < 5:
                logger.warning(f"Too few blocks ({len(block_df)}) for group {group_label}")
                continue

            run_classification(
                block_df, nc_features,
                label=f"group_{group_label}",
                output_dir=args.output_dir,
                file_prefix=file_prefix,
                min_class_count=args.min_class_count,
                use_cliques=args.use_cliques,
                scorecard=args.scorecard,
            )
    else:
        # Default: mean across all OOD sets
        merged = compute_mean_ood_score(merged, ood_cols)

        cliques = None
        if args.clique_file:
            logger.warning("--clique-file without --ood-group: the JSON contains "
                           "per-group cliques, but no group dimension is used here. "
                           "Falling back to recomputing cliques on mean OOD scores.")
        if args.use_cliques:
            logger.info(f"Computing Friedman/Conover cliques (alpha={args.clique_alpha})...")
            cliques = _compute_top_cliques(
                merged, "mean_ood_score", ascending,
                alpha=args.clique_alpha,
            )

        block_df = build_block_dataset(
            merged, "mean_ood_score", ascending, cliques=cliques,
        )

        if len(block_df) < 5:
            logger.error(f"Too few blocks ({len(block_df)})")
            return

        run_classification(
            block_df, nc_features,
            label="mean_ood",
            output_dir=args.output_dir,
            file_prefix=file_prefix,
            min_class_count=args.min_class_count,
            use_cliques=args.use_cliques,
        )


if __name__ == "__main__":
    main()

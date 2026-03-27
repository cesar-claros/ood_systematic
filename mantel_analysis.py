"""
Approach 1: Distance Matrix Correlation (Mantel Test).

Tests whether models with similar Neural Collapse profiles tend to produce
similar OOD detection method rankings.

Two distance formulations for method similarity:
  (a) Jaccard distance on top-K method sets (binary)
  (b) Spearman rank distance on full method rankings

NC similarity uses standardised Euclidean distance on the 13 NC metrics.

The Mantel test assesses the correlation between these two distance matrices
via permutation (default 9999 permutations).

Also runs per-NC-metric Mantel tests (univariate NC distance vs method distance)
to identify which NC properties most strongly predict method selection.
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
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


# ── Mantel test implementation ──────────────────────────────────────────────
def mantel_test(D1: np.ndarray, D2: np.ndarray, n_perms: int = 9999,
                method: str = "spearman") -> dict:
    """
    Mantel test: correlation between two distance matrices.

    Parameters
    ----------
    D1, D2 : square symmetric distance matrices (same size).
    n_perms : number of permutations for p-value.
    method : "spearman" or "pearson".

    Returns
    -------
    dict with r_obs, p_value, n_perms.
    """
    n = D1.shape[0]
    # Extract upper triangle (condensed form)
    idx = np.triu_indices(n, k=1)
    d1 = D1[idx]
    d2 = D2[idx]

    if method == "spearman":
        r_obs, _ = spearmanr(d1, d2)
    else:
        r_obs, _ = pearsonr(d1, d2)

    # Permutation test: permute rows/columns of D2
    rng = np.random.default_rng(42)
    count_ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(n)
        D2_perm = D2[np.ix_(perm, perm)]
        d2_perm = D2_perm[idx]
        if method == "spearman":
            r_perm, _ = spearmanr(d1, d2_perm)
        else:
            r_perm, _ = pearsonr(d1, d2_perm)
        if r_perm >= r_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perms + 1)
    return {"r_obs": r_obs, "p_value": p_value, "n_perms": n_perms}


# ── Distance matrices ──────────────────────────────────────────────────────
def nc_distance_matrix(models_df: pd.DataFrame,
                       nc_cols: list[str]) -> np.ndarray:
    """Standardised Euclidean distance on NC profile."""
    X = models_df[nc_cols].values.astype(float)
    # Standardise each metric to zero mean, unit variance
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_std = (X - mu) / sigma
    return squareform(pdist(X_std, metric="euclidean"))


def nc_distance_matrix_single(models_df: pd.DataFrame,
                               nc_col: str) -> np.ndarray:
    """Euclidean distance on a single NC metric (standardised)."""
    x = models_df[nc_col].values.astype(float).reshape(-1, 1)
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        sigma = 1.0
    x_std = (x - mu) / sigma
    return squareform(pdist(x_std, metric="euclidean"))


def method_rank_distance(rank_matrix: np.ndarray) -> np.ndarray:
    """
    Spearman rank distance between models based on their method rankings.
    rank_matrix: (n_models, n_methods) — rank of each method per model.
    Distance = 1 - Spearman correlation.
    """
    n = rank_matrix.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(rank_matrix[i], rank_matrix[j])
            D[i, j] = D[j, i] = 1.0 - rho
    return D


def jaccard_topk_distance(rank_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Jaccard distance based on top-K method sets.
    rank_matrix: (n_models, n_methods) — rank of each method per model (1=best).
    """
    n = rank_matrix.shape[0]
    # Binary: 1 if method in top-K
    topk = (rank_matrix <= k).astype(float)
    return squareform(pdist(topk, metric="jaccard"))


# ── Build model × method ranking matrix ────────────────────────────────────
def build_rank_matrix(merged: pd.DataFrame, score_col: str,
                      ascending: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build a (models × methods) matrix of method ranks per model.

    Returns
    -------
    models_df : DataFrame with one row per model (with NC metrics)
    rank_matrix : ndarray of shape (n_models, n_methods)
    """
    model_keys = ["dataset", "architecture", "study", "dropout", "run", "reward_val"]

    # Pivot: rows = models, columns = methods, values = score
    pivot = merged.pivot_table(
        index=model_keys, columns="methods", values=score_col,
        aggfunc="first",
    )

    # Drop models or methods with missing values
    pivot = pivot.dropna(axis=0, how="any").dropna(axis=1, how="any")
    logger.info(f"Rank matrix: {pivot.shape[0]} models × {pivot.shape[1]} methods")

    # Rank methods per model
    rank_df = pivot.rank(axis=1, ascending=ascending, method="average")
    rank_matrix = rank_df.values

    # Get NC metrics for these models
    models_idx = pivot.index.to_frame(index=False)
    # Merge NC metrics back
    nc_cols_present = [c for c in NC_METRICS if c in merged.columns or c + "_nc" in merged.columns]
    # Get unique model NC metrics
    nc_key_cols = model_keys + [c for c in merged.columns
                                 if c in NC_METRICS or c.endswith("_nc")]
    model_nc = merged[nc_key_cols].drop_duplicates(subset=model_keys)

    models_df = models_idx.merge(model_nc, on=model_keys, how="inner")

    # Resolve NC metric column names (may have _nc suffix from join)
    for m in NC_METRICS:
        if m not in models_df.columns and m + "_nc" in models_df.columns:
            models_df[m] = models_df[m + "_nc"]

    # Align models_df rows with rank_matrix rows
    models_df = models_df.set_index(model_keys).loc[pivot.index].reset_index()

    return models_df, rank_matrix, pivot.columns.tolist()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Approach 1: Mantel test — NC profile vs OOD method selection"
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
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5],
                        help="K values for Jaccard top-K distance (default: 1 3 5)")
    parser.add_argument("--n-perms", type=int, default=9999,
                        help="Number of permutations for Mantel test")
    parser.add_argument("--per-ood", action="store_true",
                        help="Run Mantel test per individual OOD set "
                             "(instead of averaging across OOD sets)")
    parser.add_argument("--ood-group", action="store_true",
                        help="Run Mantel test per OOD group (near/mid/far) "
                             "using CLIP-based grouping from --clip-dir")
    parser.add_argument("--clip-dir", type=str, default="clip_scores",
                        help="Directory with clip_distances_{dataset}.csv "
                             "(for --per-ood FID ordering)")
    parser.add_argument("--output-dir", type=str, default="mantel_outputs")
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

    # ── Decide score column(s) ───────────────────────────────────────────
    if args.per_ood:
        _run_per_ood(merged, ood_cols, ascending, args, file_prefix, datasets)
    elif args.ood_group:
        _run_ood_group(merged, ood_cols, ascending, args, file_prefix, datasets)
    else:
        merged = compute_mean_ood_score(merged, ood_cols)
        _run_mantel(merged, "mean_ood_score", ascending, args, file_prefix,
                    label="mean_ood")


def _run_mantel(merged: pd.DataFrame, score_col: str, ascending: bool,
                args, file_prefix: str, label: str):
    """Run full Mantel analysis for a given score column."""
    models_df, rank_matrix, method_names = build_rank_matrix(
        merged, score_col, ascending)

    n_models = len(models_df)
    logger.info(f"Running Mantel test on {n_models} models "
                f"({len(method_names)} methods)")

    if n_models < 5:
        logger.error(f"Too few models ({n_models}) for meaningful Mantel test.")
        return

    # ── NC distance (full 13-metric profile) ─────────────────────────────
    nc_cols = [c for c in NC_METRICS if c in models_df.columns]
    D_nc = nc_distance_matrix(models_df, nc_cols)

    # ── Method distance: Spearman rank ───────────────────────────────────
    logger.info("Computing Spearman rank distance...")
    D_rank = method_rank_distance(rank_matrix)

    logger.info("Running Mantel test (full NC profile vs Spearman rank)...")
    result_spearman = mantel_test(D_nc, D_rank, n_perms=args.n_perms)
    logger.info(f"  r = {result_spearman['r_obs']:.4f}, "
                f"p = {result_spearman['p_value']:.4f}")

    # ── Method distance: Jaccard top-K ───────────────────────────────────
    results_jaccard = {}
    for k in args.top_k:
        if k > len(method_names):
            continue
        D_jac = jaccard_topk_distance(rank_matrix, k)
        logger.info(f"Running Mantel test (full NC profile vs Jaccard top-{k})...")
        result = mantel_test(D_nc, D_jac, n_perms=args.n_perms)
        results_jaccard[k] = result
        logger.info(f"  r = {result['r_obs']:.4f}, p = {result['p_value']:.4f}")

    # ── Per-NC-metric Mantel tests ───────────────────────────────────────
    per_metric_rows = []
    for nc_col in nc_cols:
        D_nc_single = nc_distance_matrix_single(models_df, nc_col)
        # vs Spearman rank distance
        res = mantel_test(D_nc_single, D_rank, n_perms=args.n_perms)
        per_metric_rows.append({
            "nc_metric": nc_col,
            "method_distance": "spearman_rank",
            "r": res["r_obs"],
            "p": res["p_value"],
            "significant": res["p_value"] < 0.05,
        })
        # vs Jaccard top-K (use first K value)
        k0 = args.top_k[0]
        if k0 <= len(method_names):
            D_jac = jaccard_topk_distance(rank_matrix, k0)
            res_j = mantel_test(D_nc_single, D_jac, n_perms=args.n_perms)
            per_metric_rows.append({
                "nc_metric": nc_col,
                "method_distance": f"jaccard_top{k0}",
                "r": res_j["r_obs"],
                "p": res_j["p_value"],
                "significant": res_j["p_value"] < 0.05,
            })

    # ── Save results ─────────────────────────────────────────────────────
    summary_rows = []
    summary_rows.append({
        "test": "full_NC_vs_spearman_rank",
        "label": label,
        "n_models": n_models,
        "n_methods": len(method_names),
        "r": result_spearman["r_obs"],
        "p": result_spearman["p_value"],
        "significant": result_spearman["p_value"] < 0.05,
    })
    for k, res in results_jaccard.items():
        summary_rows.append({
            "test": f"full_NC_vs_jaccard_top{k}",
            "label": label,
            "n_models": n_models,
            "n_methods": len(method_names),
            "r": res["r_obs"],
            "p": res["p_value"],
            "significant": res["p_value"] < 0.05,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir,
                                f"mantel_summary_{label}_{file_prefix}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")

    per_metric_df = pd.DataFrame(per_metric_rows)
    per_metric_path = os.path.join(
        args.output_dir,
        f"mantel_per_metric_{label}_{file_prefix}.csv")
    per_metric_df.to_csv(per_metric_path, index=False)
    logger.info(f"Saved per-metric: {per_metric_path}")

    # ── Console summary ──────────────────────────────────────────────────
    logger.info(f"\n=== Mantel Test Results ({label}) ===")
    logger.info(f"Models: {n_models}, Methods: {len(method_names)}")
    logger.info(f"Full NC profile vs Spearman rank distance: "
                f"r={result_spearman['r_obs']:.4f}, "
                f"p={result_spearman['p_value']:.4f}")
    for k, res in results_jaccard.items():
        logger.info(f"Full NC profile vs Jaccard top-{k}: "
                    f"r={res['r_obs']:.4f}, p={res['p_value']:.4f}")
    logger.info("\nPer-NC-metric (vs Spearman rank distance):")
    for _, row in per_metric_df[
            per_metric_df["method_distance"] == "spearman_rank"].iterrows():
        sig = "*" if row["significant"] else ""
        logger.info(f"  {row['nc_metric']:25s}  r={row['r']:.4f}  "
                    f"p={row['p']:.4f} {sig}")


def _load_ood_groups(clip_dir: str, datasets: list[str]) -> dict[str, dict[str, int]]:
    """
    Load OOD group assignments (0=IID, 1=near, 2=mid, 3=far) from clip_distances CSVs.

    Returns {source_dataset: {ood_set: group_int}}.
    """
    GROUP_NAMES = {1: "near", 2: "mid", 3: "far"}
    all_groups = {}
    for ds in datasets:
        path = os.path.join(clip_dir, f"clip_distances_{ds}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        groups = {}
        for ood_set in df.index:
            try:
                g = int(df.loc[ood_set, ("group", "")])
            except (KeyError, ValueError):
                # Try alternative column format
                try:
                    g = int(df.loc[ood_set].iloc[-1])
                except (KeyError, ValueError):
                    continue
            if g in GROUP_NAMES:
                groups[ood_set] = GROUP_NAMES[g]
        all_groups[ds] = groups
    return all_groups


def _run_ood_group(merged: pd.DataFrame, ood_cols: list[str],
                   ascending: bool, args, file_prefix: str,
                   datasets: list[str]):
    """
    Run Mantel test per OOD group (near/mid/far).

    For each source dataset, OOD sets are grouped using CLIP-based labels.
    Within each group, scores are averaged across OOD sets belonging to that group.
    Then a Mantel test is run per group.
    """
    ood_group_map = _load_ood_groups(args.clip_dir, datasets)
    if not ood_group_map:
        logger.error(f"No OOD group labels loaded from {args.clip_dir}")
        return

    # Build a global OOD-set → group mapping (majority vote across source datasets)
    from collections import Counter
    ood_group_votes: dict[str, list[str]] = {}
    for ds_groups in ood_group_map.values():
        for ood_set, group in ds_groups.items():
            ood_group_votes.setdefault(ood_set, []).append(group)
    ood_group_global = {
        ood: Counter(votes).most_common(1)[0][0]
        for ood, votes in ood_group_votes.items()
    }
    logger.info(f"OOD group assignments: {ood_group_global}")

    # For each group, find OOD columns belonging to that group.
    # Build a normalised lookup to handle minor naming differences.
    def _normalise(s: str) -> str:
        return s.lower().strip().replace("_", " ")

    ood_group_norm = {_normalise(k): v for k, v in ood_group_global.items()}

    group_to_cols: dict[str, list[str]] = {}
    for col in ood_cols:
        group = ood_group_norm.get(_normalise(col))
        if group:
            group_to_cols.setdefault(group, []).append(col)

    logger.info("OOD columns per group:")
    for g in ["near", "mid", "far"]:
        cols = group_to_cols.get(g, [])
        logger.info(f"  {g}: {cols}")

    # Run Mantel test per group + overall (all OOD)
    all_results = []
    for group_label in ["near", "mid", "far", "all"]:
        if group_label == "all":
            cols = ood_cols
        else:
            cols = group_to_cols.get(group_label, [])
        if not cols:
            logger.warning(f"No OOD columns for group '{group_label}'")
            continue

        # Compute mean score across OOD sets in this group
        group_merged = merged.copy()
        group_merged["group_mean_score"] = group_merged[cols].mean(axis=1)

        logger.info(f"\n--- OOD group: {group_label} ({len(cols)} OOD sets) ---")
        _run_mantel(group_merged, "group_mean_score", ascending, args,
                    file_prefix, label=f"group_{group_label}")

    # Also produce a compact comparison table
    summary_rows = []
    for group_label in ["near", "mid", "far", "all"]:
        label = f"group_{group_label}"
        summary_path = os.path.join(
            args.output_dir, f"mantel_summary_{label}_{file_prefix}.csv")
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict["ood_group"] = group_label
                summary_rows.append(row_dict)

    if summary_rows:
        comparison = pd.DataFrame(summary_rows)
        comp_path = os.path.join(
            args.output_dir, f"mantel_group_comparison_{file_prefix}.csv")
        comparison.to_csv(comp_path, index=False)
        logger.info(f"\nSaved group comparison: {comp_path}")

        # Console summary
        logger.info("\n=== Mantel Test: OOD Group Comparison ===")
        spearman_rows = comparison[
            comparison["test"] == "full_NC_vs_spearman_rank"]
        for _, row in spearman_rows.iterrows():
            sig = "*" if row["significant"] else ""
            logger.info(f"  {row['ood_group']:6s}  r={row['r']:.4f}  "
                        f"p={row['p']:.4f} {sig}")


def _run_per_ood(merged: pd.DataFrame, ood_cols: list[str],
                 ascending: bool, args, file_prefix: str,
                 datasets: list[str]):
    """Run Mantel test per individual OOD set."""
    from nc_regime_analysis import melt_ood_scores, load_clip_fid

    melted = melt_ood_scores(merged, ood_cols)
    logger.info(f"Melted to {len(melted)} rows, "
                f"{melted['ood_set'].nunique()} OOD sets")

    # Optionally load CLIP FID for ordering
    clip_fid = load_clip_fid(args.clip_dir, datasets)
    avg_fid = {}
    if not clip_fid.empty:
        avg_fid = clip_fid.groupby("ood_set")["clip_fid"].mean().to_dict()

    all_summary_rows = []

    for ood_set in sorted(melted["ood_set"].unique()):
        subset = melted[melted["ood_set"] == ood_set].copy()
        logger.info(f"\n--- OOD set: {ood_set} (FID={avg_fid.get(ood_set, '?')}) ---")

        try:
            models_df, rank_matrix, method_names = build_rank_matrix(
                subset, "mean_ood_score", ascending)
        except Exception as e:
            logger.warning(f"  Skipping {ood_set}: {e}")
            continue

        n_models = len(models_df)
        if n_models < 5:
            logger.warning(f"  Skipping {ood_set}: only {n_models} models")
            continue

        nc_cols = [c for c in NC_METRICS if c in models_df.columns]
        D_nc = nc_distance_matrix(models_df, nc_cols)
        D_rank = method_rank_distance(rank_matrix)

        result = mantel_test(D_nc, D_rank, n_perms=args.n_perms)
        logger.info(f"  Full NC vs rank: r={result['r_obs']:.4f}, "
                    f"p={result['p_value']:.4f}")

        all_summary_rows.append({
            "ood_set": ood_set,
            "clip_fid": avg_fid.get(ood_set, np.nan),
            "n_models": n_models,
            "n_methods": len(method_names),
            "r_spearman": result["r_obs"],
            "p_spearman": result["p_value"],
            "significant": result["p_value"] < 0.05,
        })

        # Per-NC-metric for this OOD set
        for nc_col in nc_cols:
            D_nc_s = nc_distance_matrix_single(models_df, nc_col)
            res = mantel_test(D_nc_s, D_rank, n_perms=args.n_perms)
            all_summary_rows[-1][f"r_{nc_col}"] = res["r_obs"]
            all_summary_rows[-1][f"p_{nc_col}"] = res["p_value"]

    if all_summary_rows:
        df = pd.DataFrame(all_summary_rows)
        # Sort by FID if available
        if "clip_fid" in df.columns:
            df = df.sort_values("clip_fid")
        out_path = os.path.join(args.output_dir,
                                f"mantel_per_ood_{file_prefix}.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"\nSaved per-OOD Mantel results: {out_path}")

        # Console summary
        logger.info("\n=== Per-OOD Mantel Summary ===")
        for _, row in df.iterrows():
            sig = "*" if row["significant"] else ""
            logger.info(f"  {row['ood_set']:20s} FID={row.get('clip_fid', '?'):>6.3f}  "
                        f"r={row['r_spearman']:.4f}  p={row['p_spearman']:.4f} {sig}")


if __name__ == "__main__":
    main()

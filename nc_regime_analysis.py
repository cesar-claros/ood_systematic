"""
Approach 4: Stratified Analysis by Neural Collapse Regime.

For each NC metric, split models into terciles by metric value, then
label bins by degree of neural collapse (high_NC / mid_NC / low_NC).
All 13 NC metrics decrease as collapse increases, so:
  - lowest metric values  → high_NC (most collapsed)
  - middle metric values  → mid_NC
  - highest metric values → low_NC  (least collapsed)

Within each tercile, rank OOD detection methods, run Friedman + Conover
post-hoc tests, and extract top cliques to see if the best method shifts
across NC regimes.

Supports two modes:
- Default: averages scores across all OOD datasets, then stratifies by NC regime.
- --per-ood: keeps per-OOD-set scores and stratifies by NC regime x individual
  OOD set, annotating results with continuous CLIP FID. This avoids the heuristic
  near/mid/far clustering and uses continuous proximity instead.
"""

import os
import math
import argparse
import warnings
import itertools

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.utils_stats import (
    friedman_blocked,
    conover_posthoc_from_pivot,
    maximal_cliques_from_pmatrix,
    rank_cliques,
    greedy_exclusive_layers,
    HIGHER_BETTER,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── NC metric columns ────────────────────────────────────────────────────────
NC_METRICS = [
    "var_collapse", "cdnv_score", "bias_collapse",
    "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc",
    "max_equiangular_uc", "max_equiangular_wc",
    "self_duality", "w_etf_diff", "M_etf_diff", "wM_etf_diff",
]

# ── Key-mapping helpers ──────────────────────────────────────────────────────
ARCH_MAP = {"Conv": "VGG13", "ViT": "ViT"}
STUDY_MAP_INV = {"modelvit": "vit"}  # scores -> nc_metrics


def parse_dropout(val: str) -> str:
    """Convert scores 'do0'/'do1' to nc_metrics 'False'/'True'."""
    return "True" if val == "do1" else "False"


def parse_reward(val: str) -> float:
    """Convert scores 'rew2.2' to float 2.2."""
    return float(val.replace("rew", ""))


# ── Data loading ─────────────────────────────────────────────────────────────
def load_nc_metrics(nc_file: str) -> pd.DataFrame:
    nc = pd.read_csv(nc_file)
    nc["dropout"] = nc["dropout"].astype(str)
    nc["run"] = nc["run"].astype(int)
    nc["reward"] = nc["reward"].astype(float)
    return nc


def load_scores(scores_dir: str, score_metric: str, backbone: str,
                mcd_flag: str, datasets: list[str]) -> pd.DataFrame:
    """Load and concatenate score CSVs, harmonising keys to nc_metrics space."""
    frames = []
    for ds in datasets:
        fname = f"scores_all_{score_metric}_MCD-{mcd_flag}_{backbone}_{ds}.csv"
        path = os.path.join(scores_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Missing scores file: {path}")
            continue
        df = pd.read_csv(path)
        df["dataset"] = ds
        df["architecture"] = ARCH_MAP[backbone]
        # Harmonise study name
        df["study"] = df["model"].map(lambda x: STUDY_MAP_INV.get(x, x))
        # Harmonise dropout
        df["dropout"] = df["drop out"].map(parse_dropout)
        # Harmonise reward
        df["reward_val"] = df["reward"].map(parse_reward)
        df["run"] = df["run"].astype(int)
        frames.append(df)
    if not frames:
        raise RuntimeError("No score files loaded.")
    return pd.concat(frames, ignore_index=True)


# ── Join NC metrics to scores ────────────────────────────────────────────────
JOIN_KEYS_NC = ["dataset", "architecture", "study", "dropout", "run", "reward"]
JOIN_KEYS_SC = ["dataset", "architecture", "study", "dropout", "run", "reward_val"]


def join_nc_scores(nc: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    merged = scores.merge(
        nc, left_on=JOIN_KEYS_SC, right_on=JOIN_KEYS_NC, how="inner",
        suffixes=("_score", "_nc"),
    )
    logger.info(f"Joined table: {len(merged)} rows "
                f"({scores['study'].nunique()} studies, "
                f"{merged['methods'].nunique()} methods)")
    return merged


# ── OOD columns detection ───────────────────────────────────────────────────
META_COLS = {"model", "drop out", "methods", "reward", "run", "test",
             "dataset", "architecture", "study", "dropout", "reward_val",
             "Unnamed: 0"}


def ood_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that hold per-OOD-set scores (exclude IID 'test')."""
    return [c for c in df.columns
            if c not in META_COLS and c not in NC_METRICS
            and not c.endswith("_nc") and not c.endswith("_score")
            and c not in JOIN_KEYS_NC and c != "lr"
            and pd.api.types.is_numeric_dtype(df[c])
            and c != "test"]


# ── Core analysis ────────────────────────────────────────────────────────────
def compute_mean_ood_score(df: pd.DataFrame, ood_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["mean_ood_score"] = df[ood_cols].mean(axis=1)
    return df


def melt_ood_scores(df: pd.DataFrame, ood_cols: list[str]) -> pd.DataFrame:
    """Unpivot OOD columns so each row is one (model, method, ood_set, score)."""
    id_vars = [c for c in df.columns if c not in ood_cols]
    melted = df.melt(id_vars=id_vars, value_vars=ood_cols,
                     var_name="ood_set", value_name="mean_ood_score")
    melted = melted.dropna(subset=["mean_ood_score"])
    # Remove self-referential rows (ood_set == source_dataset)
    melted = melted[melted["ood_set"] != melted["dataset"]]
    return melted


def load_clip_fid(clip_dir: str, datasets: list[str]) -> pd.DataFrame:
    """Load CLIP FID distances for each (source_dataset, ood_set) pair."""
    rows = []
    for ds in datasets:
        path = os.path.join(clip_dir, f"clip_distances_{ds}.csv")
        if not os.path.exists(path):
            logger.warning(f"Missing CLIP distances: {path}")
            continue
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        for ood_set in df.index:
            if ood_set == "test":
                continue
            try:
                fid = float(df.loc[ood_set, ("global", "fid")])
            except (KeyError, ValueError):
                continue
            rows.append({
                "dataset": ds,
                "ood_set": ood_set,
                "clip_fid": fid,
            })
    return pd.DataFrame(rows)


BLOCK_KEYS = ["dataset", "architecture", "study", "run"]


def bin_nc_metric(df: pd.DataFrame, nc_metric: str, n_bins: int) -> pd.DataFrame:
    """Bin at block level (dataset, architecture, study, run), averaging NC metrics
    across dropout/reward since different methods selected different hyperparameters."""
    col = nc_metric if nc_metric in df.columns else nc_metric + "_nc"
    # All NC metrics decrease as collapse increases, so lowest values = most collapsed.
    # Label bins by degree of neural collapse (not raw metric value).
    labels = ["high_NC", "mid_NC", "low_NC"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    df = df.copy()
    # Average NC metric across dropout/reward per block, then bin
    block_nc = df.groupby(BLOCK_KEYS, as_index=False)[col].mean()
    block_nc["nc_bin"] = pd.qcut(block_nc[col], q=n_bins, labels=labels, duplicates="drop")
    df = df.drop(columns=["nc_bin"], errors="ignore")
    df = df.merge(block_nc[BLOCK_KEYS + ["nc_bin"]], on=BLOCK_KEYS, how="left")
    return df


def run_regime_analysis(df: pd.DataFrame, score_metric: str,
                        alpha: float = 0.05) -> dict:
    """
    Within a single NC-regime bin, run Friedman test on method rankings.
    Returns dict with stat results and top clique.
    """
    ascending = not HIGHER_BETTER.get(score_metric, True)

    # Block = (dataset, study, run). Dropout and reward are hyperparameters already
    # selected per method on validation, so each (dataset, study, run, method) has
    # exactly one row. For dg (multiple rewards), we average across reward values.
    block_cols = ["dataset", "study", "run"]
    df = df.copy()

    agg_cols = block_cols + ["methods"]
    df = df.groupby(agg_cols, as_index=False).agg(
        mean_ood_score=("mean_ood_score", "mean")
    )
    logger.info(f"After aggregation: {df['methods'].nunique()} methods, "
                f"{df[block_cols].drop_duplicates().shape[0]} blocks")

    df["block"] = df[block_cols].astype(str).agg("|".join, axis=1)

    # Standardise scores so ranking direction is always "higher = better"
    if ascending:
        df["score_std"] = -df["mean_ood_score"]
    else:
        df["score_std"] = df["mean_ood_score"]

    # Iteratively find the largest complete (blocks x methods) sub-matrix.
    # A complete sub-matrix means every block has every method and vice versa.
    n_methods_orig = df["methods"].nunique()
    n_blocks_orig = df["block"].nunique()
    for _ in range(20):  # converges in a few iterations
        n_blocks = df["block"].nunique()
        n_methods = df["methods"].nunique()
        # Drop methods not in all blocks
        method_counts = df.groupby("methods")["block"].nunique()
        complete_methods = method_counts[method_counts == n_blocks].index
        # Drop blocks not having all remaining methods
        block_counts = df.groupby("block")["methods"].nunique()
        complete_blocks = block_counts[block_counts == n_methods].index
        df_next = df[df["methods"].isin(complete_methods) & df["block"].isin(complete_blocks)]
        if len(df_next) == len(df):
            break
        df = df_next

    n_methods_final = df["methods"].nunique()
    n_blocks_final = df["block"].nunique()

    if n_methods_final < 2 or n_blocks_final < 2:
        logger.warning(f"Not enough complete data: {n_methods_final} methods, "
                       f"{n_blocks_final} blocks "
                       f"(started with {n_methods_orig} methods, {n_blocks_orig} blocks)")
        return {"stat": np.nan, "p": np.nan, "top_clique": [], "avg_ranks": None,
                "n_methods_used": n_methods_final, "n_blocks_used": n_blocks_final}

    logger.info(f"Complete sub-matrix: {n_blocks_final} blocks x "
                f"{n_methods_final} methods "
                f"(from {n_blocks_orig} x {n_methods_orig})")

    try:
        stat, p, pivot = friedman_blocked(
            df, entity_col="methods", block_col="block", value_col="score_std"
        )
    except Exception as e:
        logger.error(f"Friedman test failed: {e}")
        return {"stat": np.nan, "p": np.nan, "top_clique": [], "avg_ranks": None,
                "n_methods_used": n_methods_final, "n_blocks_used": n_blocks_final}

    if isinstance(stat, float) and math.isnan(stat):
        return {"stat": np.nan, "p": np.nan, "top_clique": [], "avg_ranks": None,
                "n_methods_used": pivot.shape[1], "n_blocks_used": pivot.shape[0]}

    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0).sort_values()

    top_clique = []
    if p < alpha:
        ph = conover_posthoc_from_pivot(pivot)
        cliques = maximal_cliques_from_pmatrix(ph, alpha)
        scored = rank_cliques(cliques, list(avg_ranks.index), avg_ranks)
        layers = greedy_exclusive_layers(scored)
        if layers:
            top_clique = layers[0]["members"]

    return {
        "stat": stat,
        "p": p,
        "top_clique": top_clique,
        "avg_ranks": avg_ranks,
        "n_methods_used": n_methods_final,
        "n_blocks_used": n_blocks_final,
    }


# ── Per-OOD pipeline ─────────────────────────────────────────────────────────
def run_per_ood_pipeline(merged: pd.DataFrame, ood_cols: list[str],
                         args, bin_labels: list[str], file_prefix: str,
                         datasets: list[str]):
    """Run NC regime analysis per individual OOD set with CLIP FID annotation."""
    # Melt OOD columns into long format
    melted = melt_ood_scores(merged, ood_cols)
    logger.info(f"Melted to {len(melted)} rows, "
                f"{melted['ood_set'].nunique()} OOD sets: "
                f"{sorted(melted['ood_set'].unique())}")

    # Load CLIP FID for continuous proximity
    clip_fid = load_clip_fid(args.clip_dir, datasets)
    if clip_fid.empty:
        logger.error(f"No CLIP distances loaded from {args.clip_dir}. "
                     "Check --clip-dir path.")
        return

    # Compute average FID per OOD set (across source datasets) for ordering
    avg_fid = clip_fid.groupby("ood_set")["clip_fid"].mean().sort_values()
    logger.info(f"OOD sets by avg FID: "
                f"{', '.join(f'{k} ({v:.3f})' for k, v in avg_fid.items())}")

    # Join per-row CLIP FID
    melted = melted.merge(clip_fid, on=["dataset", "ood_set"], how="left")

    ood_sets_ordered = list(avg_fid.index)
    summary_rows = []
    all_rank_data = []

    for nc_metric in NC_METRICS:
        logger.info(f"Analysing NC metric: {nc_metric}")
        binned = bin_nc_metric(melted, nc_metric, args.n_bins)

        for bin_label in bin_labels:
            for ood_set in ood_sets_ordered:
                subset = binned[(binned["nc_bin"] == bin_label) &
                                (binned["ood_set"] == ood_set)]
                n_models = subset[JOIN_KEYS_SC].drop_duplicates().shape[0]
                n_methods = subset["methods"].nunique()

                if subset.empty or n_methods < 2:
                    continue

                result = run_regime_analysis(subset, args.score_metric,
                                             args.alpha)

                ood_fid = avg_fid.get(ood_set, np.nan)
                summary_rows.append({
                    "nc_metric": nc_metric,
                    "bin": bin_label,
                    "ood_set": ood_set,
                    "clip_fid": ood_fid,
                    "n_models": n_models,
                    "n_methods_input": n_methods,
                    "n_methods_used": result.get("n_methods_used", ""),
                    "n_blocks_used": result.get("n_blocks_used", ""),
                    "friedman_stat": result["stat"],
                    "friedman_p": result["p"],
                    "significant": (result["p"] < args.alpha
                                    if not math.isnan(result["p"]) else False),
                    "top_clique": (", ".join(result["top_clique"])
                                   if result["top_clique"] else ""),
                    "top_clique_size": len(result["top_clique"]),
                })

                if result["avg_ranks"] is not None:
                    for method, rank in result["avg_ranks"].items():
                        all_rank_data.append({
                            "nc_metric": nc_metric,
                            "bin": bin_label,
                            "ood_set": ood_set,
                            "clip_fid": ood_fid,
                            "method": method,
                            "avg_rank": rank,
                        })

    if not summary_rows:
        logger.error("No per-OOD results produced.")
        return

    # ── Output CSVs ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir,
                                f"nc_regime_ood_summary_{file_prefix}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved per-OOD summary: {summary_path}")

    rank_df = pd.DataFrame(all_rank_data)
    rank_path = os.path.join(args.output_dir,
                             f"nc_regime_ood_ranks_{file_prefix}.csv")
    rank_df.to_csv(rank_path, index=False)
    logger.info(f"Saved per-OOD ranks: {rank_path}")

    # ── 2D clique shift table: NC bin x OOD set (ordered by FID) ─────────
    for nc_metric in NC_METRICS:
        sub = summary_df[summary_df["nc_metric"] == nc_metric]
        if sub.empty:
            continue
        pivot = sub.pivot(index="ood_set", columns="bin", values="top_clique")
        pivot = pivot.reindex(columns=bin_labels)
        # Order rows by average FID
        ood_order = [o for o in ood_sets_ordered if o in pivot.index]
        pivot = pivot.reindex(ood_order)
        # Add FID column for reference
        pivot.insert(0, "avg_clip_fid",
                     [avg_fid.get(o, np.nan) for o in pivot.index])
        clique_path = os.path.join(
            args.output_dir,
            f"nc_ood_clique_shift_{nc_metric}_{file_prefix}.csv")
        pivot.to_csv(clique_path)

    # ── Combined clique shift across all NC metrics ──────────────────────
    # One row per (nc_metric, ood_set), columns = NC bins
    all_clique_rows = []
    for nc_metric in NC_METRICS:
        sub = summary_df[summary_df["nc_metric"] == nc_metric]
        for ood_set in ood_sets_ordered:
            row = {"nc_metric": nc_metric, "ood_set": ood_set,
                   "clip_fid": avg_fid.get(ood_set, np.nan)}
            for bl in bin_labels:
                cell = sub[(sub["ood_set"] == ood_set) & (sub["bin"] == bl)]
                row[bl] = cell["top_clique"].values[0] if len(cell) else ""
            all_clique_rows.append(row)

    all_clique_df = pd.DataFrame(all_clique_rows)
    all_clique_path = os.path.join(
        args.output_dir, f"nc_ood_clique_shift_all_{file_prefix}.csv")
    all_clique_df.to_csv(all_clique_path, index=False)
    logger.info(f"Saved combined clique shift: {all_clique_path}")

    # ── Heatmap per NC metric: method rank across (NC bin x OOD set) ─────
    if rank_df.empty:
        logger.warning("No rank data for per-OOD heatmaps.")
        return

    for nc_metric in NC_METRICS:
        sub = rank_df[rank_df["nc_metric"] == nc_metric]
        if sub.empty:
            continue

        # For each bin, create a panel showing method ranks across OOD sets
        fig, axes = plt.subplots(1, len(bin_labels),
                                 figsize=(5 * len(bin_labels),
                                          max(4, sub["method"].nunique() * 0.3)),
                                 sharey=True)
        if len(bin_labels) == 1:
            axes = [axes]

        for ax, bl in zip(axes, bin_labels):
            bl_sub = sub[sub["bin"] == bl]
            if bl_sub.empty:
                ax.set_title(f"{bl}\n(no data)")
                continue
            hm = bl_sub.pivot(index="method", columns="ood_set",
                              values="avg_rank")
            # Order OOD columns by FID
            ood_order = [o for o in ood_sets_ordered if o in hm.columns]
            hm = hm.reindex(columns=ood_order)
            # Sort methods by mean rank across OOD sets
            hm = hm.loc[hm.mean(axis=1).sort_values().index]

            # Annotate column labels with FID
            col_labels = [f"{o}\n({avg_fid.get(o, 0):.2f})"
                          for o in ood_order]

            sns.heatmap(hm, annot=True, fmt=".1f", cmap="RdYlGn_r",
                        linewidths=0.3, ax=ax, cbar=bl == bin_labels[-1],
                        xticklabels=col_labels)
            ax.set_title(f"{bl}")
            ax.set_xlabel("OOD set (FID)")
            if bl != bin_labels[0]:
                ax.set_ylabel("")

        fig.suptitle(f"Method rank: {nc_metric} regime x OOD proximity\n"
                     f"({args.score_metric}, {args.backbone}, "
                     f"study={args.study or 'all'})",
                     fontsize=12)
        fig.tight_layout()
        fig_path = os.path.join(args.output_dir,
                                f"heatmap_ood_{nc_metric}_{file_prefix}")
        fig.savefig(fig_path + ".pdf", bbox_inches="tight")
        fig.savefig(fig_path + ".jpeg", bbox_inches="tight", dpi=150)
        plt.close(fig)

    logger.success(f"Per-OOD outputs saved to {args.output_dir}/")

    # ── Console summary ──────────────────────────────────────────────────
    sig_count = summary_df["significant"].sum()
    total = len(summary_df)
    logger.info(f"\n=== Per-OOD Summary: {sig_count}/{total} tests significant ===")
    for nc_metric in NC_METRICS:
        sub = summary_df[summary_df["nc_metric"] == nc_metric]
        sig = sub[sub["significant"]]
        logger.info(f"  {nc_metric}: {len(sig)}/{len(sub)} significant")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Approach 4: Stratified analysis of OOD methods by Neural Collapse regime"
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
                        help="Filter to a single study (e.g. confidnet, devries, dg, vit). Default: all.")
    parser.add_argument("--n-bins", type=int, default=3,
                        help="Number of bins for NC metric stratification (default: 3 = terciles)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--filter-methods", action="store_true",
                        help="Exclude methods containing 'global' or 'class' (except PCA/KPCA RecError global)")
    parser.add_argument("--per-ood", action="store_true",
                        help="Run per-OOD-set analysis instead of averaging across OOD sets. "
                             "Annotates results with continuous CLIP FID.")
    parser.add_argument("--clip-dir", type=str, default="clip_scores",
                        help="Directory with clip_distances_{dataset}.csv files (for --per-ood)")
    parser.add_argument("--output-dir", type=str, default="nc_analysis_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading NC metrics...")
    nc = load_nc_metrics(args.nc_file)

    logger.info("Loading scores...")
    # nc_metrics uses 'supercifar' but scores filenames use 'supercifar100'
    nc["dataset"] = nc["dataset"].replace({"supercifar": "supercifar100"})
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
        logger.error("No rows after join. Check key mappings.")
        return

    ood_cols = ood_columns(merged)
    logger.info(f"OOD columns: {ood_cols}")

    # ── Per-OOD mode: stratify by NC regime x individual OOD set ─────────
    if args.per_ood:
        study_tag = f"_{args.study}" if args.study else ""
        file_prefix = f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"
        bin_labels = (["high_NC", "mid_NC", "low_NC"] if args.n_bins == 3
                      else [f"bin{i}" for i in range(args.n_bins)])
        run_per_ood_pipeline(merged, ood_cols, args, bin_labels,
                             file_prefix, datasets)
        return

    merged = compute_mean_ood_score(merged, ood_cols)

    # ── Run stratified analysis per NC metric ────────────────────────────
    summary_rows = []
    all_rank_data = []

    bin_labels = ["high_NC", "mid_NC", "low_NC"] if args.n_bins == 3 else [f"bin{i}" for i in range(args.n_bins)]

    for nc_metric in NC_METRICS:
        logger.info(f"Analysing NC metric: {nc_metric}")
        binned = bin_nc_metric(merged, nc_metric, args.n_bins)

        for bin_label in bin_labels:
            subset = binned[binned["nc_bin"] == bin_label]
            n_models = subset[JOIN_KEYS_SC].drop_duplicates().shape[0]
            n_methods = subset["methods"].nunique()

            if subset.empty or n_methods < 2:
                logger.warning(f"  {nc_metric}/{bin_label}: not enough data (n_models={n_models})")
                continue

            result = run_regime_analysis(subset, args.score_metric, args.alpha)

            summary_rows.append({
                "nc_metric": nc_metric,
                "bin": bin_label,
                "n_models": n_models,
                "n_methods_input": n_methods,
                "n_methods_used": result.get("n_methods_used", ""),
                "n_blocks_used": result.get("n_blocks_used", ""),
                "friedman_stat": result["stat"],
                "friedman_p": result["p"],
                "significant": result["p"] < args.alpha if not math.isnan(result["p"]) else False,
                "top_clique": ", ".join(result["top_clique"]) if result["top_clique"] else "",
                "top_clique_size": len(result["top_clique"]),
            })

            if result["avg_ranks"] is not None:
                for method, rank in result["avg_ranks"].items():
                    all_rank_data.append({
                        "nc_metric": nc_metric,
                        "bin": bin_label,
                        "method": method,
                        "avg_rank": rank,
                    })

    if not summary_rows:
        logger.error("No results produced.")
        return

    # ── Output CSVs ──────────────────────────────────────────────────────
    study_tag = f"_{args.study}" if args.study else ""
    file_prefix = f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, f"nc_regime_summary_{file_prefix}.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")

    rank_df = pd.DataFrame(all_rank_data)
    rank_path = os.path.join(args.output_dir, f"nc_regime_ranks_{file_prefix}.csv")
    rank_df.to_csv(rank_path, index=False)
    logger.info(f"Saved ranks: {rank_path}")

    # ── Clique shift table ───────────────────────────────────────────────
    clique_pivot = summary_df.pivot(index="nc_metric", columns="bin", values="top_clique")
    clique_pivot = clique_pivot.reindex(columns=bin_labels)
    clique_path = os.path.join(args.output_dir, f"nc_clique_shift_{file_prefix}.csv")
    clique_pivot.to_csv(clique_path)
    logger.info(f"Saved clique shift table: {clique_path}")

    # ── Heatmap: mean rank per method across NC regimes ──────────────────
    if rank_df.empty:
        logger.warning("No rank data to plot heatmaps (no significant bins found).")
    for nc_metric in NC_METRICS:
        if rank_df.empty:
            break
        sub = rank_df[rank_df["nc_metric"] == nc_metric]
        if sub.empty:
            continue
        heatmap_data = sub.pivot(index="method", columns="bin", values="avg_rank")
        heatmap_data = heatmap_data.reindex(columns=bin_labels)
        heatmap_data = heatmap_data.sort_values(by=bin_labels[0], ascending=True)

        fig, ax = plt.subplots(figsize=(6, max(4, len(heatmap_data) * 0.35)))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="RdYlGn_r",
                    linewidths=0.5, ax=ax)
        ax.set_title(f"Mean rank by {nc_metric} regime\n({args.score_metric}, {args.backbone}, study={args.study or 'all'})")
        ax.set_xlabel("NC regime")
        ax.set_ylabel("")
        fig_path = os.path.join(args.output_dir, f"heatmap_{nc_metric}_{file_prefix}")
        fig.savefig(fig_path + ".pdf", bbox_inches="tight")
        fig.savefig(fig_path + ".jpeg", bbox_inches="tight")
        plt.close(fig)

    logger.success(f"All outputs saved to {args.output_dir}/")

    # ── Print summary to console ─────────────────────────────────────────
    logger.info("\n=== Top Clique Shifts Across NC Regimes ===")
    for nc_metric in NC_METRICS:
        rows = summary_df[summary_df["nc_metric"] == nc_metric]
        cliques = rows.set_index("bin")["top_clique"].to_dict()
        unique_cliques = set(c for c in cliques.values() if c)
        shifted = len(unique_cliques) > 1
        marker = " << SHIFT" if shifted else ""
        logger.info(f"  {nc_metric}:{marker}")
        for b in bin_labels:
            c = cliques.get(b, "")
            logger.info(f"    {b}: {c if c else '(no significant clique)'}")


if __name__ == "__main__":
    main()

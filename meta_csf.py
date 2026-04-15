"""Meta-CSF: NC-metric-guided OOD detection method selector.

Given a model's Neural Collapse metric profile, the meta-CSF selects the
best OOD detection method by partitioning the most predictive NC metric
into segments and assigning the best-performing method to each segment.

Pipeline
--------
1. Load NC metrics + AUGRC scores  →  regression dataset
2. For each OOD group, identify the most predictive NC metric (by IV)
3. Optimally bin that metric; within each bin pick the method with the
   lowest mean avg_ood_rank
4. Evaluate: for every model instance, the meta-CSF recommends a method
   based on the bin → compare actual AUGRC against oracle, best single
   method, and clique mean
5. Visualise: NC metric on x-axis, method ranks as lines, bin boundaries
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
    # Create a dummy binary target: top-3 overall
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


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_meta_csf(
    df: pd.DataFrame,
    feature: str,
    group_label: str,
    splits: list[float],
    bin_methods: list[str],
    cliques: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Evaluate the meta-CSF against oracle and best single method.

    For each model instance (dataset, study, dropout, reward, run):
      - meta-CSF picks a method based on the NC metric bin
      - oracle picks the actual best method
      - best_single picks the method with the lowest overall mean rank

    Returns a DataFrame with one row per model instance.
    """
    gdf = df[df["group"] == group_label].copy()

    # Assign bins
    bin_edges = [-np.inf] + splits + [np.inf]
    gdf["nc_bin"] = pd.cut(
        gdf[feature], bins=bin_edges, labels=False, include_lowest=True,
    )

    # Best single method (lowest overall mean rank in this group)
    overall_ranks = gdf.groupby("method")["avg_ood_rank"].mean().sort_values()
    best_single_method = overall_ranks.index[0]
    logger.info(
        f"  Best single method: {best_single_method} "
        f"(mean rank={overall_ranks.iloc[0]:.2f})"
    )

    # Evaluate per model instance
    instance_keys = ["dataset", "study", "dropout", "reward", "run"]
    results = []

    for keys, inst in gdf.groupby(instance_keys):
        key_dict = dict(zip(instance_keys, keys))
        dataset = key_dict["dataset"]

        # NC metric value (should be same for all methods in this instance)
        nc_val = inst[feature].iloc[0]
        bin_id = inst["nc_bin"].iloc[0]

        # Actual AUGRC per method
        augrc_map = dict(zip(inst["method"], inst["augrc"]))
        rank_map = dict(zip(inst["method"], inst["avg_ood_rank"]))

        # Oracle: actual best
        oracle_method = min(augrc_map, key=augrc_map.get)
        oracle_augrc = augrc_map[oracle_method]

        # Best single method
        single_augrc = augrc_map.get(best_single_method, np.nan)

        # Meta-CSF recommendation
        if pd.notna(bin_id) and int(bin_id) < len(bin_methods):
            meta_method = bin_methods[int(bin_id)]
            meta_augrc = augrc_map.get(meta_method, np.nan)
        else:
            meta_method = best_single_method
            meta_augrc = single_augrc

        # Regret computations
        meta_regret = meta_augrc - oracle_augrc if not np.isnan(meta_augrc) else np.nan
        single_regret = single_augrc - oracle_augrc if not np.isnan(single_augrc) else np.nan
        meta_norm_regret = meta_regret / oracle_augrc if oracle_augrc else 0
        single_norm_regret = single_regret / oracle_augrc if oracle_augrc else 0

        # Clique evaluation
        clique_hit = False
        clique_augrc = np.nan
        if cliques and dataset in cliques:
            clique_members = set(cliques[dataset])
            clique_hit = meta_method in clique_members
            clique_vals = [augrc_map[m] for m in clique_members if m in augrc_map]
            if clique_vals:
                clique_augrc = np.mean(clique_vals)

        row = {
            **key_dict,
            f"{feature}": nc_val,
            "nc_bin": int(bin_id) if pd.notna(bin_id) else -1,
            "oracle_method": oracle_method,
            "oracle_augrc": oracle_augrc,
            "best_single_method": best_single_method,
            "best_single_augrc": single_augrc,
            "best_single_regret": single_regret,
            "best_single_norm_regret": single_norm_regret,
            "meta_method": meta_method,
            "meta_augrc": meta_augrc,
            "meta_regret": meta_regret,
            "meta_norm_regret": meta_norm_regret,
            "clique_hit": clique_hit,
            "clique_mean_augrc": clique_augrc,
        }
        results.append(row)

    eval_df = pd.DataFrame(results)

    # Summary
    logger.info(f"  --- Meta-CSF evaluation ({group_label}) ---")
    logger.info(
        f"  Meta-CSF mean regret:    {eval_df['meta_regret'].mean():.4f} "
        f"({eval_df['meta_norm_regret'].mean():.1%})"
    )
    logger.info(
        f"  Best-single mean regret: {eval_df['best_single_regret'].mean():.4f} "
        f"({eval_df['best_single_norm_regret'].mean():.1%})"
    )
    if cliques:
        logger.info(
            f"  Meta-CSF clique hit rate: {eval_df['clique_hit'].mean():.1%}"
        )
    # How often meta-CSF beats or ties best single
    meta_wins = (eval_df["meta_augrc"] <= eval_df["best_single_augrc"]).mean()
    logger.info(f"  Meta-CSF <= best single: {meta_wins:.1%}")

    return eval_df


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
) -> None:
    """Visualise the meta-CSF: method rank lines + bin selection bands.

    X-axis: the NC metric (continuous).
    Lines: avg_ood_rank per method (scatter + loess-like smoothing).
    Vertical dashed: bin boundaries.
    Coloured bands: which method the meta-CSF selects.
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

    # Aggregate: mean rank per method per NC metric value
    # (group by rounded NC metric for cleaner lines)
    gdf["nc_rounded"] = gdf[feature].round(3)
    agg = (
        gdf[gdf["method"].isin(plot_methods)]
        .groupby(["method", "nc_rounded"])["avg_ood_rank"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Coloured background bands for meta-CSF bins
    bin_edges = [-np.inf] + splits + [np.inf]
    x_min = gdf[feature].min()
    x_max = gdf[feature].max()
    band_colors = plt.cm.Set3(np.linspace(0, 1, len(bin_methods)))

    for idx, method in enumerate(bin_methods):
        lo = max(bin_edges[idx], x_min - 0.02)
        hi = min(bin_edges[idx + 1], x_max + 0.02)
        ax.axvspan(lo, hi, alpha=0.15, color=band_colors[idx],
                   label=f"meta→{method}")

    # Vertical bin boundaries
    for s in splits:
        ax.axvline(s, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # Method lines
    cmap = plt.cm.tab10
    for i, method in enumerate(plot_methods):
        msub = agg[agg["method"] == method].sort_values("nc_rounded")
        ax.plot(
            msub["nc_rounded"], msub["avg_ood_rank"],
            marker=".", markersize=4, linewidth=1.5,
            color=cmap(i % 10), label=method, alpha=0.8,
        )

    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("avg OOD rank (lower = better)", fontsize=12)
    ax.set_title(
        f"Meta-CSF — {group_label}\n"
        f"Bands show the method selected by the meta-CSF in each {feature} segment",
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
    ax.set_title(f"Regret comparison — {group_label}", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"meta_csf_regret_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved regret comparison: meta_csf_regret_{tag}.pdf")


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

    all_evals = []
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

        # Feature selection
        if args.feature:
            best_feature = args.feature
            iv_table = None
            logger.info(f"  Using forced feature: {best_feature}")
        else:
            best_feature, iv_table = select_top_feature(
                reg_df, nc_features, group_label, top_k=args.top_k,
            )

        tag = f"{file_prefix}_group_{group_label}"

        if iv_table is not None:
            iv_table.to_csv(
                os.path.join(args.output_dir, f"meta_csf_iv_{tag}.csv"),
                index=False,
            )

        # Build meta-CSF
        splits, bin_methods, bin_summary = build_meta_csf(
            reg_df, best_feature, group_label,
            max_n_bins=args.max_bins,
        )
        bin_summary.to_csv(
            os.path.join(args.output_dir, f"meta_csf_bins_{tag}.csv"),
            index=False,
        )

        # Evaluate
        eval_df = evaluate_meta_csf(
            reg_df, best_feature, group_label,
            splits, bin_methods,
            cliques=cliques_for_group,
        )
        eval_df.to_csv(
            os.path.join(args.output_dir, f"meta_csf_eval_{tag}.csv"),
            index=False,
        )
        all_evals.append(eval_df.assign(group=group_label))

        # Plots
        plot_meta_csf(
            reg_df, best_feature, group_label,
            splits, bin_methods,
            cliques=cliques_for_group,
            output_dir=args.output_dir,
            tag=tag,
        )
        plot_regret_comparison(
            eval_df, group_label, args.output_dir, tag,
        )

    # Global summary
    if all_evals:
        combined = pd.concat(all_evals, ignore_index=True)
        summary = combined.groupby("group").agg(
            meta_mean_regret=("meta_regret", "mean"),
            meta_mean_norm_regret=("meta_norm_regret", "mean"),
            single_mean_regret=("best_single_regret", "mean"),
            single_mean_norm_regret=("best_single_norm_regret", "mean"),
            meta_wins=("meta_augrc", lambda x: (
                x <= combined.loc[x.index, "best_single_augrc"]
            ).mean()),
            clique_hit_rate=("clique_hit", "mean"),
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

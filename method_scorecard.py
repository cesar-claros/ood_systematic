"""Per-method scorecard & WoE analysis from Neural Collapse metrics.

For each OOD detection method, build an interpretable scorecard that maps
NC metric bins to the probability of the method being "top-ranked"
(i.e., belonging to the top-k by average OOD rank).

Pipeline
--------
1. Load NC metrics (features) and AUGRC scores (to derive avg_ood_rank)
2. For each method, binarize the target:  avg_ood_rank <= k  →  1 (good)
3. Optimal binning of each NC metric against the binary target
4. Extract WoE / IV per (method, feature) for interpretability
5. Build logistic-regression scorecard on WoE-transformed features
6. Summary visualisations: WoE heatmap and scorecard tables
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from optbinning import OptimalBinning, BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression
from loguru import logger

# Reuse data-loading utilities from the regression script
from method_augrc_prediction import (
    PAPYAN_NC_METRICS,
    ARCH_MAP,
    STUDY_MAP,
    SOURCES,
    load_scores,
    build_regression_dataset,
)
from nc_regime_analysis import NC_METRICS, load_nc_metrics


# ── Binarise target ──────────────────────────────────────────────────────────
def add_binary_target(
    df: pd.DataFrame,
    top_k: int = 3,
    cliques: dict[str, list[str]] | None = None,
    mode: str = "rank",
) -> pd.DataFrame:
    """Add a binary ``is_top`` column to the regression dataset.

    Parameters
    ----------
    mode : str
        "rank"   – 1 if the method's avg_ood_rank is <= top_k within its
                    (dataset, group, study, dropout, reward, run) setting.
        "clique" – 1 if the method appears in the clique for its
                    (dataset, group) combination.
    """
    df = df.copy()
    if mode == "rank":
        rank_keys = [
            "dataset", "group", "study", "dropout", "reward", "run",
        ]
        df["setting_rank"] = df.groupby(rank_keys)["avg_ood_rank"].rank(
            method="average", ascending=True,
        )
        df["is_top"] = (df["setting_rank"] <= top_k).astype(int)
        logger.info(
            f"Binary target (rank <= {top_k}): "
            f"{df['is_top'].sum()} positives / {len(df)} total "
            f"({df['is_top'].mean():.1%})"
        )
    elif mode == "clique":
        if cliques is None:
            raise ValueError("clique mode requires --clique-file")
        def _in_clique(row):
            members = cliques.get(row["dataset"], [])
            return int(row["method"] in members)
        df["is_top"] = df.apply(_in_clique, axis=1)
        logger.info(
            f"Binary target (clique membership): "
            f"{df['is_top'].sum()} positives / {len(df)} total "
            f"({df['is_top'].mean():.1%})"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return df


# ── WoE / IV analysis ────────────────────────────────────────────────────────
def compute_woe_iv(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, OptimalBinning]]]:
    """Per-method optimal binning → WoE and IV for each NC feature.

    Returns
    -------
    woe_df : DataFrame
        Columns: method, feature, bin, count, event_rate, woe, iv
    fitted_bins : dict
        {method: {feature: fitted OptimalBinning object}}
    """
    gdf = df[df["group"] == group_label]
    avail = [f for f in nc_features if f in gdf.columns]
    methods = sorted(gdf["method"].unique())

    rows = []
    fitted_bins: dict[str, dict[str, OptimalBinning]] = {}

    for method in methods:
        msub = gdf[gdf["method"] == method]
        y = msub["is_top"].values
        # Skip if constant target (all 0 or all 1)
        if y.sum() == 0 or y.sum() == len(y):
            logger.warning(
                f"  WoE {method}/{group_label}: constant target, skipping"
            )
            continue

        for feat in avail:
            x = msub[feat].values
            try:
                ob = OptimalBinning(
                    name=feat,
                    dtype="numerical",
                    solver="cp",
                    min_bin_size=0.05,
                    max_n_bins=6,
                )
                ob.fit(x, y)
                table = ob.binning_table.build()
                # Drop the Totals/Special/Missing summary rows
                table = table[
                    ~table["Bin"].isin(["Special", "Missing", "Totals"])
                ].copy()
                for _, r in table.iterrows():
                    rows.append({
                        "method": method,
                        "feature": feat,
                        "bin": r["Bin"],
                        "count": r["Count"],
                        "event_rate": r["Event rate"],
                        "woe": r["WoE"],
                        "iv": r["IV"],
                    })
                fitted_bins.setdefault(method, {})[feat] = ob
            except Exception as e:
                logger.debug(f"  Binning failed {method}/{feat}: {e}")

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        for col in ("count", "event_rate", "woe", "iv"):
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    return df_out, fitted_bins


def compute_iv_summary(woe_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IV per (method, feature) — total IV across bins."""
    if woe_df.empty:
        return pd.DataFrame(columns=["method", "feature", "total_iv"])
    iv_sum = (
        woe_df.groupby(["method", "feature"])["iv"]
        .sum()
        .reset_index()
        .rename(columns={"iv": "total_iv"})
    )
    return iv_sum


# ── Scorecard ─────────────────────────────────────────────────────────────────
def build_scorecard(
    df: pd.DataFrame,
    nc_features: list[str],
    group_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-method logistic scorecard on WoE-transformed features.

    Returns
    -------
    card_df : DataFrame
        Scorecard table: method, feature, bin, points
    coef_df : DataFrame
        Logistic regression coefficients per (method, feature)
    """
    gdf = df[df["group"] == group_label]
    avail = [f for f in nc_features if f in gdf.columns]
    methods = sorted(gdf["method"].unique())

    card_rows = []
    coef_rows = []

    for method in methods:
        msub = gdf[gdf["method"] == method].copy()
        y = msub["is_top"].values
        if y.sum() == 0 or y.sum() == len(y):
            continue

        X = msub[avail].copy()

        # BinningProcess for all features at once
        try:
            bp = BinningProcess(
                variable_names=avail,
                min_bin_size=0.05,
                max_n_bins=6,
            )
            bp.fit(X, y)
            X_woe = bp.transform(X, metric="woe")
        except Exception as e:
            logger.warning(f"  BinningProcess failed for {method}: {e}")
            continue

        # Logistic regression on WoE features
        lr = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
        )
        try:
            lr.fit(X_woe, y)
        except Exception as e:
            logger.warning(f"  LogReg failed for {method}: {e}")
            continue

        # Store coefficients
        for feat, coef in zip(avail, lr.coef_[0]):
            coef_rows.append({
                "method": method,
                "feature": feat,
                "coefficient": coef,
            })

        # Build scorecard points
        # Scorecard scaling: base_points=600, pdo=20 (standard)
        try:
            sc = Scorecard(
                binning_process=bp,
                estimator=lr,
                scaling_method="min_max",
                scaling_method_params={"min": 300, "max": 850},
            )
            sc.fit(X, y)
            sc_table = sc.table(style="detailed")
            for _, r in sc_table.iterrows():
                card_rows.append({
                    "method": method,
                    "feature": r["Variable"],
                    "bin": r["Bin"],
                    "points": r["Points"],
                })
        except Exception as e:
            logger.debug(f"  Scorecard table failed for {method}: {e}")

    card_df = pd.DataFrame(card_rows)
    coef_df = pd.DataFrame(coef_rows)
    return card_df, coef_df


# ── Visualisations ────────────────────────────────────────────────────────────
def _plot_binning_table(
    ob: OptimalBinning,
    ax: plt.Axes,
    metric: str = "woe",
    show_bin_labels: bool = True,
    compact: bool = True,
) -> None:
    """Replicate optbinning's BinningTable.plot() into a given axes.

    Draws a stacked bar chart (non-event / event counts) on ``ax`` and
    overlays the WoE (or event-rate) line on a twin y-axis.

    Parameters
    ----------
    ob : OptimalBinning
        A fitted OptimalBinning object.
    ax : matplotlib Axes
        Target axes for the bar chart (count).
    metric : str
        ``"woe"`` or ``"event_rate"``.
    show_bin_labels : bool
        Show bin interval labels on the x-axis instead of bin IDs.
    compact : bool
        If True, use smaller fonts and hide axis labels (for grids).
    """
    bt = ob.binning_table

    # Number of regular bins (excluding special + missing)
    n_records = len(bt._n_records)
    n_specials = bt._n_specials
    n_regular = n_records - 1 - n_specials  # exclude missing row

    n_event = list(bt.n_event)
    n_nonevent = list(bt.n_nonevent)

    # Remove special bins
    for _ in range(n_specials):
        n_event.pop(-2)
        n_nonevent.pop(-2)
    # Remove missing bin
    n_event.pop(-1)
    n_nonevent.pop(-1)

    n_bins = len(n_event)

    if metric == "woe":
        metric_values = bt._woe
        metric_label = "WoE"
    else:
        metric_values = bt._event_rate
        metric_label = "Event rate"

    # Stacked bars: non-event + event
    x = np.arange(n_bins)
    p2 = ax.bar(x, n_event, color="tab:red", alpha=0.7)
    p1 = ax.bar(x, n_nonevent, color="tab:blue", bottom=n_event, alpha=0.7)

    # Twin axis for metric line
    ax2 = ax.twinx()
    ax2.plot(
        np.arange(n_regular), metric_values[:n_regular],
        linestyle="solid", marker="o", color="black",
        markersize=3, linewidth=1.2,
    )

    # Tick / label formatting
    fs = 7 if compact else 10
    ax.set_xticks(x)

    if show_bin_labels and hasattr(bt, "_bin_str"):
        bin_str = bt._bin_str
        # Strip special + missing
        bin_str = bin_str[:n_regular]
        # Truncate long labels
        bin_str = [s[:20] for s in bin_str]
        ax.set_xticklabels(bin_str, fontsize=fs - 1, rotation=45, ha="right")
    else:
        ax.set_xticklabels(x, fontsize=fs)

    ax.tick_params(axis="y", labelsize=fs)
    ax2.tick_params(axis="y", labelsize=fs)

    if not compact:
        ax.set_xlabel("Bin", fontsize=fs + 1)
        ax.set_ylabel("Count", fontsize=fs + 1)
        ax2.set_ylabel(metric_label, fontsize=fs + 1)


def plot_woe_grid(
    fitted_bins: dict[str, dict[str, OptimalBinning]],
    group_label: str,
    output_dir: str,
    tag: str,
) -> None:
    """Grid of WoE-per-bin plots (optbinning style).

    Rows = OOD methods, Columns = NC features.
    Each cell: stacked event/non-event bars + WoE line on twin axis.
    """
    if not fitted_bins:
        logger.warning("  No fitted binnings, skipping WoE grid")
        return

    methods = sorted(fitted_bins.keys())
    features = sorted(
        {f for feats in fitted_bins.values() for f in feats}
    )
    n_methods = len(methods)
    n_features = len(features)

    fig, axes = plt.subplots(
        n_methods, n_features,
        figsize=(3.5 * n_features, 2.5 * n_methods),
        squeeze=False,
    )

    for i, method in enumerate(methods):
        for j, feat in enumerate(features):
            ax = axes[i][j]
            ob = fitted_bins.get(method, {}).get(feat)
            if ob is None:
                ax.set_visible(False)
                continue

            _plot_binning_table(ob, ax, metric="woe", compact=True)

            if j == 0:
                ax.set_ylabel(method, fontsize=9, fontweight="bold")
            if i == 0:
                ax.set_title(feat, fontsize=9, fontweight="bold")

    fig.suptitle(
        f"WoE across bins — {group_label}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        os.path.join(output_dir, f"woe_grid_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved WoE grid: woe_grid_{tag}.pdf")


def plot_iv_summary(
    iv_df: pd.DataFrame,
    group_label: str,
    output_dir: str,
    tag: str,
) -> None:
    """Bar chart of mean IV per feature (averaged across methods)."""
    if iv_df.empty:
        return

    mean_iv = (
        iv_df.groupby("feature")["total_iv"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(mean_iv) + 1.5))
    mean_iv.plot.barh(ax=ax, color="steelblue", edgecolor="gray")
    ax.set_xlabel("Mean IV (across methods)", fontsize=10)
    ax.set_ylabel("")
    ax.set_title(
        f"Feature predictive power (IV) — {group_label}", fontsize=12,
    )
    # IV interpretation thresholds
    for thresh, label in [(0.02, "weak"), (0.1, "medium"), (0.3, "strong")]:
        if mean_iv.max() > thresh:
            ax.axvline(thresh, color="gray", linestyle="--", linewidth=0.8)
            ax.text(
                thresh, len(mean_iv) - 0.5, f" {label}",
                fontsize=7, color="gray",
            )
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"iv_summary_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved IV summary: iv_summary_{tag}.pdf")


def plot_scorecard_heatmap(
    card_df: pd.DataFrame,
    group_label: str,
    output_dir: str,
    tag: str,
) -> None:
    """Heatmap of mean scorecard points per (method, feature)."""
    if card_df.empty:
        return

    mean_pts = (
        card_df.groupby(["method", "feature"])["points"]
        .mean()
        .reset_index()
        .pivot(index="feature", columns="method", values="points")
    )

    fig, ax = plt.subplots(
        figsize=(0.8 * len(mean_pts.columns) + 2, 0.5 * len(mean_pts) + 2)
    )
    sns.heatmap(
        mean_pts,
        ax=ax,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "mean points"},
    )
    ax.set_title(f"Scorecard points — {group_label}", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"scorecard_heatmap_{tag}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info(f"  Saved scorecard heatmap: scorecard_heatmap_{tag}.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-method scorecard & WoE from NC metrics",
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
        help="JSON clique file (from stats_eval.py) for clique-based target",
    )
    parser.add_argument(
        "--filter-methods", action="store_true",
        help="Exclude methods containing 'global'/'class' (except PCA/KPCA)",
    )
    parser.add_argument(
        "--papyan-only", action="store_true",
        help="Use only Papyan NC metrics (8 features)",
    )
    parser.add_argument(
        "--groups", type=str, nargs="*", default=None,
        help="OOD groups to analyse (default: near mid far all)",
    )
    parser.add_argument(
        "--target-mode", type=str, default="rank",
        choices=["rank", "clique"],
        help="How to binarize: 'rank' (top-k by avg_ood_rank) "
             "or 'clique' (membership in clique file)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Rank threshold for 'rank' target mode (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="scorecard_outputs",
    )
    parser.add_argument(
        "--mcd", type=str, default="False",
    )
    parser.add_argument(
        "--score-metric", type=str, default="AUGRC",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── NC metrics ──
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

    # ── Scores ──
    scores = load_scores(
        args.scores_dir, args.backbone, args.mcd,
        args.clip_dir, SOURCES,
        filter_methods=args.filter_methods,
        study_filter=args.study,
    )

    # ── Build dataset ──
    reg_df = build_regression_dataset(nc, scores, nc_features)

    # ── Cliques ──
    clique_data = None
    if args.clique_file and os.path.exists(args.clique_file):
        with open(args.clique_file) as f:
            raw = json.load(f)
        raw.pop("_ranks", None)
        clique_data = raw

    # ── Binarise ──
    groups_to_run = args.groups or ["near", "mid", "far", "all"]

    study_tag = f"_{args.study}" if args.study else ""
    file_prefix = (
        f"{args.score_metric}_{args.backbone}_MCD-{args.mcd}{study_tag}"
    )

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

        gdf = reg_df[reg_df["group"] == group_label].copy()
        if gdf.empty:
            logger.warning(f"  No data for group {group_label}")
            continue

        gdf = add_binary_target(
            gdf,
            top_k=args.top_k,
            cliques=cliques_for_group,
            mode=args.target_mode,
        )

        tag = f"{file_prefix}_group_{group_label}"

        # ── WoE / IV ──
        logger.info("  Computing WoE / IV ...")
        woe_df, fitted_bins = compute_woe_iv(gdf, nc_features, group_label)
        iv_df = compute_iv_summary(woe_df)

        woe_df.to_csv(
            os.path.join(args.output_dir, f"woe_{tag}.csv"),
            index=False,
        )
        iv_df.to_csv(
            os.path.join(args.output_dir, f"iv_{tag}.csv"),
            index=False,
        )

        plot_woe_grid(fitted_bins, group_label, args.output_dir, tag)
        plot_iv_summary(iv_df, group_label, args.output_dir, tag)

        # ── Scorecard ──
        logger.info("  Building scorecards ...")
        card_df, coef_df = build_scorecard(gdf, nc_features, group_label)

        if not card_df.empty:
            card_df.to_csv(
                os.path.join(args.output_dir, f"scorecard_{tag}.csv"),
                index=False,
            )
            coef_df.to_csv(
                os.path.join(args.output_dir, f"scorecard_coefs_{tag}.csv"),
                index=False,
            )
            plot_scorecard_heatmap(card_df, group_label, args.output_dir, tag)

        # ── Log top features by IV ──
        if not iv_df.empty:
            global_iv = (
                iv_df.groupby("feature")["total_iv"]
                .mean()
                .sort_values(ascending=False)
            )
            logger.info(
                f"  Top features by IV: "
                + ", ".join(
                    f"{f}={v:.4f}" for f, v in global_iv.head(5).items()
                )
            )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()

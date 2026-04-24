import os, re, math, json, itertools, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
from src.utils_stats import *
from loguru import logger
import argparse

# Set Default Environment Variables if not present
os.environ.setdefault("EXPERIMENT_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/experiments')
os.environ.setdefault("DATASET_ROOT_DIR", '/work/cniel/sw/FD_Shifts/project/datasets')

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 200)

DISPLAY_RENAME = {
    "KPCA RecError global": "KPCA RecError",
    "PCA RecError global": "PCA RecError",
    "MCD-KPCA RecError global": "MCD-KPCA RecError",
    "MCD-PCA RecError global": "MCD-PCA RecError",
}

ALL_PARADIGMS_CONFIGS = [
    ("Conv", ["confidnet"], "ConfidNet (Vanilla CE)"),
    ("Conv", ["devries"],   "DeVries"),
    ("Conv", ["dg"],        "Deep Gamblers"),
    ("ViT",  ["modelvit"],  "ViT (finetune)"),
]

# Per-backbone aggregation: one panel pools the three CNN paradigms,
# the other shows ViT. Friedman blocks include 'model' so pooling across
# paradigms keeps each (paradigm, dataset, run) as its own block.
ALL_BACKBONES_CONFIGS = [
    ("Conv", ["confidnet", "devries", "dg"], "VGG-13 (all paradigms)"),
    ("ViT",  ["modelvit"],                    "ViT (finetune)"),
]

# Raw CSF-name classification by which side of the network the score reads
# from. Kept consistent with scripts/paradigm_nc_clique_distance.py.
HEAD_SIDE_CSFS = {
    "REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
    "GradNorm", "Energy", "Confidence", "pNML",
}
FEATURE_SIDE_CSFS = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}
# Background tint per side for the all-paradigms grid.
SIDE_BG_COLOR = {"head": "#a6cde4", "feature": "#f3e2cc"}
SIDE_BG_ALPHA = 0.35


def _compute_members_for_config(
    backbone: str,
    model_filter: list[str] | None,
    metric: list[str],
    mcd_flag: str,
    filter_methods: bool,
    clip_dir: str,
    alpha: float,
    sources_all: list[str],
) -> dict:
    """Run the per-(backbone, model) Friedman/Conover/clique pipeline.

    Returns a dict with:
      - members_all: DataFrame (index='{src}->{regime}', bool columns per CSF
        that appears in at least one top clique within this configuration)
      - full_csfs: list of all CSFs seen after filtering (union across sources)
      - sources, active_groups: bookkeeping for downstream plotting
    """
    clip_names = {'Unnamed: 0_level_1': 'dataset',
                  'Unnamed: 5_level_1': 'group'}
    distance_dict = {'0': 'test', '1': 'near', '2': 'mid', '3': 'far'}

    sources = [s for s in sources_all
               if os.path.exists(os.path.join(clip_dir, f"clip_distances_{s}.csv"))]
    logger.info(f"[{backbone}/{model_filter}] active sources: {sources}")

    df_all = []
    for SOURCE in sources:
        CONFIG = {
            "FPR@95TPR":    f"scores_risk/scores_all_FPR@95TPR_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "AUROC_f":      f"scores_risk/scores_all_AUROC_f_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "AUGRC":        f"scores_risk/scores_all_AUGRC_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "AURC":         f"scores_risk/scores_all_AURC_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "ECE_L1":       f"scores_calibration/scores_all_ECE_L1_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "ECE_L2":       f"scores_calibration/scores_all_ECE_L2_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "MCE":          f"scores_calibration/scores_all_MCE_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "ECE_L1_BOUND": f"scores_calibration/scores_all_ECE_L1_BOUND_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "ECE_L2_BOUND": f"scores_calibration/scores_all_ECE_L2_BOUND_MCD-{mcd_flag}_{backbone}_{SOURCE}.csv",
            "CLIP_FILE":    f"{clip_dir}/clip_distances_{SOURCE}.csv",
            "OUTDIR":       ".",
            "ALPHA":        alpha,
            "N_BOOT":       2000,
        }
        try:
            df = load_all_scores(CONFIG)
            exclude_methods = {"CTMmean", "CTMmeanOC", "MCD-CTMmean", "MCD-CTMmeanOC"}
            df = df[~df['methods'].isin(exclude_methods)]
            if filter_methods:
                keep_exceptions = {
                    "KPCA RecError global", "PCA RecError global",
                    "MCD-KPCA RecError global", "MCD-PCA RecError global",
                }
                mask = df['methods'].str.contains('global|class', case=False, na=False)
                mask &= ~df['methods'].isin(keep_exceptions)
                df = df[~mask]
            if model_filter is not None:
                df = df[df['model'].isin(model_filter)]
            df['source'] = SOURCE

            if os.path.exists(CONFIG["CLIP_FILE"]):
                clip = pd.read_csv(CONFIG["CLIP_FILE"], header=[0, 1])
                clip.columns = clip.columns.droplevel(0)
                clip = clip.rename(clip_names, axis='columns')
                if "group" in clip.columns:
                    merged = df.merge(clip[["dataset", "group"]], on="dataset", how="left")
                    merged["group"] = merged["group"].apply(
                        lambda x: str(int(x)) if pd.notna(x) else x)
                else:
                    merged = df
            else:
                merged = df
            df_all.append(choose_baseline_rows(merged))
        except Exception as e:
            logger.error(f"Error processing {SOURCE} for {backbone}/{model_filter}: {e}")
            continue

    if not df_all:
        raise RuntimeError(f"No data loaded for backbone={backbone}, model={model_filter}")

    rank_group = ["dataset", 'model', "metric", 'group', 'run']
    blocks = ['dataset', 'model', 'metric', 'group', 'run']
    members_list: list[pd.DataFrame] = []
    df_combined = pd.concat(df_all, axis=0)

    full_csfs_set: set[str] = set()

    for source_ in sources:
        df_ = df_combined[df_combined['source'] == source_].copy()
        if backbone == 'ViT':
            df_ = df_[df_['methods'] != 'Confidence']
        if df_.empty:
            continue
        df_met = df_[(df_['metric'] == metric[0]) | (df_['metric'] == metric[1])].copy()
        if df_met.empty or 'group' not in df_met.columns:
            continue
        df_met["rank"] = df_met.groupby(rank_group)["score_std"].rank(
            ascending=False, method="average", pct=True)
        full_csfs_set.update(df_met['methods'].dropna().unique().tolist())

        layered_cliques: dict[str, list] = {}
        for dataset_group, g in df_met.groupby('group'):
            sub = g.copy()
            sub["block"] = sub[blocks].astype(str).agg("|".join, axis=1)
            try:
                stat, p, pivot = friedman_blocked(
                    sub, entity_col="methods", block_col="block", value_col="score_std")
                if isinstance(stat, float) and not math.isnan(stat):
                    ph = conover_posthoc_from_pivot(pivot)
                    ranks_ = pivot.rank(axis=1, ascending=False)
                    avg_ranks_ = ranks_.mean(axis=0).sort_values()
                    cliques = maximal_cliques_from_pmatrix(ph, alpha)
                    scored = rank_cliques(cliques, list(avg_ranks_.index), avg_ranks_)
                    layered_cliques[str(dataset_group)] = greedy_exclusive_layers(scored)
            except Exception as e:
                logger.error(f"Friedman/Posthoc failed ({source_}, {dataset_group}): {e}")

        datasets_order = [g for g in ['0', '1', '2', '3'] if g in layered_cliques]
        if not datasets_order:
            continue
        clique_members = [layered_cliques[c][0]['members'] for c in datasets_order]
        member_df = pd.DataFrame([{name: True for name in names} for names in clique_members])
        member_df = member_df.where(member_df == True, False)
        member_df.index = [f"{source_}->{distance_dict.get(d, d)}" for d in datasets_order]
        members_list.append(member_df)

    if not members_list:
        raise RuntimeError(f"No top cliques produced for backbone={backbone}, model={model_filter}")

    group_order = ['test', 'near', 'mid', 'far']
    present_labels: set[str] = set()
    for mdf in members_list:
        for idx in mdf.index:
            present_labels.add(idx.split('->')[-1])
    active_groups = [g for g in group_order if g in present_labels]

    reorder_index = [f'{src}->{gl}' for gl in active_groups for src in sources]
    members_all = pd.concat(members_list, axis=0)
    valid_reorder = [idx for idx in reorder_index if idx in members_all.index]
    members_all = members_all.loc[valid_reorder]
    members_all = members_all.where(members_all == True, False)
    members_all = members_all[sorted(members_all.columns, key=str.casefold)]

    return {
        "members_all": members_all,
        "full_csfs": sorted(full_csfs_set, key=str.casefold),
        "sources": sources,
        "active_groups": active_groups,
    }


def _run_all_paradigms(args, metric: list[str]) -> None:
    """1x4 combined top-cliques grid across training paradigms."""
    _render_panels_grid(
        args=args, metric=metric,
        configs=ALL_PARADIGMS_CONFIGS,
        out_stem="top_cliques_all_paradigms",
        suptitle_prefix="Top cliques across training paradigms",
    )


def _run_all_backbones(args, metric: list[str]) -> None:
    """1x2 combined top-cliques grid: pooled Conv paradigms vs ViT."""
    _render_panels_grid(
        args=args, metric=metric,
        configs=ALL_BACKBONES_CONFIGS,
        out_stem="top_cliques_all_backbones",
        suptitle_prefix="Top cliques by backbone",
    )


def _render_panels_grid(args, metric: list[str],
                        configs: list[tuple],
                        out_stem: str,
                        suptitle_prefix: str) -> None:
    """Shared panel-grid renderer with head/feature row split.

    For each (backbone, model_filter, label) entry in ``configs`` we build a
    panel via ``_compute_members_for_config``. Rows use the same
    regime-center-of-mass ordering as the all-paradigms view, with head-side
    CSFs on top and feature-side on bottom; the three most-frequent CSFs per
    panel get a dotted rectangle marker.
    """
    sources_all = ['cifar10', 'supercifar100', 'cifar100', 'tinyimagenet']
    panels = []
    for backbone, model_filter, label in configs:
        logger.info(f"=== {label} (backbone={backbone}, model={model_filter}) ===")
        try:
            res = _compute_members_for_config(
                backbone=backbone,
                model_filter=model_filter,
                metric=metric,
                mcd_flag=str(args.mcd),
                filter_methods=True,
                clip_dir=args.clip_dir,
                alpha=args.alpha,
                sources_all=sources_all,
            )
            panels.append((label, backbone, res))
        except Exception as e:
            logger.error(f"Panel {label} failed: {e}")
            continue

    if not panels:
        logger.error("No panels produced data; aborting plot.")
        return

    if getattr(args, "exclude_test", False):
        for _, _, r in panels:
            mdf = r["members_all"]
            keep = [idx for idx in mdf.index if idx.split("->")[-1] != "test"]
            r["members_all"] = mdf.loc[keep]
            r["active_groups"] = [g for g in r["active_groups"] if g != "test"]

    # Union of all filtered CSFs across panels (the 20-ish shared y-axis).
    all_csfs = sorted(
        {c for _, _, r in panels for c in r["full_csfs"]},
        key=str.casefold,
    )
    # Reindex each panel to the shared raw-name column set *before* renaming,
    # so we can compute the row order on raw names.
    for _, _, r in panels:
        r["members_all"] = r["members_all"].reindex(columns=all_csfs, fill_value=False)

    # Classify each CSF as head-side / feature-side / other (on raw names).
    side_of = {}
    for c in all_csfs:
        if c in HEAD_SIDE_CSFS:
            side_of[c] = "head"
        elif c in FEATURE_SIDE_CSFS:
            side_of[c] = "feature"
        else:
            side_of[c] = "other"
    unclassified = [c for c in all_csfs if side_of[c] == "other"]
    if unclassified:
        logger.warning(f"CSFs not in head/feature sets (placed below feature block): "
                       f"{unclassified}")

    if args.row_order == "regime-cm":
        regime_weights = {"near": 1.0, "mid": 2.0, "far": 3.0}
        pooled = pd.DataFrame(0.0, index=all_csfs, columns=["near", "mid", "far"])
        for _, _, r in panels:
            m = r["members_all"]
            for idx in m.index:
                regime = idx.split("->")[-1]
                if regime not in regime_weights:
                    continue
                pooled[regime] += m.loc[idx].astype(float)
        totals = pooled.sum(axis=1)
        weighted = (pooled["near"] * regime_weights["near"]
                    + pooled["mid"] * regime_weights["mid"]
                    + pooled["far"] * regime_weights["far"])
        cm = weighted / totals.replace(0, np.nan)

        # Within each side, sort descending in cm so visual bottom = far-dominant
        # and visual top = near-dominant. NaN (empty) -> +inf, placed at the
        # visual bottom of their own side block.
        def _sort_group(names):
            key = cm.reindex(names).fillna(np.inf)
            df = pd.DataFrame({"cm": key.values, "name": list(names)})
            df = df.sort_values(by=["cm", "name"],
                                ascending=[False, True], kind="mergesort")
            return df["name"].tolist()

        # Visual layout (y=0 is visual bottom in this plot):
        #   bottom -> other (if any), feature-side, head-side <- top
        other_sorted   = _sort_group([c for c in all_csfs if side_of[c] == "other"])
        feature_sorted = _sort_group([c for c in all_csfs if side_of[c] == "feature"])
        head_sorted    = _sort_group([c for c in all_csfs if side_of[c] == "head"])
        ordered_csfs = other_sorted + feature_sorted + head_sorted
    else:
        # Keep the head-on-top / feature-on-bottom grouping even in alpha mode.
        other_sorted   = sorted([c for c in all_csfs if side_of[c] == "other"],
                                key=str.casefold)
        feature_sorted = sorted([c for c in all_csfs if side_of[c] == "feature"],
                                key=str.casefold)
        head_sorted    = sorted([c for c in all_csfs if side_of[c] == "head"],
                                key=str.casefold)
        ordered_csfs = other_sorted + feature_sorted + head_sorted

    display_csfs = [DISPLAY_RENAME.get(c, c) for c in ordered_csfs]
    for _, _, r in panels:
        m = r["members_all"].reindex(columns=ordered_csfs, fill_value=False)
        r["members_all"] = m.rename(columns=DISPLAY_RENAME)

    group_colors = {'test': 'tab:green', 'near': 'tab:blue', 'mid': 'tab:red', 'far': 'tab:orange'}

    n = len(panels)
    fig, axes = plt.subplots(
        1, n,
        figsize=(max(3.2 * n, 10), max(0.32 * len(display_csfs) + 2.5, 6)),
        sharey=True,
    )
    if n == 1:
        axes = [axes]

    # Side-block y-ranges in ordered_csfs (y=0 is visual bottom).
    def _side_range(side_name: str) -> tuple[float, float] | None:
        ys = [i for i, c in enumerate(ordered_csfs) if side_of[c] == side_name]
        if not ys:
            return None
        return min(ys) - 0.5, max(ys) + 0.5

    feature_range = _side_range("feature")
    head_range = _side_range("head")

    for ax, (label, backbone, r) in zip(axes, panels):
        # Side-of-network background bands (behind everything).
        if feature_range is not None:
            ax.axhspan(feature_range[0], feature_range[1],
                       facecolor=SIDE_BG_COLOR["feature"],
                       alpha=SIDE_BG_ALPHA, zorder=0, edgecolor='none')
        if head_range is not None:
            ax.axhspan(head_range[0], head_range[1],
                       facecolor=SIDE_BG_COLOR["head"],
                       alpha=SIDE_BG_ALPHA, zorder=0, edgecolor='none')

        mdf = r["members_all"]
        c_list = [group_colors.get(idx.split('->')[-1], 'black') for idx in mdf.index]
        plot_grid(mdf, color_dotline=c_list, ax=ax, zorder=10)

        counts = mdf.sum(axis=0)
        top3 = list(counts.sort_values(ascending=False).head(3).index)
        display_cols = list(mdf.columns)
        tf = blended_transform_factory(ax.transAxes, ax.transData)
        for csf_name in top3:
            if counts[csf_name] <= 0:
                continue
            y = display_cols.index(csf_name)
            ax.add_patch(mpatches.Rectangle(
                (0.0, y - 0.45), 1.0, 0.9,
                transform=tf, fill=False,
                edgecolor='black', linestyle=':', linewidth=1.4,
                zorder=20, clip_on=False,
            ))

        ax.set_title(label, fontsize=11)

    handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None',
                             markersize=8, label=g)
               for g, c in group_colors.items()
               if any(g in r["active_groups"] for _, _, r in panels)]
    side_handles = []
    if head_range is not None:
        side_handles.append(mpatches.Patch(
            facecolor=SIDE_BG_COLOR["head"], alpha=SIDE_BG_ALPHA,
            edgecolor='none', label='head-side'))
    if feature_range is not None:
        side_handles.append(mpatches.Patch(
            facecolor=SIDE_BG_COLOR["feature"], alpha=SIDE_BG_ALPHA,
            edgecolor='none', label='feature-side'))
    top3_handle = mpatches.Patch(
        facecolor='none', edgecolor='black', linestyle=':', linewidth=1.4,
        label='top-3 by clique count')
    legend_handles = handles + side_handles + [top3_handle]
    if legend_handles:
        fig.legend(handles=legend_handles,
                   loc='lower center', ncol=len(legend_handles),
                   bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)

    grouping_label = (f', Grouping: {os.path.basename(args.clip_dir)}'
                      if args.clip_dir != 'clip_scores' else '')
    fig.suptitle(f'{suptitle_prefix} '
                 f'(Metrics={metric}{grouping_label})',
                 fontsize=12, y=1.00)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))

    clip_suffix = (f'_{os.path.basename(args.clip_dir)}'
                   if args.clip_dir != 'clip_scores' else '')
    test_suffix = '_notest' if getattr(args, "exclude_test", False) else ''
    out_filename = f'{out_stem}_{str(args.mcd)}_{args.metric_group}{test_suffix}{clip_suffix}'
    out_path = os.path.join(args.output_dir, out_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    fig.savefig(out_path + '.jpeg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.success(f"Saved combined plot to: {out_path}.pdf")


def main():
    parser = argparse.ArgumentParser(description="OOD Systematic Eval Analysis")
    parser.add_argument("--mcd", action="store_true", help="Set MCD flag (default: False)")
    parser.add_argument("--backbone", type=str, required=False, default=None, choices=['Conv', 'ViT'],
                        help="Backbone type (required unless --all-paradigms is set)")
    parser.add_argument("--all-paradigms", action="store_true",
                        help="Produce a 1x4 combined plot across ConfidNet, DeVries, DG, ViT, "
                             "showing all filtered CSFs on a shared y-axis. Forces --filter-methods; "
                             "--backbone and --model are ignored in this mode.")
    parser.add_argument("--all-backbones", action="store_true",
                        help="Produce a 1x2 combined plot: pooled VGG-13 (all CNN paradigms) vs ViT, "
                             "with the same head/feature row split and regime-CM ordering. "
                             "--backbone and --model are ignored in this mode.")
    parser.add_argument("--row-order", type=str, default="regime-cm",
                        choices=["alpha", "regime-cm"],
                        help="CSF row ordering for --all-paradigms. 'alpha' = case-fold alphabetical. "
                             "'regime-cm' = pooled regime center-of-mass across panels "
                             "(near=1, mid=2, far=3): near-dominant CSFs at top, far-dominant at bottom, "
                             "CSFs never in any clique at the very bottom. Default: regime-cm.")
    parser.add_argument("--metric-group", type=str, required=True, choices=['RC', 'ROC', 'CE', 'CE_BOUND'], help="Metric group: RC=['AUGRC', 'AURC'], ROC=['AUROC_f', 'FPR@95TPR'], CE=['ECE_L1','ECE_L2'], CE_BOUND=['ECE_L1_BOUND','ECE_L2_BOUND']")
    parser.add_argument("--output-dir", type=str, default="ood_eval_outputs", help="Output directory")
    parser.add_argument("--filter-methods", action="store_true", help="Exclude methods containing 'global' or 'class' (except PCA/KPCA RecError global variants)")
    parser.add_argument("--model", type=str, nargs='+', default=None,
                        help="Filter by model(s). Conv options: confidnet, devries, dg. ViT options: modelvit. Default: all models.")
    parser.add_argument("--clip-dir", type=str, default="clip_scores",
                        help="Directory containing clip_distances_{source}.csv files (default: clip_scores)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level for Conover post-hoc tests (default: 0.05)")
    parser.add_argument("--exclude-test", action="store_true",
                        help="Drop the 'test' (ID / misclassification) regime from "
                             "all top-clique panels. Only affects --all-paradigms / "
                             "--all-backbones modes.")

    args = parser.parse_args()

    if not args.all_paradigms and not args.all_backbones and args.backbone is None:
        parser.error("--backbone is required unless --all-paradigms or --all-backbones is set.")

    # Validate --model choices against --backbone (only for single-config mode)
    if not args.all_paradigms and not args.all_backbones and args.model is not None:
        valid_conv = {'confidnet', 'devries', 'dg'}
        valid_vit = {'modelvit'}
        valid = valid_vit if args.backbone == 'ViT' else valid_conv
        invalid = set(args.model) - valid
        if invalid:
            parser.error(f"Invalid model(s) {invalid} for backbone {args.backbone}. Valid options: {valid}")

    MCD_flag = str(args.mcd) # 'True' or 'False' as string for file paths if that's the convention
    BACKBONE = args.backbone
    OUTDIR = args.output_dir
    
    if args.metric_group == 'RC':
        metric = ['AUGRC','AURC']
    elif args.metric_group == 'ROC':
        metric = ['AUROC_f','FPR@95TPR']
    elif args.metric_group == 'CE_BOUND':
        metric = ['ECE_L1_BOUND','ECE_L2_BOUND']
    elif args.metric_group == 'CE':
        metric = ['ECE_L1','ECE_L2']
    else:
        raise ValueError(f"Unknown metric group: {args.metric_group}")

    if args.all_paradigms:
        logger.info(f"[all-paradigms] MCD={MCD_flag}, MetricGroup={args.metric_group} ({metric}), "
                    f"alpha={args.alpha}, CLIP dir={args.clip_dir}")
        os.makedirs(OUTDIR, exist_ok=True)
        _run_all_paradigms(args, metric)
        return

    if args.all_backbones:
        logger.info(f"[all-backbones] MCD={MCD_flag}, MetricGroup={args.metric_group} ({metric}), "
                    f"alpha={args.alpha}, CLIP dir={args.clip_dir}")
        os.makedirs(OUTDIR, exist_ok=True)
        _run_all_backbones(args, metric)
        return

    logger.info(f"Starting stats eval with: MCD={MCD_flag}, Backbone={BACKBONE}, MetricGroup={args.metric_group} ({metric}), alpha={args.alpha}, CLIP dir={args.clip_dir}")
    logger.info(f"Output directory: {OUTDIR}")

    os.makedirs(OUTDIR, exist_ok=True)

    clip_names = {'Unnamed: 0_level_1':'dataset',
                    'Unnamed: 5_level_1':'group'}
    distance_dict_full = {'0':'test','1':'near','2':'mid','3':'far'}
    distance_dict = distance_dict_full  # may be narrowed after data loading

    df_all = []
    
    # =========================
    # CONFIG: edit as needed
    # =========================
    # Derive sources from available CLIP CSV files in the clip directory
    _all_sources = ['cifar10', 'supercifar100', 'cifar100', 'tinyimagenet']
    sources = [s for s in _all_sources if os.path.exists(os.path.join(args.clip_dir, f"clip_distances_{s}.csv"))]
    logger.info(f"Active sources (from {args.clip_dir}): {sources}")
    for SOURCE in sources:
        logger.info(f"Processing source: {SOURCE}")
        CONFIG = {
            # Map metric -> CSV path. You can point these to your real files per metric
            # NOTE: Assuming file naming convention matches what was in script. 
            # If MCD flag in filename uses 'True'/'False' string, this works.
            "FPR@95TPR":       f"scores_risk/scores_all_FPR@95TPR_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AUROC_f":         f"scores_risk/scores_all_AUROC_f_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AUGRC":           f"scores_risk/scores_all_AUGRC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "AURC":            f"scores_risk/scores_all_AURC_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L1":          f"scores_calibration/scores_all_ECE_L1_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L2":          f"scores_calibration/scores_all_ECE_L2_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "MCE":             f"scores_calibration/scores_all_MCE_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L1_BOUND":    f"scores_calibration/scores_all_ECE_L1_BOUND_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            "ECE_L2_BOUND":    f"scores_calibration/scores_all_ECE_L2_BOUND_MCD-{MCD_flag}_{BACKBONE}_{SOURCE}.csv",
            # Optional CLIP distances / groupings file (columns: dataset, features..., e.g., 'group', 'clip_dist_id_ood', etc.)
            "CLIP_FILE": f"{args.clip_dir}/clip_distances_{SOURCE}.csv",  # set to None if not available
            # Output dir
            "OUTDIR": OUTDIR,
            # Alpha for significance
            "ALPHA": args.alpha,
            # Bootstraps for CIs
            "N_BOOT": 2000
        }

        alpha = CONFIG["ALPHA"]
        try:
            df = load_all_scores(CONFIG)

            exclude_methods = {"CTMmean", "CTMmeanOC", "MCD-CTMmean", "MCD-CTMmeanOC"}
            n_excluded = df['methods'].isin(exclude_methods).sum()
            df = df[~df['methods'].isin(exclude_methods)]
            logger.info(f"Excluded {n_excluded} rows with CTMmean/CTMmeanOC methods for {SOURCE}")

            if args.filter_methods:
                keep_exceptions = {
                    "KPCA RecError global",
                    "PCA RecError global",
                    "MCD-KPCA RecError global",
                    "MCD-PCA RecError global",
                }
                mask = df['methods'].str.contains('global|class', case=False, na=False)
                mask &= ~df['methods'].isin(keep_exceptions)
                n_removed = mask.sum()
                df = df[~mask]
                logger.info(f"Filtered {n_removed} rows with 'global'/'class' methods for {SOURCE}")

            if args.model is not None:
                df = df[df['model'].isin(args.model)]
                logger.info(f"Filtered to models {args.model} for {SOURCE} ({len(df)} rows remaining)")

            df['source'] = SOURCE

            if os.path.exists(CONFIG["CLIP_FILE"]):
                clip = pd.read_csv(CONFIG["CLIP_FILE"],header=[0,1])
                clip.columns = clip.columns.droplevel(0)
                clip = clip.rename(clip_names,axis='columns')
                # Expect columns: dataset, group (optional), and numeric features like clip_dist_to_ID, etc.
                # Keep only columns present
                numeric_features = [c for c in clip.columns if c not in ["dataset", "group"]]
                if len(numeric_features) == 0:
                   logger.warning("CLIP file found but no numeric features detected; skipping LODO.")
                   merged = df # fallback
                else:
                    # Stability: variance within vs between clusters if 'group' present
                    if "group" in clip.columns:
                        merged = df.merge(clip[["dataset", "group"]], on="dataset", how="left")
                        # Convert group to string to avoid float keys from NaN promotion
                        merged["group"] = merged["group"].apply(lambda x: str(int(x)) if pd.notna(x) else x)
                    else:
                        merged = df
            else:
                 logger.warning(f"CLIP file not found: {CONFIG['CLIP_FILE']}")
                 merged = df

            df_all.append(choose_baseline_rows(merged))
        except Exception as e:
            logger.error(f"Error processing {SOURCE}: {e}")
            continue

    if not df_all:
        logger.error("No data loaded. Exiting.")
        return

    rank_group = ["dataset", 'model', "metric", 'group', 'run' ]
    blocks = ['dataset', 'model', 'metric', 'group', 'run']

    members_list = []
    # Collect mean ranks per (source, group) for JSON export
    all_avg_ranks: dict[str, dict[str, pd.Series]] = {}

    # Combined Dataframe
    df_combined = pd.concat(df_all,axis=0)

    for source_ in sources:
        logger.info(f"Computing ranks for source: {source_}")
        
        # Filter for backbone specifics if needed
        # Assuming df_all logic was correct, re-applying checks
        
        df_ = df_combined[df_combined['source']==source_].copy()
        
        if BACKBONE=='ViT':
             df_ = df_[df_['methods']!='Confidence']
             
        if df_.empty:
            logger.warning(f"No data for source {source_} after filtering.")
            continue

        df_["rank"] = df_.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
        # Average rank per method
        avg_rank = df_.groupby(["methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
        # Average rank per method per metric
        df_met = df_[(df_['metric']==metric[0])|(df_['metric']==metric[1])].copy()
        
        if df_met.empty:
             logger.warning(f"No data for metrics {metric} in source {source_}.")
             continue
             
        df_met["rank"] = df_met.groupby(rank_group)["score_std"].rank(ascending=False, method="average", pct=True)
        # avg_rank_group_metric = df_met.groupby(['group','metric',"methods"])["rank"].mean().sort_values().rename("avg_rank").reset_index()
        # parsed = avg_rank_group_metric["methods"].apply(parse_method_variation)
        # avg_rank_group_metric["method_base"] = parsed.apply(lambda x: x[0])
        # avg_rank_group_metric["variation"] = parsed.apply(lambda x: x[1])
        
        friedman_results = []
        layered_cliques = {}
        layered_ranks = []
        
        # Group by 'group' column (merged from CLIP)
        if 'group' not in df_met.columns:
             logger.warning(f"Group column missing in data for {source_}. Skipping friedman analysis.")
             continue

        for dataset_group, g in df_met.groupby('group'):
            sub = g.copy()
            sub["block"] = sub[blocks].astype(str).agg("|".join, axis=1)
            try:
                stat, p, pivot = friedman_blocked(sub, entity_col="methods", block_col="block", value_col="score_std")
                friedman_results.append({"metric": metric, "friedman_stat": stat, "p": p, "n_blocks": pivot.shape[0], "n_methods": pivot.shape[1]})
                
                if isinstance(stat, float) and not math.isnan(stat):
                    ph = conover_posthoc_from_pivot(pivot)
                    ranks_ = pivot.rank(axis=1, ascending=False)
                    avg_ranks_ = ranks_.mean(axis=0).sort_values()
                    avg_ranks_.name = dataset_group
                    cliques = maximal_cliques_from_pmatrix(ph, alpha)
                    scored = rank_cliques(cliques, list(avg_ranks_.index), avg_ranks_)
                    layers = greedy_exclusive_layers(scored)   # disjoint “Top-1, Top-2, …” layers
                    layered_cliques.update({f'{dataset_group}':layers})
                    layered_ranks.append(avg_ranks_)
                    all_avg_ranks.setdefault(source_, {})[dataset_group] = avg_ranks_
            except Exception as e:
                logger.error(f"Error in Friedman/Posthoc for {source_} group {dataset_group}: {e}")
        clique_members = []
        clique_avg = []

        # Determine which groups are present in the data for this source
        available_groups = sorted(layered_cliques.keys())
        # Build datasets_order from available groups (e.g. ['0','1','2','3'] or ['0','1','3'])
        datasets_order = [g for g in ['0','1','2','3'] if g in available_groups]

        # Verify all available groups have cliques
        all_groups_present = True
        for c in datasets_order:
             if c not in layered_cliques:
                  all_groups_present = False
                  logger.warning(f"Group {c} ({distance_dict.get(c,c)}) missing in layers for {source_}")

        if all_groups_present and len(datasets_order) > 0:
            for c in datasets_order:
                clique_members.append( layered_cliques[c][0]['members'] )
                clique_avg.append(layered_cliques[c][0]['mean_rank'])

            member_df = pd.DataFrame([{name: True for name in names} for names in clique_members])
            member_df = member_df.where(member_df==True, False)
            member_df.index = [source_+'->'+distance_dict.get(d, d) for d in datasets_order]
            members_list.append(member_df)
        else:
             logger.warning(f"Skipping members plot for {source_} due to missing groups.")

    if not members_list:
        logger.error("No members lists created. Exiting.")
        return

    # Build reorder_index dynamically from the groups present in the data
    group_order = ['test', 'near', 'mid', 'far']
    # Detect which group labels are actually present in the members_list indices
    all_present_labels = set()
    for mdf in members_list:
        for idx in mdf.index:
            label = idx.split('->')[-1]
            all_present_labels.add(label)
    active_groups = [g for g in group_order if g in all_present_labels]

    reorder_index = []
    for group_label in active_groups:
        for src in sources:
            key = f'{src}->{group_label}'
            reorder_index.append(key)

    members_all = pd.concat(members_list,axis=0)
    
    valid_reorder = [idx for idx in reorder_index if idx in members_all.index]
    members_all = members_all.loc[valid_reorder]
    
    members_all = members_all.where(members_all==True, False)
    # Filter to existing indices from reorder_index
    members_all = members_all[sorted(members_all.columns, key=str.casefold)]

    members_all = members_all.loc[:, members_all.sum(axis=0) > 0]
    # Plotting Logic — auto-size based on data
    n_rows = len(members_all)
    n_cols = len(members_all.columns)
    figsize = (max(4, 0.25 * n_cols + 2), max(4, 0.35 * n_rows + 2))

    logger.info("Generating plot...")
    fig, ax = plt.subplots(1,2,figsize=figsize,width_ratios=[0.8,0.2],sharey='all')
    plt.subplots_adjust(wspace=0.05)
    
    # Build c_list dynamically: one color per group, 4 sources per group
    group_colors = ['tab:green', 'tab:blue', 'tab:red', 'tab:orange']
    n_sources = len(sources)
    c_list = []
    for i, _ in enumerate(active_groups):
        c_list.extend([group_colors[i % len(group_colors)]] * n_sources)
    # Extend if more rows than expected
    if len(members_all) > len(c_list):
        c_list = (c_list * (len(members_all)//len(c_list) + 1))[:len(members_all)]


    display_rename = {
        "KPCA RecError global": "KPCA RecError",
        "PCA RecError global": "PCA RecError",
        "MCD-KPCA RecError global": "MCD-KPCA RecError",
        "MCD-PCA RecError global": "MCD-PCA RecError",
    }
    members_plot = members_all.rename(columns=display_rename)

    try:
        plot_grid(members_plot, color_dotline=c_list, ax=ax[0], zorder=10)
    except Exception as e:
        logger.error(f"Error plotting grid: {e}")
        return

    grouping_label = f', Grouping: {os.path.basename(args.clip_dir)}' if args.clip_dir != 'clip_scores' else ''
    ax[0].set_title(f'Top cliques\n(Backbone:{f"Convolutional" if BACKBONE=="Conv" else "Transformer"},\nMetrics={metric}{grouping_label})')

    sns.barplot(x=members_plot.sum(axis=0), y=members_plot.columns, color='gray', ax=ax[1])
    # For modern matplotlib versions
    try:
        ax[1].bar_label(ax[1].containers[0])
    except:
        pass # Skip if old matplotlib
        
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)
    for x in ["top", "bottom", "right"]:
        ax[1].spines[x].set_visible(False)

    model_suffix = f'_{"_".join(args.model)}' if args.model is not None else ''
    clip_suffix = f'_{os.path.basename(args.clip_dir)}' if args.clip_dir != 'clip_scores' else ''
    out_filename = f'top_cliques_{BACKBONE}_{MCD_flag}_{args.metric_group}{model_suffix}{clip_suffix}'
    out_path = os.path.join(OUTDIR, out_filename)
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    fig.savefig(out_path + '.jpeg', bbox_inches='tight')
    logger.success(f"Saved plot to: {out_path}")

    # Export top cliques as JSON for use by multinomial_analysis.py
    # Format: {source: {group_name: [method1, method2, ...]}}
    # Methods are sorted by mean rank (best first).
    # Ranks stored under "_ranks": {source: {group: {method: rank}}}
    cliques_export: dict[str, dict] = {}
    ranks_export: dict[str, dict[str, dict[str, float]]] = {}
    for idx in members_all.index:
        source, group_name = idx.split('->')
        methods = [col for col in members_all.columns if members_all.loc[idx, col]]
        # Sort methods by mean rank (best first) using the numeric group key
        group_key = {v: k for k, v in distance_dict.items()}.get(group_name, group_name)
        if source in all_avg_ranks and group_key in all_avg_ranks[source]:
            avg_r = all_avg_ranks[source][group_key]
            methods = sorted(methods, key=lambda m: avg_r.get(m, float("inf")))
            ranks_export.setdefault(source, {})[group_name] = {
                m: round(float(avg_r.get(m, float("nan"))), 4) for m in methods
            }
        cliques_export.setdefault(source, {})[group_name] = methods

    # Add "all" cliques (pooled OOD groups) — not shown in plot but needed by multinomial_analysis.py
    for source_ in sources:
        df_src = df_combined[df_combined['source'] == source_].copy()
        if BACKBONE == 'ViT':
            df_src = df_src[df_src['methods'] != 'Confidence']
        df_src_met = df_src[(df_src['metric'] == metric[0]) | (df_src['metric'] == metric[1])].copy()
        ood_groups = [g for g in df_src_met['group'].unique() if g != '0']
        if not ood_groups:
            continue
        sub_all = df_src_met[df_src_met['group'].isin(ood_groups)].copy()
        sub_all["block"] = sub_all[blocks].astype(str).agg("|".join, axis=1)
        try:
            stat, p, pivot = friedman_blocked(sub_all, entity_col="methods", block_col="block", value_col="score_std")
            if isinstance(stat, float) and not math.isnan(stat):
                ph = conover_posthoc_from_pivot(pivot)
                ranks_ = pivot.rank(axis=1, ascending=False)
                avg_ranks_ = ranks_.mean(axis=0).sort_values()
                all_cliques = maximal_cliques_from_pmatrix(ph, alpha)
                scored = rank_cliques(all_cliques, list(avg_ranks_.index), avg_ranks_)
                layers = greedy_exclusive_layers(scored)
                if layers:
                    # Sort members by mean rank (best first)
                    members = sorted(layers[0]["members"],
                                     key=lambda m: avg_ranks_.get(m, float("inf")))
                    cliques_export.setdefault(source_, {})["all"] = members
                    ranks_export.setdefault(source_, {})["all"] = {
                        m: round(float(avg_ranks_.get(m, float("nan"))), 4) for m in members
                    }
                    all_avg_ranks.setdefault(source_, {})["all_pooled"] = avg_ranks_
        except Exception:
            pass

    # Attach ranks under a reserved key
    cliques_export["_ranks"] = ranks_export

    cliques_path = out_path + '_cliques.json'
    with open(cliques_path, 'w') as f:
        json.dump(cliques_export, f, indent=2)
    logger.success(f"Saved cliques JSON to: {cliques_path}")

if __name__ == "__main__":
    main()

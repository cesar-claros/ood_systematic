"""Step 16: Reporting figures and tables.

Produces (per protocol §12, reduced scope):

  outputs/figures/regret_table.md
      Markdown summary of step-14 aggregates: best NC predictor vs best
      baseline per (track, split, regime, side), with bootstrap 95% CI.

  outputs/figures/nc_feature_importance.{pdf,png}
      Regression perm-importance and multilabel binary mean |coef| side by
      side, faceted by split (xarch, lopo).

  outputs/figures/regret_by_side.{pdf,png}
      Per (regime, side) bar chart: best NC binary head vs best baseline vs
      regression top-1, with 95% CI error bars. Cross-arch headline.

  outputs/figures/competitive_heatmap_xarch.{pdf,png}
      Per-row predicted vs true competitive set on ResNet18 cross-arch test,
      faceted by paradigm.

  outputs/figures/mantel_scatter.{pdf,png}
      Pairwise NC distance vs AUGRC-vector cosine distance for the xarch
      test pool, with linear fit + Mantel r/p annotation.

  outputs/figures/wilcoxon_summary.md
      Tally of NC predictors that significantly beat each baseline
      (Holm-corrected) per (split, regime, side).

  outputs/14_reporting_check.md
      One-page index of generated artifacts with thumbnails-style notes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]
HEAD_SIDE_CSFS = {
    "REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
    "GradNorm", "Energy", "Confidence", "pNML",
}
FEATURE_SIDE_CSFS = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}


def add_id_track1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_id"] = (
        df["architecture"].astype(str) + "|"
        + df["paradigm"].astype(str) + "|"
        + df["source"].astype(str) + "|"
        + df["run"].astype(int).astype(str) + "|"
        + df["dropout"].astype(int).astype(str) + "|"
        + df["reward"].apply(lambda x: f"{x:g}")
    )
    return df


def save_figure(fig, base: Path) -> None:
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---- 1. Regret table markdown ----

def write_regret_table(out_root: Path, fig_dir: Path) -> None:
    rows = []
    for track in (1, 2):
        for split_dir in (out_root / f"track{track}").glob("*/baselines/aggregate.parquet"):
            split = split_dir.parent.parent.name
            ag = pq.read_table(split_dir).to_pandas()
            for (regime, side), g in ag.groupby(["regime", "side"]):
                # Best NC (lowest regret_raw_mean among predictor_imputed or predictor_raw=regression)
                nc = g[(g["comparator_kind"] == "predictor_imputed")
                       | ((g["comparator_kind"] == "predictor_raw")
                          & (g["comparator_name"] == "regression"))]
                if nc.empty:
                    continue
                nc_best = nc.loc[nc["regret_raw_mean"].idxmin()]
                # Best baseline
                bl = g[g["comparator_kind"] == "baseline"]
                if bl.empty:
                    continue
                bl_best = bl.loc[bl["regret_raw_mean"].idxmin()]
                rows.append({
                    "track": track, "split": split,
                    "regime": regime, "side": side,
                    "best_nc": nc_best["comparator_name"],
                    "nc_regret": nc_best["regret_raw_mean"],
                    "nc_ci_lo": nc_best["regret_raw_ci_lo"],
                    "nc_ci_hi": nc_best["regret_raw_ci_hi"],
                    "best_baseline": bl_best["comparator_name"],
                    "bl_regret": bl_best["regret_raw_mean"],
                    "bl_ci_lo": bl_best["regret_raw_ci_lo"],
                    "bl_ci_hi": bl_best["regret_raw_ci_hi"],
                    "nc_minus_bl": nc_best["regret_raw_mean"] - bl_best["regret_raw_mean"],
                })
    if not rows:
        return
    df = pd.DataFrame(rows).round(3)
    df = df.sort_values(["track", "split", "regime", "side"])
    md_lines = ["# Regret table — best NC predictor vs best baseline\n\n"]
    md_lines.append("Per (track, split, regime, side): the NC predictor with "
                    "lowest mean regret (using imputation for binary heads "
                    "or raw top-1 for regression) is compared to the best "
                    "baseline. `nc_minus_bl < 0` ⇒ NC wins.\n\n")
    md_lines.append("```\n" + df.to_string(index=False) + "\n```\n")
    (fig_dir / "regret_table.md").write_text("".join(md_lines))


# ---- 2. NC feature importance ----

def plot_nc_feature_importance(out_root: Path, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    splits = ["xarch", "lopo"]
    for col, split in enumerate(splits):
        # Regression permutation importance (mean across folds)
        fi_path = out_root / "track1" / split / "regression" / "feature_importance.parquet"
        if fi_path.exists():
            fi = pq.read_table(fi_path).to_pandas()
            fi_nc = fi[fi["feature"].isin(NC_PRIMARY)]
            agg = fi_nc.groupby("feature")["importance_mean"].mean().reindex(NC_PRIMARY)
            axes[0, col].barh(agg.index, agg.values, color="#1f77b4")
            axes[0, col].set_title(f"Regression — perm. importance ({split})")
            axes[0, col].set_xlabel("permutation MSE increase")
            axes[0, col].invert_yaxis()
        # Binary multilabel head mean |coefficient| (clique label rule)
        cp = (out_root / "track1" / split / "multilabel_competitive"
              / "clique" / "coefficients.parquet")
        if cp.exists():
            coefs = pq.read_table(cp).to_pandas()
            nc_only = coefs[coefs["feature"].isin(NC_PRIMARY)]
            rank = (nc_only.groupby("feature")["coefficient"]
                    .apply(lambda s: float(np.mean(np.abs(s))))
                    .reindex(NC_PRIMARY))
            axes[1, col].barh(rank.index, rank.values, color="#ff7f0e")
            axes[1, col].set_title(f"Multilabel binary — mean |coef| ({split})")
            axes[1, col].set_xlabel("mean |logistic coef| across CSFs")
            axes[1, col].invert_yaxis()
    fig.suptitle("NC feature importance: regression vs multilabel binary head", y=1.01)
    plt.tight_layout()
    save_figure(fig, fig_dir / "nc_feature_importance")


# ---- 3. Regret by side ----

def plot_regret_by_side(out_root: Path, fig_dir: Path) -> None:
    """xarch headline: best NC predictor (imputed) + best baseline + regression
    per (regime, side)."""
    ag_path = out_root / "track1" / "xarch" / "baselines" / "aggregate.parquet"
    if not ag_path.exists():
        return
    ag = pq.read_table(ag_path).to_pandas()
    regimes = ["near", "mid", "far"]
    sides = ["all", "feature", "head"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for j, side in enumerate(sides):
        labels, means, lo, hi, colors = [], [], [], [], []
        for regime in regimes:
            sub = ag[(ag["regime"] == regime) & (ag["side"] == side)]
            if sub.empty:
                continue
            nc = sub[sub["comparator_kind"] == "predictor_imputed"]
            nc_best = nc.loc[nc["regret_raw_mean"].idxmin()] if not nc.empty else None
            bl = sub[sub["comparator_kind"] == "baseline"]
            bl_best = bl.loc[bl["regret_raw_mean"].idxmin()] if not bl.empty else None
            reg = sub[(sub["comparator_kind"] == "predictor_raw")
                      & (sub["comparator_name"] == "regression")]
            reg_row = reg.iloc[0] if not reg.empty else None

            if nc_best is not None:
                labels.append(f"{regime}\nbest NC")
                means.append(nc_best["regret_raw_mean"])
                lo.append(nc_best["regret_raw_mean"] - nc_best["regret_raw_ci_lo"])
                hi.append(nc_best["regret_raw_ci_hi"] - nc_best["regret_raw_mean"])
                colors.append("#2ca02c")
            if reg_row is not None:
                labels.append(f"{regime}\nregression")
                means.append(reg_row["regret_raw_mean"])
                lo.append(reg_row["regret_raw_mean"] - reg_row["regret_raw_ci_lo"])
                hi.append(reg_row["regret_raw_ci_hi"] - reg_row["regret_raw_mean"])
                colors.append("#1f77b4")
            if bl_best is not None:
                labels.append(f"{regime}\n{bl_best['comparator_name']}")
                means.append(bl_best["regret_raw_mean"])
                lo.append(bl_best["regret_raw_mean"] - bl_best["regret_raw_ci_lo"])
                hi.append(bl_best["regret_raw_ci_hi"] - bl_best["regret_raw_mean"])
                colors.append("#d62728")

        x = np.arange(len(labels))
        axes[j].bar(x, means, yerr=[lo, hi], capsize=4, color=colors,
                    edgecolor="black", linewidth=0.5)
        axes[j].set_xticks(x)
        axes[j].set_xticklabels(labels, rotation=0, fontsize=8)
        axes[j].set_title(f"side = {side}")
        axes[j].set_ylabel("Mean regret (raw AUGRC) ± 95% bootstrap CI")
        axes[j].grid(axis="y", alpha=0.3)
    fig.suptitle("xarch (Track 1, VGG13 → ResNet18) — best NC vs best baseline", y=1.02)
    plt.tight_layout()
    save_figure(fig, fig_dir / "regret_by_side")


# ---- 4. Competitive heatmap ----

def plot_competitive_heatmap(out_root: Path, fig_dir: Path) -> None:
    """For each ResNet18 test model row, show predicted competitive set
    (multilabel/within_eps_majority) vs the oracle CSF."""
    pp = (out_root / "track1" / "xarch" / "multilabel_competitive"
          / "within_eps_majority" / "preds.parquet")
    if not pp.exists():
        return
    preds = pq.read_table(pp).to_pandas()
    # Use 'all' regime predictions
    preds = preds[preds["regime"] == "near"]  # use near for compactness

    oracle_path = out_root / "track1" / "dataset" / "oracle.parquet"
    oracle = pq.read_table(oracle_path).to_pandas()
    oracle = add_id_track1(oracle)
    oracle = oracle[(oracle["regime"] == "near")
                    & (oracle["architecture"] == "ResNet18")]

    csfs = sorted(preds["csf"].unique(), key=str.casefold)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, paradigm in zip(axes, ["confidnet", "devries", "dg"]):
        # Restrict to this paradigm's ResNet18 rows
        ids_for_par = oracle[oracle["paradigm"] == paradigm]["model_id"].unique()
        if len(ids_for_par) == 0:
            continue
        sorted_ids = sorted(ids_for_par)
        # Build matrix: rows = csfs, cols = (model_id, eval_dataset)
        rows_pred, rows_oracle = [], []
        col_labels = []
        for mid in sorted_ids:
            evs = oracle[oracle["model_id"] == mid]["eval_dataset"].unique()
            for ev in sorted(evs):
                col_labels.append(f"{mid.split('|')[2]}/{ev}")
                pred_set = set(preds[(preds["model_id"] == mid)
                                     & preds["predicted_competitive"]]["csf"].unique())
                oracle_csf = oracle[(oracle["model_id"] == mid)
                                    & (oracle["eval_dataset"] == ev)
                                    ]["oracle_csf"].iloc[0]
                rows_pred.append([1 if c in pred_set else 0 for c in csfs])
                rows_oracle.append([1 if c == oracle_csf else 0 for c in csfs])
        if not col_labels:
            continue
        pred_mat = np.array(rows_pred).T  # csfs × cols
        ora_mat = np.array(rows_oracle).T
        # Combine: 0 = neither, 1 = predicted only, 2 = oracle only, 3 = both
        combo = pred_mat + 2 * ora_mat
        cmap = plt.cm.colors.ListedColormap(["white", "#1f77b4", "#d62728", "#9467bd"])
        ax.imshow(combo, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        ax.set_yticks(range(len(csfs)))
        ax.set_yticklabels(csfs, fontsize=7)
        ax.set_title(f"paradigm = {paradigm} (n cols = {len(col_labels)})")
        ax.set_xlabel("test rows (model × eval)")

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="predicted only"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="oracle only"),
        plt.Rectangle((0, 0), 1, 1, color="#9467bd", label="both"),
        plt.Rectangle((0, 0), 1, 1, color="white", edgecolor="grey", label="neither"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("xarch / regime=near — predicted competitive (multilabel/within_eps_majority) "
                 "vs oracle CSF", y=1.01)
    plt.tight_layout()
    save_figure(fig, fig_dir / "competitive_heatmap_xarch")


# ---- 5. Mantel scatter ----

def plot_mantel_scatter(out_root: Path, fig_dir: Path) -> None:
    """Reproduce the Mantel computation visually for the xarch test pool."""
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_id_track1(long_df)
    long_df = long_df[(long_df["regime"] != "test")
                      & (long_df["architecture"] == "ResNet18")]
    nc_per = (long_df[["model_id"] + NC_PRIMARY]
              .drop_duplicates(subset=["model_id"]).reset_index(drop=True))
    if len(nc_per) > 100:
        nc_per = nc_per.sample(100, random_state=0).reset_index(drop=True)
    ids = nc_per["model_id"].tolist()
    Xs = StandardScaler().fit_transform(nc_per[NC_PRIMARY].values)
    n = len(ids)
    aug_long = long_df[long_df["model_id"].isin(ids)]
    pivot = (aug_long.groupby(["model_id", "csf"])["augrc"].mean()
             .unstack(fill_value=0).reindex(ids))

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    nc_d = np.array([np.linalg.norm(Xs[i] - Xs[j]) for i, j in pairs])
    perf_d = np.array([cosine(pivot.iloc[i].values, pivot.iloc[j].values)
                       for i, j in pairs])

    r = float(np.corrcoef(nc_d, perf_d)[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(nc_d, perf_d, alpha=0.25, s=10, color="#1f77b4", edgecolor="none")
    z = np.polyfit(nc_d, perf_d, 1)
    xs = np.linspace(nc_d.min(), nc_d.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color="#d62728", linewidth=2,
            label=f"linear fit (slope={z[0]:.4f})")
    ax.set_xlabel("Pairwise NC distance (Euclidean, standardized)")
    ax.set_ylabel("Pairwise CSF-AUGRC distance (cosine)")
    ax.set_title(f"Mantel scatter — xarch ResNet18 pool (n={n}, "
                 f"{len(pairs)} pairs)\nMantel r = {r:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, fig_dir / "mantel_scatter")


# ---- 6. Wilcoxon summary ----

def write_wilcoxon_summary(out_root: Path, fig_dir: Path) -> None:
    """Tally NC predictors that significantly beat each baseline (Holm-corrected)."""
    rows = []
    for jp in (out_root / "track1").glob("*/stats.json"):
        split = jp.parent.name
        with open(jp) as f:
            d = json.load(f)
        for r in d["tests"].get("wilcoxon", []):
            if r.get("reject_holm_05") and (r.get("p_holm") or 1) <= 0.05:
                rows.append({
                    "split": split, "regime": r["regime"], "side": r["side"],
                    "predictor": r["predictor"].split("::", 1)[-1],
                    "baseline": r["baseline"].split("::", 1)[-1],
                    "median_diff": r["median_diff"],
                    "p_holm": r["p_holm"],
                })
    if not rows:
        (fig_dir / "wilcoxon_summary.md").write_text("No significant Wilcoxon wins.")
        return
    df = pd.DataFrame(rows).round(4).sort_values(
        ["split", "regime", "side", "p_holm"])
    lines = ["# Wilcoxon — NC predictors that significantly beat baselines "
             "(Holm-corrected, α=0.05)\n\n"]
    lines.append("Sign convention: `median_diff = regret(predictor) − "
                 "regret(baseline)`. Negative ⇒ NC wins.\n\n")
    lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")
    counts = df.groupby(["split", "predictor"]).size().rename("n_baseline_wins").reset_index()
    counts = counts.sort_values(["split", "n_baseline_wins"], ascending=[True, False])
    lines.append("## Per-(split, predictor) tally of baselines beaten\n\n")
    lines.append("```\n" + counts.to_string(index=False) + "\n```\n")
    (fig_dir / "wilcoxon_summary.md").write_text("".join(lines))


# ---- Final report ----

def report(fig_dir: Path, out_path: Path) -> None:
    artifacts = sorted(fig_dir.iterdir())
    lines = ["# Step 16 — Reporting figures and tables\n\n"]
    lines.append("**Date:** 2026-05-04\n")
    lines.append("**Source:** `code/nc_csf_predictivity/evaluation/report.py`\n\n")
    lines.append("## Generated artifacts\n\n")
    for a in artifacts:
        size_kb = a.stat().st_size / 1024 if a.exists() else 0
        lines.append(f"- `{a.relative_to(fig_dir.parent)}`  ({size_kb:.0f} KB)\n")
    lines.append("\n## Notes\n\n")
    lines.append(
        "All PDF/PNG pairs are saved at 150 dpi for PDF (vector) and PNG. "
        "`regret_table.md` and `wilcoxon_summary.md` are markdown extracts "
        "of the underlying parquet aggregates from steps 14–15. The Figure-1 "
        "candidate is `competitive_heatmap_xarch.{pdf,png}`; key support "
        "figure for the activation-vs-weight finding is "
        "`nc_feature_importance.{pdf,png}`. Mantel scatter visualizes the "
        "step-15 Mantel statistic for the xarch ResNet18 test pool.\n"
    )
    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("regret_table.md ...")
    write_regret_table(out_root, fig_dir)
    print("nc_feature_importance ...")
    plot_nc_feature_importance(out_root, fig_dir)
    print("regret_by_side ...")
    plot_regret_by_side(out_root, fig_dir)
    print("competitive_heatmap_xarch ...")
    plot_competitive_heatmap(out_root, fig_dir)
    print("mantel_scatter ...")
    plot_mantel_scatter(out_root, fig_dir)
    print("wilcoxon_summary.md ...")
    write_wilcoxon_summary(out_root, fig_dir)

    report(fig_dir, out_root / "14_reporting_check.md")
    print(f"wrote {out_root / '14_reporting_check.md'}")


if __name__ == "__main__":
    main()

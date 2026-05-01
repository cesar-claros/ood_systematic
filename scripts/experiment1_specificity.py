"""Experiment 1 - Specificity figure (2x2 Spearman matrix).

Implements documentation/head_feature_metric_predictivity_protocol.md, Sec 3.

For each (CSF c, regime r, risk metric m in {AUGRC, AURC}):
  - target = mean rank of c within its family at this (paradigm, source) cell.
    Ranks are computed within each (config, ood_dataset) PER FAMILY, then
    averaged across (config, ood_dataset) within the cell+regime support.
  - predictors = NC-Papyan (8 features) OR HL-recipe (20 features).
  - 5-fold CV ridge with inner-CV alpha selection.
  - Score: held-out Spearman rho (primary) and R^2 (secondary).

Aggregate by (feature_set, csf_family) -> 2x2 boxplot.
Compute the specificity gap: median rho(F -> matched) - median rho(F -> other).

Outputs in code/ood_eval_outputs/experiment1_specificity/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_SIDE_CSFS = ["REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE", "GradNorm",
                  "Energy", "Confidence", "pNML"]
FEATURE_SIDE_CSFS = ["PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM",
                     "Maha", "fDBD", "KPCA RecError global", "Residual"]

NC_PAPYAN = ["var_collapse", "equiangular_uc", "equiangular_wc",
             "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
             "max_equiangular_wc", "self_duality"]

HL_RECIPE = [
    # NC-on-logits (8)
    "logit_classmean_cv_norm", "logit_classmean_cos_mse",
    "logit_classmean_cos_max_dev", "logit_classmean_cos_mean",
    "logit_within_between_ratio", "logit_classmean_mean_norm",
    "logit_cov_participation", "logit_cov_effrank",
    # Norm-scaled (3)
    "scaled_logitnorm_mean", "scaled_logitnorm_std", "scaled_logitnorm_p50",
    # Confidence-scaled (9)
    "scaled_entropy_mean", "scaled_entropy_std", "scaled_entropy_p50",
    "scaled_maxprob_mean", "scaled_maxprob_std", "scaled_maxprob_p50",
    "scaled_kl_uniform_mean", "scaled_kl_uniform_std", "scaled_kl_uniform_p50",
]

STUDY_TO_MODEL = {"confidnet": "confidnet", "devries": "devries",
                  "dg": "dg", "vit": "modelvit"}
MODEL_TO_STUDY = {v: k for k, v in STUDY_TO_MODEL.items()}
SOURCE_FILE_TO_NORM = {"cifar10": "cifar10", "cifar100": "cifar100",
                       "supercifar100": "supercifar100",
                       "tinyimagenet": "tinyimagenet"}
OOD_GROUP_LABELS = {0: "test", 1: "near", 2: "mid", 3: "far"}

PARADIGMS = ["confidnet", "devries", "dg", "vit"]
SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
REGIMES = ["test", "near", "mid", "far"]

OUTDIR = CODE_DIR / "ood_eval_outputs" / "experiment1_specificity"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_clip_groupings() -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for src in SOURCES:
        f = CODE_DIR / "clip_scores" / f"clip_distances_{src}.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f, header=[0, 1])
        c.columns = c.columns.droplevel(0)
        c = c.rename(columns={"Unnamed: 0_level_1": "dataset",
                              "Unnamed: 5_level_1": "group"})
        for _, row in c.iterrows():
            out[(src, str(row["dataset"]))] = OOD_GROUP_LABELS[int(row["group"])]
    return out


def _melt_scores(metric_name: str) -> pd.DataFrame:
    """Long-format scores for one risk metric, both backbones, all sources."""
    rows: list[pd.DataFrame] = []
    keep = set(HEAD_SIDE_CSFS) | set(FEATURE_SIDE_CSFS)
    for src in SOURCES:
        for bb in ("Conv", "ViT"):
            f = (CODE_DIR / "scores_risk" /
                 f"scores_{metric_name}_MCD-False_{bb}_{src}_fix-config.csv")
            if not f.exists():
                logger.warning(f"missing: {f}")
                continue
            df = pd.read_csv(f)
            df = df[df["methods"].isin(keep)].copy()
            df["source"] = src
            df["backbone"] = bb
            id_cols = ["source", "backbone", "model", "drop out",
                       "methods", "reward"]
            ood_cols = [c for c in df.columns if c not in id_cols]
            long = df.melt(id_vars=id_cols, value_vars=ood_cols,
                           var_name="ood_dataset", value_name="score")
            rows.append(long)
    return pd.concat(rows, ignore_index=True)


def build_targets(metric_name: str) -> pd.DataFrame:
    """Per (paradigm, source, csf, regime) mean rank within family."""
    scores = _melt_scores(metric_name)
    scores["paradigm"] = scores["model"].map(MODEL_TO_STUDY)
    scores["dropout"] = (scores["drop out"] == "do1")
    scores["reward"] = scores["reward"].str.replace("rew", "").astype(float)

    clip_map = _load_clip_groupings()
    scores["regime"] = scores.apply(
        lambda r: clip_map.get((r["source"], r["ood_dataset"]), None), axis=1)
    scores = scores.dropna(subset=["regime"])

    scores["csf_family"] = scores["methods"].apply(
        lambda c: "head" if c in HEAD_SIDE_CSFS else
                  ("feature" if c in FEATURE_SIDE_CSFS else None))
    scores = scores.dropna(subset=["csf_family"])

    # Rank within each (config, ood_dataset) PER family (1 = lowest risk).
    rank_keys = ["source", "backbone", "model", "drop out", "reward",
                 "ood_dataset", "csf_family"]
    scores["rank"] = scores.groupby(rank_keys)["score"].rank(
        ascending=True, method="average")

    cell_keys = ["paradigm", "source", "methods", "regime"]
    targets = scores.groupby(cell_keys, as_index=False)["rank"].mean()
    targets = targets.rename(columns={"methods": "csf", "rank": "mean_rank"})
    targets["metric"] = metric_name
    targets["csf_family"] = targets["csf"].apply(
        lambda c: "head" if c in HEAD_SIDE_CSFS else "feature")
    return targets


def build_centroids() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Per (paradigm, source) cell: NC-Papyan and HL-recipe centroids."""
    nc = pd.read_csv(CODE_DIR / "neural_collapse_metrics" / "nc_metrics.csv",
                     index_col=0)
    nc = nc.rename(columns={"dataset": "source", "study": "paradigm"})
    nc["source"] = nc["source"].replace({"supercifar": "supercifar100"})
    nc = nc[nc["architecture"] != "ResNet18"].copy()
    excl_n = ((nc["paradigm"] == "dg") & (nc["source"] == "supercifar100")
              & np.isclose(nc["reward"].astype(float), 2.2))
    nc = nc.loc[~excl_n].copy()
    nc[NC_PAPYAN] = (nc[NC_PAPYAN] - nc[NC_PAPYAN].mean()) / nc[NC_PAPYAN].std(ddof=0)

    hl = pd.read_csv(CODE_DIR / "head_logit_metrics" / "hl_metrics.csv",
                     index_col=0)
    hl = hl.rename(columns={"dataset": "source", "study": "paradigm"})
    hl["source"] = hl["source"].replace({"supercifar": "supercifar100"})
    hl = hl[hl["architecture"] != "ResNet18"].copy()
    excl_h = ((hl["paradigm"] == "dg") & (hl["source"] == "supercifar100")
              & np.isclose(hl["reward"].astype(float), 2.2))
    hl = hl.loc[~excl_h].copy()
    missing = [c for c in HL_RECIPE if c not in hl.columns]
    if missing:
        logger.warning(f"HL recipe columns missing: {missing}")
    hl_cols = [c for c in HL_RECIPE if c in hl.columns]
    hl[hl_cols] = (hl[hl_cols] - hl[hl_cols].mean()) / hl[hl_cols].std(ddof=0)

    centroids: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for p in PARADIGMS:
        for s in SOURCES:
            nc_rows = nc[(nc["paradigm"] == p) & (nc["source"] == s)]
            hl_rows = hl[(hl["paradigm"] == p) & (hl["source"] == s)]
            if nc_rows.empty or hl_rows.empty:
                continue
            centroids[(p, s)] = {
                "NC": nc_rows[NC_PAPYAN].mean().to_numpy(),
                "HL": hl_rows[hl_cols].mean().to_numpy(),
            }
    return centroids


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def cv_score(X: np.ndarray, y: np.ndarray,
             n_splits: int = 5, seed: int = 0) -> tuple[float, float]:
    n = len(y)
    if n < n_splits + 1 or X.shape[1] == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(n)
    for tr, te in kf.split(X):
        if np.std(y[tr]) == 0:
            preds[te] = y[tr].mean()
            continue
        pipe = Pipeline([("s", StandardScaler()),
                         ("r", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
    sp, _ = spearmanr(preds, y)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(sp), r2


def run_regressions(centroids: dict, targets_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (csf, regime, metric, family), g in targets_all.groupby(
            ["csf", "regime", "metric", "csf_family"]):
        cells, ys = [], []
        for _, row in g.iterrows():
            key = (row["paradigm"], row["source"])
            if key in centroids:
                cells.append(key)
                ys.append(row["mean_rank"])
        if len(cells) < 6:
            continue
        y = np.asarray(ys, dtype=float)
        for fset_name in ("NC", "HL"):
            X = np.stack([centroids[c][fset_name] for c in cells])
            sp, r2 = cv_score(X, y, n_splits=5)
            rows.append({
                "feature_set": fset_name,
                "csf_family": family,
                "csf": csf,
                "regime": regime,
                "metric": metric,
                "n_samples": len(cells),
                "n_features": X.shape[1],
                "spearman": sp,
                "r2": r2,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def specificity_gap(per_cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fset, matched in (("NC", "feature"), ("HL", "head")):
        sub = per_cell[per_cell["feature_set"] == fset]
        unmatched = "head" if matched == "feature" else "feature"
        med_m = sub[sub["csf_family"] == matched]["spearman"].median()
        med_u = sub[sub["csf_family"] == unmatched]["spearman"].median()
        rows.append({
            "feature_set": fset,
            "matched_family": matched,
            "median_rho_matched": med_m,
            "median_rho_unmatched": med_u,
            "gap_matched_minus_unmatched": med_m - med_u,
        })
    return pd.DataFrame(rows)


def make_2x2_boxplot(per_cell: pd.DataFrame, out_path: str) -> None:
    fsets = ["NC", "HL"]
    families = ["head", "feature"]
    titles = {"NC": "NC-Papyan (8 feats)",
              "HL": "HL-recipe (20 feats)"}
    family_titles = {"head": "head-side CSFs",
                     "feature": "feature-side CSFs"}
    color_for_fset = {"NC": "#cccccc", "HL": "#a6cde4"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    for i, fs in enumerate(fsets):
        for j, fam in enumerate(families):
            ax = axes[i, j]
            sub = per_cell[(per_cell["feature_set"] == fs)
                           & (per_cell["csf_family"] == fam)]
            data = [sub[sub["regime"] == r]["spearman"].dropna().tolist()
                    for r in REGIMES]
            bp = ax.boxplot(data, positions=range(len(REGIMES)),
                            patch_artist=True, widths=0.6)
            for patch in bp["boxes"]:
                patch.set_facecolor(color_for_fset[fs])
            ax.set_xticks(range(len(REGIMES)))
            ax.set_xticklabels(REGIMES)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_ylim(-1, 1)
            ax.grid(axis="y", linestyle=":", alpha=0.4)
            if i == 0:
                ax.set_title(family_titles[fam])
            if j == 0:
                ax.set_ylabel(f"{titles[fs]}\nCV Spearman rho")
    fig.suptitle("Experiment 1 - Matched-specificity cross "
                 "(rows: feature set; cols: CSF family)", y=1.00, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.success(f"Saved 2x2 boxplot: {out_path}.pdf")


def make_summary_bar(per_cell: pd.DataFrame, out_path: str) -> None:
    """Bar chart: median Spearman per (feature_set, csf_family). 4 bars."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fsets = ["NC", "HL"]
    fams = ["head", "feature"]
    pos, heights, errs, labels, colors = [], [], [], [], []
    for j, fam in enumerate(fams):
        for i, fs in enumerate(fsets):
            sub = per_cell[(per_cell["feature_set"] == fs)
                           & (per_cell["csf_family"] == fam)]["spearman"].dropna()
            if sub.empty:
                continue
            pos.append(j * 3 + i)
            heights.append(sub.median())
            errs.append([sub.median() - sub.quantile(0.25),
                         sub.quantile(0.75) - sub.median()])
            labels.append(f"{fs}->{fam}")
            colors.append("#cccccc" if fs == "NC" else "#a6cde4")
    errs = np.array(errs).T
    ax.bar(pos, heights, color=colors, edgecolor="black", linewidth=0.8,
           yerr=errs, capsize=4)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Median CV Spearman rho (IQR bars)")
    ax.set_title("Matched-specificity cross")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.success(f"Saved summary bar: {out_path}.pdf")


# ---------------------------------------------------------------------------
def main():
    logger.info("Loading centroids and targets...")
    centroids = build_centroids()
    logger.info(f"  cells: {len(centroids)}")

    targets = pd.concat([build_targets("AUGRC"), build_targets("AURC")],
                        ignore_index=True)
    logger.info(f"  target rows: {len(targets)}")

    logger.info("Running per-cell ridge regressions...")
    per_cell = run_regressions(centroids, targets)
    per_cell.to_csv(OUTDIR / "per_cell.csv", index=False)
    logger.info(f"  per_cell rows: {len(per_cell)}")

    summary = per_cell.groupby(["feature_set", "csf_family"]).agg(
        median_spearman=("spearman", "median"),
        q25_spearman=("spearman", lambda x: x.quantile(0.25)),
        q75_spearman=("spearman", lambda x: x.quantile(0.75)),
        median_r2=("r2", "median"),
        n_csfs=("csf", "nunique"),
        n_cells=("spearman", "count"),
    ).reset_index()
    summary.to_csv(OUTDIR / "summary.csv", index=False)
    print("\n=== Summary (median CV Spearman across csf x regime x metric) ===")
    print(summary.to_string(index=False))

    gap = specificity_gap(per_cell)
    gap.to_csv(OUTDIR / "specificity_gap.csv", index=False)
    print("\n=== Specificity gap ===")
    print(gap.to_string(index=False))

    make_2x2_boxplot(per_cell, str(OUTDIR / "specificity_box"))
    make_summary_bar(per_cell, str(OUTDIR / "specificity_bar"))

    logger.success(f"Saved outputs to {OUTDIR}/")


if __name__ == "__main__":
    main()

"""Predictive comparison: do head-side logit metrics predict head-side CSF
performance better than NC metrics?

For each (head-side CSF, evaluation group, AUGRC|AURC) cell, fit two ridge
regressions on the same train/test splits:

  A. predictors = NC metrics       (nc_metrics.csv)
  B. predictors = head-logit stats (hl_metrics.csv)

target = the CSF's **mean rank** within the head-side family, computed by
ranking CSFs by AUGRC/AURC within each individual OOD dataset (1 = best /
lowest risk) and averaging those ranks across the OOD datasets that fall
into the regime. Raw AUGRC/AURC magnitudes are not commensurate across
OOD datasets, so averaging raw scores would be dominated by per-dataset
difficulty; per-OOD ranking normalises that. This matches the per-block
ranking ``stats_eval.py`` uses upstream of Conover-Holm cliques.

Spearman rho between out-of-fold predictions and the mean-rank target is
the primary metric; R^2 is reported as a secondary score.

Outputs:
  ood_eval_outputs/head_vs_nc_regression/per_cell.csv  -- one row per cell
  ood_eval_outputs/head_vs_nc_regression/summary.csv   -- aggregated diff
  ood_eval_outputs/head_vs_nc_regression/r2_box.{pdf,jpeg}
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


HEAD_SIDE_CSFS = ["REN","PE","PCE","MSR","GEN","MLS","GE","GradNorm",
                  "Energy","Confidence","pNML"]

NC_FEATURES = ["var_collapse","cdnv_score","bias_collapse",
               "equiangular_uc","equiangular_wc","equinorm_uc","equinorm_wc",
               "max_equiangular_uc","max_equiangular_wc","self_duality",
               "w_etf_diff","M_etf_diff","wM_etf_diff"]

# Mapping: study label in metric tables -> model label in score CSVs.
STUDY_TO_MODEL = {"confidnet":"confidnet", "devries":"devries",
                  "dg":"dg", "vit":"modelvit"}
MODEL_TO_STUDY = {v: k for k, v in STUDY_TO_MODEL.items()}

# Source filename in scores_risk uses 'supercifar100' for the supercifar metric rows
SOURCE_TO_DATASET = {"cifar10":"cifar10", "cifar100":"cifar100",
                     "tinyimagenet":"tinyimagenet",
                     "supercifar100":"supercifar"}

OOD_GROUP_LABELS = {0:"test", 1:"near", 2:"mid", 3:"far"}

OUTDIR = "ood_eval_outputs/head_vs_nc_regression"


def _load_clip_groupings() -> dict[tuple[str,str], str]:
    """Map (source, ood_dataset) -> {test,near,mid,far}."""
    out: dict[tuple[str,str], str] = {}
    for src in ["cifar10","cifar100","supercifar100","tinyimagenet"]:
        f = f"clip_scores/clip_distances_{src}.csv"
        if not os.path.exists(f):
            continue
        c = pd.read_csv(f, header=[0,1])
        c.columns = c.columns.droplevel(0)
        c = c.rename(columns={"Unnamed: 0_level_1":"dataset",
                              "Unnamed: 5_level_1":"group"})
        for _, row in c.iterrows():
            out[(src, str(row["dataset"]))] = OOD_GROUP_LABELS[int(row["group"])]
    return out


def _melt_scores(metric_name: str) -> pd.DataFrame:
    """Long-format scores for one metric, all sources/backbones.

    Restricted to base head-side CSFs (HEAD_SIDE_CSFS): the exact-match
    `isin` filter excludes projection-filtered variants (\"MSR class\",
    \"MSR global\", \"MSR class avg\", \"MSR class pred\") and feature-side
    methods including CTMmean / CTMmeanOC.
    """
    rows: list[pd.DataFrame] = []
    sources = ["cifar10","cifar100","supercifar100","tinyimagenet"]
    backbones = ["Conv","ViT"]
    for src in sources:
        for bb in backbones:
            f = (f"scores_risk/scores_{metric_name}_MCD-False_"
                 f"{bb}_{src}_fix-config.csv")
            if not os.path.exists(f):
                logger.warning(f"missing: {f}")
                continue
            df = pd.read_csv(f)
            df = df[df["methods"].isin(HEAD_SIDE_CSFS)].copy()
            df["source"] = src
            df["backbone"] = bb
            id_cols = ["source","backbone","model","drop out",
                       "methods","reward"]
            ood_cols = [c for c in df.columns if c not in id_cols]
            long = df.melt(id_vars=id_cols, value_vars=ood_cols,
                           var_name="ood_dataset", value_name="score")
            rows.append(long)
    if not rows:
        raise RuntimeError("No score files loaded.")
    return pd.concat(rows, axis=0, ignore_index=True)


def build_dataset(metric_name: str) -> pd.DataFrame:
    """Long-format frame with NC + HL features + target score.

    Score files have one row per (model, drop out, methods, reward) — runs
    are already averaged. So we average HL/NC metrics across runs to match,
    then join on (dataset, study, dropout, reward).
    """
    hl = pd.read_csv("head_logit_metrics/hl_metrics.csv")
    nc = pd.read_csv("neural_collapse_metrics/nc_metrics.csv")

    hl = hl[hl["architecture"] != "ResNet18"].copy()
    nc = nc[nc["architecture"] != "ResNet18"].copy()
    keys = ["dataset","architecture","study","dropout","run","reward","lr"]
    hl_feats = [c for c in hl.columns
                if c not in keys and c not in ["Unnamed: 0","temperature_mcd"]]
    metrics_join = hl.merge(nc[keys + NC_FEATURES], on=keys, how="inner")
    logger.info(f"metrics-join rows: {len(metrics_join)} "
                f"(hl={len(hl)}, nc={len(nc)})")

    scores = _melt_scores(metric_name)
    scores["dataset"] = scores["source"].map(SOURCE_TO_DATASET)
    scores["study"] = scores["model"].map(MODEL_TO_STUDY)
    scores["dropout"] = (scores["drop out"] == "do1")
    scores["reward"] = scores["reward"].str.replace("rew","").astype(float)

    clip_map = _load_clip_groupings()
    scores["group"] = scores.apply(
        lambda r: clip_map.get((r["source"], r["ood_dataset"]), None), axis=1)
    scores = scores.dropna(subset=["group"])

    # Rank head-side CSFs within each (model config, OOD dataset). 1 = lowest
    # risk = best. Averaging ranks rather than raw scores normalises each
    # OOD dataset's intrinsic scale.
    rank_keys = ["source","backbone","model","drop out","reward","ood_dataset"]
    scores["rank"] = scores.groupby(rank_keys)["score"].rank(
        ascending=True, method="average")

    agg_keys = ["dataset","study","dropout","reward","methods","group"]
    agg = scores.groupby(agg_keys, as_index=False)["rank"].mean()
    agg = agg.rename(columns={"methods":"csf", "rank":"target"})

    # Average metrics across runs/lr per (dataset, study, dropout, reward).
    feat_keys = ["dataset","study","dropout","reward"]
    feat_cols = hl_feats + NC_FEATURES
    feats = (metrics_join
             .groupby(feat_keys + ["architecture"], as_index=False)[feat_cols]
             .mean())
    feats = feats.groupby(feat_keys, as_index=False).first()

    full = agg.merge(feats[feat_keys + ["architecture"] + feat_cols],
                     on=feat_keys, how="inner")
    logger.info(f"final long-format rows: {len(full)} "
                f"(unique cells: {full[['csf','group']].drop_duplicates().shape[0]})")
    full.attrs["hl_features"] = hl_feats
    full.attrs["nc_features"] = NC_FEATURES
    return full


def cv_r2_spearman(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                   random_state: int = 0) -> tuple[float, float]:
    if len(y) < n_splits + 1 or X.shape[1] == 0:
        return float("nan"), float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    preds = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        if np.std(y[tr]) == 0:
            preds[te] = y[tr].mean()
            continue
        pipe = Pipeline([("s", StandardScaler()),
                         ("r", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sp, _ = spearmanr(preds, y)
    return r2, float(sp)


def run_regression_per_cell(full: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    hl_feats = full.attrs["hl_features"]
    nc_feats = full.attrs["nc_features"]
    rows: list[dict] = []
    for (csf, group), g in full.groupby(["csf","group"]):
        g = g.dropna(subset=["target"])
        # Drop columns with all-NaN within the cell
        Xhl = g[hl_feats].dropna(axis=1, how="all").fillna(g[hl_feats].mean())
        Xnc = g[nc_feats].dropna(axis=1, how="all").fillna(g[nc_feats].mean())
        y = g["target"].to_numpy(dtype=float)
        r2_hl, sp_hl = cv_r2_spearman(Xhl.to_numpy(dtype=float), y)
        r2_nc, sp_nc = cv_r2_spearman(Xnc.to_numpy(dtype=float), y)
        rows.append({
            "metric": metric_name,
            "csf": csf,
            "group": group,
            "n_samples": len(g),
            "n_hl_feats": Xhl.shape[1],
            "n_nc_feats": Xnc.shape[1],
            "r2_hl": r2_hl, "r2_nc": r2_nc,
            "delta_r2_hl_minus_nc": r2_hl - r2_nc,
            "spearman_hl": sp_hl, "spearman_nc": sp_nc,
            "delta_spearman_hl_minus_nc": sp_hl - sp_nc,
        })
    return pd.DataFrame(rows)


def make_summary(per_cell: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (metric, group), g in per_cell.groupby(["metric","group"]):
        out.append({
            "metric": metric, "group": group,
            "median_spearman_hl": g["spearman_hl"].median(),
            "median_spearman_nc": g["spearman_nc"].median(),
            "median_delta_spearman": g["delta_spearman_hl_minus_nc"].median(),
            "median_r2_hl": g["r2_hl"].median(),
            "median_r2_nc": g["r2_nc"].median(),
            "median_delta_r2": g["delta_r2_hl_minus_nc"].median(),
            "n_csfs": len(g),
            "n_csfs_hl_better_spearman":
                (g["delta_spearman_hl_minus_nc"] > 0).sum(),
        })
    return pd.DataFrame(out)


def make_box_plot(per_cell: pd.DataFrame, out_path: str,
                  score: str = "spearman") -> None:
    """Boxplot of NC vs HL predictive performance.

    score: "spearman" (primary) or "r2" (secondary). Spearman is the
    natural agreement score for the rank-valued target.
    """
    col_nc = f"{score}_nc"
    col_hl = f"{score}_hl"
    ylabel = "CV Spearman rho" if score == "spearman" else "CV R^2"
    metrics = sorted(per_cell["metric"].unique())
    groups = ["test","near","mid","far"]
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(4.0 * len(metrics), 4.5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = per_cell[per_cell["metric"] == metric]
        positions, data, labels = [], [], []
        for i, g in enumerate(groups):
            v_nc = sub[sub["group"] == g][col_nc].dropna().tolist()
            v_hl = sub[sub["group"] == g][col_hl].dropna().tolist()
            positions.extend([i*2.5, i*2.5 + 1.0])
            data.extend([v_nc, v_hl])
            labels.extend([f"NC\n{g}", f"HL\n{g}"])
        bp = ax.boxplot(data, positions=positions, widths=0.85,
                        patch_artist=True)
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor("#cccccc" if j % 2 == 0 else "#a6cde4")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".jpeg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.success(f"Saved boxplot: {out_path}.pdf")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    per_cell_all = []
    for metric in ["AUGRC","AURC"]:
        logger.info(f"=== {metric} ===")
        full = build_dataset(metric)
        per_cell = run_regression_per_cell(full, metric)
        per_cell_all.append(per_cell)
    per_cell = pd.concat(per_cell_all, ignore_index=True)
    per_cell.to_csv(os.path.join(OUTDIR, "per_cell.csv"), index=False)
    summary = make_summary(per_cell)
    summary.to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
    print("\n=== summary (median CV scores per metric x group, rank target) ===")
    print(summary.to_string(index=False))
    make_box_plot(per_cell, os.path.join(OUTDIR, "spearman_box"),
                  score="spearman")
    make_box_plot(per_cell, os.path.join(OUTDIR, "r2_box"), score="r2")


if __name__ == "__main__":
    main()

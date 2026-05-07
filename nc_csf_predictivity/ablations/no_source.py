"""Ablation: NC features only (drop `source` from categorical features).

Tests whether NC metrics alone carry enough information to distinguish
which CSFs to pick. The with-source models from steps 10–12 are the
baselines; this script retrains the same predictor families with `source`
removed from the feature set, on the headline splits (xarch, lopo).

Predictor changes:
  - Regression:        categoricals = ['csf', 'regime']        (was ['csf', 'source', 'regime'])
  - Multilabel binary: categoricals = ['regime']                (was ['source', 'regime'])
  - Per-CSF binary:    categoricals = ['regime']                (was ['source', 'regime'])

Label rule for binary heads: within_eps_rank only (the headline rule).

Comparison artifacts produced:
  outputs/ablations/no_source/track1/<split>/
    regression/preds.parquet
    multilabel_competitive/within_eps_rank/preds.parquet
    per_csf_binary/within_eps_rank/preds.parquet
    metrics/per_row.parquet, aggregate.parquet
    baselines/per_row.parquet, aggregate.parquet
  outputs/15_ablation_no_source.md
      side-by-side regret comparison vs the with-source headline numbers.

Baselines (always-X, random, oracle-on-train) don't depend on the predictor
features, so they're identical across with-source and no-source runs and
just re-attached for the comparison report.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats as scstats
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
CODE_DIR = PIPELINE_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"
ABLATION_ROOT = "ablations/no_source"

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
ALWAYS_BASELINES = ["MSR", "Energy", "MLS", "CTM", "fDBD", "NNGuide"]
SIDES = ["all", "head", "feature"]
TRAIN_REGIMES = ["near", "mid", "far", "all"]
SPLITS = ["xarch", "lopo"]
ALPHAS = np.logspace(-3, 3, 7)
LR_C = 1.0
LR_CS_PERCSF = 10
LR_CV_PERCSF = 5
N_BOOT = 2000
SEED = 0
ALPHA = 0.05


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def add_model_id(df: pd.DataFrame) -> pd.DataFrame:
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


# ---- Regression (no source) ----

def build_regression_pipeline_no_source() -> Pipeline:
    pre = ColumnTransformer([
        ("nc", StandardScaler(), NC_PRIMARY),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         ["csf", "regime"]),
    ])
    return Pipeline([("preprocess", pre), ("ridge", RidgeCV(alphas=ALPHAS))])


def run_regression_no_source(long_df: pd.DataFrame, split: str,
                             out_root: Path) -> int:
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    long_ood = long_df[long_df["regime"] != "test"]
    feat_cols = NC_PRIMARY + ["csf", "regime"]
    pieces = []
    for fold_id, fold_grp in sp.groupby("fold_id"):
        fold_label = fold_grp["fold_label"].iloc[0]
        train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
        test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
        tr = long_ood[long_ood["model_id"].isin(train_ids)]
        te = long_ood[long_ood["model_id"].isin(test_ids)]
        if tr.empty or te.empty:
            continue
        pipe = build_regression_pipeline_no_source()
        pipe.fit(tr[feat_cols], tr["augrc_rank"])
        preds = pipe.predict(te[feat_cols])
        out = te[["model_id", "eval_dataset", "regime", "csf"]].copy()
        out["split_name"] = split
        out["fold_id"] = fold_id
        out["fold_label"] = fold_label
        out["predicted_score"] = preds
        out["true_score"] = te["augrc_rank"].values
        out["raw_augrc"] = te["augrc"].values
        pieces.append(out)
    if not pieces:
        return 0
    out_dir = out_root / ABLATION_ROOT / "track1" / split / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_preds = pd.concat(pieces, ignore_index=True)
    write_parquet(all_preds, out_dir / "preds.parquet")
    return len(all_preds)


# ---- Binary (no source) ----

def build_binary_pipeline_no_source(per_csf: bool) -> Pipeline:
    pre = ColumnTransformer([
        ("nc", StandardScaler(), NC_PRIMARY),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         ["regime"]),
    ])
    if per_csf:
        base = LogisticRegressionCV(Cs=LR_CS_PERCSF, cv=LR_CV_PERCSF,
                                    max_iter=2000, scoring="neg_log_loss",
                                    n_jobs=1)
        return Pipeline([("preprocess", pre), ("clf", base)])
    base = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
    return Pipeline([("preprocess", pre),
                     ("clf", MultiOutputClassifier(base, n_jobs=1))])


def get_unique_models(long_df: pd.DataFrame) -> pd.DataFrame:
    keep = (["model_id", "architecture", "paradigm", "source", "run",
             "dropout", "reward"] + NC_PRIMARY)
    return long_df[keep].drop_duplicates(subset=["model_id"]).reset_index(drop=True)


def build_train_table(long_df: pd.DataFrame, label_df: pd.DataFrame,
                      train_model_ids: list[str]) -> tuple:
    nc_per_model = get_unique_models(long_df)
    pool = nc_per_model[(nc_per_model["architecture"] == "VGG13")
                        & nc_per_model["model_id"].isin(train_model_ids)]
    if pool.empty:
        return None, None, None, []
    rows = []
    for _, m in pool.iterrows():
        for regime in TRAIN_REGIMES:
            r = m.to_dict()
            r["regime"] = regime
            rows.append(r)
    feats = pd.DataFrame(rows)
    label_wide = (label_df.pivot_table(
        index=["paradigm", "source", "dropout", "reward", "regime"],
        columns="csf", values="label", aggfunc="first").reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns
                if c not in ["paradigm", "source", "dropout", "reward", "regime"]]
    label_wide[csf_cols] = label_wide[csf_cols].astype(int)
    merged = feats.merge(label_wide,
                         on=["paradigm", "source", "dropout", "reward", "regime"],
                         how="inner")
    if merged.empty:
        return None, None, None, []
    X = merged[NC_PRIMARY + ["regime"]]  # NO source
    Y = merged[csf_cols]
    meta = merged[["model_id", "architecture", "paradigm", "source",
                   "run", "dropout", "reward", "regime"]]
    return X, Y, meta, csf_cols


def build_test_table(long_df: pd.DataFrame, test_model_ids: list[str]) -> tuple:
    nc_per_model = get_unique_models(long_df)
    pool = nc_per_model[nc_per_model["model_id"].isin(test_model_ids)]
    rows = []
    for _, m in pool.iterrows():
        for regime in TRAIN_REGIMES:
            r = m.to_dict()
            r["regime"] = regime
            rows.append(r)
    feats = pd.DataFrame(rows)
    if feats.empty:
        return None, None
    X = feats[NC_PRIMARY + ["regime"]]
    meta = feats[["model_id", "architecture", "paradigm", "source",
                  "run", "dropout", "reward", "regime"]]
    return X, meta


def run_binary_no_source(long_df: pd.DataFrame, label_df: pd.DataFrame,
                         split: str, out_root: Path,
                         per_csf: bool) -> int:
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    pieces_preds = []
    for fold_id, fold_grp in sp.groupby("fold_id"):
        fold_label = fold_grp["fold_label"].iloc[0]
        train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
        test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
        X_tr, Y_tr, _, csf_cols = build_train_table(long_df, label_df, train_ids)
        X_te, meta_te = build_test_table(long_df, test_ids)
        if X_tr is None or X_te is None or not csf_cols:
            continue
        keep = [c for c in csf_cols if Y_tr[c].nunique() > 1]
        if not keep:
            continue
        Y_tr = Y_tr[keep]

        if per_csf:
            for csf in keep:
                y = Y_tr[csf].values
                counts = np.bincount(y, minlength=2)
                if min(counts) < LR_CV_PERCSF:
                    continue
                try:
                    pipe = build_binary_pipeline_no_source(per_csf=True)
                    pipe.fit(X_tr, y)
                    proba = pipe.predict_proba(X_te)[:, 1]
                except (ValueError, RuntimeError):
                    continue
                pred = pd.DataFrame({
                    "model_id": meta_te["model_id"].values,
                    "regime": meta_te["regime"].values,
                    "csf": csf,
                    "p_competitive": proba,
                    "predicted_competitive": proba >= 0.5,
                    "split_name": split,
                    "fold_id": fold_id,
                    "fold_label": fold_label,
                    "label_rule": "within_eps_rank",
                })
                pieces_preds.append(pred)
        else:
            pipe = build_binary_pipeline_no_source(per_csf=False)
            pipe.fit(X_tr, Y_tr.values)
            proba_per_csf = pipe.predict_proba(X_te)
            for j, csf in enumerate(keep):
                p1 = proba_per_csf[j][:, 1]
                pred = pd.DataFrame({
                    "model_id": meta_te["model_id"].values,
                    "regime": meta_te["regime"].values,
                    "csf": csf,
                    "p_competitive": p1,
                    "predicted_competitive": p1 >= 0.5,
                    "split_name": split,
                    "fold_id": fold_id,
                    "fold_label": fold_label,
                    "label_rule": "within_eps_rank",
                })
                pieces_preds.append(pred)
    if not pieces_preds:
        return 0
    head_dir = "per_csf_binary" if per_csf else "multilabel_competitive"
    out_dir = (out_root / ABLATION_ROOT / "track1" / split
               / head_dir / "within_eps_rank")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_preds = pd.concat(pieces_preds, ignore_index=True)
    write_parquet(all_preds, out_dir / "preds.parquet")
    return len(all_preds)


# ---- Metrics + comparison ----

def filter_to_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if side == "all":
        return df
    if side == "head":
        return df[df["csf"].isin(HEAD_SIDE_CSFS)]
    if side == "feature":
        return df[df["csf"].isin(FEATURE_SIDE_CSFS)]
    raise ValueError(side)


def oracle_for_side(oracle: pd.DataFrame, side: str) -> pd.DataFrame:
    suffix = "" if side == "all" else f"_{side}"
    return oracle.assign(
        o_csf=oracle[f"oracle_csf{suffix}"],
        o_augrc=oracle[f"oracle_augrc{suffix}"],
        w_csf=oracle[f"worst_csf{suffix}"],
        w_augrc=oracle[f"worst_augrc{suffix}"],
    )


def regression_per_row(preds: pd.DataFrame, oracle_side: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame()
    p = preds.sort_values(["model_id", "eval_dataset", "predicted_score"]).copy()
    p["pred_rank"] = p.groupby(["model_id", "eval_dataset"]).cumcount() + 1
    top1 = p[p["pred_rank"] == 1][["model_id", "eval_dataset", "csf", "raw_augrc"]]
    top1 = top1.rename(columns={"csf": "top1_csf", "raw_augrc": "top1_augrc"})
    base = oracle_side[["model_id", "eval_dataset", "regime",
                        "o_csf", "o_augrc", "w_csf", "w_augrc"]]
    metrics = base.merge(top1, on=["model_id", "eval_dataset"], how="inner")
    if metrics.empty:
        return pd.DataFrame()
    metrics["top1_regret_raw"] = metrics["top1_augrc"] - metrics["o_augrc"]
    denom = (metrics["w_augrc"] - metrics["o_augrc"]).replace(0, np.nan)
    metrics["top1_regret_norm"] = metrics["top1_regret_raw"] / denom
    rows = []
    for (uid, ev), g in p.groupby(["model_id", "eval_dataset"]):
        if len(g) < 2:
            continue
        rho, _ = spearmanr(g["predicted_score"].values, g["raw_augrc"].values)
        ocsf = base[(base["model_id"] == uid) & (base["eval_dataset"] == ev)]["o_csf"]
        if ocsf.empty:
            continue
        ocsf = ocsf.iloc[0]
        ranks = g.set_index("csf")["pred_rank"]
        mrr = (1.0 / float(ranks.loc[ocsf])) if ocsf in ranks.index else np.nan
        rows.append({"model_id": uid, "eval_dataset": ev,
                     "spearman_rho": rho, "mrr": mrr})
    rk = pd.DataFrame(rows)
    if not rk.empty:
        metrics = metrics.merge(rk, on=["model_id", "eval_dataset"], how="left")
    return metrics


def binary_per_row(preds_bin: pd.DataFrame, eval_long: pd.DataFrame,
                   oracle_side: pd.DataFrame) -> pd.DataFrame:
    if preds_bin.empty or eval_long.empty:
        return pd.DataFrame()
    test_ids = preds_bin["model_id"].unique()
    eval_long = eval_long[eval_long["model_id"].isin(test_ids)]
    oracle_side = oracle_side[oracle_side["model_id"].isin(test_ids)]
    joined = eval_long.merge(preds_bin[["model_id", "regime", "csf",
                                         "predicted_competitive"]],
                             on=["model_id", "regime", "csf"], how="left")
    joined["predicted_competitive"] = joined["predicted_competitive"].fillna(False)
    set_size = (joined[joined["predicted_competitive"]]
                .groupby(["model_id", "eval_dataset"])["csf"].nunique()
                .rename("set_size").reset_index())
    set_min = (joined[joined["predicted_competitive"]]
               .groupby(["model_id", "eval_dataset"])["raw_augrc"].min()
               .rename("set_min_augrc").reset_index())
    base = oracle_side[["model_id", "eval_dataset", "regime",
                        "o_csf", "o_augrc", "w_csf", "w_augrc"]]
    metrics = (base.merge(set_min, on=["model_id", "eval_dataset"], how="left")
                   .merge(set_size, on=["model_id", "eval_dataset"], how="left"))
    metrics["set_size"] = metrics["set_size"].fillna(0).astype(int)
    metrics["empty_set"] = metrics["set_size"] == 0
    metrics["set_regret_raw"] = (metrics["set_min_augrc"] - metrics["o_augrc"]).clip(lower=0)
    metrics["set_regret_raw_imputed"] = metrics["set_regret_raw"].fillna(metrics["w_augrc"] - metrics["o_augrc"])
    return metrics


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def aggregate_per_row(per_row_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["predictor", "label_rule", "regime", "side"]
    metric_cols = ["top1_regret_raw", "top1_regret_norm",
                   "set_regret_raw", "set_regret_raw_imputed",
                   "set_size", "spearman_rho", "mrr"]
    for keyset, g in per_row_long.groupby(keys, dropna=False):
        rec = dict(zip(keys, keyset))
        rec["n"] = len(g)
        for m in metric_cols:
            if m not in g.columns or g[m].dropna().empty:
                continue
            mean, lo, hi = bootstrap_mean_ci(g[m].astype(float).values)
            rec[f"{m}_mean"] = mean
            rec[f"{m}_ci_lo"] = lo
            rec[f"{m}_ci_hi"] = hi
        if "empty_set" in g.columns:
            rec["empty_set_share"] = float(g["empty_set"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_metrics_no_source(out_root: Path, split: str,
                              long_df: pd.DataFrame, oracle_df: pd.DataFrame
                              ) -> pd.DataFrame:
    eval_long = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})

    pieces = []
    for predictor, head_dir, label_rule in [
        ("regression", "regression", ""),
        ("multilabel", "multilabel_competitive", "within_eps_rank"),
        ("per_csf_binary", "per_csf_binary", "within_eps_rank"),
    ]:
        if predictor == "regression":
            p_path = out_root / ABLATION_ROOT / "track1" / split / "regression" / "preds.parquet"
        else:
            p_path = out_root / ABLATION_ROOT / "track1" / split / head_dir / label_rule / "preds.parquet"
        if not p_path.exists():
            continue
        preds = pq.read_table(p_path).to_pandas()
        for side in SIDES:
            if predictor == "regression":
                preds_side = filter_to_side(preds, side)
                m = regression_per_row(preds_side, oracle_for_side(oracle_df, side))
            else:
                eval_side = filter_to_side(eval_long, side)
                m = binary_per_row(preds, eval_side, oracle_for_side(oracle_df, side))
            if m.empty:
                continue
            m["predictor"] = predictor
            m["label_rule"] = label_rule
            m["side"] = side
            pieces.append(m)
    if not pieces:
        return pd.DataFrame()
    all_rows = pd.concat(pieces, ignore_index=True)
    out_dir = out_root / ABLATION_ROOT / "track1" / split / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(all_rows, out_dir / "per_row.parquet")
    agg = aggregate_per_row(all_rows)
    write_parquet(agg, out_dir / "aggregate.parquet")
    return agg


# ---- Comparison report ----

def with_source_aggregate(out_root: Path, split: str) -> pd.DataFrame:
    """Reload the with-source aggregates for the headline rule (within_eps_rank)
    + regression."""
    p = out_root / "track1" / split / "metrics" / "aggregate.parquet"
    if not p.exists():
        return pd.DataFrame()
    ag = pq.read_table(p).to_pandas()
    keep = ag[(ag["predictor"] == "regression")
              | ((ag["label_rule"] == "within_eps_rank")
                 & ag["predictor"].isin(["multilabel", "per_csf_binary"]))]
    return keep


def report_comparison(out_root: Path, out_path: Path) -> None:
    lines = ["# Ablation — NC-only features (drop `source`)\n\n"]
    lines.append("**Date:** 2026-05-04\n")
    lines.append("**Source:** `code/nc_csf_predictivity/ablations/no_source.py`\n")
    lines.append("**Hypothesis:** NC metrics alone carry enough information to "
                 "predict CSF competitive sets. If true, removing `source` from "
                 "the categorical features should not materially hurt the "
                 "predictor's regret on held-out architectures or paradigms.\n\n")
    lines.append("**Comparison:** with-source numbers come from the headline "
                 "step-13 metrics (regression top-1 + within_eps_rank for both "
                 "binary heads). No-source numbers come from this script's "
                 "retrained predictors on the same splits.\n\n")

    for split in SPLITS:
        no_src_path = out_root / ABLATION_ROOT / "track1" / split / "metrics" / "aggregate.parquet"
        if not no_src_path.exists():
            continue
        no_src = pq.read_table(no_src_path).to_pandas()
        with_src = with_source_aggregate(out_root, split)

        lines.append(f"## {split}\n\n")
        for side in SIDES:
            lines.append(f"### `side = {side}`\n\n")
            rows = []
            for regime in ["near", "mid", "far"]:
                for predictor, label, metric_col in [
                    ("regression", "—", "top1_regret_raw"),
                    ("multilabel", "within_eps_rank", "set_regret_raw_imputed"),
                    ("per_csf_binary", "within_eps_rank", "set_regret_raw_imputed"),
                ]:
                    sel_no = no_src[(no_src["predictor"] == predictor)
                                    & (no_src["regime"] == regime)
                                    & (no_src["side"] == side)
                                    & (no_src["label_rule"] == label.replace("—", ""))]
                    sel_w = with_src[(with_src["predictor"] == predictor)
                                     & (with_src["regime"] == regime)
                                     & (with_src["side"] == side)]
                    if predictor != "regression":
                        sel_w = sel_w[sel_w["label_rule"] == label]
                    no_val = float(sel_no[f"{metric_col}_mean"].iloc[0]) if not sel_no.empty and f"{metric_col}_mean" in sel_no.columns else np.nan
                    w_metric_col = "top1_regret_raw" if predictor == "regression" else "set_regret_raw"
                    # For with-source binary, use raw set_regret (step 13 already
                    # includes both raw and imputed; for fair comparison, use the
                    # same imputed concept — but step 13 only has raw)
                    if predictor == "regression":
                        w_val = float(sel_w[f"{w_metric_col}_mean"].iloc[0]) if not sel_w.empty and f"{w_metric_col}_mean" in sel_w.columns else np.nan
                    else:
                        # Use the raw set_regret (non-imputed) from with-source for
                        # comparability with the no-source raw set_regret column.
                        # The imputed comparison can be made against step-14 baselines
                        # which we don't reload here.
                        w_val = float(sel_w["set_regret_raw_mean"].iloc[0]) if not sel_w.empty and "set_regret_raw_mean" in sel_w.columns else np.nan
                    delta = no_val - w_val if not np.isnan(no_val) and not np.isnan(w_val) else np.nan
                    rows.append({
                        "regime": regime,
                        "predictor": f"{predictor}/{label}" if label != "—" else predictor,
                        "with_source": round(w_val, 3) if not np.isnan(w_val) else None,
                        "no_source": round(no_val, 3) if not np.isnan(no_val) else None,
                        "delta_no_minus_with": round(delta, 3) if not np.isnan(delta) else None,
                    })
            df = pd.DataFrame(rows)
            lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")

        # Spearman ρ for regression specifically (the ranking metric)
        lines.append("### Spearman ρ side-asymmetry — no-source\n\n")
        rg = no_src[(no_src["predictor"] == "regression")
                    & no_src["regime"].isin(["near", "mid", "far"])]
        if "spearman_rho_mean" in rg.columns:
            sp = rg[["regime", "side", "n", "spearman_rho_mean",
                     "spearman_rho_ci_lo", "spearman_rho_ci_hi"]].round(3)
            sp = sp.sort_values(["regime", "side"])
            lines.append("```\n" + sp.to_string(index=False) + "\n```\n\n")
        # Spearman ρ for regression with-source for direct comparison
        lines.append("### Spearman ρ side-asymmetry — with-source (reference)\n\n")
        rg2 = with_src[(with_src["predictor"] == "regression")
                       & with_src["regime"].isin(["near", "mid", "far"])]
        if not rg2.empty and "spearman_rho_mean" in rg2.columns:
            sp2 = rg2[["regime", "side", "n", "spearman_rho_mean",
                       "spearman_rho_ci_lo", "spearman_rho_ci_hi"]].round(3)
            sp2 = sp2.sort_values(["regime", "side"])
            lines.append("```\n" + sp2.to_string(index=False) + "\n```\n\n")

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    print("Loading data ...")
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_model_id(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    # Load within_eps_rank labels
    within = pq.read_table(out_root / "track1" / "labels" / "within_eps.parquet").to_pandas()
    label_df = (within[["paradigm", "source", "dropout", "reward", "regime",
                        "csf", "in_within_eps_set_rank"]]
                .rename(columns={"in_within_eps_set_rank": "label"})
                .copy())
    label_df["label"] = label_df["label"].astype(int)

    for split in SPLITS:
        print(f"\n=== {split} ===")
        n = run_regression_no_source(long_df, split, out_root)
        print(f"  regression: {n:,} pred rows")
        n = run_binary_no_source(long_df, label_df, split, out_root, per_csf=False)
        print(f"  multilabel/within_eps_rank: {n:,} pred rows")
        n = run_binary_no_source(long_df, label_df, split, out_root, per_csf=True)
        print(f"  per_csf_binary/within_eps_rank: {n:,} pred rows")
        agg = compute_metrics_no_source(out_root, split, long_df, oracle_df)
        print(f"  metrics aggregated: {len(agg)} rows")

    report_comparison(out_root, out_root / "15_ablation_no_source.md")
    print(f"\nwrote {out_root / '15_ablation_no_source.md'}")


if __name__ == "__main__":
    main()

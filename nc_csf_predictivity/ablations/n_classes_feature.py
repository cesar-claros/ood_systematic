"""Ablation: replace `source` one-hot with `n_classes` as a scaled ordinal.

n_classes mapping (per Jaeger's scorecard for SuperCIFAR-100):
  cifar10        : 10
  supercifar100  : 19
  cifar100       : 100
  tinyimagenet   : 200

Hypothesis: number of classes is a principled ordinal property of the
training dataset that captures source-induced calibration shifts more
generalizably than a one-hot source identity. With n_classes as a single
numeric feature, the predictor can in principle extrapolate to source
datasets it never saw at training time (e.g., a new K-class problem).

Predictors:
  Per-CSF binary (LogisticRegressionCV) with label_rule = within_eps_rank.
  Multilabel binary (LogisticRegression(C=1.0)) with same label rule.

Splits:
  xarch (cross-arch headline), lopo (cross-paradigm headline).

Comparison: vs with-source baseline and NC-only ablation.

Outputs:
  outputs/ablations/n_classes/track1/<split>/
    multilabel_competitive/within_eps_rank/preds.parquet, coefficients.parquet
    per_csf_binary/within_eps_rank/preds.parquet, coefficients.parquet
    metrics/per_row.parquet, aggregate.parquet
  outputs/16_ablation_n_classes.md
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"
ABLATION_ROOT = "ablations/n_classes"

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

N_CLASSES_MAP = {
    "cifar10": 10,
    "supercifar100": 19,   # Jaeger's scorecard convention
    "cifar100": 100,
    "tinyimagenet": 200,
}

SIDES = ["all", "head", "feature"]
TRAIN_REGIMES = ["near", "mid", "far", "all"]
SPLITS = ["xarch", "lopo"]
LR_C = 1.0
LR_CS_PERCSF = 10
LR_CV_PERCSF = 5
N_BOOT = 2000
SEED = 0


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


def add_n_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["n_classes"] = df["source"].map(N_CLASSES_MAP).astype(float)
    return df


def build_pipeline(per_csf: bool) -> Pipeline:
    pre = ColumnTransformer([
        ("nc", StandardScaler(), NC_PRIMARY),
        ("nclass", StandardScaler(), ["n_classes"]),
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
    keep = (["model_id", "architecture", "paradigm", "source", "n_classes",
             "run", "dropout", "reward"] + NC_PRIMARY)
    return long_df[keep].drop_duplicates(subset=["model_id"]).reset_index(drop=True)


def build_train(long_df: pd.DataFrame, label_df: pd.DataFrame,
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
    X = merged[NC_PRIMARY + ["n_classes", "regime"]]
    Y = merged[csf_cols]
    meta = merged[["model_id", "architecture", "paradigm", "source",
                   "run", "dropout", "reward", "regime"]]
    return X, Y, meta, csf_cols


def build_test(long_df: pd.DataFrame, test_model_ids: list[str]) -> tuple:
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
    X = feats[NC_PRIMARY + ["n_classes", "regime"]]
    meta = feats[["model_id", "architecture", "paradigm", "source",
                  "run", "dropout", "reward", "regime"]]
    return X, meta


def run_binary(long_df: pd.DataFrame, label_df: pd.DataFrame,
               split: str, out_root: Path, per_csf: bool) -> int:
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    pieces = []
    coef_pieces = []
    for fold_id, fold_grp in sp.groupby("fold_id"):
        fold_label = fold_grp["fold_label"].iloc[0]
        train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
        test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
        X_tr, Y_tr, _, csf_cols = build_train(long_df, label_df, train_ids)
        X_te, meta_te = build_test(long_df, test_ids)
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
                    pipe = build_pipeline(per_csf=True)
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
                pieces.append(pred)
                clf = pipe.named_steps["clf"]
                # feature names after preprocess: 8 NC + n_classes + regime one-hots
                pre = pipe.named_steps["preprocess"]
                regime_names = list(pre.named_transformers_["cat"].get_feature_names_out(["regime"]))
                feat_names = NC_PRIMARY + ["n_classes"] + regime_names
                coef_dict = {"(intercept)": float(clf.intercept_[0])}
                for fname, c in zip(feat_names, clf.coef_[0]):
                    coef_dict[fname] = float(c)
                coef_dict["chosen_C"] = float(clf.C_[0])
                row = {"csf": csf, "fold_id": fold_id, "fold_label": fold_label,
                       "split_name": split, "label_rule": "within_eps_rank"}
                for fname, c in coef_dict.items():
                    row2 = {**row, "feature": fname, "coefficient": c}
                    coef_pieces.append(row2)
        else:
            pipe = build_pipeline(per_csf=False)
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
                pieces.append(pred)

    if not pieces:
        return 0
    head_dir = "per_csf_binary" if per_csf else "multilabel_competitive"
    out_dir = (out_root / ABLATION_ROOT / "track1" / split
               / head_dir / "within_eps_rank")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_preds = pd.concat(pieces, ignore_index=True)
    write_parquet(all_preds, out_dir / "preds.parquet")
    if coef_pieces:
        write_parquet(pd.DataFrame(coef_pieces), out_dir / "coefficients.parquet")
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


def aggregate(per_row_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["predictor", "label_rule", "regime", "side"]
    metric_cols = ["set_regret_raw", "set_regret_raw_imputed", "set_size"]
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


def compute_metrics(out_root: Path, split: str,
                    long_df: pd.DataFrame, oracle_df: pd.DataFrame
                    ) -> pd.DataFrame:
    eval_long = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})

    pieces = []
    for predictor, head_dir in [
        ("multilabel", "multilabel_competitive"),
        ("per_csf_binary", "per_csf_binary"),
    ]:
        p_path = (out_root / ABLATION_ROOT / "track1" / split / head_dir
                  / "within_eps_rank" / "preds.parquet")
        if not p_path.exists():
            continue
        preds = pq.read_table(p_path).to_pandas()
        for side in SIDES:
            eval_side = filter_to_side(eval_long, side)
            m = binary_per_row(preds, eval_side, oracle_for_side(oracle_df, side))
            if m.empty:
                continue
            m["predictor"] = predictor
            m["label_rule"] = "within_eps_rank"
            m["side"] = side
            pieces.append(m)
    if not pieces:
        return pd.DataFrame()
    all_rows = pd.concat(pieces, ignore_index=True)
    out_dir = out_root / ABLATION_ROOT / "track1" / split / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(all_rows, out_dir / "per_row.parquet")
    agg = aggregate(all_rows)
    write_parquet(agg, out_dir / "aggregate.parquet")
    return agg


# ---- Comparison report ----

def with_source_aggregate(out_root: Path, split: str) -> pd.DataFrame:
    p = out_root / "track1" / split / "metrics" / "aggregate.parquet"
    if not p.exists():
        return pd.DataFrame()
    ag = pq.read_table(p).to_pandas()
    return ag[(ag["label_rule"] == "within_eps_rank")
              & ag["predictor"].isin(["multilabel", "per_csf_binary"])]


def no_source_aggregate(out_root: Path, split: str) -> pd.DataFrame:
    p = out_root / "ablations" / "no_source" / "track1" / split / "metrics" / "aggregate.parquet"
    if not p.exists():
        return pd.DataFrame()
    ag = pq.read_table(p).to_pandas()
    return ag[(ag["label_rule"] == "within_eps_rank")
              & ag["predictor"].isin(["multilabel", "per_csf_binary"])]


def report_comparison(out_root: Path, out_path: Path) -> None:
    lines = ["# Ablation — `n_classes` ordinal instead of `source` one-hot\n\n"]
    lines.append("**Date:** 2026-05-04\n")
    lines.append("**Source:** `code/nc_csf_predictivity/ablations/n_classes_feature.py`\n")
    lines.append("**Mapping:** cifar10 → 10, supercifar100 → 19, cifar100 → 100, "
                 "tinyimagenet → 200 (Jaeger's scorecard convention).\n")
    lines.append("**Comparison:** vs with-source one-hot (step 11/12) and "
                 "vs NC-only (step 18 ablation).\n\n")
    lines.append("Headline metric: `set_regret_raw_imputed` (always-predicts via "
                 "empty-set imputation; fair vs always-X baselines).\n\n")

    for split in SPLITS:
        nc_path = out_root / ABLATION_ROOT / "track1" / split / "metrics" / "aggregate.parquet"
        if not nc_path.exists():
            continue
        n_cls = pq.read_table(nc_path).to_pandas()
        with_src = with_source_aggregate(out_root, split)
        no_src = no_source_aggregate(out_root, split)

        lines.append(f"## {split}\n\n")
        for predictor in ["per_csf_binary", "multilabel"]:
            lines.append(f"### `predictor = {predictor}`\n\n")
            for side in SIDES:
                rows = []
                for regime in ["near", "mid", "far"]:
                    sel_n = n_cls[(n_cls["predictor"] == predictor)
                                  & (n_cls["regime"] == regime)
                                  & (n_cls["side"] == side)]
                    sel_w = with_src[(with_src["predictor"] == predictor)
                                     & (with_src["regime"] == regime)
                                     & (with_src["side"] == side)]
                    sel_o = no_src[(no_src["predictor"] == predictor)
                                   & (no_src["regime"] == regime)
                                   & (no_src["side"] == side)]
                    n_val = float(sel_n["set_regret_raw_imputed_mean"].iloc[0]) if not sel_n.empty and "set_regret_raw_imputed_mean" in sel_n.columns else np.nan
                    # with-source uses raw set_regret (step 13 doesn't have imputed in same row)
                    w_val = float(sel_w["set_regret_raw_mean"].iloc[0]) if not sel_w.empty and "set_regret_raw_mean" in sel_w.columns else np.nan
                    o_val = float(sel_o["set_regret_raw_imputed_mean"].iloc[0]) if not sel_o.empty and "set_regret_raw_imputed_mean" in sel_o.columns else np.nan
                    rows.append({
                        "regime": regime, "side": side,
                        "with_source_raw": round(w_val, 3) if not np.isnan(w_val) else None,
                        "n_classes_imputed": round(n_val, 3) if not np.isnan(n_val) else None,
                        "no_source_imputed": round(o_val, 3) if not np.isnan(o_val) else None,
                        "Δ(n_classes − with_source)": round(n_val - w_val, 3)
                            if not np.isnan(n_val) and not np.isnan(w_val) else None,
                        "Δ(n_classes − no_source)": round(n_val - o_val, 3)
                            if not np.isnan(n_val) and not np.isnan(o_val) else None,
                    })
                df = pd.DataFrame(rows)
                lines.append(f"`side = {side}`\n\n")
                lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")

        # Also report per-CSF chosen C distribution under n_classes for one split
        cp = (out_root / ABLATION_ROOT / "track1" / split / "per_csf_binary"
              / "within_eps_rank" / "coefficients.parquet")
        if cp.exists():
            coefs = pq.read_table(cp).to_pandas()
            cc = coefs[coefs["feature"] == "(intercept)"][["csf"]].drop_duplicates()
            chosen = (coefs[coefs["feature"] == "(intercept)"]
                      .merge(coefs[coefs["feature"] == "n_classes"][["csf", "fold_id", "coefficient"]]
                             .rename(columns={"coefficient": "n_classes_coef"}),
                             on=["csf", "fold_id"], how="left"))
            chosen_c = (coefs[coefs["feature"] == "(intercept)"]
                        [["csf", "fold_id"]]
                        .merge(coefs.drop_duplicates(["csf", "fold_id"])
                               [["csf", "fold_id"]], on=["csf", "fold_id"]))
            # Get the chosen_C from a row that has it (may not be in all per-CSF rows)
            # Simpler: pull from the n_classes coefficient entry
            n_cls_coefs = (coefs[coefs["feature"] == "n_classes"]
                           [["csf", "coefficient"]]
                           .rename(columns={"coefficient": "n_classes_coef"})
                           .round(3))
            lines.append(f"### `per_csf_binary` n_classes coefficient per CSF ({split})\n\n")
            lines.append("On the standardized n_classes scale (so the coefficient "
                         "is the change in log-odds per 1 standard deviation increase "
                         "in n_classes across the training pool).\n\n")
            n_cls_coefs = n_cls_coefs.sort_values("n_classes_coef")
            lines.append("```\n" + n_cls_coefs.to_string(index=False) + "\n```\n\n")

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    print("Loading data ...")
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    long_df = add_n_classes(long_df)
    oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_model_id(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    within = pq.read_table(out_root / "track1" / "labels" / "within_eps.parquet").to_pandas()
    label_df = (within[["paradigm", "source", "dropout", "reward", "regime",
                        "csf", "in_within_eps_set_rank"]]
                .rename(columns={"in_within_eps_set_rank": "label"})
                .copy())
    label_df["label"] = label_df["label"].astype(int)

    for split in SPLITS:
        print(f"\n=== {split} ===")
        n = run_binary(long_df, label_df, split, out_root, per_csf=False)
        print(f"  multilabel/within_eps_rank: {n:,} pred rows")
        n = run_binary(long_df, label_df, split, out_root, per_csf=True)
        print(f"  per_csf_binary/within_eps_rank: {n:,} pred rows")
        agg = compute_metrics(out_root, split, long_df, oracle_df)
        print(f"  metrics aggregated: {len(agg)} rows")

    report_comparison(out_root, out_root / "16_ablation_n_classes.md")
    print(f"\nwrote {out_root / '16_ablation_n_classes.md'}")


if __name__ == "__main__":
    main()

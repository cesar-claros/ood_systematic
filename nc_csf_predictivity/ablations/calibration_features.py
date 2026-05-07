"""Calibrated (b+c) per-CSF binary across three context-feature options.

Holds the calibration fixes constant (class_weight='balanced' + per-arch
NC standardization, the winning variant from the previous ablation) and
varies only the dataset-context features:

  source     : 8 NC + source one-hot + regime one-hot   (12 features)
  n_classes  : 8 NC + n_classes (scaled) + regime       (10 features)
  none       : 8 NC + regime                            ( 9 features)

Goal: identify which dataset-context signal NC actually needs once
calibration is fixed. If `none` performs comparably to `source`, the source
covariate isn't carrying real signal — it's just compensating for class
imbalance and distribution shift, which (b+c) already handles.

All variants use:
  - L2 logistic regression, LogisticRegressionCV(Cs=50, cv=5)
  - class_weight='balanced'
  - NC features pre-standardized per architecture (passthrough in pipeline)
  - within_eps_rank labels, VGG13 training pool, splits xarch + lopo

Outputs:
  outputs/ablations/calib_features/track1/<split>/<config>/
    preds.parquet
    coefficients.parquet
  outputs/22_ablation_calib_features.md
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"
ABLATION_ROOT = "ablations/calib_features"

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]
HEAD_SIDE_CSFS = [
    "Confidence", "Energy", "GE", "GEN", "GradNorm",
    "MLS", "MSR", "PCE", "PE", "REN", "pNML",
]
FEATURE_SIDE_CSFS = [
    "CTM", "fDBD", "KPCA RecError global", "Maha", "NeCo",
    "NNGuide", "PCA RecError global", "Residual", "ViM",
]
N_CLASSES_MAP = {
    "cifar10": 10, "supercifar100": 19, "cifar100": 100, "tinyimagenet": 200,
}
SIDES = ["all", "head", "feature"]
TRAIN_REGIMES = ["near", "mid", "far", "all"]
SPLITS = ["xarch", "lopo"]
CONFIGS = ["source", "n_classes", "none"]
L2_CS = 50
L2_CV = 5
L2_MAX_ITER = 2000
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


def standardize_per_architecture(long_df: pd.DataFrame, nc_cols: list[str]
                                 ) -> pd.DataFrame:
    out = long_df.copy()
    for arch, sub in long_df.groupby("architecture"):
        for col in nc_cols:
            mean = sub[col].mean()
            std = sub[col].std()
            if std > 0:
                out.loc[sub.index, col] = (sub[col] - mean) / std
            else:
                out.loc[sub.index, col] = 0.0
    return out


def build_pipeline(config: str) -> Pipeline:
    """NC features always passthrough (pre-standardized per architecture).
    Categoricals depend on config."""
    transformers = [("nc", "passthrough", NC_PRIMARY)]
    if config == "source":
        transformers.append((
            "cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ["source", "regime"]
        ))
    elif config == "n_classes":
        transformers.append(("nclass", StandardScaler(), ["n_classes"]))
        transformers.append((
            "cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ["regime"]
        ))
    elif config == "none":
        transformers.append((
            "cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ["regime"]
        ))
    else:
        raise ValueError(config)
    pre = ColumnTransformer(transformers)
    base = LogisticRegressionCV(
        Cs=L2_CS, cv=L2_CV, penalty="l2", solver="lbfgs",
        max_iter=L2_MAX_ITER, scoring="neg_log_loss", n_jobs=1,
        class_weight="balanced",
    )
    return Pipeline([("preprocess", pre), ("clf", base)])


def feature_columns_for_config(config: str) -> list[str]:
    if config == "source":
        return NC_PRIMARY + ["source", "regime"]
    if config == "n_classes":
        return NC_PRIMARY + ["n_classes", "regime"]
    if config == "none":
        return NC_PRIMARY + ["regime"]
    raise ValueError(config)


def feature_names_after_preprocess(pipe: Pipeline, config: str) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    nc_names = list(pre.transformers_[0][2])
    extras = []
    if config == "source":
        ohe = pre.named_transformers_["cat"]
        extras = list(ohe.get_feature_names_out(["source", "regime"]))
    elif config == "n_classes":
        extras = list(pre.transformers_[1][2])
        ohe = pre.named_transformers_["cat"]
        extras = extras + list(ohe.get_feature_names_out(["regime"]))
    elif config == "none":
        ohe = pre.named_transformers_["cat"]
        extras = list(ohe.get_feature_names_out(["regime"]))
    return nc_names + extras


def get_unique_models(long_df: pd.DataFrame, config: str) -> pd.DataFrame:
    extra = ["n_classes"] if config == "n_classes" else []
    keep = (["model_id", "architecture", "paradigm", "source", "run",
             "dropout", "reward"] + NC_PRIMARY + extra)
    return long_df[keep].drop_duplicates(subset=["model_id"]).reset_index(drop=True)


def build_train(long_df, label_df, train_model_ids, config):
    nc_per_model = get_unique_models(long_df, config)
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
    X = merged[feature_columns_for_config(config)]
    Y = merged[csf_cols]
    meta = merged[["model_id", "architecture", "paradigm", "source",
                   "run", "dropout", "reward", "regime"]]
    return X, Y, meta, csf_cols


def build_test(long_df, test_model_ids, config):
    nc_per_model = get_unique_models(long_df, config)
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
    X = feats[feature_columns_for_config(config)]
    meta = feats[["model_id", "architecture", "paradigm", "source",
                  "run", "dropout", "reward", "regime"]]
    return X, meta


def fit_one_csf(X_tr, y_tr, X_te, config):
    counts = np.bincount(y_tr, minlength=2)
    if min(counts) < L2_CV:
        return None
    try:
        pipe = build_pipeline(config)
        pipe.fit(X_tr, y_tr)
    except (ValueError, RuntimeError):
        return None
    proba = pipe.predict_proba(X_te)[:, 1]
    clf = pipe.named_steps["clf"]
    feat_names = feature_names_after_preprocess(pipe, config)
    chosen_C = float(clf.C_[0])
    coef_dict = {"(intercept)": float(clf.intercept_[0]), "_chosen_C": chosen_C}
    for fname, c in zip(feat_names, clf.coef_[0]):
        coef_dict[fname] = float(c)
    return proba, coef_dict, chosen_C


def run_config(long_df_used, label_df, split, out_root, config):
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    pred_pieces, coef_pieces = [], []
    for fold_id, fold_grp in sp.groupby("fold_id"):
        fold_label = fold_grp["fold_label"].iloc[0]
        train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
        test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
        X_tr, Y_tr, _, csf_cols = build_train(long_df_used, label_df, train_ids, config)
        X_te, meta_te = build_test(long_df_used, test_ids, config)
        if X_tr is None or X_te is None or not csf_cols:
            continue
        keep = [c for c in csf_cols if Y_tr[c].nunique() > 1]
        for csf in keep:
            y = Y_tr[csf].values
            res = fit_one_csf(X_tr, y, X_te, config)
            if res is None:
                continue
            proba, coef_dict, chosen_C = res
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
                "config": config,
            })
            pred_pieces.append(pred)
            for fname, c in coef_dict.items():
                if fname == "_chosen_C":
                    continue
                coef_pieces.append({
                    "csf": csf, "feature": fname, "coefficient": c,
                    "chosen_C": chosen_C,
                    "fold_id": fold_id, "fold_label": fold_label,
                    "split_name": split, "config": config,
                    "label_rule": "within_eps_rank",
                })
    if not pred_pieces:
        return 0
    out_dir = out_root / ABLATION_ROOT / "track1" / split / config
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.concat(pred_pieces, ignore_index=True), out_dir / "preds.parquet")
    write_parquet(pd.DataFrame(coef_pieces), out_dir / "coefficients.parquet")
    return len(pred_pieces)


def filter_to_side(df, side):
    if side == "all":
        return df
    if side == "head":
        return df[df["csf"].isin(HEAD_SIDE_CSFS)]
    return df[df["csf"].isin(FEATURE_SIDE_CSFS)]


def oracle_for_side(oracle, side):
    suffix = "" if side == "all" else f"_{side}"
    return oracle.assign(
        o_csf=oracle[f"oracle_csf{suffix}"],
        o_augrc=oracle[f"oracle_augrc{suffix}"],
        w_csf=oracle[f"worst_csf{suffix}"],
        w_augrc=oracle[f"worst_augrc{suffix}"],
    )


def binary_per_row(preds_bin, eval_long, oracle_side):
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


def bootstrap_mean_ci(values):
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


def report(out_root, out_path):
    lines = ["# Ablation — calibrated (b+c) across three context-feature configs\n\n"]
    lines.append("**Date:** 2026-05-05\n")
    lines.append(f"**Source:** `code/nc_csf_predictivity/ablations/calibration_features.py`\n")
    lines.append("**Calibration (held constant):** L2 Cs=50, cv=5, "
                 "class_weight='balanced', per-architecture NC standardization.\n")
    lines.append("**Configs:**\n"
                 "- `source`: NC + source one-hot + regime\n"
                 "- `n_classes`: NC + n_classes (scaled) + regime\n"
                 "- `none`: NC + regime only\n\n")

    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    eval_long = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})
    oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_model_id(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    for split in SPLITS:
        lines.append(f"## {split}\n\n")
        for side in SIDES:
            rows = []
            cfg_metrics = {c: {} for c in CONFIGS}
            cfg_emp = {c: {} for c in CONFIGS}
            for config in CONFIGS:
                pp = out_root / ABLATION_ROOT / "track1" / split / config / "preds.parquet"
                if not pp.exists():
                    continue
                preds = pq.read_table(pp).to_pandas()
                eval_side = filter_to_side(eval_long, side)
                os_ = oracle_for_side(oracle_df, side)
                m = binary_per_row(preds, eval_side, os_)
                for regime in ["near", "mid", "far"]:
                    sub = m[m["regime"] == regime]
                    if not sub.empty:
                        vals = sub["set_regret_raw_imputed"].astype(float).values
                        mean, _, _ = bootstrap_mean_ci(vals)
                        cfg_metrics[config][regime] = mean
                        cfg_emp[config][regime] = float(sub["empty_set"].mean())
            for regime in ["near", "mid", "far"]:
                rows.append({
                    "regime": regime,
                    "src_regret": round(cfg_metrics["source"].get(regime, np.nan), 2) if regime in cfg_metrics["source"] else None,
                    "src_empty%": round(cfg_emp["source"].get(regime, np.nan), 2) if regime in cfg_emp["source"] else None,
                    "ncl_regret": round(cfg_metrics["n_classes"].get(regime, np.nan), 2) if regime in cfg_metrics["n_classes"] else None,
                    "ncl_empty%": round(cfg_emp["n_classes"].get(regime, np.nan), 2) if regime in cfg_emp["n_classes"] else None,
                    "none_regret": round(cfg_metrics["none"].get(regime, np.nan), 2) if regime in cfg_metrics["none"] else None,
                    "none_empty%": round(cfg_emp["none"].get(regime, np.nan), 2) if regime in cfg_emp["none"] else None,
                })
            df = pd.DataFrame(rows)
            lines.append(f"### `side = {side}` (imputed regret + empty share)\n\n")
            lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")

        # Per-CSF chosen C dispersion across configs
        lines.append(f"### Coefficient stability across configs ({split})\n\n")
        lines.append("Pearson correlation between NC coefficient vectors per CSF "
                     "across config pairs (averaged across folds).\n\n")
        coef_paths = {c: out_root / ABLATION_ROOT / "track1" / split / c
                      / "coefficients.parquet" for c in CONFIGS}
        if all(p.exists() for p in coef_paths.values()):
            coefs = {c: pq.read_table(p).to_pandas() for c, p in coef_paths.items()}
            avg = {}
            for c, df in coefs.items():
                df = df[(df["feature"] != "(intercept)")
                        & df["feature"].isin(NC_PRIMARY)]
                avg[c] = df.groupby(["csf", "feature"])["coefficient"].mean().reset_index()
            corrs = []
            csfs = sorted(set.intersection(*(set(avg[c]["csf"].unique()) for c in CONFIGS)))
            for csf in csfs:
                vec_src = avg["source"][avg["source"]["csf"] == csf].set_index("feature")["coefficient"].reindex(NC_PRIMARY).values
                vec_ncl = avg["n_classes"][avg["n_classes"]["csf"] == csf].set_index("feature")["coefficient"].reindex(NC_PRIMARY).values
                vec_none = avg["none"][avg["none"]["csf"] == csf].set_index("feature")["coefficient"].reindex(NC_PRIMARY).values
                if any(np.all(np.isnan(v)) for v in (vec_src, vec_ncl, vec_none)):
                    continue
                try:
                    r_sn = float(np.corrcoef(vec_src, vec_ncl)[0, 1])
                    r_se = float(np.corrcoef(vec_src, vec_none)[0, 1])
                    r_ne = float(np.corrcoef(vec_ncl, vec_none)[0, 1])
                    corrs.append({"csf": csf, "r(src,ncl)": round(r_sn, 3),
                                  "r(src,none)": round(r_se, 3), "r(ncl,none)": round(r_ne, 3)})
                except Exception:
                    pass
            if corrs:
                df_r = pd.DataFrame(corrs).sort_values("r(src,none)")
                lines.append("```\n" + df_r.to_string(index=False) + "\n```\n\n")

    out_path.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)

    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    long_df = add_n_classes(long_df)
    long_df = standardize_per_architecture(long_df, NC_PRIMARY)

    within = pq.read_table(out_root / "track1" / "labels" / "within_eps.parquet").to_pandas()
    label_df = (within[["paradigm", "source", "dropout", "reward", "regime",
                        "csf", "in_within_eps_set_rank"]]
                .rename(columns={"in_within_eps_set_rank": "label"}).copy())
    label_df["label"] = label_df["label"].astype(int)

    for split in SPLITS:
        print(f"\n=== {split} ===")
        for config in CONFIGS:
            n = run_config(long_df, label_df, split, out_root, config)
            print(f"  {config}: {n} per-CSF heads")

    report(out_root, out_root / "22_ablation_calib_features.md")
    print(f"\nwrote {out_root / '22_ablation_calib_features.md'}")


if __name__ == "__main__":
    main()

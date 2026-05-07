"""Calibration ablations for the empty-set issue (options b and c).

Runs three variants on top of the L2 Cs=50 per_csf binary baseline:

  (b)   class_weighted     — `LogisticRegressionCV(class_weight='balanced')`
                             rescales the loss so positive/negative examples
                             contribute equally, centering predictions near
                             0.5 instead of near the marginal positive rate.

  (c)   per_arch_std        — NC features are pre-standardized per architecture
                             (z-score within architecture), so a ResNet18 row
                             with 'high' self_duality for ResNet18 maps to the
                             same scaled value as a VGG13 row with 'high'
                             self_duality for VGG13.

  (b+c) class_weighted_perarch_std — both fixes combined.

All variants use:
  - L2 logistic regression, LogisticRegressionCV(Cs=50, cv=5)
  - Same features (8 NC + source one-hot + regime one-hot, n_classes excluded)
  - Same labels (within_eps_rank)
  - Same training pool (VGG13)
  - Splits xarch + lopo

Outputs:
  outputs/ablations/calibration/track1/<split>/<variant>/
    preds.parquet
    coefficients.parquet
  outputs/21_ablation_calibration.md
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
ABLATION_ROOT = "ablations/calibration"

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
SIDES = ["all", "head", "feature"]
TRAIN_REGIMES = ["near", "mid", "far", "all"]
SPLITS = ["xarch", "lopo"]
VARIANTS = ["class_weighted", "per_arch_std", "class_weighted_perarch_std"]
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


def standardize_per_architecture(long_df: pd.DataFrame, nc_cols: list[str]
                                 ) -> pd.DataFrame:
    """Replace NC columns with per-architecture z-scores.

    For each architecture, compute mean and std of each NC feature on that
    architecture's rows, then z-score within architecture. Test rows are
    standardized against the test architecture's own distribution; train
    rows against the training architecture's distribution. Fully unsupervised
    — only uses NC values, no labels."""
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


def build_pipeline(class_weight: str | None, prestandardized: bool) -> Pipeline:
    """If prestandardized=True, NC features bypass StandardScaler (already z-scored)."""
    nc_transform = "passthrough" if prestandardized else StandardScaler()
    pre = ColumnTransformer([
        ("nc", nc_transform, NC_PRIMARY),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         ["source", "regime"]),
    ])
    base = LogisticRegressionCV(
        Cs=L2_CS, cv=L2_CV, penalty="l2", solver="lbfgs",
        max_iter=L2_MAX_ITER, scoring="neg_log_loss", n_jobs=1,
        class_weight=class_weight,
    )
    return Pipeline([("preprocess", pre), ("clf", base)])


def feature_names_after_preprocess(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    nc_names = list(pre.transformers_[0][2])
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["source", "regime"]))
    return nc_names + cat_names


def get_unique_models(long_df: pd.DataFrame) -> pd.DataFrame:
    keep = (["model_id", "architecture", "paradigm", "source", "run",
             "dropout", "reward"] + NC_PRIMARY)
    return long_df[keep].drop_duplicates(subset=["model_id"]).reset_index(drop=True)


def build_train(long_df, label_df, train_model_ids):
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
    X = merged[NC_PRIMARY + ["source", "regime"]]
    Y = merged[csf_cols]
    meta = merged[["model_id", "architecture", "paradigm", "source",
                   "run", "dropout", "reward", "regime"]]
    return X, Y, meta, csf_cols


def build_test(long_df, test_model_ids):
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
    X = feats[NC_PRIMARY + ["source", "regime"]]
    meta = feats[["model_id", "architecture", "paradigm", "source",
                  "run", "dropout", "reward", "regime"]]
    return X, meta


def fit_one_csf(X_tr, y_tr, X_te, class_weight, prestandardized):
    counts = np.bincount(y_tr, minlength=2)
    if min(counts) < L2_CV:
        return None
    try:
        pipe = build_pipeline(class_weight=class_weight, prestandardized=prestandardized)
        pipe.fit(X_tr, y_tr)
    except (ValueError, RuntimeError):
        return None
    proba = pipe.predict_proba(X_te)[:, 1]
    clf = pipe.named_steps["clf"]
    feat_names = feature_names_after_preprocess(pipe)
    chosen_C = float(clf.C_[0])
    coef_dict = {"(intercept)": float(clf.intercept_[0]), "_chosen_C": chosen_C}
    for fname, c in zip(feat_names, clf.coef_[0]):
        coef_dict[fname] = float(c)
    return proba, coef_dict, chosen_C


def variant_settings(variant: str) -> tuple[str | None, bool]:
    if variant == "class_weighted":
        return ("balanced", False)
    if variant == "per_arch_std":
        return (None, True)
    if variant == "class_weighted_perarch_std":
        return ("balanced", True)
    raise ValueError(variant)


def run_variant(long_df_used, label_df, split, out_root, variant):
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    cw, prestd = variant_settings(variant)
    pred_pieces, coef_pieces = [], []
    for fold_id, fold_grp in sp.groupby("fold_id"):
        fold_label = fold_grp["fold_label"].iloc[0]
        train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
        test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
        X_tr, Y_tr, _, csf_cols = build_train(long_df_used, label_df, train_ids)
        X_te, meta_te = build_test(long_df_used, test_ids)
        if X_tr is None or X_te is None or not csf_cols:
            continue
        keep = [c for c in csf_cols if Y_tr[c].nunique() > 1]
        for csf in keep:
            y = Y_tr[csf].values
            res = fit_one_csf(X_tr, y, X_te, class_weight=cw, prestandardized=prestd)
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
                "variant": variant,
            })
            pred_pieces.append(pred)
            for fname, c in coef_dict.items():
                if fname == "_chosen_C":
                    continue
                coef_pieces.append({
                    "csf": csf, "feature": fname, "coefficient": c,
                    "chosen_C": chosen_C,
                    "fold_id": fold_id, "fold_label": fold_label,
                    "split_name": split, "variant": variant,
                    "label_rule": "within_eps_rank",
                })
    if not pred_pieces:
        return 0
    out_dir = out_root / ABLATION_ROOT / "track1" / split / variant
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
    lines = ["# Ablation — class-weighted + per-arch standardization (options b, c)\n\n"]
    lines.append("**Date:** 2026-05-05\n")
    lines.append(f"**Source:** `code/nc_csf_predictivity/ablations/calibration_balanced_perarch.py`\n")
    lines.append("**Model:** `LogisticRegressionCV(Cs=50, cv=5, penalty='l2')` per CSF\n")
    lines.append("**Variants:**\n"
                 "- `baseline_cs50`: pooled standardization, class_weight=None (step 23)\n"
                 "- `class_weighted`: pooled standardization, class_weight='balanced'\n"
                 "- `per_arch_std`: per-architecture standardization, class_weight=None\n"
                 "- `class_weighted_perarch_std`: both fixes\n\n")
    lines.append("Reference numbers:\n"
                 "- `Cs10_headline` = step 12 (over-regularized; 0% empty by collapsing CTM to constant)\n"
                 "- `Cs50_baseline` = step 23 (clean comparison point)\n\n")

    for split in SPLITS:
        lines.append(f"## {split}\n\n")
        for side in SIDES:
            rows = []
            # Cs10 headline imputed
            ws = pq.read_table(out_root / "track1" / split / "baselines" / "aggregate.parquet").to_pandas()
            cs10_imp = ws[(ws["comparator_kind"] == "predictor_imputed")
                          & (ws["comparator_name"] == "per_csf_binary/within_eps_rank")
                          & (ws["side"] == side)]
            # Cs50 baseline (recompute on the fly)
            cs50_path = out_root / "ablations" / "l2_cs50" / "track1" / split / "preds.parquet"
            cs50_metrics = {}
            cs50_emp = {}
            if cs50_path.exists():
                p_old = pq.read_table(cs50_path).to_pandas()
                long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
                long_df = add_model_id(long_df)
                eval_long = long_df[long_df["regime"] != "test"][[
                    "model_id", "eval_dataset", "regime", "csf", "augrc"
                ]].rename(columns={"augrc": "raw_augrc"})
                oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
                oracle_df = add_model_id(oracle_df)
                oracle_df = oracle_df[oracle_df["regime"] != "test"]
                eval_side = filter_to_side(eval_long, side)
                os_ = oracle_for_side(oracle_df, side)
                m_old = binary_per_row(p_old, eval_side, os_)
                for regime in ["near", "mid", "far"]:
                    sub = m_old[m_old["regime"] == regime]
                    if not sub.empty:
                        vals = sub["set_regret_raw_imputed"].astype(float).values
                        mean, _, _ = bootstrap_mean_ci(vals)
                        cs50_metrics[regime] = mean
                        cs50_emp[regime] = float(sub["empty_set"].mean())

            # New variants
            variant_metrics = {v: {} for v in VARIANTS}
            variant_emp = {v: {} for v in VARIANTS}
            for variant in VARIANTS:
                pp = out_root / ABLATION_ROOT / "track1" / split / variant / "preds.parquet"
                if not pp.exists():
                    continue
                preds = pq.read_table(pp).to_pandas()
                long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
                long_df = add_model_id(long_df)
                eval_long = long_df[long_df["regime"] != "test"][[
                    "model_id", "eval_dataset", "regime", "csf", "augrc"
                ]].rename(columns={"augrc": "raw_augrc"})
                oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
                oracle_df = add_model_id(oracle_df)
                oracle_df = oracle_df[oracle_df["regime"] != "test"]
                eval_side = filter_to_side(eval_long, side)
                os_ = oracle_for_side(oracle_df, side)
                m = binary_per_row(preds, eval_side, os_)
                for regime in ["near", "mid", "far"]:
                    sub = m[m["regime"] == regime]
                    if not sub.empty:
                        vals = sub["set_regret_raw_imputed"].astype(float).values
                        mean, _, _ = bootstrap_mean_ci(vals)
                        variant_metrics[variant][regime] = mean
                        variant_emp[variant][regime] = float(sub["empty_set"].mean())

            for regime in ["near", "mid", "far"]:
                cs10_val = float(cs10_imp[cs10_imp["regime"] == regime]["regret_raw_mean"].iloc[0]) if not cs10_imp[cs10_imp["regime"] == regime].empty else np.nan
                rows.append({
                    "regime": regime,
                    "Cs10_headline": round(cs10_val, 2) if not np.isnan(cs10_val) else None,
                    "Cs50_baseline": round(cs50_metrics.get(regime, np.nan), 2) if regime in cs50_metrics else None,
                    "  empty%_baseline": round(cs50_emp.get(regime, np.nan), 2) if regime in cs50_emp else None,
                    "(b)_class_weighted": round(variant_metrics["class_weighted"].get(regime, np.nan), 2) if regime in variant_metrics["class_weighted"] else None,
                    "  empty%_(b)": round(variant_emp["class_weighted"].get(regime, np.nan), 2) if regime in variant_emp["class_weighted"] else None,
                    "(c)_per_arch_std": round(variant_metrics["per_arch_std"].get(regime, np.nan), 2) if regime in variant_metrics["per_arch_std"] else None,
                    "  empty%_(c)": round(variant_emp["per_arch_std"].get(regime, np.nan), 2) if regime in variant_emp["per_arch_std"] else None,
                    "(b+c)_combined": round(variant_metrics["class_weighted_perarch_std"].get(regime, np.nan), 2) if regime in variant_metrics["class_weighted_perarch_std"] else None,
                    "  empty%_(b+c)": round(variant_emp["class_weighted_perarch_std"].get(regime, np.nan), 2) if regime in variant_emp["class_weighted_perarch_std"] else None,
                })
            df = pd.DataFrame(rows)
            lines.append(f"### `side = {side}` (imputed regret + empty share)\n\n")
            lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")

    out_path.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)

    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    long_df_perarch = standardize_per_architecture(long_df, NC_PRIMARY)

    within = pq.read_table(out_root / "track1" / "labels" / "within_eps.parquet").to_pandas()
    label_df = (within[["paradigm", "source", "dropout", "reward", "regime",
                        "csf", "in_within_eps_set_rank"]]
                .rename(columns={"in_within_eps_set_rank": "label"}).copy())
    label_df["label"] = label_df["label"].astype(int)

    for split in SPLITS:
        print(f"\n=== {split} ===")
        for variant in VARIANTS:
            cw, prestd = variant_settings(variant)
            df_used = long_df_perarch if prestd else long_df
            n = run_variant(df_used, label_df, split, out_root, variant)
            print(f"  {variant}: {n} folds × CSFs trained")

    report(out_root, out_root / "21_ablation_calibration.md")
    print(f"\nwrote {out_root / '21_ablation_calibration.md'}")


if __name__ == "__main__":
    main()

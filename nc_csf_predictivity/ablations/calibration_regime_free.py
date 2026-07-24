"""Regime-free ablation: per-CSF clique predictor WITHOUT the OOD-regime input.

Addresses the reviewer concern that the NC-based predictor consumes a
near/mid/far regime indicator whose provenance at deployment is unclear.
Two regime-free variants are trained and evaluated with the exact protocol of
`calibration_features_clique.py` (L2 logistic, Cs=50, cv=5, class-balanced,
per-architecture NC standardization, clique labels, VGG13-only training pool):

  marginal : train on the near/mid/far rows with the regime feature removed.
             Each model contributes three rows with identical features and
             regime-specific labels, so the head learns the regime-marginal
             competitiveness probability. One shortlist per model, applied
             to every regime at test time.
  pooled   : train on the regime='all' rows only (labels = top cliques over
             the pooled near+mid+far blocks). One shortlist per model.

Feature configs mirror the originals minus regime:
  source_nr : NC + source one-hot
  none_nr   : NC only

Outputs:
  outputs/ablations/calib_cliques_regime_free/track1/<split>/<config>_<variant>/
    preds.parquet, coefficients.parquet
  outputs/25_ablation_regime_free.md
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as scstats
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    L2_CS,
    L2_CV,
    L2_MAX_ITER,
    NC_PRIMARY,
    SIDES,
    SPLITS,
    add_model_id,
    binary_per_row,
    bootstrap_mean_ci,
    filter_to_side,
    get_unique_models,
    oracle_for_side,
    standardize_per_architecture,
    write_parquet,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

ABLATION_ROOT = "ablations/calib_cliques_regime_free"
CONFIGS_NR = ["source_nr", "none_nr"]
VARIANTS = ["marginal", "pooled"]
EVAL_REGIMES = ["near", "mid", "far"]


def build_pipeline_nr(config: str) -> Pipeline:
    """Logistic pipeline without any regime feature."""
    transformers = [("nc", "passthrough", NC_PRIMARY)]
    if config == "source_nr":
        transformers.append((
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore",
                          sparse_output=False),
            ["source"],
        ))
    elif config != "none_nr":
        raise ValueError(config)
    pre = ColumnTransformer(transformers)
    base = LogisticRegressionCV(
        Cs=L2_CS, cv=L2_CV, penalty="l2", solver="lbfgs",
        max_iter=L2_MAX_ITER, scoring="neg_log_loss", n_jobs=1,
        class_weight="balanced",
    )
    return Pipeline([("preprocess", pre), ("clf", base)])


def feature_columns_nr(config: str) -> list[str]:
    """Model input columns for a regime-free config."""
    if config == "source_nr":
        return NC_PRIMARY + ["source"]
    if config == "none_nr":
        return list(NC_PRIMARY)
    raise ValueError(config)


def build_train_nr(long_df: pd.DataFrame, label_df: pd.DataFrame,
                   train_model_ids: list[str], config: str, variant: str
                   ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """Training matrix without regime features.

    Args:
        long_df: Harmonized long frame with per-architecture-standardized NC.
        label_df: Clique labels with a regime column (near/mid/far/all/test).
        train_model_ids: Model ids in the training fold.
        config: Regime-free feature config name.
        variant: 'marginal' (near/mid/far label rows) or 'pooled' ('all' rows).

    Returns:
        (X, Y, csf_cols); X has one row per (model, label regime) for the
        marginal variant and one row per model for the pooled variant.
    """
    nc_per_model = get_unique_models(long_df, "source")
    pool = nc_per_model[(nc_per_model["architecture"] == "VGG13")
                        & nc_per_model["model_id"].isin(train_model_ids)]
    if pool.empty:
        return None, None, []
    label_regimes = EVAL_REGIMES if variant == "marginal" else ["all"]
    rows = []
    for _, m in pool.iterrows():
        for regime in label_regimes:
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
    merged = feats.merge(
        label_wide, on=["paradigm", "source", "dropout", "reward", "regime"],
        how="inner")
    if merged.empty:
        return None, None, []
    return merged[feature_columns_nr(config)], merged[csf_cols], csf_cols


def build_test_nr(long_df: pd.DataFrame, test_model_ids: list[str], config: str
                  ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """One feature row per test model (regime-independent)."""
    nc_per_model = get_unique_models(long_df, "source")
    pool = nc_per_model[nc_per_model["model_id"].isin(test_model_ids)]
    if pool.empty:
        return None, None
    X = pool[feature_columns_nr(config)]
    meta = pool[["model_id", "architecture", "paradigm", "source",
                 "run", "dropout", "reward"]]
    return X, meta


def run_config_variant(long_df: pd.DataFrame, label_df: pd.DataFrame,
                       split: str, out_root: Path, config: str,
                       variant: str) -> int:
    """Train per-CSF heads for one (split, config, variant); write preds."""
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    pred_pieces, coef_pieces = [], []
    for fold_id, fold_grp in sp.groupby("fold_id"):
        fold_label = fold_grp["fold_label"].iloc[0]
        train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
        test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
        X_tr, Y_tr, csf_cols = build_train_nr(
            long_df, label_df, train_ids, config, variant)
        X_te, meta_te = build_test_nr(long_df, test_ids, config)
        if X_tr is None or X_te is None or not csf_cols:
            continue
        for csf in [c for c in csf_cols if Y_tr[c].nunique() > 1]:
            y = Y_tr[csf].values
            if min(np.bincount(y, minlength=2)) < L2_CV:
                continue
            try:
                pipe = build_pipeline_nr(config)
                pipe.fit(X_tr, y)
            except (ValueError, RuntimeError):
                continue
            proba = pipe.predict_proba(X_te)[:, 1]
            clf = pipe.named_steps["clf"]
            for regime in EVAL_REGIMES:
                pred_pieces.append(pd.DataFrame({
                    "model_id": meta_te["model_id"].values,
                    "regime": regime,
                    "csf": csf,
                    "p_competitive": proba,
                    "predicted_competitive": proba >= 0.5,
                    "split_name": split,
                    "fold_id": fold_id,
                    "fold_label": fold_label,
                    "label_rule": "clique",
                    "config": f"{config}_{variant}",
                }))
            coef_pieces.append({
                "csf": csf, "fold_id": fold_id, "fold_label": fold_label,
                "split_name": split, "config": f"{config}_{variant}",
                "chosen_C": float(clf.C_[0]),
                "intercept": float(clf.intercept_[0]),
            })
    if not pred_pieces:
        return 0
    out_dir = out_root / ABLATION_ROOT / "track1" / split / f"{config}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.concat(pred_pieces, ignore_index=True),
                  out_dir / "preds.parquet")
    write_parquet(pd.DataFrame(coef_pieces), out_dir / "coefficients.parquet")
    return len(pred_pieces)


def per_row_metrics(preds: pd.DataFrame, eval_long: pd.DataFrame,
                    oracle_df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Imputed set-regret rows for one predictor, filtered to a side."""
    eval_side = filter_to_side(eval_long, side)
    return binary_per_row(preds, eval_side, oracle_for_side(oracle_df, side))


def wilcoxon_vs_reference(m_new: pd.DataFrame, m_ref: pd.DataFrame
                          ) -> tuple[float, float, int]:
    """Paired Wilcoxon on per-(model, eval set) imputed regret vs a reference.

    Returns (median paired difference new-ref, two-sided p, n pairs).
    """
    keys = ["model_id", "eval_dataset"]
    a = m_new.set_index(keys)["set_regret_raw_imputed"]
    b = m_ref.set_index(keys)["set_regret_raw_imputed"]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).astype(float).dropna()
    if len(d) < 10 or (d == 0).all():
        return float("nan"), float("nan"), len(d)
    res = scstats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return float(d.median()), float(res.pvalue), len(d)


def report(out_root: Path, out_path: Path) -> None:
    """Regret comparison: regime-free variants vs regime-ful configs and baselines."""
    long_df = pq.read_table(
        out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    eval_long = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})
    oracle_df = pq.read_table(
        out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_model_id(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    lines = ["# Regime-free predictor ablation (rebuttal experiment E-B)\n\n",
             "**Source:** `nc_csf_predictivity/ablations/calibration_regime_free.py`\n",
             "**Protocol:** identical to `calibration_features_clique.py` "
             "except the regime input is removed (marginal/pooled label "
             "variants). Regret = imputed set-regret, bootstrap 95% CI.\n\n"]

    for split in SPLITS:
        lines.append(f"## {split}\n\n")
        bl = pq.read_table(out_root / "track1" / split / "baselines"
                           / "aggregate.parquet").to_pandas()
        bl_pred = bl[bl["comparator_kind"] == "baseline"]

        ref_path = (out_root / "ablations" / "calib_cliques" / "track1"
                    / split / "source" / "preds.parquet")
        ref_preds = pq.read_table(ref_path).to_pandas() if ref_path.exists() else None

        arms: dict[str, pd.DataFrame] = {}
        if ref_preds is not None:
            arms["with-regime (source)"] = ref_preds
        for config in CONFIGS_NR:
            for variant in VARIANTS:
                pp = (out_root / ABLATION_ROOT / "track1" / split
                      / f"{config}_{variant}" / "preds.parquet")
                if pp.exists():
                    arms[f"{config}_{variant}"] = pq.read_table(pp).to_pandas()

        for side in SIDES:
            rows = []
            ref_metrics_by_regime: dict[str, pd.DataFrame] = {}
            for regime in EVAL_REGIMES:
                bl_cell = bl_pred[(bl_pred["regime"] == regime)
                                  & (bl_pred["side"] == side)]
                best_bl = (bl_cell.loc[bl_cell["regret_raw_mean"].idxmin()]
                           if not bl_cell.empty else None)
                rec = {"regime": regime,
                       "best_baseline": (best_bl["comparator_name"]
                                         if best_bl is not None else None),
                       "bl_regret": (round(float(best_bl["regret_raw_mean"]), 2)
                                     if best_bl is not None else None)}
                for arm_name, preds in arms.items():
                    m = per_row_metrics(preds, eval_long, oracle_df, side)
                    m_r = m[m["regime"] == regime]
                    if arm_name == "with-regime (source)":
                        ref_metrics_by_regime[regime] = m_r
                    if m_r.empty:
                        continue
                    vals = m_r["set_regret_raw_imputed"].astype(float).values
                    mean, lo, hi = bootstrap_mean_ci(vals)
                    rec[arm_name] = f"{mean:.2f} [{lo:.2f},{hi:.2f}]"
                    rec[f"{arm_name}|empty%"] = round(
                        float(m_r["empty_set"].mean()) * 100, 1)
                    if (arm_name != "with-regime (source)"
                            and regime in ref_metrics_by_regime):
                        med_d, p, n = wilcoxon_vs_reference(
                            m_r, ref_metrics_by_regime[regime])
                        rec[f"{arm_name}|p_vs_regime"] = (
                            f"{p:.3g} (n={n})" if p == p else "NA")
                rows.append(rec)
            lines.append(f"### side = {side}\n\n")
            lines.append("```\n" + pd.DataFrame(rows).to_string(index=False)
                         + "\n```\n\n")

    out_path.write_text("".join(lines))


def main() -> None:
    """Train regime-free variants on xarch and lopo; write report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)

    long_df = pq.read_table(
        out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    long_df = standardize_per_architecture(long_df, NC_PRIMARY)

    cliques = pq.read_table(
        out_root / "track1" / "cliques" / "cliques.parquet").to_pandas()
    label_df = (cliques[["paradigm", "source", "dropout", "reward", "regime",
                         "csf", "in_top_clique"]]
                .rename(columns={"in_top_clique": "label"}))
    label_df = label_df.assign(label=label_df["label"].astype(int))

    for split in SPLITS:
        print(f"=== {split} ===")
        for config in CONFIGS_NR:
            for variant in VARIANTS:
                n = run_config_variant(long_df, label_df, split, out_root,
                                       config, variant)
                print(f"  {config}_{variant}: {n} pred pieces")

    report(out_root, out_root / "25_ablation_regime_free.md")
    print(f"wrote {out_root / '25_ablation_regime_free.md'}")


if __name__ == "__main__":
    main()

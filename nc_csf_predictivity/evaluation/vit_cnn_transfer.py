"""Cross-family predictor transfer: ViT -> CNN and CNN -> ViT (rebuttal E-C).

Direction 1 (new): train the per-CSF clique predictor on the 40 ViT models
(clique labels computed here from the ViT blocks with the paper's
Friedman-Conover pipeline) and apply it, without retraining, to the VGG-13
and ResNet-18 pools.

Direction 2 (existing): the `lopo_modelvit` fold of the lopo split already
trains on CNN paradigms and tests on ViT; its predictions are re-evaluated
here against direction-matched baselines so both directions share one report.

Protocol is otherwise identical to `calibration_features_clique.py` (L2
logistic, Cs=50, cv=5, class-balanced, per-architecture NC standardization,
clique labels, imputed set-regret, bootstrap 95% CI).

Outputs:
  outputs/track1/cliques/cliques_vit.parquet
  outputs/xfam/<fold>/<config>/preds.parquet
  outputs/26_vit_cnn_transfer.md
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

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVAL_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR / "ablations"))
sys.path.insert(0, str(PIPELINE_DIR / "data"))
sys.path.insert(0, str(EVAL_DIR))

from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    NC_PRIMARY,
    SIDES,
    add_model_id,
    binary_per_row,
    bootstrap_mean_ci,
    build_pipeline,
    build_test,
    feature_columns_for_config,
    filter_to_side,
    get_unique_models,
    oracle_for_side,
    standardize_per_architecture,
    write_parquet,
)
from baselines import compute_baselines_per_row  # noqa: E402
from cliques_track1 import compute_track1_cliques  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

CONFIGS = ["source", "none"]
EVAL_REGIMES = ["near", "mid", "far"]
TRAIN_REGIMES = ["near", "mid", "far", "all"]
FOLDS = [
    {"fold_label": "vit_to_vgg13", "train_arch": "ViT", "test_arch": "VGG13"},
    {"fold_label": "vit_to_resnet18", "train_arch": "ViT",
     "test_arch": "ResNet18"},
]


def vit_cliques(long_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    """Compute (or load cached) clique labels for the ViT blocks."""
    path = out_root / "track1" / "cliques" / "cliques_vit.parquet"
    if path.exists():
        return pq.read_table(path).to_pandas()
    vit = long_df[(long_df["architecture"] == "ViT")
                  & (long_df["paradigm"] == "modelvit")]
    flat, _ = compute_track1_cliques(vit)
    write_parquet(flat, path)
    print(f"wrote {path} ({len(flat):,} rows)")
    return flat


def build_train_arch(long_df: pd.DataFrame, label_df: pd.DataFrame,
                     train_arch: str, config: str
                     ) -> tuple[pd.DataFrame | None, pd.DataFrame | None,
                                list[str]]:
    """Training matrix for an arbitrary training architecture.

    Mirrors `calibration_features_clique.build_train` but parameterizes the
    hardcoded VGG13 pool filter.
    """
    nc_per_model = get_unique_models(long_df, config)
    pool = nc_per_model[nc_per_model["architecture"] == train_arch]
    if pool.empty:
        return None, None, []
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
                if c not in ["paradigm", "source", "dropout", "reward",
                             "regime"]]
    label_wide[csf_cols] = label_wide[csf_cols].astype(int)
    merged = feats.merge(
        label_wide, on=["paradigm", "source", "dropout", "reward", "regime"],
        how="inner")
    if merged.empty:
        return None, None, []
    return (merged[feature_columns_for_config(config)], merged[csf_cols],
            csf_cols)


def train_fold(long_df: pd.DataFrame, label_df: pd.DataFrame,
               fold: dict, config: str, out_root: Path) -> int:
    """Fit per-CSF heads on the train architecture; predict the test pool."""
    test_ids = (long_df[long_df["architecture"] == fold["test_arch"]]
                ["model_id"].unique().tolist())
    X_tr, Y_tr, csf_cols = build_train_arch(
        long_df, label_df, fold["train_arch"], config)
    X_te, meta_te = build_test(long_df, test_ids, config)
    if X_tr is None or X_te is None or not csf_cols:
        return 0
    pred_pieces = []
    for csf in [c for c in csf_cols if Y_tr[c].nunique() > 1]:
        y = Y_tr[csf].values
        if min(np.bincount(y, minlength=2)) < 5:
            continue
        try:
            pipe = build_pipeline(config)
            pipe.fit(X_tr, y)
        except (ValueError, RuntimeError):
            continue
        proba = pipe.predict_proba(X_te)[:, 1]
        pred_pieces.append(pd.DataFrame({
            "model_id": meta_te["model_id"].values,
            "regime": meta_te["regime"].values,
            "csf": csf,
            "p_competitive": proba,
            "predicted_competitive": proba >= 0.5,
            "fold_label": fold["fold_label"],
            "config": config,
            "label_rule": "clique",
        }))
    if not pred_pieces:
        return 0
    out_dir = out_root / "xfam" / fold["fold_label"] / config
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.concat(pred_pieces, ignore_index=True),
                  out_dir / "preds.parquet")
    return len(pred_pieces)


def direction_tables(preds: pd.DataFrame, eval_long: pd.DataFrame,
                     oracle_df: pd.DataFrame, train_long: pd.DataFrame,
                     all_csfs: list[str], label: str) -> list[str]:
    """Regret table per (side, regime): predictor vs direction-matched baselines."""
    test_ids = preds["model_id"].unique()
    eval_test = eval_long[eval_long["model_id"].isin(test_ids)]
    oracle_test = oracle_df[oracle_df["model_id"].isin(test_ids)]
    bl = compute_baselines_per_row(eval_test, oracle_test, train_long,
                                   "model_id", all_csfs)
    lines = [f"### {label}\n\n"]
    for side in SIDES:
        m = binary_per_row(preds, filter_to_side(eval_long, side),
                           oracle_for_side(oracle_df, side))
        rows = []
        for regime in EVAL_REGIMES:
            m_r = m[m["regime"] == regime]
            bl_r = bl[(bl["regime"] == regime) & (bl["side"] == side)]
            best = (bl_r.groupby("comparator_name")["regret_raw"].mean()
                    .sort_values())
            rec = {"regime": regime}
            if not m_r.empty:
                vals = m_r["set_regret_raw_imputed"].astype(float).values
                mean, lo, hi = bootstrap_mean_ci(vals)
                rec["predictor"] = f"{mean:.2f} [{lo:.2f},{hi:.2f}]"
                rec["empty%"] = round(float(m_r["empty_set"].mean()) * 100, 1)
                rec["set_size"] = round(float(m_r["set_size"].mean()), 1)
            if not best.empty:
                rec["best_baseline"] = best.index[0]
                rec["bl_regret"] = round(float(best.iloc[0]), 2)
                if not m_r.empty:
                    b_r = bl_r[bl_r["comparator_name"] == best.index[0]]
                    keys = ["model_id", "eval_dataset"]
                    a = m_r.set_index(keys)["set_regret_raw_imputed"]
                    b = b_r.set_index(keys)["regret_raw"]
                    common = a.index.intersection(b.index)
                    d = (a.loc[common] - b.loc[common]).astype(float).dropna()
                    if len(d) >= 10 and not (d == 0).all():
                        p = scstats.wilcoxon(
                            d, zero_method="wilcox",
                            alternative="two-sided").pvalue
                        rec["wilcoxon_p"] = f"{p:.3g}"
            rows.append(rec)
        lines.append(f"`side = {side}`\n\n```\n"
                     + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    return lines


def main() -> None:
    """Run both transfer directions and write the combined report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)

    raw = pq.read_table(
        out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    raw = add_model_id(raw)
    long_std = standardize_per_architecture(raw, NC_PRIMARY)

    cnn_cliques = pq.read_table(
        out_root / "track1" / "cliques" / "cliques.parquet").to_pandas()
    vit_flat = vit_cliques(raw, out_root)
    label_vit = (vit_flat[["paradigm", "source", "dropout", "reward", "regime",
                           "csf", "in_top_clique"]]
                 .rename(columns={"in_top_clique": "label"}))
    label_vit = label_vit.assign(label=label_vit["label"].astype(int))

    for fold in FOLDS:
        for config in CONFIGS:
            n = train_fold(long_std, label_vit, fold, config, out_root)
            print(f"{fold['fold_label']}/{config}: {n} per-CSF heads")

    eval_long = raw[raw["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})
    oracle_df = pq.read_table(
        out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_model_id(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]
    all_csfs = sorted(eval_long["csf"].unique())

    lines = ["# Cross-family predictor transfer (rebuttal experiment E-C)\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/vit_cnn_transfer.py`\n",
             "**Labels:** CNN direction = step-5 cliques; ViT direction = "
             "`cliques_vit.parquet` computed here with the same pipeline.\n",
             "**Regret:** imputed set-regret, bootstrap 95% CI; Wilcoxon "
             "two-sided vs the direction's best fixed baseline.\n\n",
             "## Direction 1: ViT -> CNN (new)\n\n"]

    vit_train_long = eval_long[eval_long["model_id"].str.startswith("ViT|")]
    for fold in FOLDS:
        for config in CONFIGS:
            pp = (out_root / "xfam" / fold["fold_label"] / config
                  / "preds.parquet")
            if not pp.exists():
                continue
            preds = pq.read_table(pp).to_pandas()
            lines.extend(direction_tables(
                preds, eval_long, oracle_df, vit_train_long, all_csfs,
                f"{fold['fold_label']} (config={config})"))

    lines.append("## Direction 2: CNN -> ViT (existing lopo_modelvit fold)\n\n")
    lopo_path = (out_root / "ablations" / "calib_cliques" / "track1" / "lopo"
                 / "source" / "preds.parquet")
    if lopo_path.exists():
        lp = pq.read_table(lopo_path).to_pandas()
        lp = lp[lp["fold_label"] == "lopo_modelvit"]
        cnn_train_long = eval_long[
            ~eval_long["model_id"].str.startswith("ViT|")]
        lines.extend(direction_tables(
            lp, eval_long, oracle_df, cnn_train_long, all_csfs,
            "lopo_modelvit (config=source)"))

    out_path = out_root / "26_vit_cnn_transfer.md"
    out_path.write_text("".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

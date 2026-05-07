"""Random forest per CSF — non-linear feature-importance comparison.

The headline classifier (per_csf_binary, LogisticRegressionCV) is a linear
model. Its near-zero coefficients for CTM might mean either (a) NC carries
no signal for CTM, or (b) NC carries non-linear signal that a linear model
can't extract. This ablation trains a `RandomForestClassifier` per CSF on
the same training data and compares feature importance to the linear
coefficient magnitudes.

Setup:
  Predictor:    RandomForestClassifier(n_estimators=300, max_depth=6,
                                       min_samples_leaf=5, random_state=0)
  Features:     8 NC + 3 source one-hot + 3 regime one-hot (same as headline)
  Target:       within_eps_rank competitive (binary, per CSF)
  Train pool:   VGG13 only (same as the headline binary heads)
  Splits:       xarch, lopo

Outputs:
  outputs/ablations/random_forest/track1/<split>/per_csf_rf/
    preds.parquet               (test-row p_competitive + predicted_competitive)
    feature_importance.parquet  (per (csf, feature) impurity-based importance)
  outputs/figures/rf_feature_importance_xarch.{pdf,png}
  outputs/figures/rf_feature_importance_lopo.{pdf,png}
  outputs/17_ablation_random_forest.md
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"
ABLATION_ROOT = "ablations/random_forest"

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
RF_N_EST = 300
RF_MAX_DEPTH = 6
RF_MIN_LEAF = 5
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


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("nc", StandardScaler(), NC_PRIMARY),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         ["source", "regime"]),
    ])
    rf = RandomForestClassifier(
        n_estimators=RF_N_EST, max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_LEAF, random_state=SEED, n_jobs=1,
    )
    return Pipeline([("preprocess", pre), ("clf", rf)])


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


def feature_names_after_preprocess(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    nc_names = list(pre.transformers_[0][2])
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(["source", "regime"]))
    return nc_names + cat_names


def run_rf(long_df, label_df, split, out_root):
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    pred_pieces, imp_pieces = [], []
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
        for csf in keep:
            y = Y_tr[csf].values
            try:
                pipe = build_pipeline()
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
            pred_pieces.append(pred)

            rf = pipe.named_steps["clf"]
            feat_names = feature_names_after_preprocess(pipe)
            imps = rf.feature_importances_
            imp = pd.DataFrame({
                "csf": csf,
                "fold_id": fold_id,
                "fold_label": fold_label,
                "split_name": split,
                "feature": feat_names,
                "importance": imps,
            })
            imp_pieces.append(imp)
    if not pred_pieces:
        return 0
    out_dir = out_root / ABLATION_ROOT / "track1" / split / "per_csf_rf"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_preds = pd.concat(pred_pieces, ignore_index=True)
    all_imps = pd.concat(imp_pieces, ignore_index=True)
    write_parquet(all_preds, out_dir / "preds.parquet")
    write_parquet(all_imps, out_dir / "feature_importance.parquet")
    return len(all_preds)


def order_features(present: list[str]) -> list[str]:
    nc_order = [f for f in NC_PRIMARY if f in present]
    src = sorted([f for f in present if f.startswith("source_")])
    reg = sorted([f for f in present if f.startswith("regime_")])
    rest = [f for f in present
            if f not in nc_order and not f.startswith(("source_", "regime_"))]
    return nc_order + src + reg + rest


def order_csfs(present: set[str]) -> list[str]:
    feat = [c for c in FEATURE_SIDE_CSFS if c in present]
    head = [c for c in HEAD_SIDE_CSFS if c in present]
    return feat + head


def plot_importance_heatmap(imps: pd.DataFrame, title: str, out_base: Path) -> None:
    avg = imps.groupby(["csf", "feature"])["importance"].mean().reset_index()
    pivot = avg.pivot(index="feature", columns="csf", values="importance")
    feature_order = order_features(list(pivot.index))
    csf_order = order_csfs(set(pivot.columns))
    pivot = pivot.reindex(index=feature_order, columns=csf_order)
    n_features = len(feature_order)
    n_csfs = len(csf_order)

    vmax = float(np.nanmax(pivot.values))
    fig, ax = plt.subplots(figsize=(max(n_csfs * 0.7 + 3, 12),
                                     max(n_features * 0.5 + 3, 7)))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(n_csfs))
    ax.set_xticklabels(csf_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_order, fontsize=9)
    for i in range(n_features):
        for j in range(n_csfs):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            colour = "white" if v < 0.5 * vmax else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=colour)

    sep_after_nc = sum(1 for f in feature_order if f in NC_PRIMARY) - 0.5
    sep_after_src = sep_after_nc + sum(1 for f in feature_order if f.startswith("source_"))
    if 0 < sep_after_nc < n_features - 1:
        ax.axhline(sep_after_nc, color="white", lw=0.8, alpha=0.7)
    if sep_after_nc < sep_after_src < n_features - 1:
        ax.axhline(sep_after_src, color="white", lw=0.8, alpha=0.7)
    n_feat_csfs = sum(1 for c in csf_order if c in FEATURE_SIDE_CSFS)
    if 0 < n_feat_csfs < n_csfs:
        ax.axvline(n_feat_csfs - 0.5, color="white", lw=0.8, alpha=0.7)

    fig.text(0.5 * (n_feat_csfs - 0.5) / (n_csfs - 1) * 0.78 + 0.11,
             0.92, "feature-side CSFs",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)
    fig.text((n_feat_csfs + (n_csfs - 1 - n_feat_csfs) * 0.5) / (n_csfs - 1) * 0.78 + 0.11,
             0.92, "head-side CSFs",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)
    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Random forest impurity-based feature importance\n"
                   "(non-negative; sums to 1 per CSF)", fontsize=9)
    ax.set_xlabel("CSF (output dimension)", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    fig.suptitle(title, fontsize=11, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)


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
    lines = ["# Ablation — Random forest per CSF (non-linear feature importance)\n\n"]
    lines.append("**Date:** 2026-05-04\n")
    lines.append("**Source:** `code/nc_csf_predictivity/ablations/random_forest.py`\n")
    lines.append(f"**Model:** RandomForestClassifier(n_estimators={RF_N_EST}, "
                 f"max_depth={RF_MAX_DEPTH}, min_samples_leaf={RF_MIN_LEAF})\n")
    lines.append("**Features, label rule, training pool, splits:** identical to "
                 "the headline `per_csf_binary` from steps 11–12.\n\n")

    lines.append("## CTM specific check — does RF find non-linear NC signal?\n\n")
    for split in SPLITS:
        ip = (out_root / ABLATION_ROOT / "track1" / split / "per_csf_rf"
              / "feature_importance.parquet")
        if not ip.exists():
            continue
        imps = pq.read_table(ip).to_pandas()
        ctm = imps[imps["csf"] == "CTM"]
        if ctm.empty:
            continue
        avg = ctm.groupby("feature")["importance"].mean().sort_values(ascending=False)
        lines.append(f"### {split} — CTM RF feature importance (averaged across folds)\n\n")
        lines.append("```\n" + avg.round(3).to_string() + "\n```\n")
        nc_total = float(avg.reindex(NC_PRIMARY).fillna(0).sum())
        src_total = float(avg[avg.index.str.startswith("source_")].sum())
        reg_total = float(avg[avg.index.str.startswith("regime_")].sum())
        lines.append(f"\nGroup totals (NC vs source vs regime):\n"
                     f"- NC features: {nc_total:.3f}\n"
                     f"- source one-hots: {src_total:.3f}\n"
                     f"- regime one-hots: {reg_total:.3f}\n\n")

    lines.append("## Set-regret comparison (RF vs linear per_csf binary)\n\n")
    for split in SPLITS:
        rf_pp = out_root / ABLATION_ROOT / "track1" / split / "per_csf_rf" / "preds.parquet"
        if not rf_pp.exists():
            continue
        long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
        long_df = add_model_id(long_df)
        eval_long = long_df[long_df["regime"] != "test"][[
            "model_id", "eval_dataset", "regime", "csf", "augrc"
        ]].rename(columns={"augrc": "raw_augrc"})
        oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
        oracle_df = add_model_id(oracle_df)
        oracle_df = oracle_df[oracle_df["regime"] != "test"]
        rf_preds = pq.read_table(rf_pp).to_pandas()

        # With-source linear binary (step 14 imputed for fair comparison)
        ws = pq.read_table(out_root / "track1" / split / "baselines" / "aggregate.parquet").to_pandas()
        lin_imp = ws[(ws["comparator_kind"] == "predictor_imputed")
                     & (ws["comparator_name"] == "per_csf_binary/within_eps_rank")]

        rows = []
        for side in SIDES:
            eval_side = filter_to_side(eval_long, side)
            os_ = oracle_for_side(oracle_df, side)
            m = binary_per_row(rf_preds, eval_side, os_)
            for regime in ["near", "mid", "far"]:
                sub = m[m["regime"] == regime]
                if sub.empty:
                    continue
                vals = sub["set_regret_raw_imputed"].astype(float).values
                rf_mean, rf_lo, rf_hi = bootstrap_mean_ci(vals)
                emp = float(sub["empty_set"].mean())
                lin_sel = lin_imp[(lin_imp["regime"] == regime)
                                  & (lin_imp["side"] == side)]
                lin_val = float(lin_sel["regret_raw_mean"].iloc[0]) if not lin_sel.empty else np.nan
                rows.append({
                    "regime": regime, "side": side,
                    "linear_imputed": round(lin_val, 3) if not np.isnan(lin_val) else None,
                    "rf_imputed": round(rf_mean, 3),
                    "rf_ci_lo": round(rf_lo, 3),
                    "rf_ci_hi": round(rf_hi, 3),
                    "rf_empty_share": round(emp, 3),
                    "delta": round(rf_mean - lin_val, 3) if not np.isnan(lin_val) else None,
                })
        df = pd.DataFrame(rows)
        lines.append(f"### {split}\n\n")
        lines.append("```\n" + df.to_string(index=False) + "\n```\n\n")

    out_path.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)

    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)

    within = pq.read_table(out_root / "track1" / "labels" / "within_eps.parquet").to_pandas()
    label_df = (within[["paradigm", "source", "dropout", "reward", "regime",
                        "csf", "in_within_eps_set_rank"]]
                .rename(columns={"in_within_eps_set_rank": "label"}).copy())
    label_df["label"] = label_df["label"].astype(int)

    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"\n=== {split} ===")
        n = run_rf(long_df, label_df, split, out_root)
        print(f"  per_csf_rf: {n:,} pred rows")
        ip = (out_root / ABLATION_ROOT / "track1" / split
              / "per_csf_rf" / "feature_importance.parquet")
        if ip.exists():
            imps = pq.read_table(ip).to_pandas()
            label = ("xarch — VGG13 → ResNet18, 1 fold" if split == "xarch"
                     else f"{split} — averaged across folds")
            plot_importance_heatmap(
                imps,
                title=("Per-CSF random forest feature importance\n"
                       f"(n_est={RF_N_EST}, max_depth={RF_MAX_DEPTH}, "
                       f"min_samples_leaf={RF_MIN_LEAF}; split = {label})"),
                out_base=fig_dir / f"rf_feature_importance_{split}",
            )
            print(f"  wrote {fig_dir / f'rf_feature_importance_{split}.pdf'}")

    report(out_root, out_root / "17_ablation_random_forest.md")
    print(f"wrote {out_root / '17_ablation_random_forest.md'}")


if __name__ == "__main__":
    main()

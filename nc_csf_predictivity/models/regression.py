"""Step 10: Regression head — predicts harmonized AUGRC per (eval row, candidate CSF).

Model: RidgeCV (alphas = 10^[-3..3], 7 values) with internal LOOCV.
Features:
  - 8 strict Papyan NC metrics (per protocol §6 primary feature set)
  - one-hot encoded categoricals: csf, source, regime
Target: `augrc_rank` (within-(source, eval_dataset) percentile rank from
        step 4; rank is the primary scheme per step-4 verdict).

Per (track, split), iterates over folds; trains on the fold's training rows;
predicts on the test rows; saves predictions and permutation-importance.

Inputs:
  outputs/<track>/dataset/long_harmonized.parquet
  outputs/splits/<split>.parquet

Outputs:
  outputs/<track>/<split>/regression/preds.parquet
      columns: split_name, fold_id, fold_label, model_id|cell_id,
               eval_dataset, regime, csf, predicted_score, true_score, raw_augrc
  outputs/<track>/<split>/regression/feature_importance.parquet
      columns: split_name, fold_id, fold_label, feature, importance_mean, importance_std
  outputs/08_regression_check.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]
NC_TRACK2 = [f"nc_mean_{c}" for c in NC_PRIMARY]
CAT_FEATURES = ["csf", "source", "regime"]
TARGET_COL = "augrc_rank"

ALPHAS = np.logspace(-3, 3, 7)
PERM_REPEATS = 5
PERM_SEED = 0
TRACK1_SPLITS = ["xarch", "lopo", "lodo_vgg13", "pxs_vgg13",
                 "single_vgg13", "xarch_vit_in", "lopo_cnn_only"]
TRACK2_SPLITS = ["track2_loo"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def add_track1_model_id(df: pd.DataFrame) -> pd.DataFrame:
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


def add_track2_cell_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell_id"] = (
        df["architecture"].astype(str) + "|"
        + df["paradigm"].astype(str) + "|"
        + df["source"].astype(str)
    )
    return df


def build_pipeline(nc_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("nc", StandardScaler(), nc_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         CAT_FEATURES),
    ])
    return Pipeline([
        ("preprocess", pre),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])


def feature_names(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    nc_names = list(pre.transformers_[0][2])
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    return nc_names + cat_names


def fit_predict_one_fold(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         id_col: str, nc_cols: list[str],
                         fold_id: int, fold_label: str, split_name: str
                         ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    feat_cols = nc_cols + CAT_FEATURES
    X_train = train_df[feat_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feat_cols]
    y_test = test_df[TARGET_COL]

    pipe = build_pipeline(nc_cols)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    out = test_df[[id_col, "eval_dataset", "regime", "csf"]].copy()
    out["split_name"] = split_name
    out["fold_id"] = fold_id
    out["fold_label"] = fold_label
    out["predicted_score"] = preds
    out["true_score"] = y_test.values
    out["raw_augrc"] = test_df["augrc"].values

    # Permutation importance — on raw input columns (NC + 3 categorical).
    pi = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=PERM_REPEATS, random_state=PERM_SEED,
        scoring="neg_mean_squared_error", n_jobs=1,
    )
    imp = pd.DataFrame({
        "split_name": split_name,
        "fold_id": fold_id,
        "fold_label": fold_label,
        "feature": feat_cols,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    })

    # Train R^2 for sanity (not the headline metric)
    train_r2 = pipe.score(X_train, y_train)
    return out, imp, train_r2


def run_track1(out_root: Path) -> dict:
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_track1_model_id(long_df)
    # Primary scope: OOD only (drop regime='test')
    long_ood = long_df[long_df["regime"] != "test"].reset_index(drop=True)

    summary = {}
    for split in TRACK1_SPLITS:
        sp_path = out_root / "splits" / f"{split}.parquet"
        if not sp_path.exists():
            print(f"  splits parquet missing: {sp_path}")
            continue
        sp = pq.read_table(sp_path).to_pandas()
        preds_pieces, imp_pieces, train_r2s = [], [], []
        for fold_id, fold_grp in sp.groupby("fold_id"):
            fold_label = fold_grp["fold_label"].iloc[0]
            train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
            test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
            train_df = long_ood[long_ood["model_id"].isin(train_ids)]
            test_df = long_ood[long_ood["model_id"].isin(test_ids)]
            if train_df.empty or test_df.empty:
                continue
            preds, imp, train_r2 = fit_predict_one_fold(
                train_df, test_df, "model_id", NC_PRIMARY,
                fold_id, fold_label, split,
            )
            preds_pieces.append(preds)
            imp_pieces.append(imp)
            train_r2s.append(train_r2)

        if not preds_pieces:
            continue
        out_dir = out_root / "track1" / split / "regression"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_preds = pd.concat(preds_pieces, ignore_index=True)
        all_imp = pd.concat(imp_pieces, ignore_index=True)
        write_parquet(all_preds, out_dir / "preds.parquet")
        write_parquet(all_imp, out_dir / "feature_importance.parquet")
        summary[split] = {
            "n_folds": len(preds_pieces),
            "n_test_rows": len(all_preds),
            "mean_train_r2": float(np.mean(train_r2s)),
        }
        print(f"  track1/{split}: {len(preds_pieces)} folds, "
              f"{len(all_preds):,} test rows, "
              f"train R² ≈ {np.mean(train_r2s):.3f}")
    return summary


def run_track2(out_root: Path) -> dict:
    long_df = pq.read_table(out_root / "track2" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_track2_cell_id(long_df)
    long_ood = long_df[long_df["regime"] != "test"].reset_index(drop=True)

    summary = {}
    for split in TRACK2_SPLITS:
        sp_path = out_root / "splits" / f"{split}.parquet"
        if not sp_path.exists():
            print(f"  splits parquet missing: {sp_path}")
            continue
        sp = pq.read_table(sp_path).to_pandas()
        preds_pieces, imp_pieces, train_r2s = [], [], []
        for fold_id, fold_grp in sp.groupby("fold_id"):
            fold_label = fold_grp["fold_label"].iloc[0]
            train_ids = fold_grp[fold_grp["role"] == "train"]["cell_id"].tolist()
            test_ids = fold_grp[fold_grp["role"] == "test"]["cell_id"].tolist()
            train_df = long_ood[long_ood["cell_id"].isin(train_ids)]
            test_df = long_ood[long_ood["cell_id"].isin(test_ids)]
            if train_df.empty or test_df.empty:
                continue
            preds, imp, train_r2 = fit_predict_one_fold(
                train_df, test_df, "cell_id", NC_TRACK2,
                fold_id, fold_label, split,
            )
            preds_pieces.append(preds)
            imp_pieces.append(imp)
            train_r2s.append(train_r2)

        if not preds_pieces:
            continue
        out_dir = out_root / "track2" / split / "regression"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_preds = pd.concat(preds_pieces, ignore_index=True)
        all_imp = pd.concat(imp_pieces, ignore_index=True)
        write_parquet(all_preds, out_dir / "preds.parquet")
        write_parquet(all_imp, out_dir / "feature_importance.parquet")
        summary[split] = {
            "n_folds": len(preds_pieces),
            "n_test_rows": len(all_preds),
            "mean_train_r2": float(np.mean(train_r2s)),
        }
        print(f"  track2/{split}: {len(preds_pieces)} folds, "
              f"{len(all_preds):,} test rows, "
              f"train R² ≈ {np.mean(train_r2s):.3f}")
    return summary


def report(t1: dict, t2: dict, out_path: Path, out_root: Path) -> None:
    lines = ["# Step 10 — Regression head\n\n"]
    lines.append("**Date:** 2026-05-03\n")
    lines.append("**Source:** `code/nc_csf_predictivity/models/regression.py`\n")
    lines.append("**Model:** `RidgeCV(alphas=10^[-3..3])`, internal LOOCV α selection.\n")
    lines.append(f"**Features:** 8 NC ({', '.join(NC_PRIMARY)}) + one-hot of {CAT_FEATURES}.\n")
    lines.append(
        "**Suffix convention (Papyan 2020):** `_uc` = computed from class "
        "means of last-layer activations (feature-space geometry); `_wc` = "
        "computed from classifier weight rows (weight-space geometry).\n"
    )
    lines.append(f"**Target:** `augrc_rank` (rank is primary per step-4 verdict).\n")
    lines.append(f"**Scope:** OOD only (regime ∈ {{near, mid, far}}); test regime excluded.\n\n")

    lines.append("## Worked example — one prediction row from `xarch`\n\n")
    xarch_pred_path = out_root / "track1" / "xarch" / "regression" / "preds.parquet"
    if xarch_pred_path.exists():
        preds = pq.read_table(xarch_pred_path).to_pandas()
        # Pick the first ResNet18 confidnet cifar10 row, eval=tinyimagenet
        s = preds[preds["model_id"].str.contains("ResNet18\\|confidnet\\|cifar10\\|1\\|0\\|2.2")]
        if not s.empty:
            ev = sorted(s["eval_dataset"].unique())[0]
            ex = s[s["eval_dataset"] == ev].head(8)
            lines.append(
                f"From the cross-arch test set, model row "
                f"`{ex['model_id'].iloc[0]}` evaluated on `{ev}` "
                f"(regime = `{ex['regime'].iloc[0]}`). The Ridge predicts a "
                "score per CSF (lower = better, since target is rank ∈ [0, 1]):\n\n"
            )
            ex_show = ex[["csf", "predicted_score", "true_score", "raw_augrc"]].round(3)
            lines.append("```\n" + ex_show.to_string(index=False) + "\n```\n\n")
            top1_pred = ex.sort_values("predicted_score").iloc[0]
            top1_true = ex.sort_values("true_score").iloc[0]
            lines.append(
                f"Top-1 predicted CSF: **{top1_pred['csf']}** "
                f"(predicted_score = {top1_pred['predicted_score']:.3f}, "
                f"raw AUGRC = {top1_pred['raw_augrc']:.2f}).\n"
                f"Top-1 true (oracle) CSF: **{top1_true['csf']}** "
                f"(true_score = {top1_true['true_score']:.3f}, "
                f"raw AUGRC = {top1_true['raw_augrc']:.2f}).\n"
                f"Top-1 regret on this eval row = "
                f"{top1_pred['raw_augrc'] - top1_true['raw_augrc']:.2f} "
                f"raw AUGRC (computed downstream in step 13).\n\n"
            )

    lines.append("## Worked example — feature importance computation\n\n")
    lines.append(
        "Permutation importance shuffles each input column on the test set "
        "and measures the increase in MSE (negative R²). Each NC and "
        "categorical feature is permuted "
        f"`n_repeats={PERM_REPEATS}` times with `random_state={PERM_SEED}`. "
        "The importance is the mean MSE increase; std is across repeats. "
        "Higher = more important.\n\n"
        "Toy: with 3 features (NC_1, NC_2, csf) and a Ridge fitted on 100 "
        "rows, suppose permuting NC_1 increases MSE from 0.20 to 0.27 (Δ = "
        "+0.07) and permuting csf increases it from 0.20 to 0.50 (Δ = +0.30). "
        "Then `csf` is much more important than `NC_1` for this model. "
        "Categoricals usually dominate when the OneHot encoding has many "
        "levels (here csf has 19 of them after dropping the first), so the "
        "*relative* ordering of NC features is the more interesting readout.\n\n"
    )
    imp_path = out_root / "track1" / "xarch" / "regression" / "feature_importance.parquet"
    if imp_path.exists():
        imp = pq.read_table(imp_path).to_pandas()
        lines.append("### Feature importance on `xarch` (ResNet18 test set)\n\n")
        nc_imp = (imp[imp["feature"].isin(NC_PRIMARY)]
                  .sort_values("importance_mean", ascending=False))
        cat_imp = imp[imp["feature"].isin(CAT_FEATURES)]
        lines.append("NC features (sorted by importance):\n\n")
        lines.append("```\n" + nc_imp[["feature","importance_mean","importance_std"]]
                     .round(4).to_string(index=False) + "\n```\n\n")
        if not cat_imp.empty:
            lines.append("Categorical features:\n\n")
            lines.append("```\n" + cat_imp[["feature","importance_mean","importance_std"]]
                         .round(4).to_string(index=False) + "\n```\n\n")

    lines.append("## Run summary\n\n")
    rows = []
    for split, info in t1.items():
        rows.append({"track": "1", "split": split, **info})
    for split, info in t2.items():
        rows.append({"track": "2", "split": split, **info})
    if rows:
        lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    lines.append(
        "`mean_train_r2` is reported for sanity only — it does not measure "
        "out-of-fold transfer. Test-fold metrics (top-1 / set / top-k regret, "
        "ranking, baselines) are computed in step 13.\n"
    )

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    print("Track 1:")
    t1 = run_track1(out_root)
    print("Track 2:")
    t2 = run_track2(out_root)

    report(t1, t2, out_root / "08_regression_check.md", out_root)
    print(f"wrote {out_root / '08_regression_check.md'}")


if __name__ == "__main__":
    main()

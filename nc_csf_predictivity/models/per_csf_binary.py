"""Step 12: Per-CSF binary ablation.

Structural counterpart to step 11's multi-output head. Differences:

1. **Per-CSF C tuning.** Each CSF's binary head is fitted with
   `LogisticRegressionCV(Cs=10, cv=5)` — each picks its own regularization.
   Step 11 used fixed `C=1.0` for tractability across 20 outputs ×
   ~30 (split, fold, rule) triples.

2. **No MultiOutput wrapper.** 20 independent `Pipeline`s, each saved
   separately. This makes "does cross-CSF coupling help?" testable via
   per-row McNemar on the same test rows (step 15).

Reduced scope: Track 1 splits limited to {xarch, lopo, lodo_vgg13} for
runtime tractability. Track 2 runs on track2_loo. Other splits can be added
later if a productive ablation question warrants the compute.

Outputs:
  outputs/<track>/<split>/per_csf_binary/<label_rule>/
    preds.parquet         — same schema as step 11
    coefficients.parquet  — adds `chosen_C` column per (csf, feature)
  outputs/10_per_csf_binary_check.md
"""
from __future__ import annotations

import argparse
import json
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

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
CODE_DIR = PIPELINE_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

# Reuse helpers from the multilabel module to guarantee identical featurization.
import sys
sys.path.insert(0, str(DATA_DIR))
from multilabel_competitive import (  # noqa: E402
    NC_PRIMARY, NC_TRACK2, CAT_FEATURES, TRAIN_REGIMES,
    add_track1_model_id, add_track2_cell_id,
    load_track1_label_rules, load_track2_label_rules,
    build_train_track1, build_test_track1,
    build_train_track2, build_test_track2,
    feature_names_after_preprocess,
)

LR_CS = 10
LR_CV = 5
LR_CV_TRACK2 = 3  # track 2 has only 44 training rows
TRACK1_SPLITS = ["xarch", "lopo", "lodo_vgg13"]
TRACK2_SPLITS = ["track2_loo"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def build_per_csf_pipeline(nc_cols: list[str], cv: int) -> Pipeline:
    pre = ColumnTransformer([
        ("nc", StandardScaler(), nc_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         CAT_FEATURES),
    ])
    base = LogisticRegressionCV(Cs=LR_CS, cv=cv, max_iter=2000,
                                scoring="neg_log_loss", n_jobs=1)
    return Pipeline([("preprocess", pre), ("clf", base)])


def fit_one_csf(X_tr: pd.DataFrame, y_tr: np.ndarray,
                X_te: pd.DataFrame, nc_cols: list[str], cv: int
                ) -> tuple[np.ndarray, dict, float] | None:
    """Returns None if the fit can't be performed (e.g., minority class too small)."""
    counts = np.bincount(y_tr, minlength=2)
    if min(counts) < cv:
        # LogisticRegressionCV will fail with all-one-label CV folds.
        return None
    try:
        pipe = build_per_csf_pipeline(nc_cols, cv)
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


def run_track1(out_root: Path) -> dict:
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_track1_model_id(long_df)
    rules = load_track1_label_rules(out_root)
    summary = {}

    for split in TRACK1_SPLITS:
        sp_path = out_root / "splits" / f"{split}.parquet"
        if not sp_path.exists():
            continue
        sp = pq.read_table(sp_path).to_pandas()

        for rule, label_df in rules.items():
            preds_pieces, coefs_pieces = [], []
            for fold_id, fold_grp in sp.groupby("fold_id"):
                fold_label = fold_grp["fold_label"].iloc[0]
                train_ids = fold_grp[fold_grp["role"] == "train"]["model_id"].tolist()
                test_ids = fold_grp[fold_grp["role"] == "test"]["model_id"].tolist()
                X_tr, Y_tr, _, csf_cols = build_train_track1(long_df, label_df, train_ids)
                X_te, meta_te = build_test_track1(long_df, test_ids)
                if X_tr is None or X_te is None or not csf_cols:
                    continue

                for csf in csf_cols:
                    y = Y_tr[csf].values
                    if len(np.unique(y)) < 2:
                        continue
                    res = fit_one_csf(X_tr, y, X_te, NC_PRIMARY, LR_CV)
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
                        "label_rule": rule,
                    })
                    preds_pieces.append(pred)
                    coef_rows = []
                    for fname, c in coef_dict.items():
                        if fname == "_chosen_C":
                            continue
                        coef_rows.append({
                            "csf": csf, "feature": fname,
                            "coefficient": c, "chosen_C": chosen_C,
                        })
                    coef_df = pd.DataFrame(coef_rows)
                    coef_df["split_name"] = split
                    coef_df["fold_id"] = fold_id
                    coef_df["fold_label"] = fold_label
                    coef_df["label_rule"] = rule
                    coefs_pieces.append(coef_df)

            if not preds_pieces:
                continue
            out_dir = out_root / "track1" / split / "per_csf_binary" / rule
            out_dir.mkdir(parents=True, exist_ok=True)
            all_preds = pd.concat(preds_pieces, ignore_index=True)
            all_coefs = pd.concat(coefs_pieces, ignore_index=True)
            write_parquet(all_preds, out_dir / "preds.parquet")
            write_parquet(all_coefs, out_dir / "coefficients.parquet")
            summary[(split, rule)] = {
                "n_folds": all_preds["fold_id"].nunique(),
                "n_csf_fits": all_coefs.groupby(["fold_id", "csf"]).ngroups,
                "n_preds": len(all_preds),
                "median_chosen_C": float(all_coefs["chosen_C"].median()),
            }
            print(f"  track1/{split}/{rule}: "
                  f"{summary[(split, rule)]['n_csf_fits']} CSF fits, "
                  f"{len(all_preds):,} pred rows, "
                  f"median C = {summary[(split, rule)]['median_chosen_C']:g}")
    return summary


def run_track2(out_root: Path) -> dict:
    long_df = pq.read_table(out_root / "track2" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_track2_cell_id(long_df)
    rules = load_track2_label_rules(out_root)
    summary = {}

    for split in TRACK2_SPLITS:
        sp_path = out_root / "splits" / f"{split}.parquet"
        if not sp_path.exists():
            continue
        sp = pq.read_table(sp_path).to_pandas()

        for rule, label_df in rules.items():
            preds_pieces, coefs_pieces = [], []
            for fold_id, fold_grp in sp.groupby("fold_id"):
                fold_label = fold_grp["fold_label"].iloc[0]
                train_ids = fold_grp[fold_grp["role"] == "train"]["cell_id"].tolist()
                test_ids = fold_grp[fold_grp["role"] == "test"]["cell_id"].tolist()
                X_tr, Y_tr, _, csf_cols = build_train_track2(long_df, label_df, train_ids)
                X_te, meta_te = build_test_track2(long_df, test_ids)
                if X_tr is None or X_te is None or not csf_cols:
                    continue
                for csf in csf_cols:
                    y = Y_tr[csf].values
                    if len(np.unique(y)) < 2:
                        continue
                    res = fit_one_csf(X_tr, y, X_te, NC_TRACK2, LR_CV_TRACK2)
                    if res is None:
                        continue
                    proba, coef_dict, chosen_C = res
                    pred = pd.DataFrame({
                        "cell_id": meta_te["cell_id"].values,
                        "regime": meta_te["regime"].values,
                        "csf": csf,
                        "p_competitive": proba,
                        "predicted_competitive": proba >= 0.5,
                        "split_name": split,
                        "fold_id": fold_id,
                        "fold_label": fold_label,
                        "label_rule": rule,
                    })
                    preds_pieces.append(pred)
                    coef_rows = []
                    for fname, c in coef_dict.items():
                        if fname == "_chosen_C":
                            continue
                        coef_rows.append({
                            "csf": csf, "feature": fname,
                            "coefficient": c, "chosen_C": chosen_C,
                        })
                    coef_df = pd.DataFrame(coef_rows)
                    coef_df["split_name"] = split
                    coef_df["fold_id"] = fold_id
                    coef_df["fold_label"] = fold_label
                    coef_df["label_rule"] = rule
                    coefs_pieces.append(coef_df)

            if not preds_pieces:
                continue
            out_dir = out_root / "track2" / split / "per_csf_binary" / rule
            out_dir.mkdir(parents=True, exist_ok=True)
            all_preds = pd.concat(preds_pieces, ignore_index=True)
            all_coefs = pd.concat(coefs_pieces, ignore_index=True)
            write_parquet(all_preds, out_dir / "preds.parquet")
            write_parquet(all_coefs, out_dir / "coefficients.parquet")
            summary[(split, rule)] = {
                "n_folds": all_preds["fold_id"].nunique(),
                "n_csf_fits": all_coefs.groupby(["fold_id", "csf"]).ngroups,
                "n_preds": len(all_preds),
                "median_chosen_C": float(all_coefs["chosen_C"].median()),
            }
            print(f"  track2/{split}/{rule}: "
                  f"{summary[(split, rule)]['n_csf_fits']} CSF fits, "
                  f"{len(all_preds):,} pred rows, "
                  f"median C = {summary[(split, rule)]['median_chosen_C']:g}")
    return summary


def report(t1: dict, t2: dict, out_root: Path, out_path: Path) -> None:
    lines = ["# Step 12 — Per-CSF binary ablation\n\n"]
    lines.append("**Date:** 2026-05-03\n")
    lines.append("**Source:** `code/nc_csf_predictivity/models/per_csf_binary.py`\n")
    lines.append(f"**Model:** `LogisticRegressionCV(Cs={LR_CS}, cv={LR_CV})` per CSF, "
                 "fitted independently. Track 2 uses cv=3 due to small sample size.\n\n")

    lines.append("## Worked example — per-CSF chosen C on `xarch/clique`\n\n")
    cp = out_root / "track1" / "xarch" / "per_csf_binary" / "clique" / "coefficients.parquet"
    if cp.exists():
        coefs = pq.read_table(cp).to_pandas()
        chosen = coefs[["csf", "chosen_C"]].drop_duplicates().sort_values("chosen_C")
        lines.append(
            "Each CSF's binary head selected its own C from "
            f"{LR_CS} log-spaced values via internal {LR_CV}-fold CV. "
            "Step 11's multilabel head used a single `C=1.0` for all CSFs.\n\n"
        )
        lines.append("```\n" + chosen.to_string(index=False) + "\n```\n\n")
        lo = chosen["chosen_C"].min()
        hi = chosen["chosen_C"].max()
        lines.append(f"Range: chosen C ∈ [{lo:g}, {hi:g}]. CSFs that picked "
                     "C far from 1.0 are the ones whose 'competitive' decision "
                     "boundary differs most from the multilabel head's fixed "
                     "regularization.\n\n")

    lines.append("## Worked example — predicted competitive set on the same xarch row\n\n")
    pp = out_root / "track1" / "xarch" / "per_csf_binary" / "clique" / "preds.parquet"
    pp_multi = out_root / "track1" / "xarch" / "multilabel_competitive" / "clique" / "preds.parquet"
    if pp.exists() and pp_multi.exists():
        per = pq.read_table(pp).to_pandas()
        multi = pq.read_table(pp_multi).to_pandas()
        target_id = "ResNet18|confidnet|cifar10|1|0|2.2"
        per_row = per[(per["model_id"] == target_id) & (per["regime"] == "near")]
        multi_row = multi[(multi["model_id"] == target_id) & (multi["regime"] == "near")]
        merged = per_row[["csf", "p_competitive", "predicted_competitive"]].merge(
            multi_row[["csf", "p_competitive", "predicted_competitive"]],
            on="csf", suffixes=("_per_csf", "_multi"))
        merged["agree"] = merged["predicted_competitive_per_csf"] == merged["predicted_competitive_multi"]
        merged = merged.sort_values("p_competitive_per_csf", ascending=False).round(3).head(10)
        lines.append(
            f"Same row as step 10/11 (`{target_id}`, regime=near). "
            "Side-by-side: per-CSF binary (this step) vs multilabel binary (step 11).\n\n"
        )
        lines.append("```\n" + merged.to_string(index=False) + "\n```\n\n")
        n_agree = int(merged["agree"].sum())
        n_total = len(merged)
        lines.append(f"Per-row agreement on `predicted_competitive`: "
                     f"{n_agree}/{n_total} CSFs in this top-10 view.\n\n")

    lines.append("## Run summary\n\n")
    rows = []
    for (split, rule), info in t1.items():
        rows.append({"track": "1", "split": split, "label_rule": rule, **info})
    for (split, rule), info in t2.items():
        rows.append({"track": "2", "split": split, "label_rule": rule, **info})
    if rows:
        lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    lines.append(
        "McNemar comparison of per-CSF vs multilabel predictions on the same "
        "test rows lives in step 15.\n"
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

    report(t1, t2, out_root, out_root / "10_per_csf_binary_check.md")
    print(f"wrote {out_root / '10_per_csf_binary_check.md'}")


if __name__ == "__main__":
    main()

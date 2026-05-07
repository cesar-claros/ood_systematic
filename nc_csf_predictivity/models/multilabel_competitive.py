"""Step 11: Multi-output binary 'is competitive' head.

For each (track, split, fold, label_rule):
  Train a MultiOutputClassifier(LogisticRegressionCV) with one binary head per
  CSF. Features = NC vector + one-hot of (source, regime). `csf` is NOT a
  feature — it is the output dimension.

Per-row construction:
  Training rows are per (model_id, regime) for the model_ids in the split's
  training pool, restricted to VGG13 because those are the only rows with
  reliable per-config labels (Tracks 1: from steps 5–6; Track 2: from
  published per-paradigm JSONs, which are also constructed from VGG13/ViT).
  Each cell's binary label vector broadcasts across its 5 runs.
  Test rows are per (model_id, regime) for the test model_ids, regardless of
  architecture — the predictor outputs probabilities for any architecture.

Label rules:
  Track 1: clique, within_eps_raw, within_eps_rank, within_eps_majority,
           within_eps_unanimous.
  Track 2: clique (from published per-paradigm JSONs).

Outputs:
  outputs/<track>/<split>/multilabel_competitive/<label_rule>/
    preds.parquet         — split, fold, model|cell id, regime, csf, p, predicted_competitive
    coefficients.parquet  — per (csf, feature) logistic coefficients
  outputs/09_multilabel_competitive_check.md
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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
CODE_DIR = PIPELINE_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]
NC_TRACK2 = [f"nc_mean_{c}" for c in NC_PRIMARY]
CAT_FEATURES = ["source", "regime"]
TRAIN_REGIMES = ["near", "mid", "far", "all"]
TRACK1_SPLITS = ["xarch", "lopo", "lodo_vgg13", "pxs_vgg13",
                 "single_vgg13", "lopo_cnn_only"]
# xarch_vit_in skipped: VGG13 binary-head training data is identical to xarch.
TRACK2_SPLITS = ["track2_loo"]

LR_C = 1.0  # fixed regularization; CV across 20 binary heads × all (split, fold, rule) triples is too slow

CONV_PARADIGMS = ["confidnet", "devries", "dg"]
ALL_PARADIGMS = ["confidnet", "devries", "dg", "modelvit"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


# ---- ID helpers (must match splits.py exactly) ----

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


# ---- Label loading ----

def load_track1_label_rules(out_root: Path) -> dict[str, pd.DataFrame]:
    cliques = pq.read_table(out_root / "track1" / "cliques" / "cliques.parquet").to_pandas()
    within = pq.read_table(out_root / "track1" / "labels" / "within_eps.parquet").to_pandas()

    rules = {}
    rules["clique"] = (cliques[["paradigm","source","dropout","reward","regime","csf","in_top_clique"]]
                       .rename(columns={"in_top_clique": "label"}))
    for src_col, name in [
        ("in_within_eps_set_raw", "within_eps_raw"),
        ("in_within_eps_set_rank", "within_eps_rank"),
        ("in_within_eps_set_majority", "within_eps_majority"),
        ("in_within_eps_set_unanimous", "within_eps_unanimous"),
    ]:
        rules[name] = (within[["paradigm","source","dropout","reward","regime","csf",src_col]]
                       .rename(columns={src_col: "label"}))
    for k in rules:
        rules[k]["label"] = rules[k]["label"].astype(int)
    return rules


def load_track2_label_rules(out_root: Path) -> dict[str, pd.DataFrame]:
    """Track 2 label = published per-paradigm clique membership.
    Build long table: (architecture, paradigm, source, regime, csf, label)."""
    rules_dir = CODE_DIR / "ood_eval_outputs"
    pieces = []
    for paradigm, fn in [("confidnet", "top_cliques_Conv_False_RC_confidnet_cliques.json"),
                         ("devries", "top_cliques_Conv_False_RC_devries_cliques.json"),
                         ("dg", "top_cliques_Conv_False_RC_dg_cliques.json"),
                         ("modelvit", "top_cliques_ViT_False_RC_cliques.json")]:
        path = rules_dir / fn
        if not path.exists():
            print(f"  missing {path}")
            continue
        with open(path) as f:
            d = json.load(f)
        arch = "ViT" if paradigm == "modelvit" else "VGG13"
        for source, regimes in d.items():
            if source.startswith("_"):
                continue
            for regime, csfs in regimes.items():
                for csf in csfs:
                    pieces.append({"architecture": arch, "paradigm": paradigm,
                                   "source": source, "regime": regime,
                                   "csf": csf, "label": 1})
    cliques = pd.DataFrame(pieces)
    return {"clique": cliques}


# ---- Build features and labels ----

def get_unique_model_features(long_df: pd.DataFrame, nc_cols: list[str], id_col: str) -> pd.DataFrame:
    keep = [id_col, "architecture", "paradigm", "source", "run", "dropout", "reward"] + nc_cols
    keep = [c for c in keep if c in long_df.columns]
    return long_df[keep].drop_duplicates(subset=[id_col]).reset_index(drop=True)


def build_train_track1(long_df: pd.DataFrame, label_df: pd.DataFrame,
                       train_model_ids: list[str]) -> tuple:
    nc_per_model = get_unique_model_features(long_df, NC_PRIMARY, "model_id")
    # restrict to VGG13 IDs that are in train pool
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

    X = merged[NC_PRIMARY + CAT_FEATURES]
    Y = merged[csf_cols]
    meta = merged[["model_id", "architecture", "paradigm", "source",
                   "run", "dropout", "reward", "regime"]]
    return X, Y, meta, csf_cols


def build_test_track1(long_df: pd.DataFrame, test_model_ids: list[str]) -> tuple:
    nc_per_model = get_unique_model_features(long_df, NC_PRIMARY, "model_id")
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
    X = feats[NC_PRIMARY + CAT_FEATURES]
    meta = feats[["model_id", "architecture", "paradigm", "source",
                  "run", "dropout", "reward", "regime"]]
    return X, meta


def build_train_track2(long_df: pd.DataFrame, label_df: pd.DataFrame,
                       train_cell_ids: list[str]) -> tuple:
    cell_keys = ["architecture", "paradigm", "source"]
    nc_per_cell = (long_df[[*cell_keys, "cell_id"] + NC_TRACK2]
                   .drop_duplicates(subset=["cell_id"]).reset_index(drop=True))
    pool = nc_per_cell[nc_per_cell["cell_id"].isin(train_cell_ids)]
    if pool.empty:
        return None, None, None, []
    rows = []
    for _, c in pool.iterrows():
        for regime in TRAIN_REGIMES:
            r = c.to_dict()
            r["regime"] = regime
            rows.append(r)
    feats = pd.DataFrame(rows)

    label_wide = (label_df.pivot_table(
        index=cell_keys + ["regime"], columns="csf",
        values="label", aggfunc="first").reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns if c not in cell_keys + ["regime"]]
    label_wide[csf_cols] = label_wide[csf_cols].astype(int)

    merged = feats.merge(label_wide, on=cell_keys + ["regime"], how="inner")
    if merged.empty:
        return None, None, None, []

    X = merged[NC_TRACK2 + CAT_FEATURES]
    Y = merged[csf_cols]
    meta = merged[["cell_id", "architecture", "paradigm", "source", "regime"]]
    return X, Y, meta, csf_cols


def build_test_track2(long_df: pd.DataFrame, test_cell_ids: list[str]) -> tuple:
    cell_keys = ["architecture", "paradigm", "source"]
    nc_per_cell = (long_df[[*cell_keys, "cell_id"] + NC_TRACK2]
                   .drop_duplicates(subset=["cell_id"]).reset_index(drop=True))
    pool = nc_per_cell[nc_per_cell["cell_id"].isin(test_cell_ids)]
    rows = []
    for _, c in pool.iterrows():
        for regime in TRAIN_REGIMES:
            r = c.to_dict()
            r["regime"] = regime
            rows.append(r)
    feats = pd.DataFrame(rows)
    if feats.empty:
        return None, None
    X = feats[NC_TRACK2 + CAT_FEATURES]
    meta = feats[["cell_id", "architecture", "paradigm", "source", "regime"]]
    return X, meta


# ---- Pipeline ----

def build_pipeline(nc_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("nc", StandardScaler(), nc_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         CAT_FEATURES),
    ])
    base = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
    return Pipeline([("preprocess", pre),
                     ("clf", MultiOutputClassifier(base, n_jobs=1))])


def feature_names_after_preprocess(pipe: Pipeline) -> list[str]:
    pre = pipe.named_steps["preprocess"]
    nc_names = list(pre.transformers_[0][2])
    ohe = pre.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CAT_FEATURES))
    return nc_names + cat_names


def predict_proba_to_long(pipe: Pipeline, X: pd.DataFrame, meta: pd.DataFrame,
                          csf_cols: list[str], id_col: str) -> pd.DataFrame:
    proba_per_csf = pipe.predict_proba(X)  # list of arrays [n_rows, 2]
    n = len(meta)
    rows = []
    for i in range(n):
        meta_i = meta.iloc[i]
        for j, csf in enumerate(csf_cols):
            p1 = float(proba_per_csf[j][i, 1])
            rows.append({
                id_col: meta_i[id_col],
                "regime": meta_i["regime"],
                "csf": csf,
                "p_competitive": p1,
                "predicted_competitive": p1 >= 0.5,
            })
    return pd.DataFrame(rows)


def coefficients_to_long(pipe: Pipeline, csf_cols: list[str]) -> pd.DataFrame:
    moc = pipe.named_steps["clf"]
    feat_names = feature_names_after_preprocess(pipe)
    rows = []
    for i, est in enumerate(moc.estimators_):
        csf = csf_cols[i]
        coefs = est.coef_[0]
        intercept = float(est.intercept_[0])
        rows.append({"csf": csf, "feature": "(intercept)",
                     "coefficient": intercept})
        for j, fname in enumerate(feat_names):
            rows.append({"csf": csf, "feature": fname,
                         "coefficient": float(coefs[j])})
    return pd.DataFrame(rows)


# ---- Run drivers ----

def run_track1(out_root: Path) -> dict:
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_track1_model_id(long_df)
    rules = load_track1_label_rules(out_root)
    summary = {}

    for split in TRACK1_SPLITS:
        sp_path = out_root / "splits" / f"{split}.parquet"
        if not sp_path.exists():
            print(f"  splits parquet missing: {sp_path}")
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
                # Drop CSFs that are constant in training (e.g., always 0)
                keep_csfs = [c for c in csf_cols if Y_tr[c].nunique() > 1]
                if not keep_csfs:
                    continue
                Y_tr = Y_tr[keep_csfs]
                pipe = build_pipeline(NC_PRIMARY)
                pipe.fit(X_tr, Y_tr.values)

                preds = predict_proba_to_long(pipe, X_te, meta_te, keep_csfs, "model_id")
                preds["split_name"] = split
                preds["fold_id"] = fold_id
                preds["fold_label"] = fold_label
                preds["label_rule"] = rule
                preds_pieces.append(preds)

                coefs = coefficients_to_long(pipe, keep_csfs)
                coefs["split_name"] = split
                coefs["fold_id"] = fold_id
                coefs["fold_label"] = fold_label
                coefs["label_rule"] = rule
                coefs_pieces.append(coefs)

            if not preds_pieces:
                continue
            out_dir = out_root / "track1" / split / "multilabel_competitive" / rule
            out_dir.mkdir(parents=True, exist_ok=True)
            all_preds = pd.concat(preds_pieces, ignore_index=True)
            all_coefs = pd.concat(coefs_pieces, ignore_index=True)
            write_parquet(all_preds, out_dir / "preds.parquet")
            write_parquet(all_coefs, out_dir / "coefficients.parquet")
            summary[(split, rule)] = {
                "n_folds": len(preds_pieces),
                "n_preds": len(all_preds),
                "n_csfs_avg": float(all_coefs.groupby("fold_id")["csf"].nunique().mean()),
            }
            print(f"  track1/{split}/{rule}: {len(preds_pieces)} folds, "
                  f"{len(all_preds):,} pred rows, "
                  f"~{summary[(split, rule)]['n_csfs_avg']:.1f} CSFs/fold")
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
                keep_csfs = [c for c in csf_cols if Y_tr[c].nunique() > 1]
                if not keep_csfs:
                    continue
                Y_tr = Y_tr[keep_csfs]
                pipe = build_pipeline(NC_TRACK2)
                # Track 2 has only 11 training cells × 4 regimes = 44 rows.
                # LogisticRegressionCV's default cv=5 is fine, but might be tight.
                pipe.fit(X_tr, Y_tr.values)

                preds = predict_proba_to_long(pipe, X_te, meta_te, keep_csfs, "cell_id")
                preds["split_name"] = split
                preds["fold_id"] = fold_id
                preds["fold_label"] = fold_label
                preds["label_rule"] = rule
                preds_pieces.append(preds)

                coefs = coefficients_to_long(pipe, keep_csfs)
                coefs["split_name"] = split
                coefs["fold_id"] = fold_id
                coefs["fold_label"] = fold_label
                coefs["label_rule"] = rule
                coefs_pieces.append(coefs)

            if not preds_pieces:
                continue
            out_dir = out_root / "track2" / split / "multilabel_competitive" / rule
            out_dir.mkdir(parents=True, exist_ok=True)
            all_preds = pd.concat(preds_pieces, ignore_index=True)
            all_coefs = pd.concat(coefs_pieces, ignore_index=True)
            write_parquet(all_preds, out_dir / "preds.parquet")
            write_parquet(all_coefs, out_dir / "coefficients.parquet")
            summary[(split, rule)] = {
                "n_folds": len(preds_pieces),
                "n_preds": len(all_preds),
                "n_csfs_avg": float(all_coefs.groupby("fold_id")["csf"].nunique().mean()),
            }
            print(f"  track2/{split}/{rule}: {len(preds_pieces)} folds, "
                  f"{len(all_preds):,} pred rows, "
                  f"~{summary[(split, rule)]['n_csfs_avg']:.1f} CSFs/fold")
    return summary


def report(t1: dict, t2: dict, out_root: Path, out_path: Path) -> None:
    lines = ["# Step 11 — Multi-output binary 'is competitive' head\n\n"]
    lines.append("**Date:** 2026-05-03\n")
    lines.append("**Source:** `code/nc_csf_predictivity/models/multilabel_competitive.py`\n")
    lines.append(f"**Model:** `MultiOutputClassifier(LogisticRegression(C={LR_C}, lbfgs))`. "
                 "Fixed C — internal CV across 20 binary heads × every "
                 "(split, fold, rule) is prohibitively slow; the per-CSF "
                 "ablation in step 12 will sweep C per CSF.\n")
    lines.append("**Features:** 8 NC + one-hot of (source, regime). `csf` is the OUTPUT dimension.\n\n")

    lines.append("## Worked example — predicted competitive set on one xarch row\n\n")
    pp = out_root / "track1" / "xarch" / "multilabel_competitive" / "clique" / "preds.parquet"
    if pp.exists():
        preds = pq.read_table(pp).to_pandas()
        # Pick same row as regression worked example
        s = preds[(preds["model_id"] == "ResNet18|confidnet|cifar10|1|0|2.2")
                  & (preds["regime"] == "near")]
        if not s.empty:
            s = s.sort_values("p_competitive", ascending=False)
            lines.append(
                f"From the cross-arch test set, model row "
                f"`ResNet18|confidnet|cifar10|1|0|2.2` at regime = `near`. "
                f"Each of the 20 binary heads outputs a probability that the "
                f"corresponding CSF is competitive on this row. Predicted "
                f"competitive set = `{{p ≥ 0.5}}`.\n\n"
            )
            show = s[["csf", "p_competitive", "predicted_competitive"]].head(10).round(3)
            lines.append("```\n" + show.to_string(index=False) + "\n```\n\n")
            n_pos = int(s["predicted_competitive"].sum())
            top = s[s["predicted_competitive"]]["csf"].tolist()
            lines.append(
                f"Predicted competitive set on this row (`label_rule=clique`): "
                f"`{top}` (size {n_pos}).\n\n"
                "Set-regret on this row will be `min(raw_augrc) over the "
                "predicted set − oracle raw_augrc`, computed in step 13.\n\n"
            )

    lines.append("## Worked example — per-CSF logistic coefficients (NC features only)\n\n")
    cp = out_root / "track1" / "xarch" / "multilabel_competitive" / "clique" / "coefficients.parquet"
    if cp.exists():
        coefs = pq.read_table(cp).to_pandas()
        nc_only = coefs[coefs["feature"].isin(NC_PRIMARY)]
        # Rank top 5 NC features by mean abs coefficient across CSFs
        rank = (nc_only.groupby("feature")["coefficient"]
                .apply(lambda s: float(np.mean(np.abs(s))))
                .sort_values(ascending=False).round(4))
        lines.append("Mean |coefficient| across CSFs, per NC feature (xarch fold, "
                     "label_rule=clique):\n\n")
        lines.append("```\n" + rank.to_string() + "\n```\n\n")
        # Show one CSF's full coefficient row for illustration
        ex_csf = "CTM"
        if ex_csf in coefs["csf"].unique():
            row = coefs[(coefs["csf"] == ex_csf) & coefs["feature"].isin(NC_PRIMARY)]
            row = row.sort_values("coefficient", key=lambda s: s.abs(), ascending=False).round(3)
            lines.append(f"Per-NC coefficients for the **{ex_csf}** binary head:\n\n")
            lines.append("```\n" + row[["feature", "coefficient"]].to_string(index=False) + "\n```\n\n")
            lines.append(
                "Positive coefficient ⇒ higher value of that NC metric "
                "increases the predicted probability that this CSF is "
                "competitive. Magnitude is on the standardized-feature scale.\n\n"
            )

    lines.append("## Run summary\n\n")
    rows = []
    for (split, rule), info in t1.items():
        rows.append({"track": "1", "split": split, "label_rule": rule, **info})
    for (split, rule), info in t2.items():
        rows.append({"track": "2", "split": split, "label_rule": rule, **info})
    if rows:
        lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n\n")
    lines.append(
        "`n_csfs_avg` is the average number of binary heads actually trained "
        "per fold (some CSFs may be constant=0 in a given fold's training "
        "labels, in which case their head is skipped — e.g., a CSF that is "
        "never in any cliquer/within-ε set across the training cells of that "
        "fold).\n\n"
        "Predictor outputs `predicted_competitive` (bool, threshold 0.5 on "
        "`p_competitive`). Set-regret and per-side metrics will be computed "
        "in step 13 by joining these predictions with the oracle table from "
        "step 7.\n"
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

    report(t1, t2, out_root, out_root / "09_multilabel_competitive_check.md")
    print(f"wrote {out_root / '09_multilabel_competitive_check.md'}")


if __name__ == "__main__":
    main()

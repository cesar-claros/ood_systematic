"""Step 13: Regret + ranking metrics with bootstrap CIs.

Per-eval-row metrics from each predictor's outputs joined with the oracle
table from step 7, then aggregated per (split, regime, side, predictor,
label_rule) with 95% CI from a bootstrap over eval rows (n=2000, seed=0).

Per-eval-row metrics
====================

REGRESSION  (predictor=regression, no label_rule)
  - top1_csf, top1_regret_raw, top1_regret_norm
  - top3_regret_raw, top5_regret_raw  (min of raw_augrc over predicted top-k)
  - spearman_rho, kendall_tau, mrr_oracle
    (where MRR = 1 / rank of oracle CSF in predicted ranking)

BINARY  (predictor=multilabel_competitive | per_csf_binary, with label_rule)
  - set_size                   (|predicted competitive set ∩ side|)
  - set_regret_raw / norm       (min raw_augrc over set − oracle_augrc;
                                divided by worst − oracle for norm)
  - empty_set                   (True iff |predicted set ∩ side| = 0)

Side stratification
===================
Each row is computed three times: side ∈ {all, head, feature}. For head/
feature the candidate pool is restricted to that side's CSFs and the oracle
/ worst columns are taken from the per-side fields in oracle.parquet.

Outputs
=======
  outputs/<track>/<split>/metrics/per_row.parquet   (one row per eval × side
       × predictor × label_rule)
  outputs/<track>/<split>/metrics/aggregate.parquet (per regime × side ×
       predictor × label_rule with bootstrap CIs)
  outputs/11_metrics_check.md
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import kendalltau, spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

HEAD_SIDE_CSFS = {
    "REN", "PE", "PCE", "MSR", "GEN", "MLS", "GE",
    "GradNorm", "Energy", "Confidence", "pNML",
}
FEATURE_SIDE_CSFS = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}
SIDES = ["all", "head", "feature"]
N_BOOT = 2000
SEED = 0
LABEL_RULES_BIN = ["clique", "within_eps_raw", "within_eps_rank",
                   "within_eps_majority", "within_eps_unanimous"]
TRACK1_SPLITS = ["xarch", "lopo", "lodo_vgg13", "pxs_vgg13",
                 "single_vgg13", "lopo_cnn_only"]
TRACK2_SPLITS = ["track2_loo"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def filter_to_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if side == "all":
        return df
    if side == "head":
        return df[df["csf"].isin(HEAD_SIDE_CSFS)]
    if side == "feature":
        return df[df["csf"].isin(FEATURE_SIDE_CSFS)]
    raise ValueError(side)


def oracle_for_side(oracle: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return oracle/worst columns appropriate for the side stratum."""
    if side == "all":
        return oracle.assign(
            o_csf=oracle["oracle_csf"], o_augrc=oracle["oracle_augrc"],
            w_csf=oracle["worst_csf"], w_augrc=oracle["worst_augrc"],
        )
    if side == "head":
        return oracle.assign(
            o_csf=oracle["oracle_csf_head"], o_augrc=oracle["oracle_augrc_head"],
            w_csf=oracle["worst_csf_head"], w_augrc=oracle["worst_augrc_head"],
        )
    if side == "feature":
        return oracle.assign(
            o_csf=oracle["oracle_csf_feature"], o_augrc=oracle["oracle_augrc_feature"],
            w_csf=oracle["worst_csf_feature"], w_augrc=oracle["worst_augrc_feature"],
        )
    raise ValueError(side)


# ---- Regression metrics ----

def regression_per_row(preds: pd.DataFrame, oracle_side: pd.DataFrame,
                       id_col: str) -> pd.DataFrame:
    """preds columns required: id_col, eval_dataset, csf, predicted_score, raw_augrc.
    oracle_side columns: id_col, eval_dataset, regime, o_csf, o_augrc, w_csf, w_augrc."""
    if preds.empty:
        return pd.DataFrame()

    # Sort within each (id_col, eval_dataset) by predicted_score ascending
    p_sorted = preds.sort_values([id_col, "eval_dataset", "predicted_score"]).copy()
    p_sorted["pred_rank"] = (p_sorted.groupby([id_col, "eval_dataset"])
                             .cumcount() + 1)

    # Top-1 row per (id, eval)
    top1 = p_sorted[p_sorted["pred_rank"] == 1][
        [id_col, "eval_dataset", "csf", "raw_augrc"]
    ].rename(columns={"csf": "top1_csf", "raw_augrc": "top1_augrc"})

    # Top-k min raw_augrc
    def topk(df, k):
        return (df[df["pred_rank"] <= k]
                .groupby([id_col, "eval_dataset"])["raw_augrc"].min()
                .rename(f"top{k}_min_augrc").reset_index())

    top3 = topk(p_sorted, 3)
    top5 = topk(p_sorted, 5)

    metrics = (oracle_side[[id_col, "eval_dataset", "regime",
                            "o_csf", "o_augrc", "w_csf", "w_augrc"]]
               .merge(top1, on=[id_col, "eval_dataset"], how="inner")
               .merge(top3, on=[id_col, "eval_dataset"], how="left")
               .merge(top5, on=[id_col, "eval_dataset"], how="left"))
    if metrics.empty:
        return pd.DataFrame()

    metrics["top1_regret_raw"] = metrics["top1_augrc"] - metrics["o_augrc"]
    denom = (metrics["w_augrc"] - metrics["o_augrc"]).replace(0, np.nan)
    metrics["top1_regret_norm"] = metrics["top1_regret_raw"] / denom
    metrics["top3_regret_raw"] = (metrics["top3_min_augrc"] - metrics["o_augrc"]).clip(lower=0)
    metrics["top5_regret_raw"] = (metrics["top5_min_augrc"] - metrics["o_augrc"]).clip(lower=0)

    # Per-(id, eval) Spearman/Kendall + MRR
    rows = []
    grp = p_sorted.groupby([id_col, "eval_dataset"])
    for (uid, ev), g in grp:
        if len(g) < 2:
            continue
        rho, _ = spearmanr(g["predicted_score"].values, g["raw_augrc"].values)
        tau, _ = kendalltau(g["predicted_score"].values, g["raw_augrc"].values)
        # MRR of oracle: oracle_csf is row's o_csf
        oracle_csf_for_row = oracle_side[(oracle_side[id_col] == uid)
                                         & (oracle_side["eval_dataset"] == ev)]["o_csf"]
        if oracle_csf_for_row.empty:
            continue
        ocsf = oracle_csf_for_row.iloc[0]
        ranks = g.set_index("csf")["pred_rank"]
        mrr = (1.0 / float(ranks.loc[ocsf])) if ocsf in ranks.index else np.nan
        rows.append({id_col: uid, "eval_dataset": ev,
                     "spearman_rho": rho, "kendall_tau": tau, "mrr": mrr})
    rk = pd.DataFrame(rows)
    if not rk.empty:
        metrics = metrics.merge(rk, on=[id_col, "eval_dataset"], how="left")
    return metrics


# ---- Binary set-regret metrics ----

def binary_per_row(preds_bin: pd.DataFrame, eval_long: pd.DataFrame,
                   oracle_side: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """preds_bin: id_col, regime, csf, predicted_competitive (bool).
    eval_long: id_col, eval_dataset, regime, csf, raw_augrc, side info.
    oracle_side: id_col, eval_dataset, regime, o_*, w_*."""
    if preds_bin.empty or eval_long.empty:
        return pd.DataFrame()

    # Restrict eval rows and oracle to the model/cell IDs that the predictor
    # actually scored — otherwise non-test rows get empty predicted sets.
    test_ids = preds_bin[id_col].unique()
    eval_long = eval_long[eval_long[id_col].isin(test_ids)]
    oracle_side = oracle_side[oracle_side[id_col].isin(test_ids)]
    if eval_long.empty:
        return pd.DataFrame()

    # Join: per (id, eval, csf) get predicted_competitive (from same regime)
    # eval_long already filtered to side-relevant CSFs.
    joined = eval_long.merge(preds_bin[[id_col, "regime", "csf",
                                         "predicted_competitive"]],
                             on=[id_col, "regime", "csf"], how="left")
    joined["predicted_competitive"] = joined["predicted_competitive"].fillna(False)

    # Set size per (id, eval) — count distinct CSFs predicted competitive
    set_size = (joined[joined["predicted_competitive"]]
                .groupby([id_col, "eval_dataset"])["csf"].nunique()
                .rename("set_size").reset_index())

    # min raw_augrc within predicted set per (id, eval)
    set_min = (joined[joined["predicted_competitive"]]
               .groupby([id_col, "eval_dataset"])["raw_augrc"].min()
               .rename("set_min_augrc").reset_index())

    base = oracle_side[[id_col, "eval_dataset", "regime",
                        "o_csf", "o_augrc", "w_csf", "w_augrc"]]
    metrics = (base.merge(set_min, on=[id_col, "eval_dataset"], how="left")
                   .merge(set_size, on=[id_col, "eval_dataset"], how="left"))
    metrics["set_size"] = metrics["set_size"].fillna(0).astype(int)
    metrics["empty_set"] = metrics["set_size"] == 0
    metrics["set_regret_raw"] = (metrics["set_min_augrc"] - metrics["o_augrc"]).clip(lower=0)
    denom = (metrics["w_augrc"] - metrics["o_augrc"]).replace(0, np.nan)
    metrics["set_regret_norm"] = metrics["set_regret_raw"] / denom
    return metrics


# ---- Bootstrap aggregation ----

def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT,
                      seed: int = SEED) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def aggregate_metrics(per_row_long: pd.DataFrame) -> pd.DataFrame:
    """Group by (predictor, label_rule, regime, side) and bootstrap mean CIs."""
    if per_row_long.empty:
        return pd.DataFrame()
    rows = []
    metric_cols = ["top1_regret_raw", "top1_regret_norm",
                   "top3_regret_raw", "top5_regret_raw",
                   "set_regret_raw", "set_regret_norm", "set_size",
                   "spearman_rho", "kendall_tau", "mrr"]
    keys = ["predictor", "label_rule", "regime", "side"]
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


# ---- Per-track drivers ----

def add_id_track1(df: pd.DataFrame) -> pd.DataFrame:
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


def add_id_track2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cell_id"] = (
        df["architecture"].astype(str) + "|"
        + df["paradigm"].astype(str) + "|"
        + df["source"].astype(str)
    )
    return df


def run_one_split(track: int, split: str, out_root: Path,
                  long_df: pd.DataFrame, oracle_df: pd.DataFrame,
                  id_col: str) -> int:
    metrics_dir = out_root / f"track{track}" / split / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    per_row_pieces = []

    # Regression
    reg_path = out_root / f"track{track}" / split / "regression" / "preds.parquet"
    if reg_path.exists():
        reg = pq.read_table(reg_path).to_pandas()
        for side in SIDES:
            reg_side = filter_to_side(reg, side)
            oracle_side = oracle_for_side(oracle_df, side)
            m = regression_per_row(reg_side, oracle_side, id_col)
            if m.empty:
                continue
            m["predictor"] = "regression"
            m["label_rule"] = ""
            m["side"] = side
            per_row_pieces.append(m)

    # Binary heads
    for predictor_dir, predictor_name in [
        ("multilabel_competitive", "multilabel"),
        ("per_csf_binary", "per_csf_binary"),
    ]:
        for rule in LABEL_RULES_BIN:
            p_path = (out_root / f"track{track}" / split / predictor_dir
                      / rule / "preds.parquet")
            if not p_path.exists():
                continue
            preds = pq.read_table(p_path).to_pandas()
            # Build eval_long with the side info
            long_with_side = long_df.copy()
            long_with_side["csf_side"] = long_with_side["csf"].apply(
                lambda c: "head" if c in HEAD_SIDE_CSFS
                else "feature" if c in FEATURE_SIDE_CSFS
                else "other"
            )
            for side in SIDES:
                eval_side = filter_to_side(long_with_side, side)
                oracle_side = oracle_for_side(oracle_df, side)
                m = binary_per_row(preds, eval_side, oracle_side, id_col)
                if m.empty:
                    continue
                m["predictor"] = predictor_name
                m["label_rule"] = rule
                m["side"] = side
                per_row_pieces.append(m)

    if not per_row_pieces:
        return 0
    all_per_row = pd.concat(per_row_pieces, ignore_index=True)
    write_parquet(all_per_row, metrics_dir / "per_row.parquet")
    agg = aggregate_metrics(all_per_row)
    write_parquet(agg, metrics_dir / "aggregate.parquet")
    print(f"  track{track}/{split}: {len(all_per_row):,} per-row rows, "
          f"{len(agg)} aggregate rows")
    return len(all_per_row)


def run_track1(out_root: Path) -> None:
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_id_track1(long_df)
    long_df = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})

    oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_id_track1(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    for split in TRACK1_SPLITS:
        run_one_split(1, split, out_root, long_df, oracle_df, "model_id")


def run_track2(out_root: Path) -> None:
    long_df = pq.read_table(out_root / "track2" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_id_track2(long_df)
    long_df = long_df[long_df["regime"] != "test"][[
        "cell_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})

    oracle_df = pq.read_table(out_root / "track2" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_id_track2(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    for split in TRACK2_SPLITS:
        run_one_split(2, split, out_root, long_df, oracle_df, "cell_id")


def report(out_root: Path, out_path: Path) -> None:
    lines = ["# Step 13 — Regret + ranking metrics\n\n"]
    lines.append("**Date:** 2026-05-03\n")
    lines.append("**Source:** `code/nc_csf_predictivity/evaluation/regret.py`\n")
    lines.append(f"**Bootstrap:** n={N_BOOT}, seed={SEED}, percentile 95% CI on the mean.\n\n")

    lines.append("## Worked example — top-1 / set / per-side regret on one xarch row\n\n")
    pr = out_root / "track1" / "xarch" / "metrics" / "per_row.parquet"
    if pr.exists():
        per_row = pq.read_table(pr).to_pandas()
        target = "ResNet18|confidnet|cifar10|1|0|2.2"
        s = per_row[(per_row["model_id"] == target)
                    & (per_row["eval_dataset"] == "cifar100")]
        if not s.empty:
            lines.append(
                f"Same row as steps 10–12: `{target}`, eval=`cifar100`, "
                "regime=`near`. Showing all (predictor, label_rule, side) "
                "combinations evaluated on this row.\n\n"
            )
            cols = ["predictor", "label_rule", "side", "o_csf", "o_augrc",
                    "w_augrc", "top1_csf", "top1_regret_raw", "top1_regret_norm",
                    "set_size", "set_regret_raw"]
            present = [c for c in cols if c in s.columns]
            lines.append("```\n" + s[present].round(3).to_string(index=False) + "\n```\n\n")

    lines.append("## Aggregate — `xarch` (Track 1, primary headline)\n\n")
    ag = out_root / "track1" / "xarch" / "metrics" / "aggregate.parquet"
    if ag.exists():
        agg = pq.read_table(ag).to_pandas()
        # Highlight top-1 regret and set regret per (regime, side, predictor, label_rule)
        cols = ["predictor", "label_rule", "regime", "side", "n",
                "top1_regret_raw_mean", "top1_regret_raw_ci_lo", "top1_regret_raw_ci_hi",
                "top1_regret_norm_mean",
                "set_regret_raw_mean", "set_regret_raw_ci_lo", "set_regret_raw_ci_hi",
                "set_size_mean", "spearman_rho_mean", "mrr_mean", "empty_set_share"]
        present = [c for c in cols if c in agg.columns]
        agg_show = agg[present].sort_values(["regime", "side", "predictor", "label_rule"]).round(3)
        lines.append("```\n" + agg_show.to_string(index=False) + "\n```\n\n")
        lines.append(
            "Reading: `top1_regret_raw_mean` is the average top-1 raw AUGRC "
            "regret on the cross-arch test set (lower = better; a value of 0 "
            "means the predictor always picked the oracle). 95% CIs come from "
            f"the {N_BOOT}-resample bootstrap. `set_regret_raw_mean` is the "
            "set-regret for binary heads (min raw_augrc within predicted set "
            "− oracle). `set_size_mean` is paired so we can read precision-"
            "vs-recall trade-offs. `empty_set_share` flags rows where the "
            "predicted competitive set was empty (binary head outputs all p < "
            "0.5 on the side restriction).\n\n"
        )

    lines.append("## Aggregate — `lopo` (4 folds pooled)\n\n")
    ag_lopo = out_root / "track1" / "lopo" / "metrics" / "aggregate.parquet"
    if ag_lopo.exists():
        agg = pq.read_table(ag_lopo).to_pandas()
        cols = ["predictor", "label_rule", "regime", "side", "n",
                "top1_regret_raw_mean", "set_regret_raw_mean",
                "set_size_mean", "spearman_rho_mean"]
        present = [c for c in cols if c in agg.columns]
        agg_show = agg[present].sort_values(["regime", "side", "predictor", "label_rule"]).round(3)
        lines.append("```\n" + agg_show.to_string(index=False) + "\n```\n\n")

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    print("Track 1:")
    run_track1(out_root)
    print("Track 2:")
    run_track2(out_root)

    report(out_root, out_root / "11_metrics_check.md")
    print(f"wrote {out_root / '11_metrics_check.md'}")


if __name__ == "__main__":
    main()

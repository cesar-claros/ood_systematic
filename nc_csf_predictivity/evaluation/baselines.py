"""Step 14: Baseline comparisons + empty-set imputation for binary heads.

For every test eval row, compute the regret of:
  Always-X baselines for X ∈ {MSR, Energy, MLS, CTM, fDBD, NNGuide}
  Random-CSF baseline (analytical mean = avg regret over all CSFs on side)
  Oracle-on-train baseline (CSF with lowest mean train AUGRC on side)

Side stratification:
  - For per-side analyses, the candidate pool restricts to the side's CSFs.
  - Always-X baselines are dropped from the per-side report if X is not on
    that side (e.g., Always-MSR is head-side; absent from feature/-only).
  - Random-CSF and Oracle-on-train respect the side restriction.

Empty-set imputation:
  Binary head set_regret values are NaN on rows where the predicted competitive
  set ∩ side is empty (step 13). Step 14 also produces an imputed variant
  where empty rows are filled with `worst_augrc − oracle_augrc` (worst-case
  penalty), giving a fair like-for-like with baselines that always predict.

Output (one unified comparator table per (track, split)):
  outputs/<track>/<split>/baselines/per_row.parquet
  outputs/<track>/<split>/baselines/aggregate.parquet
      with comparator_kind ∈ {baseline, predictor_raw, predictor_imputed}
      and comparator_name describing the predictor / baseline identity.
  outputs/12_baselines_check.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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
ALWAYS_BASELINES = ["MSR", "Energy", "MLS", "CTM", "fDBD", "NNGuide"]
SIDES = ["all", "head", "feature"]
N_BOOT = 2000
SEED = 0
TRACK1_SPLITS = ["xarch", "lopo", "lodo_vgg13", "pxs_vgg13",
                 "single_vgg13", "lopo_cnn_only"]
TRACK2_SPLITS = ["track2_loo"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def csfs_on_side(side: str, all_csfs: list[str]) -> set[str]:
    if side == "all":
        return set(all_csfs)
    if side == "head":
        return set(all_csfs) & HEAD_SIDE_CSFS
    if side == "feature":
        return set(all_csfs) & FEATURE_SIDE_CSFS
    raise ValueError(side)


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
    return (float(values.mean()),
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


# ---- Per-row baseline computation ----

def compute_baselines_per_row(eval_long: pd.DataFrame, oracle_df: pd.DataFrame,
                              train_long: pd.DataFrame, id_col: str,
                              all_csfs: list[str]) -> pd.DataFrame:
    """eval_long: id_col, eval_dataset, regime, csf, raw_augrc.
    oracle_df: id_col, eval_dataset, regime, oracle/worst columns per side.
    train_long: same shape as eval_long but for training rows; used to pick
                Oracle-on-train CSF per side.
    Returns long table per (id, eval_dataset, side, comparator_name)."""
    rows = []

    # Pre-compute Oracle-on-train CSF per side
    train_best_csf = {}
    for side in SIDES:
        side_set = csfs_on_side(side, all_csfs)
        sub = train_long[train_long["csf"].isin(side_set)]
        if sub.empty:
            train_best_csf[side] = None
            continue
        means = sub.groupby("csf")["raw_augrc"].mean()
        train_best_csf[side] = means.idxmin()

    # Mean-augrc per (id, eval, side) over the side's CSFs (= analytical Random-CSF mean)
    for side in SIDES:
        side_set = csfs_on_side(side, all_csfs)
        eval_side = eval_long[eval_long["csf"].isin(side_set)]
        if eval_side.empty:
            continue

        # Random-CSF analytical mean per row
        rand_mean = (eval_side.groupby([id_col, "eval_dataset"])["raw_augrc"]
                     .mean().rename("rand_mean_augrc").reset_index())

        # Oracle-on-train CSF AUGRC per row
        train_csf = train_best_csf[side]
        if train_csf is not None:
            train_csf_aug = (eval_side[eval_side["csf"] == train_csf]
                             .groupby([id_col, "eval_dataset"])["raw_augrc"]
                             .min().rename("train_best_augrc").reset_index())
        else:
            train_csf_aug = None

        # Always-X baselines per row
        always_pieces = {}
        for x in ALWAYS_BASELINES:
            if x not in side_set:
                continue
            sub = (eval_side[eval_side["csf"] == x]
                   .groupby([id_col, "eval_dataset"])["raw_augrc"]
                   .min().rename(f"always_{x}_augrc").reset_index())
            always_pieces[x] = sub

        # Join with oracle for this side
        side_oracle = oracle_df[[id_col, "eval_dataset", "regime",
                                 f"oracle_csf{'' if side == 'all' else '_' + side}",
                                 f"oracle_augrc{'' if side == 'all' else '_' + side}",
                                 f"worst_csf{'' if side == 'all' else '_' + side}",
                                 f"worst_augrc{'' if side == 'all' else '_' + side}"]]
        ocol = f"oracle_augrc{'' if side == 'all' else '_' + side}"
        wcol = f"worst_augrc{'' if side == 'all' else '_' + side}"
        side_oracle = side_oracle.rename(columns={ocol: "o_augrc", wcol: "w_augrc"})

        merged = side_oracle.merge(rand_mean, on=[id_col, "eval_dataset"], how="left")
        if train_csf_aug is not None:
            merged = merged.merge(train_csf_aug, on=[id_col, "eval_dataset"], how="left")
        for x, sub in always_pieces.items():
            merged = merged.merge(sub, on=[id_col, "eval_dataset"], how="left")

        denom = (merged["w_augrc"] - merged["o_augrc"]).replace(0, np.nan)

        # Random-CSF
        rows.extend(_emit_baseline(merged, "rand_mean_augrc", "Random-CSF",
                                   side, denom, id_col))
        # Oracle-on-train
        if train_csf is not None:
            rows.extend(_emit_baseline(merged, "train_best_augrc",
                                       f"Oracle-on-train ({train_csf})",
                                       side, denom, id_col))
        # Always-X
        for x in always_pieces:
            rows.extend(_emit_baseline(merged, f"always_{x}_augrc",
                                       f"Always-{x}", side, denom, id_col))

    return pd.DataFrame(rows)


def _emit_baseline(merged: pd.DataFrame, augrc_col: str, name: str,
                   side: str, denom: pd.Series, id_col: str) -> list[dict]:
    out = []
    for _, r in merged.iterrows():
        aug = r.get(augrc_col, np.nan)
        if np.isnan(aug):
            continue
        regret = max(float(aug) - float(r["o_augrc"]), 0.0)
        d = float(r["w_augrc"]) - float(r["o_augrc"])
        norm = regret / d if d > 0 else np.nan
        out.append({
            id_col: r[id_col], "eval_dataset": r["eval_dataset"],
            "regime": r["regime"], "side": side,
            "comparator_kind": "baseline", "comparator_name": name,
            "regret_raw": regret, "regret_norm": norm,
        })
    return out


# ---- Imputed binary metrics from step 13 per_row ----

def compute_imputed_binary(per_row_path: Path, id_col: str) -> pd.DataFrame:
    if not per_row_path.exists():
        return pd.DataFrame()
    pr = pq.read_table(per_row_path).to_pandas()
    bin_mask = pr["predictor"].isin(["multilabel", "per_csf_binary"])
    b = pr[bin_mask].copy()
    if b.empty:
        return pd.DataFrame()
    b["set_regret_raw_imputed"] = b["set_regret_raw"].fillna(b["w_augrc"] - b["o_augrc"])
    denom = (b["w_augrc"] - b["o_augrc"]).replace(0, np.nan)
    b["set_regret_norm_imputed"] = b["set_regret_raw_imputed"] / denom

    out = b[[id_col, "eval_dataset", "regime", "side",
             "predictor", "label_rule",
             "set_regret_raw_imputed", "set_regret_norm_imputed"]].copy()
    out["comparator_kind"] = "predictor_imputed"
    out["comparator_name"] = (out["predictor"].astype(str) + "/"
                              + out["label_rule"].astype(str))
    out = out.rename(columns={
        "set_regret_raw_imputed": "regret_raw",
        "set_regret_norm_imputed": "regret_norm",
    })
    return out[[id_col, "eval_dataset", "regime", "side",
                "comparator_kind", "comparator_name",
                "regret_raw", "regret_norm"]]


def compute_raw_predictor(per_row_path: Path, id_col: str) -> pd.DataFrame:
    """Pull the raw (non-imputed) per-row predictor regrets from step 13."""
    if not per_row_path.exists():
        return pd.DataFrame()
    pr = pq.read_table(per_row_path).to_pandas()

    pieces = []
    # Regression top-1
    reg = pr[pr["predictor"] == "regression"].copy()
    if not reg.empty:
        reg["comparator_kind"] = "predictor_raw"
        reg["comparator_name"] = "regression"
        out = reg[[id_col, "eval_dataset", "regime", "side",
                   "comparator_kind", "comparator_name"]].copy()
        out["regret_raw"] = reg["top1_regret_raw"].values
        out["regret_norm"] = reg["top1_regret_norm"].values
        pieces.append(out)

    # Binary set-regret (raw, NaN on empty)
    binr = pr[pr["predictor"].isin(["multilabel", "per_csf_binary"])].copy()
    if not binr.empty:
        binr["comparator_kind"] = "predictor_raw"
        binr["comparator_name"] = (binr["predictor"].astype(str) + "/"
                                   + binr["label_rule"].astype(str))
        out = binr[[id_col, "eval_dataset", "regime", "side",
                    "comparator_kind", "comparator_name"]].copy()
        out["regret_raw"] = binr["set_regret_raw"].values
        out["regret_norm"] = binr["set_regret_norm"].values
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def aggregate(per_row_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keyset, g in per_row_long.groupby(["comparator_kind", "comparator_name",
                                           "regime", "side"], dropna=False):
        rec = dict(zip(["comparator_kind", "comparator_name", "regime", "side"],
                       keyset))
        rec["n"] = len(g)
        for col in ("regret_raw", "regret_norm"):
            mean, lo, hi = bootstrap_mean_ci(g[col].astype(float).values)
            rec[f"{col}_mean"] = mean
            rec[f"{col}_ci_lo"] = lo
            rec[f"{col}_ci_hi"] = hi
        rows.append(rec)
    return pd.DataFrame(rows)


# ---- Per-track drivers ----

def get_test_ids_for_split(out_root: Path, split: str, id_col: str) -> set[str]:
    sp_path = out_root / "splits" / f"{split}.parquet"
    if not sp_path.exists():
        return set()
    sp = pq.read_table(sp_path).to_pandas()
    return set(sp[sp["role"] == "test"][id_col].tolist())


def get_train_ids_for_split(out_root: Path, split: str, id_col: str) -> set[str]:
    sp_path = out_root / "splits" / f"{split}.parquet"
    if not sp_path.exists():
        return set()
    sp = pq.read_table(sp_path).to_pandas()
    return set(sp[sp["role"] == "train"][id_col].tolist())


def run_track1(out_root: Path) -> None:
    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_id_track1(long_df)
    long_df = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})

    oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_id_track1(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    all_csfs = sorted(long_df["csf"].unique())

    for split in TRACK1_SPLITS:
        test_ids = get_test_ids_for_split(out_root, split, "model_id")
        train_ids = get_train_ids_for_split(out_root, split, "model_id")
        if not test_ids:
            continue
        eval_test = long_df[long_df["model_id"].isin(test_ids)]
        oracle_test = oracle_df[oracle_df["model_id"].isin(test_ids)]
        eval_train = long_df[long_df["model_id"].isin(train_ids)]

        baseline_pr = compute_baselines_per_row(
            eval_test, oracle_test, eval_train, "model_id", all_csfs)

        per_row_path = out_root / "track1" / split / "metrics" / "per_row.parquet"
        raw_pr = compute_raw_predictor(per_row_path, "model_id")
        imp_pr = compute_imputed_binary(per_row_path, "model_id")

        all_pr = pd.concat([baseline_pr, raw_pr, imp_pr], ignore_index=True)
        out_dir = out_root / "track1" / split / "baselines"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(all_pr, out_dir / "per_row.parquet")

        agg = aggregate(all_pr)
        write_parquet(agg, out_dir / "aggregate.parquet")
        print(f"  track1/{split}: {len(all_pr):,} per-row, {len(agg)} agg rows")


def run_track2(out_root: Path) -> None:
    long_df = pq.read_table(out_root / "track2" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_id_track2(long_df)
    long_df = long_df[long_df["regime"] != "test"][[
        "cell_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})

    oracle_df = pq.read_table(out_root / "track2" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_id_track2(oracle_df)
    oracle_df = oracle_df[oracle_df["regime"] != "test"]

    all_csfs = sorted(long_df["csf"].unique())

    for split in TRACK2_SPLITS:
        test_ids = get_test_ids_for_split(out_root, split, "cell_id")
        train_ids = get_train_ids_for_split(out_root, split, "cell_id")
        if not test_ids:
            continue
        eval_test = long_df[long_df["cell_id"].isin(test_ids)]
        oracle_test = oracle_df[oracle_df["cell_id"].isin(test_ids)]
        eval_train = long_df[long_df["cell_id"].isin(train_ids)]

        baseline_pr = compute_baselines_per_row(
            eval_test, oracle_test, eval_train, "cell_id", all_csfs)

        per_row_path = out_root / "track2" / split / "metrics" / "per_row.parquet"
        raw_pr = compute_raw_predictor(per_row_path, "cell_id")
        imp_pr = compute_imputed_binary(per_row_path, "cell_id")

        all_pr = pd.concat([baseline_pr, raw_pr, imp_pr], ignore_index=True)
        out_dir = out_root / "track2" / split / "baselines"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(all_pr, out_dir / "per_row.parquet")

        agg = aggregate(all_pr)
        write_parquet(agg, out_dir / "aggregate.parquet")
        print(f"  track2/{split}: {len(all_pr):,} per-row, {len(agg)} agg rows")


def report(out_root: Path, out_path: Path) -> None:
    lines = ["# Step 14 — Baseline comparisons + empty-set imputation\n\n"]
    lines.append("**Date:** 2026-05-04\n")
    lines.append("**Source:** `code/nc_csf_predictivity/evaluation/baselines.py`\n")
    lines.append(f"**Bootstrap:** n={N_BOOT}, seed={SEED}.\n\n")

    lines.append("## Worked example — per-row baselines on one xarch row\n\n")
    target = "ResNet18|confidnet|cifar10|1|0|2.2"
    pr_path = out_root / "track1" / "xarch" / "baselines" / "per_row.parquet"
    if pr_path.exists():
        pr = pq.read_table(pr_path).to_pandas()
        s = pr[(pr["model_id"] == target) & (pr["eval_dataset"] == "cifar100")]
        if not s.empty:
            lines.append(
                f"Same row as steps 10–13: `{target}`, eval=`cifar100`, "
                "regime=`near`. Showing all comparators (baselines, raw "
                "predictors, and imputed binary predictors):\n\n"
            )
            cols = ["comparator_kind", "comparator_name", "side",
                    "regret_raw", "regret_norm"]
            lines.append("```\n" + s[cols].sort_values(["side","comparator_kind","comparator_name"])
                         .round(3).to_string(index=False) + "\n```\n\n")
            lines.append(
                "Reading: `predictor_raw` = the raw step-13 predictor regret "
                "(NaN for binary heads when set ∩ side was empty). "
                "`predictor_imputed` = empty rows filled with worst-case "
                "regret (= worst − oracle for that side). `baseline` rows "
                "include Always-X, Random-CSF, and Oracle-on-train.\n\n"
            )

    lines.append("## Aggregate — `xarch` cross-arch headline\n\n")
    ag = out_root / "track1" / "xarch" / "baselines" / "aggregate.parquet"
    if ag.exists():
        agg = pq.read_table(ag).to_pandas()
        agg = agg.sort_values(["regime", "side", "comparator_kind",
                               "regret_raw_mean"])
        cols = ["comparator_kind", "comparator_name", "regime", "side", "n",
                "regret_raw_mean", "regret_raw_ci_lo", "regret_raw_ci_hi",
                "regret_norm_mean"]
        lines.append("Sorted within each (regime, side) by `regret_raw_mean` "
                     "ascending (lower = better):\n\n")
        # Show one (regime, side) at a time for readability
        for regime in ["near", "mid", "far"]:
            for side in ["all", "feature", "head"]:
                sub = agg[(agg["regime"] == regime) & (agg["side"] == side)]
                if sub.empty:
                    continue
                lines.append(f"### `regime={regime}, side={side}`\n\n")
                lines.append("```\n" + sub[cols].round(3).to_string(index=False)
                             + "\n```\n\n")

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

    report(out_root, out_root / "12_baselines_check.md")
    print(f"wrote {out_root / '12_baselines_check.md'}")


if __name__ == "__main__":
    main()

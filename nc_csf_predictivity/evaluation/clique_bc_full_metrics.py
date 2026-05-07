"""Full metrics + baselines + statistical tests on the clique (b+c) predictor.

Runs the equivalents of pipeline steps 13, 14, and 15 on the new headline
predictor (per-CSF binary, L2 Cs=50, class_weight='balanced', per-arch
NC standardization, clique label rule) across xarch + lopo and the three
feature configs (source / n_classes / none).

Step 13: per-row regret + ranking metrics, aggregated per (regime, side,
         config) with bootstrap 95% CIs.
Step 14: baseline (always-X / random / oracle-on-train) per-row regret;
         predictor regret with empty-set imputation for fair comparison.
Step 15: paired Wilcoxon (alt='less') of predictor vs each baseline,
         Holm-Bonferroni corrected within (regime, side); per-row Spearman ρ
         (binary preds don't have ranking, so this is omitted).

Outputs:
  outputs/clique_bc/track1/<split>/<config>/
    metrics/per_row.parquet
    metrics/aggregate.parquet
    baselines/per_row.parquet
    baselines/aggregate.parquet
    stats.json
  outputs/24_clique_bc_full_metrics.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats as scstats
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"
RESULT_ROOT = "clique_bc"

NC_PRIMARY = [
    "var_collapse", "equiangular_uc", "equiangular_wc",
    "equinorm_uc", "equinorm_wc", "max_equiangular_uc",
    "max_equiangular_wc", "self_duality",
]
HEAD_SIDE_CSFS = {
    "Confidence", "Energy", "GE", "GEN", "GradNorm",
    "MLS", "MSR", "PCE", "PE", "REN", "pNML",
}
FEATURE_SIDE_CSFS = {
    "PCA RecError global", "NeCo", "NNGuide", "CTM", "ViM", "Maha",
    "fDBD", "KPCA RecError global", "Residual",
}
ALWAYS_BASELINES = ["MSR", "Energy", "MLS", "CTM", "fDBD", "NNGuide"]
SIDES = ["all", "head", "feature"]
SPLITS = ["xarch", "lopo"]
CONFIGS = ["source", "n_classes", "none"]
N_BOOT = 2000
SEED = 0
ALPHA = 0.05


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


def filter_to_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if side == "all":
        return df
    if side == "head":
        return df[df["csf"].isin(HEAD_SIDE_CSFS)]
    return df[df["csf"].isin(FEATURE_SIDE_CSFS)]


def csfs_on_side(side: str, all_csfs: list[str]) -> set[str]:
    if side == "all":
        return set(all_csfs)
    if side == "head":
        return set(all_csfs) & HEAD_SIDE_CSFS
    return set(all_csfs) & FEATURE_SIDE_CSFS


def oracle_for_side(oracle: pd.DataFrame, side: str) -> pd.DataFrame:
    suffix = "" if side == "all" else f"_{side}"
    return oracle.assign(
        o_csf=oracle[f"oracle_csf{suffix}"],
        o_augrc=oracle[f"oracle_augrc{suffix}"],
        w_csf=oracle[f"worst_csf{suffix}"],
        w_augrc=oracle[f"worst_augrc{suffix}"],
    )


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOT
                      ) -> tuple[float, float, float]:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


# ---- Step 13: per-row binary metrics ----

def binary_per_row(preds_bin: pd.DataFrame, eval_long: pd.DataFrame,
                   oracle_side: pd.DataFrame, side: str) -> pd.DataFrame:
    if preds_bin.empty or eval_long.empty:
        return pd.DataFrame()
    test_ids = preds_bin["model_id"].unique()
    eval_long = eval_long[eval_long["model_id"].isin(test_ids)]
    oracle_side = oracle_side[oracle_side["model_id"].isin(test_ids)]
    # Restrict eval_long to side's CSFs so set_min is computed only over
    # CSFs eligible for the side-restricted comparison.
    eval_long = filter_to_side(eval_long, side)
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
    metrics["set_regret_imputed"] = metrics["set_regret_raw"].fillna(metrics["w_augrc"] - metrics["o_augrc"])
    denom = (metrics["w_augrc"] - metrics["o_augrc"]).replace(0, np.nan)
    metrics["set_regret_norm_imputed"] = metrics["set_regret_imputed"] / denom
    metrics["side"] = side
    return metrics


def aggregate(per_row: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    metric_cols = ["set_regret_raw", "set_regret_imputed",
                   "set_regret_norm_imputed", "set_size"]
    for keyset, g in per_row.groupby(group_cols):
        rec = dict(zip(group_cols, keyset)) if isinstance(keyset, tuple) else {group_cols[0]: keyset}
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


# ---- Step 14: baselines ----

def compute_baselines_per_row(eval_long: pd.DataFrame, oracle_side: pd.DataFrame,
                              train_long: pd.DataFrame, side: str,
                              all_csfs: list[str]) -> pd.DataFrame:
    """Per (model, eval) row, compute regret of each baseline (for one side)."""
    side_set = csfs_on_side(side, all_csfs)
    eval_side = eval_long[eval_long["csf"].isin(side_set)]
    if eval_side.empty:
        return pd.DataFrame()

    # Random-CSF analytical mean
    rand_mean = (eval_side.groupby(["model_id", "eval_dataset"])["raw_augrc"]
                 .mean().rename("rand_mean_augrc").reset_index())

    # Oracle-on-train: best CSF on training rows (side-restricted)
    train_side = train_long[train_long["csf"].isin(side_set)]
    train_csf = None
    if not train_side.empty:
        train_csf = train_side.groupby("csf")["raw_augrc"].mean().idxmin()
    train_aug = None
    if train_csf is not None:
        train_aug = (eval_side[eval_side["csf"] == train_csf]
                     .groupby(["model_id", "eval_dataset"])["raw_augrc"]
                     .min().rename("train_best_augrc").reset_index())

    # Always-X
    always_pieces = {}
    for x in ALWAYS_BASELINES:
        if x not in side_set:
            continue
        sub = (eval_side[eval_side["csf"] == x]
               .groupby(["model_id", "eval_dataset"])["raw_augrc"]
               .min().rename(f"always_{x}_augrc").reset_index())
        always_pieces[x] = sub

    base = oracle_side[["model_id", "eval_dataset", "regime",
                        "o_csf", "o_augrc", "w_csf", "w_augrc"]]
    merged = base.merge(rand_mean, on=["model_id", "eval_dataset"], how="left")
    if train_aug is not None:
        merged = merged.merge(train_aug, on=["model_id", "eval_dataset"], how="left")
    for x, sub in always_pieces.items():
        merged = merged.merge(sub, on=["model_id", "eval_dataset"], how="left")

    rows = []
    denom = (merged["w_augrc"] - merged["o_augrc"]).replace(0, np.nan)
    for _, r in merged.iterrows():
        d = float(r["w_augrc"]) - float(r["o_augrc"])
        # Random-CSF
        if not np.isnan(r["rand_mean_augrc"]):
            regret = max(float(r["rand_mean_augrc"]) - float(r["o_augrc"]), 0.0)
            rows.append({"model_id": r["model_id"], "eval_dataset": r["eval_dataset"],
                         "regime": r["regime"], "side": side,
                         "comparator": "Random-CSF",
                         "regret_raw": regret,
                         "regret_norm": regret / d if d > 0 else np.nan})
        # Oracle-on-train
        if train_csf is not None and "train_best_augrc" in r and not np.isnan(r["train_best_augrc"]):
            regret = max(float(r["train_best_augrc"]) - float(r["o_augrc"]), 0.0)
            rows.append({"model_id": r["model_id"], "eval_dataset": r["eval_dataset"],
                         "regime": r["regime"], "side": side,
                         "comparator": f"Oracle-on-train ({train_csf})",
                         "regret_raw": regret,
                         "regret_norm": regret / d if d > 0 else np.nan})
        # Always-X
        for x in always_pieces:
            col = f"always_{x}_augrc"
            if col in r and not np.isnan(r[col]):
                regret = max(float(r[col]) - float(r["o_augrc"]), 0.0)
                rows.append({"model_id": r["model_id"], "eval_dataset": r["eval_dataset"],
                             "regime": r["regime"], "side": side,
                             "comparator": f"Always-{x}",
                             "regret_raw": regret,
                             "regret_norm": regret / d if d > 0 else np.nan})
    return pd.DataFrame(rows)


# ---- Step 15: paired Wilcoxon ----

def wilcoxon_predictor_vs_baselines(predictor_per_row: pd.DataFrame,
                                    baselines_per_row: pd.DataFrame
                                    ) -> list[dict]:
    """Per (regime, side): paired Wilcoxon of predictor's set_regret_imputed
    vs each baseline's regret_raw, joined on (model_id, eval_dataset)."""
    out = []
    for (regime, side), g_pred in predictor_per_row.groupby(["regime", "side"]):
        g_pred = g_pred[["model_id", "eval_dataset", "set_regret_imputed"]].rename(
            columns={"set_regret_imputed": "pred_regret"})
        g_bl = baselines_per_row[(baselines_per_row["regime"] == regime)
                                  & (baselines_per_row["side"] == side)]
        rows = []
        for comparator, gb in g_bl.groupby("comparator"):
            merged = g_pred.merge(gb[["model_id", "eval_dataset", "regret_raw"]],
                                  on=["model_id", "eval_dataset"])
            if len(merged) < 5:
                continue
            diff = merged["pred_regret"].values - merged["regret_raw"].values
            if np.allclose(diff, 0):
                W, p = float("nan"), 1.0
            else:
                try:
                    W, p = scstats.wilcoxon(diff, alternative="less",
                                            zero_method="wilcox")
                except ValueError:
                    W, p = float("nan"), float("nan")
            rows.append({
                "regime": regime, "side": side, "comparator": comparator,
                "n": int(len(merged)),
                "median_diff": float(np.median(diff)),
                "W": float(W) if not np.isnan(W) else None,
                "p": float(p) if not np.isnan(p) else None,
            })
        # Holm-Bonferroni within (regime, side)
        if rows:
            ps = [r["p"] if r["p"] is not None else 1.0 for r in rows]
            reject, p_holm, _, _ = multipletests(ps, method="holm")
            for i, r in enumerate(rows):
                r["p_holm"] = float(p_holm[i])
                r["reject_holm_05"] = bool(reject[i])
        out.extend(rows)
    return out


# ---- Driver ----

def run_one(out_root: Path, split: str, config: str,
            long_df: pd.DataFrame, oracle_df: pd.DataFrame,
            train_long_global: dict[str, pd.DataFrame]) -> dict:
    pp = (out_root / "ablations" / "calib_cliques" / "track1"
          / split / config / "preds.parquet")
    if not pp.exists():
        return {}
    preds = pq.read_table(pp).to_pandas()
    eval_long = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})
    oracle = oracle_df[oracle_df["regime"] != "test"]
    all_csfs = sorted(eval_long["csf"].unique())

    # Per-side per-row metrics
    per_row_pieces = []
    for side in SIDES:
        m = binary_per_row(preds, eval_long, oracle_for_side(oracle, side), side)
        if not m.empty:
            per_row_pieces.append(m)
    per_row = pd.concat(per_row_pieces, ignore_index=True) if per_row_pieces else pd.DataFrame()

    out_dir = out_root / RESULT_ROOT / "track1" / split / config
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines").mkdir(parents=True, exist_ok=True)

    if not per_row.empty:
        write_parquet(per_row, out_dir / "metrics" / "per_row.parquet")
        agg = aggregate(per_row, ["regime", "side"])
        write_parquet(agg, out_dir / "metrics" / "aggregate.parquet")

    # Baselines
    train_long = train_long_global[split]
    bl_pieces = []
    for side in SIDES:
        bl = compute_baselines_per_row(
            eval_long[eval_long["model_id"].isin(preds["model_id"].unique())],
            oracle_for_side(oracle[oracle["model_id"].isin(preds["model_id"].unique())], side),
            train_long, side, all_csfs)
        if not bl.empty:
            bl_pieces.append(bl)
    bl_per_row = pd.concat(bl_pieces, ignore_index=True) if bl_pieces else pd.DataFrame()
    if not bl_per_row.empty:
        write_parquet(bl_per_row, out_dir / "baselines" / "per_row.parquet")
        bl_agg = aggregate(bl_per_row.rename(columns={"regret_raw": "set_regret_raw"}),
                           ["regime", "side", "comparator"])
        write_parquet(bl_agg, out_dir / "baselines" / "aggregate.parquet")

    # Wilcoxon tests
    stats_results = {}
    if not per_row.empty and not bl_per_row.empty:
        wil = wilcoxon_predictor_vs_baselines(per_row, bl_per_row)
        stats_results["wilcoxon"] = wil

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats_results, f, indent=2, default=str)

    return {
        "n_metrics_rows": len(per_row),
        "n_baseline_rows": len(bl_per_row),
        "n_wilcoxon_tests": len(stats_results.get("wilcoxon", [])),
    }


def get_split_train_long(out_root: Path, split: str, long_df: pd.DataFrame) -> pd.DataFrame:
    sp = pq.read_table(out_root / "splits" / f"{split}.parquet").to_pandas()
    train_ids = sp[sp["role"] == "train"]["model_id"].unique()
    el = long_df[long_df["regime"] != "test"][[
        "model_id", "eval_dataset", "regime", "csf", "augrc"
    ]].rename(columns={"augrc": "raw_augrc"})
    return el[el["model_id"].isin(train_ids)]


def report(out_root: Path, summary: dict, out_path: Path) -> None:
    lines = ["# Full metrics suite — clique-rule (b+c) headline predictor\n\n"]
    lines.append("**Date:** 2026-05-05\n")
    lines.append("**Source:** `code/nc_csf_predictivity/evaluation/clique_bc_full_metrics.py`\n")
    lines.append("**Predictor:** L2 LogisticRegressionCV(Cs=50, cv=5, "
                 "class_weight='balanced'), NC pre-standardized per architecture\n")
    lines.append("**Label rule:** clique (per-(paradigm, source, dropout, reward, regime) "
                 "Friedman-Conover top cliques)\n")
    lines.append(f"**Bootstrap:** n={N_BOOT}, seed={SEED}; **Holm-Bonferroni** within (regime, side)\n\n")

    for split in SPLITS:
        lines.append(f"## {split}\n\n")
        for config in CONFIGS:
            agg_path = out_root / RESULT_ROOT / "track1" / split / config / "metrics" / "aggregate.parquet"
            if not agg_path.exists():
                continue
            agg = pq.read_table(agg_path).to_pandas()
            agg = agg.sort_values(["side", "regime"])

            lines.append(f"### config = `{config}`\n\n")
            lines.append("**Predictor regret (imputed) per (regime, side):**\n\n")
            cols = ["regime", "side", "n",
                    "set_regret_imputed_mean", "set_regret_imputed_ci_lo",
                    "set_regret_imputed_ci_hi", "set_size_mean",
                    "empty_set_share"]
            present = [c for c in cols if c in agg.columns]
            lines.append("```\n" + agg[present].round(3).to_string(index=False) + "\n```\n\n")

            # Wilcoxon Holm-corrected wins
            stats_path = out_root / RESULT_ROOT / "track1" / split / config / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    sd = json.load(f)
                wil = sd.get("wilcoxon", [])
                wins = [r for r in wil if r.get("reject_holm_05")]
                wins_df = pd.DataFrame(wins)
                lines.append("**Holm-corrected Wilcoxon wins** (predictor regret < baseline at α=0.05):\n\n")
                if wins_df.empty:
                    lines.append("none.\n\n")
                else:
                    show = wins_df[["regime", "side", "comparator", "n",
                                    "median_diff", "W", "p", "p_holm"]].round(4)
                    lines.append("```\n" + show.to_string(index=False) + "\n```\n\n")

                # Also surface losses (where baseline beats predictor at α=0.05, alt='greater')
                # We tested 'less'; a high p means we did NOT reject. To be honest about losses,
                # report the number of comparators that the predictor did NOT beat.
                no_win = [r for r in wil if not r.get("reject_holm_05")]
                lines.append(f"**Comparators not significantly beaten** (after Holm): "
                             f"{len(no_win)} of {len(wil)} pairs.\n\n")

    lines.append("## Run summary\n\n")
    rows = []
    for (split, config), info in summary.items():
        rec = {"split": split, "config": config, **info}
        rows.append(rec)
    if rows:
        lines.append("```\n" + pd.DataFrame(rows).to_string(index=False) + "\n```\n")

    out_path.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)

    long_df = pq.read_table(out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()
    long_df = add_model_id(long_df)
    oracle_df = pq.read_table(out_root / "track1" / "dataset" / "oracle.parquet").to_pandas()
    oracle_df = add_model_id(oracle_df)

    train_long = {split: get_split_train_long(out_root, split, long_df)
                  for split in SPLITS}

    summary = {}
    for split in SPLITS:
        for config in CONFIGS:
            print(f"  {split}/{config} ...")
            info = run_one(out_root, split, config, long_df, oracle_df, train_long)
            summary[(split, config)] = info

    report(out_root, summary, out_root / "24_clique_bc_full_metrics.md")
    print(f"wrote {out_root / '24_clique_bc_full_metrics.md'}")


if __name__ == "__main__":
    main()

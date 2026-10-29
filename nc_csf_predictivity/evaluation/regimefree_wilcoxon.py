"""Holm-corrected paired Wilcoxon tests for the regime-free predictor arms.

Rebuilds the 54-comparison test behind the paper's Sec 4.4 count (per
(regime, side) cell: one-sided paired Wilcoxon of per-row imputed set-regret,
H1 predictor < baseline, Holm within the cell) and applies it to the arms of
`ablations/calibration_regime_free.py` on the VGG13 -> ResNet18 split.

Gates (abort before any regime-free number is computed):
  G1  the rebuilt test on the stored clique_bc per-row metrics reproduces the
      stored clique_bc stats.json (p and Holm p) for all three paper configs.
  G2  `binary_per_row` on the with-regime ablation predictions reproduces the
      stored clique_bc per-row regret, so the regime-free arms are scored by
      the same path as the paper.

Each comparison also gets the opposite one-sided test (H1 predictor >
baseline, Holm within the cell), so cells where the predictor is worse are
reported as such.

The Wilcoxon signed-rank test ranks the paired differences, so it does not
test the mean regret that the paper reports; with heavy-tailed regret (large
misses on a few rows) it can call a predictor better whose mean is worse.
Every comparison therefore also gets a test of the mean difference: a paired
sign-flip permutation test clustered by model (all rows of one model flip
together), both directions, Holm within the cell. The published configs are
tested the same way for comparison.

Outputs:
  outputs/regimefree_wilcoxon/xarch_tests.csv
  outputs/regimefree_wilcoxon/xarch_aggregate.csv  (per-cell mean, CI, set size)
  outputs/48_regimefree_wilcoxon.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats as scstats
from scipy.special import comb
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ablations"))
from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    SIDES,
    add_model_id,
    binary_per_row,
    bootstrap_mean_ci,
    filter_to_side,
    oracle_for_side,
)

SPLIT = "xarch"
REGIMES = ["near", "mid", "far"]
PAPER_CONFIGS = {"source": 49, "n_classes": 48, "none": 46}
ARMS = ["source_nr_marginal", "none_nr_marginal",
        "source_nr_pooled", "none_nr_pooled"]
CONTROL_ARMS = ["none_nr_marginal", "source_nr_marginal", "paper_source"]
ROW_KEYS = ["model_id", "eval_dataset", "regime", "side"]
ALPHA = 0.05
TOL = 1e-12
N_FLIPS = 100_000
SEED = 0


def one_sided_p(diff: np.ndarray, alternative: str) -> float:
    """Paired Wilcoxon p-value on predictor-minus-baseline differences.

    Args:
        diff: Per-row regret differences (predictor - baseline).
        alternative: 'less' (predictor better) or 'greater' (predictor worse).

    Returns:
        One-sided p-value; 1.0 when every difference is zero, matching
        `stats/tests.py`.
    """
    if np.allclose(diff, 0):
        return 1.0
    return float(scstats.wilcoxon(diff, alternative=alternative,
                                  zero_method="wilcox").pvalue)


def signflip_mean_p(cell: pd.DataFrame, flips: np.ndarray) -> tuple[float, float]:
    """Model-clustered paired sign-flip test on the mean regret difference.

    Under H0 the sign of each model's summed difference is exchangeable, so
    the null distribution of the total difference is sum_k eps_k * S_k with
    random signs eps_k over the per-model sums S_k.

    Args:
        cell: Rows of one comparison with 'model_id' and 'diff'.
        flips: Random +-1 matrix with at least as many columns as models.

    Returns:
        (p predictor better, p predictor worse), each with the +1 correction.
    """
    sums = cell.groupby("model_id")["diff"].sum().to_numpy()
    if np.allclose(sums, 0):
        return 1.0, 1.0
    observed = sums.sum()
    null = flips[:, : len(sums)] @ sums
    n_draws = len(null) + 1
    return (float((np.sum(null <= observed) + 1) / n_draws),
            float((np.sum(null >= observed) + 1) / n_draws))


def wilcoxon_cells(pred_rows: pd.DataFrame, baseline_rows: pd.DataFrame,
                   flips: np.ndarray) -> pd.DataFrame:
    """Predictor-vs-baseline tests per (regime, side), Holm within each cell.

    Args:
        pred_rows: Per-row predictor regret with ROW_KEYS + 'regret'.
        baseline_rows: Per-row baseline regret with ROW_KEYS + 'comparator'
            + 'regret_raw'.
        flips: Random +-1 matrix for the sign-flip mean test.

    Returns:
        One row per (regime, side, comparator) with n, mean regrets, median
        difference, Wilcoxon and sign-flip mean p-values in both directions,
        and their Holm-adjusted p-values and rejections within the
        (regime, side) cell.
    """
    paired = baseline_rows.merge(pred_rows, on=ROW_KEYS, how="inner")
    paired["diff"] = paired["regret"] - paired["regret_raw"]
    groups = paired.groupby(["regime", "side", "comparator"])
    tests = (groups
             .agg(n=("diff", "size"), pred_mean=("regret", "mean"),
                  baseline_mean=("regret_raw", "mean"),
                  median_diff=("diff", "median"),
                  p=("diff", lambda d: one_sided_p(d.to_numpy(), "less")),
                  p_worse=("diff",
                           lambda d: one_sided_p(d.to_numpy(), "greater")))
             .reset_index())
    mean_ps = np.array([signflip_mean_p(cell, flips) for _, cell in groups])
    tests["p_mean"], tests["p_mean_worse"] = mean_ps[:, 0], mean_ps[:, 1]
    for col in ("p", "p_worse", "p_mean", "p_mean_worse"):
        holm = tests.groupby(["regime", "side"])[col].transform(
            lambda ps: multipletests(ps, alpha=ALPHA, method="holm")[1])
        tests[f"{col}_holm"] = holm
        tests[f"reject_{col}"] = holm < ALPHA
    return tests


def load_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into pandas."""
    return pq.read_table(path).to_pandas()


def paper_rows(clique_root: Path) -> dict[str, pd.DataFrame]:
    """Stored per-row regret of the three published (with-regime) configs."""
    return {
        config: load_parquet(clique_root / config / "metrics" / "per_row.parquet")
        .rename(columns={"set_regret_imputed": "regret"})
        for config in PAPER_CONFIGS}


def gate_stored_stats(clique_root: Path, rebuilt_by_config: dict[str, pd.DataFrame]
                      ) -> None:
    """G1: rebuilt test reproduces the stored clique_bc Wilcoxon results.

    Raises:
        RuntimeError: If any p-value, Holm p-value, or rejection count differs.
    """
    for config, expected in PAPER_CONFIGS.items():
        rebuilt = rebuilt_by_config[config]
        stored = pd.DataFrame(orjson.loads(
            (clique_root / config / "stats.json").read_bytes())["wilcoxon"])
        both = stored.merge(rebuilt, on=["regime", "side", "comparator"],
                            suffixes=("_stored", ""))
        max_dev = float(np.max(np.abs(np.r_[
            both["p_stored"] - both["p"],
            both["p_holm_stored"] - both["p_holm"]])))
        n_reject = int(rebuilt["reject_p"].sum())
        if len(both) != len(stored) or max_dev > TOL or n_reject != expected:
            raise RuntimeError(
                f"G1 failed for {config}: matched {len(both)}/{len(stored)}, "
                f"max |dp| {max_dev:.3g}, rejections {n_reject} vs {expected}")
        print(f"G1 {config}: {n_reject}/{len(rebuilt)} reproduced, "
              f"max |dp| {max_dev:.1e}")


def arm_rows(preds: pd.DataFrame, eval_long: pd.DataFrame,
             oracle_df: pd.DataFrame) -> pd.DataFrame:
    """Per-row imputed set-regret on all three sides for one predictor arm."""
    pieces = [binary_per_row(preds, filter_to_side(eval_long, side),
                             oracle_for_side(oracle_df, side)).assign(side=side)
              for side in SIDES]
    rows = pd.concat(pieces, ignore_index=True)
    return rows.rename(columns={"set_regret_raw_imputed": "regret"})


def gate_scoring_path(out_root: Path, clique_root: Path, eval_long: pd.DataFrame,
                      oracle_df: pd.DataFrame) -> None:
    """G2: the ablation scoring path reproduces the stored paper per-row regret.

    Raises:
        RuntimeError: If row sets or regret values differ.
    """
    preds = load_parquet(out_root / "ablations" / "calib_cliques" / "track1"
                         / SPLIT / "source" / "preds.parquet")
    rebuilt = arm_rows(preds, eval_long, oracle_df)[ROW_KEYS + ["regret"]]
    stored = load_parquet(clique_root / "source" / "metrics" / "per_row.parquet")
    both = stored.merge(rebuilt, on=ROW_KEYS, how="outer", indicator=True)
    max_dev = float((both["set_regret_imputed"] - both["regret"]).abs().max())
    if (both["_merge"] != "both").any() or max_dev > 1e-9:
        raise RuntimeError(f"G2 failed: unmatched "
                           f"{int((both['_merge'] != 'both').sum())}, "
                           f"max |d regret| {max_dev:.3g}")
    print(f"G2: {len(both)} rows reproduced, max |d regret| {max_dev:.1e}")


def random_set_regret(rows: pd.DataFrame, eval_long: pd.DataFrame) -> pd.Series:
    """Expected regret of a uniformly random shortlist of the same size, per row.

    For sorted side AUGRCs a_1 <= ... <= a_M and set size k >= 1,
    P(min of a random k-subset = a_j) = C(M - j, k - 1) / C(M, k). Empty
    shortlists (k = 0) keep the predictor's imputed regret, so the control
    differs from the predictor only in which CSFs it picks.

    Args:
        rows: Per-row predictor metrics from `arm_rows` (needs side,
            set_size, o_augrc, regret).
        eval_long: Long AUGRC table (model_id, eval_dataset, regime, csf,
            raw_augrc).

    Returns:
        Expected random-set regret aligned with `rows.index`.
    """
    out = rows["regret"].astype(float).copy()
    for side, side_rows in rows.groupby("side"):
        sorted_augrc = (filter_to_side(eval_long, side)
                        .groupby(["model_id", "eval_dataset"])["raw_augrc"]
                        .apply(lambda s: np.sort(s.to_numpy())))
        for idx, row in side_rows[side_rows["set_size"] > 0].iterrows():
            augrc = sorted_augrc[(row["model_id"], row["eval_dataset"])]
            if abs(augrc[0] - row["o_augrc"]) > 1e-9:
                raise RuntimeError(f"oracle mismatch on {side} row {idx}")
            weights = comb(len(augrc) - np.arange(1, len(augrc) + 1),
                           row["set_size"] - 1) / comb(len(augrc), row["set_size"])
            out[idx] = float(weights @ augrc - row["o_augrc"])
    return out


def random_set_control(rows: pd.DataFrame, eval_long: pd.DataFrame,
                       flips: np.ndarray) -> pd.DataFrame:
    """Predictor vs the size-matched random-set control, per (regime, side).

    Returns:
        One row per cell with both means, the Wilcoxon p (predictor better)
        and the model-clustered sign-flip mean p, each Holm-adjusted over the
        nine cells.
    """
    paired = rows.assign(random=random_set_regret(rows, eval_long))
    paired["diff"] = paired["regret"] - paired["random"]
    groups = paired.groupby(["regime", "side"])
    control = groups.agg(pred_mean=("regret", "mean"),
                         random_mean=("random", "mean"),
                         set_size=("set_size", "mean")).reset_index()
    control["p_wilcoxon"] = [one_sided_p(cell["diff"].to_numpy(), "less")
                             for _, cell in groups]
    control["p_mean"] = [signflip_mean_p(cell, flips)[0] for _, cell in groups]
    for col in ("p_wilcoxon", "p_mean"):
        control[f"{col}_holm"] = multipletests(control[col], method="holm")[1]
    return control


def aggregate_cells(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (regime, side): mean imputed regret with the paper's bootstrap CI.

    Column names follow the clique_bc metrics/aggregate.parquet layout, so the
    regret figure reads both sources the same way.
    """
    def one_cell(cell: pd.DataFrame) -> pd.Series:
        mean, lo, hi = bootstrap_mean_ci(cell["regret"].to_numpy(dtype=float))
        return pd.Series({"n": len(cell), "set_regret_imputed_mean": mean,
                          "set_regret_imputed_ci_lo": lo,
                          "set_regret_imputed_ci_hi": hi,
                          "set_size_mean": cell["set_size"].mean(),
                          "empty_set_share": cell["empty_set"].mean()})
    return (rows.groupby(["regime", "side"])[["regret", "set_size", "empty_set"]]
            .apply(one_cell).reset_index())


def cell_summary(tests: pd.DataFrame) -> pd.DataFrame:
    """Per (regime, side): predictor mean, strongest baseline, test counts."""
    best = tests.loc[tests.groupby(["regime", "side"])["baseline_mean"].idxmin()]
    counts = (tests.groupby(["regime", "side"])
              .agg(pred_mean=("pred_mean", "first"),
                   wil_better=("reject_p", "sum"),
                   wil_worse=("reject_p_worse", "sum"),
                   mean_better=("reject_p_mean", "sum"),
                   mean_worse=("reject_p_mean_worse", "sum"),
                   tests=("comparator", "size"))
              .reset_index())
    summary = counts.merge(
        best[["regime", "side", "comparator", "baseline_mean"]]
        .rename(columns={"comparator": "best_baseline"}), on=["regime", "side"])
    summary["reduction_%"] = 100 * (1 - summary["pred_mean"]
                                    / summary["baseline_mean"])
    order = {r: i for i, r in enumerate(REGIMES)}
    side_order = {s: i for i, s in enumerate(SIDES)}
    return (summary.sort_values(["side", "regime"],
                                key=lambda c: c.map(side_order | order))
            .round({"pred_mean": 2, "baseline_mean": 2, "reduction_%": 0}))


def report(all_tests: pd.DataFrame, controls: pd.DataFrame, out_path: Path
           ) -> None:
    """Write the markdown summary of every arm and the random-set control."""
    lines = ["# Regime-free predictor: Holm-corrected paired Wilcoxon "
             "(VGG13 -> ResNet18)\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/regimefree_wilcoxon.py`\n",
             "**Protocol:** per (regime, side) cell, each predictor is compared "
             "with each NC-free baseline on per-row imputed set-regret, Holm "
             f"within the cell (alpha {ALPHA}). `wil_*` = the paper's Sec 4.4 "
             "test, rebuilt (one-sided paired Wilcoxon signed-rank). `mean_*` = "
             "paired sign-flip permutation test on the mean difference, "
             f"clustered by model ({N_FLIPS:,} draws, seed {SEED}). `better` / "
             "`worse` = significant in that direction. Gates G1 (stored paper "
             "stats reproduced: 49/48/46 of 54) and G2 (ablation scoring path "
             "reproduces the paper per-row regret) passed before these numbers "
             "were computed. Arms `paper_*` are the published with-regime "
             "configs; the others are regime-free.\n\n",
             "Why two tests: Wilcoxon ranks the paired differences, so it "
             "tests whether the predictor is usually better, not whether its "
             "mean regret is lower. With a few large misses (empty shortlists "
             "imputed at the worst AUGRC, or rare large regrets) the two can "
             "disagree; the paper reports means.\n\n"]
    for arm, tests in all_tests.groupby("arm", sort=False):
        counts = {col: int(tests[col].sum()) for col in
                  ("reject_p", "reject_p_worse", "reject_p_mean",
                   "reject_p_mean_worse")}
        lines.append(
            f"## {arm}\n\nWilcoxon: better {counts['reject_p']}, worse "
            f"{counts['reject_p_worse']} of {len(tests)}. Mean test: better "
            f"{counts['reject_p_mean']}, worse {counts['reject_p_mean_worse']} "
            f"of {len(tests)}.\n\n")
        lines.append("```\n" + cell_summary(tests).to_string(index=False)
                     + "\n```\n\n")
        split = tests[tests["reject_p"] != tests["reject_p_mean"]][
            ["regime", "side", "comparator", "pred_mean", "baseline_mean",
             "median_diff", "p_holm", "p_mean_holm", "p_mean_worse_holm"]]
        if not split.empty:
            lines.append("Comparisons where the two tests disagree:\n\n```\n"
                         + split.round(4).to_string(index=False) + "\n```\n\n")
    lines.append(
        "## Size-matched random-set control\n\n"
        "Per row, the exact expected regret of a uniformly random shortlist "
        "with the predictor's own set size, drawn from the same side "
        "(empty rows keep the imputed regret). One-sided tests, predictor "
        "better; Holm over the nine cells of each arm. Separate from the "
        "54-comparison family above.\n\n```\n"
        + controls.round(4).to_string(index=False) + "\n```\n")
    out_path.write_text("".join(lines))


def main() -> None:
    """Run the gates, test every regime-free arm, and write the outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    out_root = Path(parser.parse_args().out_root)
    clique_root = out_root / "clique_bc" / "track1" / SPLIT

    baseline_rows = load_parquet(
        clique_root / "source" / "baselines" / "per_row.parquet")
    n_models = baseline_rows["model_id"].nunique()
    flips = np.random.default_rng(SEED).choice(
        np.array([-1.0, 1.0]), size=(N_FLIPS, n_models))
    rows_by_arm = {f"paper_{config}": rows
                   for config, rows in paper_rows(clique_root).items()}
    tests_by_arm = {arm: wilcoxon_cells(rows, baseline_rows, flips)
                    for arm, rows in rows_by_arm.items()}
    gate_stored_stats(clique_root, {config: tests_by_arm[f"paper_{config}"]
                                    for config in PAPER_CONFIGS})

    long_df = add_model_id(load_parquet(
        out_root / "track1" / "dataset" / "long_harmonized.parquet"))
    eval_long = (long_df[long_df["regime"] != "test"]
                 [["model_id", "eval_dataset", "regime", "csf", "augrc"]]
                 .rename(columns={"augrc": "raw_augrc"}))
    oracle_df = add_model_id(load_parquet(
        out_root / "track1" / "dataset" / "oracle.parquet"))
    oracle_df = oracle_df[oracle_df["regime"] != "test"]
    gate_scoring_path(out_root, clique_root, eval_long, oracle_df)

    arm_root = out_root / "ablations" / "calib_cliques_regime_free" / "track1" / SPLIT
    for arm in ARMS:
        rows_by_arm[arm] = arm_rows(load_parquet(arm_root / arm / "preds.parquet"),
                                    eval_long, oracle_df)
        tests_by_arm[arm] = wilcoxon_cells(rows_by_arm[arm], baseline_rows, flips)
    all_tests = pd.concat([tests.assign(arm=arm)
                           for arm, tests in tests_by_arm.items()],
                          ignore_index=True)
    aggregates = pd.concat([aggregate_cells(rows).assign(arm=arm)
                            for arm, rows in rows_by_arm.items()],
                           ignore_index=True)

    csv_dir = out_root / "regimefree_wilcoxon"
    csv_dir.mkdir(parents=True, exist_ok=True)
    controls = pd.concat(
        [random_set_control(rows_by_arm[arm], eval_long, flips).assign(arm=arm)
         for arm in CONTROL_ARMS], ignore_index=True)
    all_tests.to_csv(csv_dir / f"{SPLIT}_tests.csv", index=False)
    aggregates.to_csv(csv_dir / f"{SPLIT}_aggregate.csv", index=False)
    controls.to_csv(csv_dir / f"{SPLIT}_random_set_control.csv", index=False)
    report(all_tests, controls, out_root / "48_regimefree_wilcoxon.md")
    print(f"wrote {out_root / '48_regimefree_wilcoxon.md'}")


if __name__ == "__main__":
    main()

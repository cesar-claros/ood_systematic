"""Step 4: Harmonize AUGRC (and AURC) within (source, eval_dataset) cells.

Two harmonized variants per metric:
  • z-score:  z = (x − μ_cell) / σ_cell   [primary, per protocol §8]
  • rank:    pct = rank_within_cell / n_cell  [fallback variant]

Cells are keyed by (source, eval_dataset). Pooling is across all
(architecture, paradigm, run, dropout, reward, csf) entries within a cell, so
the target is on a single scale across architectures. This induces mild
train/test leakage in cross-arch evaluation (μ, σ are estimated on rows that
include the held-out architecture), which we accept because the target is
what the predictor learns to output, not a feature it sees, and downstream
regret metrics in step 13 are computed on raw AUGRC.

Per-cell Shapiro-Wilk on raw AUGRC checks the protocol §8 fallback rule: if
>50% of cells fail at α=0.05, the primary scheme switches from z to rank.

Outputs:
  outputs/<track>/dataset/long_harmonized.parquet
  outputs/02_harmonize_check.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

CELL_KEYS = ["source", "eval_dataset"]
SHAPIRO_MAX_N = 5000  # scipy's recommended cap; subsample above
SHAPIRO_ALPHA = 0.05
FALLBACK_THRESHOLD = 0.50  # >this share of failing cells → fall back to rank


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def harmonize_column(df: pd.DataFrame, value_col: str) -> tuple[pd.Series, pd.Series]:
    """Per-row z-score and percentile-rank within (source, eval_dataset)."""
    grp = df.groupby(CELL_KEYS)[value_col]
    mu = grp.transform("mean")
    sigma = grp.transform("std")
    z = (df[value_col] - mu) / sigma
    pct_rank = grp.rank(method="average", pct=True)
    return z, pct_rank


def shapiro_per_cell(df: pd.DataFrame, value_col: str = "augrc") -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for (src, ev), g in df.groupby(CELL_KEYS):
        x = g[value_col].dropna().values
        n = len(x)
        if n < 3:
            stat, p = np.nan, np.nan
        else:
            x_use = x if n <= SHAPIRO_MAX_N else rng.choice(x, SHAPIRO_MAX_N, replace=False)
            stat, p = stats.shapiro(x_use)
        rows.append({
            "source": src,
            "eval_dataset": ev,
            "n": n,
            "augrc_mean": float(np.mean(x)) if n else np.nan,
            "augrc_std": float(np.std(x, ddof=1)) if n > 1 else np.nan,
            "shapiro_W": stat,
            "shapiro_p": p,
            "passes_alpha_05": (p > SHAPIRO_ALPHA) if not np.isnan(p) else None,
        })
    return pd.DataFrame(rows).sort_values(["source", "eval_dataset"])


def worked_examples_section() -> str:
    """Self-contained pedagogical examples that mirror the actual computations."""
    lines = ["## Worked examples\n\n"]

    # Example 1: z-score
    augrc = np.array([100.0, 150.0, 200.0, 250.0])
    mu = augrc.mean()
    sigma = augrc.std(ddof=1)
    z = (augrc - mu) / sigma
    lines.append("### Example 1 — z-score within (source, eval_dataset)\n\n")
    lines.append(
        "Suppose `(source=cifar10, eval_dataset=svhn)` contains four AUGRC "
        "entries across CSFs and runs (a small subset for illustration):\n\n"
    )
    lines.append("| csf    | augrc |\n|---|---|\n")
    for c, a in zip(["MSR", "Energy", "NeCo", "CTM"], augrc):
        lines.append(f"| {c} | {a:g} |\n")
    lines.append(f"\nCell mean μ = {mu:g}; sample std σ ≈ {sigma:.2f}.\n\n")
    lines.append("Apply z = (augrc − μ) / σ:\n\n")
    lines.append("| csf    | augrc | z |\n|---|---|---|\n")
    for c, a, zi in zip(["MSR", "Energy", "NeCo", "CTM"], augrc, z):
        lines.append(f"| {c} | {a:g} | {zi:+.2f} |\n")
    lines.append("\nLower z = better detection (since lower AUGRC = better).\n\n")

    # Example 2: rank
    lines.append("### Example 2 — percentile rank within cell\n\n")
    lines.append("Same 4 entries, ranked from lowest (best):\n\n")
    lines.append("| csf | augrc | rank | pct rank = rank / n |\n|---|---|---|---|\n")
    pct = np.arange(1, 5) / 4.0
    for c, a, r, p in zip(["MSR", "Energy", "NeCo", "CTM"], augrc, [1, 2, 3, 4], pct):
        lines.append(f"| {c} | {a:g} | {r} | {p:.2f} |\n")
    lines.append(
        "\nLower pct = better. Rank harmonization is invariant to monotone "
        "transformations of AUGRC, so it is the recommended fallback when "
        "the per-cell distribution is far from normal.\n\n"
    )

    # Example 3: Shapiro-Wilk
    lines.append("### Example 3 — Shapiro-Wilk normality test on a cell\n\n")
    rng = np.random.default_rng(7)
    normal_sample = rng.normal(loc=150, scale=30, size=200)
    skew_sample = np.concatenate([rng.normal(loc=150, scale=10, size=190),
                                  rng.normal(loc=400, scale=5, size=10)])
    Wn, pn = stats.shapiro(normal_sample)
    Ws, ps = stats.shapiro(skew_sample)
    lines.append(
        "H₀: the cell's AUGRC values are drawn from a normal distribution. "
        "W ∈ [0,1], W close to 1 = consistent with normal. We reject at "
        f"p ≤ {SHAPIRO_ALPHA}.\n\n"
    )
    lines.append("| sample | n | W | p | verdict at α=0.05 |\n|---|---|---|---|---|\n")
    lines.append(f"| Normal-ish (mean 150, sd 30) | 200 | {Wn:.3f} | {pn:.3g} | "
                 f"{'PASS' if pn > SHAPIRO_ALPHA else 'FAIL'} |\n")
    lines.append(f"| Heavy-tailed mixture | 200 | {Ws:.3f} | {ps:.3g} | "
                 f"{'PASS' if ps > SHAPIRO_ALPHA else 'FAIL'} |\n\n")
    lines.append(
        f"Protocol fallback rule: if more than {FALLBACK_THRESHOLD:.0%} of "
        "cells FAIL at α=0.05, switch the primary harmonization from z-score "
        "to percentile rank for downstream training. Note: at large N "
        "(thousands), Shapiro-Wilk is over-sensitive and tends to reject "
        "normality even for visually normal data — interpret pass rates with "
        "that caveat in mind.\n\n"
    )

    # Example 4: regime-level aggregation
    lines.append("### Example 4 — regime-level aggregation across eval_datasets\n\n")
    lines.append(
        "After harmonization, the per-row z lives at the "
        "(architecture, paradigm, source, run, dropout, reward, csf, eval_dataset) "
        "granularity. To produce a single regime-level number for a (model, csf) "
        "we average z across the eval_datasets in the regime.\n\n"
    )
    lines.append(
        "For `(source=cifar10, regime=mid)` the eval_datasets per the CLIP "
        "groupings are `{isun, lsun resize, lsun cropped, svhn}`. Suppose "
        "one model row uses Energy as its CSF and we observe:\n\n"
    )
    eval_z = {"isun": -1.4, "lsun resize": -0.9, "lsun cropped": -0.6, "svhn": 0.5}
    lines.append("| eval_dataset | augrc_z |\n|---|---|\n")
    for k, v in eval_z.items():
        lines.append(f"| {k} | {v:+.2f} |\n")
    mean_z = float(np.mean(list(eval_z.values())))
    lines.append(
        f"\nRegime-level z = unweighted mean = {mean_z:+.2f}. "
        "Equal weighting per eval_dataset prevents high-variance evals from "
        "dominating. This regime-level number feeds into per-regime metric "
        "and statistical-test aggregations downstream (steps 13–15).\n\n"
    )
    return "".join(lines)


def per_cell_table(shapiro_df: pd.DataFrame) -> str:
    """Markdown table of per-cell stats."""
    lines = ["| source | eval_dataset | n | mean(augrc) | std(augrc) | W | p | pass |\n"
             "|---|---|---|---|---|---|---|---|\n"]
    for _, r in shapiro_df.iterrows():
        passes = "✓" if r["passes_alpha_05"] else "✗"
        if r["passes_alpha_05"] is None:
            passes = "—"
        lines.append(
            f"| {r['source']} | {r['eval_dataset']} | {int(r['n'])} | "
            f"{r['augrc_mean']:.2f} | {r['augrc_std']:.2f} | "
            f"{r['shapiro_W']:.3f} | {r['shapiro_p']:.2e} | {passes} |\n"
        )
    return "".join(lines)


def harmonize_track(in_path: Path, out_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pq.read_table(in_path).to_pandas()
    augrc_z, augrc_rank = harmonize_column(df, "augrc")
    df["augrc_z"] = augrc_z
    df["augrc_rank"] = augrc_rank
    if "aurc" in df.columns:
        aurc_z, aurc_rank = harmonize_column(df, "aurc")
        df["aurc_z"] = aurc_z
        df["aurc_rank"] = aurc_rank
    write_parquet(df, out_path)
    shap = shapiro_per_cell(df, "augrc")
    return df, shap


def report(t1_df: pd.DataFrame | None, t1_sh: pd.DataFrame | None,
           t2_df: pd.DataFrame | None, t2_sh: pd.DataFrame | None,
           out_path: Path) -> None:
    lines = ["# Step 4 — Harmonization check\n\n"]
    lines.append("**Date:** 2026-05-02\n")
    lines.append("**Source:** `code/nc_csf_predictivity/data/harmonize.py`\n\n")

    lines.append(worked_examples_section())

    if t1_df is not None and t1_sh is not None:
        lines.append("## Track 1\n\n")
        lines.append(f"- Output: `outputs/track1/dataset/long_harmonized.parquet`\n")
        lines.append(f"- Total rows: {len(t1_df):,}\n")
        lines.append(f"- Columns added: `augrc_z`, `augrc_rank`, `aurc_z`, `aurc_rank`\n\n")

        n_cells = len(t1_sh)
        n_pass = int(t1_sh["passes_alpha_05"].fillna(False).sum())
        share_fail = (n_cells - n_pass) / n_cells if n_cells else float("nan")
        lines.append(f"### Shapiro-Wilk pass rate\n\n")
        lines.append(f"- Cells: {n_cells}\n")
        lines.append(f"- Passing α={SHAPIRO_ALPHA}: {n_pass} ({n_pass / n_cells:.0%})\n")
        lines.append(f"- Failing share: {share_fail:.0%} (fallback triggered if >{FALLBACK_THRESHOLD:.0%})\n")
        verdict = "FALL BACK to rank" if share_fail > FALLBACK_THRESHOLD else "Keep z as primary"
        lines.append(f"- **Verdict:** {verdict}\n\n")
        lines.append("### Per-cell stats\n\n")
        lines.append(per_cell_table(t1_sh))
        lines.append("\n### Sanity: z-score range and rank range per cell (must be ~symmetric and [0,1])\n\n")
        zsum = t1_df.groupby(CELL_KEYS)["augrc_z"].agg(["mean", "std", "min", "max"]).round(3)
        rsum = t1_df.groupby(CELL_KEYS)["augrc_rank"].agg(["min", "max"]).round(3)
        lines.append("z-score per cell — first 5 cells:\n\n```\n")
        lines.append(zsum.head(5).to_string() + "\n")
        lines.append("```\n\nrank per cell — first 5 cells (should be ~0 and ~1):\n\n```\n")
        lines.append(rsum.head(5).to_string() + "\n```\n\n")

    if t2_df is not None and t2_sh is not None:
        lines.append("## Track 2\n\n")
        lines.append(f"- Output: `outputs/track2/dataset/long_harmonized.parquet`\n")
        lines.append(f"- Total rows: {len(t2_df):,}\n\n")
        n_cells2 = len(t2_sh)
        n_pass2 = int(t2_sh["passes_alpha_05"].fillna(False).sum())
        share_fail2 = (n_cells2 - n_pass2) / n_cells2 if n_cells2 else float("nan")
        lines.append("### Shapiro-Wilk pass rate\n\n")
        lines.append(f"- Cells: {n_cells2}\n")
        lines.append(f"- Passing α={SHAPIRO_ALPHA}: {n_pass2} ({n_pass2 / n_cells2:.0%})\n")
        lines.append(f"- Failing share: {share_fail2:.0%}\n")
        verdict2 = "FALL BACK to rank" if share_fail2 > FALLBACK_THRESHOLD else "Keep z as primary"
        lines.append(f"- **Verdict:** {verdict2}\n\n")
        lines.append("### Per-cell stats\n\n")
        lines.append(per_cell_table(t2_sh))
        lines.append("\n")

    lines.append(
        "## Notes on harmonization scope\n\n"
        "Harmonization pools across all (architecture, paradigm, run, dropout, "
        "reward, csf) entries inside a `(source, eval_dataset)` cell. This means "
        "the harmonized target for ResNet18 rows is computed using μ, σ that also "
        "include ResNet18 rows themselves — a mild train/test leak when the cross-"
        "arch held-out evaluation runs in step 13. We accept this because (i) the "
        "harmonized values are the *target* the predictor outputs, not features it "
        "sees, and (ii) the headline regret metrics in §10 use raw AUGRC, not the "
        "harmonized value. If the leakage materially affects results, step 10's "
        "regression head can re-fit a per-fold harmonizer on training rows only.\n"
    )
    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--track", choices=["1", "2", "all"], default="all")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    t1_df = t1_sh = t2_df = t2_sh = None

    if args.track in ("1", "all"):
        in1 = out_root / "track1" / "dataset" / "long.parquet"
        out1 = out_root / "track1" / "dataset" / "long_harmonized.parquet"
        t1_df, t1_sh = harmonize_track(in1, out1)
        print(f"wrote {out1} ({len(t1_df):,} rows)")

    if args.track in ("2", "all"):
        in2 = out_root / "track2" / "dataset" / "long.parquet"
        out2 = out_root / "track2" / "dataset" / "long_harmonized.parquet"
        t2_df, t2_sh = harmonize_track(in2, out2)
        print(f"wrote {out2} ({len(t2_df):,} rows)")

    report(t1_df, t1_sh, t2_df, t2_sh, out_root / "02_harmonize_check.md")
    print(f"wrote {out_root / '02_harmonize_check.md'}")


if __name__ == "__main__":
    main()

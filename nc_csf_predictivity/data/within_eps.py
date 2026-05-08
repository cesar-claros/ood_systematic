"""Step 6: Within-ε label table for Track 1 (VGG13 only).

Three label variants are computed side-by-side per
(paradigm, source, dropout, reward, regime, csf) on VGG13:

VARIANT C — `_raw` (raw AUGRC mean, original specification).
  Per (csf, run): mean raw AUGRC across the regime's eval_datasets.
  Per csf: mean across 5 runs → `summary_augrc_raw`.
  Bootstrap best csf's 5 per-run summaries (n_boot=2000, seed=0) →
  `eps_raw` = half-width of 95% CI.
  Label `in_within_eps_set_raw` iff `summary_augrc_raw ≤ best + eps_raw`.
  Caveat: raw AUGRC is incommensurate across eval_datasets, so the regime
  mean is biased toward high-AUGRC evals (e.g., svhn dominates cifar10/mid).

VARIANT A — `_rank` (mean of `augrc_rank` from step 4).
  Same recipe but using `augrc_rank` (within-(source, eval_dataset) percentile
  rank from step 4) instead of raw AUGRC. Each eval_dataset contributes
  equally because rank ∈ [0, 1] regardless of native scale. ε is in rank units.

VARIANT B — `_perEval` (per-eval within-ε, then aggregate).
  For each (paradigm, source, dropout, reward, eval_dataset, csf):
    - per-run AUGRC × 5 runs.
    - Best csf per eval = argmin per-eval summary; ε per eval from bootstrap.
    - csf marked competitive at this eval iff per-eval summary ≤ best + ε.
  Then aggregate to regime level by counting how many of the regime's
  eval_datasets mark this csf competitive. Two thresholds reported:
    - `in_within_eps_set_majority` ⇔ count ≥ ceil(N_evals / 2)
    - `in_within_eps_set_unanimous` ⇔ count == N_evals

Output:
  outputs/track1/labels/within_eps.parquet  (one row per cell × csf, all three
                                              variants on the same row)
  outputs/track1/labels/within_eps_per_eval.parquet  (per-(cell, eval, csf)
                                              base table for variant B; useful
                                              for diagnostics)
  outputs/04_within_eps_check.md   (worked examples for all three variants)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

N_BOOT = 2000
SEED = 0
REGIMES = ["near", "mid", "far", "all", "test"]
TRAIN_PARADIGMS = ["confidnet", "devries", "dg"]


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def restrict_to_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "all":
        return df[df["regime"].isin(["near", "mid", "far"])]
    return df[df["regime"] == regime]


def bootstrap_eps(values: np.ndarray, n_boot: int = N_BOOT,
                  seed: int = SEED) -> tuple[float, float, float]:
    """Bootstrap the mean of `values`; return (eps, ci_low, ci_hi).
    eps = half-width of percentile 95% CI on the bootstrap mean."""
    if len(values) < 2:
        v = float(values[0]) if len(values) else 0.0
        return 0.0, v, v
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    ci_low, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return float(ci_hi - ci_low) / 2.0, float(ci_low), float(ci_hi)


def cell_within_eps_one_metric(cell_long: pd.DataFrame, value_col: str,
                               suffix: str) -> pd.DataFrame:
    """Compute within-ε labels with the per-(csf, run) mean across the cell's
    eval_datasets, using `value_col` as the underlying metric.
    Returns DataFrame keyed by csf with columns suffixed by `suffix`."""
    per_run = (cell_long.groupby(["csf", "run"])[value_col].mean()
               .reset_index(name=f"run_mean{suffix}"))
    per_csf = (per_run.groupby("csf")[f"run_mean{suffix}"].mean()
               .reset_index(name=f"summary{suffix}"))
    best_row = per_csf.loc[per_csf[f"summary{suffix}"].idxmin()]
    best_csf = best_row["csf"]
    best_summary = float(best_row[f"summary{suffix}"])
    best_per_run = per_run[per_run["csf"] == best_csf][f"run_mean{suffix}"].values
    eps, ci_low, ci_hi = bootstrap_eps(best_per_run)
    per_csf[f"best_csf{suffix}"] = best_csf
    per_csf[f"best{suffix}"] = best_summary
    per_csf[f"gap{suffix}"] = per_csf[f"summary{suffix}"] - best_summary
    per_csf[f"eps{suffix}"] = eps
    per_csf[f"ci_low{suffix}"] = ci_low
    per_csf[f"ci_hi{suffix}"] = ci_hi
    per_csf[f"in_within_eps_set{suffix}"] = (
        per_csf[f"summary{suffix}"] <= best_summary + eps
    )
    return per_csf


def per_eval_within_eps(cell_long: pd.DataFrame) -> pd.DataFrame:
    """Variant B base table: per (csf, eval_dataset) within-ε using raw AUGRC.
    Returns DataFrame with columns: csf, eval_dataset, summary_augrc_eval,
    eps_eval, best_csf_eval, in_set_eval."""
    pieces = []
    for eval_dataset, eg in cell_long.groupby("eval_dataset"):
        per_run = eg.groupby(["csf", "run"])["augrc"].mean().reset_index(name="run_augrc")
        per_csf = per_run.groupby("csf")["run_augrc"].mean().reset_index(name="summary_augrc_eval")
        best_row = per_csf.loc[per_csf["summary_augrc_eval"].idxmin()]
        best_csf = best_row["csf"]
        best_val = float(best_row["summary_augrc_eval"])
        best_per_run = per_run[per_run["csf"] == best_csf]["run_augrc"].values
        eps, _, _ = bootstrap_eps(best_per_run)
        per_csf["eval_dataset"] = eval_dataset
        per_csf["best_csf_eval"] = best_csf
        per_csf["eps_eval"] = eps
        per_csf["in_set_eval"] = per_csf["summary_augrc_eval"] <= best_val + eps
        pieces.append(per_csf)
    return pd.concat(pieces, ignore_index=True)


def aggregate_perEval_to_regime(per_eval: pd.DataFrame) -> pd.DataFrame:
    """Variant B aggregation: count, majority, unanimous."""
    n_evals = per_eval["eval_dataset"].nunique()
    counts = (per_eval[per_eval["in_set_eval"]]
              .groupby("csf").size().rename("set_count_per_eval")
              .reset_index())
    all_csfs = per_eval[["csf"]].drop_duplicates()
    out = all_csfs.merge(counts, on="csf", how="left").fillna({"set_count_per_eval": 0})
    out["set_count_per_eval"] = out["set_count_per_eval"].astype(int)
    out["n_evals_in_regime"] = n_evals
    out["in_within_eps_set_majority"] = out["set_count_per_eval"] >= math.ceil(n_evals / 2)
    out["in_within_eps_set_unanimous"] = out["set_count_per_eval"] == n_evals
    return out


def compute_all_variants(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (cell-level merged table, per-eval B base table)."""
    cell_keys = ["paradigm", "source", "dropout", "reward"]
    main_pieces = []
    per_eval_pieces = []
    for (paradigm, source, dropout, reward), cell in long_df.groupby(cell_keys):
        for regime in REGIMES:
            sub = restrict_to_regime(cell, regime)
            if sub.empty:
                continue

            raw = cell_within_eps_one_metric(sub, "augrc", "_raw")
            rk = cell_within_eps_one_metric(sub, "augrc_rank", "_rank")
            pe_base = per_eval_within_eps(sub)
            pe_agg = aggregate_perEval_to_regime(pe_base)

            merged = (raw
                      .merge(rk, on="csf", how="outer")
                      .merge(pe_agg, on="csf", how="outer"))
            merged["paradigm"] = paradigm
            merged["source"] = source
            merged["dropout"] = bool(dropout)
            merged["reward"] = float(reward)
            merged["regime"] = regime
            main_pieces.append(merged)

            pe_base = pe_base.copy()
            pe_base["paradigm"] = paradigm
            pe_base["source"] = source
            pe_base["dropout"] = bool(dropout)
            pe_base["reward"] = float(reward)
            pe_base["regime"] = regime
            per_eval_pieces.append(pe_base)

    main = pd.concat(main_pieces, ignore_index=True)
    per_eval = pd.concat(per_eval_pieces, ignore_index=True)
    cols_first = ["paradigm", "source", "dropout", "reward", "regime", "csf"]
    main = main[cols_first + [c for c in main.columns if c not in cols_first]]
    per_eval = per_eval[cols_first + [c for c in per_eval.columns if c not in cols_first]]
    return main, per_eval


def worked_examples_section() -> str:
    """Three side-by-side worked examples on the same toy cell."""
    lines = ["## Worked examples — three within-ε variants on the same toy cell\n\n"]
    lines.append(
        "Toy cell: `(paradigm=confidnet, source=cifar10, dropout=False, "
        "reward=2.2, regime=near)`. The CIFAR-10 near regime contains "
        "**two** eval_datasets: `{cifar100, tinyimagenet}`. Three CSFs "
        "(MSR, Energy, NeCo), 5 runs.\n\n"
    )
    rng = np.random.default_rng(42)
    csfs = ["MSR", "Energy", "NeCo"]
    runs = [1, 2, 3, 4, 5]
    # Construct per-(csf, run, eval) AUGRCs that illustrate the scale issue:
    #   cifar100 ≈ 100, tinyimagenet ≈ 250 (different native scales)
    #   ordering of CSFs differs by eval to expose Variant B sensitivity.
    base = {
        ("MSR", "cifar100"): 100, ("MSR", "tinyimagenet"): 260,
        ("Energy", "cifar100"): 105, ("Energy", "tinyimagenet"): 250,
        ("NeCo", "cifar100"): 130, ("NeCo", "tinyimagenet"): 245,
    }
    rows = []
    for csf in csfs:
        for r in runs:
            for ev in ["cifar100", "tinyimagenet"]:
                rows.append({"csf": csf, "run": r, "eval_dataset": ev,
                             "augrc": float(base[(csf, ev)] + rng.normal(0, 3))})
    df = pd.DataFrame(rows)
    # ranks within (eval_dataset)
    df["augrc_rank"] = df.groupby("eval_dataset")["augrc"].rank(pct=True)

    # show the toy table
    show = df.pivot_table(index=["csf", "run"], columns="eval_dataset",
                          values="augrc").round(1).reset_index()
    lines.append("Per-(csf, run, eval) raw AUGRC:\n\n")
    lines.append("```\n" + show.to_string(index=False) + "\n```\n\n")

    # ---------- Variant C ----------
    lines.append("### Variant C — raw AUGRC mean (original specification)\n\n")
    per_run_raw = df.groupby(["csf", "run"])["augrc"].mean().reset_index()
    per_csf_raw = per_run_raw.groupby("csf")["augrc"].mean().reset_index(name="summary_raw")
    lines.append(
        "Per-(csf, run) mean across `{cifar100, tinyimagenet}`. "
        "Note tinyimagenet (~250) dominates cifar100 (~100) in this average:\n\n"
    )
    lines.append("```\n" + per_csf_raw.round(2).sort_values("summary_raw").to_string(index=False) + "\n```\n\n")
    best_raw = per_csf_raw.loc[per_csf_raw["summary_raw"].idxmin()]
    best_per_run_raw = per_run_raw[per_run_raw["csf"] == best_raw["csf"]]["augrc"].values
    eps_r, _, _ = bootstrap_eps(best_per_run_raw)
    per_csf_raw["in_set_raw"] = per_csf_raw["summary_raw"] <= best_raw["summary_raw"] + eps_r
    lines.append(
        f"Best by raw mean: **{best_raw['csf']}** "
        f"(summary = {best_raw['summary_raw']:.2f}). "
        f"Bootstrap ε on best's 5 per-run means = **{eps_r:.2f}** (raw AUGRC units). "
        f"Set = `{per_csf_raw[per_csf_raw['in_set_raw']]['csf'].tolist()}`.\n\n"
    )

    # ---------- Variant A ----------
    lines.append("### Variant A — mean of `augrc_rank` (harmonized first)\n\n")
    per_run_rk = df.groupby(["csf", "run"])["augrc_rank"].mean().reset_index()
    per_csf_rk = per_run_rk.groupby("csf")["augrc_rank"].mean().reset_index(name="summary_rank")
    lines.append(
        "Same operation but on `augrc_rank` (within-(source, eval_dataset) "
        "percentile rank from step 4). Each eval_dataset now contributes "
        "equally because rank ∈ [0, 1] regardless of native scale:\n\n"
    )
    lines.append("```\n" + per_csf_rk.round(3).sort_values("summary_rank").to_string(index=False) + "\n```\n\n")
    best_rk = per_csf_rk.loc[per_csf_rk["summary_rank"].idxmin()]
    best_per_run_rk = per_run_rk[per_run_rk["csf"] == best_rk["csf"]]["augrc_rank"].values
    eps_a, _, _ = bootstrap_eps(best_per_run_rk)
    per_csf_rk["in_set_rank"] = per_csf_rk["summary_rank"] <= best_rk["summary_rank"] + eps_a
    lines.append(
        f"Best by rank mean: **{best_rk['csf']}** "
        f"(summary rank = {best_rk['summary_rank']:.3f}). "
        f"Bootstrap ε = **{eps_a:.3f}** (rank units). "
        f"Set = `{per_csf_rk[per_csf_rk['in_set_rank']]['csf'].tolist()}`.\n\n"
    )

    # ---------- Variant B ----------
    lines.append("### Variant B — per-eval within-ε, then aggregate\n\n")
    pe = per_eval_within_eps(df.assign(regime="near"))
    lines.append("Per-eval base table:\n\n")
    show_pe = pe[["csf", "eval_dataset", "summary_augrc_eval", "best_csf_eval",
                  "eps_eval", "in_set_eval"]].round(2)
    lines.append("```\n" + show_pe.to_string(index=False) + "\n```\n\n")
    pe_agg = aggregate_perEval_to_regime(pe)
    lines.append("Aggregation to regime level (n_evals = 2):\n\n")
    lines.append("```\n" + pe_agg.to_string(index=False) + "\n```\n\n")
    lines.append(
        "Reading: with N_evals=2, `majority` = ≥1 of 2; `unanimous` = 2 of 2. "
        "Variant B exposes per-eval disagreement that A and C smooth over: a "
        "CSF can be `majority`-competitive (passes on some evals) without "
        "being `unanimous` (passes on all). For larger regimes (CIFAR-10 mid "
        "has 4 evals; all has 8) the gap between majority and unanimous "
        "becomes more informative.\n\n"
    )
    lines.append("### How the three variants compare on this toy cell\n\n")
    cmp = pd.DataFrame({
        "csf": csfs,
        "C_raw": [c in per_csf_raw[per_csf_raw["in_set_raw"]]["csf"].tolist() for c in csfs],
        "A_rank": [c in per_csf_rk[per_csf_rk["in_set_rank"]]["csf"].tolist() for c in csfs],
        "B_majority": [bool(pe_agg[pe_agg["csf"] == c]["in_within_eps_set_majority"].iloc[0]) for c in csfs],
        "B_unanimous": [bool(pe_agg[pe_agg["csf"] == c]["in_within_eps_set_unanimous"].iloc[0]) for c in csfs],
    })
    lines.append("```\n" + cmp.to_string(index=False) + "\n```\n\n")
    return "".join(lines)


def report(out: pd.DataFrame, per_eval: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Step 6 — Within-ε label table (variants A, B, C)\n\n"]
    lines.append("**Date:** 2026-05-02\n")
    lines.append("**Source:** `code/nc_csf_predictivity/data/within_eps.py`\n\n")

    lines.append(worked_examples_section())

    lines.append("## Run summary\n\n")
    n_cells = out.groupby(["paradigm","source","dropout","reward","regime"]).ngroups
    lines.append(f"- Cells processed: {n_cells}\n")
    lines.append(f"- Total (cell × csf) rows: {len(out):,}\n")
    lines.append(f"- Per-eval base rows (Variant B intermediate): {len(per_eval):,}\n")
    lines.append(f"- Bootstrap n_boot = {N_BOOT}, seed = {SEED}\n\n")

    lines.append("## ε distribution per regime\n\n")
    eps_summary = pd.DataFrame({
        "eps_raw (AUGRC units)":
            out.drop_duplicates(["paradigm","source","dropout","reward","regime"])
               .groupby("regime")["eps_raw"].describe()["mean"].round(3),
        "eps_rank (rank units)":
            out.drop_duplicates(["paradigm","source","dropout","reward","regime"])
               .groupby("regime")["eps_rank"].describe()["mean"].round(4),
    })
    lines.append("```\n" + eps_summary.to_string() + "\n```\n\n")

    lines.append("## Within-ε set-size distribution per regime, per variant\n\n")
    set_size_pieces = []
    for col, label in [("in_within_eps_set_raw", "C_raw"),
                       ("in_within_eps_set_rank", "A_rank"),
                       ("in_within_eps_set_majority", "B_majority"),
                       ("in_within_eps_set_unanimous", "B_unanimous")]:
        sz = (out[out[col]]
              .groupby(["paradigm","source","dropout","reward","regime"]).size()
              .reset_index(name="set_size"))
        med = sz.groupby("regime")["set_size"].median().rename(f"median_set_size_{label}")
        mn = sz.groupby("regime")["set_size"].mean().rename(f"mean_set_size_{label}").round(2)
        set_size_pieces.append(pd.concat([mn, med], axis=1))
    set_size = pd.concat(set_size_pieces, axis=1)
    lines.append("```\n" + set_size.to_string() + "\n```\n\n")

    lines.append("## Pairwise agreement between variants (Jaccard)\n\n")
    lines.append(
        "Per-cell Jaccard between the in-set indicators of two variants, "
        "averaged across cells (within each regime). 1.0 = perfect agreement, "
        "0.0 = disjoint sets.\n\n"
    )
    def jaccard_per_cell(df, col_a, col_b):
        rows = []
        for keys, g in df.groupby(["paradigm","source","dropout","reward","regime"]):
            a = set(g[g[col_a]]["csf"])
            b = set(g[g[col_b]]["csf"])
            u = a | b
            j = len(a & b) / len(u) if u else float("nan")
            rows.append({"regime": keys[4], "jaccard": j})
        return pd.DataFrame(rows).groupby("regime")["jaccard"].mean().round(3)

    pairs = [
        ("in_within_eps_set_raw", "in_within_eps_set_rank", "C_raw vs A_rank"),
        ("in_within_eps_set_raw", "in_within_eps_set_majority", "C_raw vs B_majority"),
        ("in_within_eps_set_rank", "in_within_eps_set_majority", "A_rank vs B_majority"),
        ("in_within_eps_set_majority", "in_within_eps_set_unanimous", "B_majority vs B_unanimous"),
    ]
    jacc = pd.DataFrame({label: jaccard_per_cell(out, a, b) for a, b, label in pairs})
    lines.append("```\n" + jacc.to_string() + "\n```\n\n")

    lines.append("## Spot check — three variants vs Track 1 top clique\n\n")
    lines.append(
        "(dropout=False, lowest reward, regime ∈ {near, mid, far, all}) per "
        "(paradigm, source). Each row shows the top clique from step 5 "
        "alongside the three within-ε variants.\n\n"
    )
    cliques_path = (out_path.parent / "track1" / "cliques" / "cliques.parquet")
    if cliques_path.exists():
        cliques = pq.read_table(cliques_path).to_pandas()
        clq_top = cliques[cliques["in_top_clique"]]

        sub = out[(out["dropout"] == False)]
        min_rew = (sub.groupby(["paradigm", "source"])["reward"].min()
                   .rename("min_reward").reset_index())
        sub = sub.merge(min_rew, on=["paradigm", "source"])
        sub = sub[(sub["reward"] == sub["min_reward"]) &
                  sub["regime"].isin(["near", "mid", "far", "all"])]
        clq2 = clq_top.merge(min_rew, on=["paradigm", "source"])
        clq2 = clq2[(clq2["dropout"] == False) & (clq2["reward"] == clq2["min_reward"]) &
                    clq2["regime"].isin(["near", "mid", "far", "all"])]
        clq_grp = (clq2.sort_values("mean_rank")
                   .groupby(["paradigm","source","reward","regime"])["csf"]
                   .apply(lambda s: ", ".join(s.tolist())).reset_index()
                   .rename(columns={"csf": "top_clique"}))

        def grp(col):
            return (sub[sub[col]]
                    .sort_values("summary_raw")
                    .groupby(["paradigm","source","reward","regime"])["csf"]
                    .apply(lambda s: ", ".join(s.tolist())).reset_index()
                    .rename(columns={"csf": col}))

        joined = clq_grp
        for col in ("in_within_eps_set_raw", "in_within_eps_set_rank",
                    "in_within_eps_set_majority", "in_within_eps_set_unanimous"):
            joined = joined.merge(grp(col),
                                  on=["paradigm","source","reward","regime"],
                                  how="left")
        joined = joined.rename(columns={
            "in_within_eps_set_raw": "C_raw",
            "in_within_eps_set_rank": "A_rank",
            "in_within_eps_set_majority": "B_majority",
            "in_within_eps_set_unanimous": "B_unanimous",
        })
        lines.append("```\n" + joined.to_string(index=False) + "\n```\n")
    else:
        lines.append("(cliques.parquet not found — run step 5 first to enable spot check)\n")

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    in_path = out_root / "track1" / "dataset" / "long_harmonized.parquet"
    long_df = pq.read_table(in_path).to_pandas()
    long_df = long_df[(long_df["architecture"] == "VGG13")
                      & (long_df["paradigm"].isin(TRAIN_PARADIGMS))]

    main, per_eval = compute_all_variants(long_df)

    labels_dir = out_root / "track1" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(main, labels_dir / "within_eps.parquet")
    write_parquet(per_eval, labels_dir / "within_eps_per_eval.parquet")
    print(f"wrote {labels_dir / 'within_eps.parquet'} ({len(main):,} rows)")
    print(f"wrote {labels_dir / 'within_eps_per_eval.parquet'} ({len(per_eval):,} rows)")

    report(main, per_eval, out_root / "04_within_eps_check.md")
    print(f"wrote {out_root / '04_within_eps_check.md'}")


if __name__ == "__main__":
    main()

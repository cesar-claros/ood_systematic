"""Step 5: Recompute Track 1 top cliques per
`(paradigm, source, dropout, reward, regime)` cell on VGG13.

Reuses the existing pipeline from `code/src/utils_stats.py`:
  friedman_blocked → conover_posthoc_from_pivot → maximal_cliques_from_pmatrix
  → rank_cliques → greedy_exclusive_layers (top clique = layers[0]['members']).

Convention matches `code/stats_eval.py:163–180`: scores are NEGATED so higher
is better (matching the `score_std` convention upstream); `pivot.rank(axis=1,
ascending=False)` then yields rank=1 for the best CSF in each block.

Block columns: `(run, eval_dataset, metric ∈ {augrc, aurc})`. For a cell with
5 runs × 4 eval_datasets × 2 metrics = 40 blocks (when the regime contains
4 eval_datasets), Friedman has plenty of statistical power.

Per-cell JSON files written to `outputs/track1/cliques/per_cell/`; a flat
parquet `outputs/track1/cliques/cliques.parquet` joins easily downstream.
The check report has worked examples for each step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DATA_DIR.parent
CODE_DIR = PIPELINE_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

sys.path.insert(0, str(CODE_DIR))
from src.utils_stats import (  # noqa: E402
    friedman_blocked,
    conover_posthoc_from_pivot,
    maximal_cliques_from_pmatrix,
    rank_cliques,
    greedy_exclusive_layers,
)

REGIMES = ["near", "mid", "far", "all", "test"]  # all five; downstream filters
TRAIN_PARADIGMS = ["confidnet", "devries", "dg"]  # VGG13 CNN paradigms
ALPHA = 0.05


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def melt_cell_to_long(cell: pd.DataFrame) -> pd.DataFrame:
    """Stack augrc and aurc into a long table with metric ∈ {augrc, aurc}.
    Returns rows: (run, eval_dataset, regime, metric, csf, score) where
    score = -value (negated so higher = better)."""
    melted = []
    for metric_col in ("augrc", "aurc"):
        sub = cell[["run", "eval_dataset", "regime", "csf", metric_col]].copy()
        sub = sub.rename(columns={metric_col: "value"})
        sub["metric"] = metric_col
        sub["score"] = -sub["value"]
        melted.append(sub[["run", "eval_dataset", "regime", "metric", "csf", "score"]])
    return pd.concat(melted, ignore_index=True)


def cell_top_clique(cell_long: pd.DataFrame) -> dict:
    """Run the Friedman/Conover/clique pipeline on one cell's long table.
    Returns dict with: top_clique (list), ranks (dict csf→mean_rank),
    n_blocks, n_csfs, friedman_W, friedman_p, status."""
    cell_long = cell_long.copy()
    cell_long["block"] = (cell_long["run"].astype(str) + "|"
                          + cell_long["eval_dataset"].astype(str) + "|"
                          + cell_long["metric"].astype(str))
    try:
        stat, p, pivot = friedman_blocked(
            cell_long, entity_col="csf", block_col="block", value_col="score"
        )
    except Exception as e:  # noqa: BLE001
        return {"status": f"friedman_error:{e}", "top_clique": [], "ranks": {},
                "n_blocks": 0, "n_csfs": 0, "friedman_W": np.nan, "friedman_p": np.nan}

    if not isinstance(stat, float) or math.isnan(stat):
        return {"status": "degenerate", "top_clique": [], "ranks": {},
                "n_blocks": int(pivot.shape[0]), "n_csfs": int(pivot.shape[1]),
                "friedman_W": np.nan, "friedman_p": np.nan}

    ph = conover_posthoc_from_pivot(pivot)
    ranks = pivot.rank(axis=1, ascending=False)  # 1 = best
    avg_ranks = ranks.mean(axis=0).sort_values()
    cliques = maximal_cliques_from_pmatrix(ph, alpha=ALPHA)
    scored = rank_cliques(cliques, list(avg_ranks.index), avg_ranks)
    layers = greedy_exclusive_layers(scored)
    top = layers[0]["members"] if layers else []

    return {
        "status": "ok",
        "top_clique": list(top),
        "ranks": {csf: float(avg_ranks.loc[csf]) for csf in avg_ranks.index},
        "n_blocks": int(pivot.shape[0]),
        "n_csfs": int(pivot.shape[1]),
        "friedman_W": float(stat),
        "friedman_p": float(p),
    }


def restrict_to_regime(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "all":
        return df[df["regime"].isin(["near", "mid", "far"])]
    if regime in ("test", "near", "mid", "far"):
        return df[df["regime"] == regime]
    raise ValueError(f"Unknown regime: {regime}")


def compute_track1_cliques(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Iterate cells; return flat parquet rows and nested JSON-friendly dict."""
    flat_rows = []
    nested: dict = {}

    cell_keys = ["paradigm", "source", "dropout", "reward"]
    for (paradigm, source, dropout, reward), cell in long_df.groupby(cell_keys):
        cell_long = melt_cell_to_long(cell)
        cell_csfs = sorted(cell["csf"].unique())
        do_token = "do1" if dropout else "do0"
        rew_token = f"rew{reward:g}"
        cell_id = f"{do_token}_{rew_token}"
        nested.setdefault(paradigm, {}).setdefault(source, {})[cell_id] = {}

        for regime in REGIMES:
            sub = restrict_to_regime(cell_long, regime)
            if sub.empty:
                nested[paradigm][source][cell_id][regime] = {
                    "status": "empty", "top_clique": [], "ranks": {},
                    "n_blocks": 0, "n_csfs": 0,
                    "friedman_W": None, "friedman_p": None,
                }
                continue
            res = cell_top_clique(sub)
            nested[paradigm][source][cell_id][regime] = res

            top_set = set(res["top_clique"])
            for csf in cell_csfs:
                flat_rows.append({
                    "paradigm": paradigm, "source": source,
                    "dropout": bool(dropout), "reward": float(reward),
                    "regime": regime, "csf": csf,
                    "in_top_clique": csf in top_set,
                    "mean_rank": res["ranks"].get(csf, np.nan),
                    "n_blocks": res["n_blocks"],
                    "friedman_p": res["friedman_p"],
                })

    flat = pd.DataFrame(flat_rows)
    return flat, nested


def worked_example_section() -> str:
    """Tiny synthetic Friedman/Conover/clique walk-through."""
    lines = ["## Worked example — Friedman/Conover/clique on a 3×4 toy table\n\n"]
    lines.append(
        "Three CSFs (A, B, C) evaluated on four blocks. Values shown are the "
        "raw AUGRC (lower = better). The cell pipeline negates these to get "
        "scores (higher = better) so the existing Friedman/post-hoc code can "
        "be reused unchanged.\n\n"
    )

    raw = pd.DataFrame({
        "A": [10, 12, 11, 13],
        "B": [20, 18, 22, 19],
        "C": [11, 12, 12, 13],
    }, index=["block1", "block2", "block3", "block4"])
    lines.append("Raw AUGRC:\n\n```\n" + raw.to_string() + "\n```\n\n")

    score = -raw
    lines.append("After negation (`score = -augrc`, higher = better):\n\n")
    lines.append("```\n" + score.to_string() + "\n```\n\n")

    ranks = score.rank(axis=1, ascending=False, method="average")
    lines.append("Rank within each block (`ascending=False` ⇒ rank 1 = highest score = best):\n\n")
    lines.append("```\n" + ranks.to_string() + "\n```\n\n")

    avg_ranks = ranks.mean(axis=0).sort_values()
    lines.append("Mean rank per CSF (sorted):\n\n```\n" + avg_ranks.round(3).to_string() + "\n```\n\n")

    from scipy import stats as scstats
    chi, p = scstats.friedmanchisquare(*[score[c].values for c in score.columns])
    lines.append(
        f"Friedman χ² statistic = **{chi:.3f}**, p = **{p:.4f}**. With p < 0.05, "
        "we reject H₀ that all CSFs are equivalent on these blocks → there is "
        "at least one significant pairwise difference, so a Conover post-hoc "
        "is run to identify which pairs.\n\n"
    )

    import scikit_posthocs as sp
    ph = sp.posthoc_conover_friedman(score.values)
    ph.index = list(score.columns)
    ph.columns = list(score.columns)
    lines.append("Conover post-hoc p-matrix:\n\n```\n" + ph.round(4).to_string() + "\n```\n\n")

    edge_thresh = 0.05
    lines.append(
        f"Build the 'not-significant' graph: edge between CSF i and j iff "
        f"p_ij ≥ {edge_thresh}. Maximal cliques (Bron-Kerbosch) on this graph "
        f"are the candidate top cliques.\n\n"
    )
    edges = []
    for i in ph.index:
        for j in ph.columns:
            if i < j and ph.loc[i, j] >= edge_thresh:
                edges.append((i, j))
    lines.append(f"Edges (not significantly different): {edges or 'none'}\n\n")

    if edges:
        lines.append(
            "Suppose the only edge is `A — C`. Then maximal cliques are "
            "`{A, C}` and `{B}`. After scoring by best rank, the top clique "
            "(layers[0]) is `{A, C}` (because A's mean rank is 1.0 vs C's "
            "≈1.5 vs B's 3.0). The flat parquet row for this cell would have "
            "`in_top_clique=True` for A and C, `False` for B.\n\n"
        )
    else:
        lines.append(
            "With no edges, every CSF is its own clique. The top clique is "
            "the singleton with the lowest mean rank — here `{A}`.\n\n"
        )
    return "".join(lines)


def report(flat: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Step 5 — Track 1 clique recomputation\n\n"]
    lines.append("**Date:** 2026-05-02\n")
    lines.append("**Source:** `code/nc_csf_predictivity/data/cliques_track1.py`\n\n")

    lines.append(worked_example_section())

    cells_total = flat.groupby(
        ["paradigm", "source", "dropout", "reward", "regime"]
    ).ngroups
    lines.append("## Run summary\n\n")
    lines.append(f"- Cells processed: {cells_total}\n")
    by_status = flat.groupby("regime").size().rename("rows")
    lines.append("\nRows in flat parquet per regime:\n\n```\n" + by_status.to_string() + "\n```\n\n")

    lines.append("## Top-clique size distribution per regime\n\n")
    sz = (flat[flat["in_top_clique"]]
          .groupby(["paradigm", "source", "dropout", "reward", "regime"]).size()
          .reset_index(name="top_clique_size"))
    summary = sz.groupby("regime")["top_clique_size"].describe().round(2)
    lines.append("```\n" + summary.to_string() + "\n```\n\n")

    lines.append("## Spot check — top cliques for one cell per (paradigm, source)\n\n")
    lines.append(
        "Showing the top clique for `(dropout=False, lowest reward, regime ∈ "
        "{near, mid, far, all})` per (paradigm, source). This lets us eyeball "
        "the cliques against the published per-paradigm cliques in "
        "`code/ood_eval_outputs/top_cliques_Conv_False_RC_<paradigm>_cliques.json` "
        "(which pool over dropout/reward, so they are coarser than ours).\n\n"
    )
    pick = flat[(flat["dropout"] == False) & flat["in_top_clique"]].copy()
    if not pick.empty:
        pick_min = pick.merge(
            pick.groupby(["paradigm", "source"])["reward"].min().rename("min_reward").reset_index(),
            on=["paradigm", "source"]
        )
        pick_min = pick_min[pick_min["reward"] == pick_min["min_reward"]]
        spot = pick_min[pick_min["regime"].isin(["near", "mid", "far", "all"])].copy()
        spot = spot.sort_values(["paradigm", "source", "reward", "regime", "mean_rank"])
        per_cell = spot.groupby(["paradigm", "source", "reward", "regime"])["csf"].apply(
            lambda s: ", ".join(s.tolist())
        ).reset_index().rename(columns={"csf": "top_clique"})
        lines.append("```\n" + per_cell.to_string(index=False) + "\n```\n")
    return "".join(lines).rstrip() + "\n", out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    in_path = out_root / "track1" / "dataset" / "long_harmonized.parquet"
    long_df = pq.read_table(in_path).to_pandas()
    long_df = long_df[(long_df["architecture"] == "VGG13")
                      & (long_df["paradigm"].isin(TRAIN_PARADIGMS))]

    flat, nested = compute_track1_cliques(long_df)

    cliques_dir = out_root / "track1" / "cliques"
    per_cell_dir = cliques_dir / "per_cell"
    cliques_dir.mkdir(parents=True, exist_ok=True)
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(flat, cliques_dir / "cliques.parquet")
    print(f"wrote {cliques_dir / 'cliques.parquet'} ({len(flat):,} rows)")

    for paradigm, payload in nested.items():
        path = per_cell_dir / f"{paradigm}_cliques.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"wrote {path}")

    report_text, report_path = report(flat, out_root / "03_cliques_check.md")
    report_path.write_text(report_text)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

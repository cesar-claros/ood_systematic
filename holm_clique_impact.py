"""Quantify the effect of Holm adjustment on Conover clique membership.

The paper text describes Holm-adjusted Conover p-values for the indifference
graph; `src.utils_stats.conover_posthoc_from_pivot` computes unadjusted ones.
Unadjusted p-values are smaller, so edges (p >= alpha) are fewer and cliques
are stricter; Holm can only add edges and grow cliques. This script reruns the
step-5 clique computation (VGG-13 CNN cells and the ViT cells) under both
settings and reports how often the top clique changes, the Jaccard overlap,
and the size delta.

Run from `code/`:
  ./.venv/bin/python holm_clique_impact.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "nc_csf_predictivity" / "data"))

import pandas as pd
import pyarrow.parquet as pq
import scikit_posthocs as sp

import cliques_track1


def conover_holm(pivot: pd.DataFrame) -> pd.DataFrame:
    """Conover-Friedman post-hoc with Holm step-down adjustment."""
    ph = sp.posthoc_conover_friedman(pivot.values, p_adjust="holm")
    ph.index = pivot.columns
    ph.columns = pivot.columns
    return ph


def top_cliques(long_df: pd.DataFrame, holm: bool) -> pd.DataFrame:
    """Flat clique table under the requested adjustment setting."""
    original = cliques_track1.conover_posthoc_from_pivot
    if holm:
        cliques_track1.conover_posthoc_from_pivot = conover_holm
    try:
        flat, _ = cliques_track1.compute_track1_cliques(long_df)
    finally:
        cliques_track1.conover_posthoc_from_pivot = original
    return flat


def compare(flat_u: pd.DataFrame, flat_h: pd.DataFrame, label: str) -> dict:
    """Membership agreement between unadjusted and Holm cliques."""
    keys = ["paradigm", "source", "dropout", "reward", "regime"]
    u = (flat_u[flat_u["in_top_clique"]]
         .groupby(keys)["csf"].apply(frozenset))
    h = (flat_h[flat_h["in_top_clique"]]
         .groupby(keys)["csf"].apply(frozenset))
    joined = pd.concat([u.rename("u"), h.rename("h")], axis=1).dropna()
    joined = joined[[r not in ("test",) for r in
                     joined.index.get_level_values("regime")]]
    jac = joined.apply(
        lambda r: len(r["u"] & r["h"]) / len(r["u"] | r["h"]), axis=1)
    grew = joined.apply(lambda r: r["u"] < r["h"], axis=1)
    same = joined.apply(lambda r: r["u"] == r["h"], axis=1)
    winner_kept = joined.apply(lambda r: bool(r["u"] & r["h"]), axis=1)
    return {
        "pool": label,
        "cells": len(joined),
        "identical_top_clique": f"{same.mean():.1%}",
        "unadjusted_subset_of_holm": f"{(joined.apply(lambda r: r['u'] <= r['h'], axis=1)).mean():.1%}",
        "strictly_grew": f"{grew.mean():.1%}",
        "mean_jaccard": round(float(jac.mean()), 3),
        "min_jaccard": round(float(jac.min()), 3),
        "mean_size_unadj": round(float(joined["u"].apply(len).mean()), 2),
        "mean_size_holm": round(float(joined["h"].apply(len).mean()), 2),
        "winner_overlap_nonempty": f"{winner_kept.mean():.1%}",
    }


def main() -> None:
    """Compare unadjusted vs Holm cliques on the CNN and ViT pools."""
    out_root = CODE_DIR / "nc_csf_predictivity" / "outputs"
    long_df = pq.read_table(
        out_root / "track1" / "dataset" / "long_harmonized.parquet").to_pandas()

    pools = {
        "VGG13 CNN": long_df[(long_df["architecture"] == "VGG13")
                             & long_df["paradigm"].isin(
                                 ["confidnet", "devries", "dg"])],
        "ViT": long_df[(long_df["architecture"] == "ViT")
                       & (long_df["paradigm"] == "modelvit")],
    }
    rows = []
    for label, pool in pools.items():
        flat_u = top_cliques(pool, holm=False)
        flat_h = top_cliques(pool, holm=True)
        rows.append(compare(flat_u, flat_h, label))
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    out_path = CODE_DIR / "mantel_partial_outputs" / "holm_clique_impact.md"
    out_path.write_text(
        "# Holm vs unadjusted Conover cliques (step-5 granularity)\n\n"
        + summary.to_markdown(index=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

"""Figure 1 top cliques with Holm-adjusted Conover p-values (camera-ready item 0.4).

The paper's cliques use unadjusted Conover p-values while the text says
Conover-Holm. This script recomputes the 22-CSF Figure 1 inputs (paper
scores_all files plus validation-selected Maha++ and NCI rows) with Holm
adjustment over all CSF pairs of a cell, and compares every cell with the
unadjusted cliques.

Readouts: per cell, whether the unadjusted top clique is a subset of the Holm
top clique, whether they are identical, and their sizes; per (backbone,
regime), how many cells each CSF enters; per panel, the three most frequently
tied CSFs (the dotted rectangles of Figure 1), over the OOD cells and over
all 16 columns.

Label check: the same comparison on the predictor's training labels (the
22-CSF per-configuration VGG13 top cliques of `benchmark_newcsf_cliques.py`,
near/mid/far cells).

Gates (abort before any Holm number is computed):
  GF  the unadjusted run reproduces the stored 22-CSF top clique of all 48
      OOD cells of Figure 1 (outputs/40_figure1_newcsf_cliques_bo.md, extended
      column; the report does not list the 16 ID-test columns).
  GL  the unadjusted per-configuration run reproduces the stored 22-CSF
      labels (perconfig_cliques_bo.parquet) exactly.

Outputs: outputs/figure1_holm/cells.csv, outputs/56_figure1_holm_check.md
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
from benchmark_newcsf_cliques import (  # noqa: E402
    DISPLAY as LABEL_DISPLAY,
    FEATURE_SIDE,
    HEAD_SIDE,
    NEW_CSFS,
    load_base,
    perconfig_cliques,
)
from figure1_newcsf_cliques import (  # noqa: E402
    CODE_DIR,
    DEFAULT_OUT_ROOT,
    SOURCES,
    clip_groups,
    load_scores_all,
    panel_cliques,
    select_new_csf_rows,
)

STORED = "40_figure1_newcsf_cliques_bo.md"
LONG_CSV = CODE_DIR / "new_csfs_pilot_summary_long.csv"
DISPLAY = {"modelvit": "vit", "KPCA RecError global": "KPCA RecError",
           "PCA RecError global": "PCA RecError"}
PANELS = ("confidnet", "devries", "dg", "vit")
OOD = ("near", "mid", "far")


def stored_cliques(out_root: Path) -> dict[tuple[str, str, str], list[str]]:
    """Extended 22-CSF top cliques of every Figure 1 cell in the stored report."""
    cells = {}
    for line in (out_root / STORED).read_text().splitlines():
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) == 5 and parts[0] in PANELS:
            cells[tuple(parts[:3])] = sorted(m.strip().strip("*")
                                             for m in parts[4].split(",") if m.strip())
    return cells


def all_cliques(p_adjust: str | None) -> dict[tuple[str, str, str], list[str]]:
    """Top clique of every Figure 1 cell for one p-value adjustment."""
    out = {}
    for backbone in ("Conv", "ViT"):
        for source in SOURCES:
            rows = pd.concat([load_scores_all(source, backbone),
                              select_new_csf_rows(LONG_CSV, source, backbone)])
            groups = clip_groups(CODE_DIR / "clip_scores", source)
            for (panel, regime), res in panel_cliques(rows, backbone, source, groups,
                                                      p_adjust=p_adjust).items():
                out[(DISPLAY.get(panel, panel), source, regime)] = sorted(
                    DISPLAY.get(m, m) for m in res["top_clique"])
    return out


def top_three(cells: pd.DataFrame, column: str) -> pd.DataFrame:
    """The three most frequently tied CSFs per panel (ties broken alphabetically)."""
    recs = []
    for panel, g in cells.groupby("panel", sort=False):
        counts = Counter(m for clique in g[column] for m in clique)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        recs.append({"panel": panel, "top3": ", ".join(f"{m} ({n})" for m, n in ranked)})
    return pd.DataFrame(recs)


def label_check(out_root: Path) -> pd.DataFrame:
    """Unadjusted vs Holm top cliques at the predictor-label granularity.

    Raises:
        RuntimeError: If the unadjusted run differs from the stored labels (GL).
    """
    base, _ = load_base(LONG_CSV, CODE_DIR / "clip_scores")
    roster = [LABEL_DISPLAY.get(c, c) for c in HEAD_SIDE + FEATURE_SIDE] + NEW_CSFS
    keys = ["paradigm", "source", "dropout", "reward", "regime", "csf"]
    plain = perconfig_cliques(base, roster)
    stored = pd.read_parquet(out_root / "newcsf_cliques" / "perconfig_cliques_bo.parquet")
    both = stored.merge(plain, on=keys, suffixes=("", "_new"))
    if len(both) != len(stored) or (both["in_top_clique"] != both["in_top_clique_new"]).any():
        raise RuntimeError("GL failed: per-configuration labels differ from the stored ones")
    print(f"GL: {len(both):,} stored per-configuration labels reproduced")
    holm = perconfig_cliques(base, roster, p_adjust="holm")
    merged = plain.merge(holm, on=keys, suffixes=("_plain", "_holm"))
    merged = merged[merged["regime"].isin(OOD)]
    cell = merged.groupby(keys[:-1]).agg(
        plain=("in_top_clique_plain", "sum"), holm=("in_top_clique_holm", "sum"),
        lost=("in_top_clique_plain", lambda s: int((s & ~merged.loc[s.index,
                                                    "in_top_clique_holm"]).sum())))
    return pd.DataFrame([{"cells": len(cell), "subset": int((cell["lost"] == 0).sum()),
                          "identical": int((cell["plain"] == cell["holm"]).sum()),
                          "mean_size_unadjusted": cell["plain"].mean(),
                          "mean_size_holm": cell["holm"].mean(),
                          "positive_labels_unadjusted": int(cell["plain"].sum()),
                          "positive_labels_holm": int(cell["holm"].sum())}])


def main() -> None:
    """Gate, Holm cliques, comparison, and report."""
    out_root = Path(DEFAULT_OUT_ROOT)
    stored = stored_cliques(out_root)
    plain = all_cliques(None)
    bad = [k for k in stored if plain.get(k) != stored[k]]
    if bad or len(stored) != 48:
        raise RuntimeError(f"GF failed: {len(bad)} of {len(stored)} stored OOD cells "
                           f"differ; first {bad[:3]}")
    print("GF: unadjusted run reproduces all 48 stored OOD cliques of Figure 1")
    holm = all_cliques("holm")
    labels = label_check(out_root)

    cells = pd.DataFrame([{"panel": p, "source": s, "regime": r,
                           "unadjusted": plain[(p, s, r)], "holm": holm[(p, s, r)]}
                          for (p, s, r) in plain])
    cells["subset"] = [set(u) <= set(h) for u, h in zip(cells["unadjusted"], cells["holm"])]
    cells["identical"] = cells["unadjusted"] == cells["holm"]
    cells["size_unadjusted"] = cells["unadjusted"].map(len)
    cells["size_holm"] = cells["holm"].map(len)
    ood = cells[cells["regime"].isin(OOD)]
    backbone = ood["panel"].map(lambda p: "ViT" if p == "vit" else "VGG13")
    summary = ood.groupby(backbone).agg(
        cells=("subset", "size"), subset=("subset", "sum"), identical=("identical", "sum"),
        mean_size_unadjusted=("size_unadjusted", "mean"),
        mean_size_holm=("size_holm", "mean"))
    membership = []
    for column in ("unadjusted", "holm"):
        exploded = ood.assign(backbone=backbone, csf=ood[column]).explode("csf")
        membership.append(exploded.groupby(["backbone", "regime", "csf"]).size()
                          .rename(column))
    membership = (pd.concat(membership, axis=1).fillna(0).astype(int)
                  .sort_values(["backbone", "regime", "holm"], ascending=[True, True, False]))
    tops = top_three(ood, "unadjusted").merge(top_three(ood, "holm"), on="panel",
                                              suffixes=("_unadjusted", "_holm"))
    tops_all = top_three(cells, "unadjusted").merge(top_three(cells, "holm"), on="panel",
                                                    suffixes=("_unadjusted", "_holm"))

    csv_dir = out_root / "figure1_holm"
    csv_dir.mkdir(parents=True, exist_ok=True)
    cells.assign(unadjusted=cells["unadjusted"].map(", ".join),
                 holm=cells["holm"].map(", ".join)).to_csv(csv_dir / "cells.csv", index=False)
    changed = ood[~ood["identical"]]
    lines = ["# Figure 1 cliques: unadjusted vs Holm-adjusted Conover p-values\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/figure1_holm_check.py`. 22-CSF "
             "Figure 1 inputs; Holm over all CSF pairs of a cell; alpha 0.05. Gate GF "
             "passed.\n\n## OOD cells (48)\n\n```\n", summary.round(2).to_string(),
             "\n```\n\n## Predictor labels (22-CSF per-configuration VGG13 cells, "
             "near/mid/far)\n\n```\n", labels.round(2).to_string(index=False),
             "\n```\n\nCells where the unadjusted clique is NOT a subset of the Holm "
             f"clique: {int((~ood['subset']).sum())}\n\n",
             "## Changed OOD cells\n\n```\n",
             changed.assign(unadjusted=changed["unadjusted"].map(", ".join),
                            holm=changed["holm"].map(", ".join))
             [["panel", "source", "regime", "unadjusted", "holm"]].to_string(index=False),
             "\n```\n\n## Three most frequently tied CSFs per panel (OOD cells)\n\n```\n",
             tops.to_string(index=False), "\n```\n\n## Same, all 16 columns\n\n```\n",
             tops_all.to_string(index=False), "\n```\n\n",
             "## Cells entered per CSF (OOD cells)\n\n```\n", membership.to_string(),
             "\n```\n"]
    (out_root / "56_figure1_holm_check.md").write_text("".join(lines))
    print(f"wrote {out_root / '56_figure1_holm_check.md'}")


if __name__ == "__main__":
    main()

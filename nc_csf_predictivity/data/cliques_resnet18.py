"""Step 8: ResNet18 sanity-check cliques (diagnostic only).

Runs the same Friedman/Conover/maximal-clique pipeline as step 5, but on
ResNet18 rows. Because ResNet18 has only 1 run per cell, the only block
dimensions are (eval_dataset × metric ∈ {augrc, aurc}). For OOD regimes with
1 eval_dataset (e.g., source=tinyimagenet, regime=far has only `svhn`) the
cell yields only 2 blocks → Friedman has effectively no power and the top
clique collapses to all CSFs. Those degenerate cells are reported and flagged.

This step is **diagnostic only**. The headline ResNet18 evaluation uses
oracle/regret from step 7 (no Friedman blocks needed); these cliques exist
to spot-check whether the regret-based picture is consistent with what
Friedman would say if it had power.

Outputs:
  outputs/track1/cliques/resnet18_sanity_cliques.parquet
  outputs/track1/cliques/per_cell/resnet18_sanity_<paradigm>_cliques.json
  outputs/06_resnet18_sanity_check.md
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

REGIMES = ["near", "mid", "far", "all", "test"]
TEST_PARADIGMS = ["confidnet", "devries", "dg"]  # ResNet18 = CNN-only
ALPHA = 0.05


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(path))


def melt_cell_to_long(cell: pd.DataFrame) -> pd.DataFrame:
    """Stack augrc and aurc; keep regime for filtering. Score = -value."""
    melted = []
    for col in ("augrc", "aurc"):
        sub = cell[["eval_dataset", "regime", "csf", col]].copy()
        sub = sub.rename(columns={col: "value"})
        sub["metric"] = col
        sub["score"] = -sub["value"]
        melted.append(sub[["eval_dataset", "regime", "metric", "csf", "score"]])
    return pd.concat(melted, ignore_index=True)


def cell_top_clique(cell_long: pd.DataFrame) -> dict:
    """Same recipe as step 5 but blocks = (eval_dataset × metric)."""
    cell_long = cell_long.copy()
    cell_long["block"] = (cell_long["eval_dataset"].astype(str) + "|"
                          + cell_long["metric"].astype(str))
    try:
        stat, p, pivot = friedman_blocked(
            cell_long, entity_col="csf", block_col="block", value_col="score"
        )
    except Exception as e:  # noqa: BLE001
        return {"status": f"friedman_error:{e}", "top_clique": [], "ranks": {},
                "n_blocks": 0, "n_csfs": 0,
                "friedman_W": np.nan, "friedman_p": np.nan}

    if not isinstance(stat, float) or math.isnan(stat):
        return {"status": "degenerate", "top_clique": [], "ranks": {},
                "n_blocks": int(pivot.shape[0]), "n_csfs": int(pivot.shape[1]),
                "friedman_W": np.nan, "friedman_p": np.nan}

    ph = conover_posthoc_from_pivot(pivot)
    ranks = pivot.rank(axis=1, ascending=False)
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
    return df[df["regime"] == regime]


def compute_resnet18_cliques(long_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
                    "status": res["status"],
                })

    return pd.DataFrame(flat_rows), nested


def worked_examples_section() -> str:
    """Two side-by-side ResNet18 cell illustrations: degenerate vs richer."""
    lines = ["## Worked examples — degenerate vs richer block cells\n\n"]
    lines.append(
        "Because ResNet18 has 1 run per cell, the block dimensions for "
        "Friedman are only `(eval_dataset × metric ∈ {augrc, aurc})`. The "
        "block count therefore depends on how many eval_datasets the regime "
        "contains.\n\n"
    )

    lines.append("### Degenerate case — `(source=tinyimagenet, regime=far)`\n\n")
    lines.append(
        "The CIFAR/CLIP grouping for tinyimagenet puts only `svhn` into the "
        "`far` regime (cifar10/cifar100/etc. land in `near`; places365/textures "
        "in `mid`). So the cell has:\n\n"
        "- 1 eval_dataset × 2 metrics = **2 blocks**\n"
        "- 20 CSFs (entities)\n\n"
        "Friedman with 2 blocks and 20 entities is technically computable but "
        "has effectively no statistical power — it almost always fails to "
        "reject H₀, and the Conover post-hoc returns p ≈ 1 for nearly all "
        "pairs. Consequence: the 'not-significantly-different' graph is "
        "fully connected and the top clique = all 20 CSFs.\n\n"
        "These cells are flagged with `n_blocks ≤ 4` in the parquet so "
        "downstream code can drop them or assign zero weight.\n\n"
    )

    lines.append("### Richer case — `(source=cifar10, regime=all)`\n\n")
    lines.append(
        "CIFAR-10's `all` regime pools the 8 OOD eval_datasets, giving:\n\n"
        "- 8 eval_datasets × 2 metrics = **16 blocks**\n"
        "- 20 CSFs (entities)\n\n"
        "16 blocks is enough for Friedman to detect real differences across "
        "CSFs and for Conover to identify pairwise ties. The top clique here "
        "should look qualitatively similar to the corresponding VGG13 cliques "
        "from step 5 if the cross-architecture transfer hypothesis holds. "
        "Disagreement between the ResNet18 sanity clique and the VGG13 clique "
        "(at the same paradigm/source/dropout/reward/regime) flags either "
        "(a) genuine cross-arch CSF reordering — interesting for the paper, "
        "or (b) ResNet18's single-run noise creating a misleading clique. "
        "The regret-based ResNet18 evaluation in step 13 is the authoritative "
        "test; this clique is corroborative only.\n\n"
    )

    lines.append("### Hand-trace of the Friedman block construction\n\n")
    lines.append(
        "For one ResNet18 cell `(confidnet, cifar10, dropout=False, "
        "reward=2.2, regime=all)`, the Friedman pivot is shaped 16 × 20 "
        "(blocks × CSFs):\n\n"
        "```\n"
        "block                            CSF1  CSF2  ...  CSF20\n"
        "cifar100|augrc                   ...\n"
        "cifar100|aurc                    ...\n"
        "tinyimagenet|augrc               ...\n"
        "tinyimagenet|aurc                ...\n"
        "lsun resize|augrc                ...\n"
        "lsun resize|aurc                 ...\n"
        "...                              ...\n"
        "(8 evals × 2 metrics = 16 rows)\n"
        "```\n\n"
        "Each row is a single block: the per-block ranking of CSFs is what "
        "Friedman tests for consistency. With only 1 run, augrc and aurc are "
        "deterministic functions of the same logits, so the two metric blocks "
        "for a given eval are correlated — but they are not identical (AUGRC "
        "weights misclassification differently than AURC), so they do add "
        "real information.\n\n"
    )
    return "".join(lines)


def report(flat: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Step 8 — ResNet18 sanity-check cliques\n\n"]
    lines.append("**Date:** 2026-05-03\n")
    lines.append("**Source:** `code/nc_csf_predictivity/data/cliques_resnet18.py`\n\n")
    lines.append(worked_examples_section())

    lines.append("## Run summary\n\n")
    n_cells = flat.groupby(["paradigm","source","dropout","reward","regime"]).ngroups
    lines.append(f"- Cells processed: {n_cells}\n")
    lines.append(f"- Total (cell × csf) rows: {len(flat):,}\n\n")

    lines.append("### Status counts per regime\n\n")
    cell_status = (flat.drop_duplicates(["paradigm","source","dropout","reward","regime"])
                   .groupby(["regime","status"]).size().unstack(fill_value=0))
    lines.append("```\n" + cell_status.to_string() + "\n```\n\n")

    lines.append("### n_blocks distribution per regime\n\n")
    nb = (flat.drop_duplicates(["paradigm","source","dropout","reward","regime"])
          .groupby("regime")["n_blocks"].describe().round(2))
    lines.append("```\n" + nb.to_string() + "\n```\n\n")

    lines.append("### Top-clique size distribution per regime (where status==ok)\n\n")
    ok = flat[flat["status"] == "ok"]
    sz = (ok[ok["in_top_clique"]]
          .groupby(["paradigm","source","dropout","reward","regime"]).size()
          .reset_index(name="top_clique_size"))
    if not sz.empty:
        summary = sz.groupby("regime")["top_clique_size"].describe().round(2)
        lines.append("```\n" + summary.to_string() + "\n```\n\n")

    lines.append("### Spot check vs VGG13 step-5 cliques on the same cells\n\n")
    lines.append(
        "(dropout=False, lowest reward) per (paradigm, source) and regime ∈ "
        "{near, mid, far, all}. ResNet18 sanity clique vs VGG13 clique from "
        "step 5. Substantial agreement supports cross-arch transfer; "
        "disagreement is flagged for inspection.\n\n"
    )
    vgg_path = out_path.parent / "track1" / "cliques" / "cliques.parquet"
    if vgg_path.exists():
        vgg = pq.read_table(vgg_path).to_pandas()
        vgg_top = vgg[vgg["in_top_clique"]]

        sub = flat[(flat["dropout"] == False) & (flat["status"] == "ok")]
        if not sub.empty:
            min_rew = (sub.groupby(["paradigm","source"])["reward"].min()
                       .rename("min_reward").reset_index())
            sub = sub.merge(min_rew, on=["paradigm","source"])
            sub = sub[sub["reward"] == sub["min_reward"]]
            sub = sub[sub["regime"].isin(["near","mid","far","all"])]
            r18_grp = (sub[sub["in_top_clique"]]
                       .sort_values("mean_rank")
                       .groupby(["paradigm","source","reward","regime"])["csf"]
                       .apply(lambda s: ", ".join(s.tolist()))
                       .reset_index().rename(columns={"csf":"resnet18_sanity"}))

            vgg2 = vgg_top.merge(min_rew, on=["paradigm","source"])
            vgg2 = vgg2[(vgg2["dropout"] == False) & (vgg2["reward"] == vgg2["min_reward"])
                        & vgg2["regime"].isin(["near","mid","far","all"])]
            vgg_grp = (vgg2.sort_values("mean_rank")
                       .groupby(["paradigm","source","reward","regime"])["csf"]
                       .apply(lambda s: ", ".join(s.tolist()))
                       .reset_index().rename(columns={"csf":"vgg13_step5"}))

            joined = r18_grp.merge(vgg_grp, on=["paradigm","source","reward","regime"], how="outer")
            lines.append("```\n" + joined.to_string(index=False) + "\n```\n")
        else:
            lines.append("(no ok cells for spot check)\n")
    else:
        lines.append("(VGG13 cliques.parquet not found — run step 5 first)\n")

    out_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    in_path = out_root / "track1" / "dataset" / "long_harmonized.parquet"
    long_df = pq.read_table(in_path).to_pandas()
    long_df = long_df[(long_df["architecture"] == "ResNet18")
                      & (long_df["paradigm"].isin(TEST_PARADIGMS))]

    flat, nested = compute_resnet18_cliques(long_df)

    cliques_dir = out_root / "track1" / "cliques"
    per_cell_dir = cliques_dir / "per_cell"
    cliques_dir.mkdir(parents=True, exist_ok=True)
    per_cell_dir.mkdir(parents=True, exist_ok=True)

    write_parquet(flat, cliques_dir / "resnet18_sanity_cliques.parquet")
    print(f"wrote {cliques_dir / 'resnet18_sanity_cliques.parquet'} ({len(flat):,} rows)")

    for paradigm, payload in nested.items():
        path = per_cell_dir / f"resnet18_sanity_{paradigm}_cliques.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"wrote {path}")

    report(flat, out_root / "06_resnet18_sanity_check.md")
    print(f"wrote {out_root / '06_resnet18_sanity_check.md'}")


if __name__ == "__main__":
    main()

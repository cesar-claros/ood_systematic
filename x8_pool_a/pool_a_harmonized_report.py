"""Addendum tables for the harmonized SSL rerun: the comparisons the first
report omitted, computed from the saved parquets (no refitting).

  1. Full GLiC-2-style tables (regime x source + pooled) with the
     harmonized BEST-FIXED baseline (value + name) next to NC+source and
     NC only, at both rosters: 18 (submitted-comparable) and 21 (revision).
     The submitted claim was NC-only beating best-fixed in every pooled
     cell (1.59/1.38/0.59 vs 5.84/7.07/1.20); this shows whether it
     survives the protocol upgrade.
  2. Clique membership diff per (encoder, source, regime): newcsfs pilot
     cliques vs harmonized cliques.

Run inside the x9 container (CPU fine, ~1 min):
  python x8_pool_a/pool_a_harmonized_report.py
Output: nc_csf_predictivity/outputs/pool_a/42_pool_a_harmonized_tables.md
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a",
          CODE_DIR / "nc_csf_predictivity" / "data",
          CODE_DIR / "nc_csf_predictivity" / "ablations",
          CODE_DIR / "nc_csf_predictivity" / "evaluation"):
    sys.path.insert(0, str(p))

from pool_a_analysis import (  # noqa: E402
    FEAT_CSFS, HEAD_CSFS, OUT_ROOT, pool_cliques_for)
from calibration_features_clique import NC_PRIMARY, add_model_id  # noqa: E402
from input_ablation_grid import REGIMES, evaluate, ssl_shortlists  # noqa: E402

SUBMITTED = {"source_nr": (4.96, 3.99, 5.07), "none_nr": (1.59, 1.38, 0.59)}
SUBMITTED_BF = (5.84, 7.07, 1.20)
SRC_LABEL = {"cifar10": "C10", "cifar100": "C100",
             "supercifar100": "SC100", "tinyimagenet": "TI", "all": "all"}
SHORT = {"PCA RecError global": "PCA-RE",
         "KPCA RecError global": "KPCA-RE"}


def main() -> None:
    out_dir = OUT_ROOT / "pool_a"
    long_df = pd.read_parquet(out_dir / "long_pool_a_harmonized.parquet")
    models_df = pd.read_parquet(out_dir / "models_pool_a_harmonized.parquet")
    ref_long = pd.read_parquet(out_dir / "long_pool_a_newcsfs.parquet")

    vgg_long = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    vgg_long = add_model_id(vgg_long)
    cliques = pool_cliques_for(("VGG13",), vgg_long)
    for arch, sub_a in vgg_long.groupby("architecture"):
        for c in NC_PRIMARY:
            vgg_long.loc[sub_a.index, c] = (
                (sub_a[c] - sub_a[c].mean()) / (sub_a[c].std() + 1e-12))
    label_wide = (cliques.assign(label=cliques["in_top_clique"].astype(int))
                  .pivot_table(index=["paradigm", "source", "dropout",
                                      "reward", "regime"],
                               columns="csf", values="label",
                               aggfunc="first").reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns if c not in
                ["paradigm", "source", "dropout", "reward", "regime"]]
    vgg_models = (vgg_long[vgg_long["architecture"] == "VGG13"]
                  [["model_id", "paradigm", "source", "dropout", "reward"]
                   + NC_PRIMARY].drop_duplicates("model_id"))
    tr_marginal = pd.DataFrame(
        [{**m.to_dict(), "regime": r} for _, m in vgg_models.iterrows()
         for r in REGIMES]).merge(
        label_wide, on=["paradigm", "source", "dropout", "reward", "regime"],
        how="inner")

    long_df["model_id"] = (long_df["paradigm"] + "|" + long_df["source"]
                           + "|" + long_df["run"].astype(str))
    # Roster decomposition: the submitted GLiC-2 table used the ORIGINAL 18
    # CSFs (no MahaPP/NCI/KPCA on either side). Adding the new CSFs to the
    # rows changes the oracle and the best-fixed family while the
    # VGG-trained heads carry no labels for them (they can never be
    # shortlisted), so 18-vs-20 isolates that roster asymmetry and
    # 18-vs-submitted isolates the pure protocol effect.
    legacy18 = sorted(set(HEAD_CSFS + FEAT_CSFS))
    rosters = {
        "18 legacy (submitted-comparable)":
            long_df[long_df["csf"].isin(legacy18)],
        "20 (+MahaPP/NCI in rows, no heads for them)":
            long_df[long_df["csf"] != "KPCA RecError global"],
        "21 (revision, no heads for new CSFs yet)": long_df,
    }

    lines = ["# Harmonized SSL selector vs best fixed (addendum to 41)\n\n",
             "**Source:** `x8_pool_a/pool_a_harmonized_report.py`. "
             "Submitted pooled reference: NC+source 4.96/3.99/5.07, "
             "NC only 1.59/1.38/0.59, best fixed 5.84/7.07/1.20 (pilot "
             "protocol, 18-CSF roster). W = beats the harmonized best "
             "fixed in that cell.\n"]
    for roster_name, rows_r in rosters.items():
        fam = sorted(rows_r["csf"].unique())
        ood = rows_r[rows_r["regime"].isin(REGIMES)][
            ["model_id", "eval_dataset", "source", "regime", "csf", "augrc"]]
        results = {}
        for config in SUBMITTED:
            sl = ssl_shortlists(config, models_df, tr_marginal, tr_marginal,
                                csf_cols)
            results[config] = evaluate(ood, sl, always=fam)
        lines.append(f"\n## Roster {roster_name}\n\n"
                     "| Regime | Source | Best fixed CSF | NC+source | "
                     "NC only |\n|---|---|---|---|---|\n")
        wins = {c: 0 for c in SUBMITTED}
        pooled_wins = {c: 0 for c in SUBMITTED}
        for regime in REGIMES:
            for src in ["cifar10", "cifar100", "supercifar100",
                        "tinyimagenet", "all"]:
                key = (src, regime)
                bf = results["source_nr"][key]
                cells = []
                for config in SUBMITTED:
                    v = results[config][key]["predictor"]
                    beat = v < results[config][key]["best_fixed"]
                    if beat:
                        (pooled_wins if src == "all" else wins)[config] += 1
                    cells.append(f"{v:.2f}{' W' if beat else ''}")
                bname = SHORT.get(bf["best_fixed_name"],
                                  bf["best_fixed_name"])
                lines.append(f"| {regime} | {SRC_LABEL[src]} | "
                             f"{bf['best_fixed']:.2f} ({bname}) | "
                             + " | ".join(cells) + " |\n")
        lines.append(f"\nPooled wins vs best fixed: NC+source "
                     f"{pooled_wins['source_nr']}/3, NC only "
                     f"{pooled_wins['none_nr']}/3. Per-source: NC+source "
                     f"{wins['source_nr']}/12, NC only "
                     f"{wins['none_nr']}/12.\n")
        logger.info(f"{roster_name}: pooled wins {pooled_wins}, "
                    f"per-source {wins}")

    old = pd.read_parquet(out_dir / "cliques_pool_a_newcsfs.parquet")
    new = pd.read_parquet(out_dir / "cliques_pool_a_harmonized.parquet")

    def tops(df):
        t = df[df["in_top_clique"]
               & df["regime"].isin(["near", "mid", "far"])]
        return t.groupby(["paradigm", "source", "regime"])["csf"].apply(
            lambda s: frozenset(s))

    a, b = tops(old), tops(new)
    lines.append("\n## Clique membership changes (newcsfs pilot -> "
                 "harmonized)\n\n```\n")
    for k in sorted(set(a.index) | set(b.index),
                    key=lambda k: (k[0], k[1],
                                   ["near", "mid", "far"].index(k[2]))):
        oa, ob = a.get(k, frozenset()), b.get(k, frozenset())
        if oa == ob:
            continue
        enc = "CLIP" if "clip" in k[0] else "DINOv2"
        lines.append(f"{enc:6s} {k[1]:13s} {k[2]:4s}: "
                     f"{sorted(oa)} -> {sorted(ob)}\n")
    lines.append("```\n")

    report = out_dir / "42_pool_a_harmonized_tables.md"
    report.write_text("".join(lines))
    logger.info(f"wrote {report}")
    print("".join(lines))


if __name__ == "__main__":
    main()

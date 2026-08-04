"""Expanded GLiC Q2 table: VGG13+ViT -> SSL probes per (regime, source).

Companion to `tj47_expanded_table.py` for the self-supervised pool. The
first GLiC reply reported the pilot pooled over sources; this table
disaggregates by OOD regime and source and shows all five input
configurations together, under the two strengthenings adopted for the
meta-review grid: the widest benchmark training pool (280 VGG-13 + 40
fine-tuned ViT models) and a best-fixed baseline drawn from the FULL
18-CSF roster per cell (no Always restriction; no prior clique knowledge
exists for this pool).

Configs: NC+source+regime | NC+n_cls+regime | NC+regime | NC+source |
NC only (regime-removed configs are the E-B marginal variant). The
NC+n_cls arm is newly fit here (class-count ordinal replacing the source
one-hot); the other four reproduce the meta-review grid and gate on it.

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/ssl_expanded_table.py
  ./.venv/bin/python nc_csf_predictivity/evaluation/ssl_expanded_table.py --log2-n-classes
Output: nc_csf_predictivity/outputs/33_ssl_expanded_table.md
(the --log2-n-classes variant, written with a _log2 suffix, is the scaling
diagnostic: log-then-standardize the class count instead of standardizing
the raw count; it changes only the NC+n_cls column and shows the TI-mid
failure is semantic mistransfer, not a coding artifact.)
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

import pandas as pd
from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[2]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a",
          CODE_DIR / "nc_csf_predictivity" / "data",
          CODE_DIR / "nc_csf_predictivity" / "ablations",
          CODE_DIR / "nc_csf_predictivity" / "evaluation"):
    sys.path.insert(0, str(p))

from pool_a_analysis import OUT_ROOT, pool_cliques_for  # noqa: E402
from calibration_features_clique import (  # noqa: E402
    NC_PRIMARY,
    add_model_id,
    add_n_classes,
)
from input_ablation_grid import (  # noqa: E402
    REGIMES,
    evaluate,
    ssl_shortlists,
)

COLS = [("source", "NC+source+regime"), ("n_classes", "NC+n_cls+regime"),
        ("none", "NC+regime"), ("source_nr", "NC+source"),
        ("none_nr", "NC only")]

# Meta-review grid pooled cells (VGG13+ViT pool, this exact machinery).
EXPECTED = {
    "source": (5.31, 5.08, 0.57),
    "none": (3.11, 3.75, 0.62),
    "source_nr": (4.64, 4.23, 10.25),
    "none_nr": (2.87, 3.01, 2.87),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log2-n-classes", action="store_true",
                    help="log2-transform the class count before the "
                         "pipeline's StandardScaler (scaling diagnostic)")
    args = ap.parse_args()
    vgg_long = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    vgg_long = add_model_id(vgg_long)
    cliques = pool_cliques_for(("VGG13", "ViT"), vgg_long)
    for arch, sub in vgg_long.groupby("architecture"):
        for c in NC_PRIMARY:
            vgg_long.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    label_wide = (cliques.assign(label=cliques["in_top_clique"].astype(int))
                  .pivot_table(index=["paradigm", "source", "dropout",
                                      "reward", "regime"],
                               columns="csf", values="label", aggfunc="first")
                  .reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns if c not in
                ["paradigm", "source", "dropout", "reward", "regime"]]
    pool_models = (vgg_long[vgg_long["architecture"].isin(
                       ["VGG13", "ViT"])]
                   [["model_id", "paradigm", "source", "dropout", "reward"]
                    + NC_PRIMARY].drop_duplicates("model_id"))

    def train_rows(regimes: list[str]) -> pd.DataFrame:
        rows = [{**m.to_dict(), "regime": r}
                for _, m in pool_models.iterrows() for r in regimes]
        return add_n_classes(pd.DataFrame(rows).merge(
            label_wide,
            on=["paradigm", "source", "dropout", "reward", "regime"],
            how="inner"))

    tr_full = train_rows(["near", "mid", "far", "all"])
    tr_marginal = train_rows(["near", "mid", "far"])

    models_df = add_n_classes(
        pd.read_parquet(OUT_ROOT / "pool_a" / "models_pool_a.parquet"))
    if args.log2_n_classes:
        for df in (tr_full, tr_marginal, models_df):
            df["n_classes"] = np.log2(df["n_classes"])
        logger.info("n_classes log2-transformed before standardization")
    long_df = pd.read_parquet(OUT_ROOT / "pool_a" / "long_pool_a.parquet")
    long_df["model_id"] = (long_df["paradigm"] + "|" + long_df["source"]
                           + "|" + long_df["run"].astype(str))
    ssl_rows = long_df[["model_id", "eval_dataset", "source", "regime",
                        "csf", "augrc"]]
    family = sorted(ssl_rows["csf"].unique())

    results = {}
    for config, _ in COLS:
        sl = ssl_shortlists(config, models_df, tr_full, tr_marginal,
                            csf_cols)
        results[config] = evaluate(ssl_rows, sl, always=family)
        logger.info(f"{config}: " + " ".join(
            f"{r}={results[config][('all', r)]['predictor']}"
            for r in REGIMES))

    bad = []
    for config, exp in EXPECTED.items():
        got = tuple(results[config][("all", r)]["predictor"]
                    for r in REGIMES)
        if any(abs(g - e) > 0.02 for g, e in zip(got, exp)):
            bad.append((config, exp, got))
    if bad:
        for b in bad:
            logger.error(f"replication mismatch: {b}")
        raise SystemExit("Replication gate FAILED; table not written.")
    logger.info(f"Replication gate PASSED ({len(EXPECTED)} pooled configs)")

    src_label = {"cifar10": "C10", "cifar100": "C100",
                 "supercifar100": "SC100", "tinyimagenet": "TI",
                 "all": "all"}
    short = {"PCA RecError global": "PCA-RE",
             "KPCA RecError global": "KPCA-RE"}
    wins = {"beat": 0, "total": 0}
    lines = [
        "# Expanded GLiC Q2 table: VGG13+ViT -> SSL per (regime, source)\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/ssl_expanded_table"
        ".py`. Joint-side mean imputed set-regret; widest benchmark "
        "training pool (VGG-13 + ViT); best fixed CSF = strongest single "
        "detector per cell from the FULL 18-CSF roster; the four "
        "previously reported configs gate on the meta-review grid, "
        "NC+n_cls is newly fit.\n\n",
        "| Regime | Source | Best fixed CSF | "
        + " | ".join(lbl for _, lbl in COLS)
        + " |\n|" + "---|" * (3 + len(COLS)) + "\n"]
    for regime in REGIMES:
        for src in ["cifar10", "cifar100", "supercifar100", "tinyimagenet",
                    "all"]:
            key = (src, regime)
            bf = results["source"][key]
            cells = []
            for config, _ in COLS:
                v = results[config][key]["predictor"]
                cells.append(f"{v:.2f}")
                if src != "all":
                    wins["total"] += 1
                    wins["beat"] += v < results[config][key]["best_fixed"]
            bname = short.get(bf["best_fixed_name"], bf["best_fixed_name"])
            lines.append(f"| {regime} | {src_label[src]} | "
                         f"{bf['best_fixed']:.2f} ({bname}) | "
                         + " | ".join(cells) + " |\n")
    lines.append(f"\nPer-source cells where the predictor beats the best "
                 f"fixed CSF: {wins['beat']} of {wins['total']}.\n")
    sfx = "_log2" if args.log2_n_classes else ""
    out = OUT_ROOT / f"33_ssl_expanded_table{sfx}.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}; per-source beats: "
                f"{wins['beat']}/{wins['total']}")
    print("".join(lines))


if __name__ == "__main__":
    main()

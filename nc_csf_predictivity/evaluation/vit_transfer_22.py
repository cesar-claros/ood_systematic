"""CNN -> ViT transfer of the headline NC-only predictor (camera-ready item 10.2).

The rebuttal's CNN -> ViT number (`vit_cnn_transfer.py`, lopo_modelvit fold)
used the with-regime 20-CSF predictor. This rerun uses the paper's headline
predictor: the regime-free NC-only heads trained on the 280 VGG13 models with
the 22-CSF labels (the heads of `x9_imagenet/phase3_transfer.py`, gate GH),
applied without re-training to the 40 fine-tuned ViT models. NC metrics are
z-scored within the ViT pool (the per-architecture convention). Evaluation
per (model, OOD set) row over the 21-CSF ViT roster (22 without the
Confidence readout), with the scoring and tests of the ImageNet transfer:
hindsight best fixed CSF per cell, Always-CTM, the size-matched random set,
one-sided Wilcoxon and model-clustered sign-flip mean tests, Holm over the
four main cells (pooled, near, mid, far); per-source cells are exploratory.

Gates (abort before any ViT number is computed):
  GH  the head fit reproduces the stored 22-CSF ResNet18 predictions.
  GV  the base arm of the pilot aggregate reproduces the harmonized ViT AUGRC
      of the 19 shared CSFs on every row, and every model has 21 CSFs on its
      8 OOD sets.

Inputs: outputs/track1/dataset/long_harmonized.parquet, code/new_csfs_pilot_summary_long.csv.
Outputs: outputs/vit_transfer_22/{per_row,cells}.csv, outputs/52_vit_transfer_22.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parents[1]
for path in (EVAL_DIR.parent / "ablations", EVAL_DIR, CODE_DIR / "x9_imagenet"):
    sys.path.insert(0, str(path))
from calibration_features_clique import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    NC_PRIMARY,
    add_model_id,
    standardize_per_architecture,
)
from calibration_regime_free import feature_columns_nr  # noqa: E402
from newcsf_heads import AGGREGATE, MODE_TO_EVAL, NEW_CSFS  # noqa: E402
from phase3_transfer import CONFIG, cell_table, fitted_heads, holm_main, per_row  # noqa: E402
from regimefree_wilcoxon import N_FLIPS, SEED, load_parquet  # noqa: E402

VIT_EXPERIMENT = (r"^vit/(?P<source>[\w-]+?)_modelvit_bbvit_lr[\d.e-]+_bs\d+"
                  r"_run(?P<run>\d+)_do(?P<do>[01])_rew(?P<rew>[\d.]+)$")
VIT_SOURCES = {"super_cifar100": "supercifar100", "tiny-imagenet-200": "tinyimagenet"}
REGIMES = ["near", "mid", "far"]
TOL = 1e-6


def load_vit(long_df: pd.DataFrame) -> pd.DataFrame:
    """OOD AUGRC rows of the 40 ViT models over the 21-CSF roster.

    Raises:
        RuntimeError: If the aggregate does not reproduce the harmonized rows
            or a model misses a CSF (gate GV).
    """
    harmonized = long_df[(long_df["architecture"] == "ViT") & (long_df["regime"] != "test")]
    agg = pd.read_csv(AGGREGATE, usecols=["experiment", "arm", "csf", "mode", "AUGRC"])
    agg = agg[(agg["arm"] == "base") & agg["mode"].isin(MODE_TO_EVAL)].copy()
    agg["csf"] = agg["csf"].str.replace("_RecError_global", " RecError global")
    parts = agg["experiment"].str.extract(VIT_EXPERIMENT)
    agg, parts = agg[parts["run"].notna()], parts[parts["run"].notna()]
    agg["model_id"] = ("ViT|modelvit|" + parts["source"].replace(VIT_SOURCES) + "|"
                       + parts["run"] + "|" + parts["do"] + "|"
                       + parts["rew"].astype(float).map(lambda r: f"{r:g}"))
    agg["eval_dataset"] = agg["mode"].map(MODE_TO_EVAL)
    both = harmonized.merge(agg, on=["model_id", "eval_dataset", "csf"], how="left")
    dev = float((both["augrc"] - both["AUGRC"]).abs().max())
    new = agg[agg["csf"].isin(NEW_CSFS)
              & agg.set_index(["model_id", "eval_dataset"]).index.isin(
                  harmonized.set_index(["model_id", "eval_dataset"]).index)]
    rows = pd.concat([harmonized[["model_id", "source", "eval_dataset", "csf", "augrc"]],
                      new.merge(harmonized[["model_id", "source", "eval_dataset"]]
                                .drop_duplicates(), on=["model_id", "eval_dataset"])
                      [["model_id", "source", "eval_dataset", "csf", "AUGRC"]]
                      .rename(columns={"AUGRC": "augrc"})])
    counts = rows.groupby("model_id")["csf"].nunique()
    per_model = rows.groupby("model_id").size()
    if (both["AUGRC"].isna().any() or dev > TOL or len(counts) != 40
            or (counts != 21).any() or (per_model != 21 * 8).any()):
        raise RuntimeError(f"GV failed: max |dAUGRC| {dev:.3g}, "
                           f"{int(both['AUGRC'].isna().sum())} missing, "
                           f"{len(counts)} models, CSFs {counts.unique()}")
    print(f"GV: {len(both):,} ViT rows reproduced (max |dAUGRC| {dev:.1e}); "
          "40 models x 21 CSFs x 8 OOD sets")
    return rows.rename(columns={"source": "family", "augrc": "raw_augrc"})


def vit_probabilities(heads: dict, long_df: pd.DataFrame) -> pd.DataFrame:
    """Predicted probabilities per (ViT model, CSF), NC z-scored within the ViT pool."""
    models = (standardize_per_architecture(long_df, NC_PRIMARY)
              .query("architecture == 'ViT'").drop_duplicates("model_id"))
    x = models[feature_columns_nr(CONFIG)]
    return pd.concat([pd.DataFrame({"model_id": models["model_id"].values, "csf": csf,
                                    "p": pipe.predict_proba(x)[:, 1]})
                      for csf, pipe in heads.items()], ignore_index=True)


def main() -> None:
    """Gates, ViT shortlists, per-cell evaluation, and the report."""
    out_root = Path(DEFAULT_OUT_ROOT)
    heads = fitted_heads(out_root)
    long_df = add_model_id(load_parquet(out_root / "track1" / "dataset"
                                        / "long_harmonized.parquet"))
    rows = load_vit(long_df)
    roster = sorted(rows["csf"].unique())
    probs = vit_probabilities(heads, long_df)

    pr = per_row(rows, probs, roster)
    regime = (long_df.loc[long_df["architecture"] == "ViT",
                          ["source", "eval_dataset", "regime"]].drop_duplicates()
              .rename(columns={"source": "family"}))
    pr = pr.merge(regime, on=["family", "eval_dataset"], how="left")
    cells = {"pooled": pd.Series(True, index=pr.index),
             **{reg: pr["regime"] == reg for reg in REGIMES}}
    n_main = len(cells)
    for source in sorted(pr["family"].unique()):
        for reg in REGIMES:
            cells[f"{source} {reg}"] = (pr["family"] == source) & (pr["regime"] == reg)
    flips = np.random.default_rng(SEED).choice(np.array([-1.0, 1.0]),
                                               size=(N_FLIPS, pr["model_id"].nunique()))
    table = holm_main(cell_table(pr, cells, flips), n_main)
    hits = pd.DataFrame({
        "cell": list(cells),
        "contains_oracle": [(pr.loc[m, "regret"] == 0).mean() for m in cells.values()],
        "random_expected": [(pr.loc[m, "set_size"] / len(roster)).mean()
                            for m in cells.values()]})
    table = table.merge(hits, on="cell")

    csv_dir = out_root / "vit_transfer_22"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pr.to_csv(csv_dir / "per_row.csv", index=False)
    table.to_csv(csv_dir / "cells.csv", index=False)
    show = ["cell", "rows", "nc_only", "set_size", "best_fixed", "best_fixed_regret",
            "always_ctm", "random_set", "contains_oracle", "random_expected"]
    main_p = [c for c in table.columns if c.startswith("p_") and c.endswith("_holm")]
    raw_p = ["p_wil_best", "p_mean_best", "p_wil_random", "p_mean_random"]
    lines = ["# CNN -> ViT transfer of the NC-only predictor (22-CSF heads)\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/vit_transfer_22.py`. VGG13-"
             "trained regime-free NC-only heads applied without re-training to the 40 "
             "fine-tuned ViT models; NC z-scored within the ViT pool; mean imputed "
             "set-regret (AUGRC x1000) over the 21-CSF ViT roster; best fixed CSF chosen "
             "with hindsight per cell; tests one-sided (NC only better), Holm over the "
             "four main cells; per-source cells exploratory (raw p).\n\n",
             "Gates GH and GV passed.\n\n",
             "## Main cells\n\n```\n",
             table.iloc[:n_main][show + main_p].round(4).to_string(index=False),
             "\n```\n\n## Per source (exploratory, raw p)\n\n```\n",
             table.iloc[n_main:][show + raw_p].round(4).to_string(index=False), "\n```\n"]
    (out_root / "52_vit_transfer_22.md").write_text("".join(lines))
    print(f"wrote {out_root / '52_vit_transfer_22.md'}")


if __name__ == "__main__":
    main()

"""LORO-ViT input grid: VGG + leave-one-ViT-run-out training pools.

Addresses the fairness concern that the CNN-derived Always-of-6 baseline
family and a zero-ViT training pool jointly understate what a practitioner
could do on the ViT target. Protocol:

  - 5 folds over the ViT runs. Fold f trains the per-CSF heads on all 280
    VGG-13 models plus the 32 ViT models with run != f, and predicts the 8
    held-out ViT models (run == f). Every ViT model is therefore predicted
    exactly once, out of fold.
  - ViT clique labels for training are recomputed PER FOLD from the four
    in-fold runs only (Friedman-Conover on run != f blocks), so the
    held-out run never contributes to its own training labels. VGG labels
    are the published step-5 cliques (all VGG models are in-pool).
  - All four input configs are fit (NC+source+regime, NC+source marginal,
    NC+regime, NC only marginal), mirroring `input_ablation_grid.py`.
  - The best-fixed-CSF baseline family is the union of the ViT panels' own
    top-clique members across near/mid/far (12 CSFs), i.e. the fixed
    detectors a ViT-aware practitioner would actually shortlist.
  - NC values are z-scored per architecture over each full pool, matching
    the protocol used everywhere else in the paper and responses.

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/loro_vit_grid.py
Output: nc_csf_predictivity/outputs/31_loro_vit_grid.md
"""
from __future__ import annotations

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

from pool_a_analysis import OUT_ROOT  # noqa: E402
from cliques_track1 import compute_track1_cliques  # noqa: E402
from calibration_features_clique import (  # noqa: E402
    NC_PRIMARY,
    add_model_id,
    build_pipeline,
    feature_columns_for_config,
)
from calibration_regime_free import (  # noqa: E402
    build_pipeline_nr,
    feature_columns_nr,
)
from input_ablation_grid import (  # noqa: E402
    CONFIGS,
    REGIMES,
    benchmark_rows,
    evaluate,
)

# Union of the ViT panels' top-clique members across near/mid/far.
VIT_CLIQUE_FAMILY = ["pNML", "PCA RecError global", "MLS", "GE", "PE",
                     "NeCo", "ViM", "Residual", "NNGuide", "Maha",
                     "GradNorm", "KPCA RecError global"]


def main() -> None:
    lh = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    lh = add_model_id(lh)
    vit_raw = lh[(lh["architecture"] == "ViT")
                 & (lh["paradigm"] == "modelvit")].copy()

    lh_std = lh.copy()
    for arch, sub in lh_std.groupby("architecture"):
        for c in NC_PRIMARY:
            lh_std.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))

    vgg_cliques = pd.read_parquet(
        OUT_ROOT / "track1" / "cliques" / "cliques.parquet")
    key_cols = ["paradigm", "source", "dropout", "reward", "regime"]

    def label_pivot(cliques: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        wide = (cliques.assign(label=cliques["in_top_clique"].astype(int))
                .pivot_table(index=key_cols, columns="csf", values="label",
                             aggfunc="first").reset_index().fillna(0))
        cols = [c for c in wide.columns if c not in key_cols]
        wide[cols] = wide[cols].astype(int)
        return wide, cols

    models = (lh_std[["model_id", "architecture", "paradigm", "source",
                      "run", "dropout", "reward"] + NC_PRIMARY]
              .drop_duplicates("model_id"))
    vgg_models = models[models["architecture"] == "VGG13"]
    vit_models = models[models["architecture"] == "ViT"]

    shortlists: dict[str, list[pd.DataFrame]] = {c: [] for c in CONFIGS}
    for fold in range(5):
        fold_cliques, _ = compute_track1_cliques(
            vit_raw[vit_raw["run"] != fold])
        label_wide, csf_cols = label_pivot(
            pd.concat([vgg_cliques, fold_cliques], ignore_index=True))
        n_vit_cells = fold_cliques[
            fold_cliques["regime"].isin(REGIMES)].groupby(key_cols[:4]).ngroups
        logger.info(f"fold {fold}: ViT cliques from runs != {fold} "
                    f"({n_vit_cells} cells)")

        train_models = pd.concat(
            [vgg_models, vit_models[vit_models["run"] != fold]],
            ignore_index=True)
        test_models = vit_models[vit_models["run"] == fold].copy()

        def train_rows(regimes: list[str]) -> pd.DataFrame:
            rows = [{**m.to_dict(), "regime": r}
                    for _, m in train_models.iterrows() for r in regimes]
            return pd.DataFrame(rows).merge(label_wide, on=key_cols,
                                            how="inner")

        tr_full = train_rows(["near", "mid", "far", "all"])
        tr_marginal = train_rows(["near", "mid", "far"])

        for config in CONFIGS:
            if config.endswith("_nr"):
                tr, feats = tr_marginal, feature_columns_nr(config)
                te = test_models[["model_id"] + feats].copy()
                make_pipe = lambda: build_pipeline_nr(config)  # noqa: E731
                per_regime = False
            else:
                tr, feats = tr_full, feature_columns_for_config(config)
                te = pd.concat([test_models.assign(regime=r)
                                for r in REGIMES], ignore_index=True)
                te = te[["model_id", "regime"]
                        + [c for c in feats if c != "regime"]].assign(
                    regime=lambda d: d["regime"])
                te = pd.concat([test_models.assign(regime=r)
                                for r in REGIMES], ignore_index=True)[
                    ["model_id"] + feats]
                make_pipe = lambda: build_pipeline(config)  # noqa: E731
                per_regime = True
            pieces = []
            for name in csf_cols:
                y = tr[name].astype(int).values
                if y.min() == y.max() or min(np.bincount(y)) < 5:
                    continue
                pipe = make_pipe()
                pipe.fit(tr[feats], y)
                hit = pipe.predict_proba(te[feats])[:, 1] >= 0.5
                cols = ["model_id", "regime"] if per_regime else ["model_id"]
                pr = te.loc[hit, cols].copy()
                pr["csf"] = name
                pieces.append(pr)
            sl = pd.concat(pieces, ignore_index=True)
            if not per_regime:
                base = sl[["model_id", "csf"]].drop_duplicates()
                sl = pd.concat([base.assign(regime=r) for r in REGIMES],
                               ignore_index=True)
            shortlists[config].append(sl[["model_id", "regime", "csf"]])
        logger.info(f"fold {fold}: 4 configs fit on "
                    f"{len(train_models)} models, predicted "
                    f"{len(test_models)}")

    rows = benchmark_rows("ViT")
    results = {}
    for config in CONFIGS:
        sl = pd.concat(shortlists[config], ignore_index=True)
        n_pred = sl["model_id"].nunique()
        assert n_pred == 40, f"{config}: {n_pred} of 40 models predicted"
        results[config] = evaluate(rows, sl, always=VIT_CLIQUE_FAMILY)
        logger.info(f"LORO/{config}: " + " ".join(
            f"{r}={results[config][('all', r)]['predictor']}"
            for r in REGIMES))

    src_label = {"cifar10": "C10", "cifar100": "C100",
                 "supercifar100": "SC100", "tinyimagenet": "TI",
                 "all": "all"}
    short = {"PCA RecError global": "PCA-RE",
             "KPCA RecError global": "KPCA-RE"}
    col_order = [("source", "NC+source+regime"), ("source_nr", "NC+source"),
                 ("none", "NC+regime"), ("none_nr", "NC only")]
    wins = {"beat": 0, "total": 0}
    lines = [
        "# LORO-ViT input grid (VGG + leave-one-ViT-run-out)\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/loro_vit_grid.py`. "
        "Heads trained on the 280 VGG-13 models plus 4 of 5 ViT runs; "
        "every ViT model evaluated out of fold; per-fold ViT clique labels "
        "from the in-fold runs only. Best fixed CSF = strongest member of "
        "the ViT panels' own top-clique union (12 CSFs: pNML, PCA-RE, MLS, "
        "GE, PE, NeCo, ViM, Residual, NNGuide, Maha, GradNorm, KPCA-RE), "
        "recomputed per cell on the same rows. Joint-side mean imputed "
        "set-regret; oracle over the full 19-CSF ViT roster.\n\n",
        "| Regime | Source | Best fixed CSF | "
        + " | ".join(lbl for _, lbl in col_order)
        + " |\n|" + "---|" * (3 + len(col_order)) + "\n"]
    for regime in REGIMES:
        for src in ["cifar10", "cifar100", "supercifar100", "tinyimagenet",
                    "all"]:
            key = (src, regime)
            bf = results["source"][key]
            cells = []
            for config, _ in col_order:
                v = results[config][key]["predictor"]
                cell = f"{v:.2f}"
                if (config == "none_nr"
                        and v < results[config][key]["best_fixed"]):
                    cell = f"**{cell}**"  # NC-only worst case wins
                cells.append(cell)
                if src != "all":
                    wins["total"] += 1
                    wins["beat"] += v < results[config][key]["best_fixed"]
            bname = short.get(bf["best_fixed_name"], bf["best_fixed_name"])
            lines.append(f"| {regime} | {src_label[src]} | "
                         f"{bf['best_fixed']:.2f} ({bname}) | "
                         + " | ".join(cells) + " |\n")
    lines.append(f"\nPer-source cells where the predictor beats the best "
                 f"fixed CSF: {wins['beat']} of {wins['total']}. Reference "
                 f"(zero ViT models in training, Always-of-6 baseline): "
                 f"the lopo_modelvit column of 30_input_ablation_grid.md.\n")
    out = OUT_ROOT / "31_loro_vit_grid.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}; per-source beats: "
                f"{wins['beat']}/{wins['total']}")
    print("".join(lines))


if __name__ == "__main__":
    main()

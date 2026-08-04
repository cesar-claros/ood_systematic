"""Log-coded NC+n_cls counterparts for the ResNet-18 and ViT targets.

Completes the class-count coding diagnostic across all three transfers:
the SSL variant lives in `ssl_expanded_table.py --log2-n-classes`; this
script refits the NC+n_cls configuration (paper protocol: per-CSF
LogisticRegressionCV via build_pipeline('n_classes'), regimes
near/mid/far/all, published 5-run clique labels, 280 VGG-13 training
models) under two codings of the class count, raw (the paper's, gated
against the stored-prediction evaluations in the tj47/R4De follow-up
tables) and log2-before-standardization, and applies each head set to
both benchmark targets.

Gates (raw refit vs stored-preds pooled numbers):
  ResNet18 n_cls: 1.24/1.24/0.36   ViT n_cls: 2.03/5.92/1.63

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/nclasses_log_check.py
Output: nc_csf_predictivity/outputs/36_nclasses_log_check.md
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

from pool_a_analysis import OUT_ROOT, pool_cliques_for  # noqa: E402
from calibration_features_clique import (  # noqa: E402
    NC_PRIMARY,
    add_model_id,
    add_n_classes,
    build_pipeline,
    feature_columns_for_config,
)
from input_ablation_grid import REGIMES, evaluate  # noqa: E402

FEATS = feature_columns_for_config("n_classes")  # NC + n_classes + regime
EXPECTED_RAW = {"ResNet18": (1.24, 1.24, 0.36), "ViT": (2.03, 5.92, 1.63)}


def main() -> None:
    lh = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    lh = add_model_id(lh)
    cliques = pool_cliques_for(("VGG13",), lh)
    lh_std = lh.copy()
    for arch, sub in lh_std.groupby("architecture"):
        for c in NC_PRIMARY:
            lh_std.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    label_wide = (cliques.assign(label=cliques["in_top_clique"].astype(int))
                  .pivot_table(index=["paradigm", "source", "dropout",
                                      "reward", "regime"],
                               columns="csf", values="label", aggfunc="first")
                  .reset_index().fillna(0))
    csf_cols = [c for c in label_wide.columns if c not in
                ["paradigm", "source", "dropout", "reward", "regime"]]

    models = (lh_std[["model_id", "architecture", "paradigm", "source",
                      "run", "dropout", "reward"] + NC_PRIMARY]
              .drop_duplicates("model_id"))
    vgg = models[models["architecture"] == "VGG13"]
    tr_base = add_n_classes(pd.DataFrame(
        [{**m.to_dict(), "regime": r} for _, m in vgg.iterrows()
         for r in ["near", "mid", "far", "all"]]).merge(
        label_wide, on=["paradigm", "source", "dropout", "reward",
                        "regime"], how="inner"))

    targets = {}
    for arch in ["ResNet18", "ViT"]:
        te = add_n_classes(pd.DataFrame(
            [{**m.to_dict(), "regime": r}
             for _, m in models[models["architecture"] == arch].iterrows()
             for r in REGIMES]))
        rows = lh[lh["architecture"] == arch][
            ["model_id", "eval_dataset", "source", "regime", "csf",
             "augrc"]]
        targets[arch] = (te, rows)

    results: dict[tuple[str, str], dict] = {}
    for variant in ["raw", "log2"]:
        tr = tr_base.copy()
        tes = {a: te.copy() for a, (te, _) in targets.items()}
        if variant == "log2":
            tr["n_classes"] = np.log2(tr["n_classes"])
            for te in tes.values():
                te["n_classes"] = np.log2(te["n_classes"])
        heads = []
        for name in csf_cols:
            y = tr[name].astype(int).values
            if y.min() == y.max() or min(np.bincount(y)) < 5:
                continue
            pipe = build_pipeline("n_classes")
            pipe.fit(tr[FEATS], y)
            heads.append((name, pipe))
        for arch, (_, rows) in targets.items():
            te = tes[arch]
            pieces = []
            for name, pipe in heads:
                hit = pipe.predict_proba(te[FEATS])[:, 1] >= 0.5
                pr = te.loc[hit, ["model_id", "regime"]].copy()
                pr["csf"] = name
                pieces.append(pr)
            sl = pd.concat(pieces, ignore_index=True)
            results[(arch, variant)] = evaluate(rows, sl)
            logger.info(f"{arch}/{variant}: " + " ".join(
                f"{r}={results[(arch, variant)][('all', r)]['predictor']}"
                for r in REGIMES))

    bad = []
    for arch, exp in EXPECTED_RAW.items():
        got = tuple(results[(arch, "raw")][("all", r)]["predictor"]
                    for r in REGIMES)
        if any(abs(g - e) > 0.05 for g, e in zip(got, exp)):
            bad.append((arch, exp, got))
    if bad:
        for b in bad:
            logger.error(f"raw-refit gate mismatch: {b}")
        raise SystemExit("Gates FAILED; report not written.")
    logger.info("Raw-refit gates PASSED (both targets)")

    src_label = {"cifar10": "C10", "cifar100": "C100",
                 "supercifar100": "SC100", "tinyimagenet": "TI",
                 "all": "all"}
    lines = [
        "# NC+n_cls coding check: raw vs log2 class count, benchmark "
        "targets\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/nclasses_log_check"
        ".py`. Paper-protocol NC+n_cls heads refit under raw and "
        "log2-before-standardization codings of the class count; raw "
        "refits gate against the stored-prediction evaluations of the "
        "tj47/R4De follow-up tables. SSL counterpart: "
        "`33_ssl_expanded_table_log2.md`.\n\n"]
    for arch in ["ResNet18", "ViT"]:
        lines.append(f"\n## VGG-13 -> {arch}\n\n"
                     "| Regime | Source | Best fixed CSF | n_cls raw | "
                     "n_cls log2 |\n|---|---|---|---|---|\n")
        for regime in REGIMES:
            for src in ["cifar10", "cifar100", "supercifar100",
                        "tinyimagenet", "all"]:
                key = (src, regime)
                bf = results[(arch, "raw")][key]
                cells = " | ".join(
                    f"{results[(arch, v)][key]['predictor']:.2f}"
                    for v in ["raw", "log2"])
                lines.append(f"| {regime} | {src_label[src]} | "
                             f"{bf['best_fixed']:.2f} "
                             f"({bf['best_fixed_name']}) | {cells} |\n")
    out = OUT_ROOT / "36_nclasses_log_check.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}")
    print("".join(lines))


if __name__ == "__main__":
    main()

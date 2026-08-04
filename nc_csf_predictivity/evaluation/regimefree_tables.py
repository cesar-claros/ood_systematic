"""Regime-free response tables: NC+source and NC-only, three transfers.

Generates the tables used in the reviewer follow-ups and the meta-review
response after the decision to show ONLY configurations that use no
regime input: NC + source (one shortlist per model, applied to every
regime) and NC alone. Heads are trained on the 280 VGG-13 models in all
three tables (the paper's selector; no target model is ever seen in
training), under the E-B marginal protocol. The best-fixed baseline is
chosen per target to favor the baseline:

  ResNet-18 : the paper's six Always candidates (CNN clique members)
  ViT       : six members drawn from the ViT panels' top cliques
              (KPCA/PCA RecError, ViM, Residual, GradNorm, PE; shortened
              from the 12-member union on 2026-08-02)
  SSL       : the full 18-CSF roster of the probe pool

ResNet and ViT predictor columns come from the stored regime-free
marginal predictions; the SSL columns are refit (VGG-only pool). Gates:
the pooled rows must reproduce E-B (ResNet 1.06/0.96/0.64 and
1.23/1.16/0.80), the R4De follow-up table (ViT 1.76/2.34/2.30 and
1.35/2.24/3.94), and the Pool A regime-free addendum (SSL 4.96/3.99/5.07
and 1.59/1.38/0.59). Bold marks cells where NC only beats the best fixed
detector.

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/regimefree_tables.py
Output: nc_csf_predictivity/outputs/37_regimefree_tables.md
"""
from __future__ import annotations

import pathlib
import sys

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
)
from input_ablation_grid import (  # noqa: E402
    REGIMES,
    benchmark_rows,
    evaluate,
    ssl_shortlists,
    stored_shortlists,
)

# Shortened ViT baseline family (Cesar, 2026-08-02): drawn from the ViT
# panels' top cliques; per-source winners are unchanged relative to the
# 12-member union, only the pooled-near best-fixed can differ.
VIT_FAMILY = ["KPCA RecError global", "PCA RecError global", "ViM",
              "Residual", "GradNorm", "PE"]

CONFIGS = [("source_nr", "NC+source"), ("none_nr", "NC only")]
EXPECTED = {
    ("ResNet18", "source_nr"): (1.06, 0.96, 0.64),
    ("ResNet18", "none_nr"): (1.23, 1.16, 0.80),
    ("ViT", "source_nr"): (1.76, 2.34, 2.30),
    ("ViT", "none_nr"): (1.35, 2.24, 3.94),
    ("SSL", "source_nr"): (4.96, 3.99, 5.07),
    ("SSL", "none_nr"): (1.59, 1.38, 0.59),
}


def main() -> None:
    results: dict[tuple[str, str], dict] = {}

    for arch, split, fold, family in [
            ("ResNet18", "xarch", "vgg13_to_resnet18", None),
            ("ViT", "lopo", "lopo_modelvit", VIT_FAMILY)]:
        rows = benchmark_rows(arch)
        for config, _ in CONFIGS:
            sl = stored_shortlists(split, config, fold)
            results[(arch, config)] = evaluate(rows, sl, always=family)
            logger.info(f"{arch}/{config}: " + " ".join(
                f"{r}={results[(arch, config)][('all', r)]['predictor']}"
                for r in REGIMES))

    # SSL: VGG-only pool, refit marginal heads
    vgg_long = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    vgg_long = add_model_id(vgg_long)
    cliques = pool_cliques_for(("VGG13",), vgg_long)
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
    vgg_models = (vgg_long[vgg_long["architecture"] == "VGG13"]
                  [["model_id", "paradigm", "source", "dropout", "reward"]
                   + NC_PRIMARY].drop_duplicates("model_id"))
    tr_marginal = pd.DataFrame(
        [{**m.to_dict(), "regime": r} for _, m in vgg_models.iterrows()
         for r in REGIMES]).merge(
        label_wide, on=["paradigm", "source", "dropout", "reward",
                        "regime"], how="inner")
    models_df = pd.read_parquet(OUT_ROOT / "pool_a" / "models_pool_a.parquet")
    long_df = pd.read_parquet(OUT_ROOT / "pool_a" / "long_pool_a.parquet")
    long_df["model_id"] = (long_df["paradigm"] + "|" + long_df["source"]
                           + "|" + long_df["run"].astype(str))
    ssl_rows = long_df[["model_id", "eval_dataset", "source", "regime",
                        "csf", "augrc"]]
    ssl_family = sorted(ssl_rows["csf"].unique())
    for config, _ in CONFIGS:
        sl = ssl_shortlists(config, models_df, tr_marginal, tr_marginal,
                            csf_cols)
        results[("SSL", config)] = evaluate(ssl_rows, sl,
                                            always=ssl_family)
        logger.info(f"SSL/{config}: " + " ".join(
            f"{r}={results[('SSL', config)][('all', r)]['predictor']}"
            for r in REGIMES))

    bad = []
    for (tgt, config), exp in EXPECTED.items():
        got = tuple(results[(tgt, config)][("all", r)]["predictor"]
                    for r in REGIMES)
        if any(abs(g - e) > 0.02 for g, e in zip(got, exp)):
            bad.append((tgt, config, exp, got))
    if bad:
        for b in bad:
            logger.error(f"gate mismatch: {b}")
        raise SystemExit("Gates FAILED; tables not written.")
    logger.info(f"Gates PASSED ({len(EXPECTED)} pooled cells)")

    src_label = {"cifar10": "C10", "cifar100": "C100",
                 "supercifar100": "SC100", "tinyimagenet": "TI",
                 "all": "all"}
    short = {"PCA RecError global": "PCA-RE",
             "KPCA RecError global": "KPCA-RE"}
    fam_note = {"ResNet18": "Always-of-6 (CNN clique members)",
                "ViT": "six members drawn from the ViT top cliques",
                "SSL": "full 18-CSF roster"}
    wins = {"src": {"beat": 0, "total": 0},
            "pooled": {"beat": 0, "total": 0}}
    nc_only_wins = {"src": 0, "pooled": 0}
    lines = [
        "# Regime-free response tables: NC+source and NC only\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/regimefree_tables"
        ".py`. Heads trained on the 280 VGG-13 models (marginal protocol, "
        "one shortlist per model applied to every regime); no regime "
        "input anywhere; best-fixed family per target chosen to favor the "
        "baseline. Pooled rows gate on E-B, the R4De follow-up, and the "
        "Pool A regime-free addendum. Bold = NC only beats the best "
        "fixed detector.\n"]
    for tgt in ["ResNet18", "ViT", "SSL"]:
        lines.append(f"\n## VGG-13 -> {tgt} (best fixed: "
                     f"{fam_note[tgt]})\n\n"
                     "| Regime | Source | Best fixed CSF | NC+source | "
                     "NC only |\n|---|---|---|---|---|\n")
        for regime in REGIMES:
            for src in ["cifar10", "cifar100", "supercifar100",
                        "tinyimagenet", "all"]:
                key = (src, regime)
                bf = results[(tgt, "source_nr")][key]
                cells = []
                for config, _ in CONFIGS:
                    v = results[(tgt, config)][key]["predictor"]
                    cell = f"{v:.2f}"
                    beat = v < results[(tgt, config)][key]["best_fixed"]
                    bucket = "pooled" if src == "all" else "src"
                    wins[bucket]["total"] += 1
                    wins[bucket]["beat"] += beat
                    if config == "none_nr" and beat:
                        nc_only_wins[bucket] += 1
                        cell = f"**{cell}**"
                    cells.append(cell)
                bname = short.get(bf["best_fixed_name"],
                                  bf["best_fixed_name"])
                lines.append(f"| {regime} | {src_label[src]} | "
                             f"{bf['best_fixed']:.2f} ({bname}) | "
                             + " | ".join(cells) + " |\n")
    lines.append(f"\nWins vs best fixed: pooled {wins['pooled']['beat']} "
                 f"of {wins['pooled']['total']} (NC only "
                 f"{nc_only_wins['pooled']} of 9); per-source "
                 f"{wins['src']['beat']} of {wins['src']['total']} "
                 f"(NC only {nc_only_wins['src']} of 36).\n")
    out = OUT_ROOT / "37_regimefree_tables.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}")
    print("".join(lines))


if __name__ == "__main__":
    main()

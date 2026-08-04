"""Predictor input-ablation grid: source x regime inputs, three transfers.

Builds the 2x2 input grid (with/without the source descriptor x with/without
the OOD-regime indicator) for the three selector transfers:

  VGG13 -> ResNet18 : stored xarch predictions (configs `source`, `none`)
                      plus the stored regime-free marginal predictions
                      (`source_nr`, `none_nr`)
  VGG13 -> ViT      : the same four configs from the stored lopo_modelvit
                      fold (training pool identical to xarch: the 280
                      VGG-13 models)
  VGG13+ViT -> SSL  : per-CSF heads refit on the widest benchmark pool
                      (280 VGG-13 + 40 fine-tuned ViT models, labels from
                      the published VGG cliques plus cliques_vit) for all
                      four configs and applied to the 40 frozen
                      DINOv2/CLIP linear probes without retraining

Configs: `source` = NC + source one-hot + regime one-hot (the paper
protocol); `none` = NC + regime; `source_nr` = NC + source (regime removed,
E-B marginal variant: one shortlist per model applied to every regime);
`none_nr` = NC only.

Reported: joint-side mean imputed set-regret per OOD regime, with the best
fixed CSF (Always-of-{MSR, Energy, MLS, CTM, fDBD, NNGuide}) alongside.
Replication gates assert every previously reported cell (paper, E-B, E-C,
Pool A pilot and its regime-free addendum) before the new cells are trusted.

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/input_ablation_grid.py
Output: nc_csf_predictivity/outputs/30_input_ablation_grid.md
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
          CODE_DIR / "nc_csf_predictivity" / "ablations"):
    sys.path.insert(0, str(p))

from pool_a_analysis import OUT_ROOT, pool_cliques_for  # noqa: E402
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

ALWAYS = ["MSR", "Energy", "MLS", "CTM", "fDBD", "NNGuide"]
REGIMES = ["near", "mid", "far"]
CONFIGS = ["source", "none", "source_nr", "none_nr"]
CONFIG_LABEL = {"source": "source+regime", "none": "regime only",
                "source_nr": "source only", "none_nr": "NC only"}

# Previously reported cells (joint side), the replication gates.
EXPECTED = {
    ("ResNet18", "source"): (1.02, 1.18, 0.39),    # paper headline
    ("ResNet18", "none"): (1.45, 1.44, 0.40),      # paper "NC alone"
    ("ResNet18", "source_nr"): (1.06, 0.96, 0.64), # E-B marginal
    ("ResNet18", "none_nr"): (1.23, 1.16, 0.80),   # E-B none_nr
    ("ViT", "source"): (1.90, 3.66, 2.07),         # E-C lopo_modelvit
    ("SSL", "source"): (5.31, 5.08, 0.57),         # Pool A pilot VGG13+ViT
    ("SSL", "source_nr"): (4.64, 4.23, 10.25),     # Pool A regime-free
    ("SSL", "none_nr"): (2.87, 3.01, 2.87),        # Pool A regime-free
}


def evaluate(rows_long: pd.DataFrame, shortlists: pd.DataFrame,
             always: list[str] | None = None
             ) -> dict[tuple[str, str], dict[str, float]]:
    """Joint-side mean imputed set-regret per (source, regime) and pooled.

    rows_long: model_id, eval_dataset, source, regime, csf, augrc.
    shortlists: model_id, regime, csf (predicted members only).
    always: candidate family for the best-fixed-CSF baseline (default the
    paper's Always-of-6). Keys of the result: (source, regime) plus
    ('all', regime) pooled rows; the best fixed CSF is recomputed per key
    on the same rows.
    """
    fam = ALWAYS if always is None else list(always)
    ood = rows_long[rows_long["regime"].isin(REGIMES)]
    members = (shortlists.groupby(["model_id", "regime"])["csf"]
               .apply(set).rename("members"))
    recs = []
    for (mid, _, src, regime), g in ood.groupby(
            ["model_id", "eval_dataset", "source", "regime"]):
        vals = g.set_index("csf")["augrc"]
        oracle, worst = float(vals.min()), float(vals.max())
        mem = members.get((mid, regime), set()) & set(vals.index)
        rec = {"source": src, "regime": regime,
               "pred": (float(vals.loc[list(mem)].min()) - oracle)
                       if mem else (worst - oracle)}
        for x in fam:
            rec[x] = (float(vals.loc[x]) - oracle) if x in vals.index \
                else np.nan
        recs.append(rec)
    df = pd.DataFrame(recs)

    out: dict[tuple[str, str], dict[str, float]] = {}

    def agg(sub: pd.DataFrame, key: tuple[str, str]) -> None:
        bl = {x: float(sub[x].mean()) for x in fam
              if sub[x].notna().any()}
        best = min(bl, key=bl.get)
        out[key] = {"predictor": round(float(sub["pred"].mean()), 2),
                    "best_fixed": round(bl[best], 2),
                    "best_fixed_name": best}

    for (src, regime), sub in df.groupby(["source", "regime"]):
        agg(sub, (src, regime))
    for regime, sub in df.groupby("regime"):
        agg(sub, ("all", regime))
    return out


def expand_regimes(sl: pd.DataFrame) -> pd.DataFrame:
    """Regime-free shortlists: one shortlist per model for every regime."""
    if set(sl["regime"].unique()) <= set(REGIMES):
        return sl
    base = sl[["model_id", "csf"]].drop_duplicates()
    return pd.concat([base.assign(regime=r) for r in REGIMES],
                     ignore_index=True)


def stored_shortlists(split: str, config: str, fold: str) -> pd.DataFrame:
    root = OUT_ROOT / "ablations"
    if config.endswith("_nr"):
        path = (root / "calib_cliques_regime_free" / "track1" / split
                / f"{config}_marginal" / "preds.parquet")
    else:
        path = (root / "calib_cliques" / "track1" / split / config
                / "preds.parquet")
    df = pd.read_parquet(path)
    df = df[(df["fold_label"] == fold) & df["predicted_competitive"]]
    df = df[["model_id", "regime", "csf"]].copy()
    per_regime = df[df["regime"].isin(REGIMES)]
    if not per_regime.empty:
        # regime-conditioned preds also carry regime='all' rows; drop them
        return per_regime
    return expand_regimes(df)


def benchmark_rows(arch: str) -> pd.DataFrame:
    lh = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    lh = add_model_id(lh)
    sub = lh[lh["architecture"] == arch]
    return sub[["model_id", "eval_dataset", "source", "regime", "csf",
                "augrc"]].copy()


def ssl_shortlists(config: str, models_df: pd.DataFrame,
                   tr_full: pd.DataFrame, tr_marginal: pd.DataFrame,
                   csf_cols: list[str]) -> pd.DataFrame:
    """Refit per-CSF heads on the VGG-13 pool for one config, predict on
    the probe pool. tr_full has regimes near/mid/far/all; tr_marginal only
    near/mid/far (the regime-free marginal label rows)."""
    pool = models_df.copy()
    for enc, sub in pool.groupby("paradigm"):
        for c in NC_PRIMARY:
            pool.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    pool["model_id"] = (pool["paradigm"] + "|" + pool["source"] + "|"
                        + pool["run"].astype(str))

    if config.endswith("_nr"):
        tr, feats = tr_marginal, feature_columns_nr(config)
        te = pool[["model_id"] + feats].copy() if "source" not in feats else \
            pool[["model_id"] + feats].copy()
        make_pipe = lambda: build_pipeline_nr(config)  # noqa: E731
        te_regimes = [None]
    else:
        tr, feats = tr_full, feature_columns_for_config(config)
        rows = []
        for regime in REGIMES:
            r = pool.copy()
            r["regime"] = regime
            rows.append(r)
        pool = pd.concat(rows, ignore_index=True)
        te = pool[["model_id", "regime"] + [c for c in feats
                                            if c != "regime"]].copy()
        te["regime"] = pool["regime"]
        make_pipe = lambda: build_pipeline(config)  # noqa: E731
        te_regimes = REGIMES

    pieces = []
    for name in csf_cols:
        y = tr[name].astype(int).values
        if y.min() == y.max() or min(np.bincount(y)) < 5:
            continue
        pipe = make_pipe()
        pipe.fit(tr[feats], y)
        hit = pipe.predict_proba(te[feats])[:, 1] >= 0.5
        cols = ["model_id", "regime"] if te_regimes != [None] else ["model_id"]
        pr = te.loc[hit, cols].copy()
        pr["csf"] = name
        pieces.append(pr)
    sl = pd.concat(pieces, ignore_index=True)
    if "regime" not in sl.columns:
        sl["regime"] = "all"
    logger.info(f"SSL[{config}]: {sl['csf'].nunique()} heads fired")
    return expand_regimes(sl[["model_id", "regime", "csf"]])


def main() -> None:
    results: dict[tuple[str, str], dict] = {}

    for arch, fold, split in [("ResNet18", "vgg13_to_resnet18", "xarch"),
                              ("ViT", "lopo_modelvit", "lopo")]:
        rows = benchmark_rows(arch)
        for config in CONFIGS:
            sl = stored_shortlists(split, config, fold)
            results[(arch, config)] = evaluate(rows, sl)
            logger.info(f"{arch}/{config}: " + " ".join(
                f"{r}={results[(arch, config)][('all', r)]['predictor']}"
                for r in REGIMES))

    logger.info("SSL: building shared training labels...")
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
    vgg_models = (vgg_long[vgg_long["architecture"].isin(
                      ["VGG13", "ViT"])]
                  [["model_id", "paradigm", "source", "dropout", "reward"]
                   + NC_PRIMARY].drop_duplicates("model_id"))

    def train_rows(regimes: list[str]) -> pd.DataFrame:
        rows = [{**m.to_dict(), "regime": r}
                for _, m in vgg_models.iterrows() for r in regimes]
        return pd.DataFrame(rows).merge(
            label_wide,
            on=["paradigm", "source", "dropout", "reward", "regime"],
            how="inner")

    tr_full = train_rows(["near", "mid", "far", "all"])
    tr_marginal = train_rows(["near", "mid", "far"])

    models_df = pd.read_parquet(OUT_ROOT / "pool_a" / "models_pool_a.parquet")
    long_df = pd.read_parquet(OUT_ROOT / "pool_a" / "long_pool_a.parquet")
    long_df["model_id"] = (long_df["paradigm"] + "|" + long_df["source"]
                           + "|" + long_df["run"].astype(str))
    ssl_rows = long_df[["model_id", "eval_dataset", "source", "regime",
                        "csf", "augrc"]]
    ssl_family = sorted(ssl_rows["csf"].unique())
    logger.info(f"SSL best-fixed family: full roster "
                f"({len(ssl_family)} CSFs)")
    for config in CONFIGS:
        sl = ssl_shortlists(config, models_df, tr_full, tr_marginal,
                            csf_cols)
        results[("SSL", config)] = evaluate(ssl_rows, sl,
                                            always=ssl_family)
        logger.info(f"SSL/{config}: " + " ".join(
            f"{r}={results[('SSL', config)][('all', r)]['predictor']}"
            for r in REGIMES))

    bad = []
    for (target, config), exp in EXPECTED.items():
        got = tuple(results[(target, config)][("all", r)]["predictor"]
                    for r in REGIMES)
        if any(abs(g - e) > 0.02 for g, e in zip(got, exp)):
            bad.append((target, config, exp, got))
    if bad:
        for b in bad:
            logger.error(f"replication mismatch: {b}")
        raise SystemExit("Replication gate FAILED; grid not written.")
    logger.info(f"Replication gate PASSED ({len(EXPECTED)} reported cells)")

    lines = ["# Predictor input-ablation grid (source x regime inputs)\n",
             "\n**Source:** `nc_csf_predictivity/evaluation/"
             "input_ablation_grid.py`. Joint-side mean imputed set-regret "
             "per (source, regime) and pooled ('all' rows); heads trained "
             "on the 280 VGG-13 models for the ResNet18/ViT targets and on "
             "the widest benchmark pool (VGG-13 + ViT, 320 models) for the "
             "SSL target; every configuration "
             "includes the 8 NC metrics and the columns vary only the "
             "categorical inputs; regime-removed configs use the E-B "
             "marginal variant (one shortlist per model applied to every "
             "regime). Best fixed CSF = strongest single CSF per cell "
             "from the target-appropriate family: Always-of-{MSR, Energy, "
             "MLS, CTM, fDBD, NNGuide} for the CNN-clique targets "
             "(ResNet18, ViT), the FULL roster for SSL (no prior clique "
             "knowledge exists there). Replication gate over the "
             "previously reported pooled predictor cells passed.\n"]
    src_label = {"cifar10": "C10", "cifar100": "C100",
                 "supercifar100": "SC100", "tinyimagenet": "TI",
                 "all": "all"}
    src_order = ["cifar10", "cifar100", "supercifar100", "tinyimagenet",
                 "all"]
    col_order = [("source", "NC+source+regime"), ("source_nr", "NC+source"),
                 ("none", "NC+regime"), ("none_nr", "NC only")]
    wins = {"beat": 0, "total": 0}
    for target in ["ResNet18", "ViT", "SSL"]:
        pool_lbl = "VGG-13+ViT" if target == "SSL" else "VGG-13"
        lines.append(f"\n## {pool_lbl} -> {target}\n\n")
        lines.append("| Regime | Source | Best fixed CSF | "
                     + " | ".join(lbl for _, lbl in col_order)
                     + " |\n|" + "---|" * (3 + len(col_order)) + "\n")
        for regime in REGIMES:
            for src in src_order:
                key = (src, regime)
                bf = results[(target, "source")][key]
                cells = []
                for config, _ in col_order:
                    v = results[(target, config)][key]["predictor"]
                    cell = f"{v:.2f}"
                    if (config == "none_nr"
                            and v < results[(target, config)][key]["best_fixed"]):
                        cell = f"**{cell}**"  # NC-only worst case wins
                    cells.append(cell)
                    if src != "all":
                        wins["total"] += 1
                        wins["beat"] += v < results[(target, config)][key][
                            "best_fixed"]
                short = {"PCA RecError global": "PCA-RE",
                         "KPCA RecError global": "KPCA-RE"}
                bname = short.get(bf["best_fixed_name"],
                                  bf["best_fixed_name"])
                lines.append(f"| {regime} | {src_label[src]} | "
                             f"{bf['best_fixed']:.2f} ({bname}) | "
                             + " | ".join(cells) + " |\n")
    lines.append(f"\nPer-source cells where the predictor beats the best "
                 f"fixed CSF: {wins['beat']} of {wins['total']}.\n")
    out = OUT_ROOT / "30_input_ablation_grid.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}")
    logger.info(f"per-source beat-baseline cells: "
                f"{wins['beat']}/{wins['total']}")
    print("".join(lines))


if __name__ == "__main__":
    main()

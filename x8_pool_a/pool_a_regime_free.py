"""Pool A regime-free ablation: benchmark-trained selector without the regime input.

Extends the E-B regime-free ablation (calibration_regime_free.py) to the frozen
DINOv2/CLIP probe pool. The per-CSF clique heads are trained on the benchmark
pools (VGG13, ViT, VGG13+ViT) under E-B's 'marginal' variant: near/mid/far
label rows with the regime feature removed, so each head learns the
regime-marginal top-clique probability and every probe model receives one
shortlist applied to all three regimes. Heads are then applied to the 40
probes without retraining, exactly as in `pool_a_analysis.h2_predictor`.

Configs (from E-B):
  source_nr : 8 NC metrics + source one-hot (regime removed)
  none_nr   : 8 NC metrics only (source and regime removed)

Replication gate: the with-regime `h2_predictor` is re-run first and must
reproduce the pilot's H2 table (27_pool_a_pilot.md) before the ablation arms
run.

Run from `code/`:
  ./.venv/bin/python x8_pool_a/pool_a_regime_free.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[1]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a",
          CODE_DIR / "nc_csf_predictivity" / "data",
          CODE_DIR / "nc_csf_predictivity" / "ablations"):
    sys.path.insert(0, str(p))

from pool_a_analysis import (  # noqa: E402
    ALWAYS_BASELINES,
    FEAT_CSFS,
    HEAD_CSFS,
    OUT_ROOT,
    h2_predictor,
    pool_cliques_for,
)
from calibration_features_clique import NC_PRIMARY, add_model_id  # noqa: E402
from calibration_regime_free import (  # noqa: E402
    build_pipeline_nr,
    feature_columns_nr,
)

POOL_DIR = OUT_ROOT / "pool_a"
TRAIN_POOLS = [("VGG13",), ("ViT",), ("VGG13", "ViT")]
CONFIGS_NR = ["source_nr", "none_nr"]

# Pilot H2 joint-side rows (27_pool_a_pilot.md), the replication gate.
EXPECTED_PILOT = {
    ("VGG13", "all", "near"): 5.99,
    ("VGG13", "all", "mid"): 3.22,
    ("VGG13", "all", "far"): 1.44,
    ("ViT", "all", "near"): 21.31,
    ("ViT", "all", "mid"): 78.46,
    ("ViT", "all", "far"): 18.43,
    ("VGG13+ViT", "all", "near"): 5.31,
    ("VGG13+ViT", "all", "mid"): 5.08,
    ("VGG13+ViT", "all", "far"): 0.57,
    ("VGG13", "head", "near"): 0.49,
    ("VGG13", "feature", "far"): 1.56,
}


def h2_regime_free(models_df: pd.DataFrame, long_df: pd.DataFrame,
                   train_archs: tuple, config: str) -> pd.DataFrame:
    """`h2_predictor` with the regime input removed (E-B marginal variant)."""
    vgg_long = pd.read_parquet(
        OUT_ROOT / "track1" / "dataset" / "long_harmonized.parquet")
    vgg_long = add_model_id(vgg_long)
    cliques = pool_cliques_for(train_archs, vgg_long)
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
    vgg_models = (vgg_long[vgg_long["architecture"].isin(train_archs)]
                  [["model_id", "paradigm", "source", "dropout", "reward"]
                   + NC_PRIMARY].drop_duplicates("model_id"))
    train_rows = []
    for _, m in vgg_models.iterrows():
        for regime in ["near", "mid", "far"]:  # marginal: no 'all' rows
            train_rows.append({**m.to_dict(), "regime": regime})
    tr = pd.DataFrame(train_rows).merge(
        label_wide, on=["paradigm", "source", "dropout", "reward", "regime"],
        how="inner")

    pool = models_df.copy()
    for enc, sub in pool.groupby("paradigm"):
        for c in NC_PRIMARY:
            pool.loc[sub.index, c] = (
                (sub[c] - sub[c].mean()) / (sub[c].std() + 1e-12))
    te = pool[["paradigm", "source", "run"] + NC_PRIMARY].copy()

    feats = feature_columns_nr(config)
    preds = []
    for name in csf_cols:
        y = tr[name].astype(int).values
        if y.min() == y.max() or min(np.bincount(y)) < 5:
            continue
        pipe = build_pipeline_nr(config)
        pipe.fit(tr[feats], y)
        pr = te[["paradigm", "source", "run"]].copy()
        pr["csf"] = name
        pr["hit"] = pipe.predict_proba(te[feats])[:, 1] >= 0.5
        preds.append(pr)
    preds = pd.concat(preds, ignore_index=True)
    logger.info(f"H2-NR[{'+'.join(train_archs)}|{config}]: "
                f"{preds['csf'].nunique()} per-CSF heads, "
                f"{preds.groupby(['paradigm', 'source', 'run'])['hit'].sum().mean():.1f} "
                f"mean shortlist size")

    ood = long_df[long_df["regime"] != "test"].copy()
    sides = {"all": HEAD_CSFS + FEAT_CSFS, "head": HEAD_CSFS,
             "feature": FEAT_CSFS}
    out = []
    for side, side_csfs in sides.items():
        sub = ood[ood["csf"].isin(side_csfs)]
        oracle = (sub.groupby(["paradigm", "source", "run", "eval_dataset",
                               "regime"])["augrc"]
                  .agg(["min", "max"]).reset_index()
                  .rename(columns={"min": "o", "max": "wst"}))
        chosen = sub.merge(
            preds[preds["hit"]][["paradigm", "source", "run", "csf"]],
            on=["paradigm", "source", "run", "csf"], how="inner")
        smin = (chosen.groupby(["paradigm", "source", "run", "eval_dataset",
                                "regime"])["augrc"].min()
                .rename("set_min").reset_index())
        m = oracle.merge(smin, how="left",
                         on=["paradigm", "source", "run", "eval_dataset",
                             "regime"])
        m["regret"] = (m["set_min"].fillna(m["wst"]) - m["o"]).clip(lower=0)
        for regime, g in m.groupby("regime"):
            rec = {"train_pool": "+".join(train_archs), "config": config,
                   "side": side, "regime": regime,
                   "predictor_regret": round(float(g["regret"].mean()), 2),
                   "empty_pct": round(100 * float(g["set_min"].isna().mean()), 1)}
            best_name, best_val = None, np.inf
            for x in ALWAYS_BASELINES:
                if x not in side_csfs:
                    continue
                bx = sub[(sub["csf"] == x) & (sub["regime"] == regime)]
                bxm = bx.merge(oracle, on=["paradigm", "source", "run",
                                           "eval_dataset", "regime"])
                val = float((bxm["augrc"] - bxm["o"]).clip(lower=0).mean())
                if val < best_val:
                    best_name, best_val = x, val
            rec["best_baseline"] = best_name
            rec["baseline_regret"] = round(best_val, 2)
            out.append(rec)
    return pd.DataFrame(out)


def main() -> None:
    models_df = pd.read_parquet(POOL_DIR / "models_pool_a.parquet")
    long_df = pd.read_parquet(POOL_DIR / "long_pool_a.parquet")
    logger.info(f"Loaded probe pool: {len(models_df)} models, "
                f"{len(long_df)} long rows")

    logger.info("=== Replication gate: with-regime h2_predictor ===")
    rep_frames = []
    for archs in TRAIN_POOLS:
        rep_frames.append(h2_predictor(models_df, long_df, 0, archs))
    rep = pd.concat(rep_frames, ignore_index=True)
    bad = []
    for (poolname, side, regime), exp in EXPECTED_PILOT.items():
        got = rep[(rep["train_pool"] == poolname) & (rep["side"] == side)
                  & (rep["regime"] == regime)]["predictor_regret"]
        if len(got) != 1 or abs(float(got.iloc[0]) - exp) > 0.005:
            bad.append((poolname, side, regime, exp,
                        float(got.iloc[0]) if len(got) else None))
    if bad:
        for b in bad:
            logger.error(f"replication mismatch: {b}")
        raise SystemExit("Replication gate FAILED; aborting ablation.")
    logger.info(f"Replication gate PASSED "
                f"({len(EXPECTED_PILOT)} pilot cells matched)")

    logger.info("=== Regime-free arms ===")
    frames = []
    for archs in TRAIN_POOLS:
        for config in CONFIGS_NR:
            frames.append(h2_regime_free(models_df, long_df, archs, config))
    nr = pd.concat(frames, ignore_index=True)
    nr.to_parquet(POOL_DIR / "h2_regime_free.parquet", index=False)

    rep = rep.assign(config="with_regime")
    keys = ["train_pool", "side", "regime"]
    merged = nr.merge(
        rep[keys + ["predictor_regret"]].rename(
            columns={"predictor_regret": "with_regime_regret"}),
        on=keys, how="left")

    order = {"near": 0, "mid": 1, "far": 2}
    merged["_r"] = merged["regime"].map(order)
    merged = merged.sort_values(["train_pool", "config", "side", "_r"]).drop(
        columns="_r")

    lines = [
        "# Pool A regime-free ablation (H2 without the regime input)",
        "",
        "**Source:** `x8_pool_a/pool_a_regime_free.py`. Protocol: E-B marginal "
        "variant (near/mid/far label rows, regime feature removed, one "
        "shortlist per probe model applied to every regime); heads trained on "
        "the benchmark pools and applied to the 40 probes without retraining. "
        "Replication gate against `27_pool_a_pilot.md` passed before the "
        "ablation arms ran.",
        "",
        "Configs: `source_nr` = NC + source one-hot; `none_nr` = NC only. "
        "`with_regime_regret` repeats the pilot's full-protocol result for "
        "the same cell.",
        "",
        "```",
        merged[["train_pool", "config", "side", "regime", "predictor_regret",
                "with_regime_regret", "empty_pct", "best_baseline",
                "baseline_regret"]].to_string(index=False),
        "```",
        "",
    ]
    out_md = POOL_DIR / "28_pool_a_regime_free.md"
    out_md.write_text("\n".join(lines))
    logger.info(f"Wrote {out_md}")

    joint = merged[merged["side"] == "all"]
    logger.info("Joint-side summary:\n"
                + joint[["train_pool", "config", "regime", "predictor_regret",
                         "with_regime_regret", "empty_pct",
                         "baseline_regret"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Per-source tables of the three transfer targets (camera-ready item 2.4).

The rebuttal promised regime x source tables for the transfers of the
headline predictor (regime-free NC-only heads, VGG13-trained). With the
22-CSF heads, per (regime, source) and pooled over the four sources, on the
joint side: the predictor's mean imputed set-regret, the best fixed CSF of
the cell (chosen with hindsight on the same rows), and the size-matched
random set. Every target uses `phase3_transfer.per_row`, so the regret, the
oracle, and the random set are computed the same way.

Best-fixed families (the paper's conventions):
  ResNet18   the Always-X candidates of Sec 3.6 (CTM, Energy, MLS, MSR,
             NNGuide, fDBD, Maha++, NCI), as in Sec 4.4;
  ViT        the full 21-CSF roster, as in App K;
  SSL        the full 21-CSF roster, as in App L.

Gates (abort before any table is written):
  GR  ResNet18 pooled regret and Always-CTM reproduce the stored headline
      (newcsf_heads aggregate and baseline aggregate).
  GV  ViT pooled regret reproduces App K (vit_transfer_22 cells).
  GS  SSL per-source and pooled NC-only regret and best fixed CSF reproduce
      the stored 21-CSF-head table of 42_pool_a_harmonized_tables.md.

Outputs: outputs/per_source/{resnet18,vit,ssl}.csv, outputs/57_per_source_tables.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parents[1]
for path in (EVAL_DIR.parent / "ablations", EVAL_DIR.parent / "data", EVAL_DIR,
             CODE_DIR / "x9_imagenet", CODE_DIR / "x8_pool_a"):
    sys.path.insert(0, str(path))
from calibration_features_clique import DEFAULT_OUT_ROOT, add_model_id  # noqa: E402
from input_ablation_grid import REGIMES, ssl_shortlists  # noqa: E402
from newcsf_heads import ALWAYS_BASELINES, NEW_CSFS, labels_22, prepare_pool  # noqa: E402
from phase3_transfer import per_row  # noqa: E402
from pool_a_harmonized_report import (  # noqa: E402
    marginal_training,
    standardized_vgg_models,
)
from regimefree_wilcoxon import load_parquet  # noqa: E402

SOURCES = ["cifar10", "cifar100", "supercifar100", "tinyimagenet"]
TOL = 1e-6
SSL_TOL = 0.006


def resnet18_rows(out_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Per-row ResNet18 frame of the headline predictor and its roster.

    Raises:
        RuntimeError: If the stored headline is not reproduced (gate GR).
    """
    _, _, test_ids, eval22, _, _ = prepare_pool(out_root)
    rows = eval22[eval22["model_id"].isin(test_ids)].copy()
    rows["family"] = rows["model_id"].str.split("|").str[2]
    preds = load_parquet(out_root / "newcsf_heads" / "track1" / "xarch"
                         / "none_nr_marginal" / "preds.parquet")
    probs = (preds[preds["regime"] == "near"][["model_id", "csf", "p_competitive"]]
             .rename(columns={"p_competitive": "p"}))
    roster = sorted(rows["csf"].unique())
    pr = per_row(rows[["model_id", "family", "eval_dataset", "csf", "raw_augrc"]],
                 probs, roster).merge(rows[["model_id", "eval_dataset", "regime"]]
                                      .drop_duplicates(), on=["model_id", "eval_dataset"])
    stored = pd.read_csv(out_root / "newcsf_heads" / "aggregate.csv")
    stored = stored[(stored["arm"] == "none_nr_marginal") & (stored["side"] == "all")]
    ctm = pd.read_csv(out_root / "newcsf_heads" / "baseline_aggregate.csv")
    ctm = ctm[(ctm["comparator_name"] == "Always-CTM") & (ctm["side"] == "all")]
    got = pr.groupby("regime")[["regret", "fixed::CTM"]].mean()
    dev = max((got["regret"] - stored.set_index("regime")["set_regret_imputed_mean"]).abs().max(),
              (got["fixed::CTM"] - ctm.set_index("regime")["regret_raw_mean"]).abs().max())
    if dev > TOL:
        raise RuntimeError(f"GR failed: max deviation {dev:.3g}")
    print(f"GR: ResNet18 headline reproduced (max |d| {dev:.1e})")
    return pr, roster


def vit_rows(out_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Per-row ViT frame of App K.

    Raises:
        RuntimeError: If App K's pooled regret is not reproduced (gate GV).
    """
    pr = pd.read_csv(out_root / "vit_transfer_22" / "per_row.csv")
    cells = pd.read_csv(out_root / "vit_transfer_22" / "cells.csv").set_index("cell")
    dev = (pr.groupby("regime")["regret"].mean() - cells.loc[REGIMES, "nc_only"]).abs().max()
    if dev > TOL:
        raise RuntimeError(f"GV failed: max deviation {dev:.3g}")
    print(f"GV: ViT App K regret reproduced (max |d| {dev:.1e})")
    return pr, sorted(c.removeprefix("fixed::") for c in pr.columns if c.startswith("fixed::"))


def ssl_rows(out_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Per-row SSL-probe frame of the NC-only heads trained on 22-CSF labels."""
    pool = out_root / "pool_a"
    long_df = pd.read_parquet(pool / "long_pool_a_harmonized.parquet")
    models_df = pd.read_parquet(pool / "models_pool_a_harmonized.parquet")
    long_df["model_id"] = (long_df["paradigm"] + "|" + long_df["source"] + "|"
                           + long_df["run"].astype(str))
    vgg_long = add_model_id(load_parquet(out_root / "track1" / "dataset"
                                         / "long_harmonized.parquet"))
    train, cols = marginal_training(standardized_vgg_models(vgg_long), labels_22(out_root))
    members = ssl_shortlists("none_nr", models_df, train, train, cols)
    probs = members[["model_id", "csf"]].drop_duplicates().assign(p=1.0)
    ood = long_df[long_df["regime"].isin(REGIMES)].rename(
        columns={"source": "family", "augrc": "raw_augrc"})
    roster = sorted(ood["csf"].unique())
    pr = per_row(ood[["model_id", "family", "eval_dataset", "csf", "raw_augrc"]],
                 probs, roster).merge(ood[["model_id", "eval_dataset", "regime"]]
                                      .drop_duplicates(), on=["model_id", "eval_dataset"])
    return pr, roster


def table(pr: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    """Per (regime, source) and pooled: predictor, best fixed CSF, random set."""
    recs = []
    for regime in REGIMES:
        for source in SOURCES + ["all"]:
            cell = pr[(pr["regime"] == regime)
                      & ((pr["family"] == source) if source != "all" else True)]
            means = cell[[f"fixed::{c}" for c in family]].mean()
            best = means.idxmin()
            recs.append({"regime": regime, "source": source, "rows": len(cell),
                         "nc_only": cell["regret"].mean(),
                         "best_fixed": best.removeprefix("fixed::"),
                         "best_fixed_regret": means[best],
                         "random_set": cell["random"].mean(),
                         "set_size": cell["set_size"].mean()})
    out = pd.DataFrame(recs)
    out["beats_best_fixed"] = out["nc_only"] < out["best_fixed_regret"]
    out["beats_random"] = out["nc_only"] < out["random_set"]
    return out


def gate_ssl(ssl: pd.DataFrame, out_root: Path) -> None:
    """GS: the SSL table reproduces the stored 21-CSF-head report.

    Raises:
        RuntimeError: On any cell that differs beyond rounding.
    """
    text = (out_root / "pool_a" / "42_pool_a_harmonized_tables.md").read_text()
    section = text.split("## Roster 21 with MahaPP/NCI heads")[1].split("\n## ")[0]
    label = {"C10": "cifar10", "C100": "cifar100", "SC100": "supercifar100",
             "TI": "tinyimagenet", "all": "all"}
    bad = []
    for regime, src, bf, bf_name, nc in re.findall(
            r"\| (\w+) \| (\w+) \| ([\d.]+) \(([^)]+)\) \| [^|]+ \| ([\d.]+)", section):
        row = ssl[(ssl["regime"] == regime) & (ssl["source"] == label[src])].iloc[0]
        if (abs(row["nc_only"] - float(nc)) > SSL_TOL
                or abs(row["best_fixed_regret"] - float(bf)) > SSL_TOL):
            bad.append((regime, src))
    if bad or len(ssl) != 15:
        raise RuntimeError(f"GS failed: {bad}")
    print("GS: SSL table reproduces the stored 21-CSF-head report (15 cells)")


def main() -> None:
    """Build, gate, and report the three per-source tables."""
    out_root = Path(DEFAULT_OUT_ROOT)
    targets = {}
    pr, _ = resnet18_rows(out_root)
    targets["resnet18"] = table(pr, ALWAYS_BASELINES + NEW_CSFS)
    pr, roster = vit_rows(out_root)
    targets["vit"] = table(pr, roster)
    pr, roster = ssl_rows(out_root)
    targets["ssl"] = table(pr, roster)
    gate_ssl(targets["ssl"], out_root)

    csv_dir = out_root / "per_source"
    csv_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Per-source tables of the three transfer targets\n\n",
             "**Source:** `nc_csf_predictivity/evaluation/per_source_tables.py`. Joint "
             "side, mean imputed set-regret (AUGRC x1000), 22-CSF NC-only heads trained "
             "on VGG13; best fixed CSF with hindsight per cell (ResNet18: Always-X "
             "candidates of Sec 4.4; ViT and SSL: full 21-CSF roster). Gates GR, GV, GS "
             "passed.\n\n"]
    for name, frame in targets.items():
        frame.to_csv(csv_dir / f"{name}.csv", index=False)
        per_src = frame[frame["source"] != "all"]
        pooled = frame[frame["source"] == "all"]
        lines.append(f"## {name}\n\nBeats the best fixed CSF: {int(per_src['beats_best_fixed'].sum())}"
                     f" of 12 per-source cells, {int(pooled['beats_best_fixed'].sum())} of 3 "
                     f"pooled. Beats the random set: {int(per_src['beats_random'].sum())} of 12,"
                     f" {int(pooled['beats_random'].sum())} of 3.\n\n```\n"
                     + frame.round(2).to_string(index=False) + "\n```\n\n")
    (out_root / "57_per_source_tables.md").write_text("".join(lines))
    print(f"wrote {out_root / '57_per_source_tables.md'}")


if __name__ == "__main__":
    main()

"""Per-config benchmark cliques with the new CSFs (MahaPP, NCI) — rebuild of
the lost 2026-08 script (originals were exclude-listed and never committed).

Computes top cliques at the SELECTOR-LABEL granularity from the E-F pilot
long CSV (`new_csfs_pilot_summary_long.csv`, arm == base), which covers the
full VGG-13 benchmark pool (280 models): cell = (paradigm, source, dropout,
reward, regime), blocks = run | eval_dataset | metric, extended roster
(paper Figure-1 roster + MahaPP + NCI). This mirrors
`data/cliques_track1.py` exactly and produces the in_top_clique labels used
to retrain the per-CSF selector heads with the two new columns.

NOTE: Figure-1-level cliques are NOT computed here — the published figure
selects one (do, rew) config per method by validation first; that pipeline
lives in `figure1_newcsf_cliques.py` (driven from the paper's scores_all
files plus validation-selected new-CSF rows).

Pipeline per cell, identical to `stats_eval.py` / `data/cliques_track1.py`:
  score = -metric (AUGRC and AURC stacked) -> friedman_blocked ->
  conover_posthoc_from_pivot (unadjusted, the paper's convention) ->
  maximal_cliques_from_pmatrix(alpha) -> rank_cliques ->
  greedy_exclusive_layers; top clique = layers[0].members.

Regimes come from the CLIP-distance grouping (clip_scores/clip_distances_
<source>.csv, group 0/1/2/3 = test/near/mid/far); eval modes map to dataset
names via pilot_from_configs.MODE_DATASET.

Outputs (suffixed by --tag so grid-NCI and BO runs coexist):
  outputs/newcsf_cliques/perconfig_cliques<tag>.parquet
  outputs/39_benchmark_newcsf_cliques<tag>.md

Usage:
  .venv/bin/python nc_csf_predictivity/evaluation/benchmark_newcsf_cliques.py \
      --long-csv new_csfs_pilot_summary_long.csv --tag _bo
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVAL_DIR.parent
CODE_DIR = PIPELINE_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

sys.path.insert(0, str(CODE_DIR))
from pilot_coverage import VIT_DATASET_TO_SOURCE  # noqa: E402
from pilot_from_configs import MODE_DATASET  # noqa: E402
from src.utils_stats import (  # noqa: E402
    conover_posthoc_from_pivot,
    friedman_blocked,
    greedy_exclusive_layers,
    maximal_cliques_from_pmatrix,
    rank_cliques,
)

ALPHA = 0.05
METRICS = ["AUGRC", "AURC"]  # the figure's RC metric pair
REGIME_OF_GROUP = {0: "test", 1: "near", 2: "mid", 3: "far"}
SOURCES = ["cifar10", "cifar100", "supercifar", "tinyimagenet"]
CNN_SRC_KEY = {"cifar10": "cifar10", "cifar100": "cifar100",
               "supercifar": "supercifar", "tiny-imagenet-200": "tinyimagenet"}
CLIP_KEY = {"supercifar": "supercifar100"}

# stats_eval.py roster in this pilot's stats-file vocabulary.
HEAD_SIDE = ["MSR", "MLS", "PE", "GE", "PCE", "GEN", "REN", "Energy",
             "GradNorm", "pNML", "Confidence"]
FEATURE_SIDE = ["Maha", "CTM", "ViM", "Residual", "fDBD", "NNGuide", "NeCo",
                "PCA_RecError_global", "KPCA_RecError_global"]
NEW_CSFS = ["MahaPP", "NCI"]
DISPLAY = {"PCA_RecError_global": "PCA RecError",
           "KPCA_RecError_global": "KPCA RecError"}

CNN_RE = re.compile(
    r"^(?P<src>[a-z0-9-]+)_paper_sweep/(?P<paradigm>[a-z]+)_bb"
    r"(?P<bb>vgg13|resnet18)_do(?P<do>\d)_run(?P<run>\d+)_rew(?P<rew>[\d.]+)$")
VIT_RE = re.compile(
    r"^vit/(?P<ds>.+)_modelvit_bbvit_lr(?P<lr>[\d.eE-]+)_bs\d+_run"
    r"(?P<run>\d+)_do(?P<do>\d)_rew(?P<rew>[\d.]+)$")


def parse_experiments(names: pd.Series) -> pd.DataFrame:
    rows = []
    for exp in names:
        m = CNN_RE.match(exp)
        if m:
            src = CNN_SRC_KEY.get(m["src"])
            fam = "VGG13" if m["bb"] == "vgg13" else "ResNet18"
            rows.append({"experiment": exp, "family": fam,
                         "paradigm": m["paradigm"], "source": src,
                         "dropout": int(m["do"]), "reward": float(m["rew"]),
                         "lr": np.nan, "run": f"run{m['run']}"})
            continue
        m = VIT_RE.match(exp)
        if m:
            rows.append({"experiment": exp, "family": "ViT",
                         "paradigm": "vit",
                         "source": VIT_DATASET_TO_SOURCE.get(m["ds"]),
                         "dropout": int(m["do"]), "reward": float(m["rew"]),
                         "lr": float(m["lr"]), "run": f"run{m['run']}"})
            continue
        rows.append({"experiment": exp, "family": None, "paradigm": None,
                     "source": None, "dropout": None, "reward": None,
                     "lr": None, "run": None})
    return pd.DataFrame(rows)


def clip_groups(clip_dir: Path, source: str) -> dict[str, int]:
    key = CLIP_KEY.get(source, source)
    df = pd.read_csv(clip_dir / f"clip_distances_{key}.csv",
                     header=[0, 1], index_col=0)
    gcol = [c for c in df.columns if c[0] == "group"][0]
    return df[gcol].astype(int).to_dict()


def cell_top_clique(cell: pd.DataFrame, block_cols: list[str],
                    p_adjust: str | None = None) -> dict:
    """cell rows: (block cols..., csf, score). Returns pipeline result."""
    cell = cell.copy()
    cell["block"] = cell[block_cols].astype(str).agg("|".join, axis=1)
    try:
        stat, p, pivot = friedman_blocked(
            cell, entity_col="csf", block_col="block", value_col="score")
    except Exception as e:  # noqa: BLE001
        return {"status": f"friedman_error:{e}", "top_clique": [],
                "ranks": {}, "n_blocks": 0, "friedman_p": np.nan}
    if not isinstance(stat, float) or math.isnan(stat):
        return {"status": "degenerate", "top_clique": [], "ranks": {},
                "n_blocks": int(pivot.shape[0]), "friedman_p": np.nan}
    ph = conover_posthoc_from_pivot(pivot, p_adjust=p_adjust)
    avg_ranks = pivot.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    cliques = maximal_cliques_from_pmatrix(ph, alpha=ALPHA)
    scored = rank_cliques(cliques, list(avg_ranks.index), avg_ranks)
    layers = greedy_exclusive_layers(scored)
    return {"status": "ok",
            "top_clique": list(layers[0]["members"]) if layers else [],
            "ranks": {c: float(avg_ranks.loc[c]) for c in avg_ranks.index},
            "n_blocks": int(pivot.shape[0]),
            "friedman_p": float(p)}


def melt_scores(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Stack AUGRC/AURC into (keep..., metric, score=-value)."""
    out = []
    for metric in METRICS:
        sub = df[keep + ["csf", metric]].rename(columns={metric: "value"})
        sub["metric"] = metric.lower()
        sub["score"] = -sub["value"]
        out.append(sub.drop(columns="value"))
    return pd.concat(out, ignore_index=True)


def perconfig_cliques(df: pd.DataFrame, roster: list[str],
                      family: str = "VGG13", p_adjust: str | None = None
                      ) -> pd.DataFrame:
    """One family; cells = (paradigm, source, dropout, reward, regime);
    blocks = run | eval_dataset | metric. Regime 'all' pools near+mid+far."""
    flat = []
    vgg = df[(df.family == family) & df.csf.isin(roster)]
    for (paradigm, source, do, rew), cell in vgg.groupby(
            ["paradigm", "source", "dropout", "reward"]):
        long = melt_scores(cell, ["run", "eval_dataset", "regime"])
        for regime in ["test", "near", "mid", "far", "all"]:
            sub = (long[long.regime.isin(["near", "mid", "far"])]
                   if regime == "all" else long[long.regime == regime])
            if sub.empty:
                continue
            res = cell_top_clique(sub, ["run", "eval_dataset", "metric"],
                                  p_adjust=p_adjust)
            top = set(res["top_clique"])
            for csf in sorted(set(roster) & set(cell.csf.unique())):
                flat.append({"paradigm": paradigm, "source": source,
                             "dropout": bool(do), "reward": float(rew),
                             "regime": regime, "csf": csf,
                             "in_top_clique": csf in top,
                             "mean_rank": res["ranks"].get(csf, np.nan),
                             "n_blocks": res["n_blocks"],
                             "friedman_p": res["friedman_p"],
                             "status": res["status"]})
    return pd.DataFrame(flat)


def load_base(long_csv: Path, clip_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base-arm pilot rows with family, eval dataset, and CLIP regime.

    Returns:
        (rows, parsed experiment info); rows outside the paper suite are dropped.
    """
    long_df = pd.read_csv(long_csv)
    base = long_df[long_df.arm == "base"].copy()
    # display renames so outputs speak the published figure's vocabulary
    base["csf"] = base.csf.replace(DISPLAY)

    info = parse_experiments(pd.Series(base.experiment.unique()))
    unparsed = info[info.family.isna()]
    for exp in unparsed.experiment:
        print(f"WARNING: unparsed experiment name skipped: {exp}")
    base = base.merge(info.dropna(subset=["family"]), on="experiment")

    base["eval_dataset"] = base["mode"].map(
        lambda m: "test" if m == "iid_test" else MODE_DATASET.get(m))
    groups = {s: clip_groups(clip_dir, s) for s in SOURCES}
    base["regime"] = [
        REGIME_OF_GROUP.get(groups[s].get(d)) if s in groups else None
        for s, d in zip(base.source, base.eval_dataset)]
    dropped = base[base.regime.isna()]
    if not dropped.empty:
        print("dropped (mode outside the paper suite): "
              f"{sorted(dropped['mode'].unique())}")
    return base.dropna(subset=["regime"]), info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--long-csv",
                    default=str(CODE_DIR / "new_csfs_pilot_summary_long.csv"))
    ap.add_argument("--clip-dir", default=str(CODE_DIR / "clip_scores"))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--tag", default="",
                    help="output suffix, e.g. _gridNCI or _bo")
    args = ap.parse_args()

    base, info = load_base(Path(args.long_csv), Path(args.clip_dir))
    unparsed = info[info.family.isna()]
    print("=== per-config level (VGG-13, extended roster)")
    ext_vgg_roster = [DISPLAY.get(c, c) for c in HEAD_SIDE + FEATURE_SIDE] \
        + NEW_CSFS
    percfg = perconfig_cliques(base, ext_vgg_roster)

    out_dir = Path(args.out_root) / "newcsf_cliques"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    pq.write_table(pa.Table.from_pandas(percfg, preserve_index=False),
                   str(out_dir / f"perconfig_cliques{tag}.parquet"))

    lines = [f"# 39 — Per-config benchmark cliques with MahaPP + NCI{tag}\n\n",
             f"**Input:** `{Path(args.long_csv).name}`; alpha={ALPHA}; "
             "unadjusted Conover-Friedman (paper convention); "
             "metrics AUGRC+AURC negated; top clique = first greedy layer.\n\n",
             f"Experiments: {info.family.notna().sum()} parsed "
             f"({(info.family == 'VGG13').sum()} VGG13, "
             f"{(info.family == 'ViT').sum()} ViT, "
             f"{(info.family == 'ResNet18').sum()} ResNet18); "
             f"{len(unparsed)} unparsed.\n\n",
             "## Per-config cliques (VGG-13, extended roster)\n\n"]
    sz = (percfg[percfg.in_top_clique]
          .groupby(["paradigm", "source", "dropout", "reward", "regime"])
          .size().reset_index(name="k"))
    lines.append(f"Cells: {sz.shape[0]}; top-clique size mean "
                 f"{sz.k.mean():.2f} (min {sz.k.min()}, max {sz.k.max()}).\n\n")
    for csf in NEW_CSFS:
        n = int(percfg[(percfg.csf == csf) & percfg.in_top_clique]
                .groupby(["paradigm", "source", "dropout", "reward", "regime"])
                .ngroups)
        lines.append(f"- {csf} in top clique in {n} per-config cells "
                     f"(of {sz.shape[0]}).\n")
    report = Path(args.out_root) / f"39_benchmark_newcsf_cliques{tag}.md"
    report.write_text("".join(lines))
    print(f"wrote {report}")
    print(f"wrote {out_dir / f'perconfig_cliques{tag}.parquet'} "
          f"({len(percfg):,} rows)")


if __name__ == "__main__":
    main()

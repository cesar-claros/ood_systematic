"""Figure-1 cliques with MahaPP + NCI, driven from the paper's scores_all
files — rebuild of the lost 2026-08 script.

The paper's Figure 1 is NOT computed from the raw per-config stats: each
method first gets ONE (dropout, reward) config selected per (source,
paradigm, metric) by run-averaged validation score (`retrieve_scores.py:
select_best_hyperparameters` on iid_val; joint min over (do, rew) for
lower-better metrics), and `stats_eval.py` then runs Friedman/Conover/
maximal-clique per (source, CLIP group) with blocks = dataset|model|metric|
group|run on those selected rows. The consolidated selected rows live in
`scores_risk/scores_all_<METRIC>_MCD-False_<Conv|ViT>_<source>.csv` (wide:
one column per eval dataset; methods vocabulary uses spaces, e.g.
"KPCA RecError global").

This script therefore:
  1. loads the paper's scores_all AUGRC+AURC files verbatim (legacy roster =
     published Figure 1 by construction; run as validation that the local
     pipeline reproduces the published cliques),
  2. selects MahaPP/NCI configs by the same iid_val rule from the pilot long
     CSV (requires `iid_val` rows for the new CSFs — the eval-only
     `csf_eval --test_mode val` pass over the pilot lists; when absent the
     extended run is skipped with a warning),
  3. appends the new-CSF rows and recomputes every cell,
  4. reports placements: cells entered, cliques joined, displacements.

Roster filter mirrors stats_eval: drop methods containing 'global'/'class'
except {KPCA, PCA} RecError global, drop CTMmean/CTMmeanOC, drop Confidence
on ViT. Conover is unadjusted (paper convention), alpha = 0.05.

Usage:
  .venv/bin/python nc_csf_predictivity/evaluation/figure1_newcsf_cliques.py \
      --long-csv new_csfs_pilot_summary_long.csv --tag _bo
  # legacy-only validation (no new-CSF rows needed):
  ... --legacy-only --tag _published_check
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = EVAL_DIR.parent
CODE_DIR = PIPELINE_DIR.parent
DEFAULT_OUT_ROOT = PIPELINE_DIR / "outputs"

sys.path.insert(0, str(CODE_DIR))
from pilot_from_configs import MODE_DATASET  # noqa: E402
from src.utils_stats import (  # noqa: E402
    conover_posthoc_from_pivot,
    friedman_blocked,
    greedy_exclusive_layers,
    maximal_cliques_from_pmatrix,
    rank_cliques,
)

ALPHA = 0.05
METRICS = ["AUGRC", "AURC"]
REGIME_OF_GROUP = {0: "test", 1: "near", 2: "mid", 3: "far"}
SOURCES = ["cifar10", "cifar100", "supercifar", "tinyimagenet"]
SCORES_SRC = {"cifar10": "cifar10", "cifar100": "cifar100",
              "supercifar": "supercifar100", "tinyimagenet": "tinyimagenet"}
CLIP_KEY = {"supercifar": "supercifar100"}
NEW_CSFS = ["MahaPP", "NCI"]
CNN_PARADIGMS = ["confidnet", "devries", "dg"]

# pilot stats vocabulary -> scores_all vocabulary is identity for the new
# CSFs; id columns of the wide scores_all files:
ID_COLS = ["model", "drop out", "methods", "reward", "run"]


def clip_groups(clip_dir: Path, source: str) -> dict[str, int]:
    key = CLIP_KEY.get(source, source)
    df = pd.read_csv(clip_dir / f"clip_distances_{key}.csv",
                     header=[0, 1], index_col=0)
    gcol = [c for c in df.columns if c[0] == "group"][0]
    return df[gcol].astype(int).to_dict()


def load_scores_all(source: str, backbone: str) -> pd.DataFrame:
    """Paper scores_all AUGRC+AURC for one (source, backbone), melted long:
    (model, drop out, methods, reward, run, dataset, metric, score)."""
    frames = []
    for metric in METRICS:
        path = (CODE_DIR / "scores_risk" /
                f"scores_all_{metric}_MCD-False_{backbone}_"
                f"{SCORES_SRC[source]}.csv")
        wide = pd.read_csv(path)
        long = wide.melt(id_vars=ID_COLS, var_name="dataset",
                         value_name="value")
        long["metric"] = metric
        frames.append(long)
    df = pd.concat(frames, ignore_index=True)
    df["score"] = -df["value"]  # lower-better -> higher-better
    return df.drop(columns="value")


def roster_filter(methods: pd.Series, backbone: str) -> pd.Series:
    """stats_eval filter: keep plain methods + the two RecError globals."""
    keep_exceptions = {"KPCA RecError global", "PCA RecError global"}
    excluded = {"CTMmean", "CTMmeanOC", "MCD-CTMmean", "MCD-CTMmeanOC"}
    variant = (methods.str.contains("global|class", case=False, na=False)
               & ~methods.isin(keep_exceptions))
    bad = variant | methods.isin(excluded)
    if backbone == "ViT":
        bad |= methods.eq("Confidence")
    return ~bad


def select_new_csf_rows(long_csv: Path, source: str, backbone: str,
                        arm: str = "base", csfs: tuple[str, ...] = tuple(NEW_CSFS),
                        suffix: str = "") -> pd.DataFrame | None:
    """Validation-selected pilot rows in the melted scores_all schema.

    Selection mirrors select_best_hyperparameters: per (paradigm, method,
    metric), run-averaged iid_val score, joint argmin over configs
    ((do, rew) for CNN; (lr, do) for ViT). Returns None when the pilot CSV
    has no iid_val rows for the requested CSFs.

    Args:
        long_csv: Pilot long CSV (one row per experiment, arm, CSF, mode).
        source: Source key of SOURCES.
        backbone: "Conv" or "ViT".
        arm: Pilot arm to read ("base", or an activation wrapper).
        csfs: CSFs of that arm to select.
        suffix: Appended to the method name (keeps wrapped variants apart).
    """
    long = pd.read_csv(long_csv)
    base = long[(long.arm == arm) & long.csf.isin(csfs)].copy()
    is_vit = base.experiment.str.startswith("vit/")
    base = base[is_vit] if backbone == "ViT" else base[~is_vit]

    if backbone == "ViT":
        pat = (r"^vit/(?P<ds>.+)_modelvit_bbvit_lr(?P<lr>[\d.eE-]+)_bs\d+_"
               r"run(?P<run>\d+)_do(?P<do>\d)_rew(?P<rew>[\d.]+)$")
        info = base.experiment.str.extract(pat)
        ds_to_src = {"cifar10": "cifar10", "cifar100": "cifar100",
                     "super_cifar100": "supercifar",
                     "supercifar100": "supercifar",
                     "tinyimagenet": "tinyimagenet",
                     "tiny-imagenet-200": "tinyimagenet"}
        base["source"] = info["ds"].map(ds_to_src)
        base["model"] = "modelvit"
        base["config"] = "lr" + info["lr"] + "_do" + info["do"]
        base["drop out"] = "do" + info["do"]
        base["reward"] = "rew" + info["rew"]
        base["run"] = info["run"].astype(int)
        bb_ok = pd.Series(True, index=base.index)
    else:
        pat = (r"^(?P<src>[a-z0-9-]+)_paper_sweep/(?P<paradigm>[a-z]+)_bb"
               r"(?P<bb>vgg13|resnet18)_do(?P<do>\d)_run(?P<run>\d+)_rew"
               r"(?P<rew>[\d.]+)$")
        info = base.experiment.str.extract(pat)
        src_key = {"cifar10": "cifar10", "cifar100": "cifar100",
                   "supercifar": "supercifar",
                   "tiny-imagenet-200": "tinyimagenet"}
        base["source"] = info["src"].map(src_key)
        base["model"] = info["paradigm"]
        base["config"] = "do" + info["do"] + "_rew" + info["rew"]
        base["drop out"] = "do" + info["do"]
        base["reward"] = "rew" + info["rew"]
        base["run"] = info["run"].astype(int)
        bb_ok = info["bb"] == "vgg13"  # ResNet18 handled separately if ever
    base = base[bb_ok & (base.source == source)]
    if base.empty:
        return None
    val = base[base["mode"] == "iid_val"]
    if val.empty:
        return None

    base["dataset"] = base["mode"].map(
        lambda m: "test" if m == "iid_test" else MODE_DATASET.get(m))
    rows = []
    for metric in METRICS:
        sel = (val.groupby(["model", "csf", "config"])[metric].mean()
               .groupby(["model", "csf"]).idxmin())
        for (model, csf), (_, _, config) in sel.items():
            picked = base[(base.model == model) & (base.csf == csf)
                          & (base.config == config)
                          & base["mode"].ne("iid_val")]
            for (do, rew, run), g in picked.groupby(
                    ["drop out", "reward", "run"]):
                row = {"model": model, "drop out": do, "methods": csf + suffix,
                       "reward": rew, "run": run, "metric": metric}
                for _, r in g.iterrows():
                    if pd.notna(r.dataset):
                        row[r.dataset] = r[metric]
                rows.append(row)
    out = pd.DataFrame(rows)
    data_cols = [c for c in out.columns if c not in ID_COLS + ["metric"]]
    long_out = out.melt(id_vars=ID_COLS + ["metric"], value_vars=data_cols,
                        var_name="dataset", value_name="value").dropna(
                            subset=["value"])
    long_out["score"] = -long_out["value"]
    return long_out.drop(columns="value")


def cell_top_clique(cell: pd.DataFrame, p_adjust: str | None = None) -> dict:
    cell = cell.copy()
    cell["block"] = (cell["dataset"].astype(str) + "|"
                     + cell["model"].astype(str) + "|"
                     + cell["metric"].astype(str) + "|"
                     + cell["run"].astype(str))
    try:
        stat, p, pivot = friedman_blocked(
            cell, entity_col="methods", block_col="block", value_col="score")
    except Exception as e:  # noqa: BLE001
        return {"status": f"friedman_error:{e}", "top_clique": [],
                "ranks": {}, "n_blocks": 0}
    if not isinstance(stat, float) or math.isnan(stat):
        return {"status": "degenerate", "top_clique": [], "ranks": {},
                "n_blocks": int(pivot.shape[0])}
    ph = conover_posthoc_from_pivot(pivot, p_adjust=p_adjust)
    avg_ranks = pivot.rank(axis=1, ascending=False).mean(axis=0).sort_values()
    cliques = maximal_cliques_from_pmatrix(ph, alpha=ALPHA)
    scored = rank_cliques(cliques, list(avg_ranks.index), avg_ranks)
    layers = greedy_exclusive_layers(scored)
    return {"status": "ok",
            "top_clique": list(layers[0]["members"]) if layers else [],
            "ranks": {c: float(avg_ranks.loc[c]) for c in avg_ranks.index},
            "n_blocks": int(pivot.shape[0])}


def panel_cliques(df: pd.DataFrame, backbone: str, source: str,
                  groups: dict[str, int], p_adjust: str | None = None) -> dict:
    """Top clique per (panel=model, regime) for one (backbone, source)."""
    df = df[roster_filter(df.methods, backbone)].copy()
    df["group"] = df.dataset.map(groups)
    df = df.dropna(subset=["group"])
    df["regime"] = df.group.astype(int).map(REGIME_OF_GROUP)
    out = {}
    panels = CNN_PARADIGMS if backbone == "Conv" else ["modelvit"]
    for panel in panels:
        sub = df[df.model == panel]
        for regime in ["test", "near", "mid", "far"]:
            cell = sub[sub.regime == regime]
            if cell.empty:
                continue
            out[(panel, regime)] = cell_top_clique(cell, p_adjust=p_adjust)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--long-csv",
                    default=str(CODE_DIR / "new_csfs_pilot_summary_long.csv"))
    ap.add_argument("--clip-dir", default=str(CODE_DIR / "clip_scores"))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--tag", default="")
    ap.add_argument("--legacy-only", action="store_true",
                    help="published-figure validation: no new-CSF rows")
    args = ap.parse_args()

    display = {"modelvit": "vit", "KPCA RecError global": "KPCA RecError",
               "PCA RecError global": "PCA RecError"}
    results = {}   # (backbone, panel, source, regime) -> {legacy, extended}
    skipped_ext = []
    for backbone in ["Conv", "ViT"]:
        for source in SOURCES:
            try:
                paper = load_scores_all(source, backbone)
            except FileNotFoundError as e:
                print(f"missing scores_all: {e}")
                continue
            groups = clip_groups(Path(args.clip_dir), source)
            leg = panel_cliques(paper, backbone, source, groups)
            ext = {}
            if not args.legacy_only:
                new_rows = select_new_csf_rows(Path(args.long_csv), source,
                                               backbone)
                if new_rows is None:
                    skipped_ext.append(f"{backbone}/{source}")
                else:
                    aug = pd.concat([paper, new_rows], ignore_index=True)
                    ext = panel_cliques(aug, backbone, source, groups)
            for (panel, regime), res in leg.items():
                key = (backbone, display.get(panel, panel), source, regime)
                results[key] = {"legacy": res,
                                "extended": ext.get((panel, regime))}

    if skipped_ext:
        print("EXTENDED RUN SKIPPED (no iid_val rows for MahaPP/NCI) for: "
              + ", ".join(skipped_ext))

    def disp(names):
        return [display.get(n, n) for n in names]

    lines = [f"# 40 — Figure-1 cliques from paper scores_all{args.tag}\n\n",
             f"alpha={ALPHA}; unadjusted Conover; blocks = dataset|model|"
             "metric|run; legacy roster = published Figure 1 inputs "
             "verbatim.\n\n"]
    fig1 = {k: v for k, v in results.items() if k[3] != "test"}
    lines.append(f"Figure-1 cells: {len(fig1)} (expected 48).\n\n")

    if not args.legacy_only and not skipped_ext:
        for csf in NEW_CSFS:
            entered = [(k, v) for k, v in fig1.items()
                       if v["extended"] and csf in v["extended"]["top_clique"]]
            lines.append(f"## {csf}: enters {len(entered)} of "
                         f"{len(fig1)} cells\n\n")
            for (bb, panel, source, regime), v in entered:
                ext = disp(v["extended"]["top_clique"])
                leg = disp(v["legacy"]["top_clique"])
                joined = [c for c in ext if c not in NEW_CSFS]
                displaced = [c for c in leg if c not in ext]
                note = (f" — DISPLACES {', '.join(displaced)}"
                        if displaced else "")
                lines.append(f"- {panel} / {source} / {regime}: joins "
                             f"{{{', '.join(joined) or '(alone)'}}}{note}\n")
            lines.append("\n")
        shifts = []
        for k, v in fig1.items():
            if not v["extended"]:
                continue
            leg, ext = disp(v["legacy"]["top_clique"]), \
                disp(v["extended"]["top_clique"])
            gone = [c for c in leg if c not in ext]
            came = [c for c in ext if c not in leg and c not in NEW_CSFS]
            if gone or came:
                shifts.append((k, came, gone))
        lines.append(f"## Membership shifts among legacy CSFs: "
                     f"{len(shifts)} cells\n\n")
        for (bb, panel, source, regime), came, gone in shifts:
            delta = (f"+{{{', '.join(came)}}} " if came else "") + \
                    (f"-{{{', '.join(gone)}}}" if gone else "")
            lines.append(f"- {panel} / {source} / {regime}: {delta}\n")
        lines.append("\n")

    lines.append("## All cells (legacy = published reproduction"
                 + ("" if args.legacy_only else "; extended alongside")
                 + ")\n\n| Panel | Source | Regime | Legacy top clique |"
                 + (" Extended top clique |" if not args.legacy_only else "")
                 + "\n|---|---|---|---|" +
                 ("---|" if not args.legacy_only else "") + "\n")
    panel_order = {"confidnet": 0, "devries": 1, "dg": 2, "vit": 3}
    for key in sorted(results, key=lambda k: (
            panel_order.get(k[1], 9), SOURCES.index(k[2]),
            ["test", "near", "mid", "far"].index(k[3]))):
        bb, panel, source, regime = key
        if regime == "test":
            continue
        v = results[key]
        leg = ", ".join(disp(v["legacy"]["top_clique"]))
        row = f"| {panel} | {source} | {regime} | {leg} |"
        if not args.legacy_only:
            ext = (", ".join(
                f"**{c}**" if c in NEW_CSFS else c
                for c in disp(v["extended"]["top_clique"]))
                if v["extended"] else "(skipped)")
            row += f" {ext} |"
        lines.append(row + "\n")

    report = Path(args.out_root) / f"40_figure1_newcsf_cliques{args.tag}.md"
    report.write_text("".join(lines))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()

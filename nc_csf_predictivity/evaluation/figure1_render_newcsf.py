"""Build the sandbox for regenerating Figure 1 with MahaPP + NCI.

Writes `code/fig1_newcsf_sandbox/scores_risk/` containing the paper's
scores_all files for the four risk metrics (AUGRC, AURC, AUROC_f, FPR@95TPR)
with MahaPP and NCI rows appended, selected per (source, paradigm, metric)
by the paper's rule (run-averaged iid_val value of that same metric over
configs; max for AUROC_f, min otherwise — retrieve_scores.py
select_best_hyperparameters). scores_calibration/ and clip_scores/ are
symlinked. Render from inside the sandbox with the paper's own plotting:

  cd fig1_newcsf_sandbox
  for m in confidnet devries dg; do
    ../.venv/bin/python ../stats_eval.py --backbone Conv --model $m \
        --metric-group RC --filter-methods --output-dir fig1_out
  done
  ../.venv/bin/python ../stats_eval.py --backbone ViT --model modelvit \
      --metric-group RC --filter-methods --output-dir fig1_out
  ../.venv/bin/python ../stats_eval.py --all-paradigms --metric-group RC \
      --filter-methods --output-dir fig1_out --font-scale 1.3 --no-suptitle \
      --arc-links --fig-width 15.1

The legacy rows are the paper's files verbatim, so every published clique is
reproduced by construction (validated 48/48 against the published
*_cliques.json); the only change in the figure is the two added rows.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
CODE_DIR = EVAL_DIR.parent.parent
sys.path.insert(0, str(CODE_DIR))
from pilot_from_configs import MODE_DATASET  # noqa: E402

SANDBOX = CODE_DIR / "fig1_newcsf_sandbox"
METRIC_SENSE = {"AUGRC": "min", "AURC": "min",
                "AUROC_f": "max", "FPR@95TPR": "min"}
SOURCES = ["cifar10", "cifar100", "supercifar", "tinyimagenet"]
SCORES_SRC = {"cifar10": "cifar10", "cifar100": "cifar100",
              "supercifar": "supercifar100", "tinyimagenet": "tinyimagenet"}
NEW_CSFS = ["MahaPP", "NCI"]
ID_COLS = ["model", "drop out", "methods", "reward", "run"]

CNN_RE = re.compile(
    r"^(?P<src>[a-z0-9-]+)_paper_sweep/(?P<paradigm>[a-z]+)_bbvgg13_do"
    r"(?P<do>\d)_run(?P<run>\d+)_rew(?P<rew>[\d.]+)$")
VIT_RE = re.compile(
    r"^vit/(?P<ds>.+)_modelvit_bbvit_lr(?P<lr>[\d.eE-]+)_bs\d+_run"
    r"(?P<run>\d+)_do(?P<do>\d)_rew(?P<rew>[\d.]+)$")
CNN_SRC = {"cifar10": "cifar10", "cifar100": "cifar100",
           "supercifar": "supercifar", "tiny-imagenet-200": "tinyimagenet"}
VIT_DS = {"cifar10": "cifar10", "cifar100": "cifar100",
          "super_cifar100": "supercifar", "supercifar100": "supercifar",
          "tinyimagenet": "tinyimagenet", "tiny-imagenet-200": "tinyimagenet"}


def parse(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp in base.experiment.unique():
        m = CNN_RE.match(exp)
        if m:
            rows.append({"experiment": exp, "backbone": "Conv",
                         "model": m["paradigm"],
                         "source": CNN_SRC.get(m["src"]),
                         "do": f"do{m['do']}", "reward": f"rew{m['rew']}",
                         "run": int(m["run"]),
                         "config": f"do{m['do']}_rew{m['rew']}"})
            continue
        m = VIT_RE.match(exp)
        if m:
            rows.append({"experiment": exp, "backbone": "ViT",
                         "model": "modelvit",
                         "source": VIT_DS.get(m["ds"]),
                         "do": f"do{m['do']}", "reward": f"rew{m['rew']}",
                         "run": int(m["run"]),
                         "config": f"lr{m['lr']}_do{m['do']}"})
    return pd.DataFrame(rows)


def main() -> None:
    long = pd.read_csv(CODE_DIR / "new_csfs_pilot_summary_long.csv",
                       low_memory=False)
    base = long[(long.arm == "base") & long.csf.isin(NEW_CSFS)].copy()
    info = parse(base)
    base = base.merge(info, on="experiment")
    base["dataset"] = base["mode"].map(
        lambda m: "test" if m == "iid_test" else MODE_DATASET.get(m))

    out_scores = SANDBOX / "scores_risk"
    out_scores.mkdir(parents=True, exist_ok=True)
    for link, target in [("scores_calibration", CODE_DIR / "scores_calibration"),
                         ("clip_scores", CODE_DIR / "clip_scores")]:
        lp = SANDBOX / link
        if not lp.exists():
            lp.symlink_to(target)

    for backbone in ["Conv", "ViT"]:
        pool = base[base.backbone == backbone]
        for source in SOURCES:
            sub = pool[pool.source == source]
            val = sub[sub["mode"] == "iid_val"]
            test = sub[sub["mode"] != "iid_val"]
            for metric, sense in METRIC_SENSE.items():
                fname = (f"scores_all_{metric}_MCD-False_{backbone}_"
                         f"{SCORES_SRC[source]}.csv")
                paper = pd.read_csv(CODE_DIR / "scores_risk" / fname)
                data_cols = [c for c in paper.columns if c not in ID_COLS]
                grouped = val.groupby(["model", "csf", "config"])[metric].mean()
                sel = (grouped.groupby(["model", "csf"]).idxmax() if sense == "max"
                       else grouped.groupby(["model", "csf"]).idxmin())
                new_rows = []
                for (model, csf), (_, _, config) in sel.items():
                    picked = test[(test.model == model) & (test.csf == csf)
                                  & (test.config == config)]
                    for (do, rew, run), g in picked.groupby(
                            ["do", "reward", "run"]):
                        row = {"model": model, "drop out": do, "methods": csf,
                               "reward": rew, "run": run}
                        vals = dict(zip(g.dataset, g[metric]))
                        for c in data_cols:
                            row[c] = vals.get(c, np.nan)
                        new_rows.append(row)
                nr = pd.DataFrame(new_rows, columns=list(paper.columns))
                nan_cells = int(nr[data_cols].isna().sum().sum())
                if nan_cells:
                    missing = [c for c in data_cols if nr[c].isna().any()]
                    print(f"WARNING {backbone}/{source}/{metric}: "
                          f"{nan_cells} unfilled cells in {missing}")
                aug = pd.concat([paper, nr], ignore_index=True)
                aug.to_csv(out_scores / fname, index=False)
            n = len(nr)
            print(f"{backbone:4s} {source:12s}: +{n} new-CSF rows per metric "
                  f"({sorted(nr.methods.unique())}, "
                  f"configs {sorted(set(zip(nr.methods, nr['drop out'], nr.reward)))})")
    print(f"\nsandbox ready: {SANDBOX}")


if __name__ == "__main__":
    main()

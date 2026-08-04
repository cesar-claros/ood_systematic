"""Expanded tj47 Q2 table: VGG13 -> ResNet18 per (regime, source).

The first tj47 reply shared the regime ablation pooled over sources. This
table disaggregates the same transfer by OOD regime and source dataset and
adds the paper's third input configuration (NC + n_cls + regime, the
ordinal class count replacing the source one-hot), so every input
combination appears together:

  NC+source+regime (paper) | NC+n_cls+regime | NC+regime | NC+source | NC only

All five configurations are evaluated from the stored xarch predictions
(`calib_cliques` for the regime-conditioned configs, `calib_cliques_regime_free`
marginal for the regime-removed ones); nothing is refit. The best-fixed
column is the strongest Always-of-{MSR, Energy, MLS, CTM, fDBD, NNGuide}
per cell, the paper's baseline family (appropriate here: the CNN clique
members). Replication gates: the five pooled rows must reproduce the
paper (1.02/1.18/0.39 source; 1.24/1.24/0.36 n_cls; 1.45/1.44/0.40 none)
and E-B (1.06/0.96/0.64 source_nr; 1.23/1.16/0.80 none_nr).

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/tj47_expanded_table.py
Output: nc_csf_predictivity/outputs/32_tj47_expanded_table.md
"""
from __future__ import annotations

import pathlib
import sys

from loguru import logger

CODE_DIR = pathlib.Path(__file__).resolve().parents[2]
for p in (CODE_DIR, CODE_DIR / "x8_pool_a",
          CODE_DIR / "nc_csf_predictivity" / "data",
          CODE_DIR / "nc_csf_predictivity" / "ablations",
          CODE_DIR / "nc_csf_predictivity" / "evaluation"):
    sys.path.insert(0, str(p))

from pool_a_analysis import OUT_ROOT  # noqa: E402
from input_ablation_grid import (  # noqa: E402
    REGIMES,
    benchmark_rows,
    evaluate,
    stored_shortlists,
)

COLS = [("source", "NC+source+regime"), ("n_classes", "NC+n_cls+regime"),
        ("none", "NC+regime"), ("source_nr", "NC+source"),
        ("none_nr", "NC only")]

EXPECTED = {
    "source": (1.02, 1.18, 0.39),
    "n_classes": (1.24, 1.24, 0.36),
    "none": (1.45, 1.44, 0.40),
    "source_nr": (1.06, 0.96, 0.64),
    "none_nr": (1.23, 1.16, 0.80),
}


def main() -> None:
    rows = benchmark_rows("ResNet18")
    results = {}
    for config, _ in COLS:
        sl = stored_shortlists("xarch", config, "vgg13_to_resnet18")
        results[config] = evaluate(rows, sl)
        logger.info(f"{config}: " + " ".join(
            f"{r}={results[config][('all', r)]['predictor']}"
            for r in REGIMES))

    bad = []
    for config, exp in EXPECTED.items():
        got = tuple(results[config][("all", r)]["predictor"]
                    for r in REGIMES)
        if any(abs(g - e) > 0.02 for g, e in zip(got, exp)):
            bad.append((config, exp, got))
    if bad:
        for b in bad:
            logger.error(f"replication mismatch: {b}")
        raise SystemExit("Replication gate FAILED; table not written.")
    logger.info(f"Replication gate PASSED ({len(EXPECTED)} pooled configs)")

    src_label = {"cifar10": "C10", "cifar100": "C100",
                 "supercifar100": "SC100", "tinyimagenet": "TI",
                 "all": "all"}
    wins = {"beat": 0, "total": 0}
    lines = [
        "# Expanded tj47 Q2 table: VGG13 -> ResNet18 per (regime, source)\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/tj47_expanded_table.py`."
        " Joint-side mean imputed set-regret from the stored xarch"
        " predictions; best fixed CSF = strongest Always-of-6 per cell;"
        " pooled 'all' rows reproduce the paper and E-B tables exactly.\n\n",
        "| Regime | Source | Best fixed CSF | "
        + " | ".join(lbl for _, lbl in COLS)
        + " |\n|" + "---|" * (3 + len(COLS)) + "\n"]
    for regime in REGIMES:
        for src in ["cifar10", "cifar100", "supercifar100", "tinyimagenet",
                    "all"]:
            key = (src, regime)
            bf = results["source"][key]
            cells = []
            for config, _ in COLS:
                v = results[config][key]["predictor"]
                cells.append(f"{v:.2f}")
                if src != "all":
                    wins["total"] += 1
                    wins["beat"] += v < results[config][key]["best_fixed"]
            lines.append(f"| {regime} | {src_label[src]} | "
                         f"{bf['best_fixed']:.2f} "
                         f"({bf['best_fixed_name']}) | "
                         + " | ".join(cells) + " |\n")
    lines.append(f"\nPer-source cells where the predictor beats the best "
                 f"fixed CSF: {wins['beat']} of {wins['total']}.\n")
    out = OUT_ROOT / "32_tj47_expanded_table.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}; per-source beats: "
                f"{wins['beat']}/{wins['total']}")
    print("".join(lines))


if __name__ == "__main__":
    main()

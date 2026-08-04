"""Expanded R4De W3/Q5 table: VGG13 -> ViT per (regime, source).

The first R4De reply reported the cross-family transfer pooled over
sources (joint regret 1.90/3.66/2.07 against direction-matched fixed
baselines 6.25/12.20/10.62). This table disaggregates the SAME evaluation
by OOD regime and source and shows all five input configurations:

  NC+source+regime | NC+n_cls+regime | NC+regime | NC+source | NC only

Protocol unchanged from the first reply: heads trained on the 280 VGG-13
models only (zero ViT models in training, the pure cross-family test,
stored lopo_modelvit predictions), and the best-fixed baseline is the
strongest Always-of-{MSR, Energy, MLS, CTM, fDBD, NNGuide} per cell, the
paper's CNN-clique family. The stricter variant (leave-one-ViT-run-out
training, baselines from the ViT panels' own top-clique union) lives in
the meta-review grid (`31_loro_vit_grid.md`).

Replication gates: the four previously reported pooled configs
(E-C source 1.90/3.66/2.07; grid source_nr 1.76/2.34/2.30,
none 2.50/6.46/0.63, none_nr 1.35/2.24/3.94); NC+n_cls is newly evaluated
from its stored predictions.

Run from `code/`:
  ./.venv/bin/python nc_csf_predictivity/evaluation/r4de_expanded_table.py
Output: nc_csf_predictivity/outputs/34_r4de_expanded_table.md
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
    "source": (1.90, 3.66, 2.07),
    "none": (2.50, 6.46, 0.63),
    "source_nr": (1.76, 2.34, 2.30),
    "none_nr": (1.35, 2.24, 3.94),
}


def main() -> None:
    rows = benchmark_rows("ViT")
    results = {}
    for config, _ in COLS:
        sl = stored_shortlists("lopo", config, "lopo_modelvit")
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
        "# Expanded R4De W3/Q5 table: VGG13 -> ViT per (regime, source)\n",
        "\n**Source:** `nc_csf_predictivity/evaluation/r4de_expanded_table"
        ".py`. Joint-side mean imputed set-regret from the stored "
        "lopo_modelvit predictions (zero ViT models in training); best "
        "fixed CSF = strongest Always-of-6 per cell (the paper's CNN "
        "family, as in the first R4De reply); pooled rows reproduce the "
        "previously reported numbers. Stricter variant (LORO training, "
        "ViT-clique-family baselines): 31_loro_vit_grid.md.\n\n",
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
    out = OUT_ROOT / "34_r4de_expanded_table.md"
    out.write_text("".join(lines))
    logger.info(f"Wrote {out}; per-source beats: "
                f"{wins['beat']}/{wins['total']}")
    print("".join(lines))


if __name__ == "__main__":
    main()

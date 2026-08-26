"""EXPLORATORY roster-surgery sensitivity probes (post audit-#6 closure).

Status: post-hoc, NOT frozen analyses (the pool is redefined after outcomes
were known); they answer user sensitivity questions of 2026-08-25 and are
reported as exploratory only. Everything downstream of the pool filter reuses
the registered machinery unchanged: the frozen per-cell theory cache
(theory_cell_predictions.parquet), run_folds (fold seed 2027), paired_ci
(checkpoint-clustered, B=2000), e2_gate3 (train-source tertile boundaries,
frozen RETAINED/REVERSED/INCONCLUSIVE rules), and the crossing-audit
estimator machinery.

Probe A ("no_supercifar"): drop the supercifar100 source entirely
(3 sources, 190 checkpoints). Question: is any conclusion supercifar-driven?
Probe B ("confidnet_devries"): drop the DG paradigm entirely (2 paradigms,
80 checkpoints, single reward). Question: how much rides on DG's
reward-induced geometry variation?

Declared seeds: A folds 2027 / CIs 61,161,62,162; B folds 2027 / CIs
71,171,72,172; crossing bands 73.

Usage (from code/): python stage2_sensitivity_probes.py
Outputs: nc_csf_predictivity/outputs/track1/stage2_sensitivity_no_supercifar.md
         nc_csf_predictivity/outputs/track1/stage2_sensitivity_confidnet_devries.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crossing_robustness_audit import (analyze_curve, crossing_value,
                                       ordering_retained, tertiles)
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, load_coords,
                                       run_folds)
from stage2_closure import (OUT_DIR, build_cells_with_severity, e2_gate3,
                            paired_ci, theory_full)


def evaluate_world(name: str, sub: pd.DataFrame, th: pd.Series,
                   seeds: tuple[int, int, int, int, int]) -> list[str]:
    s_ck, s_ckb, s_lo, s_lob, s_band = seeds
    L = [f"pool: {sub.cell.nunique()} checkpoints, {len(sub)} cells"]
    g = sub.groupby("cell").var_collapse.first()
    L.append(f"var_collapse spread: [{g.min():.3f}, {g.max():.3f}]")
    for mode, cs, cbs in (("ckpt5", s_ck, s_ckb), ("loso", s_lo, s_lob)):
        fitted = run_folds(sub, th, mode, np.random.default_rng(FOLD_SEED))
        mat = fitted[np.abs(fitted.gap) >= MATERIALITY].dropna(
            subset=["geometry", "severity"])
        L.append(
            f"[{mode}] material {len(mat)}, frac+ {(mat.gap > 0).mean():.2f}; "
            f"theory {accuracy(mat.theory.values, mat.gap.values):.3f}, "
            f"severity {accuracy(mat.severity.values, mat.gap.values):.3f}, "
            f"geometry {accuracy(mat.geometry.values, mat.gap.values):.3f}")
        L.append(f"  G-S sign: "
                 f"{paired_ci(mat, 'geometry', 'severity', accuracy, cs)}")
        L.append(f"  G-S balanced: "
                 f"{paired_ci(mat, 'geometry', 'severity', balanced_accuracy, cbs)}")
        if mode == "loso":
            for src, gg in mat.groupby("source"):
                L.append(f"  held-out {src}: n {len(gg)}, "
                         f"frac+ {(gg.gap > 0).mean():.2f}, "
                         f"geo {accuracy(gg.geometry.values, gg.gap.values):.3f}, "
                         f"sev {accuracy(gg.severity.values, gg.gap.values):.3f}")
    data: dict[str, list] = {}
    for r in sub.itertuples():
        data.setdefault(r.cell, []).append((float(r.d), float(r.gap)))
    fine = np.linspace(sub.d.min(), sub.d.max(), 301)
    rng = np.random.default_rng(s_band)
    rec = analyze_curve("pava", data, sorted(data), fine, 2000, rng)
    L.append(f"pooled first up-crossing {rec['first_up_crossing']}, "
             f"tie region {rec.get('tie_region')}")
    res = {}
    for sname, members in tertiles(sub).items():
        d2 = {c: v for c, v in data.items() if c in members}
        res[sname] = analyze_curve("pava", d2, sorted(d2), fine, 2000, rng)
    L.append("in-sample tertile crossings: "
             + str({k: crossing_value(v) for k, v in res.items()})
             + f", ordering retained: {ordering_retained(res)}")
    g3 = e2_gate3(sub)
    L.append(f"Gate-3-style held-out ordering: {g3['outcomes']} -> "
             f"{g3['verdict']}")
    return L


def main() -> None:
    cells = build_cells_with_severity()
    coords, problems = load_coords(Path("pilot0/pool_coords"))
    assert not problems, problems
    fr = theory_full(cells, coords)
    theory_all = pd.Series(fr.pred_gap.values, index=cells.index)

    worlds = {
        "no_supercifar": (
            cells.source != "supercifar100", (61, 161, 62, 162, 63),
            "Pool without the supercifar100 source (3 sources). "
            "Question: is any conclusion supercifar-driven?"),
        "confidnet_devries": (
            cells.cell.str.split("|").str[0].isin(["confidnet", "devries"]),
            (71, 171, 72, 172, 73),
            "Pool restricted to confidnet+devries (no DG paradigm, single "
            "reward). Question: how much rides on DG's reward-induced "
            "geometry variation?"),
    }
    for name, (mask, seeds, question) in worlds.items():
        sub = cells[mask].reset_index(drop=True)
        th = pd.Series(theory_all[mask].values, index=sub.index)
        lines = evaluate_world(name, sub, th, seeds)
        body = "\n".join(
            [f"# Exploratory sensitivity probe: {name}", "",
             "EXPLORATORY, post-hoc roster surgery (2026-08-25); NOT a frozen "
             "analysis. " + question,
             "Reproduce: `python stage2_sensitivity_probes.py` (from code/; "
             "seeds in the module docstring; reuses the frozen theory cache "
             "and registered fold/CI/gate machinery).", "", "```",
             *lines, "```"])
        (OUT_DIR / f"stage2_sensitivity_{name}.md").write_text(body + "\n")
        print(f"=== {name} ===")
        print("\n".join(lines))
    print("PROBES DONE")


if __name__ == "__main__":
    main()

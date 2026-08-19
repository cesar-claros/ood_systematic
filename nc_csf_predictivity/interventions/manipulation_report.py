"""Pilot 1 manipulation report: gates M1/M2 before any CSF scoring.

Consumes the per-run geometry JSONs from `extract_manipulation.py`
(checkpoint tag `last`) and renders the pre-registered manipulation
report (code/pilot1/MANIFEST.md section 4). Contains NO OOD or detector
information; this report is committed before outcomes are unblinded.

Gates:
  M1: at least one active arm moves self_duality by >= 1 baseline-seed SD,
      AND the ordering across the five dose levels is monotone
      (Spearman(dose rank, self_duality) <= -0.8; ranks:
      lam -0.1 < 0.0 < 0.3 < 1.0 < hard, prediction: decreasing metric).
  M2: median val-accuracy drop vs the baseline mean <= 1.5 pp for at
      least one active arm (all runs stay in the ITT regardless).
Selectivity is reported, not gated (joint-intervention relabeling per the
manifest); the range target compares arms against the benchmark span
[0.03, 0.13].

Usage (from code/):
    python nc_csf_predictivity/interventions/manipulation_report.py \
        [--geometry_dir nc_csf_predictivity/interventions/geometry] \
        [--out nc_csf_predictivity/interventions/manipulation_report.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DOSE_RANK = {"-0.1": 0, "0.0": 1, "0.3": 2, "1.0": 3, "hard": 4}
ARM_ORDER = ("-0.1", "0.0", "0.3", "1.0", "hard")
ARM_LABEL = {"-0.1": "A1-", "0.0": "baseline", "0.3": "A1+",
             "1.0": "A1++", "hard": "A2"}
PAPYAN = ("var_collapse", "equinorm_uc", "equinorm_wc", "equiangular_uc",
          "equiangular_wc", "max_equiangular_uc", "max_equiangular_wc",
          "self_duality")
BENCHMARK_SPAN = (0.03, 0.13)


def load_records(geometry_dir: Path) -> list[dict]:
    """Final-checkpoint records only, sorted by (dose rank, run)."""
    records = [json.loads(p.read_text())
               for p in sorted(geometry_dir.glob("*__last.json"))]
    for rec in records:
        if rec["lam"] not in DOSE_RANK:
            raise ValueError(f"unknown lam '{rec['lam']}' in "
                             f"{rec['experiment']}")
        rec["dose_rank"] = DOSE_RANK[rec["lam"]]
    return sorted(records, key=lambda r: (r["dose_rank"], r["run"]))


def arm_stats(records: list[dict], key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for lam in ARM_ORDER:
        vals = np.array([r[key] for r in records if r["lam"] == lam])
        if len(vals):
            out[lam] = {"mean": float(vals.mean()),
                        "sd": float(vals.std(ddof=1)) if len(vals) > 1
                        else float("nan"),
                        "median": float(np.median(vals)), "n": len(vals)}
    return out


def evaluate(records: list[dict]) -> dict:
    """Compute gates and tables; pure function for testability."""
    sd_stats = arm_stats(records, "self_duality")
    base = sd_stats["0.0"]
    active = [lam for lam in ARM_ORDER if lam != "0.0" and lam in sd_stats]

    moves = {lam: abs(sd_stats[lam]["mean"] - base["mean"]) / base["sd"]
             for lam in active}
    rho = float(spearmanr([r["dose_rank"] for r in records],
                          [r["self_duality"] for r in records]).statistic)
    m1_strength = any(v >= 1.0 for v in moves.values())
    m1_monotone = rho <= -0.8
    m1 = m1_strength and m1_monotone

    acc_stats = arm_stats(records, "val_acc")
    acc_drops = {lam: (acc_stats["0.0"]["mean"] - acc_stats[lam]["median"])
                 * 100.0 for lam in active}
    m2 = any(d <= 1.5 for d in acc_drops.values())

    selectivity: dict[str, dict] = {}
    for lam in active:
        target_move = moves[lam]
        rows = {}
        for coord in PAPYAN:
            if coord == "self_duality":
                continue
            stats_c = arm_stats(records, coord)
            base_c = stats_c["0.0"]
            std_move = (abs(stats_c[lam]["mean"] - base_c["mean"])
                        / base_c["sd"] if base_c["sd"] > 0 else float("inf"))
            rows[coord] = {
                "std_move": std_move,
                "ratio_to_target": (std_move / target_move
                                    if target_move > 0 else float("inf")),
            }
        selectivity[lam] = rows

    exits_span = {lam: not (BENCHMARK_SPAN[0] <= sd_stats[lam]["mean"]
                            <= BENCHMARK_SPAN[1]) for lam in active}
    return {"sd_stats": sd_stats, "moves": moves, "spearman": rho,
            "M1_strength": m1_strength, "M1_monotone": m1_monotone,
            "M1": m1, "acc_stats": acc_stats, "acc_drops": acc_drops,
            "M2": m2, "selectivity": selectivity, "exits_span": exits_span}


def render(records: list[dict], ev: dict) -> str:
    lines = ["# Pilot 1 Manipulation Report (blinded stage)", ""]
    lines.append(f"Runs measured: {len(records)} (final checkpoints). "
                 "No OOD or detector information enters this report.")
    lines.append("")
    lines.append(f"- **M1 manipulation**: strength "
                 f"{'PASS' if ev['M1_strength'] else 'FAIL'} "
                 f"(max standardized move "
                 f"{max(ev['moves'].values()):.2f} baseline SDs); "
                 f"monotone {'PASS' if ev['M1_monotone'] else 'FAIL'} "
                 f"(Spearman(dose, self_duality) = {ev['spearman']:+.3f}, "
                 f"gate <= -0.8) => **{'PASS' if ev['M1'] else 'FAIL'}**")
    lines.append(f"- **M2 accuracy**: "
                 f"**{'PASS' if ev['M2'] else 'FAIL'}** "
                 "(median val-acc drop vs baseline, pp: "
                 + ", ".join(f"{ARM_LABEL[k]} {v:+.2f}"
                             for k, v in ev["acc_drops"].items()) + ")")
    lines.append("")
    lines.append("## Self-duality and accuracy by arm")
    lines.append("")
    lines.append("| arm | self_duality mean (sd) | std move vs baseline "
                 "| exits benchmark span [0.03, 0.13] | val acc median |")
    lines.append("|---|---|---|---|---|")
    for lam in ARM_ORDER:
        if lam not in ev["sd_stats"]:
            continue
        s = ev["sd_stats"][lam]
        move = (f"{ev['moves'][lam]:.2f}" if lam in ev["moves"] else "-")
        span = ("yes" if ev["exits_span"].get(lam) else
                ("no" if lam in ev["exits_span"] else "-"))
        lines.append(f"| {ARM_LABEL[lam]} | {s['mean']:.4f} ({s['sd']:.4f}) "
                     f"| {move} | {span} "
                     f"| {ev['acc_stats'][lam]['median']:.4f} |")
    lines.append("")
    lines.append("## Selectivity (reported, not gated; ratio > 0.25 flags "
                 "joint-intervention relabeling)")
    lines.append("")
    for lam, rows in ev["selectivity"].items():
        flagged = {c: r for c, r in rows.items()
                   if r["ratio_to_target"] > 0.25}
        summary = (", ".join(f"{c} ({r['ratio_to_target']:.2f}x)"
                             for c, r in sorted(
                                 flagged.items(),
                                 key=lambda kv: -kv[1]["ratio_to_target"]))
                   or "none")
        lines.append(f"- {ARM_LABEL[lam]}: {summary}")
    lines.append("")
    lines.append("## Per-run record (final checkpoints)")
    lines.append("")
    lines.append("| experiment | self_dual | var_col | eqn_uc | eqa_wc "
                 "| head_resid | logit_scale | train acc | val acc |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in records:
        lines.append(
            f"| {r['experiment'].split('/')[-1]} | {r['self_duality']:.4f} "
            f"| {r['var_collapse']:.4f} | {r['equinorm_uc']:.4f} "
            f"| {r['equiangular_wc']:.4f} | {r['head_residual_fraction']:.4f} "
            f"| {r['logit_scale']:.2f} | {r['train_acc']:.4f} "
            f"| {r['val_acc']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 1 manipulation report")
    parser.add_argument(
        "--geometry_dir", type=str,
        default="nc_csf_predictivity/interventions/geometry")
    parser.add_argument(
        "--out", type=str,
        default="nc_csf_predictivity/interventions/manipulation_report.md")
    args = parser.parse_args()
    records = load_records(Path(args.geometry_dir))
    ev = evaluate(records)
    Path(args.out).write_text(render(records, ev))
    print(f"M1={'PASS' if ev['M1'] else 'FAIL'} "
          f"(strength {ev['M1_strength']}, monotone rho={ev['spearman']:+.3f}) "
          f"M2={'PASS' if ev['M2'] else 'FAIL'}; wrote {args.out}")


if __name__ == "__main__":
    main()

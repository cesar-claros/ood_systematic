#!/usr/bin/env python
"""Pilot 1 gate evaluation over the deep-stage JSONs (protocol of record:
documentation/X6_gradpca_pilot1_protocol.md, §3).

Reads every non-scores JSON in --out_dir, groups checkpoints into regimes
(vgg = from-scratch CNN, vit = fine-tuned transformer), and evaluates:

  gate 1  equivalence sanity: stage-internal GradPCA_head_sum vs
          ActPCA_cmeans metric rows identical (Theorem E1; tol 1e-9).
  gate 2  material value: for a matched contrast (lastlayer_sum - head_sum,
          lastlayer_max - head_max), the per-seed MEDIAN over OOD modes of
          the AUROC_f delta reaches >= +0.01 in at least one seed of a
          regime, with the same (positive) sign in >= 2 of 3 seeds.
  gate 3  breadth: the winning contrast's gain is not confined to one OOD
          dataset (fraction of OOD modes with positive delta reported), and
          the other regime's median degradation on that contrast is not
          larger than the gain.

AUGRC deltas are reported as secondary (sign convention: negative = deep
better). Feasibility numbers (k*, throughput, peak memory, self-check) are
summarized for the protocol's gate 5 record. Output: markdown report to
stdout and <out_dir>/pilot1_gate_report.md.

Usage:  python x6_gradpca/aggregate_pilot1.py [--out_dir x6_gradpca/outputs]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from statistics import median

MATERIAL = 0.01  # AUROC_f percentage-point gate (protocol §3, plan gate 2)
CONTRASTS = {
    "lastlayer_sum_minus_head_sum": ("GradPCA_lastlayer_sum", "GradPCA_head_sum"),
    "lastlayer_max_minus_head_max": ("GradPCA_lastlayer_max", "GradPCA_head_max"),
}
L = []  # report lines


def say(s=""):
    L.append(s)
    print(s)


def regime_of(rec):
    return "vit" if rec["study"] == "vit" or rec["model_path"].startswith("vit/") else "vgg"


def seed_of(rec):
    m = re.search(r"run(\d+)", rec["model_path"])
    return m.group(1) if m else rec["model_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="x6_gradpca/outputs")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.out_dir, "*.json"))
                   if not f.endswith("_scores.json"))
    recs = [json.load(open(f)) for f in files]
    recs = [r for r in recs if "modes" in r and "deltas" in r]
    if not recs:
        sys.exit(f"ERROR: no stage JSONs found in {args.out_dir}")
    say(f"# Pilot 1 gate report ({len(recs)} checkpoints)\n")

    # ---- gate 1: equivalence sanity -----------------------------------
    say("## Gate 1 — equivalence sanity (head_sum == ActPCA_cmeans)\n")
    g1_ok, worst = True, 0.0
    for r in recs:
        for mode, rows in r["modes"].items():
            for metric in ("AUROC_f", "AUGRC"):
                d = abs(rows["GradPCA_head_sum"][metric] - rows["ActPCA_cmeans"][metric])
                worst = max(worst, d)
                if d > 1e-9:
                    g1_ok = False
                    say(f"- VIOLATION {r['model_path']} {mode} {metric}: gap {d:.3e}")
    say(f"- max |head_sum - cmeans| over all checkpoints/modes/metrics: {worst:.3e}")
    say(f"- **verdict: {'PASS' if g1_ok else 'FAIL'}**\n")

    # ---- per-checkpoint contrast summaries ----------------------------
    by_regime = {"vgg": [], "vit": []}
    for r in recs:
        ood = sorted(m for m in r["deltas"] if m.startswith("ood_"))
        entry = {"seed": seed_of(r), "model_path": r["model_path"], "ood_modes": ood}
        for cname in CONTRASTS:
            au = [r["deltas"][m][cname]["AUROC_f"] for m in ood]
            ag = [r["deltas"][m][cname]["AUGRC"] for m in ood]
            entry[cname] = {
                "median_auroc": median(au), "per_mode_auroc": dict(zip(ood, au)),
                "median_augrc": median(ag),
                "frac_positive": sum(a > 0 for a in au) / len(au),
            }
        by_regime[regime_of(r)].append(entry)

    say("## Matched contrasts (AUROC_f delta, deep minus head; positive = deep better)\n")
    for regime, entries in by_regime.items():
        if not entries:
            continue
        say(f"### regime: {regime}\n")
        say("| seed | contrast | median OOD dAUROC_f | frac OOD modes > 0 | median OOD dAUGRC |")
        say("|---|---|---|---|---|")
        for e in sorted(entries, key=lambda x: x["seed"]):
            for cname in CONTRASTS:
                c = e[cname]
                say(f"| {e['seed']} | {cname} | {c['median_auroc']:+.4f} | "
                    f"{c['frac_positive']:.2f} | {c['median_augrc']:+.3f} |")
        say("")
        say("Per-mode dAUROC_f (seed-averaged):\n")
        modes = entries[0]["ood_modes"]
        say("| mode | " + " | ".join(CONTRASTS) + " |")
        say("|---|" + "---|" * len(CONTRASTS))
        for m in modes:
            vals = []
            for cname in CONTRASTS:
                xs = [e[cname]["per_mode_auroc"].get(m) for e in entries]
                xs = [x for x in xs if x is not None]
                vals.append(f"{sum(xs) / len(xs):+.4f}" if xs else "n/a")
            say(f"| {m} | " + " | ".join(vals) + " |")
        say("")

    # ---- gate 2: material value ---------------------------------------
    say("## Gate 2 — material deep-gradient value\n")
    winners = []
    for regime, entries in by_regime.items():
        if not entries:
            continue
        for cname in CONTRASTS:
            meds = [e[cname]["median_auroc"] for e in entries]
            hit = max(meds) >= MATERIAL
            consistent = sum(m > 0 for m in meds) >= 2 or len(meds) < 3
            verdict = hit and consistent
            say(f"- {regime} / {cname}: per-seed medians "
                f"{['%+.4f' % m for m in meds]} -> "
                f"max {max(meds):+.4f} vs {MATERIAL:+.2f}, "
                f"sign-consistent {sum(m > 0 for m in meds)}/{len(meds)} "
                f"-> **{'PASS' if verdict else 'fail'}**")
            if verdict:
                winners.append((regime, cname, median(meds)))
    say(f"- **verdict: {'PASS' if winners else 'FAIL'}"
        + (f" (winning: {[(r, c) for r, c, _ in winners]})" if winners else "") + "**\n")

    # ---- gate 3: breadth ----------------------------------------------
    say("## Gate 3 — breadth\n")
    g3_notes = []
    for regime, cname, gain in winners:
        entries = by_regime[regime]
        fracs = [e[cname]["frac_positive"] for e in entries]
        other = "vit" if regime == "vgg" else "vgg"
        other_meds = [e[cname]["median_auroc"] for e in by_regime.get(other, [])]
        other_med = median(other_meds) if other_meds else None
        broad = median(fracs) > 0.5  # majority of seeds broadly positive
        # (matches gate 2's 2/3 logic: the dissenting seed does not veto)
        no_collapse = other_med is None or other_med > -abs(gain)
        say(f"- {regime} / {cname}: frac positive OOD modes per seed "
            f"{['%.2f' % f for f in fracs]}; other-regime ({other}) median "
            f"{'n/a' if other_med is None else '%+.4f' % other_med} vs gain {gain:+.4f} "
            f"-> **{'PASS' if broad and no_collapse else 'fail'}**")
        g3_notes.append(broad and no_collapse)
    if not winners:
        say("- not applicable (gate 2 failed)")
    say("")

    # ---- feasibility record (gate 5) ----------------------------------
    say("## Feasibility record\n")
    say("| checkpoint | k lastlayer sum/max | fit s | peak GPU GB | selfcheck f64 max | fp32 max |")
    say("|---|---|---|---|---|---|")
    for r in recs:
        sc = r.get("selfcheck") or {}
        f64 = max(sc.get("f64_sum", float("nan")), sc.get("f64_max", float("nan")))
        f32 = max(sc.get("fp32_sum", float("nan")), sc.get("fp32_max", float("nan")))
        say(f"| {r['model_path']} | {r.get('k_GradPCA_lastlayer_sum')}/"
            f"{r.get('k_GradPCA_lastlayer_max')} | {r['runtime_sec'].get('fit')} | "
            f"{r.get('peak_gpu_mem_gb', 'n/a')} | {f64:.1e} | {f32:.1e} |")
    say("")

    overall = g1_ok and bool(winners) and (all(g3_notes) if g3_notes else False)
    say(f"## Overall: gates 1-3 {'PASS -> proceed to Pilot 2' if overall else 'NOT all passed -> see protocol pivots (Pivot A if no material value anywhere; Pivot D if ViT-only)'}")

    path = os.path.join(args.out_dir, "pilot1_gate_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

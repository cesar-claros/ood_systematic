#!/usr/bin/env python
"""EXPLORATORY post hoc mechanism autopsy for the Pilot 1 deep-gradient negative.

Everything here is post hoc (the pilot's registered question is settled:
protocol §6, Pivot A). This script asks WHY the deep single-layer GradPCA
variants lose to their matched head baselines, using only the retained
per-set scores in <slug>_scores.npz (projectors were not persisted).

Three candidate mechanisms and their score-level fingerprints:

  A  in-span capture: the deep class-mean span captures the OOD
     displacement, so OOD retained-energy scores rise toward (or above) the
     ID level; fingerprint = OOD score elevation ~ 0 or positive, per-cell
     AUROC depressed toward or below 0.5 (the normalized-score inversion
     regime of X6 Result 2, row 3).
  B  ID-side degradation: deep ID gradients concentrate poorly in the span;
     fingerprint = inflated ID score spread (CV) with OOD elevation similar
     to the head's.
  C  signal-plus-noise: the deep score is a noisy monotone copy of the head
     score; fingerprint = high deep-vs-head rank correlation with uniform
     mild AUROC loss.

Per (checkpoint, OOD mode, matched pair) the script reports: ID/OOD score
means and spreads, OOD elevation (OOD_mean - ID_mean)/ID_std, joint AUROC
(correct-ID iid_test rows vs OOD, higher = more ID; < 0.5 flagged as
INVERSION), deep-vs-head Spearman on the pooled joint set, and the overlap
of the top-decile most-ID-looking OOD samples. Regime-level medians and a
mechanism scorecard close the report.

Usage (HPC, code/ root):  python x6_gradpca/autopsy_pilot1.py
Writes <out_dir>/pilot1_autopsy_report.md and prints it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from statistics import median

import numpy as np

PAIRS = {"sum": ("GradPCA_lastlayer_sum", "GradPCA_head_sum"),
         "max": ("GradPCA_lastlayer_max", "GradPCA_head_max")}
L = []


def say(s=""):
    L.append(s)
    print(s)


def auroc(id_scores, ood_scores):
    """P(ID score > OOD score), ties 1/2 (higher = more ID)."""
    s = np.concatenate([id_scores, ood_scores])
    # midranks (ties get the average rank)
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    ranks = np.empty_like(sr)
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    n_id = len(id_scores)
    return float((r[:n_id].sum() - n_id * (n_id + 1) / 2) / (n_id * len(ood_scores)))


def spearman(a, b):
    ra = a.argsort(kind="mergesort").argsort().astype(np.float64)
    rb = b.argsort(kind="mergesort").argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="x6_gradpca/outputs")
    args = ap.parse_args()

    cells = []  # one dict per (checkpoint, ood_mode, pair)
    npz_files = sorted(glob.glob(os.path.join(args.out_dir, "*_scores.npz")))
    if not npz_files:
        sys.exit(f"ERROR: no *_scores.npz in {args.out_dir}")
    say("# Pilot 1 mechanism autopsy (EXPLORATORY, post hoc)\n")
    say(f"{len(npz_files)} checkpoints; mechanisms and fingerprints in the script docstring.\n")

    for nf in npz_files:
        slug = os.path.basename(nf)[:-len("_scores.npz")]
        rec = json.load(open(os.path.join(args.out_dir, f"{slug}.json")))
        regime = "vit" if rec["study"] == "vit" else "vgg"
        seed = (re.search(r"run(\d+)", rec["model_path"]) or [None, "?"])[1]
        z = np.load(nf)
        id_correct = z["iid_test__correct"].astype(bool)
        ood_modes = sorted({k.split("__")[0] for k in z.files
                            if k.startswith("ood_") and k.endswith("__correct")})
        for mode in ood_modes:
            for pname, (deep, head) in PAIRS.items():
                row = {"regime": regime, "seed": seed, "mode": mode, "pair": pname}
                for tag, var in (("deep", deep), ("head", head)):
                    sid = z[f"iid_test__{var}"][id_correct].astype(np.float64)
                    sood = z[f"{mode}__{var}"].astype(np.float64)
                    row[f"{tag}_id_mean"] = sid.mean()
                    row[f"{tag}_id_cv"] = sid.std() / max(abs(sid.mean()), 1e-12)
                    row[f"{tag}_ood_mean"] = sood.mean()
                    row[f"{tag}_elev"] = (sood.mean() - sid.mean()) / max(sid.std(), 1e-12)
                    row[f"{tag}_auroc"] = auroc(sid, sood)
                d_j = np.concatenate([z[f"iid_test__{deep}"][id_correct], z[f"{mode}__{deep}"]]).astype(np.float64)
                h_j = np.concatenate([z[f"iid_test__{head}"][id_correct], z[f"{mode}__{head}"]]).astype(np.float64)
                row["spearman"] = spearman(d_j, h_j)
                k = max(1, len(z[f"{mode}__{deep}"]) // 10)
                top_d = set(np.argsort(z[f"{mode}__{deep}"])[-k:])
                top_h = set(np.argsort(z[f"{mode}__{head}"])[-k:])
                row["topdecile_overlap"] = len(top_d & top_h) / k
                cells.append(row)

    # ---- per-regime tables ---------------------------------------------
    for regime in ("vgg", "vit"):
        sub = [c for c in cells if c["regime"] == regime]
        if not sub:
            continue
        say(f"## regime: {regime} (per-cell means over seeds)\n")
        say("| mode | pair | AUROC deep/head | OOD elev deep/head | ID CV deep/head | spearman | top-decile overlap |")
        say("|---|---|---|---|---|---|---|")
        modes = sorted({c["mode"] for c in sub})
        for mode in modes:
            for pname in PAIRS:
                cc = [c for c in sub if c["mode"] == mode and c["pair"] == pname]
                m = lambda key: sum(c[key] for c in cc) / len(cc)
                inv = " **INV**" if m("deep_auroc") < 0.5 else ""
                say(f"| {mode} | {pname} | {m('deep_auroc'):.3f}/{m('head_auroc'):.3f}{inv} | "
                    f"{m('deep_elev'):+.2f}/{m('head_elev'):+.2f} | "
                    f"{m('deep_id_cv'):.3f}/{m('head_id_cv'):.3f} | "
                    f"{m('spearman'):.3f} | {m('topdecile_overlap'):.2f} |")
        say("")

    # ---- mechanism scorecard -------------------------------------------
    say("## Mechanism scorecard (medians over cells)\n")
    say("| regime | pair | dAUROC | OOD elev deep | OOD elev head | ID CV ratio deep/head | spearman | inversions |")
    say("|---|---|---|---|---|---|---|---|")
    for regime in ("vgg", "vit"):
        for pname in PAIRS:
            cc = [c for c in cells if c["regime"] == regime and c["pair"] == pname]
            if not cc:
                continue
            md = lambda key: median(c[key] for c in cc)
            n_inv = sum(c["deep_auroc"] < 0.5 for c in cc)
            say(f"| {regime} | {pname} | {md('deep_auroc') - md('head_auroc'):+.3f} | "
                f"{md('deep_elev'):+.2f} | {md('head_elev'):+.2f} | "
                f"{md('deep_id_cv') / max(md('head_id_cv'), 1e-12):.2f} | "
                f"{md('spearman'):.3f} | {n_inv}/{len(cc)} |")
    say("")
    say("Reading guide: mechanism A (in-span capture / normalized-score inversion) = deep OOD "
        "elevation much closer to 0 (or positive) than the head's, inversion cells present; "
        "mechanism B (ID-side degradation) = deep ID CV ratio >> 1 with similar elevations; "
        "mechanism C (signal-plus-noise) = spearman near 1 with uniform mild AUROC loss. "
        "All conclusions are post hoc and exploratory; any mechanistic claim for the paper "
        "needs a prospective test (X6 house rule).")

    path = os.path.join(args.out_dir, "pilot1_autopsy_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""TEMPERATURE AUDIT (recommendation F of the 2026-09-11 evaluation; DIAGNOSTIC,
post-outcome). Two facts to state and one variant to evaluate.

Facts: (1) the four-shift extractors compute Energy on raw logits, T = 1
(`pilot0.scores.head_scores`); (2) the companion benchmark's main pipeline fits
a temperature by validation cross-entropy and evaluates Energy as
T log sum exp(z/T) (`src/csfs/temperature_scaling.py`, `src/csfs/base_detectors.py`).
So every Energy number in the campaign and the replication is the T = 1
Energy, not the benchmark's temperature-scaled Energy.

Variant, from the stored per-example logits of the version-2 records (no
forward pass): a CROSS-FITTED temperature on the ID test set (fit on the
even-index half by minimizing NLL over log T, evaluate on the odd half, and
vice versa; the OOD examples are scored with the fold's T and the two
half-AUROCs averaged), plus the diagnostic full-ID fit. Records per
checkpoint: fitted T per fold, NLL and Brier before/after, centered logit
spread s_z and s_z/T, the common-offset distribution (mean and sd of the
per-example mean logit) on ID and on each OOD set, and per set the T = 1
Energy AUROC (stored), the cross-fitted temperature-scaled Energy AUROC,
the CTM AUROC, and whether the Energy-vs-CTM winner flips. The benchmark's
own fitted temperatures (validation split) are NOT available locally; if
they are synced they replace the cross-fit as the primary variant.
MLS and CTM are invariant to a positive per-model temperature; MSR and
Energy are not.

Usage (from code/): python rn18_handoff_replication/temperature_energy_audit.py
Output: rn18_handoff_replication/outputs/temperature_energy_audit.json/.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from pilot0.scores import auroc

OUT = Path("rn18_handoff_replication/outputs")
V2 = OUT / "fourshift_v2_rn18"
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
LABEL = "TEMPERATURE AUDIT (post-outcome, diagnostic; cross-fitted T on the ID test set; benchmark validation temperatures not available locally)"


def nll(z: np.ndarray, y: np.ndarray, T: float) -> float:
    zs = z / T
    return float(np.mean(logsumexp(zs, axis=1) - zs[np.arange(len(y)), y]))


def brier(z: np.ndarray, y: np.ndarray, T: float) -> float:
    p = np.exp(z / T - logsumexp(z / T, axis=1, keepdims=True)); oh = np.zeros_like(p); oh[np.arange(len(y)), y] = 1
    return float(np.mean(((p - oh) ** 2).sum(1)))


def fit_T(z: np.ndarray, y: np.ndarray) -> float:
    r = minimize_scalar(lambda lt: nll(z, y, np.exp(lt)), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def energy_T(z: np.ndarray, T: float) -> np.ndarray:
    return T * logsumexp(z / T, axis=1)


def audit_record(p: Path) -> dict:
    rec = json.loads(p.read_text()); z = np.load(V2 / f"{p.stem}.npz")
    zid, y = z["id__logits"].astype(np.float64), z["id__labels"].astype(np.int64)
    ids = z["id__ids"]; fold = (ids % 2).astype(bool)                    # even canonical ids -> fold A (False), odd -> fold B (True)
    T = {"A": fit_T(zid[~fold], y[~fold]), "B": fit_T(zid[fold], y[fold]), "full_diagnostic": fit_T(zid, y)}
    sz = float(np.sqrt(np.mean(np.var(zid, axis=1))))
    row = {"slug": rec["slug"], "source": rec["source"], "component": rec["component"], "n_id": int(len(y)),
           "T": T, "nll_before": nll(zid, y, 1.0), "nll_after_crossfit": float(np.mean([nll(zid[fold], y[fold], T["A"]), nll(zid[~fold], y[~fold], T["B"])])),
           "brier_before": brier(zid, y, 1.0), "brier_after_crossfit": float(np.mean([brier(zid[fold], y[fold], T["A"]), brier(zid[~fold], y[~fold], T["B"])])),
           "logit_spread_sz": sz, "logit_spread_sz_over_T": sz / T["full_diagnostic"],
           "common_offset_id": {"mean": float(zid.mean(1).mean()), "sd": float(zid.mean(1).std())}, "sets": {}}
    eid1 = energy_T(zid, 1.0)
    ctm_id = z["id__score__CTM"]
    for s in SHIFTS:
        zo = z[f"set__{s}__logits"].astype(np.float64); ctm_o = z[f"set__{s}__score__CTM"]
        a1 = auroc(eid1, energy_T(zo, 1.0))
        aT = float(np.mean([auroc(energy_T(zid[fold], T["A"]), energy_T(zo, T["A"])), auroc(energy_T(zid[~fold], T["B"]), energy_T(zo, T["B"]))]))
        aFull = auroc(energy_T(zid, T["full_diagnostic"]), energy_T(zo, T["full_diagnostic"]))
        aC = auroc(ctm_id, ctm_o)
        stored = rec["v2"]["ood_unrounded"][s]["auroc_id_vs_ood"]["Energy"]
        row["sets"][s] = {"energy_T1_recomputed": a1, "energy_T1_stored": stored, "energy_T_crossfit": aT, "energy_T_fullfit_diagnostic": aFull, "ctm": aC,
                          "winner_T1": "CTM" if aC > a1 else "Energy", "winner_Tcrossfit": "CTM" if aC > aT else "Energy",
                          "gap_T1": aC - a1, "gap_Tcrossfit": aC - aT, "common_offset_ood": {"mean": float(zo.mean(1).mean()), "sd": float(zo.mean(1).std())}}
    return row


def main() -> None:
    rows = [audit_record(p) for p in sorted(V2.glob("*.json")) if not p.name.startswith("FAILED_")]
    assert all(abs(v["energy_T1_recomputed"] - v["energy_T1_stored"]) < 1e-9 for r in rows for v in r["sets"].values()), "stored T=1 Energy AUROC not reproduced"
    by_src = {}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        cells = [v for r in rs for v in r["sets"].values()]
        by_src[src] = {"n_checkpoints": len(rs), "T_crossfit_range": [min(min(r["T"]["A"], r["T"]["B"]) for r in rs), max(max(r["T"]["A"], r["T"]["B"]) for r in rs)],
                       "T_fold_disagreement_max": max(abs(r["T"]["A"] - r["T"]["B"]) for r in rs),
                       "nll_before_after": [float(np.mean([r["nll_before"] for r in rs])), float(np.mean([r["nll_after_crossfit"] for r in rs]))],
                       "energy_auroc_T1_mean": float(np.mean([c["energy_T1_recomputed"] for c in cells])), "energy_auroc_Tcrossfit_mean": float(np.mean([c["energy_T_crossfit"] for c in cells])),
                       "mean_abs_change": float(np.mean([abs(c["energy_T_crossfit"] - c["energy_T1_recomputed"]) for c in cells])),
                       "winner_flips": int(sum(c["winner_T1"] != c["winner_Tcrossfit"] for c in cells)), "n_cells": len(cells),
                       "ctm_wins_T1": int(sum(c["winner_T1"] == "CTM" for c in cells)), "ctm_wins_Tcrossfit": int(sum(c["winner_Tcrossfit"] == "CTM" for c in cells))}
    rep = {"label": LABEL, "facts": {"campaign_energy_temperature": 1.0, "benchmark_pipeline": "fitted T by validation NLL; Energy = T logsumexp(z/T)",
                                     "fitter_positivity": "the benchmark fitter optimizes T directly without a positivity constraint (implementation note, no invalid value observed here)"},
           "by_source": by_src, "records": rows}
    (OUT / "temperature_energy_audit.json").write_text(json.dumps(rep, indent=1, default=str))
    (OUT / "temperature_energy_audit.md").write_text("# Temperature audit\n\n```\n" + json.dumps({k: v for k, v in rep.items() if k != "records"}, indent=1, default=str) + "\n```\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "records"}, indent=1, default=str))


if __name__ == "__main__":
    main()

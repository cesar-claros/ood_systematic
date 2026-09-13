"""BOUNDARY AUDIT of the closed forms on the LITERAL model (recommendation C
of the 2026-09-11 evaluation): analytic AUROCs against Monte Carlo on the
exact ETF-EXACT-v2 construction (not the historical decoder) in the
regions where decisions are made.

Regions: (1) the MLS norm-confound boundary, gamma a in {0.8 ... 1.25} on the
canonical single-alignment profile; (2) the Energy-versus-CTM boundary, the
gamma at which the analytic gap crosses zero, and +-10 percent around it.
Grid: (C, D) in {(10, 128), (100, 512)}; s = R in {6, 16}; a in {0.4, 0.8};
theta in {0, 20 degrees}; L_par = 10; rho = 1.

Per configuration: analytic and Monte Carlo AUROC for MLS, Energy, CTM
(mean prototypes) and Mahalanobis (adaptive MC to SE <= 0.0025, the frozen
audit's rule); absolute error; error relative to the Monte Carlo
Energy-minus-CTM gap; ID and OOD argmax switching probabilities from a
separate 20,000-draw sample with the union bound (C-1) Phi(-s sqrt(C/(2(C-1))))
for ID; the MLS boundary displacement (gamma a at which MC AUROC crosses 1/2
by interpolation on the ladder, against the predicted 1); the Energy-CTM
boundary displacement (MC zero crossing of the gap against the analytic
one, by interpolation over the three gammas).

Usage (from code/): python rn18_handoff_replication/theory/boundary_audit.py [--quick]
Output: rn18_handoff_replication/outputs/boundary_audit.json/.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_ROOT))

from mc_phase_audit import analytic_aurocs, mc_aurocs
from rn18_handoff_replication.theory.etf_constructor import construct

OUT = Path("rn18_handoff_replication/outputs")
GA = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)
SCORES = {"MLS": "MLS", "Energy": "Energy", "CTM": "CTM_mean", "Maha": "Maha"}


def literal_model(C, D, s, a, gamma, theta, L=10.0, rho=1.0, seed=0) -> dict:
    m = construct(C, D, R=s, a=a, gamma=gamma, rho=rho, theta=theta, L_par=L, seed=seed)
    return {"means": m["means"], "w": m["W"], "b": m["b"], "cov_id": m["cov_id"], "cov_ood": m["cov_ood"], "m_ood": m["m_ood"], "sigma": 1.0, "class_freq": m["class_freq"]}


def switching(model: dict, n: int = 20000, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed); C, D = model["means"].shape
    y = rng.integers(0, C, n); h_id = model["means"][y] + rng.standard_normal((n, D)); h_ood = model["m_ood"] + np.sqrt(model["cov_ood"][0, 0]) * rng.standard_normal((n, D))
    g_id = h_id @ model["w"].T + model["b"]; g_ood = h_ood @ model["w"].T + model["b"]
    return {"id_switch": float(np.mean(g_id.argmax(1) != y)), "ood_switch_from_class1": float(np.mean(g_ood.argmax(1) != 0))}


def union_bound(C, s) -> float:
    return float((C - 1) * norm.cdf(-s * np.sqrt(C / (2 * (C - 1)))))


def audit_point(C, D, s, a, gamma, theta, region, seed) -> dict:
    model = literal_model(C, D, s, a, gamma, theta)
    ana = analytic_aurocs(model); mc, se, n = mc_aurocs(model, seed)
    row = {"C": C, "D": D, "s": s, "a": a, "gamma": gamma, "ga": gamma * a, "theta_deg": float(np.degrees(theta)), "region": region, "n_mc": n, "analytic": {}, "mc": {}, "abs_err": {}}
    for k, key in SCORES.items():
        row["analytic"][k] = ana[key]; row["mc"][k] = mc[key]; row["abs_err"][k] = abs(ana[key] - mc[key])
    row["gap_mc"] = mc["CTM_mean"] - mc["Energy"]; row["gap_analytic"] = ana["CTM_mean"] - ana["Energy"]
    row["gap_err"] = abs(row["gap_analytic"] - row["gap_mc"]); row["gap_err_over_abs_gap"] = row["gap_err"] / max(abs(row["gap_mc"]), 1e-6)
    row["sign_agree"] = bool(np.sign(row["gap_analytic"]) == np.sign(row["gap_mc"]))
    row.update(switching(model)); row["id_union_bound"] = union_bound(C, s)
    return row


def energy_ctm_crossing(C, D, s, a, theta, L=10.0) -> float | None:
    gs = np.geomspace(0.05, 5.0, 60) / a
    vals = []
    for g in gs:
        ana = analytic_aurocs(literal_model(C, D, s, a, g, theta, L)); vals.append(ana["CTM_mean"] - ana["Energy"])
    vals = np.array(vals)
    for i in range(len(gs) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]) and vals[i] != 0:
            return float(gs[i] + (gs[i + 1] - gs[i]) * (-vals[i]) / (vals[i + 1] - vals[i]))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0]); ap.add_argument("--quick", action="store_true"); a_ = ap.parse_args()
    blocks = [(10, 128)] if a_.quick else [(10, 128), (100, 512)]
    S = (6, 16); A = (0.4, 0.8); TH = (0.0, np.deg2rad(20))
    rows, t0, k = [], time.time(), 0
    for (C, D) in blocks:
        for s in S:
            for a in A:
                for th in TH:
                    for ga in (GA if not a_.quick else (0.9, 1.0, 1.1)):
                        rows.append(audit_point(C, D, s, a, ga / a, th, "mls_boundary", seed=k)); k += 1
                    gx = energy_ctm_crossing(C, D, s, a, th)
                    if gx is not None:
                        for f in (0.9, 1.0, 1.1):
                            rows.append(audit_point(C, D, s, a, gx * f, th, "energy_ctm_boundary", seed=k)); k += 1
                    print(f"[boundary] C={C} D={D} s={s} a={a} theta={np.degrees(th):.0f}: {len(rows)} points ({time.time() - t0:.0f}s)", flush=True)
    # displacements
    disp = []
    for (C, D) in blocks:
        for s in S:
            for a in A:
                for th in TH:
                    lad = sorted([r for r in rows if r["region"] == "mls_boundary" and r["C"] == C and r["s"] == s and r["a"] == a and abs(r["theta_deg"] - np.degrees(th)) < 1e-6], key=lambda r: r["ga"])
                    x = np.array([r["ga"] for r in lad]); y = np.array([r["mc"]["MLS"] - 0.5 for r in lad])
                    cross = None
                    for i in range(len(x) - 1):
                        if np.sign(y[i]) != np.sign(y[i + 1]) and y[i] != 0:
                            cross = float(x[i] + (x[i + 1] - x[i]) * (-y[i]) / (y[i + 1] - y[i])); break
                    ec = [r for r in rows if r["region"] == "energy_ctm_boundary" and r["C"] == C and r["s"] == s and r["a"] == a and abs(r["theta_deg"] - np.degrees(th)) < 1e-6]
                    ec = sorted(ec, key=lambda r: r["gamma"]); mc_cross = None
                    if len(ec) == 3:
                        gx = ec[1]["gamma"]; yv = np.array([r["gap_mc"] for r in ec]); xv = np.array([r["gamma"] for r in ec])
                        for i in range(2):
                            if np.sign(yv[i]) != np.sign(yv[i + 1]) and yv[i] != 0:
                                mc_cross = float(xv[i] + (xv[i + 1] - xv[i]) * (-yv[i]) / (yv[i + 1] - yv[i])); break
                        disp.append({"C": C, "D": D, "s": s, "a": a, "theta_deg": float(np.degrees(th)), "mls_boundary_ga_mc": cross, "mls_boundary_ga_predicted": 1.0,
                                     "energy_ctm_gamma_analytic": gx, "energy_ctm_gamma_mc": mc_cross, "energy_ctm_rel_displacement": (None if mc_cross is None else (mc_cross - gx) / gx)})
                    else:
                        disp.append({"C": C, "D": D, "s": s, "a": a, "theta_deg": float(np.degrees(th)), "mls_boundary_ga_mc": cross, "mls_boundary_ga_predicted": 1.0, "energy_ctm_gamma_analytic": None})
    summ = {"n_points": len(rows)}
    for reg in ("mls_boundary", "energy_ctm_boundary"):
        rs = [r for r in rows if r["region"] == reg]
        if not rs: continue
        summ[reg] = {"n": len(rs), **{f"max_abs_err_{k}": max(r["abs_err"][k] for r in rs) for k in SCORES}, **{f"frac_within_0.01_{k}": float(np.mean([r["abs_err"][k] <= 0.01 for r in rs])) for k in SCORES},
                     "max_gap_err": max(r["gap_err"] for r in rs), "median_gap_err_over_abs_gap": float(np.median([r["gap_err_over_abs_gap"] for r in rs])), "sign_agreement": float(np.mean([r["sign_agree"] for r in rs])),
                     "id_switch_max": max(r["id_switch"] for r in rs), "ood_switch_max": max(r["ood_switch_from_class1"] for r in rs), "id_switch_vs_union_bound_max_ratio": max(r["id_switch"] / max(r["id_union_bound"], 1e-300) for r in rs)}
    summ["mls_boundary_displacement_max_abs"] = max((abs(d["mls_boundary_ga_mc"] - 1.0) for d in disp if d.get("mls_boundary_ga_mc") is not None), default=None)
    summ["energy_ctm_rel_displacement_max_abs"] = max((abs(d["energy_ctm_rel_displacement"]) for d in disp if d.get("energy_ctm_rel_displacement") is not None), default=None)
    rep = {"label": "BOUNDARY AUDIT on the literal ETF-EXACT-v2 model (analytic vs Monte Carlo)", "grid": {"blocks": blocks, "s": S, "a": A, "theta_deg": [0, 20], "ga_ladder": GA, "L_par": 10.0, "rho": 1.0}, "summary": summ, "displacements": disp, "points": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "boundary_audit.json").write_text(json.dumps(rep, indent=1, default=str))
    (OUT / "boundary_audit.md").write_text("# Boundary audit (literal model)\n\n```\n" + json.dumps({"summary": summ, "displacements": disp}, indent=1, default=str) + "\n```\n")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()

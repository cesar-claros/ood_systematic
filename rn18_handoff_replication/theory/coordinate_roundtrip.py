"""Coordinate round-trip through the HISTORICAL decoder (plan v3, carried
v2 section 6.2): construct an exactly compatible ETF model
(ETF-EXACT-v2), measure its coordinates exactly, feed them to the frozen
`frozen_cfg -> build_config_model` path, and measure what the decoder
actually realizes. The three review discrepancies are quantified per
grid point: (1) selected-class versus maximum alignment, (2) own-class
versus full-span head rotation, (3) measured target logit versus the
unrotated scale. Active frozen clamps are recorded. The historical
decoder is NOT modified.

Also writes the proof-and-approximation ledger (v2 section 6.6) as
theorem_code_ledger.md.

Usage (from code/): python rn18_handoff_replication/theory/coordinate_roundtrip.py
Outputs: rn18_handoff_replication/theory/coordinate_roundtrip_report.json, theorem_code_ledger.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_ROOT))

from mc_phase_audit import build_config_model
from rn18_handoff_replication.theory.etf_constructor import construct, measure
from tail_space_audit import frozen_cfg

OUT = Path("rn18_handoff_replication/theory")
GRID = [(C, 512, R, th, a, g, rho, 10.0)
        for C in (10, 100) for R in (5.0, 10.0, 24.0)
        for th in (0.0, np.deg2rad(6.0), np.deg2rad(30.0))
        for a in (0.05, 0.3, 0.9) for g in (0.5, 1.0) for rho in (0.5, 2.0)]


def decoder_model(C, D, cfg) -> dict:
    m = build_config_model(C, D, cfg, seed=0)
    return {"C": C, "D": D, "means": m["means"], "W": m["w"], "b": m["b"],
            "cov_id": m["cov_id"], "cov_ood": m["cov_ood"], "m_ood": m["m_ood"],
            "class_freq": m["class_freq"]}


def active_clamps(s, theta_deg, L, eta, gamma, a, rho) -> list[str]:
    out = []
    if s < 3.0: out.append("s>=3")
    if theta_deg > 85: out.append("theta<=85")
    if L < 1e-3: out.append("logit>=1e-3")
    if eta > 0.5: out.append("eta<=0.5")
    if gamma * a < 1e-4: out.append("ga>=1e-4")
    if not (1e-3 <= a <= 0.999): out.append("a in [1e-3,0.999]")
    if rho < 0.05: out.append("rho>=0.05")
    return out


def main() -> None:
    rows = []
    for C, D, R, th, a, g, rho, L in GRID:
        m = construct(C, D, R, a, g, rho, th, L)
        x = measure(m)                                  # exact requested coordinates
        s, theta_deg, eta = x["s_dict"], np.degrees(x["theta_from_sd"]) if th > 0 else 0.0, x["radius_cv"]
        cfg = frozen_cfg(s, theta_deg, x["logit_scale"], eta, x["gamma"], x["a_max"], x["rho"])
        dm = decoder_model(C, D, cfg)
        y = measure(dm)                                 # realized by the decoder
        rows.append({
            "request": dict(C=C, D=D, R=R, theta_deg=round(float(np.degrees(th)), 3), a=a,
                            gamma=g, rho=rho, L_par=L),
            "active_clamps": active_clamps(s, theta_deg, x["logit_scale"], eta, x["gamma"],
                                           x["a_max"], x["rho"]),
            "realized": {k: round(float(y[k]), 6) for k in
                         ("R", "radius_cv", "s_dict", "a_max", "a_other_max", "gamma", "rho",
                          "logit_scale", "L_base", "theta_from_sd", "head_complement_fraction")},
            "discrepancy": {
                "alignment_selected_vs_max": round(float(y["a_max"] - a), 6),
                "logit_measured_vs_unrotated_scale": round(float(y["logit_scale"] - L), 6),
                "logit_over_L_par": round(float(y["logit_scale"] / L), 6),
                "theta_realized_minus_requested_deg": round(float(np.degrees(y["theta_from_sd"]) - np.degrees(th)), 6),
                "radius_cv_injected_by_eta_draw": round(float(y["radius_cv"]), 6),
                "s_dict_realized_over_requested": round(float(y["s_dict"] / R), 6),
                "head_complement_fraction_realized": round(float(y["head_complement_fraction"]), 6),
                "head_complement_fraction_requested": round(float(np.sin(th) ** 2), 6)},
        })
    disc = np.array([[r["discrepancy"]["alignment_selected_vs_max"],
                      r["discrepancy"]["logit_over_L_par"],
                      r["discrepancy"]["theta_realized_minus_requested_deg"],
                      r["discrepancy"]["s_dict_realized_over_requested"]] for r in rows])
    summary = {"n_grid": len(rows),
               "alignment_selected_vs_max": {"min": float(disc[:, 0].min()), "max": float(disc[:, 0].max())},
               "logit_over_L_par": {"min": float(disc[:, 1].min()), "max": float(disc[:, 1].max()),
                                    "note": "decoder scale g = L/R with rotated rows gives L cos(theta) at the unperturbed radius; eta draws perturb further"},
               "theta_realized_minus_requested_deg": {"min": float(disc[:, 2].min()), "max": float(disc[:, 2].max()),
                                                      "note": "own-class rotation by a random direction per row versus the constructor's full-span complement"},
               "s_dict_realized_over_requested": {"min": float(disc[:, 3].min()), "max": float(disc[:, 3].max()),
                                                  "note": "eta_std = radius_cv clamp [0, 0.5]; the exact constructor has zero spread so eta = 0 here"},
               "clamps_ever_active": sorted({c for r in rows for c in r["active_clamps"]})}
    (OUT / "coordinate_roundtrip_report.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=1))
    ledger = """# Theorem / code ledger (plan v3, carried v2 section 6.6)

| Result | Assumptions | Class | Implementation | Independent check |
|---|---|---|---|---|
| AUGRC = pi^2/2 + pi(1-pi)(1 - A^f); Delta^G = pi(1-pi) Delta^F | common examples, residuals, half-credit ties | exact identity (established, Traub et al.) | `set_outcomes` + analysis bridge test | float64 identity to 1e-10 (test 1) |
| NC1 low-rank form sum_i u_i' Sigma_W u_i / s_i^2 with cutoff sqrt(1e-6) s_max | Sigma_B = MM'/C | exact identity | `papyan_metrics` (direct pinv) | phase-1 bootstrap arithmetic asserted equal at unit weights on 96 checkpoints |
| s_dict = (C-1)/sqrt(C NC1) = R/sigma | isotropic equal-radius exact ETF only | exact identity under stated model | `record_params` | ETF-EXACT-v2 `check()` on 5 grid points |
| ETF-EXACT-v2 constructor identities (rank, cosines, alignment a and -a/(C-1), L_par, L_base = t R, self-duality 2(1-cos theta), complement fraction sin^2 theta) | D >= 2C+1, a in [0,1], theta in [0, pi/2) | exact identities | `etf_constructor.construct` | `measure()` independent of the constructor's algebra |
| Historical decoder realizes requested coordinates | none | approximation with documented discrepancies | `build_config_model` (unchanged) | this round-trip report: selected-class vs max alignment, own-class rotation, L cos(theta) scale, eta draws |
| Taylor moments m_f + tr(H Sigma)/2, grad' Sigma grad + tr(H Sigma H Sigma)/2 | Gaussian component, smooth fixed branch | exact for the quadratic surrogate, approximation for the score | `gaussian_diagnostics.taylor_moments` | finite-difference gradients/Hessians; MC agreement (test 5, 6) |
| Energy AUROC via binormal mixture of Taylor moments | fixed softmax branch, Gaussian components | approximation (curvature + distribution) | `A0/A1-TAYLOR` | G0/G1-MC discrepancy reported separately from observed |
| Max-cosine (CTM) AUROC via fixed-branch Taylor | no branch switching, r bounded away from 0 | approximation (branch switching + norm fluctuation) | `A0/A1-TAYLOR`, undefined at ties | switching rate and norm concentration diagnostics reported |
| Fixed-index MLS argmax bound | ID and OOD switching probabilities | bound (coupling), NOT an exact error rate | manuscript statement | qualification stated in text |
| Mahalanobis normal-CDF expression | fixed-prototype model | chi-square-difference approximation | manuscript statement | labeled approximate |
| Delete-family jackknife SE and t reference | approximately independent families | approximate inference candidate | `icml_campaign_analysis`-style jackknife | qualified only in the enumerated simulations (phase 4) |
| Partial conjunction 2 p_(3) | valid marginal p-values | exact under stated validity | retired with DIST | n/a |

Every approximate row is written as approximate in the manuscript; no entry claims exactness outside its stated model.
"""
    (OUT / "theorem_code_ledger.md").write_text(ledger)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

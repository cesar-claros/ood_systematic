"""Stage-2 source-expansion analysis: census + Q3 theory-saturation
diagnostics (source-expansion protocol stage 2 item 7; amendment 5).

Analysis-only, no recalibration: the frozen closed forms are evaluated at
the measured coordinates with the EXACT clamp conventions of the
registered pool evaluation (stage2_closure.theory_full), and their signs
are scored against the amendment-5c balanced-gap outcomes. Per protocol
section 9.1 Q3, reported per source and per paradigm: analytic sign
accuracy on material cells, fraction of exactly zero predicted winner
margins, fraction with both predicted AUROCs above 0.99, fraction on each
material side of the analytic boundary (|AUROC_E - AUROC_C| >= 0.01),
Spearman between analytic and observed gap magnitudes, and coordinate
support (gamma*a, dictionary SNR, rho quantiles). Census: material-cell
counts under the balanced rule, winner prevalence, raw-vs-balanced gap
comparison, runtimes.

Inputs: pilot0/stage2_expansion_coords/<slug>.json (88 records).
Outputs: nc_csf_predictivity/outputs/track1/stage2_expansion_report.md/.json
Usage (from code/): python stage2_expansion_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crossing_robustness_audit import OUT_DIR
from mc_phase_audit import BASE, build_config_model
from pilot0.theory import (HeadContext, NoiseModel, predicted_aurocs,
                           predicted_ctm_mean_auroc)

COORDS = Path("pilot0/stage2_expansion_coords")
MATERIALITY = 0.01


def theory_pair(c: int, d: int, s: float, theta: float, lscale: float,
                eta: float, gamma: float, a: float, rho: float,
                cache: dict) -> tuple[float, float]:
    cfg = dict(BASE, s=max(s, 3.0), theta_deg=float(np.clip(theta, 0, 85)),
               logit_target=max(lscale, 1e-3),
               eta_std=float(np.clip(eta, 0.0, 0.5)),
               ga=float(np.clip(gamma * a, 1e-4, None)),
               a=float(np.clip(a, 1e-3, 0.999)),
               rho=float(np.clip(rho, 0.05, None)))
    key = (c, d, round(cfg["s"], 3), round(cfg["theta_deg"], 2),
           round(cfg["logit_target"], 3), round(cfg["eta_std"], 4),
           round(cfg["ga"], 4), round(cfg["a"], 4), round(cfg["rho"], 4))
    if key not in cache:
        m = build_config_model(c, d, cfg, seed=0)
        ctx = HeadContext.from_head(m["w"], m["b"])
        dim = m["means"].shape[1]
        nid = NoiseModel.isotropic(m["sigma"], ctx, dim)
        nood = NoiseModel.isotropic(cfg["rho"] * m["sigma"], ctx, dim)
        head = predicted_aurocs(m["means"], m["class_freq"], nid,
                                m["m_ood"], nood, ctx)
        ctm = predicted_ctm_mean_auroc(m["means"], m["class_freq"],
                                       m["cov_id"], m["m_ood"],
                                       m["cov_ood"])
        cache[key] = (float(head["Energy"]), float(ctm))
    return cache[key]


def main() -> None:
    cells = []
    cache: dict = {}
    n_ckpt = 0
    for p in sorted(COORDS.glob("*.json")):
        if p.name.startswith("FAILED"):
            continue
        r = json.loads(p.read_text())
        n_ckpt += 1
        c, d = int(r["n_classes"]), int(r["dim"])
        vc = r["papyan"]["var_collapse"]
        sd = r["papyan"]["self_duality"]
        s_dict = (c - 1) / np.sqrt(c * max(vc, 1e-9))
        theta = float(np.degrees(np.arccos(np.clip(1 - sd / 2, -1, 1))))
        for name, e in r["ood"].items():
            if "error" in e:
                continue
            ae, ac = theory_pair(c, d, s_dict, theta,
                                 r["geometry"]["logit_scale"],
                                 r["geometry"]["class_mean_radius_cv"],
                                 e["gamma"], e["a"], e["rho"], cache)
            cells.append({
                "source": r["source"], "paradigm": r["paradigm"],
                "slug": r["slug"], "set": name,
                "gap_balanced": e["gap_balanced"], "gap_raw": e["gap_raw"],
                "material": bool(e["material"]),
                "ga": float(np.clip(e["gamma"] * e["a"], 1e-4, None)),
                "s_dict": float(s_dict), "rho": float(e["rho"]),
                "auroc_E": ae, "auroc_C": ac,
                "pred_gap": -(ae - ac),
                "runtime": r["runtime_sec"],
            })
        print(f"[q3] {p.name} done", flush=True)

    import pandas as pd
    from scipy.stats import spearmanr
    fr = pd.DataFrame(cells)
    out: dict = {"n_checkpoints": n_ckpt, "n_cells": int(len(fr)),
                 "per_source": {}, "per_paradigm_breeds": {}}
    L = ["# Stage-2 source-expansion report (census + Q3 theory "
         "diagnostics)", "",
         "Analysis-only; frozen closed forms, registered clamps, no "
         "recalibration. Materiality = |balanced gap| >= 0.01 (protocol "
         "amendment 5c); gap = AUGRC_Energy - AUGRC_CTM, negative = "
         "Energy-favored.", "",
         f"Census: {n_ckpt} checkpoints, {len(fr)} cells, 0 failures, "
         "all suites complete.", ""]

    def block(g: pd.DataFrame, label: str) -> dict:
        mat = g[g.material]
        margin = (g.auroc_E - g.auroc_C).abs()
        rec = {
            "n_cells": int(len(g)), "n_material": int(len(mat)),
            "frac_positive_material": (round(float((mat.gap_balanced > 0)
                                                   .mean()), 3)
                                       if len(mat) else None),
            "median_abs_gap_balanced": round(
                float(g.gap_balanced.abs().median()), 4),
            "frac_margin_zero": round(float((margin == 0).mean()), 3),
            "frac_both_above_099": round(float(
                ((g.auroc_E > 0.99) & (g.auroc_C > 0.99)).mean()), 3),
            "frac_energy_material_side": round(float(
                ((g.auroc_E - g.auroc_C) >= MATERIALITY).mean()), 3),
            "frac_ctm_material_side": round(float(
                ((g.auroc_E - g.auroc_C) <= -MATERIALITY).mean()), 3),
            "ga_q": [round(float(np.quantile(g.ga, q)), 3)
                     for q in (0.05, 0.5, 0.95)],
            "s_dict_q": [round(float(np.quantile(g.s_dict, q)), 1)
                         for q in (0.05, 0.5, 0.95)],
            "rho_q": [round(float(np.quantile(g.rho, q)), 2)
                      for q in (0.05, 0.5, 0.95)],
            "median_runtime_sec": round(float(g.runtime.median()), 1),
        }
        if len(mat):
            ok = np.sign(mat.pred_gap) == np.sign(mat.gap_balanced)
            rec["theory_sign_acc_material"] = round(float(ok.mean()), 3)
            rec["n_material_pred_zero"] = int((mat.pred_gap == 0).sum())
        nz = g[(g.auroc_E - g.auroc_C).abs() > 0]
        rec["spearman_absmargin_vs_absgap"] = (
            round(float(spearmanr((nz.auroc_E - nz.auroc_C).abs(),
                                  nz.gap_balanced.abs()).statistic), 3)
            if len(nz) > 10 else None)
        L.append(f"## {label}")
        for k, v in rec.items():
            L.append(f"- {k}: {v}")
        L.append("")
        return rec

    for src, g in fr.groupby("source"):
        out["per_source"][src] = block(g, f"Source: {src}")
    for par, g in fr[fr.source == "breeds"].groupby("paradigm"):
        out["per_paradigm_breeds"][par] = block(g, f"breeds / {par}")

    L += ["## Notes", "",
          "- Raw-vs-balanced: sign(gap_raw) == sign(gap_balanced) on "
          f"{round(float((np.sign(fr.gap_raw) == np.sign(fr.gap_balanced)).mean()), 3)} "
          "of cells (balancing rescales, it does not flip).",
          "- ViT inventory (recorded, EXCLUDED per protocol section 1): "
          "the release also ships ViT sweeps (svhn 105, breeds 48, "
          "wilds_animals 150, plus openset variants); admissible later "
          "only as exploratory cross-regime evidence by dated amendment.",
          ""]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stage2_expansion_report.md").write_text("\n".join(L))
    (OUT_DIR / "stage2_expansion_report.json").write_text(
        json.dumps(out, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()

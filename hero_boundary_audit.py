"""R4 (audit #5, section 13): pair-specific Monte Carlo audit of exactly the
contours displayed in the hero figure.

Panels and pairs (hero_phase_diagram.py): A = Energy vs CTM (class-mean; MC
counterpart CTM_mean) over gamma*a x s (geomspace 8..130); B = MLS vs Maha
over gamma*a x theta_w (0..60 deg); C = 100, D = 512, isotropic, BASE config.

Frozen rules (claim contract 2026-08-24, BEFORE this audit ran):
  - conditioning: local analytic slope m = |d gap / d(gamma a)| at the
    boundary contour; well-conditioned iff m >= M_MIN = 0.2, which bounds the
    implied displacement by TOL/M_MIN = 0.05 gamma-a units;
  - a contour point is DISPLAYED SHARP iff well-conditioned AND |MC
    displacement| <= 0.05 AND bracket pixels sign-agree at MC precision;
    otherwise it is masked or drawn as a band;
  - all boundary points are reported, with displacement and sign performance
    both before and after the conditioning filter.

The audited contour is the drawn "boundary" level gap = -TOL (= -0.01, the
material-advantage boundary); the -0.5/-0.1 depth guides get analytic slopes
only. Sign agreement is additionally sampled on a fixed pixel subgrid (every
4th pixel per axis).

Usage (from code/): python hero_boundary_audit.py [--quick]
Outputs: nc_csf_predictivity/outputs/track1/hero_boundary_audit_report.md/.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from hero_phase_diagram import C_MODEL, D_MODEL, analytic_pixel
from mc_phase_audit import BASE, OUT_DIR, TOL, build_config_model, mc_aurocs

M_MIN = 0.2
DISP_MAX = TOL / M_MIN
SEED_BASE = 777

PANELS = {
    "A": {"pair": ("Energy", "CTM"), "mc_pair": ("Energy", "CTM_mean"),
          "y_axis": "s", "y_grid": lambda n: np.geomspace(8, 130, n)},
    "B": {"pair": ("MLS", "Maha"), "mc_pair": ("MLS", "Maha"),
          "y_axis": "theta", "y_grid": lambda n: np.linspace(0, 60, n)},
}


def model_at(ga: float, y: float, y_axis: str, seed: int = 0) -> dict:
    cfg = dict(BASE, ga=float(ga))
    if y_axis == "s":
        cfg["s"] = float(y)
    else:
        cfg["theta_deg"] = float(y)
    cfg.update({"C": C_MODEL, "D": D_MODEL, "family": "hero_audit",
                "cluster": None, "draw": 0})
    return build_config_model(C_MODEL, D_MODEL, cfg, seed=seed)


def analytic_gap(pair, ga, y, y_axis) -> float:
    au = analytic_pixel(model_at(ga, y, y_axis))
    return float(au[pair[0]] - au[pair[1]])


def mc_gap(mc_pair, ga, y, y_axis, seed) -> tuple[float, float]:
    aucs, ses, _ = mc_aurocs(model_at(ga, y, y_axis), seed=seed)
    g = float(aucs[mc_pair[0]] - aucs[mc_pair[1]])
    se = float(np.hypot(ses[mc_pair[0]], ses[mc_pair[1]]))
    return g, se


def row_contour(ga_grid, gaps, level) -> list[tuple[float, float]]:
    """(gamma_a, slope) for each crossing of `level` along the row."""
    out = []
    for j in range(len(ga_grid) - 1):
        a, b = gaps[j] - level, gaps[j + 1] - level
        if a == 0 or (a < 0) != (b < 0):
            f = a / (a - b) if a != b else 0.0
            ga_c = ga_grid[j] + f * (ga_grid[j + 1] - ga_grid[j])
            slope = abs((gaps[j + 1] - gaps[j]) / (ga_grid[j + 1] - ga_grid[j]))
            out.append((float(ga_c), float(slope)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    n_ga, n_y = (16, 10) if args.quick else (41, 26)
    ga_grid = np.linspace(0.2, 2.2, n_ga)

    result: dict = {"m_min": M_MIN, "disp_max": DISP_MAX, "tol": TOL,
                    "grid": [n_ga, n_y], "panels": {}}
    lines = ["# Hero-figure pair-specific boundary audit (R4)", "",
             f"Frozen rules: m_min = {M_MIN} (gap units per gamma-a unit), "
             f"displacement bound tol/m_min = {DISP_MAX:.2f}; boundary level = -tol = {-TOL}; "
             f"grid {n_ga}x{n_y}; MC SE target 0.0025 per score; seed base {SEED_BASE}.", ""]
    seed = SEED_BASE

    for pname, spec in PANELS.items():
        y_grid = spec["y_grid"](n_y)
        pair, mc_pair, y_axis = spec["pair"], spec["mc_pair"], spec["y_axis"]
        gap_rows = np.array([[analytic_gap(pair, ga, y, y_axis)
                              for ga in ga_grid] for y in y_grid])

        boundary_pts = []
        for i, y in enumerate(y_grid):
            for ga_c, slope in row_contour(ga_grid, gap_rows[i], -TOL):
                j = int(np.clip(np.searchsorted(ga_grid, ga_c) - 1, 0, n_ga - 2))
                g_lo, se_lo = mc_gap(mc_pair, ga_grid[j], y, y_axis, seed)
                seed += 1
                g_hi, se_hi = mc_gap(mc_pair, ga_grid[j + 1], y, y_axis, seed)
                seed += 1
                span = (min(g_lo, g_hi) - (-TOL) <= 0 <= max(g_lo, g_hi) - (-TOL))
                disp = None
                if span and g_lo != g_hi:
                    f = (g_lo - (-TOL)) / (g_lo - g_hi)
                    disp = float(ga_grid[j] + f * (ga_grid[j + 1] - ga_grid[j]) - ga_c)
                sign_ok = True
                for gm, sem, ja in ((g_lo, se_lo, j), (g_hi, se_hi, j + 1)):
                    ga_pix = ga_grid[ja]
                    g_an = gap_rows[i, ja]
                    if abs(gm) > 2 * sem and abs(g_an) > TOL / 2:
                        sign_ok &= (np.sign(gm) == np.sign(g_an))
                well = slope >= M_MIN
                sharp = bool(well and sign_ok and disp is not None
                             and abs(disp) <= DISP_MAX)
                boundary_pts.append({
                    "y": round(float(y), 2), "ga": round(ga_c, 3),
                    "slope": round(slope, 3), "well_conditioned": bool(well),
                    "mc_bracket": [round(g_lo, 4), round(g_hi, 4)],
                    "bracket_se": [round(se_lo, 4), round(se_hi, 4)],
                    "displacement": None if disp is None else round(disp, 3),
                    "sign_agree": bool(sign_ok), "display_sharp": sharp})

        rng = np.random.default_rng(SEED_BASE + 10_000)
        sign_rows = []
        for i in range(0, n_y, 4):
            for j in range(0, n_ga, 4):
                g_an = gap_rows[i, j]
                gm, sem = mc_gap(mc_pair, ga_grid[j], y_grid[i], y_axis, seed)
                seed += 1
                resolvable = abs(gm) > 2 * sem
                sign_rows.append({"y": round(float(y_grid[i]), 2),
                                  "ga": round(float(ga_grid[j]), 3),
                                  "analytic": round(g_an, 4),
                                  "mc": round(gm, 4), "se": round(sem, 4),
                                  "resolvable": bool(resolvable),
                                  "agree": bool(np.sign(gm) == np.sign(g_an))
                                  if resolvable else None,
                                  "abs_err": round(abs(gm - g_an), 4)})

        pts = boundary_pts
        disp_all = [abs(r["displacement"]) for r in pts if r["displacement"] is not None]
        kept = [r for r in pts if r["well_conditioned"]]
        disp_kept = [abs(r["displacement"]) for r in kept if r["displacement"] is not None]
        res = [r for r in sign_rows if r["resolvable"]]
        agree = [r for r in res if r["agree"]]
        errs = [r["abs_err"] for r in sign_rows]
        panel_summary = {
            "boundary_points": len(pts),
            "well_conditioned": len(kept),
            "displacement_max_all": (round(max(disp_all), 3) if disp_all else None),
            "displacement_max_well_conditioned": (round(max(disp_kept), 3)
                                                  if disp_kept else None),
            "unresolved_brackets": sum(1 for r in pts if r["displacement"] is None),
            "display_sharp": sum(1 for r in pts if r["display_sharp"]),
            "sign_pixels_resolvable": len(res),
            "sign_agreement": (round(len(agree) / len(res), 3) if res else None),
            "gap_abs_err_p95": round(float(np.percentile(errs, 95)), 4),
            "gap_abs_err_max": round(float(max(errs)), 4)}
        result["panels"][pname] = {"summary": panel_summary,
                                   "boundary": pts, "sign_pixels": sign_rows}

        lines += [f"## Panel {pname}: {pair[0]} vs {pair[1]} over (gamma a, {y_axis})", "",
                  f"- boundary points: {panel_summary['boundary_points']}; "
                  f"well-conditioned (slope >= {M_MIN}): {panel_summary['well_conditioned']}; "
                  f"display-sharp after all rules: {panel_summary['display_sharp']}",
                  f"- max |displacement|: all points {panel_summary['displacement_max_all']}; "
                  f"well-conditioned only {panel_summary['displacement_max_well_conditioned']}; "
                  f"unresolved brackets {panel_summary['unresolved_brackets']}",
                  f"- sign agreement on resolvable subgrid pixels: {panel_summary['sign_agreement']} "
                  f"({panel_summary['sign_pixels_resolvable']} resolvable)",
                  f"- |analytic - MC| gap error on subgrid: p95 {panel_summary['gap_abs_err_p95']}, "
                  f"max {panel_summary['gap_abs_err_max']}", "",
                  "| y | gamma_a | slope | well-cond | MC bracket | displacement | sign | sharp |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in pts:
            lines.append(f"| {r['y']} | {r['ga']} | {r['slope']} | "
                         f"{r['well_conditioned']} | {r['mc_bracket']} | "
                         f"{r['displacement']} | {r['sign_agree']} | {r['display_sharp']} |")
        lines.append("")

    (OUT_DIR / "hero_boundary_audit_report.md").write_text("\n".join(lines))
    (OUT_DIR / "hero_boundary_audit_report.json").write_text(
        json.dumps(result, indent=1))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()

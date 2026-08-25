"""R1 (audit #5, section 13): reconcile the two crossing implementations.

Estimator A (per-score): decreasing isotonic fit to Energy and CTM AUGRC
separately, continuum interpolation, subtraction, simultaneous cluster-
bootstrap band inversion. This is the lineage of the manuscript's narrow
interval (documentation/x6_spectral_scripts/x4_real_fit.py, which used B=300
and z-unit severity; recorded here as historical context only).

Estimator B (direct gap): increasing isotonic fit to the per-row gap
AUGRC_Energy - AUGRC_CTM, identical band machinery. This is the robustness
audit's primary estimator (code/crossing_robustness_audit.py).

Frozen equalities (contract, 2026-08-24): identical severity construction
(four-metric composite, per-source z), identical fine grid (FINE_N=301),
identical checkpoint clusters, B=2000 each, identical simultaneous tie-set
definition {d : |Delta(d)| <= q95(sup-dev)}. Declared seeds: A=11, B=0 (B's
seed matches the robustness audit so its pooled row is reproduced exactly).

Usage (from code/): python crossing_reconcile.py [--b 2000]
Outputs: nc_csf_predictivity/outputs/track1/crossing_reconcile_report.md/.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from crossing_robustness_audit import (FINE_N, METRICS, OUT_DIR, PARQUET,
                                       attach_d, build_cells, crossings,
                                       load_severity_rows, pava_inc,
                                       severity_map)


def pava_dec(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return -pava_inc(-y, w)


def per_d_means(vals: dict[str, list], active: list[str]):
    acc: dict[float, list[float]] = {}
    for c in active:
        for d, g in vals[c]:
            acc.setdefault(d, []).append(g)
    ds = np.array(sorted(acc))
    w = np.array([len(acc[d]) for d in ds], float)
    ym = np.array([float(np.mean(acc[d])) for d in ds])
    return ds, ym, w


def curve_gap(data_gap: dict, active: list[str], fine: np.ndarray) -> np.ndarray:
    ds, ym, w = per_d_means(data_gap, active)
    return np.interp(fine, ds, pava_inc(ym, w))


def curve_per_score(data_e: dict, data_c: dict, active: list[str],
                    fine: np.ndarray) -> np.ndarray:
    ds_e, ym_e, w_e = per_d_means(data_e, active)
    ds_c, ym_c, w_c = per_d_means(data_c, active)
    fe = np.interp(fine, ds_e, pava_dec(ym_e, w_e))
    fc = np.interp(fine, ds_c, pava_dec(ym_c, w_c))
    return fe - fc


def analyze(name: str, fit, active: list[str], fine: np.ndarray, b: int,
            seed: int) -> dict:
    t0 = time.time()
    g0 = fit(active)
    xs = crossings(fine, g0)
    up = [x for x in xs
          if np.interp(x + 1e-6, fine, g0) > np.interp(x - 1e-6, fine, g0)]
    rng = np.random.default_rng(seed)
    devs = np.empty(b)
    for i in range(b):
        boot = list(rng.choice(active, len(active), replace=True))
        devs[i] = np.nanmax(np.abs(fit(boot) - g0))
    q = float(np.quantile(devs, 0.95))
    inside = np.abs(g0) <= q
    segs = []
    if inside.any():
        idx = np.where(inside)[0]
        start = idx[0]
        for a_i, b_i in zip(idx[:-1], idx[1:]):
            if b_i != a_i + 1:
                segs.append([float(fine[start]), float(fine[a_i])])
                start = b_i
        segs.append([float(fine[start]), float(fine[idx[-1]])])
    return {"estimator": name, "seed": seed, "b": b,
            "first_up_crossing": round(up[0], 3) if up else None,
            "all_crossings": [round(x, 3) for x in xs],
            "n_sign_changes": len(xs),
            "band_q95": round(q, 3),
            "zero_set_span": ([round(float(fine[inside].min()), 3),
                               round(float(fine[inside].max()), 3)]
                              if inside.any() else None),
            "zero_set_segments": [[round(a, 3), round(c, 3)] for a, c in segs],
            "zero_set_bounded_right": (bool(fine[inside].max() < fine[-1] - 1e-9)
                                       if inside.any() else None),
            "g_at_min_d": round(float(g0[0]), 2),
            "g_at_max_d": round(float(g0[-1]), 2),
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="R1 crossing reconciliation")
    parser.add_argument("--b", type=int, default=2000)
    args = parser.parse_args()

    df = pd.read_parquet(PARQUET)
    sev = severity_map(load_severity_rows(), METRICS)
    cells = attach_d(build_cells(df), sev)
    cells = cells.dropna(subset=["d"])

    data_gap: dict[str, list] = {}
    data_e: dict[str, list] = {}
    data_c: dict[str, list] = {}
    for r in cells.itertuples():
        data_gap.setdefault(r.cell, []).append((float(r.d), float(r.gap)))
        data_e.setdefault(r.cell, []).append((float(r.d), float(r.Energy)))
        data_c.setdefault(r.cell, []).append((float(r.d), float(r.CTM)))
    active = sorted(data_gap)
    fine = np.linspace(cells.d.min(), cells.d.max(), FINE_N)

    rec_a = analyze("A_per_score_isotonic",
                    lambda act: curve_per_score(data_e, data_c, act, fine),
                    active, fine, args.b, seed=11)
    rec_b = analyze("B_direct_gap_isotonic",
                    lambda act: curve_gap(data_gap, act, fine),
                    active, fine, args.b, seed=0)

    result = {"n_checkpoints": len(active), "n_rows": int(len(cells)),
              "fine_range": [round(float(fine[0]), 3), round(float(fine[-1]), 3)],
              "estimators": [rec_a, rec_b],
              "historical_note": {
                  "script": "documentation/x6_spectral_scripts/x4_real_fit.py",
                  "reported": {"d_star": -1.097, "ci": [-1.116, -1.057]},
                  "b": 300, "severity_units": "z-units (different axis)",
                  "status": "superseded by this reconciliation"}}

    lines = ["# Crossing reconciliation (R1; frozen equalities per claim contract 2026-08-24)",
             "",
             f"Checkpoints: {len(active)}; (checkpoint, OOD set) rows: {len(cells)}; "
             f"fine grid {FINE_N} points over [{fine[0]:.3f}, {fine[-1]:.3f}]; "
             f"B = {args.b}; seeds A=11, B=0; identical severity, grid, clusters, tie definition.",
             "",
             "| estimator | first up-crossing | all crossings | sign changes | band q95 | zero-set span | zero-set segments | bounded right | g(d_min) | g(d_max) | seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in (rec_a, rec_b):
        lines.append(
            f"| {r['estimator']} | {r['first_up_crossing']} | {r['all_crossings']} | "
            f"{r['n_sign_changes']} | {r['band_q95']} | {r['zero_set_span']} | "
            f"{r['zero_set_segments']} | {r['zero_set_bounded_right']} | "
            f"{r['g_at_min_d']} | {r['g_at_max_d']} | {r['seconds']} |")
    da, db = rec_a["first_up_crossing"], rec_b["first_up_crossing"]
    lines += ["",
              f"Point-estimate agreement: |d_A - d_B| = "
              f"{abs(da - db):.3f}" if None not in (da, db) else "Point estimates not both defined.",
              "",
              "Historical note: the manuscript's narrow interval (-1.097, CI [-1.116, -1.057]) came from "
              "`documentation/x6_spectral_scripts/x4_real_fit.py` at B=300 on a z-unit severity axis; it is superseded by the table above and may only be cited as a shape-constrained, conditional interval next to the simultaneous zero set.", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "crossing_reconcile_report.md").write_text("\n".join(lines))
    (OUT_DIR / "crossing_reconcile_report.json").write_text(json.dumps(result, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

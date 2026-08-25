"""R2 (audit #5, sections 6.6/13): realistic coverage simulation for the
crossing estimators, calibrated to the real VGG design.

Frozen design (claim contract 2026-08-24): 280 checkpoint clusters split
60/70/90/60 across the four ID sources exactly as in the real table; each
cluster contributes its source's eight observed severity supports; truth
curves, source offsets, checkpoint random effects (with Energy-CTM
correlation), and residual noise are all fitted from the real table; five
scenarios (single crossing, none, multiple, left-censored, right-censored);
both estimators (A per-score isotonic, B direct-gap isotonic) with the
simultaneous band tie-set definition; REPS=200, B_BOOT=500 (declared: reduced
from the reporting default 2000 for simulation runtime; the band quantile at
B=500 is adequate for coverage estimation), base seed 2026.

Evaluated per audit section 6.6: (1) first-handoff coverage when it exists;
(2) complete zero-set coverage; (3) false crossing declarations when none
exists; (4) censoring classification; (5) per-score vs direct-gap fitting;
(6) coverage of the window-selected (first contiguous zero segment) interval.

Usage (from code/): python crossing_coverage_sim.py [--reps 200] [--b 500]
Outputs: nc_csf_predictivity/outputs/track1/crossing_coverage_sim_report.md/.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from crossing_robustness_audit import (METRICS, OUT_DIR, PARQUET, attach_d,
                                       build_cells, crossings, curve,
                                       load_severity_rows, severity_map)

FINE_N = 301


def pava_w(y: np.ndarray, w: np.ndarray, increasing: bool) -> np.ndarray:
    if not increasing:
        return -pava_w(-y, w, True)
    y = y.astype(float).copy()
    w = w.astype(float).copy()
    blocks = [[y[i] * w[i], w[i], i, i] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            s2 = out.pop()
            s1 = out.pop()
            out.append([s1[0] + s2[0], s1[1] + s2[1], s1[2], s2[3]])
    fit = np.empty_like(y)
    for s, wt, i0, i1 in out:
        fit[i0:i1 + 1] = s / wt
    return fit


def calibrate() -> dict:
    df = pd.read_parquet(PARQUET)
    sev = severity_map(load_severity_rows(), METRICS)
    cells = attach_d(build_cells(df), sev).dropna(subset=["d"])
    data_gap: dict[str, list] = {}
    for r in cells.itertuples():
        data_gap.setdefault(r.cell, []).append((float(r.d), float(r.gap)))
    fine = np.linspace(cells.d.min(), cells.d.max(), FINE_N)
    mu = curve("loclin", data_gap, sorted(data_gap), fine)

    cells = cells.copy()
    cells["mu_at_d"] = np.interp(cells.d, fine, mu)
    for score in ("Energy", "CTM"):
        cells[f"res_{score}"] = np.nan
    # per-score smooth truths via loclin on each score
    per_score_mu = {}
    for score in ("Energy", "CTM"):
        dd: dict[str, list] = {}
        for r in cells.itertuples():
            dd.setdefault(r.cell, []).append((float(r.d), float(getattr(r, score))))
        per_score_mu[score] = curve("loclin", dd, sorted(dd), fine)
        cells[f"res_{score}"] = (getattr(cells, score)
                                 - np.interp(cells.d, fine, per_score_mu[score]))
    # source offsets on the per-score residuals
    off = {score: cells.groupby("source")[f"res_{score}"].mean().to_dict()
           for score in ("Energy", "CTM")}
    for score in ("Energy", "CTM"):
        cells[f"res_{score}"] -= cells.source.map(off[score])
    # cluster effects and residual noise, with Energy-CTM correlation
    cl_mean = cells.groupby("cell")[["res_Energy", "res_CTM"]].mean()
    tau = cl_mean.std(ddof=1)
    r_tau = float(cl_mean.corr().iloc[0, 1])
    within = cells.set_index("cell")[["res_Energy", "res_CTM"]] - cl_mean
    sig = within.std(ddof=1)
    r_sig = float(within.corr().iloc[0, 1])
    supports = (cells.groupby(["source", "eval_dataset"]).d.first()
                .reset_index())
    n_per_src = cells.groupby("source").cell.nunique().to_dict()
    # Per-source per-score noiseless values at that source's own supports
    # (S6 sensitivity variant: no additive-offset approximation).
    src_vals: dict[tuple, tuple[float, float]] = {}
    for src in n_per_src:
        sub = cells[cells.source == src]
        sup_d = np.sort(sub.d.unique())
        fitted = {}
        for score in ("Energy", "CTM"):
            dd: dict[str, list] = {}
            for r in sub.itertuples():
                dd.setdefault(r.cell, []).append((float(r.d), float(getattr(r, score))))
            fitted[score] = curve("loclin", dd, sorted(dd), sup_d)
        for k, d in enumerate(sup_d):
            src_vals[(src, round(float(d), 9))] = (float(fitted["Energy"][k]),
                                                   float(fitted["CTM"][k]))
    return {"fine": fine, "mu": mu, "mu_E": per_score_mu["Energy"],
            "mu_C": per_score_mu["CTM"],
            "tau": (float(tau.res_Energy), float(tau.res_CTM), r_tau),
            "sig": (float(sig.res_Energy), float(sig.res_CTM), r_sig),
            "supports": supports, "n_per_src": n_per_src,
            "src_off": off, "src_vals": src_vals}


def scenario_truths(fine: np.ndarray, mu: np.ndarray) -> dict[str, np.ndarray]:
    w = np.ones_like(mu)
    mu1 = pava_w(mu, w, True)
    truths = {"S1_single": mu1,
              "S2_none": mu1 - (mu1.max() + 2.0),
              "S4_left_censored": mu1 - mu1[0] + 1.0,
              "S5_right_censored": mu1 - mu1[-1] - 1.0}
    mu3 = mu.copy()
    if len(crossings(fine, mu3)) < 2:
        ramp = np.clip((fine - 0.4) / (fine[-1] - 0.4), 0, None)
        mu3 = mu1 - (mu1[-1] + 1.5) * ramp
    truths["S3_multiple"] = mu3
    return truths


def build_design(cal: dict, rng: np.random.Generator):
    rows = []
    cid = 0
    for src, n in sorted(cal["n_per_src"].items()):
        sup = cal["supports"][cal["supports"].source == src]
        for _ in range(n):
            for d in sup.d.values:
                rows.append((cid, src, float(d)))
            cid += 1
    arr = pd.DataFrame(rows, columns=["cid", "src", "d"])
    return arr, cid


def simulate_rep(cal, design, n_cl, truth_delta, rng, per_source_truth=False):
    tauE, tauC, r_tau = cal["tau"]
    sigE, sigC, r_sig = cal["sig"]
    cov_tau = np.array([[tauE ** 2, r_tau * tauE * tauC],
                        [r_tau * tauE * tauC, tauC ** 2]])
    cov_sig = np.array([[sigE ** 2, r_sig * sigE * sigC],
                        [r_sig * sigE * sigC, sigC ** 2]])
    b = rng.multivariate_normal([0, 0], cov_tau, size=n_cl)
    e = rng.multivariate_normal([0, 0], cov_sig, size=len(design))
    fine = cal["fine"]
    if per_source_truth:
        keys = [(s, round(float(d), 9)) for s, d in zip(design.src, design.d)]
        muE_d = np.array([cal["src_vals"][k][0] for k in keys])
        muC_d = np.array([cal["src_vals"][k][1] for k in keys])
        offE = offC = 0.0
    else:
        muE_d = np.interp(design.d, fine, cal["mu_E"] + truth_delta)
        muC_d = np.interp(design.d, fine, cal["mu_C"])
        offE = design.src.map(cal["src_off"]["Energy"]).values
        offC = design.src.map(cal["src_off"]["CTM"]).values
    cidx = design.cid.values
    E = muE_d + offE + b[cidx, 0] + e[:, 0]
    Cv = muC_d + offC + b[cidx, 1] + e[:, 1]
    return E, Cv


def fit_and_band(design, E, Cv, n_cl, fine, b_boot, rng, per_score: bool):
    d_vals = design.d.values
    sup_d, sup_inv = np.unique(d_vals, return_inverse=True)
    cidx = design.cid.values
    n_sup = len(sup_d)
    sumE = np.zeros((n_cl, n_sup)); sumC = np.zeros((n_cl, n_sup))
    cnt = np.zeros((n_cl, n_sup))
    np.add.at(sumE, (cidx, sup_inv), E)
    np.add.at(sumC, (cidx, sup_inv), Cv)
    np.add.at(cnt, (cidx, sup_inv), 1.0)

    def curve_from(choice):
        cE = sumE[choice].sum(0); cC = sumC[choice].sum(0)
        cn = cnt[choice].sum(0)
        ok = cn > 0
        if per_score:
            fE = pava_w(cE[ok] / cn[ok], cn[ok], False)
            fC = pava_w(cC[ok] / cn[ok], cn[ok], False)
            return np.interp(fine, sup_d[ok], fE) - np.interp(fine, sup_d[ok], fC)
        g = (cE[ok] - cC[ok]) / cn[ok]
        return np.interp(fine, sup_d[ok], pava_w(g, cn[ok], True))

    all_idx = np.arange(n_cl)
    g0 = curve_from(all_idx)
    devs = np.empty(b_boot)
    for i in range(b_boot):
        devs[i] = np.max(np.abs(curve_from(rng.integers(0, n_cl, n_cl)) - g0))
    q = float(np.quantile(devs, 0.95))
    return g0, q


def summarize_fit(fine, g0, q, true_zeros, scenario):
    xs = crossings(fine, g0)
    up = [x for x in xs
          if np.interp(x + 1e-6, fine, g0) > np.interp(x - 1e-6, fine, g0)]
    inside = np.abs(g0) <= q
    segs = []
    if inside.any():
        idx = np.where(inside)[0]
        start = idx[0]
        for a_i, b_i in zip(idx[:-1], idx[1:]):
            if b_i != a_i + 1:
                segs.append((fine[start], fine[a_i])); start = b_i
        segs.append((fine[start], fine[idx[-1]]))
    first = up[0] if up else None
    window = None
    if first is not None:
        for a_i, b_i in segs:
            if a_i - 1e-9 <= first <= b_i + 1e-9:
                window = (a_i, b_i)
                break
    out = {"declared_crossing": first is not None,
           "n_sign_changes": len(xs),
           "zero_set_bounded": bool(inside.any() and inside[0] == False and inside[-1] == False),  # noqa: E712
           "left_censor_call": bool(first is None and g0[0] > 0),
           "right_censor_call": bool(first is None and g0[-1] < 0)}
    if true_zeros:
        out["zero_set_covers_all_true"] = bool(
            inside.any() and all(any(a - 1e-9 <= z <= b + 1e-9 for a, b in segs)
                                 for z in true_zeros))
        z1 = true_zeros[0]
        out["window_covers_first_true"] = bool(
            window is not None and window[0] - 1e-9 <= z1 <= window[1] + 1e-9)
        out["first_err"] = (abs(first - z1) if first is not None else None)
        out["window_width"] = (window[1] - window[0]) if window else None
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--b", type=int, default=500)
    args = p.parse_args()

    cal = calibrate()
    fine = cal["fine"]
    truths = scenario_truths(fine, cal["mu"])
    rng = np.random.default_rng(2026)
    design, n_cl = build_design(cal, rng)

    results: dict = {"calibration": {
        "tau": cal["tau"], "sig": cal["sig"], "n_clusters": n_cl,
        "reps": args.reps, "b_boot": args.b, "seed": 2026}}
    lines = ["# Crossing-estimator coverage under the realistic design (R2)", "",
             f"Clusters: {n_cl} (real source split); supports: 8 per source (real severities); "
             f"reps {args.reps}; B_boot {args.b}; seed 2026; noise calibrated from the real table "
             f"(cluster sd Energy/CTM = {cal['tau'][0]:.1f}/{cal['tau'][1]:.1f}, corr {cal['tau'][2]:.2f}; "
             f"residual sd = {cal['sig'][0]:.1f}/{cal['sig'][1]:.1f}, corr {cal['sig'][2]:.2f}).", ""]

    # S6 sensitivity: truth = per-source per-score curves (no additive-offset
    # approximation); the population first-handoff is the zero of the
    # noiseless pooled isotonic gap summary.
    sup_all = np.sort(design.d.unique())
    g_noiseless = np.array(
        [cal["src_vals"][(design.src[design.d == d].iloc[0], round(float(d), 9))][0]
         - cal["src_vals"][(design.src[design.d == d].iloc[0], round(float(d), 9))][1]
         for d in sup_all])
    w_sup = np.array([float((design.d == d).sum()) for d in sup_all])
    s6_truth = np.interp(fine, sup_all, pava_w(g_noiseless, w_sup, True))
    truths["S6_per_source_realistic"] = s6_truth

    for scen, mu_t in truths.items():
        tz = sorted(crossings(fine, mu_t))
        delta = mu_t - cal["mu"]
        agg: dict[str, dict[str, list]] = {"A": {}, "B": {}}
        for rep in range(args.reps):
            E, Cv = simulate_rep(cal, design, n_cl, delta, rng,
                                 per_source_truth=scen.startswith("S6"))
            for est, per_score in (("A", True), ("B", False)):
                g0, q = fit_and_band(design, E, Cv, n_cl, fine, args.b, rng,
                                     per_score)
                rec = summarize_fit(fine, g0, q, tz, scen)
                for k, v in rec.items():
                    agg[est].setdefault(k, []).append(v)
        results[scen] = {"true_zeros": [round(z, 3) for z in tz]}
        lines += [f"## {scen} (true zeros: {[round(z, 3) for z in tz] or 'none in range'})", "",
                  "| estimator | declared crossing | zero-set covers all true | window covers first | median |first err| | median window width | mean sign changes | bounded zero set | left-censor call | right-censor call |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for est in ("A", "B"):
            a = agg[est]

            def rate(k):
                v = a.get(k)
                return "-" if not v else f"{np.mean([x for x in v]):.3f}"

            def med(k):
                v = [x for x in a.get(k, []) if x is not None]
                return "-" if not v else f"{np.median(v):.3f}"

            results[scen][est] = {k: (float(np.mean([x for x in v if x is not None]))
                                      if any(x is not None for x in v) else None)
                                  for k, v in a.items()}
            lines.append(f"| {est} | {rate('declared_crossing')} | "
                         f"{rate('zero_set_covers_all_true')} | "
                         f"{rate('window_covers_first_true')} | {med('first_err')} | "
                         f"{med('window_width')} | "
                         f"{np.mean(a['n_sign_changes']):.1f} | {rate('zero_set_bounded')} | "
                         f"{rate('left_censor_call')} | {rate('right_censor_call')} |")
        lines.append("")

    (OUT_DIR / "crossing_coverage_sim_report.md").write_text("\n".join(lines))
    (OUT_DIR / "crossing_coverage_sim_report.json").write_text(
        json.dumps(results, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

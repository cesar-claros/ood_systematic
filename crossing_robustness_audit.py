"""P0 crossing robustness audit (companion_phase_diagram_required_experiments
section 2; frozen spec, 2026-08-24).

FROZEN before running on real data:
  Pair/gap:  Delta(d) = AUGRC_Energy - AUGRC_CTM (raw AUGRC units), VGG-13
             cells only; cell = paradigm|source|run|reward|dropout; gap
             aggregated by mean within (cell, eval_dataset).
  Severity:  per-source z-standardization of the metric set over the
             included OOD datasets, averaged. PRIMARY composite replicates
             the existing pipeline exactly (kid, fd, text_align,
             img_centroid, no sign flip). Single-axis variants use
             kid / fd / img_centroid as-is and INVERSE text alignment
             (-z(text_align)), labeled the known unstable axis. CLIP
             model variants: NOT AVAILABLE (single CLIP model in
             clip_severity.csv); reported as such per "where available".
  Estimators (Analysis A):
             pava       weighted increasing isotonic (existing estimator);
             loclin     unconstrained local-linear, Gaussian kernel,
                        Silverman bandwidth on the pooled d values;
             spline     unconstrained least-squares cubic B-spline,
                        interior knots at the 25/50/75% quantiles of the
                        unique d support (falls back to fewer knots on
                        thin strata);
             piecewise  per-d weighted means, linear interpolation, no
                        projection.
  Bands:     cluster bootstrap over cells (resample cells with
             replacement), B = 2000, simultaneous 95% max-deviation band;
             tie region = {d : |ghat| <= q95}.
  Strata:    var-collapse tertiles over cells; T1 = lowest var_collapse
             (strongest collapse).
  Decision (section 2.8): the claim survives if strong-collapse models
             cross earlier while weak-collapse models show no crossing or
             a materially delayed one, stably across unconstrained fits,
             leave-one-OOD-out, leave-one-source-out, and severity
             definitions. Two-way checkpoint x shift bootstrap is NOT
             reported (only ~8 OOD sets; would require a calibration
             simulation per section 2.7).

Usage (from code/):
    python crossing_robustness_audit.py [--b 2000] [--self-test]
Outputs: nc_csf_predictivity/outputs/track1/crossing_robustness_report.md
         (+ .json).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
PARQUET = CODE / "nc_csf_predictivity/outputs/track1/dataset/long_harmonized.parquet"
SEVERITY_CSV = (CODE.parent
                / "documentation/x6_spectral_scripts/clip_severity.csv")
OUT_DIR = CODE / "nc_csf_predictivity/outputs/track1"
METRICS = ("kid", "fd", "text_align", "img_centroid")
SINGLE_AXES = {"kid_only": ("kid",), "fd_only": ("fd",),
               "img_centroid_only": ("img_centroid",),
               "inverse_text_align_only": ("text_align",)}
FINE_N = 301
ESTIMATORS = ("pava", "loclin", "spline", "piecewise")


# ---------------------------------------------------------------------------
# Severity construction.
# ---------------------------------------------------------------------------

def load_severity_rows() -> list[dict]:
    with open(SEVERITY_CSV) as fh:
        return list(csv.DictReader(fh))


def severity_map(rows: list[dict], metrics: tuple[str, ...],
                 exclude_set: str | None = None,
                 invert_text: bool = False) -> dict[tuple[str, str], float]:
    """(source, eval_dataset) -> d, z-standardized per source over the
    included OOD sets. invert_text flips text_align (single-axis variant)."""
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        if r["eval_dataset"] == exclude_set:
            continue
        by_source.setdefault(r["source"], []).append(r)
    out = {}
    for source, rr in by_source.items():
        mat = np.array([[float(r[m]) for m in metrics] for r in rr])
        z = (mat - mat.mean(0)) / (mat.std(0) + 1e-12)
        if invert_text and "text_align" in metrics:
            z[:, metrics.index("text_align")] *= -1.0
        for r, val in zip(rr, z.mean(1)):
            out[(source, r["eval_dataset"])] = float(val)
    return out


# ---------------------------------------------------------------------------
# Cell table.
# ---------------------------------------------------------------------------

def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (cell, eval_dataset): Energy/CTM AUGRC gap + geometry."""
    sub = df[(df.architecture == "VGG13") & (df.eval_dataset != "test")
             & df.csf.isin(["Energy", "CTM"])].copy()
    sub["cell"] = (sub.paradigm.astype(str) + "|" + sub.source.astype(str)
                   + "|" + sub["run"].astype(str) + "|"
                   + sub.reward.astype(str) + "|" + sub.dropout.astype(str))
    grouped = (sub.groupby(["cell", "source", "eval_dataset",
                            "var_collapse", "csf"])["augrc"]
               .mean().unstack("csf").reset_index())
    grouped["gap"] = grouped["Energy"] - grouped["CTM"]
    return grouped.dropna(subset=["gap"])


def attach_d(cells: pd.DataFrame,
             dmap: dict[tuple[str, str], float]) -> pd.DataFrame:
    out = cells.copy()
    out["d"] = [dmap.get((s, e)) for s, e in
                zip(out.source, out.eval_dataset)]
    return out.dropna(subset=["d"])


def tertiles(cells: pd.DataFrame) -> dict[str, set]:
    per_cell = cells.groupby("cell")["var_collapse"].first().sort_values()
    n = len(per_cell)
    ids = per_cell.index.to_list()
    return {"strong": set(ids[: n // 3]),
            "middle": set(ids[n // 3: 2 * n // 3]),
            "weak": set(ids[2 * n // 3:])}


# ---------------------------------------------------------------------------
# Estimators (all consume {cell: [(d, gap), ...]}).
# ---------------------------------------------------------------------------

def pava_inc(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    blocks: list[list[float]] = []
    for yi, wi in zip(np.asarray(y, float), np.asarray(w, float)):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            y2, w2, n2 = blocks.pop()
            y1, w1, n1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2,
                           n1 + n2])
    out: list[float] = []
    for yb, wb, nb in blocks:
        out += [yb] * nb
    return np.array(out)


def _per_d_means(data: dict[str, list], active: list[str]):
    acc: dict[float, list[float]] = {}
    for c in active:
        for d, g in data[c]:
            acc.setdefault(d, []).append(g)
    ds = np.array(sorted(acc))
    w = np.array([len(acc[d]) for d in ds], float)
    ym = np.array([float(np.mean(acc[d])) for d in ds])
    return ds, ym, w


def _points(data: dict[str, list], active: list[str]):
    pts = [(d, g) for c in active for d, g in data[c]]
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1]


def curve(estimator: str, data: dict[str, list], active: list[str],
          fine: np.ndarray) -> np.ndarray:
    ds, ym, w = _per_d_means(data, active)
    if len(ds) < 2:
        return np.full_like(fine, np.nan)
    if estimator == "pava":
        return np.interp(fine, ds, pava_inc(ym, w))
    if estimator == "piecewise":
        return np.interp(fine, ds, ym)
    if estimator == "loclin":
        x, y = _points(data, active)
        sd = x.std()
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        h = 0.9 * min(sd, iqr / 1.34 + 1e-12) * len(x) ** (-0.2)
        h = max(h, (ds.max() - ds.min()) / 50)
        out = np.empty_like(fine)
        for i, d0 in enumerate(fine):
            u = (x - d0) / h
            k = np.exp(-0.5 * u * u)
            s0, s1, s2 = k.sum(), (k * u).sum(), (k * u * u).sum()
            t0, t1 = (k * y).sum(), (k * u * y).sum()
            denom = s0 * s2 - s1 * s1
            out[i] = ((s2 * t0 - s1 * t1) / denom if abs(denom) > 1e-12
                      else t0 / max(s0, 1e-12))
        return out
    if estimator == "spline":
        from scipy.interpolate import LSQUnivariateSpline
        interior = np.quantile(ds, [0.25, 0.5, 0.75])
        interior = [t for t in interior if ds.min() < t < ds.max()]
        while len(interior) > 0 and len(ds) < len(interior) + 4:
            interior = interior[:-1]
        try:
            sp = LSQUnivariateSpline(ds, ym, interior, w=w, k=3)
        except Exception:
            sp = None
        if sp is None:
            coef = np.polyfit(ds, ym, min(3, len(ds) - 1), w=w)
            return np.polyval(coef, fine)
        return sp(np.clip(fine, ds.min(), ds.max()))
    raise ValueError(estimator)


# ---------------------------------------------------------------------------
# Curve diagnostics.
# ---------------------------------------------------------------------------

def crossings(fine: np.ndarray, g: np.ndarray) -> list[float]:
    s = np.sign(g)
    out = []
    for i in range(len(s) - 1):
        if s[i] != 0 and s[i + 1] != 0 and s[i] != s[i + 1]:
            f = g[i] / (g[i] - g[i + 1])
            out.append(float(fine[i] + f * (fine[i + 1] - fine[i])))
    return out


def analyze_curve(estimator: str, data: dict, active: list[str],
                  fine: np.ndarray, b: int,
                  rng: np.random.Generator) -> dict:
    g0 = curve(estimator, data, active, fine)
    xs = crossings(fine, g0)
    up = [x for i, x in enumerate(xs)
          if np.interp(x + 1e-6, fine, g0) > np.interp(x - 1e-6, fine, g0)]
    ds, ym, _ = _per_d_means(data, active)
    sign_flip_observed = any(np.sign(ym[i]) != np.sign(ym[i + 1])
                             and ym[i] != 0 for i in range(len(ym) - 1))
    rec = {"n_sign_changes": len(xs),
           "all_crossings": [round(x, 3) for x in xs],
           "first_up_crossing": round(up[0], 3) if up else None,
           "bracketed_by_observed": bool(sign_flip_observed),
           "g_at_min_d": float(g0[0]), "g_at_max_d": float(g0[-1])}
    if b > 0:
        devs = np.empty(b)
        for i in range(b):
            boot = list(rng.choice(active, len(active), replace=True))
            gb = curve(estimator, data, boot, fine)
            devs[i] = np.nanmax(np.abs(gb - g0))
        q = float(np.quantile(devs, 0.95))
        inside = np.abs(g0) <= q
        rec["band_q95"] = q
        rec["tie_region"] = ([round(float(fine[inside].min()), 3),
                              round(float(fine[inside].max()), 3)]
                             if inside.any() else None)
    return rec


# ---------------------------------------------------------------------------
# Analysis drivers.
# ---------------------------------------------------------------------------

def make_data(cells_d: pd.DataFrame) -> tuple[dict, list[str], np.ndarray]:
    data: dict[str, list] = {}
    for r in cells_d.itertuples():
        data.setdefault(r.cell, []).append((float(r.d), float(r.gap)))
    active = sorted(data)
    fine = np.linspace(cells_d.d.min(), cells_d.d.max(), FINE_N)
    return data, active, fine


def stratified(cells_d: pd.DataFrame, strata: dict[str, set],
               estimator: str, b: int, rng) -> dict:
    out = {}
    data, active, fine = make_data(cells_d)
    out["pooled"] = analyze_curve(estimator, data, active, fine, b, rng)
    for name, cellset in strata.items():
        sub = [c for c in active if c in cellset]
        out[name] = (analyze_curve(estimator, data, sub, fine, b, rng)
                     if len(sub) >= 5 else {"skipped": "too few cells"})
    return out


def crossing_value(rec: dict) -> float:
    """Effective crossing for ordering checks. A curve with no up-crossing
    that STARTS positive is left-censored (the crossing sits at or below
    the observed range = earliest possible, -inf); one that never turns
    positive is right-censored (no crossing in range, +inf). Documented
    reporting correction (2026-08-24): the initial rule counted
    left-censored strong strata as failures."""
    v = rec.get("first_up_crossing")
    if v is not None:
        return float(v)
    return -np.inf if rec.get("g_at_min_d", -1.0) > 0 else np.inf


def crossing_display(rec: dict):
    v = rec.get("first_up_crossing")
    if v is not None:
        return v
    return "<=min(censored)" if rec.get("g_at_min_d", -1.0) > 0 else None


def ordering_retained(res: dict) -> bool:
    """Strong crosses (possibly left-censored); weak has no crossing or a
    later one than middle and strong."""
    strong = crossing_value(res.get("strong", {}))
    middle = crossing_value(res.get("middle", {}))
    weak = crossing_value(res.get("weak", {}))
    return bool(strong < np.inf and weak >= middle - 0.05
                and weak >= strong - 0.05 and middle >= strong - 0.05)


def run_audit(df: pd.DataFrame, sev_rows: list[dict], b: int,
              seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base_cells = build_cells(df)
    result: dict = {"n_cells": int(base_cells.cell.nunique())}

    # Analysis A: four estimators, pooled + tertiles, full bands.
    dmap = severity_map(sev_rows, METRICS)
    cells_d = attach_d(base_cells, dmap)
    strata = tertiles(cells_d)
    result["A_estimators"] = {
        est: stratified(cells_d, strata, est, b, rng) for est in ESTIMATORS}

    # Analysis B: leave-one-OOD-dataset-out (pava; no bands needed).
    ood_sets = sorted({r["eval_dataset"] for r in sev_rows})
    result["B_loo_ood"] = {}
    for held in ood_sets:
        dmap_h = severity_map(sev_rows, METRICS, exclude_set=held)
        cd = attach_d(base_cells, dmap_h)
        res = stratified(cd, tertiles(cd), "pava", 0, rng)
        result["B_loo_ood"][held] = {
            "pooled": res["pooled"]["first_up_crossing"],
            "strong": crossing_display(res["strong"]),
            "middle": crossing_display(res["middle"]),
            "weak": crossing_display(res["weak"]),
            "ordering_retained": ordering_retained(res)}

    # Analysis C: leave-one-ID-source-out (pava, banded tie interval).
    result["C_loo_source"] = {}
    for held in sorted(base_cells.source.unique()):
        cd = attach_d(base_cells[base_cells.source != held], dmap)
        res = stratified(cd, tertiles(cd), "pava", b, rng)
        result["C_loo_source"][held] = {
            "pooled": res["pooled"]["first_up_crossing"],
            "tie": res["pooled"].get("tie_region"),
            "strong": crossing_display(res["strong"]),
            "middle": crossing_display(res["middle"]),
            "weak": crossing_display(res["weak"]),
            "ordering_retained": ordering_retained(res)}

    # Analysis D: severity-definition sensitivity (pava; no bands).
    variants: dict[str, dict] = {"full_composite": dmap}
    for name, metrics in SINGLE_AXES.items():
        variants[name] = severity_map(
            sev_rows, metrics, invert_text=("text_align" in metrics))
    for drop in METRICS:
        keep = tuple(m for m in METRICS if m != drop)
        variants[f"without_{drop}"] = severity_map(sev_rows, keep)
    result["D_severity"] = {}
    for name, dm in variants.items():
        cd = attach_d(base_cells, dm)
        res = stratified(cd, tertiles(cd), "pava", 0, rng)
        result["D_severity"][name] = {
            "pooled": res["pooled"]["first_up_crossing"],
            "strong": crossing_display(res["strong"]),
            "middle": crossing_display(res["middle"]),
            "weak": crossing_display(res["weak"]),
            "ordering_retained": ordering_retained(res)}
    result["D_severity"]["clip_model_variants"] = (
        "NOT AVAILABLE: severity table contains a single CLIP model")

    # Analysis E: two uncertainty targets.
    pooled = result["A_estimators"]["pava"]["pooled"]
    loo_vals = ([v["pooled"] for v in result["B_loo_ood"].values()
                 if v["pooled"] is not None]
                + [v["pooled"] for v in result["C_loo_source"].values()
                   if v["pooled"] is not None])
    result["E_uncertainty"] = {
        "conditional_checkpoint": {
            "first_up_crossing": pooled["first_up_crossing"],
            "tie_region": pooled.get("tie_region"),
            "note": "cluster bootstrap over checkpoints, conditional on "
                    "the fixed OOD suite"},
        "shift_sensitivity_range": ([round(min(loo_vals), 3),
                                     round(max(loo_vals), 3)]
                                    if loo_vals else None),
        "two_way_bootstrap": "NOT REPORTED: ~8 OOD sets; would require a "
                             "calibration simulation (section 2.7)"}

    # Section 2.8 decision.
    a_ok = all(ordering_retained({k: v for k, v in res.items()
                                  if k != "pooled"})
               for res in result["A_estimators"].values())
    b_ok = sum(v["ordering_retained"]
               for v in result["B_loo_ood"].values())
    c_ok = sum(v["ordering_retained"]
               for v in result["C_loo_source"].values())
    d_items = [v for k, v in result["D_severity"].items()
               if isinstance(v, dict)]
    d_ok = sum(v["ordering_retained"] for v in d_items)
    single_cross = all(
        res["pooled"]["n_sign_changes"] <= 1
        for est, res in result["A_estimators"].items() if est != "pava")
    result["decision"] = {
        "estimators_ordering_ok": a_ok,
        "unconstrained_single_crossing": single_cross,
        "loo_ood_ok": f"{b_ok}/{len(result['B_loo_ood'])}",
        "loo_source_ok": f"{c_ok}/{len(result['C_loo_source'])}",
        "severity_ok": f"{d_ok}/{len(d_items)}",
        "verdict": ("PASS" if a_ok and b_ok == len(result["B_loo_ood"])
                    and c_ok == len(result["C_loo_source"])
                    and d_ok == len(d_items)
                    else "CONDITIONAL" if a_ok else "FAIL"),
    }
    return result


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

def render(result: dict, b: int) -> str:
    lines = ["# Crossing robustness audit (P0; frozen spec in "
             "crossing_robustness_audit.py)", ""]
    lines.append(f"Cells: {result['n_cells']}; band bootstrap B = {b}; "
                 f"gap = AUGRC_Energy - AUGRC_CTM (raw AUGRC units); "
                 f"positive gap = CTM better.")
    lines.append("")
    lines.append("## A. Estimators (pooled and by var-collapse tertile)")
    lines.append("")
    lines.append("| estimator | stratum | sign changes | first up-crossing "
                 "| tie region | g(d_min) | g(d_max) | bracketed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for est, res in result["A_estimators"].items():
        for stratum, r in res.items():
            if "skipped" in r:
                continue
            lines.append(
                f"| {est} | {stratum} | {r['n_sign_changes']} "
                f"| {r['first_up_crossing']} "
                f"| {r.get('tie_region')} | {r['g_at_min_d']:+.1f} "
                f"| {r['g_at_max_d']:+.1f} "
                f"| {r['bracketed_by_observed']} |")
    lines.append("")
    lines.append("## B. Leave-one-OOD-dataset-out (pava)")
    lines.append("")
    lines.append("| held-out OOD | pooled | strong | middle | weak "
                 "| ordering retained |")
    lines.append("|---|---|---|---|---|---|")
    for held, r in result["B_loo_ood"].items():
        lines.append(f"| {held} | {r['pooled']} | {r['strong']} "
                     f"| {r['middle']} | {r['weak']} "
                     f"| {r['ordering_retained']} |")
    lines.append("")
    lines.append("## C. Leave-one-ID-source-out (pava)")
    lines.append("")
    lines.append("| held-out source | pooled | tie interval | strong "
                 "| middle | weak | ordering retained |")
    lines.append("|---|---|---|---|---|---|---|")
    for held, r in result["C_loo_source"].items():
        lines.append(f"| {held} | {r['pooled']} | {r['tie']} "
                     f"| {r['strong']} | {r['middle']} | {r['weak']} "
                     f"| {r['ordering_retained']} |")
    lines.append("")
    lines.append("## D. Severity-definition sensitivity (pava)")
    lines.append("")
    lines.append("| severity variant | pooled | strong | middle | weak "
                 "| ordering retained |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in result["D_severity"].items():
        if not isinstance(r, dict):
            continue
        lines.append(f"| {name} | {r['pooled']} | {r['strong']} "
                     f"| {r['middle']} | {r['weak']} "
                     f"| {r['ordering_retained']} |")
    lines.append("")
    lines.append(f"CLIP model variants: "
                 f"{result['D_severity']['clip_model_variants']}.")
    lines.append("")
    lines.append("## E. Uncertainty targets")
    lines.append("")
    e = result["E_uncertainty"]
    lines.append(f"- Conditional checkpoint uncertainty: first up-crossing "
                 f"{e['conditional_checkpoint']['first_up_crossing']}, tie "
                 f"region {e['conditional_checkpoint']['tie_region']} "
                 f"({e['conditional_checkpoint']['note']}).")
    lines.append(f"- Shift sensitivity (LOO ranges): pooled crossing spans "
                 f"{e['shift_sensitivity_range']}.")
    lines.append(f"- Two-way bootstrap: {e['two_way_bootstrap']}.")
    lines.append("")
    d = result["decision"]
    lines.append("## Decision (section 2.8)")
    lines.append("")
    lines.append(f"- ordering across estimators/strata: "
                 f"{d['estimators_ordering_ok']}; unconstrained fits "
                 f"single-crossing: {d['unconstrained_single_crossing']}")
    lines.append(f"- LOO-OOD retained {d['loo_ood_ok']}; LOO-source "
                 f"retained {d['loo_source_ok']}; severity variants "
                 f"retained {d['severity_ok']}")
    lines.append(f"- **Verdict: {d['verdict']}**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test on synthetic data.
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(1)
    rows = []
    sources = ["s1", "s2"]
    sets = [f"o{i}" for i in range(8)]
    sev_rows = []
    for s_i, source in enumerate(sources):
        for i, name in enumerate(sets):
            sev_rows.append({"source": source, "eval_dataset": name,
                             "kid": str(0.1 * i + 0.01 * s_i),
                             "fd": str(0.2 * i), "text_align": str(1 - 0.05 * i),
                             "img_centroid": str(0.15 * i)})
    for source in sources:
        for run in range(30):
            vc = rng.uniform(0.02, 0.30)
            cross_at = -1.0 + 3.0 * vc  # strong collapse crosses earlier
            for i, name in enumerate(sets):
                for csf in ("Energy", "CTM"):
                    d_proxy = (i - 3.5) / 2.0
                    gap_true = 40.0 * (d_proxy - cross_at)
                    base = 200.0 + rng.normal(0, 2)
                    aug = base + (gap_true / 2 if csf == "Energy"
                                  else -gap_true / 2)
                    rows.append({"paradigm": "p", "csf": csf, "reward": 1,
                                 "run": run, "eval_dataset": name,
                                 "augrc": aug, "dropout": False,
                                 "architecture": "VGG13",
                                 "source": source,
                                 "var_collapse": vc})
    df = pd.DataFrame(rows)
    result = run_audit(df, sev_rows, b=100, seed=0)
    d = result["decision"]
    assert d["verdict"] == "PASS", d
    a = result["A_estimators"]
    for est in ESTIMATORS:
        strong = a[est]["strong"]["first_up_crossing"]
        weak = a[est]["weak"]["first_up_crossing"]
        assert strong is not None and (weak is None or weak > strong), est
    print("self-test PASS: all estimators recover the constructed "
          "strong-before-weak ordering; decision machinery agrees.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossing robustness audit")
    parser.add_argument("--b", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    df = pd.read_parquet(PARQUET)
    sev_rows = load_severity_rows()
    result = run_audit(df, sev_rows, b=args.b, seed=args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "crossing_robustness_report.md").write_text(
        render(result, args.b))
    (OUT_DIR / "crossing_robustness_report.json").write_text(
        json.dumps(result, indent=1, default=float))
    print(render(result, args.b))


if __name__ == "__main__":
    main()

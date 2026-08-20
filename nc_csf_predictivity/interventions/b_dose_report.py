"""B-axis dose-search report (documentation/B_axis_pilot_protocol.md).

Geometry-only: consumes the manipulation-stage geometry JSONs (and,
optionally, nullspace JSONs) for the Pilot 1 reference pool (etfreg
baselines + A1 arms) and the new varreg/ctrreg dose-search runs, and
evaluates each B dose against the frozen gates GB1-GB5, the on-support
check, and the A1++ covariance-matching criterion. No OOD or detector
quantity enters anywhere.

Gates per (mechanism, dose), medians over its paired seeds:
  GB1 accuracy    val-acc drop vs the same-seed baseline <= 1.5 pp.
  GB2 material    |delta var_collapse| >= 10 x the baseline seed SD, in
                  the contraction direction for positive doses.
  GB3 selectivity off-target coordinates (self_duality, logit_scale,
                  eig_max_over_mean, head_residual_fraction) inside the
                  reference span widened by 25% of its width.
  GB4 leakage     eta_perp <= 2 x the reference maximum (evaluated only
                  when nullspace records are available for the dose).
  GB5 complete    both paired seeds measured (no FAILED/missing runs).

Support: RMS z-distance on the five theory-motivated coordinates
(var_collapse + the four off-target coordinates), standardized by the
16-model reference pool; threshold = the maximum leave-one-out distance
inside the reference pool (the audit's replacement for the
high-dimensional 70-sigma diagnostic).

Selection: among doses passing all gates AND on-support, the recommended
dose per mechanism minimizes |median var_collapse - A1++ mean
var_collapse| (the geometry-matching criterion for the fresh
confirmation); the report also lists the strongest passing movement.

Usage (from code/, after extract_manipulation on the B runs):
    python nc_csf_predictivity/interventions/b_dose_report.py \
        [--geometry_dirs .../geometry .../geometry_bpilot] \
        [--nullspace_dirs .../nullspace .../nullspace_bpilot]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REF_KIND = "etfreg"
REF_LAMS = ("0.0", "-0.1", "0.3", "1.0")
BASE_LAM = "0.0"
MATCH_LAM = "1.0"  # A1++, the covariance level to match
B_KINDS = ("varreg", "ctrreg")
TARGET = "var_collapse"
OFF_TARGET = ("self_duality", "logit_scale", "eig_max_over_mean",
              "head_residual_fraction")
SUPPORT_COORDS = (TARGET,) + OFF_TARGET
ACC_GATE_PP = 1.5
MATERIAL_SDS = 10.0
SPAN_TOL = 0.25
ETA_FACTOR = 2.0
MIN_SEEDS = 2

Key = tuple[str, str, int]  # (kind, lam, run)


def load_records(dirs: list[Path], pattern: str = "*__last.json",
                 lam_field: bool = True) -> dict[Key, dict]:
    """Merge JSON records from several dirs, keyed (kind, lam, run)."""
    out: dict[Key, dict] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob(pattern)):
            rec = json.loads(path.read_text())
            if "kind" not in rec:
                continue
            out[(rec["kind"], rec["lam"], int(rec["run"]))] = rec
    return out


def reference_stats(geo: dict[Key, dict]) -> dict:
    ref_keys = [k for k in geo
                if k[0] == REF_KIND and k[1] in REF_LAMS]
    if len(ref_keys) < 8:
        raise ValueError(f"reference pool too small: {len(ref_keys)} models")
    base_vc = [geo[k][TARGET] for k in ref_keys if k[1] == BASE_LAM]
    match_vc = [geo[k][TARGET] for k in ref_keys if k[1] == MATCH_LAM]
    spans = {}
    for coord in OFF_TARGET:
        vals = np.array([geo[k][coord] for k in ref_keys])
        lo, hi = float(vals.min()), float(vals.max())
        tol = SPAN_TOL * (hi - lo)
        spans[coord] = (lo - tol, hi + tol)
    mat = np.array([[geo[k][c] for c in SUPPORT_COORDS] for k in ref_keys])
    mu, sd = mat.mean(0), mat.std(0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    loo = []
    for i in range(len(ref_keys)):
        rest = np.delete(mat, i, axis=0)
        r_mu, r_sd = rest.mean(0), rest.std(0, ddof=1)
        r_sd = np.where(r_sd > 1e-12, r_sd, 1.0)
        loo.append(float(np.sqrt((((mat[i] - r_mu) / r_sd) ** 2).mean())))
    return {"ref_keys": ref_keys,
            "base_vc_sd": float(np.std(base_vc, ddof=1)),
            "a1pp_vc": float(np.mean(match_vc)),
            "spans": spans, "support_mu": mu, "support_sd": sd,
            "support_threshold": float(max(loo))}


def support_distance(rec: dict, stats: dict) -> float:
    v = np.array([rec[c] for c in SUPPORT_COORDS])
    z = (v - stats["support_mu"]) / stats["support_sd"]
    return float(np.sqrt((z ** 2).mean()))


def evaluate_dose(kind: str, lam: str, geo: dict[Key, dict],
                  nulls: dict[Key, dict], stats: dict) -> dict:
    runs = sorted(r for k, l, r in geo if k == kind and l == lam)
    per_seed = []
    for run in runs:
        rec = geo[(kind, lam, run)]
        base = geo.get((REF_KIND, BASE_LAM, run))
        if base is None:
            continue
        per_seed.append({
            "run": run,
            "d_acc_pp": (base["val_acc"] - rec["val_acc"]) * 100.0,
            "d_vc": rec[TARGET] - base[TARGET],
            "vc": rec[TARGET],
            "off_target": {c: rec[c] for c in OFF_TARGET},
            "support_distance": support_distance(rec, stats),
        })
    if not per_seed:
        return {"n_seeds": 0, "gates": {"GB5_complete": False},
                "all_pass": False, "on_support": False}
    med = lambda key: float(np.median([s[key] for s in per_seed]))
    med_vc_delta = med("d_vc")
    gates = {
        "GB1_accuracy": med("d_acc_pp") <= ACC_GATE_PP,
        "GB2_material": (abs(med_vc_delta)
                         >= MATERIAL_SDS * stats["base_vc_sd"]
                         and (med_vc_delta < 0 or float(lam) < 0)),
        "GB3_selectivity": all(
            stats["spans"][c][0]
            <= float(np.median([s["off_target"][c] for s in per_seed]))
            <= stats["spans"][c][1]
            for c in OFF_TARGET),
        "GB5_complete": len(per_seed) >= MIN_SEEDS,
    }
    etas = [nulls[(kind, lam, s["run"])]["eta_perp"] for s in per_seed
            if (kind, lam, s["run"]) in nulls]
    ref_etas = [nulls[k]["eta_perp"] for k in stats["ref_keys"]
                if k in nulls]
    if etas and ref_etas:
        gates["GB4_leakage"] = bool(
            np.median(etas) <= ETA_FACTOR * max(ref_etas))
    dist = med("support_distance")
    on_support = dist <= stats["support_threshold"]
    return {
        "n_seeds": len(per_seed), "per_seed": per_seed,
        "median_d_acc_pp": med("d_acc_pp"),
        "median_d_vc": med_vc_delta,
        "median_vc": med("vc"),
        "sds_moved": (abs(med_vc_delta) / stats["base_vc_sd"]
                      if stats["base_vc_sd"] > 0 else float("inf")),
        "match_dist": abs(med("vc") - stats["a1pp_vc"]),
        "support_distance": dist, "on_support": bool(on_support),
        "gates": gates, "all_pass": bool(all(gates.values())),
    }


def run_report(geo: dict[Key, dict], nulls: dict[Key, dict]) -> dict:
    stats = reference_stats(geo)
    doses: dict[str, dict[str, dict]] = {}
    for kind in B_KINDS:
        lams = sorted({l for k, l, _ in geo if k == kind},
                      key=lambda s: float(s))
        doses[kind] = {lam: evaluate_dose(kind, lam, geo, nulls, stats)
                       for lam in lams}
    recommended = {}
    for kind, table in doses.items():
        ok = {lam: r for lam, r in table.items()
              if r["all_pass"] and r["on_support"]}
        recommended[kind] = (min(ok, key=lambda lam: ok[lam]["match_dist"])
                             if ok else None)
    candidates = [(kind, lam) for kind, lam in recommended.items()
                  if lam is not None]
    overall = (min(candidates,
                   key=lambda kl: doses[kl[0]][kl[1]]["match_dist"])
               if candidates else None)
    return {"reference": {
                "base_vc_sd": stats["base_vc_sd"],
                "a1pp_vc": stats["a1pp_vc"],
                "support_threshold": stats["support_threshold"],
                "spans": {c: list(v) for c, v in stats["spans"].items()}},
            "doses": doses, "recommended": recommended,
            "overall_recommendation": overall}


def render(result: dict) -> str:
    ref = result["reference"]
    lines = ["# B-axis dose-search report (geometry only)", ""]
    lines.append(f"Reference: baseline var_collapse seed SD "
                 f"{ref['base_vc_sd']:.5f}; A1++ target var_collapse "
                 f"{ref['a1pp_vc']:.4f}; support threshold (LOO) "
                 f"{ref['support_threshold']:.2f}.")
    lines.append("")
    for kind, table in result["doses"].items():
        lines.append(f"## {kind}")
        lines.append("")
        lines.append("| lam | seeds | d_acc (pp) | d_var_collapse (SDs) "
                     "| var_collapse | match to A1++ | support dist "
                     "| gates | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for lam, r in table.items():
            if r["n_seeds"] == 0:
                lines.append(f"| {lam} | 0 | - | - | - | - | - | - "
                             f"| NO DATA |")
                continue
            failed = [g for g, ok in r["gates"].items() if not ok]
            gate_str = "all pass" if r["all_pass"] else \
                "FAIL " + ",".join(f.split("_")[0] for f in failed)
            verdict = ("QUALIFIES" if r["all_pass"] and r["on_support"]
                       else ("off-support" if r["all_pass"]
                             else "rejected"))
            lines.append(
                f"| {lam} | {r['n_seeds']} | {r['median_d_acc_pp']:+.2f} "
                f"| {r['median_d_vc']:+.4f} ({r['sds_moved']:.0f}) "
                f"| {r['median_vc']:.4f} | {r['match_dist']:.4f} "
                f"| {r['support_distance']:.2f} | {gate_str} "
                f"| {verdict} |")
        rec = result["recommended"][kind]
        lines.append("")
        lines.append(f"Recommended {kind} dose: "
                     f"{rec if rec else 'NONE (no dose qualifies)'}")
        lines.append("")
    overall = result["overall_recommendation"]
    lines.append(f"**Overall geometry-matched pick: "
                 f"{overall if overall else 'NONE'}** (closest qualifying "
                 f"var_collapse to the A1++ level; protocol section 6).")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    base = "nc_csf_predictivity/interventions"
    parser = argparse.ArgumentParser(description="B-axis dose report")
    parser.add_argument("--geometry_dirs", nargs="+", type=str,
                        default=[f"{base}/geometry",
                                 f"{base}/geometry_bpilot"])
    parser.add_argument("--nullspace_dirs", nargs="+", type=str,
                        default=[f"{base}/nullspace",
                                 f"{base}/nullspace_bpilot"])
    parser.add_argument("--out", type=str,
                        default=f"{base}/b_dose_report.md")
    args = parser.parse_args()
    geo = load_records([Path(d) for d in args.geometry_dirs])
    nulls = load_records([Path(d) for d in args.nullspace_dirs],
                         pattern="*.json")
    result = run_report(geo, nulls)
    Path(args.out).write_text(render(result))
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(result, indent=1, default=float))
    print(f"recommended: {result['recommended']}; overall "
          f"{result['overall_recommendation']}; wrote {args.out}")


if __name__ == "__main__":
    main()

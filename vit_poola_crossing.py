"""P1 ViT and foundation-regime crossing replication
(companion_phase_diagram_required_experiments section 5; frozen spec,
2026-08-24).

External-validity analysis on EXISTING tables only (no new extraction,
probing, or training). PREDECLARED pairs, fixed before inspecting any
outcome table (section 5.3):

  fine-tuned ViT (40 cells, master table, paradigm modelvit):
    MLS vs "KPCA RecError global"   (theory pick from earlier plans)
    MLS vs Residual
    Energy vs CTM                   (common-family sensitivity)
  frozen probes (Pool A, 40 cells = DINOv2-ViT-B/14 + CLIP-ViT-B/16
  linear probes x 4 sources x 5 runs):
    MLS vs Maha                     (head-side vs feature-side pair,
                                     motivated by the X1 theory and the
                                     intervention campaign's E1 pair;
                                     chosen WITHOUT inspecting the Pool A
                                     clique tables)

Analysis = the section-2 robustness pipeline reused verbatim from
crossing_robustness_audit.py: gap Delta(d) = AUGRC_A - AUGRC_B on the
per-source CLIP severity composite; four estimators (pava, loclin,
spline, piecewise); B = 2000 cluster-bootstrap simultaneous bands for
the pava tie regions; leave-one-source-out; leave-one-OOD-set-out;
var-collapse tertile strata; explicit no-crossing outcomes. Pool A adds
encoder stratification (DINOv2 vs CLIP) and residue-energy
stratification (median split on rho_res, the X8 residue coordinate,
predeclared); probe-training-size sensitivity is NOT AVAILABLE (no size
column in the harmonized tables).

Interpretation buckets (section 5.5), computed per pair:
  sharp      first up-crossing exists and the pava tie region is bounded
             inside the observed severity range;
  smeared    crossing exists but the tie region reaches an edge;
  none       no up-crossing (the winning side is reported; this can
             still support the paper if geometry places the observed
             range on one side of the boundary);
  (pair-dependent inconsistency is reported in the summary.)
Per section 5.5, no new compute is dispatched to force a replication; if
underpowered, the precise crossing result stays scoped to VGG-13.

Usage (from code/):  python vit_poola_crossing.py [--b 2000]
Outputs: nc_csf_predictivity/outputs/track1/vit_poola_crossing_report.md
         (+ .json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossing_robustness_audit import (
    METRICS,
    analyze_curve,
    crossing_display,
    load_severity_rows,
    make_data,
    severity_map,
)

CODE = Path(__file__).resolve().parent
PARQUET = CODE / "nc_csf_predictivity/outputs/track1/dataset/long_harmonized.parquet"
POOL_A = CODE / "nc_csf_predictivity/outputs/pool_a/long_pool_a_harmonized.parquet"
POOL_A_MODELS = (CODE
                 / "nc_csf_predictivity/outputs/pool_a/models_pool_a_harmonized.parquet")
OUT_DIR = CODE / "nc_csf_predictivity/outputs/track1"
POOL_A_RENAME = {"lsun_cropped": "lsun cropped",
                 "lsun_resize": "lsun resize"}
VIT_PAIRS = (("MLS", "KPCA RecError global"),
             ("MLS", "Residual"),
             ("Energy", "CTM"))
PROBE_PAIR = ("MLS", "Maha")


# ---------------------------------------------------------------------------
# Cell builders (generalized from the VGG audit).
# ---------------------------------------------------------------------------

def build_pair_cells(df: pd.DataFrame, pair: tuple[str, str],
                     geom: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (cell, eval_dataset): AUGRC gap for the pair, plus
    var_collapse (from the table itself or a models join)."""
    sub = df[df.csf.isin(pair)].copy()
    sub["cell"] = (sub.paradigm.astype(str) + "|" + sub.source.astype(str)
                   + "|" + sub["run"].astype(str) + "|"
                   + sub.reward.astype(str) + "|" + sub.dropout.astype(str))
    keys = ["cell", "source", "eval_dataset"]
    if "var_collapse" in sub.columns:
        keys.append("var_collapse")
    grouped = (sub.groupby(keys + ["csf"])["augrc"]
               .mean().unstack("csf").reset_index())
    grouped["gap"] = grouped[pair[0]] - grouped[pair[1]]
    grouped = grouped.dropna(subset=["gap"])
    if "var_collapse" not in grouped.columns and geom is not None:
        gm = geom.copy()
        gm["cell"] = (gm.paradigm.astype(str) + "|" + gm.source.astype(str)
                      + "|" + gm["run"].astype(str) + "|"
                      + gm.reward.astype(str) + "|"
                      + gm.dropout.astype(str))
        grouped = grouped.merge(gm[["cell", "var_collapse", "rho_res"]],
                                on="cell", how="left")
    return grouped


def attach_severity(cells: pd.DataFrame, dmap: dict) -> pd.DataFrame:
    out = cells.copy()
    out["d"] = [dmap.get((s, e)) for s, e in
                zip(out.source, out.eval_dataset)]
    return out.dropna(subset=["d"])


def tertiles_of(cells: pd.DataFrame) -> dict[str, set]:
    per_cell = cells.groupby("cell")["var_collapse"].first().sort_values()
    ids = per_cell.index.to_list()
    n = len(ids)
    return {"strong": set(ids[: n // 3]),
            "middle": set(ids[n // 3: 2 * n // 3]),
            "weak": set(ids[2 * n // 3:])}


# ---------------------------------------------------------------------------
# Per-pair pipeline (section 2 machinery on an arbitrary cell table).
# ---------------------------------------------------------------------------

def classify(rec: dict, fine_range: tuple[float, float]) -> str:
    x = rec.get("first_up_crossing")
    if x is None:
        if rec.get("g_at_min_d", -1.0) > 0:
            return "none (second score better everywhere; left-censored)"
        return "none (first score better everywhere)"
    tie = rec.get("tie_region")
    if tie is None:
        return "sharp"
    lo, hi = tie
    eps = 0.02 * (fine_range[1] - fine_range[0])
    bounded = lo > fine_range[0] + eps and hi < fine_range[1] - eps
    return "sharp" if bounded else "smeared"


def pair_analysis(cells_d: pd.DataFrame, b: int, rng,
                  strata_extra: dict[str, set] | None = None) -> dict:
    data, active, fine = make_data(cells_d)
    frange = (float(fine[0]), float(fine[-1]))
    out: dict = {"n_cells": len(active),
                 "estimators": {}}
    for est in ("pava", "loclin", "spline", "piecewise"):
        rec = analyze_curve(est, data, active, fine,
                            b if est == "pava" else 0, rng)
        out["estimators"][est] = rec
    out["classification"] = classify(out["estimators"]["pava"], frange)
    strata = tertiles_of(cells_d)
    if strata_extra:
        strata.update(strata_extra)
    out["strata"] = {}
    for name, cellset in strata.items():
        sub = [c for c in active if c in cellset]
        if len(sub) < 5:
            out["strata"][name] = {"skipped": "too few cells"}
            continue
        rec = analyze_curve("pava", data, sub, fine, 0, rng)
        out["strata"][name] = {
            "crossing": crossing_display(rec),
            "g_at_min": round(rec["g_at_min_d"], 2),
            "g_at_max": round(rec["g_at_max_d"], 2)}
    # Leave-one-source-out and leave-one-OOD-set-out (pava crossings).
    out["loo_source"] = {}
    for held in sorted(cells_d.source.unique()):
        sub_cells = cells_d[cells_d.source != held]
        if sub_cells.cell.nunique() < 5:
            out["loo_source"][held] = "too few cells"
            continue
        d2, a2, f2 = make_data(sub_cells)
        out["loo_source"][held] = crossing_display(
            analyze_curve("pava", d2, a2, f2, 0, rng))
    out["loo_ood"] = {}
    for held in sorted(cells_d.eval_dataset.unique()):
        sub_cells = cells_d[cells_d.eval_dataset != held]
        d2, a2, f2 = make_data(sub_cells)
        out["loo_ood"][held] = crossing_display(
            analyze_curve("pava", d2, a2, f2, 0, rng))
    return out


def run_step8(b: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sev = load_severity_rows()
    dmap = severity_map(sev, METRICS)
    result: dict = {"vit": {}, "pool_a": {}}

    df = pd.read_parquet(PARQUET)
    vit = df[(df.architecture == "ViT") & (df.eval_dataset != "test")]
    for pair in VIT_PAIRS:
        cells = attach_severity(build_pair_cells(vit, pair), dmap)
        result["vit"][f"{pair[0]} vs {pair[1]}"] = pair_analysis(
            cells, b, rng)

    pa = pd.read_parquet(POOL_A)
    pa = pa[pa.eval_dataset != "iid"].copy()
    pa["eval_dataset"] = pa.eval_dataset.replace(POOL_A_RENAME)
    models = pd.read_parquet(POOL_A_MODELS)
    cells = attach_severity(
        build_pair_cells(pa, PROBE_PAIR, geom=models), dmap)
    # Predeclared extra strata: encoder and residue-energy median split.
    enc = {name: set(cells[cells.cell.str.startswith(name)].cell)
           for name in ("probe_dinov2_vitb14", "probe_clip_vitb16")}
    rho = cells.groupby("cell")["rho_res"].first()
    med = rho.median()
    residue = {"low_residue": set(rho[rho <= med].index),
               "high_residue": set(rho[rho > med].index)}
    result["pool_a"][f"{PROBE_PAIR[0]} vs {PROBE_PAIR[1]}"] = pair_analysis(
        cells, b, rng, strata_extra={**enc, **residue})
    result["pool_a"]["probe_training_size_sensitivity"] = (
        "NOT AVAILABLE: no probe-size column in the harmonized tables")
    return result


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------

def render(result: dict, b: int) -> str:
    lines = [("# ViT and foundation-regime crossing replication (P1; "
             "frozen spec in vit_poola_crossing.py)"), ""]
    lines.append(f"Existing tables only; predeclared pairs; band "
                 f"bootstrap B = {b}; gap = AUGRC_A - AUGRC_B (positive "
                 f"= second score better).")
    for pool, pool_name in (("vit", "Fine-tuned ViT (40 cells)"),
                            ("pool_a", ("Frozen probes, Pool A "
                                       "(40 cells)"))):
        lines.append("")
        lines.append(f"## {pool_name}")
        for pair, res in result[pool].items():
            if isinstance(res, str):
                lines.append(f"- {pair}: {res}")
                continue
            lines.append("")
            lines.append(f"### {pair}  ->  "
                         f"**{res['classification']}**")
            lines.append("")
            lines.append("| estimator | sign changes | first up-crossing "
                         "| tie region | g(d_min) | g(d_max) |")
            lines.append("|---|---|---|---|---|---|")
            for est, r in res["estimators"].items():
                lines.append(
                    f"| {est} | {r['n_sign_changes']} "
                    f"| {r['first_up_crossing']} | {r.get('tie_region')} "
                    f"| {r['g_at_min_d']:+.1f} | {r['g_at_max_d']:+.1f} |")
            lines.append("")
            strata_str = "; ".join(
                f"{name}: {rec.get('crossing', rec.get('skipped'))}"
                for name, rec in res["strata"].items())
            lines.append(f"- strata crossings: {strata_str}")
            lines.append("- leave-one-source-out crossings: "
                         + "; ".join(f"{k}: {v}" for k, v in
                                     res["loo_source"].items()))
            loo_vals = [v for v in res["loo_ood"].values()
                        if isinstance(v, float)]
            lines.append(f"- leave-one-OOD-set-out: "
                         f"{len(loo_vals)}/{len(res['loo_ood'])} refits "
                         f"keep a crossing"
                         + (f", range [{min(loo_vals):.2f}, "
                            f"{max(loo_vals):.2f}]" if loo_vals else ""))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ViT / Pool A crossing replication")
    parser.add_argument("--b", type=int, default=2000)
    args = parser.parse_args()
    result = run_step8(args.b)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = render(result, args.b)
    (OUT_DIR / "vit_poola_crossing_report.md").write_text(report)
    (OUT_DIR / "vit_poola_crossing_report.json").write_text(
        json.dumps(result, indent=1, default=float))
    print(report)


if __name__ == "__main__":
    main()

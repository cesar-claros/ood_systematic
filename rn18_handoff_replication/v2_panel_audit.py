"""Extraction-chain audit of the version-2 (lossless) panel against the
version-1 panel of record (EXTRACTION-CHAIN AUDIT, post-outcome, descriptive).

A. Inventory of the version-2 outputs (name, size, sha256 of JSON and npz),
   extractor revisions present (must be one), checkpoint identities.
B. Field-by-field consistency of every frozen rounded field between the
   version-1 records (readout of record) and the version-2 records of the
   same checkpoints: outcomes (AUROC, AUGRC raw and balanced, gaps,
   materiality, counts), coordinates (gamma, a, rho, ...), papyan panel,
   geometry, ID-test coordinates, P10 block scalars. Expected: exact
   equality (same frozen arithmetic on a deterministic forward); any
   difference is reported per field.
C. Unrounded sensitivity of the corrected readout: SEL (all-cell regret,
   reference contrast) and LEVEL recomputed from the UNROUNDED outcomes
   with the version-2 reader primitives at multiplier 1 (descriptive), and
   the handoff gate recomputed from unrounded gaps; compared with the
   corrected readout of record.
D. Per-example evidence summaries: AUGRC identity residuals, failure-AUROC
   decomposition (pi, failure-AUROC gaps), exact score ties (duplicate
   values, ceiling AUROCs), correctness partition, test/train within-class
   scale ratio, environment provenance.

Usage (from code/): python rn18_handoff_replication/v2_panel_audit.py [--b 2000]
Output: rn18_handoff_replication/outputs/v2_panel_audit.json/.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from rn18_handoff_replication import rn18_analysis as v1
from rn18_handoff_replication import rn18_analysis_v2 as R
from rn18_handoff_replication.comparators import NAMES, REFERENCE, Comparators

OUT = Path("rn18_handoff_replication/outputs")
V1 = {"rn18": OUT / "fourshift_rn18", "vgg": OUT / "fourshift_vgg_bridge"}
V2 = {"rn18": OUT / "fourshift_v2_rn18", "vgg": OUT / "fourshift_v2_vgg_bridge"}
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
SCORES = ("Energy", "CTM", "MSR", "MLS", "Maha", "fDBD")
LABEL = "EXTRACTION-CHAIN AUDIT (post-outcome, descriptive)"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if not p.name.startswith("FAILED_")}


def _flat(o, prefix=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, f"{prefix}{k}/"))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            out.update(_flat(v, f"{prefix}{i}/"))
    elif isinstance(o, (int, float, bool)) and not isinstance(o, bool):
        out[prefix.rstrip("/")] = float(o)
    elif isinstance(o, bool):
        out[prefix.rstrip("/")] = float(o)
    return out


def field_consistency(a: dict, b: dict) -> dict:
    """Per-record: compare every numeric leaf that both records share among the frozen groups."""
    groups = {"outcomes": lambda r: {s: {k: v for k, v in r["ood"][s].items() if not isinstance(v, dict)} for s in SHIFTS if s in r["ood"]},
              "coords_p10": lambda r: {s: r["ood"][s].get("p10", {}) for s in SHIFTS if s in r["ood"]},
              "gaussian": lambda r: {s: r["ood"][s].get("gaussian", {}) for s in SHIFTS if s in r["ood"]},
              "papyan": lambda r: r["papyan"], "geometry": lambda r: r["geometry"], "iid_test": lambda r: r["iid_test"],
              "record": lambda r: {k: r[k] for k in ("n_classes", "dim") if k in r}}
    out = {}
    for g, f in groups.items():
        fa, fb = _flat(f(a)), _flat(f(b))
        keys = sorted(set(fa) & set(fb))
        diffs = {k: abs(fa[k] - fb[k]) for k in keys}
        worst = max(diffs.items(), key=lambda kv: kv[1]) if diffs else (None, 0.0)
        out[g] = {"n_fields": len(keys), "n_equal": int(sum(v == 0.0 for v in diffs.values())), "max_abs_diff": worst[1], "worst_field": worst[0],
                  "only_in_v1": len(set(fa) - set(fb)), "only_in_v2": len(set(fb) - set(fa))}
    return out


def unrounded_cells(recs: list[dict], axes: dict) -> pd.DataFrame:
    """The reader's cell table with aurocE/aurocC/dG/dG_bal replaced by the unrounded values."""
    df = v1.add_geometry_percentile(v1.cells_from_records(recs, axes, with_p10=True))
    un = {}
    for r in recs:
        for s, u in r["v2"]["ood_unrounded"].items():
            un[(r["slug"], s)] = u
    for i, row in df.iterrows():
        u = un[(row.cell, row.ood_set)]
        df.at[i, "aurocE"], df.at[i, "aurocC"] = u["auroc_id_vs_ood"]["Energy"], u["auroc_id_vs_ood"]["CTM"]
        df.at[i, "dG"], df.at[i, "dG_bal"] = u["gap_raw"], u["gap_balanced"]
    df["dA"] = df.aurocC - df.aurocE
    return df


def per_example_summaries(recs: list[dict], d: Path) -> dict:
    ident, ties, ceil, pis, decomp, ratios, errs, n_sets = 0.0, [], 0, [], [], [], [], 0
    for r in recs:
        z = np.load(d / f"{r['slug']}.npz")
        errs.append(float(r["v2"]["id_test"]["id_error_rate_unrounded"]))
        if "id_test_feature_model" in r["v2"]:
            ratios.append(r["v2"]["id_test_feature_model"]["test_over_train_sigma_iso"])
        for s in SHIFTS:
            u = r["v2"]["ood_unrounded"][s]; n_sets += 1
            ident = max(ident, max(abs(x) for k in ("identity_residual_raw", "identity_residual_balanced") for x in u[k].values()))
            pis.append((u["pi_raw"], u["pi_balanced"]))
            decomp.append(u["gap_balanced_decomposition"]["failure_auroc_gap_CTM_minus_Energy"])
            for sc in ("Energy", "CTM"):
                x = np.concatenate([z[f"id__score__{sc}"], z[f"set__{s}__score__{sc}"]])
                ties.append(1.0 - len(np.unique(x)) / len(x))
                ceil += int(u["auroc_id_vs_ood"][sc] >= 1.0 - 1e-12)
    return {"n_sets": n_sets, "max_identity_residual": ident, "pi_raw_range": [min(p[0] for p in pis), max(p[0] for p in pis)],
            "pi_balanced_range": [min(p[1] for p in pis), max(p[1] for p in pis)],
            "failure_auroc_gap_balanced_CTM_minus_Energy": {"mean": float(np.mean(decomp)), "min": float(np.min(decomp)), "max": float(np.max(decomp))},
            "score_tie_fraction_energy_ctm": {"mean": float(np.mean(ties)), "max": float(np.max(ties))}, "n_ceiling_aurocs_energy_ctm": ceil,
            "id_error_rate_range": [float(np.min(errs)), float(np.max(errs))],
            "test_over_train_sigma_iso": ({"min": float(np.min(ratios)), "max": float(np.max(ratios)), "mean": float(np.mean(ratios))} if ratios else None)}


def run(b: int) -> None:
    rep = {"label": LABEL, "inventory": {}, "consistency": {}, "unrounded_sensitivity": {}, "per_example": {}}
    for tag in ("rn18", "vgg"):
        d = V2[tag]
        files = sorted(p for p in d.glob("*") if p.is_file() and not p.name.startswith("FAILED_"))
        recs = load(d)
        revs = sorted({r["v2"].get("environment", {}).get("extractor_sha256", "none") for r in recs.values()})
        rep["inventory"][tag] = {"n_json": sum(p.suffix == ".json" for p in files), "n_npz": sum(p.suffix == ".npz" for p in files),
                                 "n_failed": len(list(d.glob("FAILED_*"))), "extractor_revisions": revs,
                                 "id_methods": dict(pd.Series([r["v2"]["id_test"].get("id_method") for r in recs.values()]).value_counts()),
                                 "checkpoints": {s: {"epoch": r["v2"]["checkpoint"].get("epoch"), "file_sha256": r["v2"]["checkpoint"]["file_sha256"][:16],
                                                     "state_dict_digest": r["v2"]["checkpoint"]["state_dict_digest_loaded_module"][:16]} for s, r in recs.items()},
                                 "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]}
        v1recs = load(V1[tag])
        common = sorted(set(recs) & set(v1recs))
        cons = {s: field_consistency(v1recs[s], recs[s]) for s in common}
        agg = {}
        for g in next(iter(cons.values())):
            agg[g] = {"n_records": len(common), "n_fields_total": int(sum(c[g]["n_fields"] for c in cons.values())),
                      "n_fields_equal": int(sum(c[g]["n_equal"] for c in cons.values())),
                      "max_abs_diff": max(c[g]["max_abs_diff"] for c in cons.values()),
                      "worst": max(((c[g]["max_abs_diff"], s, c[g]["worst_field"]) for s, c in cons.items()), default=None)}
        rep["consistency"][tag] = {"n_common": len(common), "only_v1": sorted(set(v1recs) - set(recs)), "only_v2": sorted(set(recs) - set(v1recs)), "by_group": agg}
        rep["per_example"][tag] = per_example_summaries(list(recs.values()), d)
    # C. unrounded sensitivity (CE component, version-2 reader primitives, multiplier 1, descriptive)
    axes = v1.severity_axes()
    rn = list(load(V2["rn18"]).values()); vg = list(load(V2["vgg"]).values())
    import re
    for r in vg:
        r["run_label"] = int(re.search(r"_run(\d+)_", r["model_path"]).group(1)); r["component"] = "vgg_bridge"; r["paradigm"] = "confidnet"; r["dropout"] = 0
    df_u, vgg_u = unrounded_cells(rn, axes), unrounded_cells(vg, axes)
    df_r = v1.add_geometry_percentile(v1.cells_from_records(rn, axes, with_p10=True))
    vgg_r = v1.add_geometry_percentile(v1.cells_from_records(vg, axes, with_p10=False))
    ce_u, ce_r = df_u[df_u.component == "standalone_ce"], df_r[df_r.component == "standalone_ce"]
    sel_u = R.sel_endpoint(ce_u, Comparators(vgg_u, "dA"), None, "unrounded", ["descriptive"])
    sel_r = R.sel_endpoint(ce_r, Comparators(vgg_r, "dA"), None, "rounded (version-2 records)", ["descriptive"])
    lv_u, lv_r = R.level_endpoint(ce_u, None, "unrounded", ["descriptive"]), R.level_endpoint(ce_r, None, "rounded (version-2 records)", ["descriptive"])
    ho_u = v1.ho_endpoint(df_u, b)
    rec_of_record = json.loads((OUT / "rn18_report_v2.json").read_text())
    rep["unrounded_sensitivity"] = {
        "SEL_reference_contrast": {"unrounded": sel_u["comparators"][REFERENCE]["D_b"], "rounded_v2_records": sel_r["comparators"][REFERENCE]["D_b"],
                                   "corrected_readout": rec_of_record["SEL_ce"]["comparators"][REFERENCE]["D_b"],
                                   "ci_unrounded_mult1": sel_u["comparators"][REFERENCE]["ci"], "ci_corrected_readout_mult1": rec_of_record["SEL_ce"]["comparators"][REFERENCE]["ci"]},
        "SEL_mean_regret_P00": {"unrounded": sel_u["mean_regret_P00"], "rounded_v2_records": sel_r["mean_regret_P00"], "corrected_readout": rec_of_record["SEL_ce"]["mean_regret_P00"]},
        "SEL_P00_choice_counts_unrounded": sel_u["P00_choice_counts"], "SEL_material_unrounded": sel_u["material_subset"],
        "SEL_all_comparators_D_unrounded": {n: sel_u["comparators"][n].get("D_b") for n in NAMES},
        "LEVEL_delta": {"unrounded": lv_u["delta"], "rounded_v2_records": lv_r["delta"], "corrected_readout": rec_of_record["LEVEL_ce"]["delta"],
                        "ci_unrounded_mult1": lv_u["ci"], "ci_corrected_readout_mult1": rec_of_record["LEVEL_ce"]["ci"]},
        "HO_unrounded_verdicts": {ax: {s: v["verdict"] for s, v in ho_u[ax].items() if s != "global"} for ax in ("dK", "dF")},
        "HO_corrected_readout_verdicts": {ax: {s: v["verdict"] for s, v in rec_of_record["HO"][ax].items() if s != "global"} for ax in ("dK", "dF")},
        "note": "all three SEL/LEVEL columns are descriptive at multiplier 1; the reader of record used the rounded version-1 fields"}
    (OUT / "v2_panel_audit.json").write_text(json.dumps(rep, indent=1, default=str))
    small = {k: v for k, v in rep.items() if k != "inventory"} | {"inventory": {t: {k: v for k, v in i.items() if k not in ("files", "checkpoints")} for t, i in rep["inventory"].items()}}
    (OUT / "v2_panel_audit.md").write_text("# Version-2 panel: extraction-chain audit\n\n```\n" + json.dumps(small, indent=1, default=str) + "\n```\n")
    print(json.dumps(small, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--b", type=int, default=2000)
    run(ap.parse_args().b)

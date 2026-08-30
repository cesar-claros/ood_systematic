"""Phase-1 extraction diagnostics (saturation plan section 12 Phase 1,
required outputs: component occupancy, prototype-switching diagnostics,
direction-mixing magnitude, anisotropy, batch-size occupancy). Local;
descriptive only; NO detector outcome is read.

Reports, per checkpoint and pooled:
- integrity (sets returned, runtimes);
- ID nearest-prototype switch rate (train samples whose nearest
  prototype differs from their label);
- per-set component occupancy (raw/kept counts, other-component weight,
  top-component share, entropy) and the class-vs-prototype switch rate;
- the measured direction-mixing bias: kept-component max alignments
  against the global-mean max alignment (the plan's section-6.3
  quantity, now on real sets);
- anisotropy: max and mean prototype-direction variance over the
  isotropic average trS/D, for the ID within-class covariance and each
  set's global residual covariance (the plan's section-8.1 quantity);
- batch-occupancy grid medians (gate R6 support).

Usage (from code/): python repair_phase1_report.py
Output: nc_csf_predictivity/outputs/track1/repair_phase1_report.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crossing_robustness_audit import OUT_DIR

STATS_DIR = Path("pilot0/repair_phase1_stats")


def per_checkpoint(path: Path) -> dict:
    r = json.loads(path.read_text())
    z = np.load(path.with_suffix(".npz"))
    d = r["dim"]
    tr_id = r["id_scalars"]["id__trS_id"]
    dir_id = z["id__dir_id"]
    out = {
        "slug": r["slug"], "kind": r["kind"], "C": r["n_classes"],
        "D": d, "runtime_sec": r["runtime_sec"],
        "n_sets": len(r["sets"]),
        "id_prototype_switch_rate": r["id_prototype_switch_rate"],
        "id_dir_over_avg_max": float(dir_id.max() / (tr_id / d)),
        "id_dir_over_avg_mean": float(dir_id.mean() / (tr_id / d)),
        "sets": {},
    }
    for name, s in r["sets"].items():
        if "error" in s:
            out["sets"][name] = {"error": s["error"]}
            continue
        st = s["stats"]
        key = name.replace(" ", "_")
        n2g = st["global__n2"]
        a_glob = float(z[f"set__{key}__global__a"].max() / np.sqrt(n2g))
        comp_aligns, comp_weights = [], []
        i = 0
        while f"components__{i}__n" in st:
            if st[f"components__{i}__component"] >= 0:
                a_vec = z[f"set__{key}__components__{i}__a"]
                n2 = st[f"components__{i}__n2"]
                comp_aligns.append(float(a_vec.max() / np.sqrt(n2)))
                comp_weights.append(st[f"components__{i}__weight"])
            i += 1
        ca, cw = np.array(comp_aligns), np.array(comp_weights)
        tr_g = st["global__trS"]
        dir_g = z[f"set__{key}__global__dir"]
        batch = {k.split("__")[1]: v for k, v in st.items()
                 if k.startswith("batch_occupancy")}
        out["sets"][name] = {
            "n": st["diagnostics__n"],
            "n_raw": st["diagnostics__n_components_raw"],
            "n_kept": st["diagnostics__n_components_kept"],
            "other_weight": round(st["diagnostics__other_weight"], 4),
            "top_share": round(st["diagnostics__top_component_share"], 4),
            "class_switch": round(
                st["diagnostics__class_vs_prototype_switch_rate"], 4),
            "align_global": round(a_glob, 4),
            "align_comp_wmean": round(float((ca * cw).sum() / cw.sum()), 4)
            if len(ca) else None,
            "align_comp_max": round(float(ca.max()), 4) if len(ca)
            else None,
            "mixing_bias": round(float((ca * cw).sum() / cw.sum())
                                 - a_glob, 4) if len(ca) else None,
            "ood_dir_over_avg_max": round(float(dir_g.max()
                                                / (tr_g / d)), 2),
            "rho_shared": round(float(np.sqrt(
                st["resid_shared__trS"] / tr_id)), 4),
            "coords": {k: round(s["coords"][k], 4)
                       for k in ("gamma", "a", "rho")},
        }
        out["sets"][name]["batch_kept5pct"] = {
            b: v for b, v in
            ((b2.split("__")[0] if False else b2, v)
             for b2, v in batch.items()) if "n_components" in b}
    return out


def main() -> None:
    recs = [per_checkpoint(p) for p in sorted(STATS_DIR.glob("*.json"))
            if not p.name.startswith("FAILED")]
    pool = [r for r in recs if r["kind"] == "pool"]
    breeds = [r for r in recs if r["kind"] == "breeds"]

    def set_rows(rr):
        rows = []
        for r in rr:
            for name, s in r["sets"].items():
                if "error" not in s and name != "iid_test":
                    rows.append(s)
        return rows

    rows_p, rows_b = set_rows(pool), set_rows(breeds)

    def agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {"median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4)} if vals else None

    summary = {
        "n_checkpoints": len(recs), "n_pool": len(pool),
        "n_breeds": len(breeds),
        "n_ood_set_records": {"pool": len(rows_p), "breeds": len(rows_b)},
        "runtime_total_min": round(sum(r["runtime_sec"]
                                       for r in recs) / 60, 1),
        "id_prototype_switch_rate": agg(recs, "id_prototype_switch_rate"),
        "id_dir_over_avg_max": agg(recs, "id_dir_over_avg_max"),
        "pool_ood": {k: agg(rows_p, k) for k in
                     ("n_raw", "n_kept", "other_weight", "top_share",
                      "class_switch", "align_global", "align_comp_wmean",
                      "mixing_bias", "ood_dir_over_avg_max",
                      "rho_shared")},
        "breeds_ood": {k: agg(rows_b, k) for k in
                       ("n_raw", "n_kept", "other_weight", "top_share",
                        "class_switch", "align_global",
                        "align_comp_wmean", "mixing_bias",
                        "ood_dir_over_avg_max", "rho_shared")},
    }
    out = {"summary": summary, "checkpoints": recs}
    (OUT_DIR / "repair_phase1_report.json").write_text(
        json.dumps(out, indent=1, default=float))
    L = ["# Repair-campaign Phase-1 extraction diagnostics", "",
         "Descriptive; no detector outcome read. Frozen rules in "
         "pilot0/repair_stats.py; manifest sha256 e15752da...", "", "```",
         json.dumps(summary, indent=1), "```", ""]
    (OUT_DIR / "repair_phase1_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()

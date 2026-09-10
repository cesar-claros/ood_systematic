"""G0/G1-MC and A0/A1-TAYLOR diagnostics on the ResNet-18 four-shift panel
(status-review F4; DIAGNOSTIC, post-outcome, descriptive).

For every (checkpoint, new set) the driver takes the STORED inputs (head
W, b; raw CTM prototypes = normalized uncentered class means; ID model =
uncentered class means, within-class covariance, held-out class
probabilities; OOD model G0 = one Gaussian at the set mean with the set's
global covariance, G1 = the frozen nearest-prototype component mixture
with the shared residual covariance) and reports four quantities per
score, so the three explanations the review separates can be compared:

  observed AUROC (actual scores)                         [record]
  dictionary prediction (frozen corrected-dictionary arm) [record margins]
  G0-MC / G1-MC AUROC (fitted Gaussian model, MC, 20 batches, SE)
  A0 / A1-TAYLOR AUROC (quadratic surrogate under the same model)

and the errors: approximation = |TAYLOR - MC| (same fitted distribution);
distribution mismatch = |MC - observed|; mapping = |dictionary - observed|
and |dictionary - G1-MC|. Companion diagnostics: CTM branch switching
(fraction of MC draws whose arg-max prototype differs from the branch at
the component mean), analytic-undefined counts, PSD rejections of stored
covariances, and the ORIGIN diagnostic: ||global mean|| / R together with
a CENTERED variant of G1-MC (all means and prototypes translated by the
global mean; covariances unchanged), reported as the change it induces.

Panels: --panel-dir fourshift_rn18 (version-1 records, float32 covariances:
labeled PRELIMINARY) or fourshift_v2_rn18 (version-2, float64).

Usage (from code/):
    python rn18_handoff_replication/theory/run_gaussian_diagnostics.py --self-test
    python rn18_handoff_replication/theory/run_gaussian_diagnostics.py [--panel-dir fourshift_rn18] [--n 4096] [--batches 20] [--limit K]
Output: rn18_handoff_replication/outputs/gaussian_diagnostics_<panel>.json/.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CODE_ROOT))

from icml_campaign_analysis import frozen_margin, pred_auroc
from rn18_handoff_replication.theory.gaussian_diagnostics import (MASTER, AnalyticUndefined, ctm_score, energy_score, mc_aurocs,
                                                                  psd_sqrt, taylor_aurocs)

OUT = Path("rn18_handoff_replication/outputs")
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
LABEL = "DIAGNOSTIC (post-outcome, descriptive)"


def cell_key(slug: str, cname: str) -> int:
    return int(hashlib.sha256(f"{slug}|{cname}".encode()).hexdigest()[:8], 16)


def _norm_rows(P):
    return P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)


def branch_switch_rate(means, S, P, key: int, n: int = 2048) -> float:
    """Fraction of draws (one batch per component, equal weight) whose CTM
    arg-max prototype differs from the branch at the component mean."""
    rng = np.random.default_rng([MASTER, key, 991])
    rates = []
    for m in means:
        k0 = int(np.argmax(P @ m / max(np.linalg.norm(m), 1e-12)))
        h = m + rng.standard_normal((n, len(m))) @ S
        hn = h / np.clip(np.linalg.norm(h, axis=1, keepdims=True), 1e-12, None)
        rates.append(float(np.mean(np.argmax(hn @ P.T, axis=1) != k0)))
    return float(np.mean(rates))


def diagnose_set(rec: dict, z, cname: str, n: int, batches: int) -> dict:
    o = rec["ood"][cname]
    W, b = z["w"].astype(np.float64), z["b"].astype(np.float64)
    proto = z["proto_unc"].astype(np.float64)
    P = _norm_rows(proto)
    id_means = [proto[c] for c in range(len(proto))]
    counts = np.asarray(rec["iid_test"]["label_counts"], float); id_w = (counts / counts.sum()).tolist()
    R = float(rec["geometry"]["class_mean_radius"]); gm = z["global_mean"].astype(np.float64)
    out = {"observed": {"Energy": float(o["auroc_id_vs_ood_Energy"]), "CTM": float(o["auroc_id_vs_ood_CTM"])}}
    l_e, l_c = frozen_margin(rec, o)
    out["dictionary"] = {"Energy": pred_auroc(l_e), "CTM": pred_auroc(l_c)}
    out["origin_ratio_norm_gm_over_R"] = float(np.linalg.norm(gm) / R)
    key = cell_key(rec["slug"], cname)
    covs = {"sigma_w": z["sigma_w"], "cov_glob": z[f"set__{cname}__cov_glob"], "cov_res": z[f"set__{cname}__cov_res"]}
    out["covariance_dtype"] = {k: str(v.dtype) for k, v in covs.items()}
    S = {}
    for k, v in covs.items():
        try:
            S[k] = psd_sqrt(v.astype(np.float64))
        except ValueError as e:
            out.setdefault("psd_rejected", {})[k] = str(e)
    if "psd_rejected" in out:
        return out
    ood_mean = z[f"set__{cname}__ood_mean"].astype(np.float64)
    comps_mean = z[f"set__{cname}__comp_means"].astype(np.float64)
    weights = np.asarray(o["gaussian"]["weights"], float)
    assert len(weights) == len(comps_mean), (len(weights), len(comps_mean))
    G0 = [(ood_mean, 1.0)]
    G1 = [(comps_mean[i], float(weights[i])) for i in range(len(weights))]
    for tag, comps, Sood in (("G0", G0, S["cov_glob"]), ("G1", G1, S["cov_res"])):
        out[f"{tag}_MC"] = mc_aurocs(id_means, S["sigma_w"], id_w, comps, Sood, W, b, P, cell_id=key, n_batches=batches, n=n)
        out[f"A{tag[1]}_TAYLOR"] = taylor_aurocs(id_means, S["sigma_w"], id_w, comps, Sood, W, b, P, floor_scale=R)
    out["branch_switch"] = {"id": branch_switch_rate(id_means, S["sigma_w"], P, key), "ood_G1": branch_switch_rate([m for m, _ in G1], S["cov_res"], P, key + 1)}
    # origin diagnostic: centered variant of G1-MC (translate every mean and prototype by the global mean)
    Pc = _norm_rows(proto - gm)
    out["G1_MC_centered"] = mc_aurocs([m - gm for m in id_means], S["sigma_w"], id_w, [(m - gm, w) for m, w in G1], S["cov_res"], W, b, Pc, cell_id=key + 2, n_batches=batches, n=n)
    # error decomposition
    err = {}
    for sc in ("Energy", "CTM"):
        obs, dic = out["observed"][sc], out["dictionary"][sc]
        err[sc] = {"approximation_A0_vs_G0": (None if out["A0_TAYLOR"].get(sc) is None else abs(out["A0_TAYLOR"][sc] - out["G0_MC"][sc])),
                   "approximation_A1_vs_G1": (None if out["A1_TAYLOR"].get(sc) is None else abs(out["A1_TAYLOR"][sc] - out["G1_MC"][sc])),
                   "mismatch_G0_vs_observed": abs(out["G0_MC"][sc] - obs), "mismatch_G1_vs_observed": abs(out["G1_MC"][sc] - obs),
                   "mapping_dictionary_vs_observed": abs(dic - obs), "dictionary_vs_G1": abs(dic - out["G1_MC"][sc]),
                   "origin_effect_G1_centered_minus_uncentered": out["G1_MC_centered"][sc] - out["G1_MC"][sc]}
    gap_obs = out["observed"]["Energy"] - out["observed"]["CTM"]
    err["gap_sign_agreement"] = {"dictionary": bool(np.sign(out["dictionary"]["Energy"] - out["dictionary"]["CTM"]) == np.sign(gap_obs)),
                                 "G0_MC": bool(np.sign(out["G0_MC"]["diff_E_minus_C"]) == np.sign(gap_obs)),
                                 "G1_MC": bool(np.sign(out["G1_MC"]["diff_E_minus_C"]) == np.sign(gap_obs)),
                                 "A1_TAYLOR": (None if out["A1_TAYLOR"].get("CTM") is None else bool(np.sign(out["A1_TAYLOR"]["Energy"] - out["A1_TAYLOR"]["CTM"]) == np.sign(gap_obs))),
                                 "observed_gap": gap_obs, "material": bool(abs(gap_obs) >= 0.01)}
    out["errors"] = err
    return out


def summarize(cells: list[dict]) -> dict:
    ok = [c for c in cells if "psd_rejected" not in c["diag"]]
    rej = [c for c in cells if "psd_rejected" in c["diag"]]
    summ = {"n_cells": len(cells), "n_psd_rejected": len(rej), "psd_rejected": [(c["slug"], c["set"], c["diag"]["psd_rejected"]) for c in rej][:20]}
    by_src = {}
    for src in sorted({c["source"] for c in ok}):
        cs = [c for c in ok if c["source"] == src]
        e = lambda sc, k: [c["diag"]["errors"][sc][k] for c in cs if c["diag"]["errors"][sc][k] is not None]
        mat = [c for c in cs if c["diag"]["errors"]["gap_sign_agreement"]["material"]]
        agree = lambda k, pool: (float(np.mean([c["diag"]["errors"]["gap_sign_agreement"][k] for c in pool if c["diag"]["errors"]["gap_sign_agreement"][k] is not None])) if pool else None)
        by_src[src] = {"n": len(cs), "n_material": len(mat),
                       "mean_abs_error": {sc: {k: (float(np.mean(e(sc, k))) if e(sc, k) else None) for k in
                                               ("approximation_A0_vs_G0", "approximation_A1_vs_G1", "mismatch_G0_vs_observed", "mismatch_G1_vs_observed",
                                                "mapping_dictionary_vs_observed", "dictionary_vs_G1")} for sc in ("Energy", "CTM")},
                       "mean_origin_effect_G1_centered_minus_uncentered": {sc: float(np.mean(e(sc, "origin_effect_G1_centered_minus_uncentered"))) for sc in ("Energy", "CTM")},
                       "origin_ratio_mean": float(np.mean([c["diag"]["origin_ratio_norm_gm_over_R"] for c in cs])),
                       "gap_sign_agreement_material": {k: agree(k, mat) for k in ("dictionary", "G0_MC", "G1_MC", "A1_TAYLOR")},
                       "gap_sign_agreement_all": {k: agree(k, cs) for k in ("dictionary", "G0_MC", "G1_MC", "A1_TAYLOR")},
                       "A1_CTM_undefined": int(sum(c["diag"]["A1_TAYLOR"].get("CTM") is None for c in cs)),
                       "A0_CTM_undefined": int(sum(c["diag"]["A0_TAYLOR"].get("CTM") is None for c in cs)),
                       "branch_switch_mean": {k: float(np.mean([c["diag"]["branch_switch"][k] for c in cs])) for k in ("id", "ood_G1")},
                       "G1_MC_resolved_fraction": float(np.mean([c["diag"]["G1_MC"]["resolved"] for c in cs]))}
    summ["by_source"] = by_src
    return summ


def run(panel_dir: str, n: int, batches: int, limit: int | None) -> None:
    d = OUT / panel_dir
    files = sorted(p for p in d.glob("*.json") if not p.name.startswith("FAILED_"))
    if limit:
        files = files[:limit]
    prelim = panel_dir == "fourshift_rn18"
    cells, t0 = [], time.time()
    for i, p in enumerate(files, 1):
        rec = json.loads(p.read_text()); z = np.load(d / f"{p.stem}.npz")
        for cname in SHIFTS:
            if cname not in rec["ood"] or "error" in rec["ood"][cname]:
                continue
            cells.append({"slug": rec["slug"], "source": rec["source"], "component": rec.get("component"), "set": cname,
                          "diag": diagnose_set(rec, z, cname, n, batches)})
        print(f"[gauss-diag] {i}/{len(files)} {rec['slug']} ({time.time() - t0:.0f}s)", flush=True)
    rep = {"label": LABEL, "panel_dir": panel_dir, "preliminary_float32_covariances": prelim, "n": n, "batches": batches, "master_seed": MASTER,
           "summary": summarize(cells), "cells": cells}
    tag = panel_dir.replace("fourshift_", "")
    (OUT / f"gaussian_diagnostics_{tag}.json").write_text(json.dumps(rep, indent=1, default=str))
    (OUT / f"gaussian_diagnostics_{tag}.md").write_text(f"# Gaussian/Taylor diagnostics ({panel_dir}; {'PRELIMINARY float32' if prelim else 'float64'})\n\n```\n"
                                                       + json.dumps(rep["summary"], indent=1, default=str) + "\n```\n")
    print(json.dumps(rep["summary"], indent=1, default=str))


def self_test() -> None:
    rng = np.random.default_rng(3)
    C, D = 5, 16
    proto = rng.standard_normal((C, D)) * 3 + 1.0
    W = rng.standard_normal((C, D)); b = np.zeros(C)
    gm = proto.mean(0)
    rec = {"slug": "t", "n_classes": C, "dim": D, "papyan": {"var_collapse": 0.5, "self_duality": 0.3},
           "geometry": {"logit_scale": 5.0, "class_mean_radius_cv": 0.1, "class_mean_radius": float(np.linalg.norm(proto - gm, axis=1).mean())},
           "iid_test": {"label_counts": [10] * C},
           "ood": {"mnist_new": {"auroc_id_vs_ood_Energy": 0.8, "auroc_id_vs_ood_CTM": 0.7, "gamma": 0.5, "a": 0.6, "rho": 1.2,
                                 "gaussian": {"weights": [0.6, 0.4]}}}}
    z = {"w": W, "b": b, "proto_unc": proto, "global_mean": gm, "sigma_w": 0.1 * np.eye(D), "set__mnist_new__cov_glob": 0.5 * np.eye(D),
         "set__mnist_new__cov_res": 0.3 * np.eye(D), "set__mnist_new__ood_mean": gm + 2.0, "set__mnist_new__comp_means": np.stack([gm + 2.0, gm - 1.0])}
    out = diagnose_set(rec, z, "mnist_new", n=512, batches=4)
    for k in ("G0_MC", "G1_MC", "A0_TAYLOR", "A1_TAYLOR", "G1_MC_centered", "branch_switch", "errors"):
        assert k in out, k
    assert 0 <= out["G1_MC"]["Energy"] <= 1 and out["errors"]["Energy"]["mismatch_G1_vs_observed"] >= 0
    z2 = dict(z); z2["sigma_w"] = np.diag(np.r_[np.ones(D - 1), -1e-3])
    assert "psd_rejected" in diagnose_set(rec, z2, "mnist_new", 128, 2)
    s = summarize([{"slug": "t", "source": "cifar10", "component": "x", "set": "mnist_new", "diag": out}])
    assert s["by_source"]["cifar10"]["n"] == 1
    print(f"[gauss-diag] self-test PASS: G1-MC Energy {out['G1_MC']['Energy']:.3f} CTM {out['G1_MC']['CTM']}, A1 {out['A1_TAYLOR']}, "
          f"branch switch {out['branch_switch']}, PSD rejection detected")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel-dir", default="fourshift_rn18")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    self_test() if a.self_test else run(a.panel_dir, a.n, a.batches, a.limit)


if __name__ == "__main__":
    main()

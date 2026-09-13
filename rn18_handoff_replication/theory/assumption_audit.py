"""ASSUMPTION AUDIT of the perturbed-ETF reduction by matched comparisons
(recommendation B of the 2026-09-11 evaluation; DIAGNOSTIC, post-outcome).

For every (checkpoint, shift) cell of the version-2 ResNet-18 panel, the
scorer is held fixed (measured head W, b; measured training prototypes for
CTM) and the fitted Gaussian model is varied one factor at a time:

  ID means      : measured on TRAINING features | measured on ID-TEST features |
                  ETF-CONSTRAINED (closest equiangular, equinorm frame to the
                  training means in their own span, same radius; Procrustes)
  ID covariance : training within-class | test within-class |
                  isotropic with the training trace | isotropic with the test trace
  OOD model     : G0 (one Gaussian) | G1 (nearest-prototype mixture, shared
                  residual covariance)  [OOD statistics measured on the scored set]
  plus one origin check on the G1/train/train model: means and prototypes
  translated by the global mean WITH b' = b + W m (a pure coordinate change:
  Energy must be unchanged up to Monte Carlo noise; only CTM may move).

Each variant gives Monte Carlo AUROCs for Energy and CTM (10 paired batches
of 2,048 draws; master seed 2201; MC standard errors kept) and the paired
gap. Reported against the observed per-cell values: signed and absolute
level error per score, paired-gap error, winner-sign agreement on cells
material under the ID-versus-OOD AUROC gap, by source and variant, and the
one-factor-at-a-time effects.

DECISION RULE, declared before the run (thresholds in AUROC units):
  R1 "feature statistics generalize": a variant with test means AND test
     covariance has mean absolute level error <= 0.05 for both scores on at
     least three of four sources; otherwise the ID statistics do not
     transfer from training to test under either OOD model.
  R2 "the ETF mean restriction is material": for the best-fitting
     covariance/OOD choice, replacing measured training means by the ETF
     frame changes the mean absolute level error by >= 0.05 or the
     material-cell sign agreement by >= 0.10; below both, the mean geometry
     is not the limiting assumption.
  R3 "covariance shape matters": isotropic vs measured covariance (same
     trace, same means, same OOD model) differs by >= 0.05 in level error
     or >= 0.10 in sign agreement.
  R4 "a usable gap prediction exists": some variant reaches winner-sign
     agreement >= 0.75 on AUROC-material cells on every source; otherwise
     no fitted-Gaussian variant predicts winners.
The rule is evaluated mechanically in the summary; nothing is snapped.

Usage (from code/): python rn18_handoff_replication/theory/assumption_audit.py [--workers 4] [--n 2048] [--batches 10] [--limit K]
Output: rn18_handoff_replication/outputs/assumption_audit.json/.md
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

from pilot0.scores import auroc
from rn18_handoff_replication.theory.gaussian_diagnostics import MASTER, ctm_score, energy_score, psd_sqrt

OUT = Path("rn18_handoff_replication/outputs")
V2 = OUT / "fourshift_v2_rn18"
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
MEANS = ("train", "test", "etf")
COVS = ("train", "test", "iso_train", "iso_test")
OODS = ("G0", "G1")
LABEL = "ASSUMPTION AUDIT (post-outcome, diagnostic)"
MAT = 0.01
RULE = {"R1_level_err": 0.05, "R1_min_sources": 3, "R2_delta_err": 0.05, "R2_delta_sign": 0.10, "R3_delta_err": 0.05, "R3_delta_sign": 0.10, "R4_sign_agreement": 0.75}


def cell_key(slug: str, cname: str, tag: str) -> int:
    return int(hashlib.sha256(f"{slug}|{cname}|{tag}".encode()).hexdigest()[:8], 16)


def _norm_rows(P):
    return P / np.clip(np.linalg.norm(P, axis=1, keepdims=True), 1e-12, None)


def etf_means(M_centered: np.ndarray, gm: np.ndarray, R: float) -> np.ndarray:
    """Closest simplex ETF (in the Frobenius sense, Procrustes) to the centered
    class means within their own span, radius R; returned uncentered."""
    C = M_centered.shape[0]
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    B = Vt[:C - 1].T                                      # D x (C-1) basis of the mean span
    Mc = M_centered @ B                                   # C x (C-1) coordinates
    E = np.eye(C) - np.ones((C, C)) / C                   # simplex directions in R^C
    Q, _ = np.linalg.qr(E[:, :C - 1] if False else E)     # not used
    Ue, Se, Vte = np.linalg.svd(E, full_matrices=False)
    E0 = (Ue[:, :C - 1] * Se[:C - 1])                     # C x (C-1) coordinates of the simplex (rank C-1)
    E0 = E0 / np.linalg.norm(E0, axis=1, keepdims=True) * R
    A = E0.T @ Mc                                         # (C-1) x (C-1)
    Ua, _, Vta = np.linalg.svd(A)
    Omega = Ua @ Vta                                      # rotation aligning E0 to Mc
    return (E0 @ Omega) @ B.T + gm


def mc_pair(id_means, S_id, id_w, comps, S_ood, W, b, P, key, n, batches):
    C, D = len(id_means), len(id_means[0]); ood_w = np.array([w for _, w in comps]); comp_m = np.asarray([m for m, _ in comps])
    e, c = [], []
    for bt in range(batches):
        ss = np.random.SeedSequence([MASTER, key, bt]); r_cls, r_id, r_ood, r_comp = [np.random.default_rng(s) for s in ss.spawn(4)]
        yc = r_cls.choice(C, size=n, p=id_w); h_id = np.asarray(id_means)[yc] + r_id.standard_normal((n, D)) @ S_id
        kc = r_comp.choice(len(comps), size=n, p=ood_w / ood_w.sum()); h_ood = comp_m[kc] + r_ood.standard_normal((n, D)) @ S_ood
        e.append(auroc(energy_score(h_id, W, b), energy_score(h_ood, W, b))); c.append(auroc(ctm_score(h_id, P), ctm_score(h_ood, P)))
    e, c = np.array(e), np.array(c)
    return {"Energy": float(e.mean()), "Energy_se": float(e.std(ddof=1) / np.sqrt(batches)), "CTM": float(c.mean()), "CTM_se": float(c.std(ddof=1) / np.sqrt(batches)), "gap": float((c - e).mean())}


def audit_record(path: Path, n: int, batches: int) -> list[dict]:
    rec = json.loads(path.read_text()); z = np.load(V2 / f"{path.stem}.npz")
    W, b = z["w"].astype(np.float64), z["b"].astype(np.float64)
    gm_tr, gm_te = z["global_mean"].astype(np.float64), z["id__global_mean_test"].astype(np.float64)
    mc_tr, mc_te = z["class_means_centered"].astype(np.float64), z["id__class_means_centered_test"].astype(np.float64)
    R = float(rec["geometry"]["class_mean_radius"])
    means = {"train": mc_tr + gm_tr, "test": mc_te + gm_te, "etf": etf_means(mc_tr, gm_tr, R)}
    P = _norm_rows(means["train"])                              # frozen scorer: training prototypes
    counts = np.asarray(rec["iid_test"]["label_counts"], float); id_w = (counts / counts.sum()).tolist()
    covs = {"train": z["sigma_w"].astype(np.float64), "test": z["id__sigma_w_test"].astype(np.float64)}
    D = W.shape[1]
    covs["iso_train"] = np.eye(D) * np.trace(covs["train"]) / D; covs["iso_test"] = np.eye(D) * np.trace(covs["test"]) / D
    S = {k: psd_sqrt(v) for k, v in covs.items()}
    out = []
    for cname in SHIFTS:
        o = rec["ood"][cname]; u = rec["v2"]["ood_unrounded"][cname]
        obs = {"Energy": u["auroc_id_vs_ood"]["Energy"], "CTM": u["auroc_id_vs_ood"]["CTM"]}
        ood_mean = z[f"set__{cname}__ood_mean"].astype(np.float64); comps_m = z[f"set__{cname}__comp_means"].astype(np.float64)
        wts = np.asarray(o["gaussian"]["weights"], float)
        oodm = {"G0": ([(ood_mean, 1.0)], psd_sqrt(z[f"set__{cname}__cov_glob"].astype(np.float64))),
                "G1": ([(comps_m[i], float(wts[i])) for i in range(len(wts))], psd_sqrt(z[f"set__{cname}__cov_res"].astype(np.float64)))}
        row = {"slug": rec["slug"], "source": rec["source"], "component": rec["component"], "set": cname, "observed": obs, "observed_gap": obs["CTM"] - obs["Energy"], "variants": {}}
        for mk in MEANS:
            for ck in COVS:
                for ok in OODS:
                    comps, S_ood = oodm[ok]
                    row["variants"][f"{mk}|{ck}|{ok}"] = mc_pair([means[mk][i] for i in range(len(means[mk]))], S[ck], id_w, comps, S_ood, W, b, P, cell_key(rec["slug"], cname, f"{mk}{ck}{ok}"), n, batches)
        # origin check with a pure coordinate change (b' = b + W m)
        m0 = gm_tr; comps, S_ood = oodm["G1"]
        row["variants"]["train|train|G1|centered_bprime"] = mc_pair([means["train"][i] - m0 for i in range(len(means["train"]))], S["train"], id_w, [(mm - m0, w) for mm, w in comps], S_ood, W, b + W @ m0, _norm_rows(means["train"] - m0), cell_key(rec["slug"], cname, "centered"), n, batches)
        out.append(row)
    return out


def summarize(cells: list[dict]) -> dict:
    sources = sorted({c["source"] for c in cells}); variants = sorted({v for c in cells for v in c["variants"]})
    by = {}
    for v in variants:
        by[v] = {}
        for src in sources + ["all"]:
            cs = [c for c in cells if src == "all" or c["source"] == src]
            errE = np.array([c["variants"][v]["Energy"] - c["observed"]["Energy"] for c in cs]); errC = np.array([c["variants"][v]["CTM"] - c["observed"]["CTM"] for c in cs])
            gap_err = np.array([c["variants"][v]["gap"] - c["observed_gap"] for c in cs])
            mat = np.array([abs(c["observed_gap"]) >= MAT for c in cs])
            sgn = np.array([np.sign(c["variants"][v]["gap"]) == np.sign(c["observed_gap"]) for c in cs])
            by[v][src] = {"n": len(cs), "bias_E": float(errE.mean()), "bias_C": float(errC.mean()), "mae_E": float(np.abs(errE).mean()), "mae_C": float(np.abs(errC).mean()),
                          "gap_mae": float(np.abs(gap_err).mean()), "sign_agree_material": (float(sgn[mat].mean()) if mat.any() else None), "n_material": int(mat.sum())}
    # decision rule
    def ok_R1(v):
        return sum(1 for s in sources if by[v][s]["mae_E"] <= RULE["R1_level_err"] and by[v][s]["mae_C"] <= RULE["R1_level_err"]) >= RULE["R1_min_sources"]
    test_variants = [v for v in variants if v.startswith("test|test|")]
    R1 = {v: ok_R1(v) for v in test_variants}
    best = min((v for v in variants if "|" in v and v.count("|") == 2), key=lambda v: by[v]["all"]["mae_E"] + by[v]["all"]["mae_C"])
    mk, ck, ok = best.split("|")
    R2 = {}
    for ck2 in COVS:
        for ok2 in OODS:
            a, e = by[f"train|{ck2}|{ok2}"]["all"], by[f"etf|{ck2}|{ok2}"]["all"]
            R2[f"{ck2}|{ok2}"] = {"d_err": (e["mae_E"] + e["mae_C"] - a["mae_E"] - a["mae_C"]) / 2, "d_sign": (e["sign_agree_material"] or 0) - (a["sign_agree_material"] or 0)}
    R3 = {}
    for mk2 in MEANS:
        for ok2 in OODS:
            for pair in (("train", "iso_train"), ("test", "iso_test")):
                a, e = by[f"{mk2}|{pair[0]}|{ok2}"]["all"], by[f"{mk2}|{pair[1]}|{ok2}"]["all"]
                R3[f"{mk2}|{pair[0]}vs{pair[1]}|{ok2}"] = {"d_err": (e["mae_E"] + e["mae_C"] - a["mae_E"] - a["mae_C"]) / 2, "d_sign": (e["sign_agree_material"] or 0) - (a["sign_agree_material"] or 0)}
    R4 = {v: all((by[v][s]["sign_agree_material"] or 0) >= RULE["R4_sign_agreement"] for s in sources) for v in variants}
    origin = {s: {"dE": by["train|train|G1|centered_bprime"][s]["bias_E"] - by["train|train|G1"][s]["bias_E"], "dC": by["train|train|G1|centered_bprime"][s]["bias_C"] - by["train|train|G1"][s]["bias_C"]} for s in sources}
    verdict = {"R1_feature_statistics_generalize": {v: r for v, r in R1.items()},
               "R2_etf_mean_restriction_material": {k: (abs(x["d_err"]) >= RULE["R2_delta_err"] or abs(x["d_sign"]) >= RULE["R2_delta_sign"]) for k, x in R2.items()},
               "R3_covariance_shape_matters": {k: (abs(x["d_err"]) >= RULE["R3_delta_err"] or abs(x["d_sign"]) >= RULE["R3_delta_sign"]) for k, x in R3.items()},
               "R4_usable_gap_prediction": [v for v, r in R4.items() if r], "best_level_variant": best}
    return {"rule": RULE, "by_variant": by, "R2_detail": R2, "R3_detail": R3, "origin_check_bprime": origin, "verdict": verdict}


def _job(args):
    p, n, batches = args
    return audit_record(Path(p), n, batches)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--n", type=int, default=2048); ap.add_argument("--batches", type=int, default=10); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        rng = np.random.default_rng(0); M = rng.standard_normal((6, 12)); M -= M.mean(0); E = etf_means(M, np.zeros(12), 2.0)
        G = E @ E.T; off = G[~np.eye(6, dtype=bool)]
        assert np.allclose(np.linalg.norm(E, axis=1), 2.0) and np.allclose(off, off[0], atol=1e-9) and np.allclose(E.sum(0), 0, atol=1e-9)
        print("[assumption-audit] self-test PASS: ETF-constrained means are equinorm, equiangular, zero-mean"); return
    files = sorted(p for p in V2.glob("*.json") if not p.name.startswith("FAILED_"))
    if a.limit:
        files = files[:a.limit]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cells, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_job, (str(p), a.n, a.batches)): p for p in files}
        for i, f in enumerate(as_completed(futs), 1):
            cells.extend(f.result()); print(f"[assumption-audit] {i}/{len(files)} {futs[f].stem} ({time.time() - t0:.0f}s)", flush=True)
    cells.sort(key=lambda c: (c["slug"], c["set"]))
    rep = {"label": LABEL, "n": a.n, "batches": a.batches, "master_seed": MASTER, "means": MEANS, "covariances": COVS, "ood_models": OODS, "summary": summarize(cells), "cells": cells}
    (OUT / "assumption_audit.json").write_text(json.dumps(rep, indent=1, default=str))
    small = {k: v for k, v in rep["summary"].items() if k != "by_variant"} | {"by_variant_all": {v: d["all"] for v, d in rep["summary"]["by_variant"].items()}}
    (OUT / "assumption_audit.md").write_text("# Assumption audit (matched comparisons)\n\n```\n" + json.dumps(small, indent=1, default=str) + "\n```\n")
    print(json.dumps(rep["summary"]["verdict"], indent=1, default=str))


if __name__ == "__main__":
    main()

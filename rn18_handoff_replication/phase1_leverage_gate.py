"""RN18 handoff-replication plan v3, PHASE 1 (local): geometry-leverage
gate on training features only (v3 section 4). Reads
outputs/phase1_nc1/*.json (+ .npz) and reports, per source and panel:

- within-source NC1 range and max-to-min ratio (deterministic view);
- Q_g = sd_j(log NC1_j) / median_j sd_b(log NC1_j^(b)), ddof 1 both;
- pairwise order stability p_ij over the common bootstrap (half credit
  on ties) and the fraction of pairs with distinct g and p_ij >= 0.8;
- tertile sizes under the thirds rule; label-hash agreement (common
  resample validity); the TinyImageNet pseudoinverse tolerance grid
  (v2 section 5.6) recomputed from the stored Sigma_W and class means.

Panels: full (paradigm pool + standalone CE), paradigm_pool, ce_all,
ce_do0. HO eligibility per source uses the full panel: ratio >= 2,
Q_g >= 2, >= 80% stable distinct pairs, every tertile >= 5. The
section-3.4 expansion trigger reads ce_do0: ratio >= 1.5 and Q_g >= 2.

Usage (from code/): python rn18_handoff_replication/phase1_leverage_gate.py [--self-test]
Output: rn18_handoff_replication/outputs/phase1_leverage_report.json/.md
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

IN_DIR = Path("rn18_handoff_replication/outputs/phase1_nc1")
OUT_JSON = Path("rn18_handoff_replication/outputs/phase1_leverage_report.json")
TOL_GRID = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
PANELS = {"full": lambda r: True,
          "paradigm_pool": lambda r: r["component"] == "paradigm_pool",
          "ce_all": lambda r: r["component"] == "standalone_ce",
          "ce_do0": lambda r: r["component"] == "standalone_ce" and r["dropout"] == 0}


def nc1_at_tol(sigma_w, means_c, tol) -> float:
    C = len(means_c)
    sb = means_c.T @ means_c / C
    sb = (sb + sb.T) / 2
    return float(np.trace(sigma_w @ np.linalg.pinv(sb, rcond=tol, hermitian=True)) / C)


def panel_stats(recs: list[dict], boots: dict, arrays: dict) -> dict:
    g = np.array([np.log(r["deterministic_view"]["nc1_corrected"]) for r in recs])
    nc1 = np.exp(g)
    n = len(recs)
    out = {"n": n, "nc1_min": float(nc1.min()), "nc1_max": float(nc1.max()),
           "ratio_max_min": float(nc1.max() / nc1.min()),
           "labels_hash_agree": len({r["deterministic_view"]["labels_sha256"]
                                     for r in recs}) == 1}
    if n < 2:
        out.update(Q_g=None, frac_stable_pairs=None, tertiles=None, eligible=False)
        return out
    sd_obs = float(np.std(g, ddof=1))
    B = np.array([boots[r["slug"]] for r in recs])          # (n, Bboot)
    lb = np.log(B)
    med_se = float(np.median(np.std(lb, axis=1, ddof=1)))
    q = (np.inf if med_se == 0 and sd_obs > 0 else 0.0 if med_se == 0 else sd_obs / med_se)
    stable = distinct = 0
    for i, j in combinations(range(n), 2):
        if g[i] == g[j]:
            continue
        distinct += 1
        s = np.sign(g[i] - g[j])
        pres = np.sign(lb[i] - lb[j])
        p = float(np.mean(np.where(pres == 0, 0.5, pres == s)))
        stable += int(p >= 0.8)
    frac = stable / max(len(list(combinations(range(n), 2))), 1)
    order = np.argsort(g)
    t = [len(order[: n // 3]), len(order[n // 3: 2 * n // 3]), len(order[2 * n // 3:])]
    out.update(Q_g=float(q), frac_stable_pairs=float(frac), tertiles=t,
               eligible=bool(out["ratio_max_min"] >= 2 and q >= 2 and frac >= 0.8
                             and min(t) >= 5))
    ratios = []
    for tol in TOL_GRID:
        vals = [nc1_at_tol(arrays[r["slug"]]["sigma_w"],
                           arrays[r["slug"]]["class_means_centered"], tol) for r in recs]
        ratios.append(vals)
    R = np.array(ratios)                                         # (tol, n)
    per_ckpt = R.max(0) / np.where(R.min(0) > 0, R.min(0), np.nan)
    out["tolerance_grid_median_ratio"] = float(np.nanmedian(per_ckpt))
    return out


def analyze(recs, boots, arrays) -> dict:
    report = {"n_records": len(recs), "sources": {}}
    for s in sorted({r["source"] for r in recs}):
        rs = [r for r in recs if r["source"] == s]
        report["sources"][s] = {p: panel_stats([r for r in rs if f(r)], boots, arrays)
                                for p, f in PANELS.items()}
    report["HO_eligible_sources"] = [s for s, v in report["sources"].items()
                                     if v["full"]["eligible"]]
    ce = {s: v["ce_do0"] for s, v in report["sources"].items()}
    report["expansion_trigger_ce_do0"] = {
        s: bool(v["ratio_max_min"] >= 1.5 and (v["Q_g"] or 0) >= 2) for s, v in ce.items()}
    report["four_source_nc1_ineligible"] = any(
        v["full"].get("tolerance_grid_median_ratio", 0) > 10 for v in report["sources"].values())
    return report


def load(in_dir: Path):
    recs, boots, arrays = [], {}, {}
    for p in sorted(in_dir.glob("*.json")):
        if p.name.startswith("FAILED_"):
            continue
        r = json.loads(p.read_text())
        z = np.load(in_dir / f"{p.stem}.npz")
        recs.append(r)
        boots[r["slug"]] = z["boot_nc1"]
        arrays[r["slug"]] = {"sigma_w": z["sigma_w"],
                             "class_means_centered": z["class_means_centered"]}
    return recs, boots, arrays


def self_test() -> None:
    rng = np.random.default_rng(3)
    recs, boots, arrays = [], {}, {}
    C, D = 6, 16
    for k in range(12):
        comp = "paradigm_pool" if k < 7 else "standalone_ce"
        nc1 = float(np.exp(rng.uniform(-2, 1)))
        slug = f"s{k}"
        recs.append({"slug": slug, "source": "cifar10", "component": comp,
                     "dropout": k % 2,
                     "deterministic_view": {"nc1_corrected": nc1, "labels_sha256": "h"}})
        boots[slug] = nc1 * np.exp(rng.normal(0, 0.02, 100))
        M = rng.standard_normal((C, D)); M -= M.mean(0)
        arrays[slug] = {"sigma_w": np.eye(D) * nc1, "class_means_centered": M}
    rep = analyze(recs, boots, arrays)
    st = rep["sources"]["cifar10"]["full"]
    assert st["n"] == 12 and st["Q_g"] > 2 and st["frac_stable_pairs"] > 0.8, st
    assert st["tertiles"] == [4, 4, 4] and st["eligible"] is False  # tertile < 5
    assert rep["sources"]["cifar10"]["ce_do0"]["n"] == 2  # k in {8, 10}
    print("[gate] self-test PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    recs, boots, arrays = load(IN_DIR)
    rep = analyze(recs, boots, arrays)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1, default=float))
    OUT_JSON.with_suffix(".md").write_text(
        "# Phase-1 geometry-leverage gate (training features only)\n\n```\n"
        + json.dumps(rep, indent=1, default=float) + "\n```\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "sources"}, indent=1))
    for s, v in rep["sources"].items():
        f = v["full"]
        print(f"{s}: n={f['n']} ratio={f['ratio_max_min']:.2f} Q_g={f['Q_g']} "
              f"stable={f['frac_stable_pairs']} tertiles={f['tertiles']} "
              f"eligible={f['eligible']} | ce_do0 ratio="
              f"{v['ce_do0']['ratio_max_min']:.2f} Q_g={v['ce_do0']['Q_g']}")


if __name__ == "__main__":
    main()

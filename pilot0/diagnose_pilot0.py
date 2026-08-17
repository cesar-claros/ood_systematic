"""Per-cell diagnosis of a Pilot 0 run from its result JSONs.

Reads pilot0_<slug>.json files (no caches needed) and decomposes the
G2/G3a failure: sign agreement, delta correlation, and regression slope
per (predictor arm x score x rotation kind), plus a parameter-free
z-space response transport: the predicted response is expressed in
probit space and re-anchored at the EMPIRICAL baseline operating point,

    delta_z      = probit(pred_state) - probit(pred_baseline)
    delta_recal  = Phi(probit(emp_baseline) + delta_z) - emp_baseline.

This uses only the measured pre-intervention baseline level (available in
Pilot 1/2 by design), so it is a registerable plug-in variant, not
outcome peeking; whether to adopt it is decided from this diagnosis and
frozen in the manifest.

Usage (from code/):
    python pilot0/diagnose_pilot0.py pilot0/outputs/pilot0_<slug>.json [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr

ARMS = ("iso", "emp")
KINDS = ("toward", "away")
SCORES = ("MSR", "MLS", "Energy", "CTM_head")
Z_CLIP = (1e-6, 1.0 - 1e-6)


def _kind(state_name: str) -> str:
    return "toward" if state_name.startswith("toward") else "away"


def _summarize(cells: list[dict], arm: str) -> dict[str, float]:
    d_emp = np.array([c["delta_emp"] for c in cells])
    d_pred = np.array([c[f"delta_pred_{arm}"] for c in cells])
    material = np.array([c[f"material_{arm}"] for c in cells])
    out = {
        "n": len(cells),
        "mae_pred": float(np.abs(d_pred - d_emp).mean()),
        "mae_const": float(np.abs(d_emp).mean()),
        "mean_abs_dpred": float(np.abs(d_pred).mean()),
        "mean_abs_demp": float(np.abs(d_emp).mean()),
    }
    if material.sum():
        out["sign_material"] = float(
            (np.sign(d_pred[material]) == np.sign(d_emp[material])).mean())
        out["n_material"] = int(material.sum())
    if len(cells) > 2 and d_pred.std() > 0 and d_emp.std() > 0:
        out["corr"] = float(pearsonr(d_pred, d_emp).statistic)
        out["slope_emp_on_pred"] = float(
            np.polyfit(d_pred, d_emp, 1)[0])
    return out


def _z_transport(cells: list[dict], arm: str) -> list[dict]:
    """Replace each cell's predicted delta by the z-space transported one."""
    recal = []
    for c in cells:
        pred_state = np.clip(c[f"auroc_pred_{arm}"], *Z_CLIP)
        pred_base = np.clip(
            c[f"auroc_pred_{arm}"] - c[f"delta_pred_{arm}"], *Z_CLIP)
        emp_base = np.clip(c["auroc_emp"] - c["delta_emp"], *Z_CLIP)
        delta_z = norm.ppf(pred_state) - norm.ppf(pred_base)
        delta_recal = float(
            norm.cdf(norm.ppf(emp_base) + delta_z) - emp_base)
        cc = dict(c)
        cc[f"delta_pred_{arm}"] = delta_recal
        cc[f"auroc_pred_{arm}"] = emp_base + delta_recal
        recal.append(cc)
    return recal


def _print_block(title: str, stats: dict[str, float]) -> None:
    sign = stats.get("sign_material")
    corr = stats.get("corr")
    slope = stats.get("slope_emp_on_pred")
    print(f"  {title:28s} n={stats['n']:4d} "
          f"maeP={stats['mae_pred']:.4f} maeC={stats['mae_const']:.4f} "
          f"|dP|={stats['mean_abs_dpred']:.4f} |dE|={stats['mean_abs_demp']:.4f} "
          + (f"sign={sign:.3f}@{stats['n_material']}" if sign is not None
             else "sign=--")
          + (f" corr={corr:+.3f}" if corr is not None else "")
          + (f" slope={slope:+.3f}" if slope is not None else ""))


def diagnose(path: Path) -> None:
    result = json.loads(path.read_text())
    cells = result["cells"]
    print(f"\n=== {result['slug']} ===")
    print("geometry:", {k: round(v, 3)
                        for k, v in result["geometry_baseline"].items()
                        if k in ("self_duality_angle_mean_deg", "snr",
                                 "eig_max_over_mean", "effective_rank",
                                 "head_residual_fraction")})
    for arm in ARMS:
        for label, cell_set in (("raw", cells),
                                ("z-transport", _z_transport(cells, arm))):
            print(f"[arm={arm} | {label}]")
            _print_block("ALL", _summarize(cell_set, arm))
            for score in SCORES:
                sub = [c for c in cell_set if c["score"] == score]
                _print_block(f"score={score}", _summarize(sub, arm))
            for kind in KINDS:
                sub = [c for c in cell_set if _kind(c["state"]) == kind]
                _print_block(f"kind={kind}", _summarize(sub, arm))
        print()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: diagnose_pilot0.py <pilot0_*.json> [...]")
    for arg in sys.argv[1:]:
        diagnose(Path(arg))


if __name__ == "__main__":
    main()

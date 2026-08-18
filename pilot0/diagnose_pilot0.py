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

With --caches, additionally runs the w_perp LEVEL-ORDERING check for the
reconstruction family (first real-feature contact for the X1 pass-2
proposition, [[X1_pass2_wperp_reconstruction]]): per OOD set, the
predicted complement-energy AUROC is computed from the DEPLOYED subspace
(effective w_perp convention) with isotropic-within-complement noise
(declared approximation), and compared to the empirical PCA_RE/Residual
AUROC. The claim tested is the ORDERING across OOD sets (Spearman), not
absolute calibration.

Usage (from code/):
    python pilot0/diagnose_pilot0.py pilot0/outputs/pilot0_<slug>.json [...]
    python pilot0/diagnose_pilot0.py --caches pilot0/caches/<slug>.npz [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _complement_stats(h: np.ndarray, offset: np.ndarray,
                      basis: np.ndarray, keep_is_basis: bool
                      ) -> tuple[np.ndarray, int]:
    """Per-sample complement energy and complement dimension.

    keep_is_basis=True means `basis` spans the KEPT subspace (PCA_RE:
    complement energy = total - kept); False means `basis` spans the
    complement itself (Residual's res_basis).
    """
    x = h.astype(np.float64) - offset
    if keep_is_basis:
        energy = (x**2).sum(1) - ((x @ basis) ** 2).sum(1)
        d_perp = x.shape[1] - basis.shape[1]
    else:
        energy = ((x @ basis) ** 2).sum(1)
        d_perp = basis.shape[1]
    return energy, d_perp


def wperp_level_check(cache: dict) -> dict[str, dict]:
    """Ordering check: predicted vs empirical reconstruction AUROC levels.

    Returns per-family records with predicted/empirical AUROC lists and
    the ordering Spearman, for tests and post-hoc analysis.
    """
    from pilot0.run_pilot0 import PCAFamily, auroc

    slug = cache["meta"].get("slug", "checkpoint")
    n_classes = int(cache["meta"].get("n_classes", cache["w"].shape[0]))
    ood_names = list(cache["meta"]["ood_sets"])
    pca = PCAFamily(cache["h_train"], n_classes, cache["w"], cache["b"])
    print(f"\n=== w_perp level-ordering check: {slug} ===")
    families = (
        ("PCA_RE", pca.pca_re, pca.mean, pca.re_basis, True),
        ("Residual", pca.residual, pca.u, pca.res_basis, False),
    )
    out: dict[str, dict] = {}
    for name, scorer, offset, basis, keep in families:
        e_train, d_perp = _complement_stats(cache["h_train"], offset,
                                            basis, keep)
        sigma2 = float(e_train.mean()) / d_perp
        sc_id = scorer(cache["h_iid_test"])
        preds, emps, lams = [], [], []
        print(f"[{name}] D'={d_perp}")
        for ood in ood_names:
            h_o = cache[f"h_{ood}"].astype(np.float64)
            m_o = h_o.mean(0)
            lam = float(_complement_stats(m_o[None, :], offset, basis,
                                          keep)[0][0]) / sigma2
            e_resid = _complement_stats(h_o - m_o, np.zeros_like(offset),
                                        basis, keep)[0]
            rho2 = float(e_resid.mean()) / d_perp / sigma2
            pred = float(norm.cdf(
                ((rho2 - 1.0) * d_perp + lam)
                / np.sqrt(2.0 * d_perp * (1.0 + rho2**2)
                          + 4.0 * rho2 * lam)))
            emp = auroc(sc_id, scorer(cache[f"h_{ood}"]))
            preds.append(pred)
            emps.append(emp)
            lams.append(lam)
            print(f"  {ood:26s} lam_perp={lam:9.1f} rho2={rho2:5.2f} "
                  f"pred={pred:.4f} emp={emp:.4f}")
        record = {"ood_sets": ood_names, "pred": preds, "emp": emps,
                  "lam_perp": lams}
        if len(ood_names) > 2:
            record["spearman_pred_emp"] = float(
                spearmanr(preds, emps).statistic)
            record["spearman_lam_emp"] = float(
                spearmanr(lams, emps).statistic)
            print(f"  ordering Spearman(pred, emp) = "
                  f"{record['spearman_pred_emp']:+.3f}; "
                  f"model-free Spearman(lam_perp, emp) = "
                  f"{record['spearman_lam_emp']:+.3f}")
        out[name] = record
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-cell diagnosis of Pilot 0 results")
    parser.add_argument("jsons", nargs="*", help="pilot0_<slug>.json paths")
    parser.add_argument("--caches", nargs="*", default=[],
                        help="extraction NPZs for the w_perp level check")
    args = parser.parse_args()
    if not args.jsons and not args.caches:
        parser.error("provide result JSONs and/or --caches NPZs")
    for arg in args.jsons:
        diagnose(Path(arg))
    if args.caches:
        from pilot0.run_pilot0 import load_cache
        for path in args.caches:
            wperp_level_check(load_cache(Path(path)))


if __name__ == "__main__":
    main()

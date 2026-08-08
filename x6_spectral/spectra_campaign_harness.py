"""Measurement and prediction harness for the X6 real-spectra campaign (v3).

Rewritten after the re-review, which found the v2 harness invalid: its
class-dependent-residue statistic was algebraically zero, its global rule
predicted benefit from recoverability alone (contradicting the orientation
identifiability result, X6 section 2.5), one rule was applied to score
families with opposite capture dependence, and the census default did not
match the deployed preprocessing (the pipeline only centers:
TorchStandardScaler's std division is commented out in code/src/csfs/_utils.py).

Structure, per X6 section 2.5:

  Tier A (ID-only). Emits recovery and stability statements per variant plus
  ONE falsifiable one-sided prediction: where recovery fails, projection
  should not significantly help. Where recovery holds, the benefit sign is
  UNDETERMINED from ID data alone and no sign is emitted.

  Tier B (deployment-batch adaptation; explicitly OOD-side). Given a small
  unlabeled batch from the evaluation shift, estimates the displacement's
  kept/complement energy split per projector and applies the exact
  per-operator crossing rules: kept-space distance benefits iff
  a_hat > a*(lam_hat, q, D); complement/reconstruction scores benefit iff
  a_hat < a_x (the kept-vs-complement flip); projected-logit scores require
  classifier-row-space alignment of the displacement.

Census default is standardize=False (centered covariance, implementation
faithful); standardized spectra are a robustness arm only. The operator map
below is pre-registered but may be adjusted with written justification
BEFORE the freeze; the Delta-baseline semantics per CSF family in
projection_targets.csv must be pinned at campaign kickoff, before any
held-out outcome table is inspected.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncf

from spectral_diagnostics import (class_projector_heterogeneity,
                                  common_mode_fraction, spike_census,
                                  split_half_stability, viability,
                                  weight_top_class_alignment, weight_top_gap)

#: Operator classes; "kept" = distance/similarity in retained coordinates,
#: "complement" = reconstruction/residual energy in rejected coordinates,
#: "logit" = scores of backprojected logits. Adjudicated against the actual
#: implementations in code/src/csfs/ with one-line justifications in
#: x6_spectral/FREEZE.md (gate 2; fDBD moved kept -> logit there). Hybrids
#: are assigned to their dominant operator (NNGuide kept-dominant, ViM
#: complement-dominant). Pilot-only families (MahalanobisPP, NCI) are out of
#: scope until they appear in an outcome table.
FAMILY_OPERATOR: dict[str, str] = {
    "Maha": "kept", "CTM": "kept", "CTMmean": "kept", "NNGuide": "kept",
    "NeCo": "kept",
    "PCA RecError": "complement", "KPCA RecError": "complement",
    "Residual": "complement", "ViM": "complement",
    "MLS": "logit", "Energy": "logit", "MSR": "logit", "GEN": "logit",
    "GE": "logit", "PE": "logit", "PCE": "logit", "REN": "logit",
    "pNML": "logit", "GradNorm": "logit", "Confidence": "logit",
    "fDBD": "logit",
}
VARIANTS = ("global", "class pred", "class avg")


def auc_exact(q: float, lam: float) -> float:
    """Exact AUROC of ncx2_q(lam) against chi2_q (see pass3_common)."""
    if lam <= 0:
        return 0.5
    return float(1 - ncf.cdf(1.0, q, q, lam))


def a_star(lam: float, q: int, dim: int) -> float:
    """Kept-space crossing capture; small-lam limit is kappa_D/kappa_q."""
    target = auc_exact(dim, lam)
    return float(brentq(lambda a: auc_exact(q, lam * a) - target, 1e-9, 1.0))


def a_flip(lam: float, q: int, dim: int) -> float:
    """Capture where kept-space and complement-space scores exchange rank."""
    return float(brentq(
        lambda a: auc_exact(q, lam * a) - auc_exact(dim - q, lam * (1 - a)),
        1e-9, 1 - 1e-9))


def measure(h: np.ndarray, y: np.ndarray, w: np.ndarray, n_classes: int,
            k_class: int = 12, standardize: bool = False,
            het_splits: int = 3) -> dict:
    """ID-side diagnostics for one (checkpoint, source) cell."""
    n, dim = h.shape
    census = spike_census(h, standardize=standardize)
    via = viability(h, y, n_classes, standardize=standardize)
    k_stab = max(census["n_spikes"], n_classes - 1)
    stab_mean, stab_sd = split_half_stability(h, k_stab)
    het = class_projector_heterogeneity(h, y, n_classes, k_class,
                                        n_splits=het_splits)
    return {"census": census, "viability": via,
            "stability_k": k_stab, "stability": stab_mean,
            "stability_sd": stab_sd, "stability_null": k_stab / dim,
            "class_heterogeneity": het,
            "common_mode": common_mode_fraction(h),
            "w_gap": weight_top_gap(w),
            "w_top_align": weight_top_class_alignment(w),
            "n": n, "dim": dim}


def adjusted_stability(stability: float, null: float) -> float:
    """Above-chance projector agreement, (stability - null) / (1 - null).

    The raw ratio stability/null is unusable when the census saturates on
    real anisotropic spectra (hundreds of above-edge components make the
    random-subspace null k/D approach 1, so no fixed multiple of the null
    is reachable); the dev-pool stage-1 run showed exactly this. The
    normalized form is chance-corrected at any k.
    """
    return float((stability - null) / max(1.0 - null, 1e-9))


def tier_a(diag: dict, id_val_accuracy: float | None = None) -> dict:
    """ID-only statements: recovery, stability, and one-sided predictions.

    Returns per variant a dict with recovery evidence and `prediction` in
    {"no-benefit", "undetermined"}. "no-benefit" is the falsifiable claim
    (projection should not significantly help when the target subspace is
    not recoverable/stable); benefit signs are never emitted here. The
    stability gate uses adjusted (above-chance) agreement > 0.5, calibrated
    on the dev-pool stage-1 run (rule version r2, 2026-08-08; the earlier
    2x-null gate was unsatisfiable in the saturated-census regime). The
    routing screen for class-predicted variants is a conservative heuristic
    pending real-data calibration: pass4b shows naturally structured routing
    errors are far cheaper than uniform corruption, so low accuracy warns
    rather than vetoes.
    """
    via = diag["viability"]
    adj_stab = adjusted_stability(diag["stability"], diag["stability_null"])
    stab_ok = adj_stab > 0.5
    het = diag["class_heterogeneity"]
    het_ok = het["heterogeneity"] > 2 * het.get("within_sd", 0.0)
    out = {}
    global_rec = bool(via["global_viable"] and stab_ok)
    out["global"] = {
        "recoverable": global_rec,
        "stability": diag["stability"],
        "adjusted_stability": adj_stab,
        "prediction": "undetermined" if global_rec else "no-benefit",
    }
    class_rec = bool(via["class_viable"] and het_ok)
    note = None
    if id_val_accuracy is not None and id_val_accuracy < 0.5:
        note = "routing accuracy < 0.5: treat class-pred with caution"
    out["class pred"] = {
        "recoverable": class_rec,
        "heterogeneity": het["heterogeneity"],
        "prediction": "undetermined" if class_rec else "no-benefit",
        "routing_note": note,
    }
    out["class avg"] = {
        "recoverable": class_rec,
        "prediction": "no-benefit",
        "basis": "averaging back-projections mixes class affine frames; "
                 "theory-side expectation, strongly negative on dev data",
    }
    return out


def estimate_orientation(h_id: np.ndarray, ood_batch: np.ndarray,
                         projector: np.ndarray) -> dict:
    """Deployment-batch orientation estimate (explicitly OOD-side).

    Uses the mean-displacement model: delta_hat = mean(batch) - mean(ID).
    a_hat is its kept-energy fraction under `projector`; lam_hat is a
    method-of-moments noncentrality. Valid for mean-shift-like OOD; for
    covariance-only shifts the displacement vanishes and orientation must be
    assessed from per-sample energy splits instead (reported as fallback).
    """
    delta = ood_batch.mean(0) - h_id.mean(0)
    d2 = float(delta @ delta)
    a_hat = float(np.linalg.norm(projector.T @ delta) ** 2 / (d2 + 1e-12))
    sigma2 = float(np.median(np.var(h_id - h_id.mean(0), axis=0)))
    lam_hat = d2 / (sigma2 + 1e-12)
    kept_id = (h_id - h_id.mean(0)) @ projector
    frac_id = float((kept_id ** 2).sum() / ((h_id - h_id.mean(0)) ** 2).sum())
    kept_ood = ((ood_batch - h_id.mean(0)) @ projector)
    frac_ood = float((kept_ood ** 2).sum()
                     / ((ood_batch - h_id.mean(0)) ** 2).sum())
    return {"a_hat": a_hat, "lam_hat": lam_hat, "delta_norm2": d2,
            "kept_energy_id": frac_id, "kept_energy_ood": frac_ood}


def tier_b(diag: dict, orientation: dict, w: np.ndarray, q: int,
           delta: np.ndarray | None = None) -> dict:
    """Deployment-batch sign predictions, one rule per operator class."""
    dim = diag["dim"]
    lam, a_hat = orientation["lam_hat"], orientation["a_hat"]
    thr_kept = a_star(lam, q, dim)
    thr_flip = a_flip(lam, q, dim)
    signs = {"kept": +1 if a_hat > thr_kept else -1,
             "complement": +1 if a_hat < thr_flip else -1}
    if delta is not None:
        w_row = np.linalg.qr(w.T, mode="reduced")[0]
        row_align = float(np.linalg.norm(w_row.T @ delta) ** 2
                          / (delta @ delta + 1e-12))
        signs["logit"] = +1 if row_align > 0.5 else -1
        signs["logit_row_align"] = row_align
    signs.update({"a_hat": a_hat, "a_star": thr_kept, "a_flip": thr_flip,
                  "mode": "deployment-batch adaptation (not ID-only)"})
    return signs


def predictions_for_families(tier_a_out: dict, tier_b_out: dict | None
                             ) -> dict:
    """Join tiers into per-(family, variant) records for the score join.

    Tier A supplies {"no-benefit", "undetermined"}; when a Tier B record is
    available, its operator-class sign replaces "undetermined" and is
    labeled with its mode. Families absent from FAMILY_OPERATOR are skipped.
    """
    out = {}
    for family, operator in FAMILY_OPERATOR.items():
        for variant in VARIANTS:
            rec = dict(tier_a_out[variant])
            if (tier_b_out is not None
                    and rec["prediction"] == "undetermined"
                    and operator in tier_b_out):
                rec = {**rec, "prediction": int(tier_b_out[operator]),
                       "mode": tier_b_out["mode"]}
            out[(family, variant)] = rec
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n_cls, dim, k_res = 10, 256, 8
    base = np.linalg.qr(rng.standard_normal((dim, n_cls + k_res + 1)))[0]
    simplex = (np.eye(n_cls) - np.ones((n_cls, n_cls)) / n_cls)
    mu = (simplex * np.sqrt(n_cls / (n_cls - 1)) * 3.0) @ base[:, :n_cls].T
    b_res = base[:, n_cls:n_cls + k_res]
    perp = base[:, -1]

    def draw(n: int, theta: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, n_cls, n)
        h = mu[y] + rng.standard_normal((n, dim))
        resid = rng.standard_normal((n, k_res)) * np.sqrt(3.0)
        if theta == 0.0:
            h += resid @ b_res.T
        else:
            keep = np.eye(dim) - base @ base.T
            for c in range(n_cls):
                g_c = np.linalg.qr(
                    keep @ rng.standard_normal((dim, k_res)))[0][:, :k_res]
                u_c = np.sqrt(1 - theta**2) * b_res + theta * g_c
                mask = y == c
                h[mask] += resid[mask] @ u_c.T
        return h, y

    print("[self-test 1] recoverable world (N=4000), shared vs "
          "class-dependent residue subspaces")
    for theta, label in ((0.0, "shared B"), (0.7, "theta=0.7")):
        h, y = draw(4000, theta)
        diag = measure(h, y, mu, n_cls, k_class=k_res)
        het = diag["class_heterogeneity"]
        ta = tier_a(diag)
        print(f"    {label}: heterogeneity={het['heterogeneity']:+.3f} "
              f"(cross={het['cross']:.3f} within={het['within']:.3f})  "
              f"global={ta['global']['prediction']}  "
              f"class_pred={ta['class pred']['prediction']}")

    print("[self-test 2] unrecoverable world (N=280, y=0.91): Tier A must "
          "emit the one-sided no-benefit claim")
    h_small, y_small = draw(280)
    diag_small = measure(h_small, y_small, mu, n_cls, k_class=k_res)
    ta_small = tier_a(diag_small)
    print(f"    global_viable={diag_small['viability']['global_viable']} "
          f"stability={diag_small['stability']:.2f} "
          f"(null {diag_small['stability_null']:.2f}) -> "
          f"global={ta_small['global']['prediction']}")

    print("[self-test 3] Tier B flips sign with batch orientation")
    h, y = draw(4000)
    diag = measure(h, y, mu, n_cls, k_class=k_res)
    centered = h - h.mean(0)
    q = n_cls - 1 + k_res
    proj = np.linalg.eigh(centered.T @ centered / len(h))[1][:, -q:]
    for label, direction in (("in-span", mu[0] / np.linalg.norm(mu[0])),
                             ("out-of-span", perp)):
        batch = h[:200] + 12.0 * direction
        ori = estimate_orientation(h, batch, proj)
        tb = tier_b(diag, ori, mu, q, delta=direction)
        print(f"    {label}: a_hat={ori['a_hat']:.2f} "
              f"(a*={tb['a_star']:.2f}, flip={tb['a_flip']:.2f}) -> "
              f"kept={tb['kept']:+d} complement={tb['complement']:+d} "
              f"logit={tb['logit']:+d}")

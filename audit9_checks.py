"""Audit-9 checks (submission-close plan 2026-08-28, sections 8, 11, 15):
R5 unrounded + high-precision margin verification, and the optional
within-reward BREEDS sensitivities (blocked permutation and residual rank
association). Post-hoc sensitivities; frozen primaries unchanged.

R5 (section 8): (a) UNROUNDED maximum absolute and relative BREEDS margin
changes under float32 input rounding and independent relative 1e-6 input
perturbations (seed 931, same construction as audit-8 R5); (b) a
deterministic high-precision reference: the theory module's normal CDF is
replaced by mpmath.ncdf at 50 significant digits and the analytic AUROC
pair is recomputed on the SAME float64 deviates for the smallest-margin
BREEDS cell and the median-margin ImageNet-200 cell. Scope stated
narrowly: this isolates CDF-evaluation and accumulation error in the final
AUROC arithmetic; it is not an extraction-uncertainty interval.

Within-reward sensitivity (section 11, optional): across the 28 BREEDS
cells, (Option A) permute |analytic margin| only WITHIN paradigm x reward
strata (B = 10000, seed 987) and compare the observed global Spearman
against that null; (Option B) rank-residualize both |margin| and |gap|
against categorical paradigm x reward indicators and report the residual
Spearman. Strata hold 4 checkpoints each, so low power is expected and is
reported, not hidden.

Usage (from code/): python audit9_checks.py
Output: nc_csf_predictivity/outputs/track1/audit9_checks_report.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audit8_checks import expansion_cells
from crossing_robustness_audit import OUT_DIR

B_PERM = 10000


def rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def r5_unrounded(base) -> dict:
    out = {}
    for tag in ("float32", "rel1e-6"):
        alt = expansion_cells(perturb=tag)
        m = base.merge(alt, on=["slug", "set"], suffixes=("", "_alt"))
        br = m[m.source == "breeds"]
        d_abs = (br.margin - br.margin_alt).abs()
        d_rel = d_abs / np.clip(br.margin, 1e-300, None)
        out[tag] = {
            "breeds_sign_flips": int((br.pred_sign
                                      != br.pred_sign_alt).sum()),
            "breeds_rank_spearman": rho(br.margin.values,
                                        br.margin_alt.values),
            "max_abs_margin_change": float(d_abs.max()),
            "max_rel_margin_change": float(d_rel.max()),
        }
    return out


def r5_highprecision(base) -> dict:
    """Recompute the analytic pair with mpmath.ncdf at 50 digits for the
    smallest-margin BREEDS cell and the median-margin ImageNet-200 cell."""
    import mpmath

    import pilot0.theory as theory
    from stage2_expansion_analysis import theory_pair

    br = base[base.source == "breeds"].sort_values("margin")
    inn = base[base.source == "imagenet200"].sort_values("margin")
    targets = {"breeds_smallest": br.iloc[0],
               "imagenet200_median": inn.iloc[len(inn) // 2]}

    class MPNorm:
        @staticmethod
        def cdf(x):
            return float(mpmath.ncdf(mpmath.mpf(float(x))))

    def pair_for(rowlike, records):
        rec = records[rowlike.slug]
        c, d = int(rec["n_classes"]), int(rec["dim"])
        vc = rec["papyan"]["var_collapse"]
        sd = rec["papyan"]["self_duality"]
        s_dict = (c - 1) / np.sqrt(c * max(vc, 1e-9))
        theta = float(np.degrees(np.arccos(np.clip(1 - sd / 2, -1, 1))))
        e = rec["ood"][rowlike.set]
        return theory_pair(c, d, s_dict, theta,
                           rec["geometry"]["logit_scale"],
                           rec["geometry"]["class_mean_radius_cv"],
                           e["gamma"], e["a"], e["rho"], {})

    records = {}
    for d_ in ("pilot0/stage2_expansion_coords",
               "pilot0/stage3_imagenet200_coords"):
        for p in Path(d_).glob("*.json"):
            if not p.name.startswith("FAILED"):
                r = json.loads(p.read_text())
                records[r.get("slug") or r.get("run")] = r

    mpmath.mp.dps = 50
    out = {}
    orig_norm = theory.norm
    for name, rowlike in targets.items():
        m64_e, m64_c = pair_for(rowlike, records)
        theory.norm = MPNorm
        try:
            mhp_e, mhp_c = pair_for(rowlike, records)
        finally:
            theory.norm = orig_norm
        margin64 = abs(m64_e - m64_c)
        marginhp = abs(mhp_e - mhp_c)
        out[name] = {
            "cell": f"{rowlike.slug} / {rowlike.set}",
            "margin_float64": margin64, "margin_mpmath50": marginhp,
            "abs_difference": abs(margin64 - marginhp),
            "sign_agrees": bool(np.sign(m64_e - m64_c)
                                == np.sign(mhp_e - mhp_c)),
        }
    return out


def within_reward(br) -> dict:
    strata = (br.paradigm + "|" + br.reward.astype(str)).values
    x = br.margin.values
    y = np.abs(br.obs.values)
    obs = rho(x, y)
    rng = np.random.default_rng(987)
    perms = np.empty(B_PERM)
    idx_by = {s: np.where(strata == s)[0] for s in set(strata)}
    for i in range(B_PERM):
        xp = x.copy()
        for s, idx in idx_by.items():
            xp[idx] = x[rng.permutation(idx)]
        perms[i] = rho(xp, y)
    p_blocked = float((1 + (perms >= obs).sum()) / (B_PERM + 1))

    # Option B: rank-residualize both variables on stratum indicators.
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    dummies = np.column_stack([(strata == s).astype(float)
                               for s in sorted(set(strata))])
    def resid(v):
        beta, *_ = np.linalg.lstsq(dummies, v, rcond=None)
        return v - dummies @ beta
    r_resid = rho(resid(rx), resid(ry))
    return {"observed_spearman": round(obs, 3),
            "blocked_permutation_p_one_sided": round(p_blocked, 5),
            "blocked_null_q95": round(float(np.quantile(perms, 0.95)), 3),
            "residual_rank_spearman": round(r_resid, 3),
            "n_strata": len(idx_by),
            "note": "strata hold 4 checkpoints each; low power expected"}


def main() -> None:
    base = expansion_cells()
    br = base[base.source == "breeds"].reset_index(drop=True)
    out = {"R5_unrounded": r5_unrounded(base),
           "R5_highprecision": r5_highprecision(base),
           "within_reward": within_reward(br)}
    L = ["# Audit-9 checks (R5 unrounded + high precision; within-reward "
         "sensitivity)", "",
         "Post-hoc sensitivities; frozen primaries unchanged. The "
         "high-precision check isolates CDF-evaluation and accumulation "
         "error on the same float64 deviates; it is not an "
         "extraction-uncertainty interval.", "",
         "## R5 unrounded", "```",
         json.dumps(out["R5_unrounded"], indent=1), "```", "",
         "## R5 high-precision (mpmath, 50 digits)", "```",
         json.dumps(out["R5_highprecision"], indent=1), "```", "",
         "## Within-reward BREEDS sensitivity (optional, audit-9 "
         "section 11)", "```",
         json.dumps(out["within_reward"], indent=1), "```", ""]
    (OUT_DIR / "audit9_checks_report.md").write_text("\n".join(L))
    (OUT_DIR / "audit9_checks_report.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()

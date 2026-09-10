"""RN18 handoff-replication PHASE 4, VERSION 2: corrected outcome-free
qualification of the version-2 reader (repair item P1-B; status-review
finding F2, with F11's reproducibility items). The version-1
qualification (`phase4_qualification.py`, `design_manifest.json`,
`qualification_report.json`) is preserved unchanged; this is a labeled
correction with its own manifest, seeds and license file.

What version 1 exercised and this version changes:

1. ONE COHERENT FORECAST MODEL PER CELL. Observed AUROCs (Energy, CTM)
   come from the family/cell gap model; the P00 forecasts are the
   observed AUROCs plus forecast errors with a FAMILY-level and a
   cell-level component drawn from the scenario's distribution; the
   frozen-arm margin M = log(1 - predE) - log(1 - predC) is DERIVED from
   those forecasts (the reader's own formula), and the LEVEL errors are
   the absolute forecast errors of the same forecasts. P10 forecasts
   shrink the P00 errors by a family-level factor (heterogeneous by
   dropout cluster when the scenario says so). Version 1 drew M and the
   LEVEL errors independently of each other and of the family effects.
2. LITERAL CONSTANT COMPARATORS (always Energy = -1, always CTM = +1).
   The seven fitted comparators are exchangeable P00 clones (fresh
   forecast errors, same law: true D_b = 0 exactly by exchangeability)
   unless a scenario calibrates a flip probability so that the true D_b
   sits at a declared boundary (+eps, +1.5 eps, +0.005, -0.005 for the
   rotating best). Calibration is done on the persisted reference
   sample, and the reference truth is stored with its Monte Carlo
   standard error; nothing is snapped to a nominal value.
3. FAMILY-LEVEL dependence (correlation, factor model), skew (lognormal),
   heavy tails (t3), contamination, a single outlying family, and a
   two-cluster dropout effect; plus the MANDATORY bounded skewed family
   null (delta_f = 0.2 (B_f - 0.3/3.3), B_f ~ Beta(0.3, 3), exact mean 0,
   valid MAE differences at AUROC 0.5).
4. The pi = 1 source keeps its cells (SEL/LEVEL are AUROC endpoints; only
   the AUGRC gap vanishes there), the missing-cell and duplicate-family
   fixtures produce the reader's DECLARED validator failure (no
   inference), non-finite inputs produce NOT ESTIMABLE, and the
   policy-degeneracy regime (P00 = always CTM) is a named scenario.
5. HO fixtures are PER-EXAMPLE: scores are simulated per example, the
   AUGRC gap goes through the frozen outcome arithmetic
   (`failure_augrc`), and the unchanged gate (`ho_source`) is applied;
   verdict rates at b = 0 (2,000 replications) and band coverage of the
   true stratum curve at the declared b = 2000 (50 replications).
6. Decisions at full precision (reader v2 primitives; rounding is for
   display only); the vectorized fast path is asserted equal to the
   reader v2 numerically (estimates, standard errors, interval
   endpoints to 1e-10, verdict strings, and failure states) on the first
   50 replications of every scenario.
7. Stable scenario digests (sha256 of the name), every DGP parameter,
   calibration and reference truth persisted; seeds 2403 (development)
   and 2404 (audit) with separate scenario stream keys.

Usage (from code/):
    python rn18_handoff_replication/phase4_qualification_v2.py --self-test
    python rn18_handoff_replication/phase4_qualification_v2.py --write-manifest
    python rn18_handoff_replication/phase4_qualification_v2.py --stage dev   [--reps 10000]
    python rn18_handoff_replication/phase4_qualification_v2.py --stage audit [--reps 10000]
Outputs: simulations/design_manifest_v2.json, reference_truths_v2_<stage>.json,
         development_results_v2.json, development_critical_values_v2.json,
         audit_results_v2.json, qualification_report_v2.json, sample_size_decision_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm, t as student

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from crossing_robustness_audit import curve as pava_curve
from pilot0.extract_stage2_expansion import failure_augrc
from rn18_handoff_replication import rn18_analysis_v2 as R
from rn18_handoff_replication.comparators import NAMES, REFERENCE
from rn18_handoff_replication.rn18_analysis import FINE_N, ho_source

SIM = Path("rn18_handoff_replication/simulations")
SEED_DEV, SEED_AUDIT = 2403, 2404
MULTS = (1.0, 1.1, 1.25, 1.5, 2.0)
N_FS = (10, 5)
SOURCES = ("cifar10", "cifar100", "supercifar100", "tinyimagenet")
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
ALPHA_EACH = R.ALPHA_SEL / len(NAMES)
REF_REPS = 12_500                       # x 160 cells = 2,000,000 reference cells at N_f = 10
CHECK_READER = 50
NUM_TOL = 1e-10
FITTED = tuple(n for n in NAMES if n not in ("always_energy", "always_ctm"))
BASE = dict(dist="gauss", corr=0.5, sd_fam=0.02, sd_cell=0.03, base_auroc=0.75, err_scale=0.10, err_fam_frac=0.5,
            p00_bias=0.0, p10_noise=0.005, level_target=0.0, dropout_ratio=None, comp_mode="exchangeable",
            family_outlier=False, beta_family_null=False, ties_frac=0.0, ceiling=False, one_sign=False,
            pi_varies=False, pi_one_source=False, shared_id=False, shared_ood=False, missing_cell=False,
            nonfinite=False, undefined_p10=False, duplicate_family=False, ho=None, geo="reliable")
DECLARED_SEL = {"identical": "the seven fitted comparators NOT ESTIMABLE (zero jackknife SE); constants estimable",
                "degenerate_ctm": "always_ctm NOT ESTIMABLE (identical choices); the rest regular"}


def skey(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Scenario manifest v2: the 43 frozen scenarios + 4 corrections.
# ---------------------------------------------------------------------------

def scenarios() -> list[dict]:
    S = []

    def add(name, axis, held=False, **kw):
        S.append({"name": name, "axis": axis, "held_out": held, "params": {**BASE, **kw}})
    add("S0_base", "base")
    add("dist_t3", "distribution", dist="t3")
    add("dist_contaminated", "distribution", dist="contaminated")
    add("dist_lognormal_skew", "distribution", held=True, dist="lognormal")
    add("dep_corr0", "dependence", corr=0.0)
    add("dep_corr0.9", "dependence", corr=0.9)
    add("dep_factor_model", "dependence", held=True, corr="factor")
    add("geo_near_zero_spread", "geometry", ho="planted", geo="near_zero")
    add("geo_clustered_ties", "geometry", ho="planted", geo="clustered")
    add("geo_one_extreme", "geometry", ho="planted", geo="extreme")
    add("geo_shared_measurement_error", "geometry", ho="planted", geo="shared_error")
    add("null_geometry_dependent_variance", "null", ho="planted", geo="var_dep")
    add("null_sel_boundary_superior", "null", comp_mode="boundary_superior")
    add("null_level_boundary_equivalence", "null", level_target=0.01)
    add("null_level_equivalence_false", "null", level_target=0.015)
    add("null_sel_equivalence_false", "null", comp_mode="reference_off_margin")
    add("pred_p10_improvement_0.02", "predictors", level_target=0.02)
    add("pred_identical", "predictors", comp_mode="identical")
    add("pred_practically_equivalent", "predictors", comp_mode="equivalent")
    add("pred_each_best_in_turn", "predictors", comp_mode="rotating_best")
    add("pred_uniform_p00_advantage", "predictors", comp_mode="p00_advantage")
    add("pred_p10_improvement", "predictors", level_target=0.01)
    add("pred_p10_worsening", "predictors", level_target=-0.01)
    add("metric_near_ceiling", "metric", ceiling=True)
    add("metric_almost_all_ties", "metric", ties_frac=0.9)
    add("metric_class_missing_material", "metric", one_sign=True)
    add("metric_id_error_mixtures", "metric", pi_varies=True)
    add("metric_pi_one_source", "metric", pi_one_source=True)
    add("ex_shared_id_across_shifts", "examples", shared_id=True)
    add("ex_shared_ood_across_sources", "examples", shared_ood=True)
    add("impl_missing_cell", "implementation", missing_cell=True)
    add("impl_nonfinite_score", "implementation", nonfinite=True)
    add("impl_undefined_analytic_branch", "implementation", undefined_p10=True)
    add("impl_duplicate_family", "implementation", duplicate_family=True)
    add("ho_all_left_censored", "ho", ho="all_left")
    add("ho_all_right_censored", "ho", ho="all_right")
    add("ho_reversed", "ho", ho="reversed")
    add("ho_nonlinear_middle_crossing", "ho", ho="nonlinear")
    add("int_t3_corr0.9", "interaction", dist="t3", corr=0.9)
    add("int_ceiling_ties", "interaction", ceiling=True, ties_frac=0.9)
    add("int_identical_corr0.9", "interaction", comp_mode="identical", corr=0.9)
    add("int_p10_improvement_pi_one", "interaction", level_target=0.01, pi_one_source=True)
    add("int_ushape_clustered", "interaction", held=True, ho="ushape", geo="clustered")
    # corrections (2026-09-09, status-review F2 and analysis section 2.1)
    add("null_level_bounded_beta_family", "correction", beta_family_null=True)
    add("dep_dropout_two_clusters", "correction", level_target=0.011, dropout_ratio=2.1)
    add("pred_p00_degenerate_always_ctm", "correction", comp_mode="degenerate_ctm", p00_bias=0.3)
    add("dist_family_outlier", "correction", family_outlier=True)
    assert len(S) == 47 and sum(s["held_out"] for s in S) == 3
    return S


# ---------------------------------------------------------------------------
# Coherent cell generator.
# ---------------------------------------------------------------------------

def _resid(rng, shape, dist):
    if dist == "gauss":
        return rng.standard_normal(shape)
    if dist == "t3":
        return rng.standard_t(3, shape) / np.sqrt(3.0)
    if dist == "lognormal":
        z = np.exp(0.8 * rng.standard_normal(shape))
        return (z - np.exp(0.32)) / np.sqrt((np.exp(0.64) - 1) * np.exp(0.64))
    if dist == "contaminated":
        z = rng.standard_normal(shape)
        m = rng.random(shape) < 0.05
        return np.where(m, 5 * z, z) / np.sqrt(0.95 + 0.05 * 25)
    raise ValueError(dist)


def _family_effects(rng, N_f, P, ncol=4):
    """N_f x ncol family effects with the scenario's dependence and distribution (unit scale)."""
    if P["corr"] == "factor":
        lam = np.array([0.3, 0.6, 0.9, 0.5])[:ncol]
        u = lam[None, :] * _resid(rng, (N_f, 1), P["dist"]) + 0.5 * _resid(rng, (N_f, ncol), P["dist"])
    else:
        r = float(P["corr"])
        u = np.sqrt(r) * _resid(rng, (N_f, 1), P["dist"]) + np.sqrt(1 - r) * _resid(rng, (N_f, ncol), P["dist"])
    if P["family_outlier"]:
        u[0] *= 5.0
    return u


def _inside(auroc, err):
    """Forecast = AUROC + err kept strictly inside (0, 1): the error sign is
    mirrored when the forecast would leave (0.001, 0.999)."""
    hi = auroc + np.abs(err) > 0.999
    lo = auroc - np.abs(err) < 0.001
    e = np.where(hi, -np.abs(err), np.where(lo, np.abs(err), err))
    return auroc + e, e


def _shrink(P, calib, fam_do):
    s0 = calib.get("shrink", 0.0)
    if P["dropout_ratio"] is not None:
        return np.where(fam_do == 1, P["dropout_ratio"] * s0, s0)
    return np.full(len(fam_do), s0)


def sim_cells(P: dict, N_f: int, rng, mu_so: np.ndarray, calib: dict | None = None, best_k: int = 0) -> dict:
    """N_f families x 4 sources x 4 shifts, one coherent model per cell."""
    calib = calib or {}
    fam_do = np.array([0] * 5 + [1] * 5)[:N_f] if N_f == 10 else np.zeros(5, int)
    n = N_f * 16
    f_idx = np.repeat(np.arange(N_f), 16)
    s_idx = np.tile(np.repeat(np.arange(4), 4), N_f)
    o_idx = np.tile(np.arange(4), N_f * 4)
    # observed gap model
    u = P["sd_fam"] * _family_effects(rng, N_f, P)
    e = P["sd_cell"] * _resid(rng, n, P["dist"])
    if P["shared_id"]:
        e = np.sqrt(0.5) * e + np.sqrt(0.5) * P["sd_cell"] * rng.standard_normal((N_f, 4))[f_idx, s_idx]
    if P["shared_ood"]:
        e = np.sqrt(0.5) * e + np.sqrt(0.5) * P["sd_cell"] * rng.standard_normal((N_f, 4))[f_idx, o_idx]
    dA = mu_so[s_idx, o_idx] + u[f_idx, s_idx] + e
    if P["ceiling"]:
        dA = dA * 0.05
    if P["one_sign"]:
        dA = np.abs(dA)
    base = 0.5 if P["beta_family_null"] else (0.98 if P["ceiling"] else P["base_auroc"])
    aurocE = np.clip(base + 0.02 * rng.standard_normal(n), 0.01, 0.99)
    aurocC = np.clip(aurocE + dA, 0.01, 0.99)
    dA = aurocC - aurocE
    pi = np.full(n, 0.5)
    if P["pi_varies"]:
        pi = np.array([0.3, 0.5, 0.6, 0.8])[s_idx]
    if P["pi_one_source"]:
        pi = np.where(s_idx == 3, 1.0, pi)
    dG = pi * (1 - pi) * (0.6 * dA + 0.002 * rng.standard_normal(n))       # AUGRC identity: vanishes at pi = 1
    # coherent forecasts: P00 errors with family + cell components; P10 = family-shrunk P00 errors
    tau, fr = P["err_scale"], P["err_fam_frac"]
    shrink = _shrink(P, calib, fam_do)[f_idx]

    def forecast_pair(rng_, degenerate=False):
        vE, vC = _family_effects(rng_, N_f, P, 1)[:, 0], _family_effects(rng_, N_f, P, 1)[:, 0]
        wE, wC = _resid(rng_, n, P["dist"]), _resid(rng_, n, P["dist"])
        errE = tau * (np.sqrt(fr) * vE[f_idx] + np.sqrt(1 - fr) * wE)
        errC = tau * (np.sqrt(fr) * vC[f_idx] + np.sqrt(1 - fr) * wC)
        if degenerate:
            # policy-degeneracy regime: Energy always under-forecast, CTM always over-forecast (+ bias) -> P00 = always CTM
            eE00 = -np.abs(errE); eC00 = np.abs(errC) + P["p00_bias"]
            predE00 = np.clip(aurocE + eE00, 0.001, 0.999); predC00 = np.clip(aurocC + eC00, 0.001, 0.999)
            eE00, eC00 = predE00 - aurocE, predC00 - aurocC
        else:
            predE00, eE00 = _inside(aurocE, errE + P["p00_bias"])
            predC00, eC00 = _inside(aurocC, errC)
        predE10, eE10 = _inside(aurocE, eE00 * (1 - shrink) + P["p10_noise"] * rng_.standard_normal(n))
        predC10, eC10 = _inside(aurocC, eC00 * (1 - shrink) + P["p10_noise"] * rng_.standard_normal(n))
        return predE00, predC00, predE10, predC10

    if P["beta_family_null"]:
        B = rng.beta(0.3, 3.0, N_f)
        delta_f = 0.2 * (B - 0.3 / 3.3)
        zeta = rng.uniform(-1e-3, 1e-3, n)
        predE00 = aurocE + 0.2; predC00 = aurocC + 0.2 + zeta
        predE10 = aurocE + 0.2 - delta_f[f_idx]; predC10 = aurocC + 0.2 - delta_f[f_idx] + zeta
        assert (predC10 > 0).all() and (predC00 < 1).all() and (predE10 > 0).all()

        def forecast_pair(rng_, degenerate=False):                      # exchangeable clone: fresh zeta only
            z2 = rng_.uniform(-1e-3, 1e-3, n)
            return aurocE + 0.2, aurocC + 0.2 + z2, None, None
    else:
        predE00, predC00, predE10, predC10 = forecast_pair(rng, P["comp_mode"] == "degenerate_ctm")
    lE00, lC00 = np.log(1 - predE00), np.log(1 - predC00)
    lE10, lC10 = np.log(1 - predE10), np.log(1 - predC10)
    M = lE00 - lC00                                                      # the reader's frozen-arm margin
    if P["ties_frac"] > 0:
        M = np.where(rng.random(n) < P["ties_frac"], 0.0, M)
    # comparators: literal constants; fitted ones = exchangeable clones, optionally calibrated
    preds = {"always_energy": np.full(n, -1.0), "always_ctm": np.full(n, 1.0)}
    U = rng.random((len(FITTED), n))
    for k, name in enumerate(FITTED):
        mode = P["comp_mode"]
        if mode == "identical":
            preds[name] = M.copy(); continue
        pE, pC, _, _ = forecast_pair(rng)
        Mk = np.log(1 - pE) - np.log(1 - pC)
        q = 0.0
        if mode == "boundary_superior" or mode == "p00_advantage" or mode == "equivalent":
            q = calib.get("flip_q", 0.0)
        elif mode == "reference_off_margin" and name == REFERENCE:
            q = calib.get("flip_q", 0.0)
        elif mode == "rotating_best" and k == best_k:
            Mk = np.where(U[k] < calib.get("fix_q", 0.0), np.sign(dA) * np.maximum(np.abs(Mk), 1e-6), Mk)
        if q > 0:
            Mk = np.where(U[k] < q, -Mk, Mk)
        preds[name] = Mk
    keep = np.ones(n, bool)
    if P["missing_cell"]:
        keep[0] = False
    if P["nonfinite"]:
        aurocE = aurocE.copy(); aurocE[1] = np.nan; dA = aurocC - aurocE
    if P["undefined_p10"]:
        lC10 = lC10.copy(); lC10[rng.random(n) < 0.1] = np.nan
    fam = np.array([f"do{fam_do[f]}_run{(f % 5) + 1}" for f in range(N_f)])
    if P["duplicate_family"]:
        fam = fam.copy(); fam[1] = fam[0]
    return dict(n=n, N_f=N_f, f_idx=f_idx, s_idx=s_idx, o_idx=o_idx, family=fam[f_idx], dA=dA, dG=dG, M=M, preds=preds,
                aurocE=aurocE, aurocC=aurocC, lE00=lE00, lC00=lC00, lE10=lE10, lC10=lC10, keep=keep, pi=pi, U=U)


def to_frame(c: dict) -> pd.DataFrame:
    df = pd.DataFrame({"cell": [f"ck{f}_{s}" for f, s in zip(c["f_idx"], c["s_idx"])], "component": "standalone_ce",
                       "family": c["family"], "source": np.array(SOURCES)[c["s_idx"]], "ood_set": np.array(SHIFTS)[c["o_idx"]],
                       "dA": c["dA"], "dG": c["dG"], "M": c["M"], "aurocE": c["aurocE"], "aurocC": c["aurocC"],
                       "l_E": c["lE00"], "l_C": c["lC00"], "l_E_p10": c["lE10"], "l_C_p10": c["lC10"]})
    for k, v in c["preds"].items():
        df[f"pred_{k}"] = v
    return df[c["keep"]].reset_index(drop=True)


class FixedComparators:
    def predict(self, name, df):
        return df[f"pred_{name}"].to_numpy(float)


# ---------------------------------------------------------------------------
# Reader-equivalent fast paths (full precision) and the declared states.
# ---------------------------------------------------------------------------

def panel_state(c: dict) -> str | None:
    """The version-2 validator's declared failures, applied to a simulated panel."""
    if len(np.unique(c["family"])) != c["N_f"]:
        return "VALIDATION FAILED [families]"
    if not c["keep"].all():
        return "VALIDATION FAILED [key_set]"
    return None


def fast_sel(c: dict) -> dict:
    dA, M, fam = c["dA"], c["M"], c["family"]
    if not (np.isfinite(dA).all() and np.isfinite(M).all()):
        return {"state": "NOT ESTIMABLE (non-finite required input on the registered panel)"}
    fams, finv = np.unique(fam, return_inverse=True)
    N = len(fams)
    r00 = R.regret(dA, R.choice_prob(M))
    out = {}
    for name in NAMES:
        p = c["preds"][name]
        if not np.isfinite(p).all():
            out[name] = None; continue
        d = R.regret(dA, R.choice_prob(p)) - r00
        tot, cnt = d.sum(), len(d)
        fs = np.bincount(finv, weights=d, minlength=N); fc = np.bincount(finv, minlength=N)
        loo = (tot - fs) / (cnt - fc)
        se = float(np.sqrt((N - 1) / N * ((loo - loo.mean()) ** 2).sum()))
        out[name] = (float(d.mean()), se) if (np.isfinite(se) and se > 0) else None
    return {"state": None, "N": N, "D": out}


def fast_level(c: dict) -> dict:
    need = [c["lE00"], c["lC00"], c["lE10"], c["lC10"], c["aurocE"], c["aurocC"]]
    if not all(np.isfinite(a).all() for a in need):
        return {"state": "NOT ESTIMABLE (non-finite required input on the registered panel)"}
    pa = lambda l: 1 - np.exp(l)
    e00 = (np.abs(pa(c["lE00"]) - c["aurocE"]) + np.abs(pa(c["lC00"]) - c["aurocC"])) / 2
    e10 = (np.abs(pa(c["lE10"]) - c["aurocE"]) + np.abs(pa(c["lC10"]) - c["aurocC"])) / 2
    d = e00 - e10
    fams, finv = np.unique(c["family"], return_inverse=True)
    fd = pd.DataFrame({"f": finv, "s": c["s_idx"], "d": d}).groupby(["f", "s"]).d.mean().groupby("f").mean()
    N = len(fd); se = float(fd.std(ddof=1) / np.sqrt(N))
    if not np.isfinite(se) or se <= 0:
        return {"state": "NOT ESTIMABLE (zero SE)"}
    return {"state": None, "N": N, "delta": float(fd.mean()), "se": se, "family_deltas": fd.to_numpy()}


def sel_intervals(fs: dict, m: float) -> dict:
    ivs = {}
    for n, v in fs["D"].items():
        if v is None:
            ivs[n] = None
        else:
            est, se = v
            q = student.ppf(1 - ALPHA_EACH / 2, fs["N"] - 1) * m * se
            ivs[n] = [est - q, est + q]
    return ivs


def level_interval(fl: dict, m: float) -> list:
    q = student.ppf(1 - R.ALPHA_LEVEL / 2, fl["N"] - 1) * m * fl["se"]
    return [fl["delta"] - q, fl["delta"] + q]


def check_reader(c: dict, fs: dict, fl: dict, name: str) -> None:
    """Numerical agreement of the fast path with the version-2 reader (estimates,
    SEs, interval endpoints, verdicts, failure states)."""
    df = to_frame(c)
    if panel_state(c):
        return                                                            # the validator stops before the endpoints
    rs = R.sel_endpoint(df, FixedComparators(), 1.0, "sim")
    if fs["state"]:
        assert rs["verdict"].startswith("NOT ESTIMABLE"), (name, rs["verdict"])
    else:
        ivs = sel_intervals(fs, 1.0)
        assert rs["verdict"] == R.sel_verdict(ivs), (name, rs["verdict"], R.sel_verdict(ivs))
        for n in NAMES:
            rc = rs["comparators"][n]
            if fs["D"][n] is None:
                assert rc["ci"] is None, (name, n, rc)
            else:
                assert abs(rc["D_b"] - fs["D"][n][0]) <= NUM_TOL and abs(rc["se"] - fs["D"][n][1]) <= NUM_TOL, (name, n)
                assert max(abs(rc["ci"][0] - ivs[n][0]), abs(rc["ci"][1] - ivs[n][1])) <= NUM_TOL, (name, n)
    rl = R.level_endpoint(df, 1.0, "sim")
    if fl["state"]:
        assert rl["verdict"] == "NOT ESTIMABLE" or rl["verdict"].startswith("NOT ESTIMABLE"), (name, rl["verdict"])
    else:
        iv = level_interval(fl, 1.0)
        assert abs(rl["delta"] - fl["delta"]) <= NUM_TOL and abs(rl["se"] - fl["se"]) <= NUM_TOL, name
        assert max(abs(rl["ci"][0] - iv[0]), abs(rl["ci"][1] - iv[1])) <= NUM_TOL and rl["verdict"] == R.level_verdict(iv)["verdict"], name


# ---------------------------------------------------------------------------
# Reference sample, calibration and truths (persisted; no snapping).
# ---------------------------------------------------------------------------

def _ref_sample(P, N_f, seed, mu_so, calib, best_k=0, reps=REF_REPS):
    rng = np.random.default_rng([seed, 999, N_f, best_k])
    Ps = dict(P, missing_cell=False, nonfinite=False, undefined_p10=False, duplicate_family=False)
    rows = [sim_cells(Ps, N_f, rng, mu_so, calib, best_k) for _ in range(reps)]
    cat = lambda k: np.concatenate([r[k] for r in rows])
    out = {k: cat(k) for k in ("dA", "M", "lE00", "lC00", "lE10", "lC10", "aurocE", "aurocC", "f_idx", "s_idx")}
    out["preds"] = {n: cat_p for n, cat_p in ((n, np.concatenate([r["preds"][n] for r in rows])) for n in NAMES)}
    out["U"] = np.concatenate([r["U"] for r in rows], axis=1)
    out["rep"] = np.repeat(np.arange(reps), N_f * 16)
    return out


def _D_of(sample, name, q=None, k=None, fix=False):
    dA, M = sample["dA"], sample["M"]
    p = sample["preds"][name]
    if q is not None and k is not None:
        p = (np.where(sample["U"][k] < q, np.sign(dA) * np.maximum(np.abs(p), 1e-6), p) if fix
             else np.where(sample["U"][k] < q, -p, p))
    d = R.regret(dA, R.choice_prob(p)) - R.regret(dA, R.choice_prob(M))
    reps = sample["rep"]
    per_rep = np.bincount(reps, weights=d) / np.bincount(reps)
    return float(d.mean()), float(per_rep.std(ddof=1) / np.sqrt(len(per_rep)))


def _level_of(sample, N_f):
    pa = lambda l: 1 - np.exp(l)
    e00 = (np.abs(pa(sample["lE00"]) - sample["aurocE"]) + np.abs(pa(sample["lC00"]) - sample["aurocC"])) / 2
    e10 = (np.abs(pa(sample["lE10"]) - sample["aurocE"]) + np.abs(pa(sample["lC10"]) - sample["aurocC"])) / 2
    d = e00 - e10
    key = sample["rep"] * N_f + sample["f_idx"]
    fam_mean = np.bincount(key, weights=d) / np.bincount(key)
    per_rep = fam_mean.reshape(-1, N_f).mean(1)
    return float(per_rep.mean()), float(per_rep.std(ddof=1) / np.sqrt(len(per_rep)))


def _bisect(fun, target, lo, hi, iters=40):
    flo, fhi = fun(lo) - target, fun(hi) - target
    assert flo * fhi <= 0, ("bracket", flo, fhi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (fun(mid) - target) * flo <= 0:
            hi = mid
        else:
            lo, flo = mid, fun(mid) - target
    return 0.5 * (lo + hi)


def calibrate(P: dict, N_f: int, seed: int, mu_so: np.ndarray) -> tuple[dict, dict]:
    """Returns (calibration, truths). Calibration targets are hit on the
    reference sample by bisection; truths are re-measured on a fresh
    reference sample under the calibrated parameters and stored with
    their Monte Carlo standard errors (per-replication blocks)."""
    calib = {}
    # LEVEL shrink so that the reference truth equals the declared target (through the measurement path)
    if P["level_target"] != 0.0 and not P["beta_family_null"]:
        base = _ref_sample(P, N_f, seed, mu_so, {}, reps=REF_REPS // 5)

        def lv(s):
            return _level_of(_ref_sample(P, N_f, seed, mu_so, {"shrink": s}, reps=REF_REPS // 10), N_f)[0]
        calib["shrink"] = _bisect(lv, P["level_target"], -0.6, 0.6, iters=18)
        del base
    mode = P["comp_mode"]
    if mode in ("boundary_superior", "p00_advantage", "equivalent", "reference_off_margin"):
        target = {"boundary_superior": R.EPS_R, "p00_advantage": 0.005, "equivalent": 0.0005, "reference_off_margin": 1.5 * R.EPS_R}[mode]
        s = _ref_sample(P, N_f, seed, mu_so, calib, reps=REF_REPS // 5)
        k = FITTED.index(REFERENCE)
        calib["flip_q"] = _bisect(lambda q: _D_of(s, REFERENCE, q, k)[0], target, 0.0, 0.6, iters=30)
        del s
    if mode == "rotating_best":
        s = _ref_sample(P, N_f, seed, mu_so, calib, reps=REF_REPS // 5)
        calib["fix_q"] = _bisect(lambda q: _D_of(s, FITTED[0], q, 0, fix=True)[0], -0.005, 0.0, 0.9, iters=30)
        del s
    # truths under the calibrated parameters (fresh reference sample; nothing snapped)
    ks = range(len(FITTED)) if mode == "rotating_best" else [0]
    D_by_k, D_se_by_k = {}, {}
    for kk in ks:
        s = _ref_sample(P, N_f, seed + 1, mu_so, calib, best_k=kk)
        D_by_k[kk] = {}; D_se_by_k[kk] = {}
        for n in NAMES:
            D_by_k[kk][n], D_se_by_k[kk][n] = _D_of(s, n)
        if kk == 0:
            lvl, lvl_se = (0.0, 0.0) if P["beta_family_null"] else _level_of(s, N_f)
        del s
    exact = {"level_delta_exact_zero": bool(P["beta_family_null"]),
             "D_exact_zero": [n for n in FITTED if mode == "exchangeable" or (mode == "reference_off_margin" and n != REFERENCE)]}
    return calib, {"D_by_k": D_by_k, "D_se_by_k": D_se_by_k, "level_delta": lvl, "level_delta_se": lvl_se, "exact": exact,
                   "reference_cells": REF_REPS * N_f * 16}


# ---------------------------------------------------------------------------
# Inferential scenarios.
# ---------------------------------------------------------------------------

def cp_upper(k: int, n: int, level=0.975) -> float:
    return 1.0 if k >= n else float(beta_dist.ppf(level, k + 1, n - k))


def cp_lower(k: int, n: int, level=0.975) -> float:
    return 0.0 if k <= 0 else float(beta_dist.ppf(1 - level, k, n - k + 1))


def truth_D(tr: dict, kk: int, n: str) -> float:
    return 0.0 if n in tr["exact"]["D_exact_zero"] else tr["D_by_k"][kk][n]


def run_inferential(sc: dict, N_f: int, seed: int, reps: int, mults=MULTS) -> dict:
    P = sc["params"]; name = sc["name"]; key = skey(name)
    rng = np.random.default_rng([seed, key, N_f])
    mu_so = 0.03 * np.random.default_rng([7, key]).standard_normal((4, 4))
    calib, tr = calibrate(P, N_f, seed, mu_so)
    dl_true = 0.0 if tr["exact"]["level_delta_exact_zero"] else tr["level_delta"]
    mode = P["comp_mode"]
    cnt = {m: {"sel_false": 0, "sel_cov": 0, "sel_cov_n": 0, "lvl_false": 0, "lvl_cov": 0,
               "sel_claims": {"superior": 0, "inferior": 0, "equivalent": 0}, "lvl_claims": {"improvement": 0, "worsening": 0, "equivalent": 0, "one_point": 0}} for m in mults}
    declared = {"validator": 0, "sel_not_estimable": 0, "lvl_not_estimable": 0, "comparators_not_estimable": {n: 0 for n in NAMES}}
    hw_sel, hw_lvl, fam_deltas = [], [], []
    for rep in range(reps):
        kk = rep % len(FITTED) if mode == "rotating_best" else 0
        c = sim_cells(P, N_f, rng, mu_so, calib, kk)
        st = panel_state(c)
        fs = fast_sel(c) if st is None else {"state": st}
        fl = fast_level(c) if st is None else {"state": st}
        if rep < CHECK_READER:
            check_reader(c, fs, fl, name)
        if st:
            declared["validator"] += 1; continue
        if fs["state"]:
            declared["sel_not_estimable"] += 1
        else:
            for n in NAMES:
                declared["comparators_not_estimable"][n] += int(fs["D"][n] is None)
            t_sup = all(truth_D(tr, kk, n) > R.EPS_R for n in NAMES)
            D_ref = truth_D(tr, kk, REFERENCE)
            for m in mults:
                ivs = sel_intervals(fs, m)
                v = R.sel_verdict(ivs)
                false = ((v.startswith("PRACTICALLY SUPERIOR") and not t_sup)
                         or (v.startswith("PRACTICALLY INFERIOR") and not D_ref < -R.EPS_R)
                         or (v.startswith("PRACTICALLY EQUIVALENT") and not abs(D_ref) <= R.EPS_R))
                cnt[m]["sel_false"] += int(false)
                for kcl, tag in (("superior", "PRACTICALLY SUPERIOR"), ("inferior", "PRACTICALLY INFERIOR"), ("equivalent", "PRACTICALLY EQUIVALENT")):
                    cnt[m]["sel_claims"][kcl] += int(v.startswith(tag))
                est = {n: iv for n, iv in ivs.items() if iv is not None}
                if est:
                    cnt[m]["sel_cov_n"] += 1
                    cnt[m]["sel_cov"] += int(all(iv[0] <= truth_D(tr, kk, n) <= iv[1] for n, iv in est.items()))
                if m == 1.0 and est:
                    hw_sel.append(float(np.mean([(iv[1] - iv[0]) / 2 for iv in est.values()])))
        if fl["state"]:
            declared["lvl_not_estimable"] += 1
        else:
            if rep < 200:
                fam_deltas.append(fl["family_deltas"])
            for m in mults:
                iv = level_interval(fl, m)
                lv = R.level_verdict(iv)
                false = ((lv["verdict"] == "resolved improvement" and not dl_true > 0)
                         or (lv["verdict"] == "resolved worsening" and not dl_true < 0)
                         or (lv["equivalent_within_0.01"] and not abs(dl_true) <= R.MARGIN_LEVEL)
                         or (lv["at_least_one_point"] and not dl_true > R.MARGIN_LEVEL))
                cnt[m]["lvl_false"] += int(false)
                cnt[m]["lvl_cov"] += int(iv[0] <= dl_true <= iv[1])
                cnt[m]["lvl_claims"]["improvement"] += int(lv["verdict"] == "resolved improvement")
                cnt[m]["lvl_claims"]["worsening"] += int(lv["verdict"] == "resolved worsening")
                cnt[m]["lvl_claims"]["equivalent"] += int(lv["equivalent_within_0.01"])
                cnt[m]["lvl_claims"]["one_point"] += int(lv["at_least_one_point"])
                if m == 1.0:
                    hw_lvl.append((iv[1] - iv[0]) / 2)
    n_sel = reps - declared["validator"] - declared["sel_not_estimable"]
    n_lvl = reps - declared["validator"] - declared["lvl_not_estimable"]
    fd = np.concatenate(fam_deltas) if fam_deltas else np.array([np.nan])
    per_mult = lambda m, fam: ({"false_claim_rate": cnt[m]["sel_false"] / reps, "false_claim_upper": cp_upper(cnt[m]["sel_false"], reps),
                                "coverage": cnt[m]["sel_cov"] / max(cnt[m]["sel_cov_n"], 1), "coverage_lower": cp_lower(cnt[m]["sel_cov"], max(cnt[m]["sel_cov_n"], 1)),
                                "claims": {k: v / reps for k, v in cnt[m]["sel_claims"].items()}} if fam == "SEL" else
                               {"false_claim_rate": cnt[m]["lvl_false"] / reps, "false_claim_upper": cp_upper(cnt[m]["lvl_false"], reps),
                                "coverage": cnt[m]["lvl_cov"] / max(n_lvl, 1), "coverage_lower": cp_lower(cnt[m]["lvl_cov"], max(n_lvl, 1)),
                                "claims": {k: v / reps for k, v in cnt[m]["lvl_claims"].items()}})
    return {"scenario": name, "scenario_key": key, "held_out": sc["held_out"], "N_f": N_f, "seed": seed, "reps": reps,
            "params": P, "mu_so": mu_so.tolist(), "calibration": calib, "truth": tr,
            "declared": declared, "declared_expected": {"SEL": DECLARED_SEL.get(mode), "validator": bool(P["missing_cell"] or P["duplicate_family"]),
                                                          "nonfinite": bool(P["nonfinite"]), "undefined_p10": bool(P["undefined_p10"])},
            "SEL": {"n_inferential": n_sel, "per_mult": {str(m): per_mult(m, "SEL") for m in mults},
                    "median_half_width_mult1": (float(np.median(hw_sel)) if hw_sel else None)},
            "LEVEL": {"n_inferential": n_lvl, "per_mult": {str(m): per_mult(m, "LEVEL") for m in mults},
                      "median_half_width_mult1": (float(np.median(hw_lvl)) if hw_lvl else None),
                      "family_delta_skewness_first200": (float(pd.Series(fd).skew()) if np.isfinite(fd).sum() > 2 else None)}}


# ---------------------------------------------------------------------------
# HO scenarios: per-example fixtures through the frozen AUGRC arithmetic.
# ---------------------------------------------------------------------------

N_EX = 400          # examples per side per (checkpoint, shift); pi = 0.3 ID failures


def _stratum_truth(g, mode, geo):
    cross = -0.2 + 0.4 * (g + 1)
    d = np.array([-1.5, -0.5, 0.5, 1.5]) + 0.0031

    def y_of(dv):
        if mode in ("planted", None):
            return 0.02 * (dv - cross)
        if mode == "reversed":
            return 0.02 * (dv + cross)
        if mode == "all_left":
            return 0.02 * (dv + 5)
        if mode == "all_right":
            return 0.02 * (dv - 5)
        if mode == "nonlinear":
            return 0.02 * np.tanh(3 * (dv - cross)) if abs(cross) < 0.3 else 0.02 * np.sign(dv - cross) * 3
        if mode == "ushape":
            return 0.02 * (abs(dv - cross) - 0.8)
        raise ValueError(mode)
    return d, np.array([y_of(dv) for dv in d])


def expected_raw_gap(muE: float, muC: float, sC: float, pi: float, n_id: int, n_ood: int) -> float:
    """Expected raw AUGRC gap (Energy - CTM) of the construction: ID correct
    scores N(mu, s^2), ID failures N(0, s^2), OOD (all failures) N(-1, 1);
    AUGRC identity with the failure set = ID failures + OOD, whose failure
    AUROC is the failure-count-weighted mean of the two group AUROCs."""
    def A(mu, sc):
        a_f = norm.cdf(mu / (np.sqrt(2) * sc))                      # correct vs ID failure, both scale sc
        a_o = norm.cdf((mu + 1.0) / np.sqrt(sc ** 2 + 1.0))          # correct vs OOD
        nf = pi * n_id
        return (nf * a_f + n_ood * a_o) / (nf + n_ood)
    pim = (pi * n_id + n_ood) / (n_id + n_ood)
    return pim * (1 - pim) * (A(muC, sC) - A(muE, 1.0))


def sim_ho_source_examples(P: dict, rng) -> tuple[pd.DataFrame, dict]:
    """24 checkpoints (14 paradigm, 5 CE do0, 5 CE do1) x 4 shifts. Per cell:
    ID correctness (pi = 0.3), Gaussian Energy/CTM scores whose ID failure-AUROC
    gap follows the planted target through the exact identity; dG is then
    MEASURED by the frozen failure_augrc on the simulated examples (ID + OOD),
    and the TRUE curve is the expected raw gap of the same construction."""
    n_par, n_ce = 14, 10
    geo = P.get("geo", "reliable")
    g = np.concatenate([rng.uniform(-3, 1, n_par), rng.normal(-1, 0.03, 5), rng.normal(-0.2, 0.03, 5)])
    if geo == "near_zero":
        g = np.full(24, -1.0) + rng.normal(0, 1e-4, 24)
    elif geo == "clustered":
        g = np.repeat(np.array([-2.0, -1.0, 0.0]), 8) + rng.normal(0, 1e-3, 24)
    elif geo == "extreme":
        g[0] = 5.0
    elif geo == "shared_error":
        g = g + rng.normal(0, 0.5)
    pi, n_id, n_ood = 0.3, 2 * N_EX, N_EX
    rows, true_curves = [], {}
    for j in range(24):
        d, y = _stratum_truth(g[j], P["ho"], geo)
        sC = 1.0 + (0.5 * (g[j] + 1) if geo == "var_dep" else 0.0)
        tc = []
        for o in range(4):
            aE = 0.7; aC = float(np.clip(aE + y[o] / (pi * (1 - pi)), 0.02, 0.98))
            muE, muC = np.sqrt(2) * norm.ppf(aE), np.sqrt(2) * sC * norm.ppf(aC)
            tc.append((float(d[o]), expected_raw_gap(muE, muC, sC, pi, n_id, n_ood)))
            res = (rng.random(n_id) < pi).astype(float)
            sE = np.where(res == 0, muE, 0.0) + rng.standard_normal(n_id)
            sCc = np.where(res == 0, muC, 0.0) + sC * rng.standard_normal(n_id)
            sEo = rng.standard_normal(n_ood) - 1.0; sCo = rng.standard_normal(n_ood) - 1.0
            resm = np.concatenate([res, np.ones(n_ood)])
            gap_raw = failure_augrc(np.concatenate([sE, sEo]), resm) - failure_augrc(np.concatenate([sCc, sCo]), resm)
            comp = "paradigm_pool" if j < n_par else "standalone_ce"
            rows.append(dict(cell=f"c{j}", component=comp, paradigm=("dg" if j < 10 else "confidnet") if comp == "paradigm_pool" else "ce",
                             dropout=int(j >= n_par + 5) if comp == "standalone_ce" else j % 2, source="cifar10", ood_set=SHIFTS[o],
                             nc1=float(np.exp(g[j])), g=g[j], dK=d[o], dF=d[o] * 0.9, dG=float(gap_raw), M=0.0))
        true_curves[f"c{j}"] = tc
    return pd.DataFrame(rows), true_curves


def band_covers_truth(sub: pd.DataFrame, true_curves: dict, res: dict) -> dict:
    """Does each stratum's sup-norm band (q95 of bootstrap sup deviations) cover
    the stratum's TRUE PAVA curve (PAVA of the noiseless cell means)?"""
    from rn18_handoff_replication.rn18_analysis import _thirds
    strata = _thirds(sub)
    fine = np.linspace(sub.dK.min(), sub.dK.max(), FINE_N)
    out = {}
    for k, cells in strata.items():
        data_true = {c: true_curves[c] for c in cells}
        g_true = pava_curve("pava", data_true, sorted(data_true), fine)
        data_obs = {}
        for r in sub[sub.cell.isin(cells)].itertuples():
            data_obs.setdefault(r.cell, []).append((float(r.dK), float(r.dG)))
        g_obs = pava_curve("pava", data_obs, sorted(data_obs), fine)
        q = res["full_suite"][k].get("band_q95")
        out[k] = bool(q is not None and np.nanmax(np.abs(g_obs - g_true)) <= q)
    return out


def run_ho(sc: dict, seed: int, reps: int, reps_band: int, b_declared: int = 2000) -> dict:
    P = sc["params"]; key = skey(sc["name"])
    rng = np.random.default_rng([seed, key, 3])
    counts, cover = {}, {"strong": 0, "middle": 0, "weak": 0, "n": 0}
    t0 = time.time()
    for i in range(reps):
        sub, truth = sim_ho_source_examples(P, rng)
        if i < reps_band:
            res = ho_source(sub, "dK", b=b_declared)
            if "full_suite" in res:
                cv = band_covers_truth(sub, truth, res)
                for k in cv:
                    cover[k] += int(cv[k])
                cover["n"] += 1
        else:
            res = ho_source(sub, "dK", b=0)
        counts[res["verdict"]] = counts.get(res["verdict"], 0) + 1
    rates = {k: v / reps for k, v in counts.items()}
    return {"scenario": sc["name"], "scenario_key": key, "held_out": sc["held_out"], "reps": reps, "reps_band": reps_band,
            "per_example": {"n_id": 2 * N_EX, "n_ood": N_EX, "pi": 0.3, "arithmetic": "pilot0.extract_stage2_expansion.failure_augrc"},
            "verdict_rates": rates, "retained_rate": rates.get("HO-RETAINED", 0.0), "retained_upper": cp_upper(counts.get("HO-RETAINED", 0), reps),
            "band_coverage_of_true_curve": {k: (cover[k] / cover["n"] if cover["n"] else None) for k in ("strong", "middle", "weak")},
            "band_b": b_declared, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# Stages, multiplier selection, licenses.
# ---------------------------------------------------------------------------

def declared_expectation(P: dict, fam: str) -> dict | None:
    """What a declared (by-design) scenario must produce in EVERY replication;
    None for a regular scenario."""
    if P["missing_cell"] or P["duplicate_family"]:
        return {"validator": True}
    if P["nonfinite"]:
        return {"not_estimable": True}
    if fam == "SEL" and P["comp_mode"] == "identical":
        return {"comparator_not_estimable": REFERENCE}
    if fam == "SEL" and P["comp_mode"] == "degenerate_ctm":
        return {"comparator_not_estimable": "always_ctm", "regular_otherwise": True}
    if fam == "LEVEL" and P["undefined_p10"]:
        return {"not_estimable": True}
    return None


def _passes(r: dict, fam: str, m: float, alpha: float, cov_nom: float) -> bool:
    pm = r[fam]["per_mult"][str(m)]
    exp = declared_expectation(r["params"], fam)
    reps, d = r["reps"], r["declared"]
    if exp is None:
        return pm["false_claim_upper"] <= alpha and pm["coverage_lower"] >= cov_nom
    if exp.get("validator"):
        return d["validator"] == reps and pm["false_claim_rate"] == 0.0
    if exp.get("not_estimable"):
        key = "sel_not_estimable" if fam == "SEL" else "lvl_not_estimable"
        return d[key] == reps and pm["false_claim_rate"] == 0.0
    comp = exp["comparator_not_estimable"]
    ok = d["comparators_not_estimable"][comp] == r[fam]["n_inferential"] and pm["false_claim_rate"] == 0.0
    if exp.get("regular_otherwise"):
        ok = d["comparators_not_estimable"][comp] == r[fam]["n_inferential"] and pm["false_claim_upper"] <= alpha and pm["coverage_lower"] >= cov_nom
    return ok


def _nonest_ok(r: dict, fam: str) -> bool:
    if declared_expectation(r["params"], fam) is not None:
        return True
    return (r["reps"] - r[fam]["n_inferential"]) / r["reps"] < 0.01


def select_multipliers(dev: dict) -> dict:
    out = {}
    for fam, alpha in (("SEL", R.ALPHA_SEL), ("LEVEL", R.ALPHA_LEVEL)):
        for N_f in N_FS:
            chosen = None
            for m in MULTS:
                if all(_passes(r, fam, m, alpha, 1 - alpha) and _nonest_ok(r, fam) for r in dev["inferential"] if r["N_f"] == N_f):
                    chosen = m; break
            out[f"{fam}_Nf{N_f}"] = chosen
    return out


def licenses(audit: dict, mults: dict) -> dict:
    out = {}
    for fam, alpha in (("SEL", R.ALPHA_SEL), ("LEVEL", R.ALPHA_LEVEL)):
        for N_f in N_FS:
            m = mults.get(f"{fam}_Nf{N_f}")
            if m is None:
                out[f"{fam}_Nf{N_f}"] = {"licensed": False, "multiplier": None, "reason": "no multiplier passed development",
                                         "failed_scenarios": [r["scenario"] for r in audit["inferential"] if r["N_f"] == N_f
                                                              and not _passes(r, fam, MULTS[-1], alpha, 1 - alpha)]}
                continue
            fails = [r["scenario"] for r in audit["inferential"] if r["N_f"] == N_f and not _passes(r, fam, m, alpha, 1 - alpha)]
            fails += [r["scenario"] + " (non-estimable >= 1%)" for r in audit["inferential"] if r["N_f"] == N_f and not _nonest_ok(r, fam)]
            out[f"{fam}_Nf{N_f}"] = {"licensed": not fails, "multiplier": m, "failed_scenarios": fails}
    return out


def power_report(res: dict, mults: dict) -> dict:
    """Plan v3 section 7.1 reporting rule: power at the named minimal effects."""
    rep = {}
    by = {(r["scenario"], r["N_f"]): r for r in res["inferential"]}
    for N_f in N_FS:
        m_s, m_l = mults.get(f"SEL_Nf{N_f}") or 1.0, mults.get(f"LEVEL_Nf{N_f}") or 1.0
        g = lambda name, fam, m, claim: by[(name, N_f)][fam]["per_mult"][str(m)]["claims"][claim] if (name, N_f) in by else None
        rep[f"Nf{N_f}"] = {"SEL_superiority_power_at_0.005": g("pred_uniform_p00_advantage", "SEL", m_s, "superior"),
                           "SEL_equivalence_power_at_0": g("S0_base", "SEL", m_s, "equivalent"),
                           "LEVEL_improvement_power_at_0.01": g("pred_p10_improvement", "LEVEL", m_l, "improvement"),
                           "LEVEL_one_point_power_at_0.02": g("pred_p10_improvement_0.02", "LEVEL", m_l, "one_point"),
                           "LEVEL_improvement_power_two_clusters_0.011": g("dep_dropout_two_clusters", "LEVEL", m_l, "improvement"),
                           "LEVEL_false_direction_bounded_beta_null": g("null_level_bounded_beta_family", "LEVEL", m_l, "improvement")}
    return rep


def _job(args):
    kind, sc, seed, reps, reps_ho, reps_band, N_f = args
    return (kind, run_ho(sc, seed, reps_ho, reps_band) if kind == "ho" else run_inferential(sc, N_f, seed, reps))


def stage(name: str, reps: int, reps_ho: int, reps_band: int, workers: int = 1) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    seed = SEED_DEV if name == "dev" else SEED_AUDIT
    res = {"stage": name, "seed": seed, "reps": reps, "reps_ho": reps_ho, "reps_band": reps_band, "workers": workers, "inferential": [], "ho": []}
    jobs = []
    for sc in scenarios():
        if name == "dev" and sc["held_out"]:
            continue
        if sc["params"]["ho"] is not None:
            jobs.append(("ho", sc, seed, reps, reps_ho, reps_band, None))
        else:
            for N_f in N_FS:
                jobs.append(("inf", sc, seed, reps, reps_ho, reps_band, N_f))
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for f in as_completed(futs):
            kind, r = f.result()
            res[kind if kind == "ho" else "inferential"].append(r)
            done += 1
            print(f"[phase4v2/{name}] {r['scenario']} ({kind}{'' if kind == 'ho' else ' N_f=' + str(r['N_f'])}) done: {done}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
            (SIM / f"{name}_progress_v2.json").write_text(json.dumps({"done": done, "of": len(jobs), "seconds": time.time() - t0}))
    order = {s["name"]: i for i, s in enumerate(scenarios())}
    res["inferential"].sort(key=lambda r: (order[r["scenario"]], -r["N_f"]))
    res["ho"].sort(key=lambda r: order[r["scenario"]])
    return res


# ---------------------------------------------------------------------------
# Self-test (small replications): coherence, constants, exchangeability, declared states.
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(5)
    mu = np.zeros((4, 4))
    c = sim_cells(BASE, 10, rng, mu)
    # (1) M is derived from the forecasts (coherent with LEVEL), constants are literal
    assert np.allclose(c["M"], c["lE00"] - c["lC00"]) and (c["preds"]["always_energy"] == -1).all() and (c["preds"]["always_ctm"] == 1).all()
    pa = lambda l: 1 - np.exp(l)
    assert np.all((pa(c["lE00"]) > 0) & (pa(c["lE00"]) < 1) & (pa(c["lC10"]) > 0) & (pa(c["lC10"]) < 1))
    # (2) dependence and distribution reach the LEVEL contributions (F2's paired-stream test, inverted)
    a = sim_cells(dict(BASE, corr=0.0), 5, np.random.default_rng(19), mu)
    b = sim_cells(dict(BASE, corr=0.9), 5, np.random.default_rng(19), mu)
    e = lambda x: (np.abs(pa(x["lE00"]) - x["aurocE"]) + np.abs(pa(x["lC00"]) - x["aurocC"])) / 2 - (np.abs(pa(x["lE10"]) - x["aurocE"]) + np.abs(pa(x["lC10"]) - x["aurocC"])) / 2
    assert not np.array_equal(e(a), e(b))
    # (3) pi = 1 keeps every cell and zeroes the AUGRC gap there
    p1 = sim_cells(dict(BASE, pi_one_source=True), 5, np.random.default_rng(3), mu)
    assert p1["keep"].all() and np.all(p1["dG"][p1["s_idx"] == 3] == 0) and np.isfinite(p1["dA"]).all()
    # (4) declared states through the reader v2
    for kw, fam, want in ((dict(missing_cell=True), None, "VALIDATION FAILED [key_set]"), (dict(duplicate_family=True), None, "VALIDATION FAILED [families]")):
        cc = sim_cells(dict(BASE, **kw), 10, np.random.default_rng(1), mu)
        assert panel_state(cc) == want, (kw, panel_state(cc))
    nf = sim_cells(dict(BASE, nonfinite=True), 10, np.random.default_rng(1), mu)
    assert fast_sel(nf)["state"] and fast_level(nf)["state"]
    check_reader(nf, fast_sel(nf), fast_level(nf), "nonfinite")
    up = sim_cells(dict(BASE, undefined_p10=True), 10, np.random.default_rng(1), mu)
    assert fast_sel(up)["state"] is None and fast_level(up)["state"]
    check_reader(up, fast_sel(up), fast_level(up), "undefined_p10")
    ident = sim_cells(dict(BASE, comp_mode="identical"), 10, np.random.default_rng(1), mu)
    fs = fast_sel(ident); assert all(fs["D"][n] is None for n in FITTED) and all(fs["D"][n] is not None for n in ("always_energy", "always_ctm"))
    check_reader(ident, fs, fast_level(ident), "identical")
    deg = sim_cells(dict(BASE, comp_mode="degenerate_ctm", p00_bias=0.3), 10, np.random.default_rng(1), mu)
    fsd = fast_sel(deg); assert (deg["M"] > 0).all() and fsd["D"]["always_ctm"] is None and fsd["D"][REFERENCE] is not None
    check_reader(deg, fsd, fast_level(deg), "degenerate")
    # (5) bounded beta family null: exact family deltas, valid forecasts
    bb = sim_cells(dict(BASE, beta_family_null=True), 10, np.random.default_rng(2), mu)
    fl = fast_level(bb); assert fl["state"] is None and len(fl["family_deltas"]) == 10
    # (6) reader agreement on the base fixture; (7) exchangeable clone truth ~ 0 on a small reference
    check_reader(c, fast_sel(c), fast_level(c), "base")
    s = _ref_sample(BASE, 10, 11, mu, {}, reps=300)
    d0, se0 = _D_of(s, REFERENCE)
    assert abs(d0) < 4 * se0 + 1e-4, (d0, se0)
    # (8) per-example HO fixture: planted ordering retained at b=0 through failure_augrc; reversed not retained
    sub, truth = sim_ho_source_examples(dict(BASE, ho="planted"), np.random.default_rng(9))
    r_pl = ho_source(sub, "dK", b=0)["verdict"]
    sub_r, _ = sim_ho_source_examples(dict(BASE, ho="reversed"), np.random.default_rng(9))
    r_rv = ho_source(sub_r, "dK", b=0)["verdict"]
    assert r_pl == "HO-RETAINED" and r_rv != "HO-RETAINED", (r_pl, r_rv)
    res_b = ho_source(sub, "dK", b=30); cv = band_covers_truth(sub, truth, res_b)
    assert set(cv) == {"strong", "middle", "weak"}
    # (9) stable digests
    assert skey("S0_base") == skey("S0_base") and skey("S0_base") != skey("dist_t3")
    print(f"[phase4-v2] self-test PASS: coherent M/LEVEL, literal constants, family-level dependence reaches LEVEL, "
          f"pi=1 keeps cells, declared states match reader v2, exchangeable D_ref {d0:+.5f} (se {se0:.5f}), "
          f"per-example HO planted={r_pl} reversed={r_rv}, band check {cv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--stage", choices=("dev", "audit"))
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--reps-ho", type=int, default=2000)
    ap.add_argument("--reps-band", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    SIM.mkdir(parents=True, exist_ok=True)
    if args.write_manifest:
        man = {"plan": "v3 section 7.2 + correction 2026-09-09 (status-review F2, F11)", "version": 2, "n_scenarios": 47,
               "held_out": [s["name"] for s in scenarios() if s["held_out"]], "seeds": {"dev": SEED_DEV, "audit": SEED_AUDIT},
               "scenario_keys": {s["name"]: skey(s["name"]) for s in scenarios()}, "multipliers": MULTS, "N_f": N_FS,
               "reps_inferential": 10000, "reps_ho": 2000, "reps_band_at_declared_b": 50, "reference_cells": REF_REPS * 160,
               "reader": "rn18_analysis_v2.py", "scenarios": scenarios()}
        text = json.dumps(man, indent=1, default=str)
        (SIM / "design_manifest_v2.json").write_text(text)
        print("design_manifest_v2.json sha256", hashlib.sha256(text.encode()).hexdigest())
        return
    if args.stage == "dev":
        dev = stage("dev", args.reps, args.reps_ho, args.reps_band, args.workers)
        mults = select_multipliers(dev)
        (SIM / "development_results_v2.json").write_text(json.dumps(dev, indent=1, default=str))
        (SIM / "development_critical_values_v2.json").write_text(json.dumps({"multipliers": mults, "rule": f"smallest in {MULTS} passing every non-held-out scenario",
                                                                             "seed": SEED_DEV, "power": power_report(dev, mults)}, indent=1))
        print("multipliers:", mults)
    elif args.stage == "audit":
        mults = json.loads((SIM / "development_critical_values_v2.json").read_text())["multipliers"]
        audit = stage("audit", args.reps, args.reps_ho, args.reps_band, args.workers)
        lic = licenses(audit, mults)
        (SIM / "audit_results_v2.json").write_text(json.dumps(audit, indent=1, default=str))
        (SIM / "qualification_report_v2.json").write_text(json.dumps({"version": 2, "licenses": lic, "multipliers": mults, "seed": SEED_AUDIT,
                                                                        "power": power_report(audit, mults), "ho_operating_characteristics": audit["ho"],
                                                                        "correction_of": "simulations/qualification_report.json (version 1, preserved)"}, indent=1, default=str))
        (SIM / "sample_size_decision_v2.json").write_text(json.dumps({"N_f_fixed": {"SEL_LEVEL": 10, "ORG_descriptive": 5}, "no_sample_size_selection": True, "licenses": lic}, indent=1))
        print(json.dumps(lic, indent=1))


if __name__ == "__main__":
    main()

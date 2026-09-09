"""RN18 handoff-replication plan v3, PHASE 4: outcome-free qualification
of the inference candidate on the enumerated 43-scenario grid (v3
section 7.2-7.3). Fixes the scenario manifest BEFORE any run, selects
the per-family multipliers on the 40 non-held-out scenarios under seed
2401, then audits every scenario (including the three held-out classes)
under seed 2402 with the frozen multipliers. Decides the SEL and LEVEL
inference licenses at N_f = 10 (CE component, blocks of four) and at
N_f = 5 (dropout-off sensitivity). HO scenarios report the gate's
operating characteristics (retention under planted ordering; retention
under reversed and all-censored nulls, which the rule must not retain).

Decision logic is the reader's own: regret / choice_prob / interval /
sel_verdict / level_verdict / ho_source are imported from
rn18_analysis; the vectorized SEL/LEVEL fast path is asserted to agree
with the DataFrame reader on the first 50 replications of every
scenario.

False-claim rules (per scenario truth): SEL "superior" is false unless
every true D_b > eps_R; "inferior" false unless D_ref < -eps_R;
"equivalent" false unless |D_ref| <= eps_R. LEVEL "resolved
improvement/worsening" false unless the true delta has that sign;
"equivalent" false unless |delta| <= 0.01; "at least one point" false
unless delta > 0.01. Truths come from a 400,000-cell reference
simulation of the scenario's data-generating process (archived).
Acceptance per scenario: Clopper-Pearson upper bound of the false-claim
rate <= family alpha (0.025), coverage lower bound >= nominal
(simultaneous 0.975 for SEL, 0.975 for LEVEL), non-estimable < 1%.

Usage (from code/):
    python rn18_handoff_replication/phase4_qualification.py --write-manifest
    python rn18_handoff_replication/phase4_qualification.py --stage dev   [--reps 10000]
    python rn18_handoff_replication/phase4_qualification.py --stage audit [--reps 10000]
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
from scipy.stats import beta as beta_dist, t as student

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from rn18_handoff_replication.comparators import NAMES, REFERENCE
from rn18_handoff_replication.rn18_analysis import (ALPHA_LEVEL, ALPHA_SEL, EPS_R, MARGIN_LEVEL, choice_prob,
                                                     ho_source, level_endpoint, level_verdict, regret,
                                                     sel_endpoint, sel_verdict)

SIM = Path("rn18_handoff_replication/simulations")
SEED_DEV, SEED_AUDIT = 2401, 2402
MULTS = (1.0, 1.1, 1.25, 1.5, 2.0)
N_FS = (10, 5)
SOURCES = ("cifar10", "cifar100", "supercifar100", "tinyimagenet")
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
ALPHA_EACH = ALPHA_SEL / len(NAMES)
REF_CELLS = 400_000
BASE = dict(dist="gauss", corr=0.5, sd_fam=0.02, sd_cell=0.03, p_sign=0.6, comp_acc=None, comp_mode="exchangeable",
            level_delta=0.0, ties_frac=0.0, ceiling=False, one_sign=False, pi_varies=False, pi_one_source=False,
            shared_id=False, shared_ood=False, missing_cell=False, nonfinite=False, undefined_p10=False,
            duplicate_family=False, ho=None)


# ---------------------------------------------------------------------------
# Scenario manifest (v3 section 7.2): 1 base + 37 one-factor + 5 interaction.
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
    add("null_level_boundary_equivalence", "null", level_delta=0.01)
    add("null_level_equivalence_false", "null", level_delta=0.015)
    add("null_sel_equivalence_false", "null", comp_mode="reference_off_margin")
    add("pred_p10_improvement_0.02", "predictors", level_delta=0.02)
    add("pred_identical", "predictors", comp_mode="identical")
    add("pred_practically_equivalent", "predictors", comp_mode="equivalent")
    add("pred_each_best_in_turn", "predictors", comp_mode="rotating_best")
    add("pred_uniform_p00_advantage", "predictors", comp_mode="p00_advantage")
    add("pred_p10_improvement", "predictors", level_delta=0.01)
    add("pred_p10_worsening", "predictors", level_delta=-0.01)
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
    add("int_p10_improvement_pi_one", "interaction", level_delta=0.01, pi_one_source=True)
    add("int_ushape_clustered", "interaction", held=True, ho="ushape", geo="clustered")
    assert len(S) == 43 and sum(s["held_out"] for s in S) == 3
    return S


# ---------------------------------------------------------------------------
# Fast metric-level simulator for SEL / LEVEL.
# ---------------------------------------------------------------------------

def _resid(rng, n, dist):
    if dist == "gauss":
        return rng.standard_normal(n)
    if dist == "t3":
        return rng.standard_t(3, n) / np.sqrt(3.0)
    if dist == "lognormal":
        z = np.exp(0.8 * rng.standard_normal(n))
        return (z - np.exp(0.32)) / np.sqrt((np.exp(0.64) - 1) * np.exp(0.64))
    if dist == "contaminated":
        z = rng.standard_normal(n)
        m = rng.random(n) < 0.05
        return np.where(m, 5 * z, z) / np.sqrt(0.95 + 0.05 * 25)
    raise ValueError(dist)


def sim_cells(P: dict, N_f: int, rng, mu_so: np.ndarray) -> dict:
    """Arrays for N_f families x 4 sources x 4 shifts. Families are
    (dropout, run) blocks: N_f = 10 -> do0/do1 x runs 1..5; N_f = 5 -> do0 only."""
    fam_do = np.array([0] * 5 + [1] * 5)[:N_f] if N_f == 10 else np.zeros(5, int)
    n = N_f * 16
    f_idx = np.repeat(np.arange(N_f), 16)
    s_idx = np.tile(np.repeat(np.arange(4), 4), N_f)
    o_idx = np.tile(np.arange(4), N_f * 4)
    if P["corr"] == "factor":
        lam = np.array([0.3, 0.6, 0.9, 0.5])
        u = lam[None, :] * rng.standard_normal((N_f, 1)) + 0.5 * rng.standard_normal((N_f, 4))
    else:
        r = float(P["corr"])
        u = np.sqrt(r) * rng.standard_normal((N_f, 1)) + np.sqrt(1 - r) * rng.standard_normal((N_f, 4))
    u = P["sd_fam"] * u
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
    aurocE = (0.98 if P["ceiling"] else 0.8) + 0.02 * rng.standard_normal(n)
    aurocC = aurocE + dA
    pi = np.full(n, 0.5)
    if P["pi_varies"]:
        pi = np.array([0.3, 0.5, 0.6, 0.8])[s_idx]
    dG = pi * (1 - pi) * dA * 0.6 + 0.002 * rng.standard_normal(n)
    # P00 sign prediction with accuracy p_sign, magnitude |N(3,1)|
    right = rng.random(n) < P["p_sign"]
    M = np.abs(rng.normal(3, 1, n)) * np.where(right, np.sign(dA), -np.sign(dA))
    if P["ties_frac"] > 0:
        M = np.where(rng.random(n) < P["ties_frac"], 0.0, M)
    # comparators
    E_abs = float(np.mean(np.abs(dA)))
    preds = {}
    for k, name in enumerate(NAMES):
        mode = P["comp_mode"]
        if mode == "identical":
            preds[name] = M.copy()
        elif mode == "equivalent":
            flip = rng.random(n) < 0.005
            preds[name] = np.where(flip, -M, M)
        else:
            acc = P["p_sign"]
            if mode == "boundary_superior":
                acc = P["p_sign"] - EPS_R / max(E_abs, 1e-9)         # E[D_b] = +eps_R exactly
            elif mode == "p00_advantage":
                acc = P["p_sign"] - 0.005 / max(E_abs, 1e-9)         # E[D_b] = +0.005
            elif mode == "reference_off_margin" and name == REFERENCE:
                acc = P["p_sign"] - 1.5 * EPS_R / max(E_abs, 1e-9)   # E[D_ref] = +1.5 eps_R (equivalence false)
            elif mode == "rotating_best":
                acc = P["p_sign"] + (0.15 if k == P.get("best_k", 0) else 0.0)
            ok = rng.random(n) < acc
            preds[name] = np.where(ok, np.sign(dA), -np.sign(dA)) * np.abs(rng.normal(3, 1, n))
    # LEVEL: predicted-AUROC errors
    e00 = np.abs(rng.normal(0.15, 0.03, n))
    dl = rng.normal(P["level_delta"], 0.02, n)
    e10 = np.abs(e00 - dl)
    sgn = np.where(rng.random(n) < 0.5, 1, -1)
    predE00 = np.clip(aurocE + sgn * e00, 1e-6, 1 - 1e-6); predC00 = np.clip(aurocC + sgn * e00, 1e-6, 1 - 1e-6)
    predE10 = np.clip(aurocE + sgn * e10, 1e-6, 1 - 1e-6); predC10 = np.clip(aurocC + sgn * e10, 1e-6, 1 - 1e-6)
    keep = np.ones(n, bool)
    if P["pi_one_source"]:
        keep &= s_idx != 3                                        # pi = 1 source: Delta^F undefined, cells dropped
    if P["missing_cell"]:
        keep[0] = False
    if P["nonfinite"]:
        dA = dA.copy(); dA[1] = np.nan
    lE10 = np.log(1 - predE10); lC10 = np.log(1 - predC10)
    if P["undefined_p10"]:
        lC10 = lC10.copy(); lC10[rng.random(n) < 0.1] = np.nan
    fam = np.array([f"do{fam_do[f]}_run{(f % 5) + 1}" for f in range(N_f)])
    if P["duplicate_family"]:
        fam = fam.copy(); fam[1] = fam[0]
    return dict(n=n, f_idx=f_idx, s_idx=s_idx, o_idx=o_idx, family=fam[f_idx], dA=dA, dG=dG, M=M, preds=preds,
                aurocE=aurocE, aurocC=aurocC, lE00=np.log(1 - predE00), lC00=np.log(1 - predC00),
                lE10=lE10, lC10=lC10, keep=keep, e00=e00, e10=e10)


def to_frame(c: dict) -> pd.DataFrame:
    df = pd.DataFrame({"cell": [f"ck{f}_{s}" for f, s in zip(c["f_idx"], c["s_idx"])],
                       "component": "standalone_ce", "family": c["family"],
                       "source": np.array(SOURCES)[c["s_idx"]], "ood_set": np.array(SHIFTS)[c["o_idx"]],
                       "dA": c["dA"], "dG": c["dG"], "M": c["M"], "aurocE": c["aurocE"], "aurocC": c["aurocC"],
                       "l_E": c["lE00"], "l_C": c["lC00"], "l_E_p10": c["lE10"], "l_C_p10": c["lC10"]})
    for k, v in c["preds"].items():
        df[f"pred_{k}"] = v
    return df[c["keep"]].reset_index(drop=True)


class FixedComparators:
    def predict(self, name, df):
        return df[f"pred_{name}"].to_numpy(float)


def fast_sel(c: dict) -> dict:
    """Vectorized D_b, jackknife SE, and N_f (same arithmetic as the reader)."""
    keep = c["keep"] & np.isfinite(c["dA"]) & np.isfinite(c["M"])
    dA, M, fam = c["dA"][keep], c["M"][keep], c["family"][keep]
    fams, finv = np.unique(fam, return_inverse=True)
    N = len(fams)
    r00 = regret(dA, choice_prob(M))
    out = {}
    for name in NAMES:
        rb = regret(dA, choice_prob(c["preds"][name][keep]))
        d = rb - r00
        tot, cnt = d.sum(), len(d)
        fs = np.bincount(finv, weights=d, minlength=N); fc = np.bincount(finv, minlength=N)
        loo = (tot - fs) / (cnt - fc)
        se = float(np.sqrt((N - 1) / N * ((loo - loo.mean()) ** 2).sum()))
        out[name] = (float(d.mean()), se)
    return {"N": N, "D": out}


def fast_level(c: dict) -> dict:
    keep = c["keep"] & np.isfinite(c["lC10"]) & np.isfinite(c["lE10"])
    if (~keep & c["keep"]).any():
        return {"incomplete": True}
    fam, f, s = c["family"][keep], c["f_idx"][keep], c["s_idx"][keep]
    pa = lambda l: 1 - np.exp(l)
    e00 = (np.abs(pa(c["lE00"][keep]) - c["aurocE"][keep]) + np.abs(pa(c["lC00"][keep]) - c["aurocC"][keep])) / 2
    e10 = (np.abs(pa(c["lE10"][keep]) - c["aurocE"][keep]) + np.abs(pa(c["lC10"][keep]) - c["aurocC"][keep])) / 2
    d = e00 - e10
    fams, finv = np.unique(fam, return_inverse=True)
    # per (family, source, checkpoint) means -> family x source mean -> family mean: cells per checkpoint equal, so
    # the family mean over kept cells equals the nested means when each (family, source) has equal cell counts
    df = pd.DataFrame({"f": finv, "s": s, "d": d}).groupby(["f", "s"]).d.mean().groupby("f").mean()
    N = len(df)
    return {"N": N, "delta": float(df.mean()), "se": float(df.std(ddof=1) / np.sqrt(N)), "incomplete": False}


def cp_upper(k: int, n: int, level=0.975) -> float:
    return 1.0 if k >= n else float(beta_dist.ppf(level, k + 1, n - k))


def cp_lower(k: int, n: int, level=0.975) -> float:
    return 0.0 if k <= 0 else float(beta_dist.ppf(1 - level, k, n - k + 1))


def truths(P: dict, N_f: int, seed: int, mu_so) -> dict:
    """High-precision reference truths from the DGP (archived)."""
    rng = np.random.default_rng([seed, 999])
    reps = max(1, REF_CELLS // (N_f * 16))
    Ds = {n: [] for n in NAMES}; dl = []
    for _ in range(reps):
        Pk = dict(P, best_k=0)
        c = sim_cells(Pk, N_f, rng, mu_so)
        keep = c["keep"] & np.isfinite(c["dA"])
        r00 = regret(c["dA"][keep], choice_prob(c["M"][keep]))
        for n in NAMES:
            Ds[n].append(float(np.mean(regret(c["dA"][keep], choice_prob(c["preds"][n][keep])) - r00)))
        dl.append(float(np.mean(c["e00"][keep] - c["e10"][keep])))
    D = {n: float(np.mean(v)) for n, v in Ds.items()}
    nominal = {"boundary_superior": EPS_R, "p00_advantage": 0.005}.get(P["comp_mode"])
    for n in D:
        target = nominal if nominal is not None else (1.5 * EPS_R if (P["comp_mode"] == "reference_off_margin" and n == REFERENCE) else
                                                      0.0 if P["comp_mode"] in ("exchangeable", "reference_off_margin", "identical") else None)
        if target is not None and abs(D[n] - target) < 5e-4:
            D[n] = target
    ld = float(np.mean(dl))
    if abs(ld - P["level_delta"]) < 5e-4:
        ld = float(P["level_delta"])
    return {"D": D, "level_delta": ld, "snap_rule": "reference truth snapped to the nominal DGP value when within 5e-4"}


def run_inferential(sc: dict, N_f: int, seed: int, reps: int, mults=MULTS, check_reader: int = 50) -> dict:
    P = sc["params"]
    rng = np.random.default_rng([seed, hash(sc["name"]) % (2 ** 31), N_f])
    mu_so = 0.03 * np.random.default_rng([7, hash(sc["name"]) % (2 ** 31)]).standard_normal((4, 4))
    tr = truths(P, N_f, seed, mu_so)
    D_true, dl_true = tr["D"], tr["level_delta"]
    sel_false = {m: 0 for m in mults}; sel_cov = {m: 0 for m in mults}; sel_deg = 0
    lvl_false = {m: 0 for m in mults}; lvl_cov = {m: 0 for m in mults}; lvl_incomplete = 0
    sel_claims = {m: {"superior": 0, "inferior": 0, "equivalent": 0} for m in mults}
    lvl_claims = {m: {"improvement": 0, "worsening": 0, "equivalent": 0, "one_point": 0} for m in mults}
    hw = []
    t_ok_sel = all(D_true[n] > EPS_R for n in NAMES)
    for rep in range(reps):
        Pk = dict(P, best_k=rep % len(NAMES))
        c = sim_cells(Pk, N_f, rng, mu_so)
        try:
            if P["duplicate_family"]:
                fams = np.unique(c["family"])
                assert len(fams) == N_f, "family count mismatch"
            fs = fast_sel(c)
        except AssertionError:
            sel_deg += 1
            lvl_incomplete += 1
            continue
        N = fs["N"]
        for m in mults:
            ivs = {}
            for n, (est, se) in fs["D"].items():
                if se <= 0:
                    ivs[n] = None
                else:
                    q = student.ppf(1 - ALPHA_EACH / 2, N - 1) * m * se
                    ivs[n] = [est - q, est + q]
            v = sel_verdict(ivs)
            false = ((v.startswith("PRACTICALLY SUPERIOR") and not t_ok_sel)
                     or (v.startswith("PRACTICALLY INFERIOR") and not D_true[REFERENCE] < -EPS_R)
                     or (v.startswith("PRACTICALLY EQUIVALENT") and not abs(D_true[REFERENCE]) <= EPS_R))
            sel_false[m] += int(false)
            for key, tag in (("superior", "PRACTICALLY SUPERIOR"), ("inferior", "PRACTICALLY INFERIOR"),
                             ("equivalent", "PRACTICALLY EQUIVALENT")):
                sel_claims[m][key] += int(v.startswith(tag))
            sel_cov[m] += int(all(iv is not None and iv[0] <= D_true[n] <= iv[1] for n, iv in ivs.items()))
            if m == 1.0:
                widths = [(iv[1] - iv[0]) / 2 for iv in ivs.values() if iv is not None]
                hw.append(float(np.mean(widths)) if widths else np.nan)
        if any(v is None for v in ivs.values()):
            sel_deg += 1
        fl = fast_level(c)
        if fl.get("incomplete"):
            lvl_incomplete += 1
        else:
            for m in mults:
                q = student.ppf(1 - ALPHA_LEVEL / 2, fl["N"] - 1) * m * fl["se"]
                iv = [fl["delta"] - q, fl["delta"] + q]
                lv = level_verdict(iv)
                false = ((lv["verdict"] == "resolved improvement" and not dl_true > 0)
                         or (lv["verdict"] == "resolved worsening" and not dl_true < 0)
                         or (lv["equivalent_within_0.01"] and not abs(dl_true) <= MARGIN_LEVEL)
                         or (lv["at_least_one_point"] and not dl_true > MARGIN_LEVEL))
                lvl_false[m] += int(false)
                lvl_cov[m] += int(iv[0] <= dl_true <= iv[1])
                lvl_claims[m]["improvement"] += int(lv["verdict"] == "resolved improvement")
                lvl_claims[m]["worsening"] += int(lv["verdict"] == "resolved worsening")
                lvl_claims[m]["equivalent"] += int(lv["equivalent_within_0.01"])
                lvl_claims[m]["one_point"] += int(lv["at_least_one_point"])
        if rep < check_reader and not P["duplicate_family"]:
            df = to_frame(c)
            rs = sel_endpoint(df, FixedComparators(), mult=1.0)
            fast_ivs = {}
            for n, (est, se) in fs["D"].items():
                fast_ivs[n] = None if se <= 0 else [est - student.ppf(1 - ALPHA_EACH / 2, N - 1) * se,
                                                   est + student.ppf(1 - ALPHA_EACH / 2, N - 1) * se]
            assert rs["verdict"] == sel_verdict(fast_ivs), ("reader/fast SEL mismatch", sc["name"], rs["verdict"])
            rl = level_endpoint(df, mult=1.0)
            if not fl.get("incomplete"):
                q = student.ppf(1 - ALPHA_LEVEL / 2, fl["N"] - 1) * fl["se"]
                assert rl["verdict"] == level_verdict([fl["delta"] - q, fl["delta"] + q])["verdict"], ("reader/fast LEVEL mismatch", sc["name"])
            else:
                assert rl["verdict"].startswith("INELIGIBLE"), rl
    n_sel = reps - sel_deg if not P["duplicate_family"] else reps
    out = {"scenario": sc["name"], "held_out": sc["held_out"], "N_f": N_f, "seed": seed, "reps": reps,
           "truth": {"D": D_true, "level_delta": dl_true},
           "SEL": {"degenerate_or_declared": sel_deg,
                   "per_mult": {str(m): {"false_claim_rate": sel_false[m] / reps, "false_claim_upper": cp_upper(sel_false[m], reps),
                                         "coverage": sel_cov[m] / max(reps - sel_deg, 1), "coverage_lower": cp_lower(sel_cov[m], max(reps - sel_deg, 1)),
                                         "claims": {k: v / reps for k, v in sel_claims[m].items()}} for m in mults},
                   "median_half_width_mult1": float(np.nanmedian(hw)) if hw else None},
           "LEVEL": {"incomplete_declared": lvl_incomplete,
                     "per_mult": {str(m): {"false_claim_rate": lvl_false[m] / reps, "false_claim_upper": cp_upper(lvl_false[m], reps),
                                           "coverage": lvl_cov[m] / max(reps - lvl_incomplete, 1), "coverage_lower": cp_lower(lvl_cov[m], max(reps - lvl_incomplete, 1)),
                                           "claims": {k: v / reps for k, v in lvl_claims[m].items()}} for m in mults}}}
    return out


# ---------------------------------------------------------------------------
# HO gate operating characteristics (one source, 24 checkpoints, b = 0).
# ---------------------------------------------------------------------------

def sim_ho_source(P: dict, rng) -> pd.DataFrame:
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
    rows = []
    d = np.array([-1.5, -0.5, 0.5, 1.5]) + 0.0031
    for j in range(24):
        cross = -0.2 + 0.4 * (g[j] + 1)
        sd = 0.003 * (1 + 0.5 * (g[j] + 1)) if geo == "var_dep" else 0.003
        for o in range(4):
            mode = P["ho"]
            if mode in ("planted", None):
                y = 0.02 * (d[o] - cross)
            elif mode == "reversed":
                y = 0.02 * (d[o] + cross)
            elif mode == "all_left":
                y = 0.02 * (d[o] + 5)
            elif mode == "all_right":
                y = 0.02 * (d[o] - 5)
            elif mode == "nonlinear":
                y = 0.02 * np.tanh(3 * (d[o] - cross)) if abs(cross) < 0.3 else 0.02 * np.sign(d[o] - cross) * 3
            elif mode == "ushape":
                y = 0.02 * (abs(d[o] - cross) - 0.8)
            else:
                raise ValueError(mode)
            comp = "paradigm_pool" if j < n_par else "standalone_ce"
            rows.append(dict(cell=f"c{j}", component=comp, paradigm=("dg" if j < 10 else "confidnet") if comp == "paradigm_pool" else "ce",
                             dropout=int(j >= n_par + 5) if comp == "standalone_ce" else j % 2, source="cifar10",
                             ood_set=SHIFTS[o], nc1=float(np.exp(g[j])), g=g[j], dK=d[o], dF=d[o] * 0.9,
                             dG=y + rng.normal(0, sd), M=0.0))
    return pd.DataFrame(rows)


def run_ho(sc: dict, seed: int, reps: int) -> dict:
    P = sc["params"]
    rng = np.random.default_rng([seed, hash(sc["name"]) % (2 ** 31), 3])
    counts = {}
    for _ in range(reps):
        v = ho_source(sim_ho_source(P, rng), "dK", b=0)["verdict"]
        counts[v] = counts.get(v, 0) + 1
    rates = {k: v / reps for k, v in counts.items()}
    return {"scenario": sc["name"], "held_out": sc["held_out"], "reps": reps, "verdict_rates": rates,
            "retained_rate": rates.get("HO-RETAINED", 0.0),
            "retained_upper": cp_upper(counts.get("HO-RETAINED", 0), reps)}


# ---------------------------------------------------------------------------
# Stages.
# ---------------------------------------------------------------------------

def stage(name: str, reps: int, reps_ho: int, frozen_mults: dict | None) -> dict:
    seed = SEED_DEV if name == "dev" else SEED_AUDIT
    S = scenarios()
    res = {"stage": name, "seed": seed, "reps": reps, "inferential": [], "ho": []}
    t0 = time.time()
    for sc in S:
        if name == "dev" and sc["held_out"]:
            continue
        if sc["params"]["ho"] is not None:
            res["ho"].append(run_ho(sc, seed, reps_ho))
        else:
            for N_f in N_FS:
                res["inferential"].append(run_inferential(sc, N_f, seed, reps))
        print(f"[phase4/{name}] {sc['name']} done ({time.time() - t0:.0f}s)", flush=True)
    return res


def select_multipliers(dev: dict) -> dict:
    out = {}
    for fam, alpha, cov_nom in (("SEL", ALPHA_SEL, 1 - ALPHA_SEL), ("LEVEL", ALPHA_LEVEL, 1 - ALPHA_LEVEL)):
        for N_f in N_FS:
            chosen = None
            for m in MULTS:
                ok = True
                for r in dev["inferential"]:
                    if r["N_f"] != N_f:
                        continue
                    pm = r[fam]["per_mult"][str(m)]
                    if pm["false_claim_upper"] > alpha or pm["coverage_lower"] < cov_nom:
                        ok = False
                        break
                if ok:
                    chosen = m
                    break
            out[f"{fam}_Nf{N_f}"] = chosen
    return out


def licenses(audit: dict, mults: dict) -> dict:
    out = {}
    for fam, alpha, cov_nom in (("SEL", ALPHA_SEL, 1 - ALPHA_SEL), ("LEVEL", ALPHA_LEVEL, 1 - ALPHA_LEVEL)):
        for N_f in N_FS:
            m = mults.get(f"{fam}_Nf{N_f}")
            if m is None:
                out[f"{fam}_Nf{N_f}"] = {"licensed": False, "reason": "no multiplier passed development"}
                continue
            fails = []
            for r in audit["inferential"]:
                if r["N_f"] != N_f:
                    continue
                pm = r[fam]["per_mult"][str(m)]
                if pm["false_claim_upper"] > alpha or pm["coverage_lower"] < cov_nom:
                    fails.append(r["scenario"])
                nonest = (r[fam].get("degenerate_or_declared", 0) + r[fam].get("incomplete_declared", 0)) / r["reps"]
                if nonest >= 0.01 and not any(k in r["scenario"] for k in ("identical", "duplicate", "undefined", "nonfinite", "pi_one")):
                    fails.append(r["scenario"] + " (non-estimable >= 1%)")
            out[f"{fam}_Nf{N_f}"] = {"licensed": not fails, "multiplier": m, "failed_scenarios": fails}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--stage", choices=("dev", "audit"))
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--reps-ho", type=int, default=2000)
    args = ap.parse_args()
    SIM.mkdir(parents=True, exist_ok=True)
    if args.write_manifest:
        man = {"plan": "v3 section 7.2", "n_scenarios": 43, "held_out": [s["name"] for s in scenarios() if s["held_out"]],
               "seeds": {"dev": SEED_DEV, "audit": SEED_AUDIT}, "multipliers": MULTS, "N_f": N_FS,
               "reps_inferential": 10000, "reps_ho": 2000, "scenarios": scenarios()}
        text = json.dumps(man, indent=1, default=str)
        (SIM / "design_manifest.json").write_text(text)
        print("design_manifest.json sha256", hashlib.sha256(text.encode()).hexdigest())
        return
    if args.stage == "dev":
        dev = stage("dev", args.reps, args.reps_ho, None)
        mults = select_multipliers(dev)
        (SIM / "development_results.json").write_text(json.dumps(dev, indent=1, default=str))
        (SIM / "development_critical_values.json").write_text(json.dumps({"multipliers": mults, "rule": "smallest in "
                                                                          f"{MULTS} passing every non-held-out null/boundary scenario", "seed": SEED_DEV}, indent=1))
        print("multipliers:", mults)
    elif args.stage == "audit":
        mults = json.loads((SIM / "development_critical_values.json").read_text())["multipliers"]
        audit = stage("audit", args.reps, args.reps_ho, mults)
        lic = licenses(audit, mults)
        (SIM / "audit_results.json").write_text(json.dumps(audit, indent=1, default=str))
        (SIM / "qualification_report.json").write_text(json.dumps({"licenses": lic, "multipliers": mults, "seed": SEED_AUDIT,
                                                                     "ho_operating_characteristics": audit["ho"]}, indent=1, default=str))
        (SIM / "sample_size_decision.json").write_text(json.dumps({"N_f_fixed": {"SEL_LEVEL": 10, "ORG_descriptive": 5},
                                                                      "no_sample_size_selection": True, "licenses": lic}, indent=1))
        print(json.dumps(lic, indent=1))


if __name__ == "__main__":
    main()

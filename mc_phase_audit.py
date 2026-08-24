"""P0 final phase-map Monte Carlo audit (companion required-experiments
section 3; FROZEN GRID, 2026-08-24).

Every region displayed in the hero phase diagram must be checked against
simulation before the figure is drawn (section 3.1). This harness audits
the analytic AUROC formulas for seven score variants against adaptive
Monte Carlo over the frozen grid below, reusing the VERIFIED conventions:
empirical scorers from pilot0.scores (population-parameter plug-in, the
X1 verification convention), analytic predictors from pilot0.theory.
fDBD's "analytic" value is its declared degeneracy proxy (the analytic
head-CTM prediction), so its error column MEASURES the two-axis
divergence; off-collapse fDBD boundaries must come from simulation.
"CTM_mean" (max cosine to class-mean prototypes) is the pipeline CTM used
in the 280-checkpoint crossing; "CTM_head" is the head-prototype variant.

FROZEN GRID
  Blocks (C, D): (10,128), (100,128) [high class fraction], (100,512),
                 (200,512), (200,768). Full sweeps on (10,128) and
                 (100,512); reduced sweeps (base, theta_w ends, gamma*a
                 near 1, tied profile) on the other three.
  Base config:   s = 24, theta_w = 6 deg, logit-scale target 10,
                 gamma*a = 0.8 with a = 0.9 (unique profile), rho = 1.0,
                 Std(eta) = 0.1, isotropic covariance.
  Sweeps (full blocks, one-axis-at-a-time unless stated):
    plane      gamma*a in {0.25,0.5,0.8,0.9,0.95,1.0,1.05,1.1,1.25,1.6,
               2.2} x s in {10,24,65} (dense around the predicted
               norm-confound crossing at gamma*a = 1);
    theta_w    {0,6,21,40,60} deg x 5 quenched direction draws;
    equinorm   Std(eta) in {0,0.1,0.25} x 3 quenched draws;
    rho        {0.7,1.0,1.4};
    profile    {unique, tied2, diffuse} x gamma*a in {0.5,0.9,1.0,1.1,
               1.6} (the MSR tied-logit stress, section 3.5);
    aniso      {iso, spiked(eig_max/mean ~ 20, rank D//8)};
    logit scale targets {5,10,30}.
  LHS block:   120 Latin-hypercube configs over all axes on (100,512).
  Clustering stress (100,512), section 3.5: pair tie (c=0.9), isolated
               aligned class (c=0.6), cluster k=4 (c=0.7), cluster k=4
               unequal correlations {0.9,0.7,0.5,0.3}, plus 5 random
               k=4 assignments; each x gamma*a in {0.8,1.0,1.2} with the
               OOD direction aligned to a clustered class.
  Adaptive MC: start N = 8192 per population, double until the
               Hanley-McNeil SE of every score's MC AUROC is <= 0.0025
               or N = 131072; the per-config SE is reported.
  Outputs (section 3.4): per config x score: analytic, MC, MC SE,
               absolute error; per config: predicted vs empirical winner,
               empirical top-2 margin; resolvable pairwise sign accuracy;
               per-score summary (median/95th/max error, boundary sign
               accuracy, auto-annotated failure regimes); crossing
               displacement along the gamma*a sweeps for the analytic
               pairs that cross.
  Decision (section 3.6): applied per score in the report: within-
               tolerance regime, ordering-approximation label, or
               exclusion (MSR ties stated explicitly).

Usage (from code/):
    python mc_phase_audit.py [--self-test] [--quick]
Outputs: nc_csf_predictivity/outputs/track1/mc_phase_audit_report.md (+
         .json; per-config records in mc_phase_audit_records.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot0.scores import (
    MahalanobisScorer,
    auroc,
    fdbd,
    head_scores,
    max_cosine,
    normalize_rows,
)
from pilot0.theory import (
    HeadContext,
    NoiseModel,
    hanley_mcneil_se,
    predicted_aurocs,
    predicted_ctm_mean_auroc,
    predicted_maha_auroc,
)

OUT_DIR = Path(__file__).resolve().parent / "nc_csf_predictivity/outputs/track1"
SCORES = ("MSR", "MLS", "Energy", "CTM_head", "CTM_mean", "Maha", "fDBD")
TOL = 0.01
SE_TARGET = 0.0025
N_START, N_MAX = 8192, 131072
GA_GRID = (0.25, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.6, 2.2)
FULL_BLOCKS = ((10, 128), (100, 512))
REDUCED_BLOCKS = ((100, 128), (200, 512), (200, 768))
BASE = {"s": 24.0, "theta_deg": 6.0, "logit_target": 10.0, "ga": 0.8,
        "a": 0.9, "rho": 1.0, "eta_std": 0.1, "aniso": "iso",
        "profile": "unique", "cluster": None, "draw": 0}


# ---------------------------------------------------------------------------
# Model construction (numpy ETF, quenched draws keyed by config seed).
# ---------------------------------------------------------------------------

def etf_directions(c: int, d: int, rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, c)))
    simplex = (np.eye(c) - 1.0 / c) * np.sqrt(c / (c - 1))
    dirs = simplex @ q.T
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def apply_cluster(dirs: np.ndarray, cluster: dict | None,
                  rng) -> np.ndarray:
    if not cluster:
        return dirs
    idx = cluster["classes"]
    common = dirs[idx].mean(0)
    common /= np.linalg.norm(common)
    out = dirs.copy()
    for k, i in enumerate(idx):
        corr = cluster["corr"][k] if isinstance(cluster["corr"], (list,
                                                                  tuple)) \
            else cluster["corr"]
        v = np.sqrt(1 - corr) * dirs[i] + np.sqrt(corr) * common
        out[i] = v / np.linalg.norm(v)
    return out


def rotate_rows(dirs: np.ndarray, theta: float, rng) -> np.ndarray:
    if theta == 0:
        return dirs.copy()
    out = np.empty_like(dirs)
    for i, u in enumerate(dirs):
        v = rng.standard_normal(dirs.shape[1])
        v -= (v @ u) * u
        v /= np.linalg.norm(v)
        out[i] = np.cos(theta) * u + np.sin(theta) * v
    return out


def spiked_cov(d: int, sigma2: float, rng,
               ratio: float = 20.0) -> np.ndarray:
    k = max(d // 8, 1)
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    # eig profile: k spiked directions at kappa, rest at 1; rescale so
    # the mean eigenvalue stays sigma2 and eig_max/mean ~ ratio.
    kappa = ratio * d / (d - k + k * ratio)
    base = d / (d - k + k * kappa)
    cov = sigma2 * base * (np.eye(d) + (kappa - 1.0) * q @ q.T)
    return cov


def build_config_model(c: int, d: int, cfg: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dirs = etf_directions(c, d, rng)
    dirs = apply_cluster(dirs, cfg["cluster"], rng)
    sigma = 1.0
    radius = cfg["s"] * sigma
    eta = np.clip(1.0 + cfg["eta_std"] * rng.standard_normal(c), 0.2, None)
    means = radius * eta[:, None] * dirs
    theta = np.deg2rad(cfg["theta_deg"])
    w_dirs = rotate_rows(dirs, theta, rng)
    # Head scale: HeadContext logit scale ~ g * radius; target g so the
    # measured logit spread matches cfg["logit_target"] (frozen rule:
    # g = logit_target / radius).
    g = cfg["logit_target"] / radius
    w = g * w_dirs
    b = np.zeros(c)
    if cfg["aniso"] == "iso":
        cov_id = sigma ** 2 * np.eye(d)
    else:
        cov_id = spiked_cov(d, sigma ** 2, rng)
    cov_ood = cfg["rho"] ** 2 * cov_id
    # OOD mean by alignment profile.
    ga, a = cfg["ga"], cfg["a"]
    gamma = ga / a
    fresh = rng.standard_normal(d)
    if cfg["cluster"]:
        target = cfg["cluster"]["classes"][0]
    else:
        target = 0
    if cfg["profile"] == "unique":
        u = dirs[target]
    elif cfg["profile"] == "tied2":
        u = dirs[target] + dirs[(target + 1) % c]
        u = u / np.linalg.norm(u)
    else:  # diffuse
        u = dirs.mean(0)
        u = u / np.linalg.norm(u)
    fresh -= (fresh @ u) * u
    fresh /= np.linalg.norm(fresh)
    align = max(min(a, 0.999), -0.999)
    m_dir = align * u + np.sqrt(1 - align ** 2) * fresh
    m_ood = gamma * radius * m_dir
    return {"means": means, "w": w, "b": b, "cov_id": cov_id,
            "cov_ood": cov_ood, "m_ood": m_ood, "sigma": sigma,
            "class_freq": np.full(c, 1.0 / c)}


# ---------------------------------------------------------------------------
# Analytic and Monte Carlo AUROCs.
# ---------------------------------------------------------------------------

def analytic_aurocs(model: dict) -> dict[str, float]:
    ctx = HeadContext.from_head(model["w"], model["b"])
    dim = model["means"].shape[1]
    sigma = model["sigma"]
    if np.allclose(model["cov_id"], sigma ** 2 * np.eye(dim)):
        noise_id = NoiseModel.isotropic(sigma, ctx, dim)
        rho_eff = float(np.sqrt(model["cov_ood"][0, 0] / sigma ** 2))
        noise_ood = NoiseModel.isotropic(rho_eff * sigma, ctx, dim)
    else:
        noise_id = NoiseModel.empirical(model["cov_id"], ctx)
        noise_ood = NoiseModel.empirical(model["cov_ood"], ctx)
    out = predicted_aurocs(model["means"], model["class_freq"], noise_id,
                           model["m_ood"], noise_ood, ctx)
    out["fDBD"] = out["CTM_head"]  # declared degeneracy proxy
    out["CTM_mean"] = predicted_ctm_mean_auroc(
        model["means"], model["class_freq"], model["cov_id"],
        model["m_ood"], model["cov_ood"])
    precision = np.linalg.pinv(model["cov_id"], hermitian=True)
    out["Maha"] = predicted_maha_auroc(
        model["means"], precision, model["cov_id"], model["m_ood"],
        model["cov_ood"])
    return {k: float(v) for k, v in out.items()}


def _sample(model: dict, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    c, d = model["means"].shape
    labels = rng.integers(0, c, n)
    dim_sqrt_id = np.linalg.cholesky(
        model["cov_id"] + 1e-9 * np.eye(d))
    dim_sqrt_ood = np.linalg.cholesky(
        model["cov_ood"] + 1e-9 * np.eye(d))
    h_id = model["means"][labels] + rng.standard_normal((n, d)) @ dim_sqrt_id.T
    h_ood = model["m_ood"] + rng.standard_normal((n, d)) @ dim_sqrt_ood.T
    return h_id, h_ood


def mc_aurocs(model: dict, seed: int) -> tuple[dict, dict, int]:
    rng = np.random.default_rng(seed + 1)
    c, d = model["means"].shape
    precision = np.linalg.pinv(model["cov_id"], hermitian=True)
    means_n = normalize_rows(model["means"])
    w_n = normalize_rows(model["w"])
    train_mean = np.zeros(d)

    class _TrueMaha(MahalanobisScorer):
        def __init__(self):
            self.means = model["means"]
            self.precision = precision
            self._m_quad = np.einsum(
                "cd,dk,ck->c", self.means, self.precision, self.means,
                optimize=True)

    maha = _TrueMaha()
    n = N_START
    acc: dict[str, list[np.ndarray]] = {s: [] for s in SCORES}
    total = 0
    while True:
        h_id, h_ood = _sample(model, n, rng)
        for pop, h in (("id", h_id), ("ood", h_ood)):
            g = h @ model["w"].T + model["b"]
            hs = head_scores(g)
            vals = {"MSR": hs["MSR"], "MLS": hs["MLS"],
                    "Energy": hs["Energy"],
                    "CTM_head": max_cosine(normalize_rows(h), w_n),
                    "CTM_mean": max_cosine(normalize_rows(h), means_n),
                    "Maha": maha(h),
                    "fDBD": fdbd(h, g, model["w"], train_mean)}
            for s in SCORES:
                acc[s].append((pop, vals[s]))
        total += n
        aucs, ses = {}, {}
        for s in SCORES:
            s_id = np.concatenate([v for p, v in acc[s] if p == "id"])
            s_ood = np.concatenate([v for p, v in acc[s] if p == "ood"])
            aucs[s] = auroc(s_id, s_ood)
            ses[s] = hanley_mcneil_se(aucs[s], len(s_id), len(s_ood))
        if max(ses.values()) <= SE_TARGET or total >= N_MAX:
            return aucs, ses, total
        n = total  # double


# ---------------------------------------------------------------------------
# Frozen grid enumeration.
# ---------------------------------------------------------------------------

def grid_configs(quick: bool = False) -> list[dict]:
    cfgs: list[dict] = []

    def add(c, d, family, **over):
        cfg = dict(BASE, **over)
        cfg.update({"C": c, "D": d, "family": family})
        cfgs.append(cfg)

    for c, d in FULL_BLOCKS:
        for s in (10.0, 24.0, 65.0):
            for ga in GA_GRID:
                add(c, d, "plane", s=s, ga=ga)
        for theta in (0.0, 6.0, 21.0, 40.0, 60.0):
            for draw in range(5):
                add(c, d, "theta", theta_deg=theta, draw=draw)
        for eta in (0.0, 0.1, 0.25):
            for draw in range(3):
                add(c, d, "equinorm", eta_std=eta, draw=draw)
        for rho in (0.7, 1.0, 1.4):
            add(c, d, "rho", rho=rho)
        for profile in ("unique", "tied2", "diffuse"):
            for ga in (0.5, 0.9, 1.0, 1.1, 1.6):
                add(c, d, "profile", profile=profile, ga=ga)
        for aniso in ("iso", "spiked"):
            add(c, d, "aniso", aniso=aniso)
        for lt in (5.0, 10.0, 30.0):
            add(c, d, "logit_scale", logit_target=lt)
    for c, d in REDUCED_BLOCKS:
        add(c, d, "reduced")
        for theta in (0.0, 60.0):
            add(c, d, "reduced", theta_deg=theta)
        for ga in (0.9, 1.0, 1.1):
            add(c, d, "reduced", ga=ga)
        add(c, d, "reduced", profile="tied2", ga=1.0)

    rng = np.random.default_rng(2024)
    for i in range(120):
        add(100, 512, "lhs",
            s=float(rng.uniform(8, 70)),
            theta_deg=float(rng.uniform(0, 60)),
            logit_target=float(rng.uniform(5, 30)),
            ga=float(rng.uniform(0.2, 2.2)),
            a=float(rng.uniform(0.4, 0.99)),
            rho=float(rng.uniform(0.6, 1.6)),
            eta_std=float(rng.uniform(0, 0.3)),
            aniso=("iso" if rng.random() < 0.5 else "spiked"),
            profile=("unique", "tied2", "diffuse")[int(rng.integers(3))],
            draw=i)

    clusters = [("pair_tie", {"classes": [0, 1], "corr": 0.9}),
                ("isolated", {"classes": [0, 1], "corr": 0.6}),
                ("cluster4", {"classes": [0, 1, 2, 3], "corr": 0.7}),
                ("cluster4_unequal",
                 {"classes": [0, 1, 2, 3],
                  "corr": (0.9, 0.7, 0.5, 0.3)})]
    rng_c = np.random.default_rng(7)
    for k in range(5):
        classes = sorted(int(x) for x in
                         rng_c.choice(100, 4, replace=False))
        clusters.append((f"random_assign_{k}",
                         {"classes": classes, "corr": 0.7}))
    for name, cluster in clusters:
        for ga in (0.8, 1.0, 1.2):
            add(100, 512, f"cluster:{name}", cluster=cluster, ga=ga)

    if quick:
        cfgs = [c for i, c in enumerate(cfgs) if i % 12 == 0]
    return cfgs


# ---------------------------------------------------------------------------
# Audit driver.
# ---------------------------------------------------------------------------

def audit_config(cfg: dict, idx: int) -> dict:
    seed = 10_000 + idx
    model = build_config_model(cfg["C"], cfg["D"], cfg, seed)
    ana = analytic_aurocs(model)
    mc, se, n_used = mc_aurocs(model, seed)
    errs = {s: abs(ana[s] - mc[s]) for s in SCORES}
    pred_winner = max(SCORES, key=lambda s: ana[s])
    emp_winner = max(SCORES, key=lambda s: mc[s])
    emp_sorted = sorted((mc[s] for s in SCORES), reverse=True)
    pair_flags = {}
    for i, si in enumerate(SCORES):
        for sj in SCORES[i + 1:]:
            resolvable = abs(mc[si] - mc[sj]) > 2 * (se[si] + se[sj])
            if resolvable:
                pair_flags[f"{si}|{sj}"] = bool(
                    np.sign(ana[si] - ana[sj]) == np.sign(mc[si] - mc[sj]))
    boundary_dist = min(abs(ana[si] - ana[sj])
                        for i, si in enumerate(SCORES)
                        for sj in SCORES[i + 1:])
    rec = {k: cfg[k] for k in ("C", "D", "family", "s", "theta_deg",
                               "logit_target", "ga", "a", "rho",
                               "eta_std", "aniso", "profile", "draw")}
    rec["cluster"] = (cfg["cluster"] and str(cfg["cluster"])) or None
    rec.update({"analytic": ana, "mc": mc, "mc_se": se, "abs_err": errs,
                "n_mc": n_used, "pred_winner": pred_winner,
                "emp_winner": emp_winner,
                "winner_match": pred_winner == emp_winner,
                "emp_top2_margin": float(emp_sorted[0] - emp_sorted[1]),
                "boundary_distance": float(boundary_dist),
                "pair_sign_ok": pair_flags})
    return rec


def crossing_displacement(records: list[dict]) -> dict:
    """Along each (block, s) gamma*a sweep: analytic vs MC crossing of
    each score pair that crosses, in gamma*a units."""
    out = {}
    plane = [r for r in records if r["family"] == "plane"]
    for c, d in FULL_BLOCKS:
        for s in (10.0, 24.0, 65.0):
            rows = sorted((r for r in plane if r["C"] == c and r["D"] == d
                           and r["s"] == s), key=lambda r: r["ga"])
            if len(rows) < 4:
                continue
            gas = np.array([r["ga"] for r in rows])
            for i, si in enumerate(SCORES):
                for sj in SCORES[i + 1:]:
                    da = np.array([r["analytic"][si] - r["analytic"][sj]
                                   for r in rows])
                    dm = np.array([r["mc"][si] - r["mc"][sj]
                                   for r in rows])
                    xa = _first_zero(gas, da)
                    xm = _first_zero(gas, dm)
                    if xa is not None and xm is not None:
                        out[f"C{c}_s{s:g}_{si}|{sj}"] = {
                            "analytic": round(xa, 3), "mc": round(xm, 3),
                            "displacement": round(xm - xa, 3)}
    return out


def _first_zero(x: np.ndarray, y: np.ndarray):
    s = np.sign(y)
    for i in range(len(s) - 1):
        if s[i] != 0 and s[i + 1] != 0 and s[i] != s[i + 1]:
            f = y[i] / (y[i] - y[i + 1])
            return float(x[i] + f * (x[i + 1] - x[i]))
    return None


def failure_regimes(records: list[dict], score: str) -> str:
    bad = [r for r in records if r["abs_err"][score] > TOL]
    if not bad:
        return "none observed"
    tags = []
    if sum(r["profile"] in ("tied2", "diffuse") for r in bad) > len(bad) / 3:
        tags.append("tied/diffuse alignment")
    if sum(r["theta_deg"] >= 21 for r in bad) > len(bad) / 3:
        tags.append("theta_w >= 21 deg")
    if sum(r["aniso"] == "spiked" for r in bad) > len(bad) / 3:
        tags.append("spiked covariance")
    if sum(bool(r["cluster"]) for r in bad) > len(bad) / 3:
        tags.append("clustered classes")
    if sum(r["ga"] >= 1.4 for r in bad) > len(bad) / 3:
        tags.append("large gamma*a")
    if sum(r["C"] >= 100 and r["C"] - 1 >= 0.5 * r["D"]
           for r in bad) > len(bad) / 3:
        tags.append("high class fraction (C-1 >= D/2)")
    frac = len(bad) / len(records)
    return f"{', '.join(tags) or 'scattered'} ({frac:.0%} of configs)"


def summarize(records: list[dict]) -> dict:
    summary = {}
    for s in SCORES:
        errs = np.array([r["abs_err"][s] for r in records])
        pair_ok = [ok for r in records
                   for key, ok in r["pair_sign_ok"].items()
                   if s in key.split("|")]
        summary[s] = {
            "median_err": float(np.median(errs)),
            "p95_err": float(np.quantile(errs, 0.95)),
            "max_err": float(errs.max()),
            "within_tol_frac": float((errs <= TOL).mean()),
            "boundary_sign_accuracy": (float(np.mean(pair_ok))
                                       if pair_ok else None),
            "known_failure_regime": failure_regimes(records, s),
        }
    return summary


def decide(summary: dict) -> dict:
    decisions = {}
    for s, rec in summary.items():
        sign_ok = (rec["boundary_sign_accuracy"] or 0) >= 0.95
        if rec["p95_err"] <= TOL:
            verdict = "calibrated formula (within tolerance)"
        elif sign_ok:
            verdict = ("ordering approximation (winner signs reliable, "
                       "levels exceed tolerance)")
        else:
            verdict = ("boundary from simulation required (sign accuracy "
                       "below 0.95 in audited regimes)")
        if s == "MSR" and rec["p95_err"] > TOL:
            verdict += ("; EXCLUDE tied-logit regimes from the +-0.01 "
                        "claim (state in abstract and formula table)")
        if s == "fDBD":
            verdict += ("; proxy = degeneracy with head-CTM, valid near "
                        "collapse only (two-axis divergence is the "
                        "paper's own claim)")
        decisions[s] = verdict
    return decisions


def render(summary: dict, decisions: dict, crossings: dict,
           n_cfg: int, winner_acc: float) -> str:
    lines = ["# MC phase-map audit (P0; frozen grid in mc_phase_audit.py)",
             ""]
    lines.append(f"Configs audited: {n_cfg}; adaptive MC SE target "
                 f"{SE_TARGET}; tolerance {TOL}; overall predicted-winner "
                 f"accuracy {winner_acc:.3f}.")
    lines.append("")
    lines.append("| score | median err | 95th err | max err | within tol "
                 "| boundary sign acc | known failure regime |")
    lines.append("|---|---|---|---|---|---|---|")
    for s, rec in summary.items():
        acc = rec["boundary_sign_accuracy"]
        lines.append(
            f"| {s} | {rec['median_err']:.4f} | {rec['p95_err']:.4f} "
            f"| {rec['max_err']:.4f} | {rec['within_tol_frac']:.2f} "
            f"| {acc if acc is None else f'{acc:.3f}'} "
            f"| {rec['known_failure_regime']} |")
    lines.append("")
    lines.append("## Crossing displacement along gamma*a sweeps "
                 "(analytic vs MC, gamma*a units)")
    lines.append("")
    if crossings:
        lines.append("| sweep/pair | analytic | MC | displacement |")
        lines.append("|---|---|---|---|")
        for key, r in sorted(crossings.items()):
            lines.append(f"| {key} | {r['analytic']} | {r['mc']} "
                         f"| {r['displacement']} |")
    else:
        lines.append("(no crossing pairs detected)")
    lines.append("")
    lines.append("## Decision per score (section 3.6)")
    lines.append("")
    for s, verdict in decisions.items():
        lines.append(f"- **{s}**: {verdict}")
    lines.append("")
    return "\n".join(lines)


def self_test() -> None:
    cfg = dict(BASE, C=10, D=64, family="selftest", ga=0.6)
    model = build_config_model(10, 64, cfg, 1)
    ana = analytic_aurocs(model)
    global N_START, N_MAX
    mc, se, n = mc_aurocs(model, 1)
    for s in ("MLS", "Energy", "CTM_head", "CTM_mean", "Maha"):
        assert abs(ana[s] - mc[s]) < 0.03, (s, ana[s], mc[s])
    assert all(v <= SE_TARGET * 1.5 or n >= N_MAX for v in se.values())
    rec = audit_config(cfg, 0)
    assert set(rec["abs_err"]) == set(SCORES)
    print(f"self-test PASS (n_mc={n}): analytic ~ MC for the verified "
          f"scores at a benign config; record schema complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MC phase-map audit")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="1/12 subsample of the frozen grid")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    cfgs = grid_configs(quick=args.quick)
    print(f"frozen grid: {len(cfgs)} configs")
    records = []
    for i, cfg in enumerate(cfgs):
        records.append(audit_config(cfg, i))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cfgs)} done")
    summary = summarize(records)
    decisions = decide(summary)
    crossings = crossing_displacement(records)
    winner_acc = float(np.mean([r["winner_match"] for r in records]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = render(summary, decisions, crossings, len(records),
                    winner_acc)
    (OUT_DIR / "mc_phase_audit_report.md").write_text(report)
    (OUT_DIR / "mc_phase_audit_report.json").write_text(json.dumps(
        {"summary": summary, "decisions": decisions,
         "crossings": crossings, "winner_accuracy": winner_acc},
        indent=1, default=float))
    (OUT_DIR / "mc_phase_audit_records.json").write_text(
        json.dumps(records, indent=0, default=float))
    print(report)


if __name__ == "__main__":
    main()

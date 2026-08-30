"""Phase-2 factorial diagnostic P00/P01/P10/P11 (saturation plan
2026-08-28, sections 9 and 12 Phase 2). Local; development data only.

EVIDENCE CLASS: post hoc (plan section 9.2, nested-development design).
Everything runs on the Phase-1 prototype subset, which lies entirely in
the frozen development half of the repair split (plus the five BREEDS
externals, reported separately); the validation half stays untouched.
Nothing here upgrades any paper claim.

FROZEN SPECIFICATION (declared before any factorial outcome was
inspected; exactly ONE specification per arm, no search over component
or shrinkage rules - the rules are the Phase-1 freeze):
- Scores: Energy and mean-CTM (the claim-bearing pair). All arms are
  evaluated in stable log-tail space (l = log error probability); the
  margin is M = l_E - l_C, sign(M) = predicted AUGRC-gap sign; direct
  AUROC = 1 - exp(l).
- P00d: the frozen dictionary arm of record = tail_space_audit values
  (continuity reference). P00r: the same dictionary formula recomputed
  from the Phase-1 record's own papyan/geometry panel and the frozen
  per-set (gamma, a, rho); the paired base for gate R2 (its deviation
  from P00d is reported).
- P10r (mixture-aware, isotropic): the identical dictionary formula per
  measured component k with gamma_k = |m_k|/R (R = mean class-mean
  radius), a_k = max_c mu_hat_c' m_k / |m_k|, and the shared
  rho_res = sqrt(trS_res / trS_id); mixture in tail space:
  l = logsumexp_k(log omega_k + l_k). The "other" component enters with
  its own mean.
- P01m (global mean, empirical anisotropic): measured-moment machinery
  on the stored sufficient statistics. ID per class y: Energy mean =
  LSE(G_id[y] + b_eff) + 0.5 tr(H Sigma_G) (second-order log-sum-exp
  correction, Sigma_G = WSW_id, H = diag(p) - pp'), variance =
  p' WSW_id p; CTM via the FROZEN empirical-arm form (theory.ctm_stats
  arithmetic in projection space: alignments A_id[:, y], norm radii[y]^2,
  directional variances dir_id, trace trS_id). OOD side: the global mean
  projections with the global residual covariance projections (same
  forms, same second-order Energy correction). CTM second-order mean
  corrections are NOT applied anywhere (component cross terms are not
  stored; declared).
- P11m (mixture-aware, empirical): P01m's ID side; OOD components with
  the shared pooled residual covariance projections;
  l = logsumexp_{y,k}(log pi_y + log omega_k + log_ndtr(-z_yk)).
- Endpoints (plan section 9.3): primary = balanced winner-sign accuracy
  on the subset's empirically material cells (|gap| >= 10 milli,
  frozen); co-primary = Spearman(|M|, |gap|); secondary = ordinary sign
  accuracy, direct-AUROC level MAE (BREEDS only, where empirical AUROCs
  exist), z-level sign stability (10 draws of relative 1e-6
  perturbations, seed 1049, measured arms; dictionary-arm stability is
  the audit-10 R5 result).
- Comparators (section 9.4): severity-only isotonic and train-fold mean,
  fitted by grouped 5-fold (seed 2027) WITHIN the subset via the frozen
  run_folds machinery; the frozen dictionary arm P00d.
- Uncertainty: paired checkpoint-cluster bootstrap, B = 2000, seed 1031,
  for each arm-minus-P00r and arm-minus-severity balanced-accuracy
  difference. Four sources reported individually.
- BREEDS (n = 5 checkpoints, 1 shift): descriptive table only.

Gates R1-R4 are the plan's section 10; verdict lines are printed from
the numbers, R1 referencing the audit-10 tail audit.

Usage (from code/): python repair_factorial.py
Inputs: pilot0/repair_phase1_stats/, repair_phase1_manifest.json,
        tail_space_audit_cells.parquet, harmonized parquet outcomes,
        pilot0/stage2_expansion_coords/ (BREEDS outcomes)
Output: nc_csf_predictivity/outputs/track1/repair_factorial_report.md/.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, logsumexp
from scipy.stats import spearmanr

from crossing_robustness_audit import OUT_DIR
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, run_folds)
from stage2_closure import build_cells_with_severity
from tail_space_audit import frozen_cfg, tail_pair

STATS_DIR = Path("pilot0/repair_phase1_stats")
MANIFEST = Path("nc_csf_predictivity/outputs/track1/"
                "repair_phase1_manifest.json")
TAIL_CACHE = OUT_DIR / "tail_space_audit_cells.parquet"
ARMS = ("P00r", "P10r", "P01m", "P11m")
B_BOOT, BOOT_SEED, PERT_SEED, K_PERT = 2000, 1031, 1049, 10


def rho(x, y) -> float:
    return float(spearmanr(x, y, nan_policy="omit").statistic)


def ctm_stats_proj(a_vec, n2, trace, dim, dir_vec):
    """theory.ctm_stats arithmetic on stored projections."""
    omega = n2 + trace
    mean_cos = a_vec / np.sqrt(omega)
    k = int(np.argmax(mean_cos))
    p0 = float(a_vec[k])
    sigma_dir2 = float(dir_vec[k])
    sigma_avg2 = trace / dim
    q0 = n2 - p0**2 + trace
    om = p0**2 + q0
    v = (q0**2 * sigma_dir2 / om**3
         + p0**2 * (4.0 * (n2 - p0**2) * sigma_avg2
                    + 2.0 * sigma_avg2**2 * dim) / (4.0 * om**3))
    return float(mean_cos[k]), float(v)


def energy_stats(g, wsw):
    """(mean with 2nd-order LSE correction, variance) for logits mean g."""
    gm = g - g.max()
    p = np.exp(gm)
    p /= p.sum()
    lse = float(np.log(np.exp(gm).sum()) + g.max())
    var = float(p @ wsw @ p)
    corr = 0.5 * (float(p @ np.diag(wsw)) - var)
    return lse + corr, var


class Record:
    """One Phase-1 checkpoint's stats, array-hydrated."""

    def __init__(self, slug: str):
        self.r = json.loads((STATS_DIR / f"{slug}.json").read_text())
        self.z = np.load(STATS_DIR / f"{slug}.npz")
        self.C, self.D = self.r["n_classes"], self.r["dim"]
        s = self.r["id_scalars"]
        self.trS_id = s["id__trS_id"]
        self.radii = self.z["id__radii"]
        self.R = float(self.radii.mean())
        self.freq = self.z["id__class_freq"]
        self.b_eff = self.z["id__b_eff"]
        self.G_id = self.z["id__G_id"]
        self.A_id = self.z["id__A_id"]
        self.WSW_id = self.z["id__WSW_id"]
        self.dir_id = self.z["id__dir_id"]
        pap, geo = self.r["papyan"], self.r["geometry"]
        self.s_dict = (self.C - 1) / np.sqrt(
            self.C * max(pap["var_collapse"], 1e-9))
        self.theta = float(np.degrees(np.arccos(
            np.clip(1.0 - pap["self_duality"] / 2.0, -1.0, 1.0))))
        self.logit = geo["logit_scale"]
        self.eta = geo["class_mean_radius_cv"]
        # ID score moments (shared by P01m/P11m)
        self.id_E = [energy_stats(self.G_id[y] + self.b_eff, self.WSW_id)
                     for y in range(self.C)]
        self.id_C = [ctm_stats_proj(self.A_id[:, y], self.radii[y]**2,
                                    self.trS_id, self.D, self.dir_id)
                     for y in range(self.C)]

    def set_stats(self, name: str):
        st = self.r["sets"][name]["stats"]
        key = name.replace(" ", "_")

        def arr(suffix):
            return self.z[f"set__{key}__{suffix}"]
        comps = []
        i = 0
        while f"components__{i}__n" in st:
            comps.append({"weight": st[f"components__{i}__weight"],
                          "n2": st[f"components__{i}__n2"],
                          "g": arr(f"components__{i}__g"),
                          "a": arr(f"components__{i}__a")})
            i += 1
        return {
            "coords": self.r["sets"][name]["coords"],
            "glob": {"g": arr("global__g"), "a": arr("global__a"),
                     "n2": st["global__n2"], "WSW": arr("global__WSW"),
                     "dir": arr("global__dir"),
                     "trS": st["global__trS"]},
            "res": {"WSW": arr("resid_shared__WSW"),
                    "dir": arr("resid_shared__dir"),
                    "trS": st["resid_shared__trS"]},
            "comps": comps,
        }

    def dict_pair(self, gamma, a, rho_v):
        cfg = frozen_cfg(self.s_dict, self.theta, self.logit, self.eta,
                         gamma, a, rho_v)
        _, _, l_e, l_c = tail_pair(self.C, self.D, cfg, fast_ctm=True)
        return l_e, l_c

    def arm_P00r(self, ss):
        c = ss["coords"]
        return self.dict_pair(c["gamma"], c["a"], c["rho"])

    def arm_P10r(self, ss):
        rho_res = float(np.sqrt(ss["res"]["trS"] / self.trS_id))
        ls_e, ls_c, lw = [], [], []
        for cp in ss["comps"]:
            nrm = np.sqrt(cp["n2"])
            l_e, l_c = self.dict_pair(nrm / self.R,
                                      float(cp["a"].max()) / nrm, rho_res)
            ls_e.append(l_e)
            ls_c.append(l_c)
            lw.append(np.log(cp["weight"]))
        lw = np.array(lw)
        return (float(logsumexp(lw + np.array(ls_e))),
                float(logsumexp(lw + np.array(ls_c))))

    def _measured(self, ood_pops):
        """(l_E, l_C, zE, zC) for OOD populations [(logw, g, a, n2,
        WSW, dir, trS), ...] against the ID class mixture."""
        z_e = np.empty((self.C, len(ood_pops)))
        z_c = np.empty_like(z_e)
        lw = np.empty(len(ood_pops))
        for j, (logw, g, a, n2, wsw, dirv, tr) in enumerate(ood_pops):
            lw[j] = logw
            me_o, ve_o = energy_stats(g + self.b_eff, wsw)
            mc_o, vc_o = ctm_stats_proj(a, n2, tr, self.D, dirv)
            for y in range(self.C):
                me_y, ve_y = self.id_E[y]
                mc_y, vc_y = self.id_C[y]
                z_e[y, j] = (me_y - me_o) / np.sqrt(ve_y + ve_o)
                z_c[y, j] = (mc_y - mc_o) / np.sqrt(vc_y + vc_o)
        lpi = np.log(self.freq)[:, None] + lw[None, :]
        l_e = float(logsumexp(lpi + log_ndtr(-z_e)))
        l_c = float(logsumexp(lpi + log_ndtr(-z_c)))
        return l_e, l_c, z_e, z_c, lpi

    def arm_P01m(self, ss):
        g = ss["glob"]
        pops = [(0.0, g["g"], g["a"], g["n2"], g["WSW"], g["dir"],
                 g["trS"])]
        return self._measured(pops)

    def arm_P11m(self, ss):
        res = ss["res"]
        pops = [(np.log(cp["weight"]), cp["g"], cp["a"], cp["n2"],
                 res["WSW"], res["dir"], res["trS"])
                for cp in ss["comps"]]
        return self._measured(pops)


def stability(z_e, z_c, lpi, rng) -> float:
    base = np.sign(logsumexp(lpi + log_ndtr(-z_e))
                   - logsumexp(lpi + log_ndtr(-z_c)))
    same = 0
    for _ in range(K_PERT):
        pe = z_e * (1 + 1e-6 * rng.uniform(-1, 1, z_e.shape))
        pc = z_c * (1 + 1e-6 * rng.uniform(-1, 1, z_c.shape))
        m = (logsumexp(lpi + log_ndtr(-pe))
             - logsumexp(lpi + log_ndtr(-pc)))
        same += np.sign(m) == base
    return same / K_PERT


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    slug_cell = {e["model_path"].replace("/", "__"): e["cell"]
                 for e in man["pool"]}
    cells = build_cells_with_severity()
    sub = cells[cells.cell.isin(set(slug_cell.values()))].reset_index(
        drop=True)
    tail = pd.read_parquet(TAIL_CACHE)[["cell", "eval_dataset", "m_tail"]]
    sub = sub.merge(tail, on=["cell", "eval_dataset"], how="left")
    rngp = np.random.default_rng(PERT_SEED)

    margins = {a: pd.Series(np.nan, index=sub.index) for a in ARMS}
    margins["P00d"] = sub.m_tail
    stab = {a: pd.Series(np.nan, index=sub.index)
            for a in ("P01m", "P11m")}
    for slug, cell in sorted(slug_cell.items()):
        rec = Record(slug)
        rows = sub[sub.cell == cell]
        print(f"[factorial] {slug} ({len(rows)} sets)", flush=True)
        for idx, row in rows.iterrows():
            if row.eval_dataset not in rec.r["sets"]:
                continue
            ss = rec.set_stats(row.eval_dataset)
            le, lc = rec.arm_P00r(ss)
            margins["P00r"][idx] = le - lc
            le, lc = rec.arm_P10r(ss)
            margins["P10r"][idx] = le - lc
            le, lc, ze, zc, lpi = rec.arm_P01m(ss)
            margins["P01m"][idx] = le - lc
            stab["P01m"][idx] = stability(ze, zc, lpi, rngp)
            le, lc, ze, zc, lpi = rec.arm_P11m(ss)
            margins["P11m"][idx] = le - lc
            stab["P11m"][idx] = stability(ze, zc, lpi, rngp)

    folded = run_folds(sub, pd.Series(np.sign(margins["P11m"]),
                                      index=sub.index), "ckpt5",
                       np.random.default_rng(FOLD_SEED))
    for a in list(margins):
        folded[f"m_{a}"] = margins[a].values
        folded[a] = np.sign(margins[a].values)
    mat = folded[np.abs(folded.gap) >= MATERIALITY].copy()
    obs = mat.gap.values
    arms_all = ["P00d"] + list(ARMS)
    out = {"n_cells": int(len(folded)), "n_material": int(len(mat)),
           "P00r_vs_P00d": {
               "sign_agreement": round(float(
                   (np.sign(folded.m_P00r) == np.sign(folded.m_P00d))
                   .mean()), 4),
               "max_abs_margin_dev": round(float(
                   (folded.m_P00r - folded.m_P00d).abs().max()), 4)},
           "arms": {}, "comparators": {}}
    for a in arms_all:
        out["arms"][a] = {
            "sign_acc": round(accuracy(mat[a].values, obs), 4),
            "balanced_acc": round(balanced_accuracy(mat[a].values, obs),
                                  4),
            "spearman_absM_absgap_material": round(
                rho(mat[f"m_{a}"].abs(), np.abs(obs)), 4),
            "spearman_absM_absgap_all": round(
                rho(folded[f"m_{a}"].abs(), folded.gap.abs()), 4)}
    for c in ("severity", "mean"):
        out["comparators"][c] = {
            "sign_acc": round(accuracy(mat[c].values, obs), 4),
            "balanced_acc": round(
                balanced_accuracy(np.sign(mat[c].values), obs), 4)}
    out["stability_mean"] = {a: round(float(np.nanmean(stab[a])), 4)
                             for a in stab}
    out["per_source_balanced_P11m"] = {
        s: round(balanced_accuracy(g.P11m.values, g.gap.values), 3)
        for s, g in mat.groupby("source")}

    # paired checkpoint bootstrap for balanced-accuracy differences
    rng = np.random.default_rng(BOOT_SEED)
    ckpts = np.array(sorted(mat.cell.unique()))
    groups = {c: g for c, g in mat.groupby("cell")}
    pairs = ([(a, "P00r") for a in ("P10r", "P01m", "P11m")]
             + [("P11m", "P10r"), ("P11m", "P01m")]
             + [(a, "severity_sign") for a in ("P11m", "P01m")])
    mat["severity_sign"] = np.sign(mat.severity.values)
    boots = {p: np.empty(B_BOOT) for p in pairs}
    for i in range(B_BOOT):
        fr = pd.concat([groups[c] for c in
                        rng.choice(ckpts, len(ckpts), replace=True)])
        fr_sev = np.sign(fr.severity.values)
        o = fr.gap.values
        for a, b in pairs:
            vb = fr_sev if b == "severity_sign" else fr[b].values
            boots[(a, b)][i] = (balanced_accuracy(fr[a].values, o)
                                - balanced_accuracy(vb, o))
    out["paired_diffs_balanced"] = {}
    for (a, b), d in boots.items():
        vb = mat["severity_sign"] if b == "severity_sign" else mat[b]
        point = (balanced_accuracy(mat[a].values, obs)
                 - balanced_accuracy(vb.values, obs))
        out["paired_diffs_balanced"][f"{a}_minus_{b}"] = {
            "point": round(float(point), 4),
            "ci95": [round(float(np.quantile(d, q)), 4)
                     for q in (0.025, 0.975)]}

    # BREEDS descriptive block
    breeds = []
    for e in man["breeds"]:
        slug = e["model_path"].replace("/", "__")
        rec = Record(slug)
        s2 = json.loads(Path("pilot0/stage2_expansion_coords/"
                             f"{slug}.json").read_text())
        ood_key = next(iter(s2["ood"]))
        eo = s2["ood"][ood_key]
        ss = rec.set_stats(ood_key)
        row = {"slug": slug, "gap_balanced": eo["gap_balanced"],
               "auroc_E_emp": eo["auroc_id_vs_ood_Energy"],
               "auroc_C_emp": eo["auroc_id_vs_ood_CTM"]}
        for a in ARMS:
            res = getattr(rec, f"arm_{a}")(ss)
            le, lc = res[0], res[1]
            row[f"m_{a}"] = round(le - lc, 4)
            row[f"aurocs_{a}"] = [round(1 - float(np.exp(le)), 4),
                                  round(1 - float(np.exp(lc)), 4)]
        breeds.append(row)
    lev = {a: round(float(np.mean([
        abs(r[f"aurocs_{a}"][0] - r["auroc_E_emp"])
        + abs(r[f"aurocs_{a}"][1] - r["auroc_C_emp"])
        for r in breeds])) / 2, 4) for a in ARMS}
    out["breeds"] = {"rows": breeds, "level_mae": lev,
                     "sign_match": {a: int(sum(
                         np.sign(r[f"m_{a}"]) == np.sign(r["gap_balanced"])
                         for r in breeds)) for a in ARMS}}

    (OUT_DIR / "repair_factorial_report.json").write_text(
        json.dumps(out, indent=1, default=float))
    body = {k: v for k, v in out.items() if k != "breeds"}
    L = ["# Phase-2 factorial diagnostic (P00/P01/P10/P11, development "
         "subset)", "",
         "Post hoc; development half only; frozen spec in the header of "
         "repair_factorial.py; validation half untouched.", "", "```",
         json.dumps(body, indent=1), "```", "", "## BREEDS (n=5, "
         "descriptive)", "```",
         json.dumps(out["breeds"], indent=1), "```", ""]
    (OUT_DIR / "repair_factorial_report.md").write_text("\n".join(L))
    print("\n".join(L[:8]))
    print(json.dumps(body, indent=1)[:2000])


if __name__ == "__main__":
    main()

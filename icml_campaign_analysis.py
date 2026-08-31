"""ICML campaign analysis of record (frozen protocol section 8.3-8.5 +
section 9 amendment). Endpoints E1-E4 and gates GR-1..GR-4 with
code-asserted denominators, plus the severity-amendment gates A
(provenance) and B (KID stability), which run BEFORE any roster outcome
file is opened and abort the run on failure.

GR-5 COMPLIANCE: this script is committed, with its synthetic self-test
passing, BEFORE any HPC output is opened. It is the FIRST reader of
every roster outcome.

REGISTERED SPECIFICATION (verbatim bindings):
- Frozen predictor: the corrected-dictionary formula exactly as in
  corrected_dictionary_audit.py at 5732b9e: s = (C-1)/sqrt(C*NC1) and
  theta from each record's papyan panel, logit scale and equinorm
  spread from measured geometry, frozen clamps via frozen_cfg, per-set
  (gamma, a, rho) from the frozen estimate_ood_coords, stable tail
  evaluation (tail_pair fast_ctm), prediction sign(l_E - l_C). NO
  refit, NO recalibration, NO abstention. Secondary mixture arm P10
  exactly as repair_factorial.arm_P10r (per-component dictionary,
  N_MIN merge upstream, shared residual rho; tail-space logsumexp).
- E1 (primary; rosters A, B): material-cell winner-sign accuracy vs
  the fold-fitted severity-only isotonic baseline; folds grouped by
  checkpoint, 5-fold, seed 2027; roster B additionally LOSO. Paired
  checkpoint-cluster bootstrap B = 2000, seed 1201; balanced accuracy
  co-reported. Severity axis d^K primary; every verdict re-read under
  d^F (never averaged, never cross-compared).
- E2 (roster B): within-source collapse-tertile ordering over the
  new-shift severity axis, corrected panel (pool papyan var_collapse),
  frozen thirds rule, pava with B = 2000 bands (seed 1211), frozen
  ordering_retained; re-read under d^F.
- E3 (rosters A, C): per-checkpoint MAE of predicted AUROC
  (1 - exp(l), Energy and CTM) against empirical ID-vs-OOD AUROC; P10
  against P00-corrected, paired per checkpoint, bootstrap seed 1221.
- E4 (all rosters): Spearman(|l_E - l_C|, |observed gap|); no gate.
- Materiality: roster A prevalence-balanced |gap_balanced| >= 0.01;
  rosters B/C the pool convention |gap_raw| >= 10 milli AUGRC.
- Gates: GR-1 per roster (CI excludes zero in favor AND all
  leave-one-source influences positive); GR-2 (>= 2 of {A, B-ckpt,
  B-LOSO} pass, none reverses); GR-3 per source (ordering retained on
  the full new-shift suite AND >= n-1 of the n single-set deletions,
  n asserted from the extracted sets); GR-4 (E3 favors P10, CI
  excluding zero, on BOTH A and C). All denominators are computed from
  the actual rosters and asserted (the G11.B / P-4 rule).
- Severity gates (operationalization DECLARED HERE, pre-outcome; the
  adopted document left them qualitative): Gate A = provenance JSON
  complete (model, explicit tag laion2b_s34b_b79k, versions, kernel,
  seeds, cache hashes, table sha256) AND per-source Spearman >= 0.9
  between regenerated and frozen registered values for BOTH kid and
  fd. Gate B = per (roster-context, source): every independent-seed
  KID replicate ranking of that source's eval sets has Kendall tau
  >= 0.8 with the seed-0 ranking and the mean tau is >= 0.9.

Usage (from code/):
    python icml_campaign_analysis.py --self-test
    python icml_campaign_analysis.py [--b 2000]
Inputs: pilot0/icml_roster_a_coords/, pilot0/icml_roster_a_in200_coords/,
        pilot0/icml_roster_b_coords/, pilot0/icml_roster_c_outcomes/,
        pilot0/roster_c_stats/, pilot0/pool_coords/,
        pilot0/clip_severity_v2.csv (+ _provenance.json),
        documentation/x6_spectral_scripts/clip_severity.csv
Output: nc_csf_predictivity/outputs/track1/icml_campaign_report.json/.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import kendalltau, spearmanr

from crossing_robustness_audit import (OUT_DIR, SEVERITY_CSV, analyze_curve,
                                       ordering_retained, tertiles)
from heldout_theory_validation import (FOLD_SEED, MATERIALITY, accuracy,
                                       balanced_accuracy, severity_only)
from tail_space_audit import frozen_cfg, tail_pair

CODE = Path(__file__).resolve().parent
DIR_A_CIFAR = CODE / "pilot0/icml_roster_a_coords"
DIR_A_IN200 = CODE / "pilot0/icml_roster_a_in200_coords"
DIR_B = CODE / "pilot0/icml_roster_b_coords"
DIR_C_OUT = CODE / "pilot0/icml_roster_c_outcomes"
DIR_C_STATS = CODE / "pilot0/roster_c_stats"
DIR_POOL = CODE / "pilot0/pool_coords"
SEV_V2 = CODE / "pilot0/clip_severity_v2.csv"
SEV_V2_PROV = CODE / "pilot0/clip_severity_v2_provenance.json"
E1_SEED, E2_SEED, E3_SEED = 1201, 1211, 1221
B_DEFAULT = 2000
FINE_N = 301
PROV_REQUIRED = ("model", "pretrained_tag", "kernel", "kid_subsets",
                 "kid_point_seed", "kid_group_seeds", "versions",
                 "caches", "table_sha256")


def rho(x, y) -> float:
    return float(spearmanr(x, y, nan_policy="omit").statistic)


# ---------------------------------------------------------------------------
# Frozen predictor arms.
# ---------------------------------------------------------------------------

def dict_pair(c: int, d: int, s: float, theta: float, logit: float,
              eta: float, gamma: float, a: float, rho_v: float):
    cfg = frozen_cfg(s, theta, logit, eta, gamma, a, rho_v)
    _, _, l_e, l_c = tail_pair(c, d, cfg, fast_ctm=True)
    return l_e, l_c


def record_params(rec: dict) -> tuple[int, int, float, float, float, float]:
    c = int(rec["n_classes"])
    vc = rec["papyan"]["var_collapse"]
    sd = rec["papyan"]["self_duality"]
    s = float((c - 1) / np.sqrt(c * max(vc, 1e-9)))
    theta = float(np.degrees(np.arccos(
        np.clip(1.0 - sd / 2.0, -1.0, 1.0))))
    return (c, int(rec["dim"]), s, theta,
            float(rec["geometry"]["logit_scale"]),
            float(rec["geometry"]["class_mean_radius_cv"]))


def frozen_margin(rec: dict, co: dict) -> tuple[float, float]:
    c, d, s, theta, logit, eta = record_params(rec)
    return dict_pair(c, d, s, theta, logit, eta,
                     co["gamma"], co["a"], co["rho"])


def p10_margin(rec: dict, p10: dict) -> tuple[float, float]:
    """repair_factorial.arm_P10r on the compact block."""
    c, d, s, theta, logit, eta = record_params(rec)
    ls_e, ls_c, lw = [], [], []
    for cp in p10["components"]:
        nrm = float(np.sqrt(cp["n2"]))
        l_e, l_c = dict_pair(c, d, s, theta, logit, eta,
                             nrm / p10["R"], cp["a_max"] / nrm,
                             p10["rho_res"])
        ls_e.append(l_e)
        ls_c.append(l_c)
        lw.append(np.log(cp["weight"]))
    lw = np.array(lw)
    return (float(logsumexp(lw + np.array(ls_e))),
            float(logsumexp(lw + np.array(ls_c))))


# ---------------------------------------------------------------------------
# Severity axes from the v2 table.
# ---------------------------------------------------------------------------

def severity_axes(sev: pd.DataFrame, roster: str) -> dict:
    """{axis: {(source, eval_key): d}} with per-source z over the
    roster's own eval sets (the amendment's source-standardized rule)."""
    sub = sev[sev.roster == roster]
    out = {"K": {}, "F": {}}
    for axis, col in (("K", "kid_mmd2"), ("F", "frechet_clip_distance")):
        for source, g in sub.groupby("source"):
            v = g[col].values.astype(float)
            z = (v - v.mean()) / (v.std() + 1e-12)
            for key, d in zip(g.eval_key, z):
                out[axis][(source, key)] = float(d)
    return out


# ---------------------------------------------------------------------------
# Severity gates A and B (run BEFORE any roster outcome is opened).
# ---------------------------------------------------------------------------

def gate_a(sev: pd.DataFrame, prov: dict, frozen_rows: list[dict]) -> dict:
    missing = [k for k in PROV_REQUIRED if k not in prov]
    tag_ok = prov.get("pretrained_tag") == "laion2b_s34b_b79k"
    frozen = pd.DataFrame(frozen_rows)
    frozen["kid"] = frozen["kid"].astype(float)
    frozen["fd"] = frozen["fd"].astype(float)
    reg = sev[sev.roster == "registered"].merge(
        frozen, on=["source", "eval_dataset"], how="inner",
        suffixes=("", "_frozen"))
    n_expected = len(frozen)
    per_source = {}
    all_ok = tag_ok and not missing and len(reg) == n_expected
    for source, g in reg.groupby("source"):
        rk = rho(g.kid_mmd2, g.kid)
        rf = rho(g.frechet_clip_distance, g.fd)
        per_source[source] = {"spearman_kid": round(rk, 3),
                              "spearman_fd": round(rf, 3),
                              "n": int(len(g))}
        all_ok = all_ok and rk >= 0.9 and rf >= 0.9
    return {"pass": bool(all_ok), "provenance_missing": missing,
            "tag_ok": bool(tag_ok),
            "n_registered_matched": [int(len(reg)), int(n_expected)],
            "per_source": per_source}


def gate_b(sev: pd.DataFrame) -> dict:
    out, all_ok = {}, True
    for (roster, source), g in sev.groupby(["roster", "source"]):
        if len(g) < 3:
            out[f"{roster}/{source}"] = {"skipped": "fewer than 3 sets"}
            continue
        base = g.kid_mmd2.values.astype(float)
        reps = np.array([json.loads(v) if isinstance(v, str) else v
                         for v in g.kid_seed_means])
        taus = [float(kendalltau(base, reps[:, j]).statistic)
                for j in range(reps.shape[1])]
        ok = min(taus) >= 0.8 and float(np.mean(taus)) >= 0.9
        out[f"{roster}/{source}"] = {"tau_mean": round(float(
            np.mean(taus)), 3), "tau_min": round(min(taus), 3),
            "pass": bool(ok)}
        all_ok = all_ok and ok
    return {"pass": bool(all_ok), "per_group": out}


# ---------------------------------------------------------------------------
# Roster loaders -> cell tables. gap is in milli-AUGRC units throughout.
# ---------------------------------------------------------------------------

def _load_dir(path: Path, schema_key: str) -> tuple[list[dict], list[str]]:
    recs, problems = [], []
    if not path.is_dir():
        return recs, [f"missing dir {path}"]
    for p in sorted(path.glob("*.json")):
        if p.name.startswith("FAILED_"):
            problems.append(p.name)
            continue
        r = json.loads(p.read_text())
        if schema_key not in r:
            problems.append(f"{p.name}: no {schema_key}")
            continue
        recs.append(r)
    return recs, problems


def cells_from_records(recs: list[dict], gap_field: str,
                       with_p10: bool) -> pd.DataFrame:
    rows = []
    for rec in recs:
        cell = rec["slug"] if "slug" in rec else rec["run"]
        for name, e in rec["ood"].items():
            if "error" in e or gap_field not in e:
                continue
            l_e, l_c = frozen_margin(rec, e)
            row = dict(cell=cell, source=rec["source"], eval_dataset=name,
                       gap=float(e[gap_field]) * 1000.0,
                       l_E=l_e, l_C=l_c, m=l_e - l_c,
                       var_collapse=float(rec["papyan"]["var_collapse"]),
                       auroc_E=e.get("auroc_id_vs_ood_Energy"),
                       auroc_C=e.get("auroc_id_vs_ood_CTM"))
            if with_p10 and "p10" in e:
                p_e, p_c = p10_margin(rec, e["p10"])
                row.update(l_E_p10=p_e, l_C_p10=p_c)
            rows.append(row)
    return pd.DataFrame(rows)


def attach_axis(cells: pd.DataFrame, axis_map: dict) -> pd.DataFrame:
    out = cells.copy()
    out["d"] = [axis_map.get((s, e))
                for s, e in zip(out.source, out.eval_dataset)]
    dropped = int(out.d.isna().sum())
    if dropped:
        print(f"[analysis] WARNING: {dropped} cells without severity "
              f"dropped", flush=True)
    return out.dropna(subset=["d"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# E1: folds, material sign accuracy, paired bootstrap, GR-1.
# ---------------------------------------------------------------------------

def run_folds_e1(cells: pd.DataFrame, mode: str) -> pd.DataFrame:
    cells = cells.copy()
    sev_pred = pd.Series(np.nan, index=cells.index)
    mean_pred = pd.Series(np.nan, index=cells.index)
    ckpts = np.array(sorted(cells.cell.unique()))
    if mode == "ckpt5":
        rng = np.random.default_rng(FOLD_SEED)
        perm = rng.permutation(len(ckpts))
        folds = [set(ckpts[perm[i::5]]) for i in range(5)]
    else:
        folds = [set(cells[cells.source == s].cell.unique())
                 for s in sorted(cells.source.unique())]
    for held in folds:
        te = cells.cell.isin(held)
        train, test = cells[~te], cells[te]
        if not len(train) or not len(test):
            continue
        sev_pred[te] = severity_only(train, test)
        mean_pred[te] = float(train.gap.mean())
    cells["severity"] = sev_pred
    cells["mean"] = mean_pred
    return cells


def e1_block(folded: pd.DataFrame, b: int) -> dict:
    m = folded[np.abs(folded.gap) >= MATERIALITY]
    o = m.gap.values
    rng = np.random.default_rng(E1_SEED)
    ck = np.array(sorted(m.cell.unique()))
    gp = {c: g for c, g in m.groupby("cell")}
    diffs = np.empty(b)
    for i in range(b):
        bt = pd.concat([gp[c] for c in
                        rng.choice(ck, len(ck), replace=True)])
        diffs[i] = (accuracy(bt.m.values, bt.gap.values)
                    - accuracy(bt.severity.values, bt.gap.values))
    ci = [round(float(np.quantile(diffs, q)), 4) for q in (.025, .975)]
    loo = {}
    for s in sorted(m.source.unique()):
        sub = m[m.source != s]
        loo[s] = round(accuracy(sub.m.values, sub.gap.values)
                       - accuracy(sub.severity.values, sub.gap.values), 4)
    point = round(accuracy(m.m.values, o)
                  - accuracy(m.severity.values, o), 4)
    gr1 = bool(ci[0] > 0 and all(v > 0 for v in loo.values()))
    reversed_ = bool(ci[1] < 0)
    return {"n_material": int(len(m)),
            "n_checkpoints_material": int(len(ck)),
            "theory_sign_acc": round(accuracy(m.m.values, o), 4),
            "theory_balanced_acc": round(
                balanced_accuracy(m.m.values, o), 4),
            "severity_sign_acc": round(
                accuracy(m.severity.values, o), 4),
            "trainfold_mean_sign_acc": round(
                accuracy(m["mean"].values, o), 4),
            "diff_point": point, "diff_ci95": ci,
            "leave_one_source_diff": loo,
            "GR1_pass": gr1, "GR1_reversed": reversed_}


def e1_endpoint(cells: pd.DataFrame, axes: dict, modes: tuple,
                b: int) -> dict:
    out = {}
    for axis in ("K", "F"):
        withd = attach_axis(cells, axes[axis])
        out[axis] = {}
        for mode in modes:
            folded = run_folds_e1(withd, mode)
            out[axis][mode] = e1_block(folded, b)
    return out


# ---------------------------------------------------------------------------
# E2: roster-B within-source tertile ordering (GR-3).
# ---------------------------------------------------------------------------

def e2_source(cells_s: pd.DataFrame, b: int,
              rng: np.random.Generator) -> dict:
    strata = tertiles(cells_s)
    sets_here = sorted(cells_s.eval_dataset.unique())
    n_sets = len(sets_here)
    assert n_sets >= 2, f"E2 needs >= 2 sets, got {sets_here}"

    def analyze(sub: pd.DataFrame) -> dict:
        data: dict[str, list] = {}
        for r in sub.itertuples():
            data.setdefault(r.cell, []).append((float(r.d), float(r.gap)))
        fine = np.linspace(sub.d.min(), sub.d.max(), FINE_N)
        res = {}
        for name, cellset in strata.items():
            active = sorted(c for c in data if c in cellset)
            res[name] = analyze_curve("pava", data, active, fine, b, rng)
        return res

    full = analyze(cells_s)
    retained_full = ordering_retained(full)
    deletions = {}
    for drop in sets_here:
        res = analyze(cells_s[cells_s.eval_dataset != drop])
        deletions[drop] = bool(ordering_retained(res))
    n_del_ok = sum(deletions.values())
    gr3 = bool(retained_full and n_del_ok >= n_sets - 1)
    return {"n_sets": n_sets, "sets": sets_here,
            "n_cells": int(cells_s.cell.nunique()),
            "tertile_sizes": {k: len(v) for k, v in strata.items()},
            "full_suite": full, "ordering_retained_full": retained_full,
            "single_set_deletions_retained": deletions,
            "n_deletions_retained": [int(n_del_ok), int(n_sets)],
            "GR3_pass": gr3}


def e2_endpoint(cells: pd.DataFrame, axes: dict, b: int) -> dict:
    out = {}
    for axis in ("K", "F"):
        withd = attach_axis(cells, axes[axis])
        rng = np.random.default_rng(E2_SEED)
        out[axis] = {s: e2_source(g, b, rng)
                     for s, g in withd.groupby("source")}
    return out


# ---------------------------------------------------------------------------
# E3: absolute-level calibration, P10 vs P00 (GR-4 halves).
# ---------------------------------------------------------------------------

def pred_auroc(l: float) -> float:
    return float(np.clip(1.0 - np.exp(l), 0.0, 1.0))


def e3_endpoint(cells: pd.DataFrame, b: int) -> dict:
    if "l_E_p10" not in cells.columns:
        return {"skipped": "no P10 blocks in this roster"}
    sub = cells.dropna(subset=["auroc_E", "auroc_C", "l_E_p10"])

    def pa(vals) -> np.ndarray:
        return np.array([pred_auroc(v) for v in vals])

    per_ck = []
    for cell, g in sub.groupby("cell"):
        err00 = np.concatenate([
            np.abs(pa(g.l_E) - g.auroc_E.values.astype(float)),
            np.abs(pa(g.l_C) - g.auroc_C.values.astype(float))])
        err10 = np.concatenate([
            np.abs(pa(g.l_E_p10) - g.auroc_E.values.astype(float)),
            np.abs(pa(g.l_C_p10) - g.auroc_C.values.astype(float))])
        per_ck.append((cell, float(err00.mean()), float(err10.mean())))
    df = pd.DataFrame(per_ck, columns=["cell", "mae_P00", "mae_P10"])
    diff = df.mae_P00.values - df.mae_P10.values
    rng = np.random.default_rng(E3_SEED)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                      for _ in range(b)])
    ci = [round(float(np.quantile(boots, q)), 4) for q in (.025, .975)]
    favors = bool(ci[0] > 0)
    return {"n_checkpoints": int(len(df)),
            "n_cells": int(len(sub)),
            "mae_P00_mean": round(float(df.mae_P00.mean()), 4),
            "mae_P10_mean": round(float(df.mae_P10.mean()), 4),
            "diff_point": round(float(diff.mean()), 4),
            "diff_ci95": ci, "favors_P10_ci_excl_zero": favors}


# ---------------------------------------------------------------------------
# E4.
# ---------------------------------------------------------------------------

def e4_endpoint(cells: pd.DataFrame) -> dict:
    m = cells[np.abs(cells.gap) >= MATERIALITY]
    return {"spearman_absM_absgap_all": round(
                rho(cells.m.abs(), cells.gap.abs()), 4),
            "spearman_absM_absgap_material": round(
                rho(m.m.abs(), m.gap.abs()), 4) if len(m) > 2 else None}


# ---------------------------------------------------------------------------
# Roster C (stats + outcomes join).
# ---------------------------------------------------------------------------

def roster_c_cells(stats_dir: Path, out_dir: Path) -> pd.DataFrame:
    out_recs, problems = _load_dir(out_dir, "schema_icml_c")
    outcomes = {r["slug"]: r for r in out_recs}
    rows = []
    for p in sorted(stats_dir.glob("*.json")):
        if p.name.startswith("FAILED_"):
            problems.append(p.name)
            continue
        r = json.loads(p.read_text())
        z = np.load(stats_dir / f"{p.stem}.npz")
        orec = outcomes.get(r["slug"])
        if orec is None:
            problems.append(f"{r['slug']}: stats without outcomes")
            continue
        c_num, d_num = int(r["n_classes"]), int(r["dim"])
        pap, geo = r["papyan"], r["geometry"]
        s = float((c_num - 1) / np.sqrt(
            c_num * max(pap["var_collapse"], 1e-9)))
        theta = float(np.degrees(np.arccos(
            np.clip(1.0 - pap["self_duality"] / 2.0, -1.0, 1.0))))
        logit, eta = geo["logit_scale"], geo["class_mean_radius_cv"]
        trs_id = float(r["id_scalars"]["id__trS_id"])
        radius = float(z["id__radii"].mean())
        for name, entry in r["sets"].items():
            if entry.get("set_index", 0) == 0:
                continue
            oent = orec["ood"].get(name)
            if oent is None or "error" in oent:
                continue
            st = entry["stats"]
            key = name.replace(" ", "_")
            co = entry["coords"]
            l_e, l_c = dict_pair(c_num, d_num, s, theta, logit, eta,
                                 co["gamma"], co["a"], co["rho"])
            comps, i = [], 0
            while f"components__{i}__n" in st:
                comps.append({
                    "weight": st[f"components__{i}__weight"],
                    "n2": st[f"components__{i}__n2"],
                    "a_max": float(
                        z[f"set__{key}__components__{i}__a"].max())})
                i += 1
            rho_res = float(np.sqrt(st["resid_shared__trS"] / trs_id))
            p_e, p_c = p10_margin(
                {"n_classes": c_num, "dim": d_num, "papyan": pap,
                 "geometry": geo},
                {"components": comps, "R": radius, "rho_res": rho_res})
            rows.append(dict(
                cell=r["slug"], source=r["source"], eval_dataset=name,
                gap=float(oent["gap_raw"]) * 1000.0,
                l_E=l_e, l_C=l_c, m=l_e - l_c,
                l_E_p10=p_e, l_C_p10=p_c,
                auroc_E=oent.get("auroc_id_vs_ood_Energy"),
                auroc_C=oent.get("auroc_id_vs_ood_CTM")))
    if problems:
        print(f"[analysis] roster C problems: {problems[:5]} "
              f"({len(problems)} total)", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main run.
# ---------------------------------------------------------------------------

def run(b: int) -> None:
    sev = pd.read_csv(SEV_V2)
    prov = json.loads(SEV_V2_PROV.read_text())
    import csv as _csv
    with open(SEVERITY_CSV) as fh:
        frozen_rows = list(_csv.DictReader(fh))

    ga = gate_a(sev, prov, frozen_rows)
    gb = gate_b(sev)
    print(f"[analysis] Gate A (provenance): "
          f"{'PASS' if ga['pass'] else 'FAIL'}", flush=True)
    print(f"[analysis] Gate B (KID stability): "
          f"{'PASS' if gb['pass'] else 'FAIL'}", flush=True)
    report: dict = {"severity_gate_A": ga, "severity_gate_B": gb}
    if not (ga["pass"] and gb["pass"]):
        _write(report, aborted=True)
        raise SystemExit("[analysis] severity gates failed; per the "
                         "amendment no roster outcome may be read. "
                         "Regenerate the severity table first.")

    # --- rosters open only after the gates pass -------------------------
    recs_a, prob_a = _load_dir(DIR_A_CIFAR, "schema_icml_a")
    recs_a3, prob_a3 = _load_dir(DIR_A_IN200, "schema_stage3")
    recs_b, prob_b = _load_dir(DIR_B, "schema_icml_b")
    for r in recs_a3:
        r["slug"] = r["run"]
    cells_a = cells_from_records(recs_a + recs_a3, "gap_balanced",
                                 with_p10=True)
    cells_b = cells_from_records(recs_b, "gap_raw", with_p10=False)
    cells_c = roster_c_cells(DIR_C_STATS, DIR_C_OUT)
    axes_a = severity_axes(sev, "roster_a")
    axes_b = severity_axes(sev, "new_shifts")

    # Asserted denominators (P-4): computed from the ACTUAL rosters.
    den = {
        "roster_a_records": [len(recs_a), len(recs_a3)],
        "roster_a_cells": int(len(cells_a)),
        "roster_a_sources": sorted(cells_a.source.unique()),
        "roster_b_records": len(recs_b),
        "roster_b_cells": int(len(cells_b)),
        "roster_b_sources": sorted(cells_b.source.unique()),
        "roster_b_sets_per_source": {
            s: sorted(g.eval_dataset.unique())
            for s, g in cells_b.groupby("source")},
        "roster_c_cells": int(len(cells_c)),
        "roster_c_checkpoints": int(cells_c.cell.nunique())
                                if len(cells_c) else 0,
        "problems": {"a": prob_a + prob_a3, "b": prob_b},
    }
    assert len(cells_a), "roster A produced no cells"
    assert len(cells_b), "roster B produced no cells"
    assert den["roster_b_records"] <= 280
    for s, sets_ in den["roster_b_sets_per_source"].items():
        assert len(sets_) >= 2, f"roster B {s}: only {sets_}"
    report["denominators"] = den
    print(f"[analysis] denominators: {json.dumps(den, default=str)}",
          flush=True)

    report["E1_roster_a"] = e1_endpoint(cells_a, axes_a, ("ckpt5",), b)
    report["E1_roster_b"] = e1_endpoint(cells_b, axes_b,
                                        ("ckpt5", "loso"), b)
    report["E2_roster_b"] = e2_endpoint(cells_b, axes_b, b)
    report["E3_roster_a"] = e3_endpoint(cells_a, b)
    report["E3_roster_c"] = (e3_endpoint(cells_c, b) if len(cells_c)
                             else {"skipped": "no roster C cells"})
    report["E4"] = {"roster_a": e4_endpoint(cells_a),
                    "roster_b": e4_endpoint(cells_b),
                    "roster_c": (e4_endpoint(cells_c) if len(cells_c)
                                 else None)}

    # --- gates (primary axis d^K; d^F verdicts are the robustness read) -
    g1a = report["E1_roster_a"]["K"]["ckpt5"]
    g1b = report["E1_roster_b"]["K"]["ckpt5"]
    g1bl = report["E1_roster_b"]["K"]["loso"]
    passes = [g1a["GR1_pass"], g1b["GR1_pass"], g1bl["GR1_pass"]]
    reverses = [g1a["GR1_reversed"], g1b["GR1_reversed"],
                g1bl["GR1_reversed"]]
    gr2 = bool(sum(passes) >= 2 and not any(reverses))
    gr3 = {s: v["GR3_pass"]
           for s, v in report["E2_roster_b"]["K"].items()}
    e3a = report["E3_roster_a"]
    e3c = report["E3_roster_c"]
    gr4 = bool(e3a.get("favors_P10_ci_excl_zero")
               and e3c.get("favors_P10_ci_excl_zero"))
    report["gates"] = {
        "GR1": {"roster_a": g1a["GR1_pass"],
                "roster_b_ckpt5": g1b["GR1_pass"],
                "roster_b_loso": g1bl["GR1_pass"]},
        "GR2_pass": gr2,
        "GR2_dF_robustness": {
            "roster_a": report["E1_roster_a"]["F"]["ckpt5"]["GR1_pass"],
            "roster_b_ckpt5":
                report["E1_roster_b"]["F"]["ckpt5"]["GR1_pass"],
            "roster_b_loso":
                report["E1_roster_b"]["F"]["loso"]["GR1_pass"]},
        "GR3_per_source": gr3,
        "GR3_dF_robustness": {
            s: v["GR3_pass"]
            for s, v in report["E2_roster_b"]["F"].items()},
        "GR4_pass": gr4,
    }
    _write(report, aborted=False)
    for line in _verdicts(report):
        print(line, flush=True)


def _write(report: dict, aborted: bool) -> None:
    report["aborted_at_severity_gates"] = aborted
    (OUT_DIR / "icml_campaign_report.json").write_text(
        json.dumps(report, indent=1, default=str))
    lines = ["# ICML campaign analysis of record", "",
             "Spec in the script header; protocol section 8. "
             + ("ABORTED at the severity gates." if aborted else ""),
             "", "```", json.dumps(report, indent=1, default=str),
             "```", ""]
    (OUT_DIR / "icml_campaign_report.md").write_text("\n".join(lines))
    print(f"[analysis] wrote {OUT_DIR / 'icml_campaign_report.json'}",
          flush=True)


def _verdicts(report: dict) -> list[str]:
    g = report["gates"]
    out = [f"[analysis] GR-1: A={g['GR1']['roster_a']} "
           f"B/ckpt={g['GR1']['roster_b_ckpt5']} "
           f"B/loso={g['GR1']['roster_b_loso']}",
           f"[analysis] GR-2 (global predictor claim): "
           f"{'PASS' if g['GR2_pass'] else 'FAIL'}",
           f"[analysis] GR-3 per source: {g['GR3_per_source']}",
           f"[analysis] GR-4 (level claim): "
           f"{'PASS' if g['GR4_pass'] else 'FAIL'}"]
    return out


# ---------------------------------------------------------------------------
# Synthetic self-test (GR-5: must pass before commit; no real data).
# ---------------------------------------------------------------------------

def _synth_cells(rng, n_ck=48, sets=("s1", "s2", "s3", "s4"),
                 theory_strength=3.0, p10_better=True) -> pd.DataFrame:
    rows = []
    d_map = {s: i - 1.5 for i, s in enumerate(sets)}
    for k in range(n_ck):
        vc = float(rng.uniform(0.5, 5.0))
        for s in sets:
            d = d_map[s]
            base = 8.0 * d + rng.normal(0, 6)
            signal = theory_strength * rng.normal(0, 6)
            gap = base + signal
            m = 0.4 * signal + 0.05 * base + rng.normal(0, 0.5)
            true_a = 0.85 - 0.05 * d
            rows.append(dict(
                cell=f"ck{k:03d}", source=f"src{k % 4}",
                eval_dataset=s, gap=gap, m=m, l_E=np.log(1 - true_a),
                l_C=np.log(1 - true_a) - 0.1, var_collapse=vc,
                auroc_E=true_a + rng.normal(0, 0.002),
                auroc_C=true_a - 0.08 + rng.normal(0, 0.002),
                l_E_p10=np.log(1 - true_a)
                        + (0.001 if p10_better else 0.4),
                l_C_p10=np.log(1 - true_a + 0.08)
                        + (0.001 if p10_better else 0.4)))
    return pd.DataFrame(rows)


def self_test() -> None:
    rng = np.random.default_rng(11)
    sets = ("s1", "s2", "s3", "s4")
    cells = _synth_cells(rng)
    axes = {"K": {(f"src{i}", s): float(j - 1.5)
                  for i in range(4) for j, s in enumerate(sets)}}
    axes["F"] = axes["K"]

    # E1: planted theory signal beats severity-only.
    r1 = e1_endpoint(cells, axes, ("ckpt5", "loso"), b=300)
    blk = r1["K"]["ckpt5"]
    assert blk["n_material"] > 50, blk["n_material"]
    assert blk["theory_sign_acc"] > blk["severity_sign_acc"], blk
    assert blk["diff_ci95"][0] > 0 and blk["GR1_pass"], blk
    # No-signal control: theory margin pure noise. The fitted severity
    # baseline must win, GR-1 must fail, and the reversal flag (CI
    # wholly against the predictor) must fire.
    null = cells.copy()
    null["m"] = rng.normal(0, 1, len(null))
    r0 = e1_endpoint(null, axes, ("ckpt5",), b=300)["K"]["ckpt5"]
    assert r0["diff_ci95"][1] < 0, r0
    assert not r0["GR1_pass"] and r0["GR1_reversed"], r0
    print("[self-test] E1 planted-signal and null control PASS")

    # E2: gap curves crossing earlier for low-var_collapse (strong)
    # cells retain ordering; reversed construction fails.
    rows = []
    for k in range(30):
        vc = 0.5 + 4.5 * k / 29
        # strong crosses earliest; the 0.00317 offset keeps every
        # stratum-mean zero off the 301-point fine grid (an exact
        # grid-point zero is a synthetic knife edge the sign-based
        # crossing detector legitimately skips)
        cross = -1.00317 + 2.0 * k / 29
        for j, s in enumerate(sets):
            d = j - 1.5
            rows.append(dict(cell=f"c{k:02d}", source="src0",
                             eval_dataset=s, var_collapse=vc,
                             gap=4.0 * (d - cross),
                             d=float(d)))
    good = pd.DataFrame(rows)
    res_good = e2_source(good, b=50, rng=np.random.default_rng(E2_SEED))
    assert res_good["ordering_retained_full"], res_good["full_suite"]
    assert res_good["GR3_pass"], res_good
    bad = good.copy()
    bad["var_collapse"] = -bad["var_collapse"]  # reversed tertiles
    res_bad = e2_source(bad, b=50, rng=np.random.default_rng(E2_SEED))
    assert not res_bad["ordering_retained_full"], res_bad["full_suite"]
    print("[self-test] E2 ordering + reversed control PASS")

    # E3: planted better P10 levels -> CI favors P10; degraded -> not.
    r3 = e3_endpoint(cells, b=300)
    assert r3["favors_P10_ci_excl_zero"], r3
    worse = _synth_cells(np.random.default_rng(12), p10_better=False)
    r3w = e3_endpoint(worse, b=300)
    assert not r3w["favors_P10_ci_excl_zero"], r3w
    print("[self-test] E3 level endpoint + control PASS")

    # E4 smoke.
    r4 = e4_endpoint(cells)
    assert r4["spearman_absM_absgap_all"] > 0.2, r4
    print("[self-test] E4 PASS")

    # Severity gates on synthetic tables.
    frozen_rows = [{"source": "cifar10", "eval_dataset": f"e{j}",
                    "kid": str(0.0005 * (j + 1)),
                    "fd": str(0.1 * (j + 1))} for j in range(8)]
    sev_rows = []
    for j in range(8):
        kid = 0.0005 * (j + 1) * 1.02
        sev_rows.append(dict(
            roster="registered", source="cifar10",
            eval_dataset=f"e{j}", eval_key=f"e{j}", kid_mmd2=kid,
            frechet_clip_distance=0.1 * (j + 1) * 0.99,
            kid_seed_means=json.dumps(
                [kid * (1 + 0.01 * t) for t in range(8)])))
    sev = pd.DataFrame(sev_rows)
    prov = {k: "x" for k in PROV_REQUIRED}
    prov["pretrained_tag"] = "laion2b_s34b_b79k"
    ga = gate_a(sev, prov, frozen_rows)
    assert ga["pass"], ga
    gb = gate_b(sev)
    assert gb["pass"], gb
    bad_sev = sev.copy()
    bad_sev["kid_mmd2"] = list(bad_sev.kid_mmd2)[::-1]
    assert not gate_a(bad_sev, prov, frozen_rows)["pass"]
    noisy = sev.copy()
    noisy["kid_seed_means"] = [json.dumps(list(
        np.random.default_rng(5 + i).uniform(0.0, 0.005, 8)))
        for i in range(len(noisy))]
    assert not gate_b(noisy)["pass"]
    assert not gate_a(sev, {k: "x" for k in PROV_REQUIRED},
                      frozen_rows)["tag_ok"]
    print("[self-test] severity gates A/B + controls PASS")

    # Frozen-arm arithmetic: identical inputs reproduce the
    # corrected-dictionary recipe (regression against a fixed record).
    rec = {"n_classes": 10, "dim": 512,
           "papyan": {"var_collapse": 0.9, "self_duality": 0.3},
           "geometry": {"logit_scale": 12.0,
                        "class_mean_radius_cv": 0.1}}
    co = {"gamma": 0.8, "a": 0.4, "rho": 0.6}
    l_e, l_c = frozen_margin(rec, co)
    assert np.isfinite(l_e) and np.isfinite(l_c)
    l_e2, l_c2 = frozen_margin(rec, co)
    assert (l_e, l_c) == (l_e2, l_c2), "not deterministic"
    p10 = {"R": 1.0, "rho_res": 0.6,
           "components": [{"weight": 0.7, "n2": 0.64, "a_max": 0.32},
                          {"weight": 0.3, "n2": 1.0, "a_max": 0.4}]}
    p_e, p_c = p10_margin(rec, p10)
    assert np.isfinite(p_e) and np.isfinite(p_c)
    one = {"R": 1.0, "rho_res": 0.6,
           "components": [{"weight": 1.0, "n2": 0.64, "a_max": 0.32}]}
    p_e1, p_c1 = p10_margin(rec, one)
    d_e, d_c = dict_pair(10, 512, *(record_params(rec)[2:]),
                         0.8, 0.4, 0.6)
    assert np.isclose(p_e1, d_e) and np.isclose(p_c1, d_c), (
        "single-component mixture must equal the dictionary arm")
    print("[self-test] frozen/P10 arm arithmetic PASS")
    print("[self-test] ALL PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--b", type=int, default=B_DEFAULT)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    run(args.b)


if __name__ == "__main__":
    main()

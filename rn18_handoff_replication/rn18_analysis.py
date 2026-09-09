"""RN18 handoff-replication plan v3: the committed FIRST READER of the
four-shift ResNet-18 outcomes (phase 6). Endpoints exactly as adopted:

- HO (primary, registered, gate-based): per source and axis (d^K primary,
  d^F robustness) on the FULL panel, thirds by nc1_corrected, PAVA
  stratum curves with 2000 bands (seed 1211), frozen ordering_retained;
  retained on the full suite AND >= n-1 single-shift deletions AND every
  delete-one-checkpoint recomputation; informative (an observed
  crossing in some stratum) else HO-UNINFORMATIVE; global wording at
  >= 3 of 4 sources under d^K. Mandatory companions: tertile
  composition table; within-DeepGamblers and within-CE (dropout)
  sensitivities; the attribution sentence.
- SEL (alpha 0.025) and LEVEL (alpha 0.025): on the standalone-CE
  component, families = (dropout, run) seed blocks (N_f asserted,
  expected 10), delete-family jackknife with t_{N_f - 1}, multiplier 1
  until phase-4 qualification sets otherwise. SEL: all-cell regret on
  Delta^A, P00-H versus the nine frozen comparators, Bonferroni 9,
  epsilon 0.002, reference = matched-scalar ridge. LEVEL: P00-H vs P10-H
  MAE of predicted AUROC over Energy/CTM x 4 shifts, margin 0.01.
- ORG (descriptive): section-7.1 rank slopes on ce_do0 with the
  jackknife over run families (N_f = 5), equivalence reading at 0.003;
  full-panel slope with the regime-confound caveat.
- E4 (descriptive): Spearman(|M|, |Delta^G|).
Denominators asserted from the actual records (P-4). Also `--vgg-check`
runs the whole estimator chain on the 20 consumed VGG records
(v2 section 13.1 item 11) and `--self-test` on synthetic cells.

Usage (from code/): python rn18_handoff_replication/rn18_analysis.py [--self-test | --vgg-check | --b 2000]
Inputs: outputs/fourshift_rn18/*.json, outputs/phase1_nc1/*.json,
        pilot0/clip_severity_v2.csv, pilot0/icml_roster_b_coords/*confidnet_bbvgg13_do0_*.json
Output: outputs/rn18_report.json/.md
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from crossing_robustness_audit import analyze_curve, crossing_value, ordering_retained, tertiles
from icml_campaign_analysis import frozen_margin, p10_margin, pred_auroc, record_params
from rn18_handoff_replication.comparators import NAMES, REFERENCE, Comparators

OUT = Path("rn18_handoff_replication/outputs")
DIR_RN18 = OUT / "fourshift_rn18"
DIR_P1 = OUT / "phase1_nc1"
SEV = Path("pilot0/clip_severity_v2.csv")
VGG_GLOB = "pilot0/icml_roster_b_coords/*confidnet_bbvgg13_do0_*.json"      # AUG-VIEW sensitivity (consumed roster-B records)
DIR_VGG_DET = OUT / "fourshift_vgg_bridge"                                    # PRIMARY view (deterministic re-extraction)
CRIT = Path("rn18_handoff_replication/simulations/development_critical_values.json")
SHIFTS = ("mnist_new", "fashionmnist_new", "kmnist_new", "stl10_new")
SEED_HO, ALPHA_SEL, ALPHA_LEVEL, EPS_R, MARGIN_LEVEL, TIE = 1211, 0.025, 0.025, 0.002, 0.01, 1e-12
MIN_STRATUM, FINE_N = 5, 301


# ---------------------------------------------------------------------------
# Cell table.
# ---------------------------------------------------------------------------

def severity_axes() -> dict:
    sev = pd.read_csv(SEV)
    sub = sev[sev.roster == "new_shifts"]
    out = {}
    for s, g in sub.groupby("source"):
        for ax, col in (("dK", "kid_mmd2"), ("dF", "frechet_clip_distance")):
            v = g[col].values.astype(float)
            z = (v - v.mean()) / (v.std() + 1e-12)
            for k, d in zip(g.eval_key, z):
                out[(s, k, ax)] = float(d)
    return out


def cells_from_records(recs: list[dict], axes: dict, with_p10: bool) -> pd.DataFrame:
    rows = []
    for r in recs:
        c, d, s, theta, logit, eta = record_params(r)
        nc1 = float(r["papyan"]["var_collapse"])
        comp = r.get("component", "vgg_bridge")
        fam = (f"do{int(r.get('dropout', 0))}_run{int(r.get('run_label', 1))}" if comp == "standalone_ce"
               else "paradigm_block" if comp == "paradigm_pool" else f"vgg_run{int(r.get('run_label', 1))}")
        for e, o in r["ood"].items():
            if "error" in o or e not in SHIFTS:
                continue
            l_e, l_c = frozen_margin(r, o)
            row = dict(cell=r["slug"], component=comp, paradigm=r.get("paradigm"), dropout=int(r.get("dropout", 0)),
                       reward=r.get("reward"), source=r["source"], ood_set=e, family=fam, nc1=nc1, g=np.log(nc1),
                       dK=axes.get((r["source"], e, "dK")), dF=axes.get((r["source"], e, "dF")),
                       aurocE=float(o["auroc_id_vs_ood_Energy"]), aurocC=float(o["auroc_id_vs_ood_CTM"]),
                       dG=float(o["gap_raw"]), dG_bal=float(o["gap_balanced"]),
                       l_E=l_e, l_C=l_c, M=l_e - l_c,
                       logC=np.log(c), logD=np.log(d), logNC1=np.log(nc1), s_dict=s, theta_deg=theta,
                       logit=logit, eta=eta, log_gamma=np.log(max(o["gamma"], 1e-300)), a=o["a"],
                       log_rho=np.log(max(o["rho"], 1e-300)))
            row["dA"] = row["aurocC"] - row["aurocE"]
            if with_p10 and "p10" in o:
                pe, pc = p10_margin(r, o["p10"])
                row.update(l_E_p10=pe, l_C_p10=pc)
            rows.append(row)
    df = pd.DataFrame(rows)
    assert df.dK.notna().all() and df.dF.notna().all(), "missing severity for some (source, shift)"
    return df


def add_geometry_percentile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["g_pct"] = np.nan
    for s, g in df.groupby("source"):
        ck = g.groupby("cell").g.first()
        r = ck.rank(method="average")
        pct = ((r - 0.5) / len(ck)).to_dict()
        df.loc[df.source == s, "g_pct"] = df.loc[df.source == s, "cell"].map(pct)
    return df


# ---------------------------------------------------------------------------
# HO.
# ---------------------------------------------------------------------------

def _curves(sub: pd.DataFrame, strata: dict, axis: str, b: int, rng) -> dict:
    data: dict[str, list] = {}
    for r in sub.itertuples():
        data.setdefault(r.cell, []).append((float(getattr(r, axis)), float(r.dG)))
    fine = np.linspace(sub[axis].min(), sub[axis].max(), FINE_N)
    return {k: analyze_curve("pava", data, sorted(c for c in data if c in v), fine, b, rng)
            for k, v in strata.items()}


def _informative(res: dict) -> bool:
    return any(v.get("first_up_crossing") is not None for v in res.values())


def _thirds(sub: pd.DataFrame) -> dict:
    t = tertiles(sub.rename(columns={"nc1": "var_collapse"})[["cell", "var_collapse"]].drop_duplicates("cell"))
    return t


def two_stratum_retained(res: dict) -> bool:
    lo, hi = crossing_value(res["lower_nc1"]), crossing_value(res["higher_nc1"])
    return bool(lo < np.inf and hi >= lo - 0.05)


def ho_source(sub: pd.DataFrame, axis: str, b: int) -> dict:
    rng = np.random.default_rng(SEED_HO)
    strata = _thirds(sub)
    sizes = {k: len(v) for k, v in strata.items()}
    sets_here = sorted(sub.ood_set.unique())
    n_sets, n_ck = len(sets_here), sub.cell.nunique()
    out = {"n_checkpoints": n_ck, "n_sets": n_sets, "tertile_sizes": sizes}
    if min(sizes.values()) < MIN_STRATUM:
        out["verdict"] = "HO-INELIGIBLE"; out["reason"] = f"tertile below {MIN_STRATUM}"
        return out
    full = _curves(sub, strata, axis, b, rng)
    out["full_suite"] = full
    out["informative"] = _informative(full)
    out["retained_full"] = bool(ordering_retained(full))
    dele = {s: bool(ordering_retained(_curves(sub[sub.ood_set != s], strata, axis, 0, rng))) for s in sets_here}
    out["single_shift_deletions"] = dele
    ck_del = {}
    for c in sorted(sub.cell.unique()):
        s2 = sub[sub.cell != c]
        ck_del[c] = bool(ordering_retained(_curves(s2, _thirds(s2), axis, 0, rng)))
    out["n_checkpoint_deletions_retained"] = [int(sum(ck_del.values())), len(ck_del)]
    out["composition"] = {k: sub[sub.cell.isin(v)].drop_duplicates("cell")
                          .groupby(["paradigm", "dropout"]).size().rename(lambda x: str(x)).to_dict()
                          for k, v in strata.items()}
    ok = out["retained_full"] and sum(dele.values()) >= n_sets - 1 and all(ck_del.values())
    out["verdict"] = ("HO-UNINFORMATIVE" if not out["informative"] else "HO-RETAINED" if ok else "HO-NOT-RETAINED")
    # objective-fixed sensitivities (descriptive)
    dg = sub[sub.paradigm == "dg"]
    if dg.cell.nunique() >= 9:
        st = _thirds(dg)
        if min(len(v) for v in st.values()) >= 3:
            res = _curves(dg, st, axis, 0, rng)
            out["within_dg"] = {"retained": bool(ordering_retained(res)), "informative": _informative(res),
                                "sizes": {k: len(v) for k, v in st.items()}}
        else:
            out["within_dg"] = {"verdict": "UNINFORMATIVE (stratum < 3)"}
    else:
        out["within_dg"] = {"verdict": "UNINFORMATIVE (< 9 DG checkpoints)"}
    ce = sub[sub.component == "standalone_ce"]
    if ce.cell.nunique() >= 4:
        med = {d: ce[ce.dropout == d].groupby("cell").nc1.first().median() for d in (0, 1)}
        lo_d = min(med, key=med.get)
        st = {"lower_nc1": set(ce[ce.dropout == lo_d].cell), "higher_nc1": set(ce[ce.dropout != lo_d].cell)}
        res = _curves(ce, st, axis, 0, rng)
        out["within_ce_dropout"] = {"lower_nc1_dropout": lo_d, "retained": two_stratum_retained(res),
                                    "informative": _informative(res)}
    return out


def ho_endpoint(df: pd.DataFrame, b: int) -> dict:
    out = {}
    for axis in ("dK", "dF"):
        out[axis] = {s: ho_source(g, axis, b) for s, g in df.groupby("source")}
    ret = [s for s, v in out["dK"].items() if v["verdict"] == "HO-RETAINED"]
    out["global"] = {"retained_sources_dK": ret, "retained_sources_dF": [s for s, v in out["dF"].items() if v["verdict"] == "HO-RETAINED"],
                     "global_wording_available": len(ret) >= 3,
                     "attribution": "compatible with geometry organizing the handoff and equally with the "
                                    "training objective doing so; the design does not separate them"}
    return out


# ---------------------------------------------------------------------------
# Jackknife.
# ---------------------------------------------------------------------------

def jackknife(df: pd.DataFrame, stat, fam_col: str = "family") -> dict:
    fams = sorted(df[fam_col].unique())
    N = len(fams)
    full = float(stat(df))
    loo = np.array([stat(df[df[fam_col] != f]) for f in fams])
    if not np.all(np.isfinite(loo)):
        return {"estimate": full, "N_f": N, "degenerate": "INFERENCE_DEGENERATE"}
    se = float(np.sqrt((N - 1) / N * ((loo - loo.mean()) ** 2).sum()))
    return {"estimate": full, "se": se, "N_f": N, "degenerate": None if se > 0 else "INFERENCE_DEGENERATE"}


def interval(jk: dict, alpha: float, mult: float = 1.0):
    if jk.get("degenerate"):
        return None
    q = student.ppf(1 - alpha / 2, jk["N_f"] - 1) * mult * jk["se"]
    return [round(jk["estimate"] - q, 5), round(jk["estimate"] + q, 5)]


# ---------------------------------------------------------------------------
# SEL / LEVEL / ORG / E4.
# ---------------------------------------------------------------------------

def choice_prob(pred: np.ndarray) -> np.ndarray:
    return np.where(pred > TIE, 1.0, np.where(pred < -TIE, 0.0, 0.5))


def regret(delta: np.ndarray, p_c: np.ndarray) -> np.ndarray:
    return p_c * np.maximum(-delta, 0) + (1 - p_c) * np.maximum(delta, 0)


def sel_verdict(ivs: dict) -> str:
    """The SEL decision rule on the nine simultaneous intervals (shared
    with the phase-4 simulator). A missing interval (degenerate) blocks
    the superiority claim and the reference-based claims."""
    if all(iv is not None and iv[0] > EPS_R for iv in ivs.values()):
        return "PRACTICALLY SUPERIOR TO ALL DECLARED ZERO-SHOT COMPARATORS"
    ref = ivs.get(REFERENCE)
    if ref is None:
        return "UNRESOLVED"
    if ref[1] < -EPS_R:
        return "PRACTICALLY INFERIOR TO THE REFERENCE"
    if -EPS_R <= ref[0] and ref[1] <= EPS_R:
        return "PRACTICALLY EQUIVALENT TO THE REFERENCE"
    return "UNRESOLVED"


def level_verdict(iv: list) -> dict:
    """The LEVEL decision rule (shared with the phase-4 simulator)."""
    return {"verdict": ("resolved improvement" if iv[0] > 0 else "resolved worsening" if iv[1] < 0
                        else "unresolved direction"),
            "equivalent_within_0.01": bool(-MARGIN_LEVEL <= iv[0] and iv[1] <= MARGIN_LEVEL),
            "at_least_one_point": bool(iv[0] > MARGIN_LEVEL)}


def assert_families(df: pd.DataFrame, expected: int) -> None:
    fams = sorted(df.family.unique())
    assert len(fams) == expected, f"family count {len(fams)} != expected {expected}: {fams}"
    counts = df.groupby("family").cell.nunique()
    assert counts.nunique() == 1, f"unequal family sizes (duplicate label?): {counts.to_dict()}"


def sel_endpoint(ce: pd.DataFrame, comp: Comparators, mult: float = 1.0) -> dict:
    ok = np.isfinite(ce.dA.values) & np.isfinite(ce.M.values)
    n_bad = int((~ok).sum())
    ce = ce[ok].copy()
    ce["R_P00"] = regret(ce.dA.values, choice_prob(ce.M.values))
    for n in NAMES:
        ce[f"R_{n}"] = regret(ce.dA.values, choice_prob(comp.predict(n, ce)))
    fams = sorted(ce.family.unique())
    out = {"N_f": len(fams), "n_cells": int(len(ce)), "score_domain_failures_excluded": n_bad,
           "mean_regret_P00": float(ce.R_P00.mean()), "comparators": {}}
    alpha_each = ALPHA_SEL / len(NAMES)
    ivs = {}
    for n in NAMES:
        jk = jackknife(ce, lambda d, n=n: d[f"R_{n}"].mean() - d["R_P00"].mean())
        iv = interval(jk, alpha_each, mult)
        out["comparators"][n] = {"mean_regret": float(ce[f"R_{n}"].mean()), "D_b": jk["estimate"],
                                 "ci": iv, "degenerate": jk.get("degenerate")}
        ivs[n] = iv
    out["verdict"] = sel_verdict(ivs)
    out["material_sign_accuracy_dG"] = (float(np.mean(np.sign(ce.M[np.abs(ce.dG) >= 0.01]) == np.sign(ce.dG[np.abs(ce.dG) >= 0.01])))
                                        if (np.abs(ce.dG) >= 0.01).any() else None)
    return out


def _level_delta(sub: pd.DataFrame) -> pd.Series:
    sub = sub.copy()
    pa = lambda col: np.array([pred_auroc(v) for v in sub[col]])
    sub["e00"] = (np.abs(pa("l_E") - sub.aurocE.values) + np.abs(pa("l_C") - sub.aurocC.values)) / 2
    sub["e10"] = (np.abs(pa("l_E_p10") - sub.aurocE.values) + np.abs(pa("l_C_p10") - sub.aurocC.values)) / 2
    per_ck = sub.groupby(["family", "source", "cell"])[["e00", "e10"]].mean().reset_index()
    fam = per_ck.groupby(["family", "source"])[["e00", "e10"]].mean().groupby("family").mean()
    return fam.e00 - fam.e10


def level_endpoint(ce: pd.DataFrame, mult: float = 1.0) -> dict:
    if "l_E_p10" not in ce or ce.l_E_p10.isna().all():
        return {"verdict": "INELIGIBLE", "reason": "no P10 block"}
    need = ["l_E", "l_C", "l_E_p10", "l_C_p10", "aurocE", "aurocC"]
    finite = np.isfinite(ce[need].to_numpy(float)).all(1)
    n_missing = int((~finite).sum())
    sub = ce[finite].copy()
    if n_missing:
        d = _level_delta(sub)
        return {"verdict": "INELIGIBLE-INCOMPLETE-PANEL", "n_missing_cells": n_missing,
                "descriptive_delta_common_cells": float(d.mean()), "n_common_cells": int(len(sub))}
    d = _level_delta(sub)
    N = len(d)
    se = float(d.std(ddof=1) / np.sqrt(N))
    q = student.ppf(1 - ALPHA_LEVEL / 2, N - 1) * mult * se
    iv = [round(float(d.mean() - q), 5), round(float(d.mean() + q), 5)]
    return {"N_f": N, "delta": float(d.mean()), "ci": iv, **level_verdict(iv)}


def rank_slope_stat(df: pd.DataFrame, y: str = "dG") -> float:
    T = []
    for s, g in df.groupby("source"):
        ck = g.groupby("cell").g.first()
        r = ck.rank(method="average"); n = len(ck)
        x = ((r - 0.5) / n - 0.5)
        for e, ge in g.groupby("ood_set"):
            yv = ge.set_index("cell")[y].reindex(x.index)
            xc = x - x.mean()
            T.append(-0.8 * float((xc * (yv - yv.mean())).sum() / (xc ** 2).sum()))
    return float(np.mean(T))


def org_descriptive(df: pd.DataFrame) -> dict:
    ce0 = df[(df.component == "standalone_ce") & (df.dropout == 0)]
    out = {}
    if ce0.cell.nunique() >= 4:
        jk = jackknife(ce0, rank_slope_stat)
        iv = interval(jk, 0.05)
        out["ce_do0"] = {"A_G": jk["estimate"], "ci95_descriptive": iv, "N_f": jk["N_f"],
                         "equivalence_0.003": (bool(iv and -0.003 <= iv[0] and iv[1] <= 0.003) if iv else None),
                         "resolvable": bool(iv and (iv[1] - iv[0]) / 2 <= 0.003)}
    out["full_panel_descriptive"] = {"A_G": rank_slope_stat(df),
                                     "caveat": "geometry and training objective move together on this panel"}
    return out


def e4(df: pd.DataFrame) -> dict:
    m = df[np.abs(df.dG) >= 0.01]
    return {"spearman_absM_absdG_all": float(spearmanr(df.M.abs(), df.dG.abs()).statistic),
            "spearman_absM_absdG_material": (float(spearmanr(m.M.abs(), m.dG.abs()).statistic) if len(m) > 2 else None)}


# ---------------------------------------------------------------------------
# Runs.
# ---------------------------------------------------------------------------

def load_dir(d: Path, key: str) -> list[dict]:
    recs = []
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("FAILED_"):
            continue
        r = json.loads(p.read_text())
        if key in r:
            recs.append(r)
    return recs


def vgg_table_det(axes: dict) -> pd.DataFrame:
    """Primary comparator training table: the 20 backbones under the deterministic view."""
    recs = []
    import re
    for r in load_dir(DIR_VGG_DET, "schema_fourshift"):
        r["run_label"] = int(re.search(r"_run(\d+)_", r["model_path"]).group(1))
        r["component"] = "vgg_bridge"; r["paradigm"] = "confidnet"; r["dropout"] = 0
        recs.append(r)
    assert len(recs) == 20, len(recs)
    return add_geometry_percentile(cells_from_records(recs, axes, with_p10=False))


def vgg_table(axes: dict) -> pd.DataFrame:
    """AUG-VIEW sensitivity table: the consumed roster-B records (augmented train view)."""
    recs = []
    for p in sorted(glob.glob(VGG_GLOB)):
        r = json.load(open(p))
        import re
        r["run_label"] = int(re.search(r"_run(\d+)_", r["model_path"]).group(1))
        r["component"] = "vgg_bridge"; r["paradigm"] = "confidnet"; r["dropout"] = 0
        if "dim" not in r:                       # roster-B schema-1 records omit dim; join from pool_coords
            pc = Path("pilot0/pool_coords") / f"{r['model_path'].replace('/', '__')}.json"
            r["dim"] = int(json.loads(pc.read_text())["dim"])
        recs.append(r)
    return add_geometry_percentile(cells_from_records(recs, axes, with_p10=False))


def run(b: int) -> None:
    axes = severity_axes()
    recs = load_dir(DIR_RN18, "schema_fourshift")
    p1 = {r["slug"]: r for r in load_dir(DIR_P1, "schema_phase1")}
    for r in recs:                                   # phase-1 consistency (deterministic view)
        ref = p1[r["slug"]]["deterministic_view"]["nc1_corrected"]
        assert abs(r["papyan"]["var_collapse"] - ref) <= 1e-6 * ref, (r["slug"], r["papyan"]["var_collapse"], ref)
    df = add_geometry_percentile(cells_from_records(recs, axes, with_p10=True))
    vgg = vgg_table_det(axes)
    vgg_aug = vgg_table(axes)
    mults = json.loads(CRIT.read_text())["multipliers"]
    m10, m5 = mults["SEL_Nf10"], mults["SEL_Nf5"]
    ml10, ml5 = mults["LEVEL_Nf10"], mults["LEVEL_Nf5"]
    assert None not in (m10, m5, ml10, ml5), "unlicensed family"
    den = {"n_records": len(recs), "n_cells": int(len(df)),
           "per_source_checkpoints": df.groupby("source").cell.nunique().to_dict(),
           "sets_per_source": {s: sorted(g.ood_set.unique()) for s, g in df.groupby("source")},
           "ce_families": sorted(df[df.component == "standalone_ce"].family.unique()),
           "vgg_checkpoints_primary_view": int(vgg.cell.nunique()), "vgg_checkpoints_aug_view": int(vgg_aug.cell.nunique()),
           "multipliers": mults}
    ce = df[df.component == "standalone_ce"]
    assert_families(ce, 10)
    comp_A = Comparators(vgg, "dA")
    comp_A_aug = Comparators(vgg_aug, "dA")
    ce0 = ce[ce.dropout == 0]
    report = {"denominators": den, "HO": ho_endpoint(df, b),
              "SEL_ce": sel_endpoint(ce, comp_A, mult=m10),
              "SEL_ce_do0_sensitivity_Nf5": sel_endpoint(ce0, comp_A, mult=m5),
              "SEL_ce_augview_comparators_sensitivity": sel_endpoint(ce, comp_A_aug, mult=m10),
              "SEL_paradigm_pool_descriptive": {"mean_regret_P00": float(regret(df[df.component == "paradigm_pool"].dA.values, choice_prob(df[df.component == "paradigm_pool"].M.values)).mean())},
              "LEVEL_ce": level_endpoint(ce, mult=ml10),
              "LEVEL_ce_do0_sensitivity_Nf5": level_endpoint(ce0, mult=ml5),
              "bridge_A_G": {"primary_view": rank_slope_stat(vgg), "aug_view": rank_slope_stat(vgg_aug)},
              "ORG_descriptive": org_descriptive(df), "E4": {"full": e4(df), "ce": e4(ce)},
              "comparator_fits": {"primary_view": comp_A.summary(), "aug_view": comp_A_aug.summary()}}
    _write(report, "rn18_report")
    print(json.dumps({"HO_global": report["HO"]["global"], "SEL": report["SEL_ce"]["verdict"],
                      "LEVEL": report["LEVEL_ce"].get("verdict")}, indent=1))


def vgg_check(b: int) -> None:
    axes = severity_axes()
    vgg = vgg_table(axes)
    comp = Comparators(vgg, "dA")
    rep = {"n_cells": int(len(vgg)), "HO_dK": {s: ho_source(g, "dK", b) for s, g in vgg.groupby("source")},
           "in_sample_regret": {n: float(regret(vgg.dA.values, choice_prob(comp.predict(n, vgg))).mean()) for n in NAMES},
           "regret_P00": float(regret(vgg.dA.values, choice_prob(vgg.M.values)).mean()),
           "org_bridge_A_G": rank_slope_stat(vgg), "fits": comp.summary(), "e4": e4(vgg)}
    _write(rep, "vgg_mini_grid_check")
    print(json.dumps({k: v for k, v in rep.items() if k in ("n_cells", "in_sample_regret", "regret_P00", "org_bridge_A_G")}, indent=1))
    print("HO verdicts on the 5-checkpoint VGG panel:", {s: v["verdict"] for s, v in rep["HO_dK"].items()})


def _write(rep: dict, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=str))
    (OUT / f"{name}.md").write_text(f"# {name}\n\n```\n" + json.dumps(rep, indent=1, default=str) + "\n```\n")


# ---------------------------------------------------------------------------
# Self-test (synthetic cells; planted HO ordering, SEL, LEVEL structure).
# ---------------------------------------------------------------------------

def _synth(rng, n_par=18, n_ce_do=5) -> pd.DataFrame:
    rows = []
    for s in ("cifar10", "cifar100"):
        ck = []
        for k in range(n_par):
            ck.append((f"{s}_p{k}", "paradigm_pool", "dg" if k < 12 else "confidnet", k % 2, "paradigm_block", np.exp(rng.uniform(-3, 1))))
        for d in (0, 1):
            for r in range(1, n_ce_do + 1):
                ck.append((f"{s}_ce{d}{r}", "standalone_ce", "ce", d, f"do{d}_run{r}", np.exp(rng.normal(-1 + 0.8 * d, 0.03))))
        for cell, comp, par, do, fam, nc1 in ck:
            cross = -1.2 + 0.5 * np.log(nc1)          # lower NC1 crosses earlier
            for j, e in enumerate(SHIFTS):
                dK = j - 1.5 + 0.01
                dG = 0.02 * (dK - cross) + rng.normal(0, 0.003)
                M = 3.0 * np.sign(dG) + rng.normal(0, 1)
                aE, aC = 0.8, 0.8 + 2 * dG
                rows.append(dict(cell=cell, component=comp, paradigm=par, dropout=do, reward=0, source=s, ood_set=e,
                                 family=fam, nc1=nc1, g=np.log(nc1), dK=dK, dF=dK * 0.9, aurocE=aE, aurocC=aC,
                                 dG=dG, dG_bal=dG, l_E=np.log(0.2), l_C=np.log(0.2) - M, M=M, dA=aC - aE,
                                 logC=np.log(10), logD=np.log(512), logNC1=np.log(nc1), s_dict=9 / np.sqrt(10 * nc1),
                                 theta_deg=20.0, logit=10.0, eta=0.2, log_gamma=np.log(0.5), a=0.5, log_rho=0.0,
                                 l_E_p10=np.log(0.2) + 0.001, l_C_p10=np.log(0.2 + 2 * dG) + 0.001))
    return add_geometry_percentile(pd.DataFrame(rows))


def self_test() -> None:
    rng = np.random.default_rng(21)
    df = _synth(rng)
    ho = ho_endpoint(df, b=30)
    for s, v in ho["dK"].items():
        assert v["verdict"] == "HO-RETAINED", (s, v["verdict"], v.get("reason"))
        assert "composition" in v and v["within_ce_dropout"]["lower_nc1_dropout"] == 0
    vg = df[df.component == "standalone_ce"].copy()
    vg["family"] = vg["family"].str.replace("do1_", "do0_")
    comp = Comparators(vg.assign(component="vgg_bridge"), "dA")
    ce = df[df.component == "standalone_ce"]
    sel = sel_endpoint(ce, comp)
    assert sel["N_f"] == 10 and sel["verdict"] in ("PRACTICALLY SUPERIOR TO ALL DECLARED ZERO-SHOT COMPARATORS",
                                                   "UNRESOLVED", "PRACTICALLY INFERIOR TO THE REFERENCE",
                                                   "PRACTICALLY EQUIVALENT TO THE REFERENCE"), sel
    lv = level_endpoint(ce)
    assert lv["verdict"] == "resolved improvement" and lv["N_f"] == 10, lv
    org = org_descriptive(df)
    assert org["ce_do0"]["N_f"] == 5
    null = df.copy(); null["M"] = rng.normal(0, 1, len(null))
    assert abs(e4(null)["spearman_absM_absdG_all"]) < 0.2
    print(f"[rn18-analysis] self-test PASS: HO retained on planted ordering (both sources), SEL verdict "
          f"'{sel['verdict']}', LEVEL {lv['verdict']} ({lv['delta']:+.4f}), ORG ce_do0 N_f=5")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--vgg-check", action="store_true", dest="vgg_check")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.vgg_check:
        vgg_check(args.b)
    else:
        run(args.b)


if __name__ == "__main__":
    main()

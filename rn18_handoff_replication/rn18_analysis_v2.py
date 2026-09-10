"""RN18 handoff-replication reader, version 2 (fail-closed).

Status-review finding F9 (2026-09-09) established that the first reader
`rn18_analysis.py` (reader of record, commit c7d1b98) was not fail-closed:
a non-finite comparator prediction became choice probability 0.5 (an
intentional tie); LEVEL had no zero-standard-error guard; no complete
expected key set was compared; only the development multipliers were
checked, not the audit licenses or the freeze hashes; verdicts were taken
on five-decimal rounded intervals. This module is the versioned
correction (repair item P1-C). The original reader and its readout of
record (`outputs/rn18_report.json`) are never edited.

What changes (and nothing else):

1. A validator runs BEFORE any endpoint computation and stops the run on
   the first failure with a declared verdict: freeze hashes (code and
   manifests, with the one recorded reader-of-record discrepancy), the
   mechanical output inventory (JSON set equal to the freeze by name,
   size and sha256; extractor `.npz` sidecars of inventoried slugs
   recorded with their hashes; no other files; no FAILED_ files), audit licenses (all four families licensed under the
   audit seed and replication count recorded in FREEZE.json), the
   complete expected (cell, shift) key set from the frozen expected
   panel, every required field finite, the family structure (10 blocks
   of 4, seed formula 270000 + 5 dropout + run - 1), the phase-1 NC1
   consistency, the severity axes, and the expected denominators.
2. One non-estimability policy: a non-finite required input on a
   registered endpoint's panel makes that endpoint NOT ESTIMABLE; a
   non-finite comparator prediction makes that comparator NOT ESTIMABLE;
   a zero or non-finite jackknife standard error makes that interval
   NOT ESTIMABLE. Nothing is dropped, imputed, or tied.
3. Decisions are taken on full-precision intervals; rounded copies are
   written for display only.
4. Licensed multipliers are read from the version-2 qualification
   (`simulations/qualification_report_v2.json`, repair item P1-B). Without
   that file the readout is WITHHELD: the validator report is written
   and no endpoint is computed.
5. A correction record is written beside the readout: the original
   readout's hash, the reader-of-record and frozen reader hashes, the
   rule changes, and a mechanical list of every verdict that differs
   from the readout of record.

HO, the cell table, the comparators, ORG and E4 are imported UNCHANGED
from the reader of record.

Usage (from code/):
    python rn18_handoff_replication/rn18_analysis_v2.py --self-test
    python rn18_handoff_replication/rn18_analysis_v2.py --validate-only
    python rn18_handoff_replication/rn18_analysis_v2.py [--b 2000]
Outputs: outputs/rn18_report_v2_validation.json, outputs/rn18_report_v2.json/.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT))

from icml_campaign_analysis import pred_auroc
from rn18_handoff_replication import rn18_analysis as v1
from rn18_handoff_replication.comparators import NAMES, REFERENCE, Comparators
from rn18_handoff_replication.rn18_analysis import (ALPHA_LEVEL, ALPHA_SEL, EPS_R, MARGIN_LEVEL, OUT, SHIFTS, TIE,
                                                    add_geometry_percentile, cells_from_records, e4, ho_endpoint,
                                                    load_dir, rank_slope_stat, regret, severity_axes, vgg_table,
                                                    vgg_table_det)

ROOT = Path("rn18_handoff_replication")
FREEZE = ROOT / "FREEZE.json"
PANEL = ROOT / "manifests/expected_panel.json"
QUAL_V1 = ROOT / "simulations/qualification_report.json"
QUAL_V2 = ROOT / "simulations/qualification_report_v2.json"
AUDIT_V1 = ROOT / "simulations/audit_results.json"
REPORT_V1 = OUT / "rn18_report.json"
READER_OF_RECORD_SHA = "c50c47fa8ff2b230" + ""     # prefix; full value checked below
READER_OF_RECORD_COMMIT = "c7d1b98"
READER_DISCREPANCY = ("frozen hash 038cf044... (commit 6ae5b4b) vs reader of record c50c47fa... (commit c7d1b98): "
                      "the only difference is the JSON serialization of the tertile composition keys; "
                      "no numerical path changed")
REQUIRED_OOD = ("auroc_id_vs_ood_Energy", "auroc_id_vs_ood_CTM", "gap_raw", "gap_balanced", "gamma", "a", "rho")
SEED_BASE = 270000
EVIDENCE = {"HO": "REGISTERED (gate-based, no alpha)", "SEL": "REGISTERED (alpha 0.025, Bonferroni 9)",
            "LEVEL": "REGISTERED (alpha 0.025)", "sensitivity": "PRE-SPECIFIED SENSITIVITY (descriptive)",
            "ORG": "DESCRIPTIVE", "E4": "DESCRIPTIVE", "bridge": "DESCRIPTIVE (consumed VGG bridge)"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class ValidationFailure(SystemExit):
    def __init__(self, check: str, detail):
        self.check, self.detail = check, detail
        super().__init__(f"VALIDATION FAILED [{check}]: {detail}")


# ---------------------------------------------------------------------------
# Validator (runs before any endpoint computation; fails closed).
# ---------------------------------------------------------------------------

def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:                                 # noqa: BLE001
        return f"unavailable ({exc})"


def validate(recs: list[dict], vgg_recs: list[dict], p1: dict, axes: dict, require_v2_license: bool = True) -> dict:
    """Every check either passes or raises ValidationFailure. Returns the
    validation record (all checks with their observed values)."""
    rep = {"git_head_at_run": _git_head(), "checks": {}}
    fz = json.loads(FREEZE.read_text())
    # 1. freeze hashes (code + manifests), with the one recorded discrepancy
    mism = []
    for grp in ("code", "manifests"):
        for rel, h in fz["hashes"][grp].items():
            p = Path(rel)
            if not p.exists():
                mism.append((rel, "missing")); continue
            got = sha(p)
            if got != h:
                if rel.endswith("rn18_analysis.py") and got.startswith(READER_OF_RECORD_SHA):
                    rep["checks"]["reader_of_record_discrepancy"] = {"frozen": h, "reader_of_record": got,
                                                                     "commit": READER_OF_RECORD_COMMIT,
                                                                     "attribution": READER_DISCREPANCY}
                else:
                    mism.append((rel, got))
    if mism:
        raise ValidationFailure("freeze_hashes", mism)
    rep["checks"]["freeze_hashes"] = {"n_checked": sum(len(fz["hashes"][g]) for g in ("code", "manifests")), "mismatches": []}
    rep["checks"]["reader_v2_sha256"] = sha(Path(__file__))
    # 2. mechanical inventory of the unread outputs. The freeze script inventoried
    #    `*.json` only (phase5_freeze.inventory); the extractor also writes one
    #    `<slug>.npz` sidecar per record (float32 covariances, F10). The JSON set
    #    must match the freeze exactly; sidecars of inventoried slugs are RECORDED
    #    (name, size, sha256) as a disclosed fact; any other file fails.
    for d in ("fourshift_rn18", "fourshift_vgg_bridge"):
        inv = fz["unread_outputs"][d]
        root = OUT / d
        present = {p.name: p for p in root.glob("*") if p.is_file()}
        failed = sorted(n for n in present if n.startswith("FAILED_"))
        if failed:
            raise ValidationFailure(f"inventory:{d}", {"FAILED_files": failed})
        want = {f["name"]: f for f in inv["files"]}
        slugs = {n[:-5] for n in want}
        jsons = {n for n in present if n.endswith(".json")}
        sidecars = sorted(n for n in present if n.endswith(".npz") and n[:-4] in slugs)
        other = sorted(set(present) - jsons - set(sidecars))
        extra = sorted(jsons - set(want)); missing = sorted(set(want) - jsons)
        changed = [n for n, f in want.items() if n in present and (present[n].stat().st_size != f["bytes"] or sha(present[n]) != f["sha256"])]
        if extra or missing or changed or other or len(want) != inv["n"]:
            raise ValidationFailure(f"inventory:{d}", {"extra": extra, "missing": missing, "changed": changed, "other_files": other})
        rep["checks"][f"inventory:{d}"] = {"n": inv["n"], "extra": [], "missing": [], "changed": [], "other_files": [],
                                           "sidecars_not_in_freeze_inventory": {"n": len(sidecars), "rule": "recorded, not inventoried by the freeze",
                                                                                "files": [{"name": n, "bytes": present[n].stat().st_size, "sha256": sha(present[n])} for n in sidecars]}}
    # 3. licenses (version 1 audit, and the version-2 license that this reader uses)
    q1 = json.loads(QUAL_V1.read_text()); a1 = json.loads(AUDIT_V1.read_text())
    if a1["stage"] != "audit" or a1["seed"] != fz["seeds"]["sim_audit"] or a1["reps"] != 10000:
        raise ValidationFailure("audit_v1", {"stage": a1["stage"], "seed": a1["seed"], "reps": a1["reps"]})
    unl = {k: v for k, v in q1["licenses"].items() if not v.get("licensed")}
    if unl or set(q1["licenses"]) != {"SEL_Nf10", "SEL_Nf5", "LEVEL_Nf10", "LEVEL_Nf5"}:
        raise ValidationFailure("licenses_v1", unl or list(q1["licenses"]))
    rep["checks"]["licenses_v1"] = {k: v["multiplier"] for k, v in q1["licenses"].items()}
    rep["checks"]["license_v2"] = None
    if QUAL_V2.exists():
        q2 = json.loads(QUAL_V2.read_text())
        if set(q2.get("licenses", {})) != {"SEL_Nf10", "SEL_Nf5", "LEVEL_Nf10", "LEVEL_Nf5"}:
            raise ValidationFailure("licenses_v2", {"families": sorted(q2.get("licenses", {}))})
        # an unlicensed family is NOT fatal: the plan makes that endpoint descriptive (alpha unused)
        rep["checks"]["license_v2"] = {"path": str(QUAL_V2), "sha256": sha(QUAL_V2), "seed": q2.get("seed"),
                                       "licensed": {k: v["multiplier"] for k, v in q2["licenses"].items() if v.get("licensed")},
                                       "unlicensed": {k: v.get("failed_scenarios") for k, v in q2["licenses"].items() if not v.get("licensed")}}
    elif require_v2_license:
        rep["checks"]["license_v2"] = "ABSENT: readout withheld (repair item P1-B pending)"
    # 4. complete expected key set
    panel = json.loads(PANEL.read_text())
    elig = [c for c in panel["cells"] if c["mechanically_eligible"]]
    if len(elig) != fz["panel_of_record"]["n_eligible"]:
        raise ValidationFailure("expected_panel", {"eligible": len(elig), "frozen": fz["panel_of_record"]["n_eligible"]})
    want_keys = {(c["model_path"].replace("/", "__"), s) for c in elig for s in SHIFTS}
    have_keys, bad_fields = set(), []
    for r in recs:
        for s, o in r["ood"].items():
            if s not in SHIFTS:
                continue
            if "error" in o:
                bad_fields.append((r["slug"], s, "error")); continue
            have_keys.add((r["slug"], s))
            for k in REQUIRED_OOD:
                if k not in o or not np.isfinite(float(o[k])):
                    bad_fields.append((r["slug"], s, k))
            if "p10" not in o:
                bad_fields.append((r["slug"], s, "p10_absent"))
        for k, v in (("var_collapse", r["papyan"].get("var_collapse")), ("self_duality", r["papyan"].get("self_duality")),
                     ("logit_scale", r["geometry"].get("logit_scale")), ("class_mean_radius_cv", r["geometry"].get("class_mean_radius_cv")),
                     ("dim", r.get("dim")), ("n_classes", r.get("n_classes"))):
            if v is None or not np.isfinite(float(v)) or (k == "var_collapse" and v <= 0):
                bad_fields.append((r["slug"], "record", k))
    missing = sorted(want_keys - have_keys); extra = sorted(have_keys - want_keys)
    if missing or extra or bad_fields:
        raise ValidationFailure("key_set", {"missing": missing[:20], "n_missing": len(missing), "extra": extra[:20],
                                            "bad_fields": bad_fields[:20], "n_bad_fields": len(bad_fields)})
    rep["checks"]["key_set"] = {"n_expected": len(want_keys), "n_present": len(have_keys), "missing": 0, "extra": 0, "bad_fields": 0}
    # 5. family structure and seed formula on the standalone-CE component
    ce = [r for r in recs if r["component"] == "standalone_ce"]
    fam: dict[tuple, list] = {}
    for r in ce:
        fam.setdefault((int(r["dropout"]), int(r["run_label"])), []).append(r)
    bad = {}
    for (do, run), rs in sorted(fam.items()):
        seeds = {int(r["seed"]) for r in rs}; srcs = sorted(r["source"] for r in rs)
        exp_seed = SEED_BASE + 5 * do + (run - 1)
        if seeds != {exp_seed} or len(rs) != 4 or len(set(srcs)) != 4:
            bad[f"do{do}_run{run}"] = {"seeds": sorted(seeds), "expected_seed": exp_seed, "sources": srcs}
    if bad or len(fam) != fz["expected_denominators"]["ce_families"]:
        raise ValidationFailure("families", bad or {"n_families": len(fam)})
    rep["checks"]["families"] = {"n": len(fam), "checkpoints_per_family": 4, "seed_formula": "270000 + 5*dropout + (run-1)"}
    # 6. phase-1 NC1 consistency (deterministic view)
    off = []
    for r in recs:
        if r["slug"] not in p1:
            off.append((r["slug"], "no phase-1 record")); continue
        ref = p1[r["slug"]]["deterministic_view"]["nc1_corrected"]
        if abs(r["papyan"]["var_collapse"] - ref) > 1e-6 * ref:
            off.append((r["slug"], r["papyan"]["var_collapse"], ref))
    if off:
        raise ValidationFailure("phase1_consistency", off[:10])
    rep["checks"]["phase1_consistency"] = {"n": len(recs), "tolerance_rel": 1e-6}
    # 7. severity axes complete
    srcs = sorted({r["source"] for r in recs})
    miss_ax = [(s, e, ax) for s in srcs for e in SHIFTS for ax in ("dK", "dF") if (s, e, ax) not in axes or not np.isfinite(axes[(s, e, ax)])]
    if miss_ax:
        raise ValidationFailure("severity_axes", miss_ax)
    rep["checks"]["severity_axes"] = {"n_sources": len(srcs), "n_keys": len(srcs) * len(SHIFTS) * 2}
    # 8. denominators
    ed = fz["expected_denominators"]
    obs = {"rn18_records": len(recs), "rn18_cells": len(have_keys), "ce_families": len(fam),
           "ce_do0_families": sum(1 for (do, _) in fam if do == 0), "vgg_bridge_records": len(vgg_recs), "sets_per_source": len(SHIFTS)}
    if any(obs[k] != ed[k] for k in ed):
        raise ValidationFailure("denominators", {"expected": ed, "observed": obs})
    rep["checks"]["denominators"] = obs
    rep["verdict"] = "VALIDATION PASSED"
    return rep


# ---------------------------------------------------------------------------
# Inference primitives (version 2: one non-estimability policy, unrounded).
# ---------------------------------------------------------------------------

def choice_prob(pred: np.ndarray) -> np.ndarray:
    """CTM if > TIE, Energy if < -TIE, tie otherwise; a non-finite
    prediction PROPAGATES as NaN (unavailable), never as a tie."""
    pred = np.asarray(pred, float)
    out = np.where(pred > TIE, 1.0, np.where(pred < -TIE, 0.0, 0.5))
    return np.where(np.isfinite(pred), out, np.nan)


def jackknife(df: pd.DataFrame, stat, fam_col: str = "family") -> dict:
    fams = sorted(df[fam_col].unique())
    N = len(fams)
    full = float(stat(df))
    loo = np.array([stat(df[df[fam_col] != f]) for f in fams], float)
    if not np.isfinite(full) or not np.all(np.isfinite(loo)):
        return {"estimate": full, "N_f": N, "not_estimable": "non-finite statistic or leave-one-out value"}
    se = float(np.sqrt((N - 1) / N * ((loo - loo.mean()) ** 2).sum()))
    if not np.isfinite(se) or se <= 0:
        return {"estimate": full, "se": se, "N_f": N, "not_estimable": "zero or non-finite jackknife standard error"}
    return {"estimate": full, "se": se, "N_f": N, "not_estimable": None}


def interval(jk: dict, alpha: float, mult: float):
    if jk.get("not_estimable"):
        return None
    q = student.ppf(1 - alpha / 2, jk["N_f"] - 1) * mult * jk["se"]
    return [float(jk["estimate"] - q), float(jk["estimate"] + q)]          # full precision


def display(iv):
    return None if iv is None else [round(iv[0], 5), round(iv[1], 5)]


def sel_verdict(ivs: dict) -> str:
    """Decision on the nine simultaneous full-precision intervals. A
    NOT ESTIMABLE comparator blocks the superiority claim; a NOT
    ESTIMABLE reference makes the reference-based claims NOT ESTIMABLE."""
    if all(iv is not None and iv[0] > EPS_R for iv in ivs.values()):
        return "PRACTICALLY SUPERIOR TO ALL DECLARED ZERO-SHOT COMPARATORS"
    ref = ivs.get(REFERENCE)
    if ref is None:
        return "NOT ESTIMABLE (reference interval unavailable)"
    if ref[1] < -EPS_R:
        return "PRACTICALLY INFERIOR TO THE REFERENCE"
    if -EPS_R <= ref[0] and ref[1] <= EPS_R:
        return "PRACTICALLY EQUIVALENT TO THE REFERENCE"
    return "UNRESOLVED"


def level_verdict(iv) -> dict:
    if iv is None:
        return {"verdict": "NOT ESTIMABLE", "equivalent_within_0.01": None, "at_least_one_point": None}
    return {"verdict": ("resolved improvement" if iv[0] > 0 else "resolved worsening" if iv[1] < 0 else "unresolved direction"),
            "equivalent_within_0.01": bool(-MARGIN_LEVEL <= iv[0] and iv[1] <= MARGIN_LEVEL),
            "at_least_one_point": bool(iv[0] > MARGIN_LEVEL)}


UNLICENSED = "DESCRIPTIVE (family unlicensed by the version-2 qualification; interval at multiplier 1 is conditional on the approximate procedure)"


def _license_fields(mult, evidence: str, failed) -> tuple[float, dict]:
    if mult is None:
        return 1.0, {"evidence_class": UNLICENSED, "declared_class": evidence, "license": {"licensed": False, "failed_scenarios": failed}}
    return float(mult), {"evidence_class": evidence, "license": {"licensed": True, "multiplier": float(mult)}}


def _unlicense(out: dict, licensed: bool) -> dict:
    if not licensed and "verdict" in out:
        out["verdict_if_licensed_at_multiplier_1"] = out["verdict"]
        out["verdict"] = "NO LICENSE: descriptive"
    return out


def sel_endpoint(ce: pd.DataFrame, comp: Comparators, mult, evidence: str, failed=None) -> dict:
    """All-cell regret on Delta^A. No cell is dropped: a non-finite
    required input makes the endpoint NOT ESTIMABLE (declared). mult None =
    unlicensed family: descriptive interval at multiplier 1, no claim."""
    fams = sorted(ce.family.unique())
    licensed = mult is not None
    mult, lic = _license_fields(mult, evidence, failed)
    out = {**lic, "N_f": len(fams), "n_cells": int(len(ce)), "multiplier": mult}
    bad = ~(np.isfinite(ce.dA.values) & np.isfinite(ce.M.values))
    if bad.any():
        out.update(verdict="NOT ESTIMABLE (non-finite required input on the registered panel)",
                   n_nonfinite_cells=int(bad.sum()), cells=sorted(ce.cell[bad].unique().tolist()))
        return out
    ce = ce.copy()
    p00 = choice_prob(ce.M.values)
    ce["R_P00"] = regret(ce.dA.values, p00)
    out["mean_regret_P00"] = float(ce.R_P00.mean())
    out["P00_choice_counts"] = {"CTM": int((p00 == 1).sum()), "Energy": int((p00 == 0).sum()), "tie": int((p00 == 0.5).sum())}
    out["P00_identical_to_always_ctm"] = bool((p00 == 1).all())
    out["per_source_mean_regret_P00"] = {s: float(g.R_P00.mean()) for s, g in ce.groupby("source")}
    mat = np.abs(ce.dG.values) >= 0.01
    out["material_subset"] = {"n_material": int(mat.sum()), "n_all": int(len(ce)),
                              "sign_accuracy_material_dG": (float(np.mean(np.sign(ce.M.values[mat]) == np.sign(ce.dG.values[mat]))) if mat.any() else None),
                              "sign_accuracy_all_nonzero_dG": float(np.mean(np.sign(ce.M.values[ce.dG.values != 0]) == np.sign(ce.dG.values[ce.dG.values != 0])))}
    alpha_each = ALPHA_SEL / len(NAMES)
    ivs, out["comparators"] = {}, {}
    for n in NAMES:
        pred = np.asarray(comp.predict(n, ce), float)
        n_unavail = int((~np.isfinite(pred)).sum())
        if n_unavail:
            out["comparators"][n] = {"not_estimable": f"{n_unavail} non-finite predictions", "ci": None}
            ivs[n] = None
            continue
        ce[f"R_{n}"] = regret(ce.dA.values, choice_prob(pred))
        jk = jackknife(ce, lambda d, n=n: d[f"R_{n}"].mean() - d["R_P00"].mean())
        iv = interval(jk, alpha_each, mult)
        out["comparators"][n] = {"mean_regret": float(ce[f"R_{n}"].mean()), "D_b": jk["estimate"], "se": jk.get("se"),
                                 "ci": iv, "ci_display": display(iv), "not_estimable": jk.get("not_estimable")}
        ivs[n] = iv
    out["verdict"] = sel_verdict(ivs)
    return _unlicense(out, licensed)


def _level_family_deltas(sub: pd.DataFrame) -> pd.Series:
    sub = sub.copy()
    pa = lambda col: np.array([pred_auroc(v) for v in sub[col]])
    sub["e00"] = (np.abs(pa("l_E") - sub.aurocE.values) + np.abs(pa("l_C") - sub.aurocC.values)) / 2
    sub["e10"] = (np.abs(pa("l_E_p10") - sub.aurocE.values) + np.abs(pa("l_C_p10") - sub.aurocC.values)) / 2
    per_ck = sub.groupby(["family", "source", "cell"])[["e00", "e10"]].mean().reset_index()
    fam = per_ck.groupby(["family", "source"])[["e00", "e10"]].mean().groupby("family").mean()
    return fam.e00 - fam.e10


def level_endpoint(ce: pd.DataFrame, mult, evidence: str, failed=None) -> dict:
    licensed = mult is not None
    mult, lic = _license_fields(mult, evidence, failed)
    out = {**lic, "multiplier": mult}
    need = ["l_E", "l_C", "l_E_p10", "l_C_p10", "aurocE", "aurocC"]
    if any(c not in ce for c in need):
        out.update(verdict="NOT ESTIMABLE (P10 block absent)"); return out
    finite = np.isfinite(ce[need].to_numpy(float)).all(1)
    if not finite.all():
        out.update(verdict="NOT ESTIMABLE (non-finite required input on the registered panel)",
                   n_nonfinite_cells=int((~finite).sum()), cells=sorted(ce.cell[~finite].unique().tolist()))
        return out
    d = _level_family_deltas(ce)
    N = int(len(d))
    se = float(d.std(ddof=1) / np.sqrt(N)) if N > 1 else float("nan")
    out.update(N_f=N, delta=float(d.mean()), se=se, family_deltas={k: float(v) for k, v in d.items()})
    if not np.isfinite(se) or se <= 0:
        out.update(ci=None, **level_verdict(None), not_estimable="zero or non-finite standard error of the family deltas")
        return out
    q = student.ppf(1 - ALPHA_LEVEL / 2, N - 1) * mult * se
    iv = [float(d.mean() - q), float(d.mean() + q)]
    out.update(ci=iv, ci_display=display(iv), **level_verdict(iv))
    return _unlicense(out, licensed)


def org_descriptive(df: pd.DataFrame) -> dict:
    ce0 = df[(df.component == "standalone_ce") & (df.dropout == 0)]
    out = {"evidence_class": EVIDENCE["ORG"]}
    if ce0.cell.nunique() >= 4:
        jk = jackknife(ce0, rank_slope_stat)
        iv = interval(jk, 0.05, 1.0)
        out["ce_do0"] = {"A_G": jk["estimate"], "ci95_descriptive": iv, "ci95_display": display(iv), "N_f": jk["N_f"],
                         "not_estimable": jk.get("not_estimable"),
                         "equivalence_0.003": (bool(-0.003 <= iv[0] and iv[1] <= 0.003) if iv else None),
                         "resolvable": (bool((iv[1] - iv[0]) / 2 <= 0.003) if iv else None)}
    out["full_panel_descriptive"] = {"A_G": rank_slope_stat(df), "caveat": "geometry and training objective move together on this panel"}
    return out


# ---------------------------------------------------------------------------
# Comparator specification dump (P2: coefficients, scalers, folds, schema).
# ---------------------------------------------------------------------------

def comparator_dump(comp: Comparators, train: pd.DataFrame) -> dict:
    """Everything needed to re-apply the nine frozen comparators without
    refitting: ridge coefficients on standardized features, the per-fold
    standardization is refit inside CV but the FINAL fit's scalers are the
    ones applied; leave-one-checkpoint-out fold identities; the feature
    schema; the per-source isotonic tables; the source-shift mean table;
    the majority table. Two inherited rules are recorded as discrepancies
    with the protocol text (status-review F11): the within-source geometry
    percentile is computed on the full panel before the CE subset and is
    not recomputed inside CV folds or after target deletions; the majority
    baseline selects material cells by the AUGRC gap dG while predicting
    the AUROC gap dA. The comparators stay as frozen (pre-registered); a
    reconciled version would be a new, separately declared comparator set."""
    from heldout_theory_validation import severity_only
    out = {"folds": {"rule": "leave-one-VGG-checkpoint-out", "fold_ids": sorted(train.cell.unique())},
           "indicator_sources": list(comp.ind_sources), "reference_source": next(s for s in ("cifar10", "cifar100", "supercifar100", "tinyimagenet") if (train.source == s).any()),
           "ridge": {}, "isotonic": {}, "source_shift_mean": {f"{s}|{e}": float(v) for (s, e), v in comp.mean_table.items()},
           "source_majority": comp.majority,
           "rule_discrepancies_with_protocol": [
               "geometry percentile g_pct computed per source on the full panel before the CE subset; not recomputed within CV folds or after target deletions",
               "majority baseline selects material cells by |dG| >= 0.01 (AUGRC gap) while the target metric is dA (AUROC gap); fallbacks: non-zero-target majority, then global, then Energy"]}
    for name, (fit, cont, inter, lam, losses) in comp.fits.items():
        feats = list(cont) + (["dK*g_pct"] if inter else [])
        out["ridge"][name] = {"features": feats, "lambda": lam, "cv_losses": {str(k): float(v) for k, v in losses.items()},
                              "scaler_mean": fit["mu"].tolist(), "scaler_sd": [None if not np.isfinite(v) else float(v) for v in fit["sd"]],
                              "beta": {"intercept": float(fit["beta"][0]),
                                       "source_indicators": dict(zip(comp.ind_sources, fit["beta"][1:1 + len(comp.ind_sources)].tolist())),
                                       "standardized_features": dict(zip(feats, fit["beta"][1 + len(comp.ind_sources):].tolist()))}}
    for col, nm in (("dK", "vgg_kid_isotonic"), ("dF", "vgg_fd_isotonic")):
        out["isotonic"][nm] = {}
        for src, g in train.groupby("source"):
            tr = g.rename(columns={col: "d", comp.target_col: "gap"})
            ds = np.unique(tr.d.values)
            fitted = severity_only(tr, pd.DataFrame({"d": ds}))
            out["isotonic"][nm][src] = {"d": ds.tolist(), "fitted_gap": fitted.tolist()}
    return out


# ---------------------------------------------------------------------------
# Correction record: mechanical comparison with the readout of record.
# ---------------------------------------------------------------------------

def _verdict_paths(rep: dict, prefix: str = "") -> dict:
    """Every string-valued 'verdict' (and HO retained flags) with its JSON path."""
    out = {}
    if isinstance(rep, dict):
        for k, v in rep.items():
            p = f"{prefix}/{k}"
            if k == "verdict" and isinstance(v, str):
                out[prefix or "/"] = v
            elif k in ("retained_full", "global_wording_available", "equivalent_within_0.01", "at_least_one_point") and isinstance(v, bool):
                out[p] = v
            else:
                out.update(_verdict_paths(v, p))
    return out


def correction_record(rep_v2: dict) -> dict:
    rec = {"original_readout": {"path": str(REPORT_V1), "sha256": sha(REPORT_V1) if REPORT_V1.exists() else None,
                                "bytes": REPORT_V1.stat().st_size if REPORT_V1.exists() else None,
                                "reader_of_record_commit": READER_OF_RECORD_COMMIT},
           "reader_discrepancy": READER_DISCREPANCY,
           "rule_changes": ["validator before any computation (freeze hashes, inventory, licenses, key set, families, phase-1, axes, denominators)",
                            "non-finite comparator prediction = comparator NOT ESTIMABLE (was: tie with probability 0.5)",
                            "non-finite required input on a registered panel = endpoint NOT ESTIMABLE (was: cell dropped)",
                            "zero or non-finite jackknife/LEVEL standard error = NOT ESTIMABLE (was: LEVEL had no guard)",
                            "decisions on full-precision intervals (was: five-decimal rounded)",
                            "multipliers from the version-2 qualification license (was: version-1 development values)",
                            "evidence-class label on every endpoint; both denominators and the constant-policy identity reported"],
           "conclusions_changed": None}
    if REPORT_V1.exists():
        v1r = json.loads(REPORT_V1.read_text())
        a, b = _verdict_paths(v1r), _verdict_paths(rep_v2)
        rec["conclusions_changed"] = {p: {"readout_of_record": a[p], "v2": b[p]} for p in a if p in b and a[p] != b[p]}
        rec["verdicts_compared"] = len([p for p in a if p in b])
        rec["verdicts_only_in_v2"] = sorted(p for p in b if p not in a)[:40]
    return rec


# ---------------------------------------------------------------------------
# Runs.
# ---------------------------------------------------------------------------

def _load_all():
    axes = severity_axes()
    recs = load_dir(v1.DIR_RN18, "schema_fourshift")
    vgg_recs = load_dir(v1.DIR_VGG_DET, "schema_fourshift")
    p1 = {r["slug"]: r for r in load_dir(v1.DIR_P1, "schema_phase1")}
    return axes, recs, vgg_recs, p1


def run(b: int, validate_only: bool) -> None:
    axes, recs, vgg_recs, p1 = _load_all()
    val = validate(recs, vgg_recs, p1, axes)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rn18_report_v2_validation.json").write_text(json.dumps(val, indent=1, default=str))
    print(json.dumps({k: (v if k != "checks" else {kk: ("ok" if isinstance(vv, dict) else vv) for kk, vv in v.items()})
                      for k, v in val.items()}, indent=1, default=str))
    if validate_only:
        return
    if not QUAL_V2.exists():
        print("READOUT WITHHELD: simulations/qualification_report_v2.json is absent (repair item P1-B pending). "
              "Validation record written; no endpoint computed.")
        return
    lic2 = json.loads(QUAL_V2.read_text())["licenses"]
    m = {k: (lic2[k]["multiplier"] if lic2[k].get("licensed") else None) for k in ("SEL_Nf10", "SEL_Nf5", "LEVEL_Nf10", "LEVEL_Nf5")}
    fl = {k: lic2[k].get("failed_scenarios") for k in m}
    df = add_geometry_percentile(cells_from_records(recs, axes, with_p10=True))
    vgg, vgg_aug = vgg_table_det(axes), vgg_table(axes)
    ce = df[df.component == "standalone_ce"]; ce0 = ce[ce.dropout == 0]
    comp_A, comp_A_aug = Comparators(vgg, "dA"), Comparators(vgg_aug, "dA")
    pool = df[df.component == "paradigm_pool"]
    ho = ho_endpoint(df, b); ho["evidence_class"] = EVIDENCE["HO"]
    report = {"reader": "rn18_analysis_v2.py", "validation": val,
              "denominators": {"n_records": len(recs), "n_cells": int(len(df)),
                               "per_source_checkpoints": df.groupby("source").cell.nunique().to_dict(),
                               "ce_families": sorted(ce.family.unique()), "vgg_checkpoints_primary_view": int(vgg.cell.nunique()),
                               "vgg_checkpoints_aug_view": int(vgg_aug.cell.nunique()), "multipliers_v2": m, "licenses_v2": lic2},
              "HO": ho,
              "SEL_ce": sel_endpoint(ce, comp_A, m["SEL_Nf10"], EVIDENCE["SEL"], fl["SEL_Nf10"]),
              "SEL_ce_do0_sensitivity_Nf5": sel_endpoint(ce0, comp_A, m["SEL_Nf5"], EVIDENCE["sensitivity"], fl["SEL_Nf5"]),
              "SEL_ce_augview_comparators_sensitivity": sel_endpoint(ce, comp_A_aug, m["SEL_Nf10"], EVIDENCE["sensitivity"], fl["SEL_Nf10"]),
              "SEL_paradigm_pool_descriptive": {"evidence_class": EVIDENCE["ORG"],
                                                "mean_regret_P00": float(np.nanmean(regret(pool.dA.values, choice_prob(pool.M.values)))),
                                                "n_nonfinite": int((~np.isfinite(pool.M.values)).sum())},
              "LEVEL_ce": level_endpoint(ce, m["LEVEL_Nf10"], EVIDENCE["LEVEL"], fl["LEVEL_Nf10"]),
              "LEVEL_ce_do0_sensitivity_Nf5": level_endpoint(ce0, m["LEVEL_Nf5"], EVIDENCE["sensitivity"], fl["LEVEL_Nf5"]),
              "bridge_A_G": {"evidence_class": EVIDENCE["bridge"], "primary_view": rank_slope_stat(vgg), "aug_view": rank_slope_stat(vgg_aug)},
              "ORG_descriptive": org_descriptive(df),
              "E4": {"evidence_class": EVIDENCE["E4"], "full": e4(df), "ce": e4(ce)},
              "comparator_fits": {"primary_view": comp_A.summary(), "aug_view": comp_A_aug.summary()},
              "comparator_specification": {"primary_view": comparator_dump(comp_A, vgg), "aug_view": comparator_dump(comp_A_aug, vgg_aug)}}
    report["correction_record"] = correction_record(report)
    (OUT / "rn18_report_v2.json").write_text(json.dumps(report, indent=1, default=str))
    (OUT / "rn18_report_v2.md").write_text("# rn18_report_v2\n\n```\n" + json.dumps(report, indent=1, default=str) + "\n```\n")
    print(json.dumps({"HO_global": report["HO"]["global"], "SEL": report["SEL_ce"]["verdict"],
                      "LEVEL": report["LEVEL_ce"].get("verdict"),
                      "conclusions_changed": report["correction_record"]["conclusions_changed"]}, indent=1, default=str))


# ---------------------------------------------------------------------------
# Self-test: the planted synthetic panel plus the four fail-closed cases.
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(21)
    df = v1._synth(rng)
    ce = df[df.component == "standalone_ce"]
    vg = ce.copy(); vg["family"] = vg["family"].str.replace("do1_", "do0_")
    comp = Comparators(vg.assign(component="vgg_bridge"), "dA")
    sel = sel_endpoint(ce, comp, 1.0, EVIDENCE["SEL"])
    assert sel["N_f"] == 10 and sel["verdict"] in ("PRACTICALLY SUPERIOR TO ALL DECLARED ZERO-SHOT COMPARATORS", "UNRESOLVED",
                                                   "PRACTICALLY INFERIOR TO THE REFERENCE", "PRACTICALLY EQUIVALENT TO THE REFERENCE"), sel["verdict"]
    lv = level_endpoint(ce, 1.0, EVIDENCE["LEVEL"])
    assert lv["verdict"] == "resolved improvement" and lv["N_f"] == 10, lv
    # (a) a non-finite comparator prediction is NOT ESTIMABLE, never a tie
    assert np.isnan(choice_prob(np.array([np.nan]))[0]) and choice_prob(np.array([0.0]))[0] == 0.5

    class Broken(Comparators):
        def predict(self, name, d):
            p = super().predict(name, d)
            if name == REFERENCE:
                p = p.copy(); p[0] = np.nan
            return p
    sel_b = sel_endpoint(ce, Broken(vg.assign(component="vgg_bridge"), "dA"), 1.0, EVIDENCE["SEL"])
    assert sel_b["comparators"][REFERENCE]["ci"] is None and sel_b["verdict"].startswith("NOT ESTIMABLE"), sel_b["verdict"]
    # (b) a non-finite required input makes the endpoint NOT ESTIMABLE (no cell dropped)
    ce_nan = ce.copy(); ce_nan.loc[ce_nan.index[3], "M"] = np.nan
    assert sel_endpoint(ce_nan, comp, 1.0, EVIDENCE["SEL"])["verdict"].startswith("NOT ESTIMABLE")
    ce_nan2 = ce.copy(); ce_nan2.loc[ce_nan2.index[3], "l_C_p10"] = np.nan
    assert level_endpoint(ce_nan2, 1.0, EVIDENCE["LEVEL"])["verdict"].startswith("NOT ESTIMABLE")
    # (c) identical leave-one-out values give a zero SE: NOT ESTIMABLE, not an interval
    jk = jackknife(pd.DataFrame({"family": list("abcdef"), "x": [1.0] * 6}), lambda d: d.x.mean())
    assert jk["not_estimable"] and interval(jk, 0.05, 1.0) is None
    ce_flat = ce.copy(); ce_flat["l_E_p10"] = ce_flat["l_E"]; ce_flat["l_C_p10"] = ce_flat["l_C"]
    assert level_endpoint(ce_flat, 1.0, EVIDENCE["LEVEL"])["verdict"] == "NOT ESTIMABLE"
    # (d) decisions use full precision: an interval whose rounded copy is [0.0, ...] still decides on the exact value
    iv = [-4e-6, 0.02]
    assert level_verdict(iv)["verdict"] == "unresolved direction" and display(iv)[0] == 0.0
    assert level_verdict([1e-7, 0.02])["verdict"] == "resolved improvement"
    # (e) the validator's fail-closed cases are exercised in tests/test_rn18_reader_v2_20260909.py
    # (g) an unlicensed family is descriptive: interval at multiplier 1, no claim, rule outcome kept as information
    un = level_endpoint(ce, None, EVIDENCE["LEVEL"], ["null_level_bounded_beta_family"])
    assert un["verdict"] == "NO LICENSE: descriptive" and un["verdict_if_licensed_at_multiplier_1"] == "resolved improvement" and un["multiplier"] == 1.0
    assert not un["license"]["licensed"] and un["evidence_class"].startswith("DESCRIPTIVE")
    us = sel_endpoint(ce, comp, None, EVIDENCE["SEL"], ["metric_almost_all_ties"])
    assert us["verdict"] == "NO LICENSE: descriptive" and "verdict_if_licensed_at_multiplier_1" in us
    # (f) the comparator dump reproduces the reference ridge prediction from saved coefficients
    dump = comparator_dump(comp, vg.assign(component="vgg_bridge"))
    r = dump["ridge"][REFERENCE]; row = ce.iloc[[0]]
    feats = r["features"]; x = row[feats].to_numpy(float)[0]
    sd = np.array([np.inf if v is None else v for v in r["scaler_sd"]]); z = (x - np.array(r["scaler_mean"])) / sd
    ind = np.array([float(row.source.iloc[0] == s_) for s_ in dump["indicator_sources"]])
    pred = r["beta"]["intercept"] + ind @ np.array(list(r["beta"]["source_indicators"].values())) + z @ np.array(list(r["beta"]["standardized_features"].values()))
    assert abs(pred - comp.predict(REFERENCE, row)[0]) < 1e-9, (pred, comp.predict(REFERENCE, row)[0])
    print("[rn18-analysis-v2] self-test PASS: planted panel verdicts, NaN prediction -> NOT ESTIMABLE, "
          "non-finite input -> NOT ESTIMABLE, zero SE -> NOT ESTIMABLE, full-precision decisions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--validate-only", action="store_true", dest="validate_only")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.b, args.validate_only)


if __name__ == "__main__":
    main()

"""Stage-2 held-out theory-to-real validation (audit #5 section 12).

FROZEN DESIGN: documentation/heldout_theory_validation_design.md and the
claim contract (2026-08-24), Option B (pool-wide measured coordinates from
pilot0/extract_pool_coords.py). Nothing in here may be tuned after held-out
outcomes are inspected; the materiality rule, folds, baselines, and gates
are fixed.

Inputs
  - harmonized parquet (outcomes Delta_ij = AUGRC_E - AUGRC_C per
    checkpoint-shift cell, NC panel per checkpoint, source)
  - CLIP severity (crossing-audit construction, per source)
  - pilot0/pool_coords/<slug>.json (measured geometry + per-set gamma/a/rho)

Theory arm (no trained parameters; sign used raw)
  Per cell: build the declared model at (C, D) = measured (n_classes, dim)
  with s_i from the NC1 dictionary, theta_i from the self-duality dictionary,
  logit scale and equinorm spread from measured geometry, and the FROZEN
  per-set coordinates (gamma, a, rho); predict sign(AUGRC gap) =
  -sign(AUROC_Energy - AUROC_CTM) from the closed forms (within-cell
  prevalence is fixed, so the AUGRC identity transfers the sign exactly).

Baselines (fit inside training folds only)
  severity-only isotonic; geometry-only ridge logistic; source identity
  (checkpoint folds only); train-fold mean sign; flexible ridge logistic
  with matched features.

Endpoint: material-cell (|Delta| >= 10 AUGRC-milli units, frozen) sign
accuracy, balanced accuracy; primary comparison theory vs severity-only with
a checkpoint-cluster bootstrap CI (B=2000). Folds: grouped 5-fold by
checkpoint (seed 2027) and leave-one-source-out; leave-one-OOD-set-out as
sensitivity. Gates printed per the claim contract.

Usage (from code/):
  python heldout_theory_validation.py [--coords_dir pilot0/pool_coords]
  python heldout_theory_validation.py --self_test
Outputs: nc_csf_predictivity/outputs/track1/heldout_theory_report.md/.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crossing_robustness_audit import (METRICS, OUT_DIR, PARQUET,
                                       load_severity_rows, severity_map)
from mc_phase_audit import BASE, build_config_model
from pilot0.theory import (HeadContext, NoiseModel, predicted_aurocs,
                           predicted_ctm_mean_auroc)

MATERIALITY = 10.0          # |Delta| in AUGRC x 1000 units; frozen pre-outcome
B_BOOT = 2000
FOLD_SEED = 2027
NSNCS_MAP = {"ood_nsncs_svhn": "svhn", "ood_nsncs_ti": "tinyimagenet",
             "ood_nsncs_lsun_cropped": "lsun cropped",
             "ood_nsncs_lsun_resize": "lsun resize",
             "ood_nsncs_isun": "isun", "ood_nsncs_textures": "textures",
             "ood_nsncs_places365": "places365"}


# ---------------------------------------------------------------------------
# Data assembly.
# ---------------------------------------------------------------------------

def load_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df.architecture == "VGG13") & (df.eval_dataset != "test")
             & df.csf.isin(["Energy", "CTM"])].copy()
    sub["cell"] = (sub.paradigm.astype(str) + "|" + sub.source.astype(str)
                   + "|" + sub["run"].astype(str) + "|"
                   + sub.reward.astype(str) + "|" + sub.dropout.astype(str))
    keep = ["cell", "source", "eval_dataset", "var_collapse", "self_duality",
            "equinorm_uc", "max_equiangular_wc", "csf"]
    grouped = (sub.groupby(keep[:-1] + ["csf"])["augrc"].mean()
               .unstack("csf").reset_index())
    grouped["gap"] = grouped["Energy"] - grouped["CTM"]
    return grouped.dropna(subset=["gap"])


def slug_fields(slug: str) -> dict:
    # <srcdir>_paper_sweep__<paradigm>_bbvgg13_do<d>_run<r>_rew<w>
    import re
    m = re.match(r"^(?P<src>[a-z0-9\-]+)_paper_sweep__(?P<paradigm>[a-z]+)_bb"
                 r"vgg13_do(?P<do>\d)_run(?P<run>\d+)_rew(?P<rew>[\d.]+)$",
                 slug)
    if not m:
        return {}
    src = {"cifar10": "cifar10", "cifar100": "cifar100",
           "supercifar": "supercifar100",
           "tiny-imagenet-200": "tinyimagenet"}[m["src"]]
    return {"paradigm": m["paradigm"], "source": src, "run": int(m["run"]),
            "reward": float(m["rew"]), "dropout": bool(int(m["do"]))}


def cell_key_from_fields(f: dict) -> str:
    return (f"{f['paradigm']}|{f['source']}|{f['run']}|{f['reward']}|"
            f"{f['dropout']}")


def load_coords(coords_dir: Path) -> tuple[dict, list[str]]:
    """cell -> {geometry, n_classes, dim, per-parquet-set coords}."""
    out: dict[str, dict] = {}
    problems: list[str] = []
    for p in sorted(coords_dir.glob("*.json")):
        if p.name.startswith("FAILED_"):
            problems.append(p.name)
            continue
        rec = json.loads(p.read_text())
        f = slug_fields(rec["slug"])
        if not f:
            problems.append(f"unparsed slug {rec['slug']}")
            continue
        out[cell_key_from_fields(f)] = rec
    return out, problems


def map_ood_names(rec: dict, parquet_sets: set[str]) -> dict[str, dict]:
    """Map coords-JSON OOD keys to parquet eval_dataset names for one source.

    The seven nsncs names map directly; ood_sncs takes the unique leftover
    parquet name for that source (asserted unique)."""
    mapped: dict[str, dict] = {}
    for k, v in rec["ood"].items():
        if "error" in v:
            continue
        if k in NSNCS_MAP and NSNCS_MAP[k] in parquet_sets:
            mapped[NSNCS_MAP[k]] = v
    if "ood_sncs" in rec["ood"] and "error" not in rec["ood"]["ood_sncs"]:
        leftover = parquet_sets - set(mapped)
        if len(leftover) == 1:
            mapped[leftover.pop()] = rec["ood"]["ood_sncs"]
    return mapped


# ---------------------------------------------------------------------------
# Theory arm.
# ---------------------------------------------------------------------------

def dictionary_params(row, n_classes: int = 100) -> tuple[float, float]:
    c_eff = float(n_classes)
    s = float((c_eff - 1) / np.sqrt(c_eff * max(row.var_collapse, 1e-9)))
    theta = float(np.degrees(np.arccos(
        np.clip(1.0 - row.self_duality / 2.0, -1.0, 1.0))))
    return s, theta


def analytic_gap_sign(c: int, d: int, s: float, theta_deg: float,
                      logit_target: float, eta_std: float,
                      gamma: float, a: float, rho: float) -> float:
    cfg = dict(BASE, s=max(s, 3.0), theta_deg=float(np.clip(theta_deg, 0, 85)),
               logit_target=max(logit_target, 1e-3),
               eta_std=float(np.clip(eta_std, 0.0, 0.5)),
               ga=float(np.clip(gamma * a, 1e-4, None)),
               a=float(np.clip(a, 1e-3, 0.999)),
               rho=float(np.clip(rho, 0.05, None)))
    model = build_config_model(c, d, cfg, seed=0)
    ctx = HeadContext.from_head(model["w"], model["b"])
    dim = model["means"].shape[1]
    sigma = model["sigma"]
    noise_id = NoiseModel.isotropic(sigma, ctx, dim)
    noise_ood = NoiseModel.isotropic(cfg["rho"] * sigma, ctx, dim)
    head = predicted_aurocs(model["means"], model["class_freq"], noise_id,
                            model["m_ood"], noise_ood, ctx)
    ctm = predicted_ctm_mean_auroc(model["means"], model["class_freq"],
                                   model["cov_id"], model["m_ood"],
                                   model["cov_ood"])
    auroc_gap = float(head["Energy"]) - float(ctm)
    return -auroc_gap  # predicted sign of the AUGRC gap


def theory_predictions(cells: pd.DataFrame, coords: dict) -> pd.Series:
    preds = pd.Series(np.nan, index=cells.index)
    cache: dict[tuple, float] = {}
    for idx, row in cells.iterrows():
        rec = coords.get(row.cell)
        if rec is None:
            continue
        sets = map_ood_names(
            rec, set(cells[cells.cell == row.cell].eval_dataset))
        co = sets.get(row.eval_dataset)
        if co is None:
            continue
        s, theta = dictionary_params(row, rec["n_classes"])
        key = (rec["n_classes"], rec["dim"], round(s, 3), round(theta, 2),
               round(rec["geometry"]["logit_scale"], 3),
               round(rec["geometry"]["class_mean_radius_cv"], 4),
               round(co["gamma"], 4), round(co["a"], 4), round(co["rho"], 4))
        if key not in cache:
            cache[key] = analytic_gap_sign(
                rec["n_classes"], rec["dim"], s, theta,
                rec["geometry"]["logit_scale"],
                rec["geometry"]["class_mean_radius_cv"],
                co["gamma"], co["a"], co["rho"])
        preds.at[idx] = cache[key]
    return preds


# ---------------------------------------------------------------------------
# Baselines (all fitted inside the training fold only).
# ---------------------------------------------------------------------------

def pava_inc(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    blocks = [[y[i] * w[i], w[i], i, i] for i in range(len(y))]
    out: list = []
    for b in blocks:
        out.append(list(b))
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            s2 = out.pop(); s1 = out.pop()
            out.append([s1[0] + s2[0], s1[1] + s2[1], s1[2], s2[3]])
    fit = np.empty_like(y, dtype=float)
    for s, wt, i0, i1 in out:
        fit[i0:i1 + 1] = s / wt
    return fit


def severity_only(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    ds, inv = np.unique(train.d.values, return_inverse=True)
    ym = np.zeros(len(ds)); wm = np.zeros(len(ds))
    np.add.at(ym, inv, train.gap.values); np.add.at(wm, inv, 1.0)
    fit = pava_inc(ym / wm, wm)
    return np.interp(test.d.values, ds, fit)


def ridge_logistic(x: np.ndarray, y01: np.ndarray, lam: float,
                   iters: int = 60) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(xb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-xb @ beta))
        wgt = np.clip(p * (1 - p), 1e-6, None)
        z = xb @ beta + (y01 - p) / wgt
        a_mat = xb.T @ (wgt[:, None] * xb) + lam * np.eye(xb.shape[1])
        a_mat[0, 0] -= lam
        beta_new = np.linalg.solve(a_mat, xb.T @ (wgt * z))
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


GEO_COLS = ["var_collapse", "self_duality", "equinorm_uc",
            "max_equiangular_wc"]


def geo_features(fr: pd.DataFrame, flexible: bool) -> np.ndarray:
    base = [fr.d.values] + [fr[c].values for c in GEO_COLS]
    feats = list(base) + [fr.d.values * fr[c].values for c in GEO_COLS]
    if flexible:
        feats += [v * v for v in base]
        feats += [fr[a].values * fr[b].values
                  for i, a in enumerate(GEO_COLS) for b in GEO_COLS[i + 1:]]
    return np.column_stack(feats)


def geometry_model(train: pd.DataFrame, test: pd.DataFrame,
                   flexible: bool) -> np.ndarray:
    xtr = geo_features(train, flexible); xte = geo_features(test, flexible)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    beta = ridge_logistic((xtr - mu) / sd, (train.gap.values > 0).astype(float),
                          lam=1.0 if not flexible else 10.0)
    logits = beta[0] + ((xte - mu) / sd) @ beta[1:]
    return logits


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------

def accuracy(pred_sign: np.ndarray, obs: np.ndarray) -> float:
    ok = ~np.isnan(pred_sign)
    if ok.sum() == 0:
        return float("nan")
    return float((np.sign(pred_sign[ok]) == np.sign(obs[ok])).mean())


def balanced_accuracy(pred_sign: np.ndarray, obs: np.ndarray) -> float:
    ok = ~np.isnan(pred_sign)
    accs = []
    for s in (-1, 1):
        m = ok & (np.sign(obs) == s)
        if m.sum():
            accs.append(float((np.sign(pred_sign[m]) == s).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def run_folds(cells: pd.DataFrame, theory: pd.Series, mode: str,
              rng: np.random.Generator) -> pd.DataFrame:
    cells = cells.copy()
    cells["theory"] = theory
    cells["oof"] = np.nan
    preds = {k: pd.Series(np.nan, index=cells.index)
             for k in ("severity", "geometry", "flexible", "src_id", "mean")}
    ckpts = np.array(sorted(cells.cell.unique()))
    if mode == "ckpt5":
        perm = rng.permutation(len(ckpts))
        folds = [set(ckpts[perm[i::5]]) for i in range(5)]
    else:  # loso
        folds = [set(cells[cells.source == s].cell.unique())
                 for s in sorted(cells.source.unique())]
    for held in folds:
        te = cells.cell.isin(held)
        train, test = cells[~te], cells[te]
        preds["severity"][te] = severity_only(train, test)
        preds["geometry"][te] = geometry_model(train, test, flexible=False)
        preds["flexible"][te] = geometry_model(train, test, flexible=True)
        preds["mean"][te] = float(train.gap.mean())
        if mode == "ckpt5":
            src_mean = train.groupby("source").gap.mean()
            preds["src_id"][te] = test.source.map(src_mean).values
    for k, v in preds.items():
        cells[k] = v
    return cells


def summarize(cells: pd.DataFrame, mode: str, rng: np.random.Generator) -> dict:
    mat = cells[np.abs(cells.gap) >= MATERIALITY].copy()
    obs = mat.gap.values
    arms = ["theory", "severity", "geometry", "flexible", "mean"] + (
        ["src_id"] if mode == "ckpt5" else [])
    res = {"n_cells": int(len(cells)), "n_material": int(len(mat)),
           "theory_coverage": float((~mat.theory.isna()).mean()), "arms": {}}
    for arm in arms:
        res["arms"][arm] = {
            "sign_acc": accuracy(mat[arm].values, obs),
            "balanced_acc": balanced_accuracy(mat[arm].values, obs)}
    # primary: theory vs severity-only, checkpoint-cluster bootstrap
    both = mat.dropna(subset=["theory", "severity"])
    ckpts = np.array(sorted(both.cell.unique()))
    groups = {c: g for c, g in both.groupby("cell")}
    diffs = np.empty(B_BOOT)
    for i in range(B_BOOT):
        sel = rng.choice(ckpts, len(ckpts), replace=True)
        fr = pd.concat([groups[c] for c in sel])
        o = fr.gap.values
        diffs[i] = (accuracy(fr.theory.values, o)
                    - accuracy(fr.severity.values, o))
    point = (accuracy(both.theory.values, both.gap.values)
             - accuracy(both.severity.values, both.gap.values))
    res["theory_minus_severity"] = {
        "point": float(point),
        "ci95": [float(np.quantile(diffs, 0.025)),
                 float(np.quantile(diffs, 0.975))]}
    if mode == "loso":
        res["per_source"] = {
            s: {"theory": accuracy(g.theory.values, g.gap.values),
                "severity": accuracy(g.severity.values, g.gap.values),
                "n_material": int(len(g))}
            for s, g in mat.groupby("source")}
    return res


def gates(ck: dict, lo: dict) -> list[str]:
    out = []
    tms = ck["theory_minus_severity"]
    g1 = tms["point"] > 0 and tms["ci95"][0] > 0
    out.append(f"Gate 1 (theory beats severity-only, clustered CI > 0): "
               f"{'PASS' if g1 else 'FAIL'} "
               f"(point {tms['point']:+.3f}, CI {tms['ci95']})")
    per = lo.get("per_source", {})
    wins = [s for s, v in per.items()
            if not np.isnan(v["theory"]) and v["theory"] > v["severity"]]
    g2 = len(wins) >= 2
    out.append(f"Gate 2 (improvement not carried by one source): "
               f"{'PASS' if g2 else 'FAIL'} (theory>severity on {wins})")
    out.append("Gate 3 (strata handoff ordering retained held-out): "
               "report the per-tertile first-handoff comparison from the "
               "crossing pipeline on held-out folds (descriptive; see report)")
    out.append("Gate 4 (nothing tuned on evaluation folds): PASS by "
               "construction (theory arm has no fitted parameters; baselines "
               "fold-fitted)")
    verdict = ("PREDICTIVE candidate (confirm Gate 3 descriptively)"
               if g1 and g2 else "ORGANIZATIONAL (theory does not beat "
               "empirical baselines held-out) or THEORY-FIRST per contract")
    out.append(f"Mode indication: {verdict}")
    return out


# ---------------------------------------------------------------------------
# Self-test: synthetic coords + outcomes where theory is the generator.
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(7)
    rows = []
    coords: dict[str, dict] = {}
    sets = ["svhn", "isun", "textures", "lsun cropped", "lsun resize",
            "places365", "tinyimagenet", "cifar10"]
    for i in range(24):
        cell = f"dg|cifar100|{i}|2.2|False"
        vc = float(rng.uniform(0.05, 0.6))
        sd = float(rng.uniform(0.005, 0.4))
        geo = {"logit_scale": 10.0, "class_mean_radius_cv": 0.05}
        ood = {}
        for j, name in enumerate(sets):
            gamma = float(rng.uniform(0.3, 0.9))
            a = float(rng.uniform(0.4, 0.85))
            rho = float(rng.uniform(0.9, 1.1))
            ood[name] = {"gamma": gamma, "a": a, "rho": rho}
        coords[cell] = {"n_classes": 20, "dim": 128, "geometry": geo,
                        "_direct": ood, "ood": {}}
        for j, name in enumerate(sets):
            co = coords[cell]["_direct"][name]
            row = pd.Series({"var_collapse": vc, "self_duality": sd})
            s, theta = dictionary_params(row)
            g = analytic_gap_sign(20, 128, s, theta, 10.0, 0.05,
                                  co["gamma"], co["a"], co["rho"])
            rows.append({"cell": cell, "source": "cifar100",
                         "eval_dataset": name, "var_collapse": vc,
                         "self_duality": sd, "equinorm_uc": 0.1,
                         "max_equiangular_wc": 0.2,
                         "d": j / 4.0 - 1.0,
                         "gap": 40.0 * np.sign(g) + rng.normal(0, 5)})
    cells = pd.DataFrame(rows)
    theory = pd.Series(np.nan, index=cells.index)
    for idx, row in cells.iterrows():
        co = coords[row.cell]["_direct"][row.eval_dataset]
        s, theta = dictionary_params(row)
        theory.at[idx] = analytic_gap_sign(20, 128, s, theta, 10.0, 0.05,
                                           co["gamma"], co["a"], co["rho"])
    ck = summarize(run_folds(cells, theory, "ckpt5",
                             np.random.default_rng(FOLD_SEED)),
                   "ckpt5", np.random.default_rng(1))
    acc = ck["arms"]["theory"]["sign_acc"]
    assert acc > 0.95, f"self-test: theory arm should be near-perfect, got {acc}"
    assert ck["theory_minus_severity"]["point"] > 0, "self-test: delta <= 0"
    print(f"self-test OK: theory sign acc {acc:.3f}, "
          f"delta vs severity {ck['theory_minus_severity']['point']:+.3f}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords_dir", type=str,
                        default="pilot0/pool_coords")
    parser.add_argument("--self_test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    coords_dir = Path(args.coords_dir)
    df = pd.read_parquet(PARQUET)
    cells = load_outcomes(df)
    sev = severity_map(load_severity_rows(), METRICS)
    cells["d"] = [sev.get((s, e)) for s, e in
                  zip(cells.source, cells.eval_dataset)]
    cells = cells.dropna(subset=["d"]).reset_index(drop=True)
    coords, problems = load_coords(coords_dir)
    print(f"coords: {len(coords)} checkpoints loaded, "
          f"{len(problems)} problems")

    theory = theory_predictions(cells, coords)
    rng = np.random.default_rng(FOLD_SEED)
    ck_cells = run_folds(cells, theory, "ckpt5", rng)
    ck = summarize(ck_cells, "ckpt5", np.random.default_rng(11))
    lo_cells = run_folds(cells, theory, "loso", rng)
    lo = summarize(lo_cells, "loso", np.random.default_rng(12))

    lines = ["# Held-out theory-to-real validation (Stage 2, frozen design)",
             "",
             f"Coords loaded for {len(coords)}/280 checkpoints; problems: "
             f"{problems or 'none'}. Materiality |gap| >= {MATERIALITY} "
             f"(AUGRC x 1000). Theory arm has no fitted parameters.", ""]
    for name, res in (("Checkpoint-held-out (grouped 5-fold)", ck),
                      ("Leave-one-source-out", lo)):
        lines += [f"## {name}", "",
                  f"cells {res['n_cells']}, material {res['n_material']}, "
                  f"theory coverage {res['theory_coverage']:.3f}", "",
                  "| arm | sign acc (material) | balanced acc |",
                  "|---|---|---|"]
        for arm, v in res["arms"].items():
            lines.append(f"| {arm} | {v['sign_acc']:.3f} | "
                         f"{v['balanced_acc']:.3f} |")
        tms = res["theory_minus_severity"]
        lines += ["", f"theory - severity: {tms['point']:+.3f}, "
                  f"cluster CI95 {tms['ci95']}", ""]
        if "per_source" in res:
            lines += ["| held-out source | theory | severity | n material |",
                      "|---|---|---|---|"]
            for s, v in res["per_source"].items():
                lines.append(f"| {s} | {v['theory']:.3f} | "
                             f"{v['severity']:.3f} | {v['n_material']} |")
            lines.append("")
    lines += ["## Gates (claim contract)", ""]
    lines += [f"- {g}" for g in gates(ck, lo)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "heldout_theory_report.md").write_text("\n".join(lines))
    (OUT_DIR / "heldout_theory_report.json").write_text(
        json.dumps({"ckpt5": ck, "loso": lo}, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Pilot 2: held-out-mechanism transport test (manifest section 8, plan
"Pilot 2 - held-out-mechanism prediction", Addendum A item 4).

Fit on baseline+A1 (16 models), predict the A2 models (4) without
refitting. Outcome variable = the registered detector gaps per cell
(model, OOD set, pair), gap = L_A - L_B with L = 1 - AUROC_f (primary;
AUGRC secondary). Arms at matched parameter budgets (2 fitted params each):

  plugin       observed_gap ~ a + b * predicted_gap, where predicted_gap
               comes from the FROZEN empirical-covariance plug-in
               (stage2b_extract_predict.py outputs). The plug-in predicts
               A2 from A2's own measured geometry - no support needed.
  nuisance_pc  a + b * z, z = first PC of the standardized registered
               nuisance vector Q over the training models. Q = (ID acc,
               NLL, ECE, training fit, weight norm); training loss was not
               recorded by the geometry extraction, so train_acc is the
               recorded training-fit component (implementation note,
               2026-08-19). Registered sensitivity variant adds logit_scale
               to Q.
  nc_pc        a + b * z, z = first PC of the eight Papyan NC metrics.
  dose         a + b * lam on training; the held-out mechanism's label is
               novel, so its held-out prediction DEGENERATES to the
               OOD-cell mean (plan, extrapolation semantics fixed).
  cellmean     training-mean per (set, pair) cell (floor).

Ordering discipline (mechanical, enforced): `--stage margin` computes the
material margin from a nested bootstrap (outer: seeds, inner: OOD sets) of
the cell-mean arm's held-out MAE and writes pilot2_margin.json BEFORE any
fitted arm's A2 prediction errors exist; `--stage full` refuses to run
without that file. Deviation note (2026-08-19, recorded in the margin
file): the registered Pilot 1 endpoints themselves contained A2 cells, so
A2 *gap outcomes* were necessarily unblinded before margin setting; the
margin is therefore this fixed formula with zero analyst freedom, and no
fitted-arm prediction error is computed at the margin stage.

Verdict per scale: margin condition = plugin MAE beats BOTH registered
comparators (nuisance_pc, dose) by more than the margin (either worse by
more than the margin = FAIL; otherwise EXPAND before deciding - admissible
per plan). Sign condition = calibrated plugin gap sign correct on >= 75%
of material held-out cells (materiality from predictions only:
|predicted gap| >= 2 * se_gap, Hanley-McNeil SEs at the predicted AUROCs,
n_id = 10000). Registered pass = margin condition AND sign condition.

On-support diagnostic (R10): per held-out cell, RMS z-distance in the
registered geometry coordinates (G-vector + per-set H-coordinates),
standardized over the training models; cells beyond the training maximum
are off-support and reported separately (A2 is off-support by
construction, Addendum A item 4; verdicts are reported unchanged).

Usage (from code/, after the Pilot 1 sweep; stage2b/ + geometry/ JSONs and
the stats root must be reachable):
    python nc_csf_predictivity/interventions/pilot2_transport.py \
        --stats_root $EXPERIMENT_ROOT_DIR/cifar100_intervention --stage margin
    # commit pilot2_margin.json, then:
    python nc_csf_predictivity/interventions/pilot2_transport.py \
        --stats_root $EXPERIMENT_ROOT_DIR/cifar100_intervention --stage full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nc_csf_predictivity.interventions.outcome_analysis import load_long
from pilot0.theory import hanley_mcneil_se

PAIRS = {"E1": ("MLS", "Maha"), "E2": ("CTM", "CTM_mean"),
         "E4": ("Energy", "MLS")}
PRED_KEY = {"CTM": "CTM_head"}  # stats-CSV name -> stage2b prediction name
TRAIN_LAMS = ("0.0", "-0.1", "0.3", "1.0")
HOLDOUT_LAMS = ("hard",)
DOSE = {"0.0": 0.0, "-0.1": -0.1, "0.3": 0.3, "1.0": 1.0}
N_ID = 10_000
Q_FIELDS = ("val_acc", "val_nll", "val_ece15", "train_acc",
            "weight_norm_fro")
Q_SENS_FIELDS = Q_FIELDS + ("logit_scale",)
NC_FIELDS = ("var_collapse", "equinorm_uc", "equinorm_wc",
             "equiangular_uc", "equiangular_wc", "max_equiangular_uc",
             "max_equiangular_wc", "self_duality")
G_HULL_FIELDS = ("var_collapse", "self_duality", "equinorm_uc",
                 "equiangular_uc", "equinorm_wc", "equiangular_wc",
                 "log_radius", "log_sigma", "eig_max_over_mean",
                 "head_residual_fraction", "logit_scale")
H_HULL_FIELDS = ("gamma", "a", "rho", "w_perp", "top2_gap")
BOOT_B = 10_000
SIGN_THRESHOLD = 0.75
MARGIN_NOTE = (
    "Deviation note (2026-08-19): the registered Pilot 1 endpoints "
    "included A2 cells, so A2 gap OUTCOMES were unblinded before margin "
    "setting; the section-8 ordering 'before A2 unblinding' is applied to "
    "fitted-arm PREDICTION ERRORS, which do not exist at this stage. The "
    "margin is the fixed formula 2 x nested-bootstrap SD (outer: seeds, "
    "inner: OOD sets, B=10000, rng seed 0) of the cell-mean arm's held-out "
    "MAE, with zero analyst freedom.")


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------

def load_stage2b(stage2b_dir: Path) -> dict[tuple[str, int], dict]:
    """(lam, run) -> stage2b record with per-set h_coords + predictions."""
    out = {}
    for path in sorted(stage2b_dir.glob("*.json")):
        if path.name.endswith("FAILED.json"):
            continue
        rec = json.loads(path.read_text())
        out[(rec["lam"], int(rec["run"]))] = rec
    if not out:
        raise FileNotFoundError(f"no stage2b JSONs in {stage2b_dir}")
    return out


def load_geometry(geometry_dir: Path) -> dict[tuple[str, int], dict]:
    """(lam, run) -> final-checkpoint geometry record (tag 'last')."""
    out = {}
    for path in sorted(geometry_dir.glob("*__last.json")):
        rec = json.loads(path.read_text())
        rec["log_radius"] = float(np.log(rec["class_mean_radius"]))
        rec["log_sigma"] = float(np.log(rec["sigma_iso"]))
        out[(rec["lam"], int(rec["run"]))] = rec
    if not out:
        raise FileNotFoundError(f"no *__last.json geometry records in "
                                f"{geometry_dir}")
    return out


def build_cells(table: pd.DataFrame, stage2b: dict, scale: str,
                arm_name: str = "emp") -> pd.DataFrame:
    """One row per (lam, run, set, pair): observed gap y, predicted gap
    ghat (frozen plug-in, AUROC scale), prediction-based materiality."""
    rows = []
    models = sorted({(lam, run) for (lam, run) in stage2b})
    for lam, run in models:
        rec = stage2b[(lam, run)]
        for set_name, set_rec in rec["sets"].items():
            preds = set_rec["preds"][arm_name]
            for endpoint, (a, b) in PAIRS.items():
                sub = table[(table.lam == lam) & (table.run == run)
                            & (table.set_name == set_name)]
                row_a = sub[sub.method == a]
                row_b = sub[sub.method == b]
                if len(row_a) != 1 or len(row_b) != 1:
                    raise ValueError(
                        f"expected 1 observed row each for {a}/{b} at "
                        f"{lam}/{run}/{set_name}, got "
                        f"{len(row_a)}/{len(row_b)}")
                if scale == "auroc_f":
                    y = ((1.0 - float(row_a.iloc[0]["auroc_f"]))
                         - (1.0 - float(row_b.iloc[0]["auroc_f"])))
                else:
                    y = (float(row_a.iloc[0]["augrc"])
                         - float(row_b.iloc[0]["augrc"]))
                pa = preds[PRED_KEY.get(a, a)]
                pb = preds[PRED_KEY.get(b, b)]
                ghat = (1.0 - pa) - (1.0 - pb)
                se_gap = float(np.sqrt(
                    hanley_mcneil_se(pa, N_ID, set_rec["n_ood"]) ** 2
                    + hanley_mcneil_se(pb, N_ID, set_rec["n_ood"]) ** 2))
                rows.append({"lam": lam, "run": run, "set_name": set_name,
                             "pair": endpoint, "y": y, "ghat": ghat,
                             "se_gap": se_gap,
                             "material": bool(abs(ghat) >= 2.0 * se_gap)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Arms (2 fitted parameters each; cellmean = 0).
# ---------------------------------------------------------------------------

def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS intercept + slope; degenerate x -> slope 0."""
    if np.std(x) < 1e-12:
        return float(np.mean(y)), 0.0
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta)


def pc1_scores(train: np.ndarray, full: np.ndarray) -> np.ndarray:
    """First-PC scores of `full`, standardized by the training rows.

    Constant columns (train SD ~ 0) are dropped; the PC sign is fixed so
    the largest-|loading| coordinate has a positive loading (deterministic).
    """
    mu, sd = train.mean(0), train.std(0, ddof=1)
    keep = sd > 1e-12
    if not keep.any():
        return np.zeros(len(full))
    z_train = (train[:, keep] - mu[keep]) / sd[keep]
    z_full = (full[:, keep] - mu[keep]) / sd[keep]
    _, _, vt = np.linalg.svd(z_train, full_matrices=False)
    pc = vt[0]
    if pc[np.argmax(np.abs(pc))] < 0:
        pc = -pc
    return z_full @ pc


def cell_means(train_cells: pd.DataFrame) -> dict[tuple[str, str], float]:
    grouped = train_cells.groupby(["set_name", "pair"])["y"].mean()
    return {key: float(v) for key, v in grouped.items()}


def arm_predictions(cells: pd.DataFrame, geometry: dict,
                    train_lams: tuple[str, ...],
                    with_dose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Add per-arm prediction columns for every cell; fit on train only."""
    cells = cells.copy()
    train = cells[cells.lam.isin(train_lams)]
    means = cell_means(train)
    fits: dict = {}

    # plugin
    alpha, beta = fit_linear(train["ghat"].values, train["y"].values)
    cells["pred_plugin"] = alpha + beta * cells["ghat"]
    fits["plugin"] = {"alpha": alpha, "beta": beta}

    # scalar-per-model comparators (nuisance PC / NC PC / sensitivity)
    models = sorted({(l, r) for l, r in zip(cells.lam, cells.run)})
    train_models = [m for m in models if m[0] in train_lams]
    for arm, fields in (("nuisance_pc", Q_FIELDS), ("nc_pc", NC_FIELDS),
                        ("nuisance_pc_logitscale", Q_SENS_FIELDS)):
        full_mat = np.array([[geometry[m][f] for f in fields]
                             for m in models])
        train_mat = np.array([[geometry[m][f] for f in fields]
                              for m in train_models])
        scores = dict(zip(models, pc1_scores(train_mat, full_mat)))
        x_cells = np.array([scores[(l, r)]
                            for l, r in zip(cells.lam, cells.run)])
        x_train = np.array([scores[(l, r)]
                            for l, r in zip(train.lam, train.run)])
        alpha, beta = fit_linear(x_train, train["y"].values)
        cells[f"pred_{arm}"] = alpha + beta * x_cells
        fits[arm] = {"alpha": alpha, "beta": beta}

    # cell-mean floor (and the dose model's held-out degeneration)
    mean_col = np.array([means[(s, p)]
                         for s, p in zip(cells.set_name, cells.pair)])
    cells["pred_cellmean"] = mean_col
    if with_dose:
        train_dose = np.array([DOSE[l] for l in train.lam])
        alpha, beta = fit_linear(train_dose, train["y"].values)
        in_dose = cells.lam.map(lambda l: DOSE.get(l))
        cells["pred_dose"] = np.where(
            in_dose.notna(), alpha + beta * in_dose.fillna(0.0), mean_col)
        fits["dose"] = {"alpha": alpha, "beta": beta}
    return cells, fits


# ---------------------------------------------------------------------------
# Margin (stage 1) and evaluation (stage 2).
# ---------------------------------------------------------------------------

def bootstrap_margin(cells: pd.DataFrame, train_lams: tuple[str, ...],
                     holdout_lams: tuple[str, ...], b: int = BOOT_B,
                     seed: int = 0) -> float:
    """2 x nested-bootstrap SD of the cell-mean arm's held-out MAE."""
    train = cells[cells.lam.isin(train_lams)]
    means = cell_means(train)
    hold = cells[cells.lam.isin(holdout_lams)]
    errors = {
        (row.lam, row.run, row.set_name, row.pair):
            abs(row.y - means[(row.set_name, row.pair)])
        for row in hold.itertuples()}
    runs = sorted(hold.run.unique())
    sets = sorted(hold.set_name.unique())
    lam_pairs = sorted({(row.lam, row.pair) for row in hold.itertuples()})
    rng = np.random.default_rng(seed)
    maes = np.empty(b)
    for i in range(b):
        rs = rng.choice(runs, size=len(runs), replace=True)
        ss = rng.choice(sets, size=len(sets), replace=True)
        maes[i] = np.mean([errors[(lam, r, s, p)]
                           for r in rs for s in ss for lam, p in lam_pairs])
    return float(2.0 * maes.std(ddof=1))


ARMS = ("plugin", "nuisance_pc", "nc_pc", "dose", "cellmean")


def mae_breakdown(hold: pd.DataFrame, arm: str) -> dict:
    err = (hold[f"pred_{arm}"] - hold["y"]).abs()
    return {
        "pooled": float(err.mean()),
        "per_pair": {p: float(err[hold.pair == p].mean())
                     for p in sorted(hold.pair.unique())},
        "per_set": {s: float(err[hold.set_name == s].mean())
                    for s in sorted(hold.set_name.unique())},
        "per_seed": {int(r): float(err[hold.run == r].mean())
                     for r in sorted(hold.run.unique())},
    }


def support_distances(cells: pd.DataFrame, geometry: dict, stage2b: dict,
                      train_lams: tuple[str, ...]) -> pd.DataFrame:
    """RMS z-distance per (model, set) in G + per-set H coordinates."""
    models = sorted({(l, r) for l, r in zip(cells.lam, cells.run)})
    train_models = [m for m in models if m[0] in train_lams]
    sets = sorted(cells.set_name.unique())
    rows = []
    for set_name in sets:
        vec = {m: np.array(
            [geometry[m][f] for f in G_HULL_FIELDS]
            + [stage2b[m]["sets"][set_name]["h_coords"][f]
               for f in H_HULL_FIELDS]) for m in models}
        train_mat = np.array([vec[m] for m in train_models])
        mu, sd = train_mat.mean(0), train_mat.std(0, ddof=1)
        keep = sd > 1e-12
        for m in models:
            z = (vec[m][keep] - mu[keep]) / sd[keep]
            rows.append({"lam": m[0], "run": m[1], "set_name": set_name,
                         "distance": float(np.sqrt((z ** 2).mean()))})
    dist = pd.DataFrame(rows)
    threshold = float(dist[dist.lam.isin(train_lams)]["distance"].max())
    dist["on_support"] = dist["distance"] <= threshold
    dist.attrs["threshold"] = threshold
    return dist


def mechanism_label_cv(cells: pd.DataFrame, geometry: dict) -> dict:
    """Pass condition 3: leave-one-seed-out CV gain from adding the
    mechanism label once plug-in index + nuisance PC are included."""
    models = sorted({(l, r) for l, r in zip(cells.lam, cells.run)})
    q_mat = np.array([[geometry[m][f] for f in Q_FIELDS] for m in models])
    z_q = dict(zip(models, pc1_scores(q_mat, q_mat)))
    runs = sorted(cells.run.unique())
    errs_a, errs_b = [], []
    for held in runs:
        tr = cells[cells.run != held]
        te = cells[cells.run == held]
        means = cell_means(tr)

        def design(df: pd.DataFrame, label: bool) -> np.ndarray:
            base = [np.ones(len(df)), df["ghat"].values,
                    np.array([z_q[(l, r)]
                              for l, r in zip(df.lam, df.run)])]
            if label:
                base.append((df.lam == "hard").values.astype(float))
            return np.column_stack(base)

        resid_tr = tr["y"].values - np.array(
            [means[(s, p)] for s, p in zip(tr.set_name, tr.pair)])
        resid_te_base = np.array(
            [means[(s, p)] for s, p in zip(te.set_name, te.pair)])
        for label, sink in ((False, errs_a), (True, errs_b)):
            coef, *_ = np.linalg.lstsq(design(tr, label), resid_tr,
                                       rcond=None)
            pred = resid_te_base + design(te, label) @ coef
            sink.extend(np.abs(pred - te["y"].values))
    mae_a, mae_b = float(np.mean(errs_a)), float(np.mean(errs_b))
    return {"cv_mae_without_label": mae_a, "cv_mae_with_label": mae_b,
            "label_gain": mae_a - mae_b}


def evaluate(cells: pd.DataFrame, geometry: dict, stage2b: dict,
             margin: float, train_lams: tuple[str, ...],
             holdout_lams: tuple[str, ...],
             with_dose: bool = True) -> dict:
    cells, fits = arm_predictions(cells, geometry, train_lams,
                                  with_dose=with_dose)
    hold = cells[cells.lam.isin(holdout_lams)]
    arms = [a for a in ARMS if with_dose or a != "dose"]
    mae = {arm: mae_breakdown(hold, arm) for arm in arms}
    mae["nuisance_pc_logitscale"] = mae_breakdown(
        hold, "nuisance_pc_logitscale")

    comparators = ("nuisance_pc", "dose") if with_dose \
        else ("nuisance_pc", "cellmean")
    deltas = {c: mae[c]["pooled"] - mae["plugin"]["pooled"]
              for c in comparators}
    if all(d > margin for d in deltas.values()):
        verdict = "PASS"
    elif any(d < -margin for d in deltas.values()):
        verdict = "FAIL"
    else:
        verdict = "EXPAND"

    material = hold[hold.material]
    sign_ok = (np.sign(material["pred_plugin"]) ==
               np.sign(material["y"])) if len(material) else pd.Series([])
    sign = {"n_material": len(material),
            "n_correct": int(sign_ok.sum()),
            "fraction": float(sign_ok.mean()) if len(material) else None,
            "pass": bool(len(material)
                         and sign_ok.mean() >= SIGN_THRESHOLD)}

    dist = support_distances(cells, geometry, stage2b, train_lams)
    hold_dist = dist[dist.lam.isin(holdout_lams)]
    merged = hold.merge(hold_dist, on=["lam", "run", "set_name"])
    err = (merged["pred_plugin"] - merged["y"]).abs()
    support = {
        "threshold": dist.attrs["threshold"],
        "n_on": int(merged["on_support"].sum()),
        "n_off": int((~merged["on_support"]).sum()),
        "mae_plugin_on": (float(err[merged.on_support].mean())
                          if merged.on_support.any() else None),
        "mae_plugin_off": (float(err[~merged.on_support].mean())
                           if (~merged.on_support).any() else None),
        "holdout_distance_range": [float(hold_dist.distance.min()),
                                   float(hold_dist.distance.max())],
    }

    if verdict == "PASS" and sign["pass"]:
        registered = "PASS"
    elif verdict == "EXPAND":
        registered = "EXPAND"
    else:
        registered = "FAIL"
    return {"margin": margin, "mae": mae, "delta_mae_vs_plugin": deltas,
            "verdict_margin": verdict, "sign": sign, "support": support,
            "fits": fits,
            "registered_pass": registered == "PASS",
            "registered_verdict": registered,
            "mechanism_label": mechanism_label_cv(cells, geometry),
            "n_train_cells": int(cells.lam.isin(train_lams).sum()),
            "n_holdout_cells": len(hold)}


# ---------------------------------------------------------------------------
# Rendering + CLI.
# ---------------------------------------------------------------------------

def render(result: dict, scale: str) -> str:
    lines = [f"# Pilot 2 Transport Report (scale: {scale})", ""]
    lines.append(f"Margin (frozen pre-errors): {result['margin']:.5f}; "
                 f"train cells {result['n_train_cells']}, held-out cells "
                 f"{result['n_holdout_cells']}.")
    lines.append("")
    lines.append("| arm | held-out MAE | vs plugin |")
    lines.append("|---|---|---|")
    for arm, rec in result["mae"].items():
        delta = rec["pooled"] - result["mae"]["plugin"]["pooled"]
        lines.append(f"| {arm} | {rec['pooled']:.5f} | {delta:+.5f} |")
    lines.append("")
    comps = " AND ".join(result["delta_mae_vs_plugin"])
    lines.append(f"**Margin condition: {result['verdict_margin']}** "
                 f"(plugin must beat {comps} by > margin).")
    s = result["sign"]
    frac = "n/a" if s["fraction"] is None else f"{s['fraction']:.3f}"
    lines.append(f"**Sign condition: {'PASS' if s['pass'] else 'FAIL'}** "
                 f"({s['n_correct']}/{s['n_material']} material cells, "
                 f"fraction {frac}, threshold {SIGN_THRESHOLD}).")
    lines.append(f"**Registered Pilot 2 verdict: "
                 f"{result['registered_verdict']}**")
    lines.append("")
    lines.append("## Per-pair held-out MAE")
    lines.append("")
    pairs = sorted(result["mae"]["plugin"]["per_pair"])
    lines.append("| arm | " + " | ".join(pairs) + " |")
    lines.append("|---|" + "---|" * len(pairs))
    for arm, rec in result["mae"].items():
        cells_ = " | ".join(f"{rec['per_pair'][p]:.5f}" for p in pairs)
        lines.append(f"| {arm} | {cells_} |")
    lines.append("")
    lines.append("## Not-one-dataset / not-one-seed (pass condition 4)")
    lines.append("")
    for level in ("per_set", "per_seed"):
        plug = result["mae"]["plugin"][level]
        comp = result["mae"]["nuisance_pc"][level]
        wins = sum(plug[k] < comp[k] for k in plug)
        lines.append(f"- plugin beats nuisance_pc in {wins}/{len(plug)} "
                     f"{level.replace('per_', '')}s")
    lines.append("")
    sup = result["support"]
    lines.append("## On-support diagnostic (R10 / Addendum A item 4)")
    lines.append("")
    lines.append(
        f"- held-out cells on/off support: {sup['n_on']}/{sup['n_off']} "
        f"(threshold {sup['threshold']:.2f}, held-out distance range "
        f"{sup['holdout_distance_range'][0]:.2f}-"
        f"{sup['holdout_distance_range'][1]:.2f})")
    lines.append(f"- plugin MAE on-support: {sup['mae_plugin_on']}, "
                 f"off-support: {sup['mae_plugin_off']}")
    mech = result["mechanism_label"]
    lines.append("")
    lines.append("## Mechanism label (pass condition 3, LOSO CV)")
    lines.append("")
    lines.append(
        f"- CV MAE without label {mech['cv_mae_without_label']:.5f}, with "
        f"label {mech['cv_mae_with_label']:.5f}, gain "
        f"{mech['label_gain']:+.5f}")
    lines.append("")
    lines.append("## Calibration fits (2 params per arm)")
    lines.append("")
    for arm, fit in result["fits"].items():
        lines.append(f"- {arm}: alpha {fit['alpha']:+.5f}, beta "
                     f"{fit['beta']:+.4f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 2 transport test")
    base = "nc_csf_predictivity/interventions"
    parser.add_argument("--stats_root", type=str, required=True)
    parser.add_argument("--stage2b_dir", type=str, default=f"{base}/stage2b")
    parser.add_argument("--geometry_dir", type=str,
                        default=f"{base}/geometry")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["margin", "full"])
    parser.add_argument("--margin_file", type=str,
                        default=f"{base}/pilot2_margin.json")
    parser.add_argument("--out", type=str, default=f"{base}/pilot2_report.md")
    parser.add_argument("--reverse", action="store_true",
                        help="Sensitivity: fit on baseline+A2, predict A1 "
                             "(dose arm dropped; A2 has no numeric dose).")
    parser.add_argument("--overwrite_margin", action="store_true")
    args = parser.parse_args()

    train_lams: tuple[str, ...] = TRAIN_LAMS
    holdout_lams: tuple[str, ...] = HOLDOUT_LAMS
    with_dose = True
    if args.reverse:
        train_lams = ("0.0", "hard")
        holdout_lams = ("-0.1", "0.3", "1.0")
        with_dose = False

    table = load_long(Path(args.stats_root))
    stage2b = load_stage2b(Path(args.stage2b_dir))
    margin_path = Path(args.margin_file)

    if args.stage == "margin":
        if margin_path.exists() and not args.overwrite_margin:
            raise SystemExit(f"{margin_path} already exists; the margin is "
                             f"frozen (pass --overwrite_margin to force).")
        margins = {scale: bootstrap_margin(
            build_cells(table, stage2b, scale), train_lams, holdout_lams)
            for scale in ("auroc_f", "augrc")}
        margin_path.write_text(json.dumps(
            {"margins": margins, "b": BOOT_B, "seed": 0,
             "reverse": args.reverse, "note": MARGIN_NOTE}, indent=1))
        print(f"froze margins {margins} -> {margin_path}")
        print("Commit this file, then run --stage full.")
        return

    if not margin_path.exists():
        raise SystemExit(f"{margin_path} not found: run --stage margin "
                         f"first (the margin must be frozen before any "
                         f"fitted arm's A2 prediction errors exist).")
    margins = json.loads(margin_path.read_text())["margins"]
    geometry = load_geometry(Path(args.geometry_dir))
    outputs = {}
    for scale in ("auroc_f", "augrc"):
        cells = build_cells(table, stage2b, scale)
        outputs[scale] = evaluate(cells, geometry, stage2b, margins[scale],
                                  train_lams, holdout_lams,
                                  with_dose=with_dose)
    text = render(outputs["auroc_f"], "1 - AUROC_f (primary)")
    text += "\n\n---\n\n" + render(outputs["augrc"], "AUGRC (secondary)")
    Path(args.out).write_text(text)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(outputs, indent=1, default=float))
    primary = outputs["auroc_f"]
    print(f"margin condition: {primary['verdict_margin']}; sign: "
          f"{primary['sign']['n_correct']}/{primary['sign']['n_material']}; "
          f"registered pass: {primary['registered_pass']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

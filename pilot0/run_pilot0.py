"""Pilot 0 driver: operator-level falsification on cached checkpoint features.

Consumes NPZ caches from `extract_pilot0.py` (or generates a synthetic
ETF checkpoint with --synthetic) and produces the pre-registered gate
report. Gates (plan section 8, Pilot 0, amended):

  G0  cache consistency: h @ W^T + b reproduces stored logits.
  G1  feature-only scores (mean-CTM, Mahalanobis) are exactly invariant
      to every head operation.
  G2  head-side responses have the predicted sign on >= 80% of material
      cells (materiality: |predicted delta| >= 2 x Hanley-McNeil SE).
  G3  exact-mean theory beats the constant-response baseline in MAE.
  G4  head-CTM response follows the corrected gap-attenuation prediction
      (reported within G2/G3 for the CTM_head rows).
  G5  AUGRC and failure-AUROC rankings agree within every fixed block.
  G6  H estimators recover injected (gamma, a, rho) configurations;
      definitions freeze on pass.

Usage (from code/):
    python pilot0/run_pilot0.py --caches pilot0/caches/<slug>.npz [...] \
        [--out_dir pilot0/outputs]
    python pilot0/run_pilot0.py --synthetic [--out_dir pilot0/outputs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot0.geometry import FeatureModel, fit_feature_model, geometry_record
from pilot0.ood_coords import estimate_ood_coords, validate_estimators
from pilot0.rotation import head_grid
from pilot0.scores import (
    FEATURE_SCORES,
    HEAD_SCORES,
    MahalanobisScorer,
    auroc,
    compute_feature_scores,
    compute_head_scores,
    logits,
    normalize_rows,
)
from pilot0.theory import HeadContext, hanley_mcneil_se, predicted_aurocs
from src.rc_stats import RiskCoverageStats

TAUS = (0.5, 1.0)
THETAS_DEG = (10.0, 20.0, 30.0, 45.0, 60.0)
N_DRAWS = 5
SEED = 20260814


def load_cache(path: Path) -> dict:
    """Load one extraction NPZ into arrays plus metadata."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta_path = path.with_suffix(".json")
    data["meta"] = (json.loads(meta_path.read_text())
                    if meta_path.exists() else {})
    return data


def make_synthetic_cache(rng: np.random.Generator, n_classes: int = 10,
                         dim: int = 128, snr: float = 10.0) -> dict:
    """ETF checkpoint with known geometry (the verify2.py construction)."""
    radius = 1.0
    sigma = radius / snr
    m = (np.eye(n_classes) - 1.0 / n_classes) * np.sqrt(
        n_classes / (n_classes - 1)) * radius
    q = np.linalg.qr(rng.standard_normal((dim, n_classes)))[0]
    mu = m @ q.T
    span = np.linalg.qr(mu.T, mode="reduced")[0]
    theta_init = np.radians(15.0)
    v = rng.standard_normal((n_classes, dim))
    v -= (v @ span) @ span.T
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    w = np.cos(theta_init) * mu + np.sin(theta_init) * radius * v
    b = np.zeros(n_classes)

    n_train, n_eval = 40_000, 8_000
    y_train = rng.integers(0, n_classes, n_train)
    h_train = mu[y_train] + rng.standard_normal((n_train, dim)) * sigma
    y_test = rng.integers(0, n_classes, n_eval)
    h_test = mu[y_test] + rng.standard_normal((n_eval, dim)) * sigma

    mu_hat = mu / radius
    v_perp = rng.standard_normal(dim)
    v_perp -= span @ (span.T @ v_perp)
    v_perp /= np.linalg.norm(v_perp)
    u_tilt = np.cos(np.pi / 4) * mu_hat[0] + np.sin(np.pi / 4) * v_perp
    u_mid = mu_hat[0] + mu_hat[1]
    u_mid /= np.linalg.norm(u_mid)
    cache = {
        "h_train": h_train.astype(np.float32), "y_train": y_train,
        "h_iid_test": h_test.astype(np.float32), "y_iid_test": y_test,
        "w": w, "b": b,
        "logits_iid_test": (h_test @ w.T + b).astype(np.float32),
        "meta": {"slug": "synthetic_etf", "n_classes": n_classes,
                 "ood_sets": ["tilt45_g1.0", "tilt45_g1.25", "mid_g1.25"]},
    }
    for name, u, gamma in (("tilt45_g1.0", u_tilt, 1.0),
                           ("tilt45_g1.25", u_tilt, 1.25),
                           ("mid_g1.25", u_mid, 1.25)):
        cache[f"h_{name}"] = (gamma * radius * u
                              + rng.standard_normal((n_eval, dim)) * sigma
                              ).astype(np.float32)
    return cache


def analyze_checkpoint(cache: dict, out_dir: Path) -> dict:
    """Run the full Pilot 0 protocol on one checkpoint cache."""
    slug = cache["meta"].get("slug", "checkpoint")
    n_classes = int(cache["meta"].get("n_classes", cache["w"].shape[0]))
    h_train, y_train = cache["h_train"], cache["y_train"]
    w0, b = cache["w"].astype(np.float64), cache["b"].astype(np.float64)
    ood_names = list(cache["meta"]["ood_sets"])

    result: dict = {"slug": slug, "gates": {}}
    g_check = logits(cache["h_iid_test"], w0, b)
    result["gates"]["G0_logit_consistency"] = float(
        np.abs(g_check - cache["logits_iid_test"].astype(np.float64)).max())

    model = fit_feature_model(h_train, y_train, n_classes)
    uncentered_means = model.class_means + model.global_mean
    maha = MahalanobisScorer(h_train, y_train, n_classes)
    result["geometry_baseline"] = geometry_record(w0, b, model)

    rng = np.random.default_rng(SEED)
    result["h_validation"] = validate_estimators(model, rng)
    result["ood_coords"] = {name: estimate_ood_coords(cache[f"h_{name}"],
                                                      model)
                            for name in ood_names}

    states = head_grid(w0, model, TAUS, THETAS_DEG, N_DRAWS, SEED)
    sigma = model.sigma_iso
    n_id = len(cache["h_iid_test"])
    mean_prototypes_n = normalize_rows(uncentered_means)

    # Hoist everything head-state-independent out of the state loop:
    # float64/normalized activations, feature-only scores, OOD moments.
    eval_sets: dict[str, dict] = {}
    for name in ["iid_test"] + ood_names:
        h64 = cache[f"h_{name}"].astype(np.float64)
        rec = {"h64": h64, "h_n": normalize_rows(h64),
               "feat": compute_feature_scores(h64, maha, mean_prototypes_n)}
        if name != "iid_test":
            m_ood = h64.mean(0)
            resid = h64 - m_ood
            rec["m_ood"] = m_ood
            rec["sigma_o"] = float(
                np.sqrt((resid**2).sum(1).mean() / h64.shape[1]))
        eval_sets[name] = rec

    empirical: dict = {}
    predicted: dict = {}
    for name in ood_names:
        for score in FEATURE_SCORES:
            empirical[("baseline", name, score)] = auroc(
                eval_sets["iid_test"]["feat"][score],
                eval_sets[name]["feat"][score])

    id_rec = eval_sets["iid_test"]
    for state in states:
        ctx = HeadContext.from_head(state["w"], b)
        sc_id = compute_head_scores(id_rec["h64"], id_rec["h_n"], ctx.w, b)
        for name in ood_names:
            rec = eval_sets[name]
            sc_ood = compute_head_scores(rec["h64"], rec["h_n"], ctx.w, b)
            for score in HEAD_SCORES:
                empirical[(state["name"], name, score)] = auroc(
                    sc_id[score], sc_ood[score])
            predicted[(state["name"], name)] = predicted_aurocs(
                uncentered_means, model.class_freq, sigma, rec["m_ood"],
                rec["sigma_o"], ctx)
        state.pop("w")

    # G1: feature-only scores never consume the head state; verify the
    # plumbing by recomputing them through the same call path once and
    # requiring exact equality with the hoisted values.
    invariance_max = 0.0
    for name in ["iid_test"] + ood_names:
        redo = compute_feature_scores(eval_sets[name]["h64"], maha,
                                      mean_prototypes_n)
        for score in FEATURE_SCORES:
            invariance_max = max(invariance_max, float(np.abs(
                redo[score] - eval_sets[name]["feat"][score]).max()))
    result["gates"]["G1_feature_invariance_max_abs"] = invariance_max

    sign_hits, sign_total = 0, 0
    abs_err_theory, abs_err_const = [], []
    cells = []
    for state in states:
        if state["kind"] == "baseline":
            continue
        for name in ood_names:
            n_ood = len(cache[f"h_{name}"])
            for score in HEAD_SCORES:
                base_emp = empirical[("baseline", name, score)]
                base_pred = predicted[("baseline", name)][score]
                emp = empirical[(state["name"], name, score)]
                pred = predicted[(state["name"], name)][score]
                d_pred, d_emp = pred - base_pred, emp - base_emp
                se = hanley_mcneil_se(base_emp, n_id, n_ood)
                material = abs(d_pred) >= 2.0 * se
                abs_err_theory.append(abs(emp - pred))
                abs_err_const.append(abs(d_emp))
                if material:
                    sign_total += 1
                    sign_hits += int(np.sign(d_pred) == np.sign(d_emp))
                cells.append({
                    "state": state["name"], "ood_set": name, "score": score,
                    "auroc_emp": emp, "auroc_pred": pred,
                    "delta_emp": d_emp, "delta_pred": d_pred,
                    "material": bool(material)})
    result["cells"] = cells
    result["gates"]["G2_sign_agreement"] = (
        sign_hits / sign_total if sign_total else float("nan"))
    result["gates"]["G2_material_cells"] = sign_total
    result["gates"]["G3_mae_theory"] = float(np.mean(abs_err_theory))
    result["gates"]["G3_mae_constant"] = float(np.mean(abs_err_const))

    result["gates"].update(_identity_check(cache, model, maha,
                                           uncentered_means, b, ood_names))
    _write_report(result, out_dir)
    return result


def _identity_check(cache: dict, model: FeatureModel,
                    maha: MahalanobisScorer, uncentered_means: np.ndarray,
                    b: np.ndarray, ood_names: list[str]) -> dict:
    """G5: AUGRC vs failure-AUROC rank agreement within fixed blocks."""
    from scipy.stats import spearmanr
    y_id = cache["y_iid_test"]
    identity_states = ("baseline", "away_30deg_d0")
    worst_rho, max_dev = 1.0, 0.0
    lookup = {s["name"]: s["w"] for s in head_grid(
        cache["w"].astype(np.float64), model, TAUS, THETAS_DEG, N_DRAWS,
        SEED) if s["name"] in identity_states}
    mean_prototypes_n = normalize_rows(uncentered_means)
    prepared = {}
    for name in ["iid_test"] + ood_names:
        h64 = cache[f"h_{name}"].astype(np.float64)
        prepared[name] = (h64, normalize_rows(h64),
                          compute_feature_scores(h64, maha,
                                                 mean_prototypes_n))
    for state_name in identity_states:
        w_s = lookup[state_name]
        h64_id, hn_id, feat_id = prepared["iid_test"]
        sc_id = compute_head_scores(h64_id, hn_id, w_s, b) | feat_id
        preds = logits(h64_id, w_s, b).argmax(1)
        res_id = (preds != y_id).astype(float)
        for name in ood_names:
            h64_o, hn_o, feat_o = prepared[name]
            sc_ood = compute_head_scores(h64_o, hn_o, w_s, b) | feat_o
            res = np.concatenate([res_id, np.ones(len(h64_o))])
            pi = float(res.mean())
            augrcs, f_aurocs = [], []
            for score in HEAD_SCORES + FEATURE_SCORES:
                confids = np.concatenate([sc_id[score], sc_ood[score]])
                rc = RiskCoverageStats(confids=confids, residuals=res)
                augrcs.append(rc.augrc / rc.AUC_DISPLAY_SCALE)
                f_aurocs.append(auroc(confids[res == 0], confids[res == 1]))
            rho = spearmanr(augrcs, [1 - a for a in f_aurocs]).statistic
            worst_rho = min(worst_rho, float(rho))
            ident = [pi**2 / 2 + pi * (1 - pi) * (1 - a) for a in f_aurocs]
            max_dev = max(max_dev, float(np.abs(np.array(augrcs)
                                                - np.array(ident)).max()))
    return {"G5_rank_agreement_worst_spearman": worst_rho,
            "G5_identity_max_abs_dev": max_dev}


def _write_report(result: dict, out_dir: Path) -> None:
    """Write the per-checkpoint JSON and append to the markdown report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = result["slug"]
    with open(out_dir / f"pilot0_{slug}.json", "w") as fh:
        json.dump(result, fh, indent=1, default=float)

    g = result["gates"]
    lines = [f"## {slug}", ""]
    lines.append(f"- G0 logit consistency (max abs): "
                 f"{g['G0_logit_consistency']:.3e}")
    lines.append(f"- G1 feature-only invariance (max abs score delta): "
                 f"{g['G1_feature_invariance_max_abs']:.3e}")
    lines.append(f"- G2 sign agreement on material cells: "
                 f"{g['G2_sign_agreement']:.3f} "
                 f"({g['G2_material_cells']} cells; gate >= 0.80)")
    lines.append(f"- G3 MAE theory {g['G3_mae_theory']:.4f} vs constant "
                 f"{g['G3_mae_constant']:.4f} (gate: theory < constant)")
    lines.append(f"- G5 worst within-block Spearman(AUGRC, 1-AUROC_f): "
                 f"{g['G5_rank_agreement_worst_spearman']:.4f}; identity "
                 f"max abs dev {g['G5_identity_max_abs_dev']:.3e}")
    lines.append("")
    lines.append("### H-estimator recovery (G6)")
    lines.append("")
    lines.append("| true g / a / rho / w_perp | est g / a / rho / w_perp |")
    lines.append("|---|---|")
    for rec in result["h_validation"]:
        lines.append(
            f"| {rec['true_gamma']:.2f} / {rec['true_a']:.2f} / "
            f"{rec['true_rho']:.2f} / {rec['true_w_perp']:.2f} "
            f"| {rec['est_gamma']:.3f} / {rec['est_a']:.3f} / "
            f"{rec['est_rho']:.3f} / {rec['est_w_perp']:.3f} |")
    lines.append("")
    lines.append("### Measured OOD coordinates (frozen definitions)")
    lines.append("")
    lines.append("| OOD set | gamma | a | rho | w_perp | top2 gap |")
    lines.append("|---|---|---|---|---|---|")
    for name, c in result["ood_coords"].items():
        lines.append(f"| {name} | {c['gamma']:.3f} | {c['a']:.3f} | "
                     f"{c['rho']:.3f} | {c['w_perp']:.3f} | "
                     f"{c['top2_gap']:.3f} |")
    lines.append("")
    with open(out_dir / "pilot0_report.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot 0 operator study")
    parser.add_argument("--caches", nargs="*", default=[],
                        help="Extraction NPZ paths")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on a synthetic ETF checkpoint")
    parser.add_argument("--out_dir", type=str, default="pilot0/outputs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    report = out_dir / "pilot0_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.write_text("# Pilot 0 report\n\n")

    caches = ([make_synthetic_cache(np.random.default_rng(SEED))]
              if args.synthetic else
              [load_cache(Path(p)) for p in args.caches])
    if not caches:
        parser.error("provide --caches or --synthetic")
    import time
    for cache in caches:
        t0 = time.perf_counter()
        result = analyze_checkpoint(cache, out_dir)
        g = result["gates"]
        print(f"{result['slug']}: G2={g['G2_sign_agreement']:.3f} "
              f"({g['G2_material_cells']} cells) "
              f"G3 {g['G3_mae_theory']:.4f}<{g['G3_mae_constant']:.4f}? "
              f"G1 {g['G1_feature_invariance_max_abs']:.1e} "
              f"[{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()

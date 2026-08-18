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
    PCAFamily,
    auroc,
    compute_feature_scores,
    compute_head_scores,
    logits,
    normalize_rows,
)
from pilot0.theory import (
    HeadContext,
    NoiseModel,
    hanley_mcneil_se,
    predicted_aurocs,
)
from src.rc_stats import RiskCoverageStats

TAUS = (0.5, 1.0)
THETAS_DEG = (10.0, 20.0, 30.0, 45.0, 60.0)
N_DRAWS = 5
SEED = 20260814
REGISTERED_SCORES = ("MLS", "Energy", "CTM_head")


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
    pca = PCAFamily(h_train, n_classes, w0, b)
    train_mean = h_train.astype(np.float64).mean(0)
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
               "feat": compute_feature_scores(h64, maha, mean_prototypes_n,
                                              pca)}
        if name != "iid_test":
            m_ood = h64.mean(0)
            resid = h64 - m_ood
            rec["m_ood"] = m_ood
            rec["sigma_o"] = float(
                np.sqrt((resid**2).sum(1).mean() / h64.shape[1]))
            rec["cov_o"] = resid.T @ resid / len(resid)
        eval_sets[name] = rec

    empirical: dict = {}
    predicted: dict = {}
    for name in ood_names:
        for score in FEATURE_SCORES:
            empirical[("baseline", name, score)] = auroc(
                eval_sets["iid_test"]["feat"][score],
                eval_sets[name]["feat"][score])

    id_rec = eval_sets["iid_test"]
    dim = id_rec["h64"].shape[1]
    for state in states:
        ctx = HeadContext.from_head(state["w"], b)
        noise_id = {"iso": NoiseModel.isotropic(sigma, ctx, dim),
                    "emp": NoiseModel.empirical(model.sigma_w, ctx)}
        sc_id = compute_head_scores(id_rec["h64"], id_rec["h_n"], ctx.w, b,
                                    train_mean)
        if state["kind"] == "baseline":
            from scipy.stats import spearmanr
            result["fdbd_ctm_score_spearman_baseline"] = float(
                spearmanr(sc_id["fDBD"], sc_id["CTM_head"]).statistic)
        for name in ood_names:
            rec = eval_sets[name]
            sc_ood = compute_head_scores(rec["h64"], rec["h_n"], ctx.w, b,
                                         train_mean)
            for score in HEAD_SCORES + ("fDBD",):
                empirical[(state["name"], name, score)] = auroc(
                    sc_id[score], sc_ood[score])
            noise_ood = {
                "iso": NoiseModel.isotropic(rec["sigma_o"], ctx, dim),
                "emp": NoiseModel.empirical(rec["cov_o"], ctx)}
            predicted[(state["name"], name)] = {
                arm: predicted_aurocs(uncentered_means, model.class_freq,
                                      noise_id[arm], rec["m_ood"],
                                      noise_ood[arm], ctx)
                for arm in ("iso", "emp")}
        state.pop("w")

    # G1: feature-only scores never consume the head state; verify the
    # plumbing by recomputing them through the same call path once and
    # requiring exact equality with the hoisted values.
    invariance_max = 0.0
    for name in ["iid_test"] + ood_names:
        redo = compute_feature_scores(eval_sets[name]["h64"], maha,
                                      mean_prototypes_n, pca)
        for score in FEATURE_SCORES:
            invariance_max = max(invariance_max, float(np.abs(
                redo[score] - eval_sets[name]["feat"][score]).max()))
    result["gates"]["G1_feature_invariance_max_abs"] = invariance_max

    # G7 (registered secondary): the fDBD-vs-head-CTM AUROC gap should
    # grow with away-rotation angle (pairwise averaging suppresses
    # quenched leakage; tier-3 fDBD divergence result). Spearman trend of
    # gap on theta, pooled over draws, per OOD set.
    from scipy.stats import spearmanr as _spearmanr
    trends = {}
    for name in ood_names:
        thetas, gaps = [], []
        for state in states:
            if state["kind"] != "away":
                continue
            thetas.append(state["param"])
            gaps.append(empirical[(state["name"], name, "fDBD")]
                        - empirical[(state["name"], name, "CTM_head")])
        trends[name] = float(_spearmanr(thetas, gaps).statistic)
    result["fdbd_divergence_trends"] = trends
    positive = sum(t > 0 for t in trends.values())
    result["gates"]["G7_fdbd_divergence_positive_sets"] = (
        f"{positive}/{len(trends)}")
    result["gates"]["G7_fdbd_divergence_median_trend"] = float(
        np.median(list(trends.values())))

    # G2: sign agreement on material cells, per predictor arm.
    # G3a: RESPONSE-scale MAE, |delta_pred - delta_emp| vs the constant-
    #      response baseline |delta_emp| (the quantity that matters for
    #      the Pilot 2 plug-in index, which is calibrated per R2 anyway).
    # G3b: LEVEL MAE at the baseline head state, reported as the
    #      misspecification diagnostic (T4), not a pass/fail gate: the
    #      isotropic arm's level error on real anisotropic features is
    #      expected and documented, not a harness failure.
    # Confirmatory gates aggregate over REGISTERED_SCORES only: the plan's
    # section 7 pairs never include MSR, and X1 section 11 explicitly
    # deferred the MSR tie-case moments (max of correlated Gaussians) as
    # unnecessary for the registered comparisons. MSR stays in the cells
    # and the per-score table as a documented theory boundary.
    arms = ("iso", "emp")
    sign_hits: dict = {(a, s): 0 for a in arms for s in HEAD_SCORES}
    sign_total: dict = {(a, s): 0 for a in arms for s in HEAD_SCORES}
    resp_err: dict = {(a, s): [] for a in arms for s in HEAD_SCORES}
    resp_const: dict = {s: [] for s in HEAD_SCORES}
    level_err: dict = {(a, s): [] for a in arms for s in HEAD_SCORES}
    cells = []
    for state in states:
        if state["kind"] == "baseline":
            continue
        for name in ood_names:
            n_ood = len(cache[f"h_{name}"])
            for score in HEAD_SCORES:
                base_emp = empirical[("baseline", name, score)]
                emp = empirical[(state["name"], name, score)]
                d_emp = emp - base_emp
                se = hanley_mcneil_se(base_emp, n_id, n_ood)
                resp_const[score].append(abs(d_emp))
                cell = {"state": state["name"], "ood_set": name,
                        "score": score, "auroc_emp": emp, "delta_emp": d_emp}
                for arm in arms:
                    base_pred = predicted[("baseline", name)][arm][score]
                    pred = predicted[(state["name"], name)][arm][score]
                    d_pred = pred - base_pred
                    resp_err[(arm, score)].append(abs(d_pred - d_emp))
                    material = abs(d_pred) >= 2.0 * se
                    if material:
                        sign_total[(arm, score)] += 1
                        sign_hits[(arm, score)] += int(
                            np.sign(d_pred) == np.sign(d_emp))
                    cell[f"auroc_pred_{arm}"] = pred
                    cell[f"delta_pred_{arm}"] = d_pred
                    cell[f"material_{arm}"] = bool(material)
                cells.append(cell)
    result["cells"] = cells
    for name in ood_names:
        for score in HEAD_SCORES:
            for arm in arms:
                level_err[(arm, score)].append(
                    abs(predicted[("baseline", name)][arm][score]
                        - empirical[("baseline", name, score)]))

    result["per_score"] = {
        score: {arm: {
            "sign": (sign_hits[(arm, score)] / sign_total[(arm, score)]
                     if sign_total[(arm, score)] else float("nan")),
            "n_material": sign_total[(arm, score)],
            "response_mae": float(np.mean(resp_err[(arm, score)])),
            "level_mae": float(np.mean(level_err[(arm, score)])),
        } for arm in arms} | {
            "response_mae_constant": float(np.mean(resp_const[score]))}
        for score in HEAD_SCORES}

    for arm in arms:
        reg_hits = sum(sign_hits[(arm, s)] for s in REGISTERED_SCORES)
        reg_total = sum(sign_total[(arm, s)] for s in REGISTERED_SCORES)
        result["gates"][f"G2_sign_agreement_{arm}"] = (
            reg_hits / reg_total if reg_total else float("nan"))
        result["gates"][f"G2_material_cells_{arm}"] = reg_total
        result["gates"][f"G3a_response_mae_{arm}"] = float(np.mean(
            [e for s in REGISTERED_SCORES for e in resp_err[(arm, s)]]))
        result["gates"][f"G3b_level_mae_{arm}"] = float(np.mean(
            [e for s in REGISTERED_SCORES for e in level_err[(arm, s)]]))
    result["gates"]["G3a_response_mae_constant"] = float(np.mean(
        [e for s in REGISTERED_SCORES for e in resp_const[s]]))

    result["gates"].update(_identity_check(cache, model, maha, pca,
                                           train_mean, uncentered_means, b,
                                           ood_names))
    _write_report(result, out_dir)
    return result


def _identity_check(cache: dict, model: FeatureModel,
                    maha: MahalanobisScorer, pca: PCAFamily,
                    train_mean: np.ndarray, uncentered_means: np.ndarray,
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
                                                 mean_prototypes_n, pca))
    for state_name in identity_states:
        w_s = lookup[state_name]
        h64_id, hn_id, feat_id = prepared["iid_test"]
        sc_id = compute_head_scores(h64_id, hn_id, w_s, b, train_mean) | feat_id
        preds = logits(h64_id, w_s, b).argmax(1)
        res_id = (preds != y_id).astype(float)
        for name in ood_names:
            h64_o, hn_o, feat_o = prepared[name]
            sc_ood = compute_head_scores(h64_o, hn_o, w_s, b,
                                         train_mean) | feat_o
            res = np.concatenate([res_id, np.ones(len(h64_o))])
            pi = float(res.mean())
            augrcs, f_aurocs = [], []
            for score in HEAD_SCORES + ("fDBD",) + FEATURE_SCORES:
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
    lines.append(f"- G2 sign agreement, registered endpoints "
                 f"(MLS/Energy/CTM_head; gate >= 0.80): "
                 f"empirical-cov {g['G2_sign_agreement_emp']:.3f} "
                 f"({g['G2_material_cells_emp']} cells); "
                 f"isotropic {g['G2_sign_agreement_iso']:.3f} "
                 f"({g['G2_material_cells_iso']} cells)")
    lines.append(f"- G3a response MAE, registered endpoints "
                 f"(gate: emp < constant): "
                 f"empirical-cov {g['G3a_response_mae_emp']:.4f}, "
                 f"isotropic {g['G3a_response_mae_iso']:.4f}, "
                 f"constant-response {g['G3a_response_mae_constant']:.4f}")
    lines.append(f"- G3b baseline level MAE (diagnostic, not a gate): "
                 f"empirical-cov {g['G3b_level_mae_emp']:.4f}, "
                 f"isotropic {g['G3b_level_mae_iso']:.4f} "
                 f"(iso-emp gap = anisotropy misspecification, T4)")
    lines.append("")
    lines.append("### Per-score response diagnostics (empirical-cov arm)")
    lines.append("")
    lines.append("| score | sign@material | resp MAE | resp MAE const "
                 "| level MAE |")
    lines.append("|---|---|---|---|---|")
    for score, rec in result["per_score"].items():
        e = rec["emp"]
        tag = "" if score in ("MLS", "Energy", "CTM_head") else \
            " (diagnostic only, tie-case moments deferred in X1 pass 2)"
        lines.append(
            f"| {score}{tag} | {e['sign']:.3f}@{e['n_material']} "
            f"| {e['response_mae']:.4f} "
            f"| {rec['response_mae_constant']:.4f} "
            f"| {e['level_mae']:.4f} |")
    lines.append(f"- G5 worst within-block Spearman(AUGRC, 1-AUROC_f): "
                 f"{g['G5_rank_agreement_worst_spearman']:.4f}; identity "
                 f"max abs dev {g['G5_identity_max_abs_dev']:.3e}")
    lines.append(f"- G7 fDBD-CTM divergence under away-rotation "
                 f"(registered secondary; predicted: gap grows with theta): "
                 f"positive trend in {g['G7_fdbd_divergence_positive_sets']} "
                 f"OOD sets, median Spearman "
                 f"{g['G7_fdbd_divergence_median_trend']:+.3f}; baseline "
                 f"fDBD-CTM score Spearman "
                 f"{result['fdbd_ctm_score_spearman_baseline']:.3f} "
                 f"(tier-3: -> 1 at exact NC)")
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
        print(f"{result['slug']}: "
              f"G2_emp={g['G2_sign_agreement_emp']:.3f} "
              f"({g['G2_material_cells_emp']} cells) "
              f"G3a {g['G3a_response_mae_emp']:.4f}"
              f"<{g['G3a_response_mae_constant']:.4f}? "
              f"G3b lvl emp={g['G3b_level_mae_emp']:.3f} "
              f"iso={g['G3b_level_mae_iso']:.3f} "
              f"G1 {g['G1_feature_invariance_max_abs']:.1e} "
              f"[{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()

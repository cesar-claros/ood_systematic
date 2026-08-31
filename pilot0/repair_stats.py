"""Sufficient statistics for the mixture-aware anisotropic repair
(saturation plan 2026-08-28, sections 7, 8.4, and 12 Phase 1). Pure
numpy; shared by the HPC extractor and the local synthetic self-test.

FROZEN RULES (declared before any repair outcome exists; no detector
outcome, score, or ranking enters any rule):
- Space: the phase model's centered feature space (train global mean
  subtracted); prototypes are the centered train class-mean directions
  mu_hat_c, the same directions the frozen coordinate estimators use.
- Component membership (plan section 7.1, nearest-prototype option):
  K(h) = argmax_c mu_hat_c' (h - global_mean).
- Small components: components with fewer than N_MIN = 25 samples merge
  into one "other" component (id -1).
- Covariance rule (plan section 7.1, pooled option): component-specific
  means with ONE pooled residual covariance shared across components
  (Cov of h - m_{K(h)} over all samples of the set); the global
  (unpartitioned) residual covariance is stored separately for the P01
  arm. Per-component residual traces are stored as a diagnostic of the
  shared-covariance approximation, never used to refit it.
- Stored per population (plan section 8.4): mean projections onto the
  head (W m) and the prototypes (mu_hat m) plus the squared norm;
  covariance projections W Sigma W', diag(mu_hat Sigma mu_hat'),
  cross terms mu_hat Sigma m, m' Sigma m, tr Sigma, tr Sigma^2, and the
  top-16 eigenvalues for auditing. These are sufficient for the
  P00/P01/P10/P11 Energy, MLS, MSR, and CTM delta-method moments with
  first- and second-order mean corrections; Mahalanobis needs
  precision-weighted forms and is deferred (Energy-CTM is the
  claim-bearing pair).
- Batch-occupancy diagnostic (gate R6 support): at n_o in
  {64, 128, 256, 512, 1000}, a seeded subsample (BATCH_SEED + set index)
  records the number of components holding at least 5% of the subsample
  and the top-component share.

Self-test (run locally: python pilot0/repair_stats.py): verifies the
projection identities against direct computation and reproduces the
plan's section-6.3 two-mode example, where the global mean's maximum
alignment collapses while the component means retain it.
"""
from __future__ import annotations

import numpy as np

N_MIN = 25
BATCH_GRID = (64, 128, 256, 512, 1000)
BATCH_SEED = 20260830
TOP_EIGS = 16


def pop_stats(m: np.ndarray, cov: np.ndarray | None, w: np.ndarray,
              mu_hat: np.ndarray) -> dict:
    """Projection statistics for one population mean (and covariance)."""
    out = {"g": w @ m, "a": mu_hat @ m, "n2": float(m @ m)}
    if cov is not None:
        out.update(cov_stats(cov, w, mu_hat, m))
    return out


def cov_stats(cov: np.ndarray, w: np.ndarray, mu_hat: np.ndarray,
              m: np.ndarray | None = None) -> dict:
    wc = w @ cov
    mc = mu_hat @ cov
    out = {"WSW": wc @ w.T, "dir": (mc * mu_hat).sum(1),
           "trS": float(np.trace(cov)),
           "trS2": float((cov * cov).sum()),
           "eigs": np.sort(np.linalg.eigvalsh(cov))[::-1][:TOP_EIGS]}
    if m is not None:
        out["cross"] = mc @ m
        out["mSm"] = float(m @ cov @ m)
    return out


def id_stats(fm, w: np.ndarray, b: np.ndarray) -> dict:
    """ID-side sufficient statistics from the fitted FeatureModel."""
    mu = fm.class_means
    mu_hat = mu / fm.radii[:, None]
    sw = fm.sigma_w
    mc = mu_hat @ sw
    return {"class_freq": fm.class_freq, "radii": fm.radii,
            "b_eff": w @ fm.global_mean + b,
            "G_id": mu @ w.T, "A_id": mu_hat @ mu.T,
            "WSW_id": (w @ sw) @ w.T, "dir_id": (mc * mu_hat).sum(1),
            "CROSS_id": mc @ mu.T,
            "qSq_id": np.einsum("cd,de,ce->c", mu, sw, mu),
            "trS_id": float(np.trace(sw)),
            "trS2_id": float((sw * sw).sum()),
            "eigs_id": np.sort(np.linalg.eigvalsh(sw))[::-1][:TOP_EIGS]}


def assign_components(hc: np.ndarray, mu_hat: np.ndarray,
                      chunk: int = 8192) -> np.ndarray:
    """Nearest-prototype membership K(h) on centered features, chunked."""
    out = np.empty(len(hc), dtype=np.int64)
    for i in range(0, len(hc), chunk):
        out[i:i + chunk] = (hc[i:i + chunk] @ mu_hat.T).argmax(1)
    return out


def mixture_stats(hc: np.ndarray, w: np.ndarray, mu_hat: np.ndarray,
                  logits_argmax: np.ndarray | None = None,
                  set_index: int = 0, chunk: int = 8192) -> dict:
    """Component decomposition of one centered OOD feature matrix.

    Returns global-mean stats (P00/P01), the frozen component partition
    with per-component mean projections, the pooled shared residual
    covariance stats (P10/P11), switching diagnostics, and the
    batch-occupancy grid. `logits_argmax` (optional) supplies the
    per-sample argmax-logit class for the class-switch diagnostic.

    Implementation is streaming (float64 accumulators over chunks; no
    residual matrix is materialized): cov_glob = S2/n - m m',
    cov_res = (S2 - sum_k n_k m_k m_k')/n, per-component residual trace
    from row-norm sums. Mathematically identical to the direct forms.
    """
    n, d = hc.shape
    labels = assign_components(hc, mu_hat, chunk=chunk)
    raw_counts = np.bincount(labels, minlength=len(mu_hat))
    keep = np.where(raw_counts >= N_MIN)[0]
    comp_id = np.where(np.isin(labels, keep), labels, -1)
    ids = sorted(set(comp_id.tolist()))
    pos = {k: i for i, k in enumerate(ids)}
    s2 = np.zeros((d, d))
    sums = np.zeros((len(ids), d))
    norm2 = np.zeros(len(ids))
    counts = np.zeros(len(ids))
    for i in range(0, n, chunk):
        blk = hc[i:i + chunk].astype(np.float64)
        cid = comp_id[i:i + chunk]
        s2 += blk.T @ blk
        rn2 = (blk ** 2).sum(1)
        for k in np.unique(cid):
            mask = cid == k
            sums[pos[k]] += blk[mask].sum(0)
            norm2[pos[k]] += rn2[mask].sum()
            counts[pos[k]] += mask.sum()
    m_glob = sums.sum(0) / n
    cov_glob = s2 / n - np.outer(m_glob, m_glob)
    means = sums / counts[:, None]
    cov_res = (s2 - (means.T * counts) @ means) / n
    comps = [{"component": int(k), "n": int(counts[pos[k]]),
              "weight": float(counts[pos[k]] / n),
              **pop_stats(means[pos[k]], None, w, mu_hat),
              "resid_tr": float(norm2[pos[k]] / counts[pos[k]]
                                - means[pos[k]] @ means[pos[k]])}
             for k in ids]
    omega = np.array([c["weight"] for c in comps])
    diag = {
        "n": int(n),
        "n_components_raw": int((raw_counts > 0).sum()),
        "n_components_kept": int(len(keep)),
        "other_weight": float((comp_id == -1).mean()),
        "top_component_share": float(raw_counts.max() / n),
        "component_entropy": float(-(omega * np.log(omega)).sum()),
    }
    if logits_argmax is not None:
        diag["class_vs_prototype_switch_rate"] = float(
            (logits_argmax != labels).mean())
    batch = {}
    rng = np.random.default_rng(BATCH_SEED + set_index)
    for n_o in BATCH_GRID:
        if n_o >= n:
            continue
        sub = labels[rng.choice(n, n_o, replace=False)]
        cnt = np.bincount(sub, minlength=len(mu_hat))
        batch[str(n_o)] = {
            "n_components_5pct": int((cnt >= 0.05 * n_o).sum()),
            "top_share": float(cnt.max() / n_o)}
    return {"global": pop_stats(m_glob, cov_glob, w, mu_hat),
            "components": comps,
            "resid_shared": cov_stats(cov_res, w, mu_hat),
            "diagnostics": diag, "batch_occupancy": batch}


def compact_p10(h_ood: np.ndarray, fm, w: np.ndarray,
                set_index: int = 0) -> dict:
    """ADDITIVE 2026-08-31 (ICML campaign, protocol section 8.1/8.3):
    the JSON-compact per-set inputs of the P10 mixture arm, computed
    with the FROZEN rules above (assign_components N_MIN merge,
    pooled shared residual). Exactly the fields repair_factorial's
    arm_P10r consumes: per kept component (weight, n2, a_max) and the
    shared rho_res = sqrt(trS_res / trS_id). No detector outcome enters.
    """
    hc = h_ood.astype(np.float64) - fm.global_mean
    mu_hat = fm.class_means / fm.radii[:, None]
    st = mixture_stats(hc, w, mu_hat, set_index=set_index)
    trs_id = float(np.trace(fm.sigma_w))
    return {
        "rho_res": float(np.sqrt(st["resid_shared"]["trS"] / trs_id)),
        "trS_res": float(st["resid_shared"]["trS"]),
        "trS_id": trs_id, "R": float(fm.radius),
        "components": [{"component": c["component"], "n": c["n"],
                        "weight": c["weight"], "n2": c["n2"],
                        "a_max": float(np.max(c["a"]))}
                       for c in st["components"]],
        "diagnostics": st["diagnostics"],
    }


def split_record(stats: dict) -> tuple[dict, dict]:
    """Split one stats dict into JSON-safe scalars and npz arrays."""
    scalars, arrays = {}, {}

    def walk(obj, prefix):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}__{k}" if prefix else k)
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}__{i}")
        elif isinstance(obj, np.ndarray):
            arrays[prefix] = obj
        else:
            scalars[prefix] = obj
    walk(stats, "")
    return scalars, arrays


def self_test() -> None:
    rng = np.random.default_rng(7)
    c_num, d = 8, 64
    mu = rng.standard_normal((c_num, d)) * 4.0
    mu -= mu.mean(0)
    radii = np.linalg.norm(mu, axis=1)
    mu_hat = mu / radii[:, None]
    w = rng.standard_normal((c_num, d)) * 0.3
    cov_true = np.diag(rng.uniform(0.5, 2.0, d))

    # projection identities against direct computation
    m = rng.standard_normal(d)
    cs = cov_stats(cov_true, w, mu_hat, m)
    assert np.allclose(cs["WSW"], w @ cov_true @ w.T)
    assert np.allclose(cs["dir"],
                       np.einsum("cd,de,ce->c", mu_hat, cov_true, mu_hat))
    assert np.allclose(cs["cross"], mu_hat @ cov_true @ m)
    assert np.isclose(cs["mSm"], m @ cov_true @ m)
    assert np.isclose(cs["trS2"], np.trace(cov_true @ cov_true))

    # plan section 6.3: two equally weighted modes toward classes 0 and 1
    r = 3.0
    n_half = 4000
    sqrt_cov = np.sqrt(np.diag(cov_true))
    h0 = r * radii[0] * mu_hat[0] + rng.standard_normal(
        (n_half, d)) * sqrt_cov
    h1 = r * radii[1] * mu_hat[1] + rng.standard_normal(
        (n_half, d)) * sqrt_cov
    hc = np.vstack([h0, h1])
    mx = mixture_stats(hc, w, mu_hat)
    a_glob = float(mx["global"]["a"].max()
                   / np.sqrt(mx["global"]["n2"]))
    comp_aligns = [float(c["a"].max() / np.sqrt(c["n2"]))
                   for c in mx["components"] if c["component"] >= 0]
    assert len(comp_aligns) >= 2, "two modes must occupy two components"
    assert min(comp_aligns) > a_glob + 0.2, (
        "component alignments must exceed the direction-mixed global "
        f"alignment ({comp_aligns} vs {a_glob:.3f})")
    # weights recover the halves; shared residual trace ~ true trace
    kept_w = sum(c["weight"] for c in mx["components"]
                 if c["component"] >= 0)
    assert kept_w > 0.98
    assert abs(mx["resid_shared"]["trS"] - np.trace(cov_true)) \
        / np.trace(cov_true) < 0.05

    # split_record round-trip shapes
    scalars, arrays = split_record(mx)
    assert any(k.endswith("__WSW") for k in arrays)
    assert "diagnostics__n_components_raw" in scalars
    print("repair_stats self_test: projection identities, section-6.3 "
          f"two-mode example (global max-alignment {a_glob:.3f} vs "
          f"component alignments {[round(x, 3) for x in comp_aligns]}), "
          "and record round-trip all PASS")


if __name__ == "__main__":
    self_test()
